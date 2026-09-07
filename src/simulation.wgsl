// Rigid TIP4P-Ew water with a reaction-field cutoff. Angstrom, kJ/mol, amu, ps, e.
// Constants are mirrored from src/water-model.ts and guarded by tests/wgsl-constants.test.ts.
const COULOMB = 1389.35456;
const BOLTZMANN = 0.0083144626;
const FORCE_TO_ACCELERATION = 100.0;
const SIGMA_O = 3.16435;
const EPSILON_O = 0.680946;
const Q_H = 0.52422;
const Q_M = -1.04844;
const MOLECULE_MASS = 18.015324;
const INERTIA = vec3f(1.155054441, 0.614540977, 1.769595419);
const BODY_OXYGEN = vec3f(-0.065559552, 0.0, 0.0);
const BODY_HYDROGEN_A = vec3f(0.520322725, 0.756950327, 0.0);
const BODY_HYDROGEN_B = vec3f(0.520322725, -0.756950327, 0.0);
const BODY_CHARGE = vec3f(0.059440448, 0.0, 0.0);

struct Params {
  box: f32,
  cutoff: f32,
  dt: f32,
  temperature: f32,
  friction: f32,
  molecules: u32,
  step: u32,
  seed: u32,
  /** Factor applied to every centre of mass by the rescale kernel. */
  scale: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Four vec4 per molecule: centre, quaternion, velocity, body angular momentum.
@group(0) @binding(1) var<storage, read_write> state: array<vec4f>;
// Three vec4 per molecule: O, H1, H2. Oxygen is wrapped; hydrogens stay whole.
@group(0) @binding(2) var<storage, read_write> sites: array<vec4f>;
// One vec4 per molecule: the M charge site, placed relative to the wrapped oxygen.
@group(0) @binding(3) var<storage, read_write> charges: array<vec4f>;
// Two vec4 per molecule: net force, then net torque about the centre of mass.
@group(0) @binding(4) var<storage, read_write> forceTorque: array<vec4f>;
// translational temperature, rotational temperature, maximum force, non-finite flag.
@group(0) @binding(5) var<storage, read_write> stats: array<f32>;

fn quatRotate(q: vec4f, v: vec3f) -> vec3f {
  let t = 2.0 * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
}

fn quatInverseRotate(q: vec4f, v: vec3f) -> vec3f {
  return quatRotate(vec4f(-q.xyz, q.w), v);
}

/** Charge site a of molecule i: 0 and 1 are hydrogens, 2 is the M site. */
fn chargeSiteOf(molecule: u32, index: u32) -> vec3f {
  if (index == 2u) { return charges[molecule].xyz; }
  return sites[molecule * 3u + index + 1u].xyz;
}

/** Body-frame constant paired with chargeSiteOf's index, for the matching torque arm. */
fn bodyArmOf(index: u32) -> vec3f {
  if (index == 2u) { return BODY_CHARGE; }
  if (index == 0u) { return BODY_HYDROGEN_A; }
  return BODY_HYDROGEN_B;
}

fn chargeOf(index: u32) -> f32 {
  if (index == 2u) { return Q_M; }
  return Q_H;
}

/** Force-shifted Lennard-Jones magnitude; positive is repulsive. */
fn lennardJones(r: f32, cutoff: f32) -> f32 {
  let s6 = pow(SIGMA_O / r, 6.0);
  let edge = pow(SIGMA_O / cutoff, 6.0);
  let force = 24.0 * EPSILON_O * (2.0 * s6 * s6 - s6) / r;
  let edgeForce = 24.0 * EPSILON_O * (2.0 * edge * edge - edge) / cutoff;
  return force - edgeForce;
}

/** Reaction field with conducting boundary; magnitude vanishes at the cutoff. */
fn reactionField(r: f32, product: f32, cutoff: f32) -> f32 {
  return COULOMB * product * (1.0 / (r * r) - r / (cutoff * cutoff * cutoff));
}

@compute @workgroup_size(64)
fn forces(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= params.molecules) { return; }
  let q = state[i * 4u + 1u];
  let oxygen = sites[i * 3u].xyz;
  let cutoff = params.cutoff;
  var force = vec3f(0.0);
  var torque = vec3f(0.0);
  // Torque arms come from the quaternion (quatRotate(q, BODY_*)), not from `site - centre`:
  // `sites`' oxygen is wrapped into [0, box) while `state`'s centre of mass is not, so once a
  // molecule's centre and its wrapped oxygen land in different periodic images that subtraction
  // is off by a whole box vector. The body-frame constants are already relative to the centre of
  // mass, so quatRotate(q, BODY_*) gives the exact arm regardless of which image the sites wrap
  // into. Do not "simplify" this back to `site - centre`.
  let oxygenArm = quatRotate(q, BODY_OXYGEN);
  for (var j = 0u; j < params.molecules; j++) {
    if (j == i) { continue; }
    let raw = sites[j * 3u].xyz - oxygen;
    let shift = -params.box * round(raw / params.box);
    let delta = raw + shift;
    let r2 = dot(delta, delta);
    if (r2 >= cutoff * cutoff) { continue; }
    let r = sqrt(r2);
    // Lennard-Jones acts between the oxygens only.
    let pull = -lennardJones(r, cutoff) * delta / r;
    force += pull;
    torque += cross(oxygenArm, pull);
    for (var a = 0u; a < 3u; a++) {
      let here = chargeSiteOf(i, a);
      let arm = quatRotate(q, bodyArmOf(a));
      let qa = chargeOf(a);
      for (var b = 0u; b < 3u; b++) {
        let there = chargeSiteOf(j, b) + shift;
        let separation = there - here;
        let distance = length(separation);
        let magnitude = reactionField(distance, qa * chargeOf(b), cutoff);
        let contribution = -magnitude * separation / distance;
        force += contribution;
        torque += cross(arm, contribution);
      }
    }
  }
  forceTorque[i * 2u] = vec4f(force, 0.0);
  forceTorque[i * 2u + 1u] = vec4f(torque, 0.0);
}

/** Rebuilds the drawable sites and the M site from the rigid-body state. */
fn writeSites(i: u32) {
  let centre = state[i * 4u].xyz;
  let q = state[i * 4u + 1u];
  let oxygen = centre + quatRotate(q, BODY_OXYGEN);
  let wrapped = oxygen - params.box * floor(oxygen / params.box);
  sites[i * 3u] = vec4f(wrapped, 1.0);
  sites[i * 3u + 1u] = vec4f(wrapped + quatRotate(q, BODY_HYDROGEN_A - BODY_OXYGEN), 1.0);
  sites[i * 3u + 2u] = vec4f(wrapped + quatRotate(q, BODY_HYDROGEN_B - BODY_OXYGEN), 1.0);
  charges[i] = vec4f(wrapped + quatRotate(q, BODY_CHARGE - BODY_OXYGEN), 1.0);
}

/** Density changes move centres of mass; molecules stay rigid. */
@compute @workgroup_size(64)
fn rescale(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= params.molecules) { return; }
  state[i * 4u] = vec4f(state[i * 4u].xyz * params.scale, 1.0);
  writeSites(i);
}

struct Rotation {
  q: vec4f,
  l: vec3f,
}

/** PCG hash: a stateless stream indexed by molecule, step and lane. */
fn hash(value: u32) -> u32 {
  let mixed = value * 747796405u + 2891336453u;
  let word = ((mixed >> ((mixed >> 28u) + 4u)) ^ mixed) * 277803737u;
  return (word >> 22u) ^ word;
}

fn uniform01(value: u32) -> f32 {
  return f32(hash(value)) * 2.3283064365386963e-10;
}

fn gaussianPair(value: u32) -> vec2f {
  let first = max(uniform01(value), 1e-7);
  let second = uniform01(value ^ 0x9e3779b9u);
  let radius = sqrt(-2.0 * log(first));
  let angle = 6.283185307179586 * second;
  return vec2f(radius * cos(angle), radius * sin(angle));
}

fn gaussian3(value: u32) -> vec3f {
  let first = gaussianPair(value);
  let second = gaussianPair(value * 2654435761u + 12345u);
  return vec3f(first.x, first.y, second.x);
}

/** dq/dt for a body-frame angular velocity. */
fn quatDerivative(q: vec4f, w: vec3f) -> vec4f {
  return 0.5 * vec4f(q.w * w + cross(q.xyz, w), -dot(q.xyz, w));
}

/**
 * Torque-free rotation, sub-stepped so the explicit-Euler update to Euler's equations keeps its
 * first-order error small over each sub-step. That update is not itself norm-preserving: it can
 * add a component of `l` along its own direction, so ||l|| would otherwise drift upward step
 * after step. Torque-free motion conserves ||l|| exactly, so the magnitude captured on entry is
 * restored once the sub-steps are done (a zero-magnitude input is left at zero, since
 * `normalize` on a zero vector is undefined).
 */
fn freeRotation(start: vec4f, momentum: vec3f, duration: f32) -> Rotation {
  var q = start;
  var l = momentum;
  let magnitude = length(momentum);
  let h = duration / 4.0;
  for (var sub = 0u; sub < 4u; sub++) {
    let w = l / INERTIA;
    l -= h * cross(w, l);
    q = normalize(q + h * quatDerivative(q, w));
  }
  if (magnitude > 0.0) {
    l = normalize(l) * magnitude;
  } else {
    l = vec3f(0.0);
  }
  return Rotation(q, l);
}

/** One BAOAB step. The trailing half kick is the leading one of the next step. */
@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= params.molecules) { return; }
  var centre = state[i * 4u].xyz;
  var q = state[i * 4u + 1u];
  var velocity = state[i * 4u + 2u].xyz;
  var angular = state[i * 4u + 3u].xyz;
  let dt = params.dt;
  let force = forceTorque[i * 2u].xyz;
  let torque = quatInverseRotate(q, forceTorque[i * 2u + 1u].xyz);

  velocity += 0.5 * dt * FORCE_TO_ACCELERATION * force / MOLECULE_MASS;
  angular += 0.5 * dt * FORCE_TO_ACCELERATION * torque;

  centre += 0.5 * dt * velocity;
  var rotation = freeRotation(q, angular, 0.5 * dt);
  q = rotation.q;
  angular = rotation.l;

  let decay = exp(-params.friction * dt);
  let spread = sqrt(1.0 - decay * decay);
  let energy = BOLTZMANN * params.temperature * FORCE_TO_ACCELERATION;
  // Mix molecule and step sequentially through the hash rather than combining them with XOR
  // and multiplication directly: that combination is not injective (distinct (i, step) pairs
  // can land on the same stream), where chaining the hash spreads both indices' bits fully
  // before they interact.
  let stream = hash(params.seed ^ hash(i ^ hash(params.step)));
  velocity = decay * velocity + spread * sqrt(energy / MOLECULE_MASS) * gaussian3(stream);
  angular = decay * angular + spread * sqrt(energy * INERTIA) * gaussian3(stream ^ 0x5bf03635u);

  centre += 0.5 * dt * velocity;
  rotation = freeRotation(q, angular, 0.5 * dt);
  q = rotation.q;
  angular = rotation.l;

  // Wrapping the centre keeps f32 coordinates small over a long session.
  centre -= params.box * floor(centre / params.box);
  state[i * 4u] = vec4f(centre, 1.0);
  state[i * 4u + 1u] = q;
  state[i * 4u + 2u] = vec4f(velocity, 0.0);
  state[i * 4u + 3u] = vec4f(angular, 0.0);
  writeSites(i);
}
