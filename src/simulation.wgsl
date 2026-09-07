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
