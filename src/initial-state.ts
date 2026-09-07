import { BOLTZMANN, FORCE_TO_ACCELERATION, ICE_CELL, MASS_H, MASS_O, MOLECULE_MASS,
  bodySites, boxLength, cellsFor, principalMoments, type Vector3 } from './water-model';

export type Quaternion = [number, number, number, number];
export interface InitialState {
  molecules: number;
  box: number;
  /** Four vec4 per molecule: centre, quaternion, velocity, body angular momentum. */
  state: Float32Array;
  /** Three vec4 per molecule: O, H1, H2. */
  sites: Float32Array;
  /** One vec4 per molecule: the M charge site. */
  charges: Float32Array;
}

const cross = (a: Vector3, b: Vector3): Vector3 =>
  [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = (v: Vector3): Vector3 => {
  const norm = Math.hypot(...v);
  if (norm < 1e-10) throw new Error('Degenerate water orientation');
  return [v[0] / norm, v[1] / norm, v[2] / norm];
};

/** Deterministic 32-bit generator, so a seed reproduces a run exactly. */
export function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const gaussian = (random: () => number): number =>
  Math.sqrt(-2 * Math.log(Math.max(random(), 1e-12))) * Math.cos(2 * Math.PI * random());

/** Body-to-space quaternion: x along the bisector, y from the second hydrogen to the first. */
export function orientationOf(oxygen: Vector3, first: Vector3, second: Vector3): Quaternion {
  const toFirst = first.map((value, axis) => value - oxygen[axis]) as Vector3;
  const toSecond = second.map((value, axis) => value - oxygen[axis]) as Vector3;
  const x = unit(toFirst.map((value, axis) => value + toSecond[axis]) as Vector3);
  const plane = unit(toFirst.map((value, axis) => value - toSecond[axis]) as Vector3);
  const z = unit(cross(x, plane));
  const y = cross(z, x);
  const m = [x, y, z];                                  // rows are the body axes in space
  const trace = m[0][0] + m[1][1] + m[2][2];
  let q: Quaternion;
  if (trace > 0) {
    const s = 2 * Math.sqrt(trace + 1);
    q = [(m[1][2] - m[2][1]) / s, (m[2][0] - m[0][2]) / s, (m[0][1] - m[1][0]) / s, s / 4];
  } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
    const s = 2 * Math.sqrt(1 + m[0][0] - m[1][1] - m[2][2]);
    q = [s / 4, (m[1][0] + m[0][1]) / s, (m[2][0] + m[0][2]) / s, (m[1][2] - m[2][1]) / s];
  } else if (m[1][1] > m[2][2]) {
    const s = 2 * Math.sqrt(1 + m[1][1] - m[0][0] - m[2][2]);
    q = [(m[1][0] + m[0][1]) / s, s / 4, (m[2][1] + m[1][2]) / s, (m[2][0] - m[0][2]) / s];
  } else {
    const s = 2 * Math.sqrt(1 + m[2][2] - m[0][0] - m[1][1]);
    q = [(m[2][0] + m[0][2]) / s, (m[2][1] + m[1][2]) / s, s / 4, (m[0][1] - m[1][0]) / s];
  }
  const norm = Math.hypot(...q);
  return q.map(value => value / norm) as Quaternion;
}

const rotate = (q: Quaternion, v: Vector3): Vector3 => {
  const t = cross([q[0], q[1], q[2]], v).map(value => 2 * value) as Vector3;
  const c = cross([q[0], q[1], q[2]], t);
  return v.map((value, axis) => value + q[3] * t[axis] + c[axis]) as Vector3;
};

const BODY = bodySites();

/** Writes O, H1, H2 as vec4 with the oxygen wrapped and the hydrogens whole. */
export function moleculeSites(centre: Vector3, quaternion: Quaternion, box: number,
    out: Float32Array, offset: number): void {
  const oxygen = rotate(quaternion, BODY.oxygen).map((value, axis) => centre[axis] + value) as Vector3;
  const wrapped = oxygen.map(value => ((value % box) + box) % box) as Vector3;
  for (const axis of [0, 1, 2]) out[offset + axis] = wrapped[axis];
  out[offset + 3] = 1;
  for (const hydrogen of [0, 1]) {
    const relative = rotate(quaternion, BODY.hydrogen[hydrogen].map((value, axis) =>
      value - BODY.oxygen[axis]) as Vector3);
    for (const axis of [0, 1, 2]) out[offset + (hydrogen + 1) * 4 + axis] = wrapped[axis] + relative[axis];
    out[offset + (hydrogen + 1) * 4 + 3] = 1;
  }
}

/** The M site, placed relative to the already wrapped oxygen. */
export function chargeSite(centre: Vector3, quaternion: Quaternion, oxygen: Vector3): Vector3 {
  const relative = rotate(quaternion, BODY.charge.map((value, axis) => value - BODY.oxygen[axis]) as Vector3);
  return oxygen.map((value, axis) => value + relative[axis]) as Vector3;
}

export function buildInitialState(packed: Float32Array,
    options: { molecules: number; densityRatio: number; temperature: number; seed: number }): InitialState {
  const { molecules, densityRatio, temperature, seed } = options;
  if (packed.length !== molecules * 9) throw new Error('Ice configuration does not match the molecule count');
  const latticeBox = ICE_CELL * cellsFor(molecules);
  const box = boxLength(molecules, densityRatio);
  const scale = box / latticeBox;
  const random = mulberry32(seed);
  const moments = principalMoments();
  const state = new Float32Array(molecules * 16);
  const sites = new Float32Array(molecules * 12);
  const charges = new Float32Array(molecules * 4);
  const centres: Vector3[] = [];
  const quaternions: Quaternion[] = [];

  for (let molecule = 0; molecule < molecules; molecule++) {
    const read = (site: number): Vector3 =>
      [packed[molecule * 9 + site * 3], packed[molecule * 9 + site * 3 + 1], packed[molecule * 9 + site * 3 + 2]];
    const oxygen = read(0), first = read(1), second = read(2);
    const centre = [0, 1, 2].map(axis =>
      (MASS_O * oxygen[axis] + MASS_H * (first[axis] + second[axis])) / MOLECULE_MASS) as Vector3;
    // Only the centres move with density; the molecule itself stays rigid.
    centres.push(centre.map(value => value * scale) as Vector3);
    quaternions.push(orientationOf(oxygen, first, second));
  }

  const velocities: Vector3[] = [], angular: Vector3[] = [];
  for (let molecule = 0; molecule < molecules; molecule++) {
    velocities.push([gaussian(random), gaussian(random), gaussian(random)]);
    angular.push([gaussian(random), gaussian(random), gaussian(random)]);
  }
  // Remove the net momentum, then scale both sets to hit the requested temperature exactly.
  const drift = [0, 1, 2].map(axis =>
    velocities.reduce((sum, velocity) => sum + velocity[axis], 0) / molecules);
  for (const velocity of velocities) for (const axis of [0, 1, 2]) velocity[axis] -= drift[axis];
  const degrees = 3 * molecules;
  const target = degrees * BOLTZMANN * temperature * FORCE_TO_ACCELERATION;
  const translational = velocities.reduce((sum, velocity) =>
    sum + MOLECULE_MASS * velocity.reduce((inner, value) => inner + value * value, 0), 0);
  const rotational = angular.reduce((sum, momentum) =>
    sum + momentum.reduce((inner, value, axis) => inner + value * value / moments[axis], 0), 0);
  const translationScale = Math.sqrt(target / translational);
  // Each L component is drawn from one unit normal and scaled by this single common factor,
  // so per-axis variance is not I_k kB T; the thermostat's noise term corrects the per-axis
  // distribution within the first picosecond, and only the total rotational temperature matters here.
  const rotationScale = Math.sqrt(target / rotational);

  for (let molecule = 0; molecule < molecules; molecule++) {
    const base = molecule * 16;
    for (const axis of [0, 1, 2]) {
      state[base + axis] = centres[molecule][axis];
      state[base + 8 + axis] = velocities[molecule][axis] * translationScale;
      state[base + 12 + axis] = angular[molecule][axis] * rotationScale;
    }
    state[base + 3] = 1;
    for (let component = 0; component < 4; component++) state[base + 4 + component] = quaternions[molecule][component];
    moleculeSites(centres[molecule], quaternions[molecule], box, sites, molecule * 12);
    const oxygen: Vector3 = [sites[molecule * 12], sites[molecule * 12 + 1], sites[molecule * 12 + 2]];
    const charge = chargeSite(centres[molecule], quaternions[molecule], oxygen);
    for (const axis of [0, 1, 2]) charges[molecule * 4 + axis] = charge[axis];
    charges[molecule * 4 + 3] = 1;
  }
  return { molecules, box, state, sites, charges };
}
