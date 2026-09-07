import { BOLTZMANN, FORCE_TO_ACCELERATION, MOLECULE_MASS } from './water-model';

export const SPEED_BINS = 12;
export const SPEED_AXIS_MAX = 18;
export const DENSITY_AXIS_MAX = .26;
export const SMOOTHING = 1 / 3;
type Histogram = Float32Array | Float64Array;

/** Speed density in ps/Å, with translational velocity measured in Å/ps. */
export function maxwellBoltzmann(speed: number, temperature: number): number {
  if (!Number.isFinite(speed) || speed < 0 || !Number.isFinite(temperature) || temperature <= 0) return 0;
  const a = MOLECULE_MASS / (2 * BOLTZMANN * FORCE_TO_ACCELERATION * temperature);
  return 4 / Math.sqrt(Math.PI) * a ** 1.5 * speed ** 2 * Math.exp(-a * speed ** 2);
}

export function mostProbableSpeed(temperature: number): number {
  return Number.isFinite(temperature) && temperature > 0
    ? Math.sqrt(2 * BOLTZMANN * FORCE_TO_ACCELERATION * temperature / MOLECULE_MASS) : 0;
}

/** Read only COM velocity xyz (8–10); neither atom velocities nor angular momentum. */
export function speedHistogram(state: Float32Array, molecules: number, out: Histogram): void {
  if (!Number.isInteger(molecules) || molecules < 1 || state.length !== molecules * 16 || out.length !== SPEED_BINS) {
    throw new Error('Speed histogram requires matching molecular state and twelve bins');
  }
  const width = SPEED_AXIS_MAX / SPEED_BINS;
  out.fill(0);
  for (let molecule = 0; molecule < molecules; molecule++) {
    const offset = molecule * 16 + 8;
    const speed = Math.hypot(state[offset], state[offset + 1], state[offset + 2]);
    if (!Number.isFinite(speed)) throw new Error('Nonfinite translational velocity');
    out[Math.min(SPEED_BINS - 1, Math.floor(speed / width))] += 1;
  }
  for (let bin = 0; bin < SPEED_BINS; bin++) out[bin] /= molecules * width;
}

/** Normalize the first sample by copying it; apply this only to subsequent samples. */
export function smoothHistogram(current: Histogram, sample: Histogram): void {
  if (current.length !== SPEED_BINS || sample.length !== SPEED_BINS) throw new Error('Expected twelve bins');
  for (let bin = 0; bin < SPEED_BINS; bin++) current[bin] += SMOOTHING * (sample[bin] - current[bin]);
}
