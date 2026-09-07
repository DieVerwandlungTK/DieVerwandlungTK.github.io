// No imports: this module must load under plain Node (no Vite, no ?raw shader import),
// since tests/simulation-guard.test.ts exercises it without a GPU.
export interface SimulationStats {
  translationalTemperature: number;
  rotationalTemperature: number;
  maximumForce: number;
  nonFinite: boolean;
}

/** Forces above this mean the f32 integration has diverged, not that water is hot. */
export const FORCE_LIMIT = 5e4;

export const needsRestart = (stats: SimulationStats): boolean =>
  stats.nonFinite || !Number.isFinite(stats.maximumForce) || stats.maximumForce > FORCE_LIMIT;
