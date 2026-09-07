import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { BOLTZMANN, FORCE_TO_ACCELERATION, MOLECULE_MASS, OH_LENGTH, principalMoments } from '../../src/water-model';

interface Fixture {
  box: number; cutoff: number; molecules: number;
  sites: number[]; forces: number[]; torques: number[]; potentialEnergy: number;
}

async function ready(page: import('@playwright/test').Page) {
  await page.goto('/');
  await expect(page.locator('.molecular-stage')).toHaveClass(/ready/, { timeout: 20000 });
  await page.waitForFunction(() => Boolean((window as any).waterSimulation));
}

/** Worst-case relative error of `actual` against `expected`, scaled by the expected magnitude. */
function error(actual: number[], expected: number[]): number {
  const scale = Math.max(...expected.map(Math.abs));
  const worst = Math.max(...expected.map((value, index) => Math.abs(value - actual[index])));
  return worst / scale;
}

test('GPU forces and torques match the Python reference within f32 precision', async ({ page }) => {
  const fixture: Fixture = JSON.parse(await readFile('tests/fixtures/reference-forces-64.json', 'utf8'));
  await ready(page);
  const result = await page.evaluate(async data => {
    const simulation = (window as any).waterSimulation;
    simulation.setMolecules(data.molecules);
    simulation.setDensity(1);
    const { forces, torques } = await simulation.evaluateForces(new Float32Array(data.sites));
    return { forces: Array.from(forces as Float32Array), torques: Array.from(torques as Float32Array) };
  }, fixture);
  expect(result.forces.length).toBe(fixture.molecules * 3);
  expect(error(result.forces, fixture.forces)).toBeLessThan(1e-3);
  expect(error(result.torques, fixture.torques)).toBeLessThan(1e-3);
});

test('forces and torques are invariant under a whole-system translation across a box face', async ({ page }) => {
  const fixture: Fixture = JSON.parse(await readFile('tests/fixtures/reference-forces-64.json', 'utf8'));
  // Shift every site of every molecule by the same vector, half the box on each axis. Forces
  // and torques in a periodic system do not depend on which periodic image a molecule sits in,
  // so the shifted configuration must reproduce the same reference values. This exercises the
  // wrap boundary that a per-molecule offset (e.g. Task 6's integrator wrapping centres of mass
  // independently) would also cross: some oxygens land on the far side of [0, box) from where
  // their molecule's centre of mass would naively be.
  const box = fixture.box;
  const offset = box / 2;
  const translated = fixture.sites.map(value => value + offset);
  // Count oxygen coordinates that cross the box edge once shifted, as a proxy for the actual
  // failure mechanism this test exercises: a molecule's centre of mass landing outside [0, box)
  // while its oxygen gets wrapped back in, so a lever arm built from the wrapped oxygen would
  // differ from the unwrapped centre by a whole box vector. A nonzero count here confirms the
  // fixture actually forces that scenario, before the assertions below check it has no effect.
  let crossings = 0;
  for (let molecule = 0; molecule < fixture.molecules; molecule++) {
    for (const axis of [0, 1, 2]) {
      const before = ((fixture.sites[molecule * 9 + axis] % box) + box) % box;
      if (before + offset >= box) crossings++;
    }
  }
  expect(crossings).toBeGreaterThan(0);

  await ready(page);
  const result = await page.evaluate(async data => {
    const simulation = (window as any).waterSimulation;
    simulation.setMolecules(data.molecules);
    simulation.setDensity(1);
    const { forces, torques } = await simulation.evaluateForces(new Float32Array(data.sites));
    return { forces: Array.from(forces as Float32Array), torques: Array.from(torques as Float32Array) };
  }, { molecules: fixture.molecules, sites: translated });
  expect(result.forces.length).toBe(fixture.molecules * 3);
  expect(error(result.forces, fixture.forces)).toBeLessThan(1e-3);
  expect(error(result.torques, fixture.torques)).toBeLessThan(1e-3);
});

/** Kinetic temperatures computed from the raw state, the way the reduction kernel will. */
function temperatures(state: number[], molecules: number) {
  const inertia = principalMoments();
  let translational = 0, rotational = 0;
  const momentum = [0, 0, 0];
  for (let molecule = 0; molecule < molecules; molecule++) {
    for (const axis of [0, 1, 2]) {
      const velocity = state[molecule * 16 + 8 + axis];
      const angular = state[molecule * 16 + 12 + axis];
      translational += MOLECULE_MASS * velocity * velocity;
      rotational += angular * angular / inertia[axis];
      momentum[axis] += MOLECULE_MASS * velocity;
    }
  }
  const degrees = 3 * molecules * BOLTZMANN * FORCE_TO_ACCELERATION;
  return { translational: translational / degrees, rotational: rotational / degrees, momentum };
}

test('molecules stay rigid after five thousand steps', async ({ page }) => {
  test.setTimeout(120000);
  await ready(page);
  const result = await page.evaluate(async LENGTH => {
    const simulation = (window as any).waterSimulation;
    simulation.setMolecules(64);
    simulation.setTemperature(300);
    for (let batch = 0; batch < 50; batch++) simulation.step(100);
    const sites = await simulation.readSites();
    const state = Array.from(await simulation.readState() as Float32Array);
    let worstBond = 0, worstAngle = 0, worstQuaternionNorm = 0;
    for (let molecule = 0; molecule < simulation.molecules; molecule++) {
      const site = (index: number) => [0, 1, 2].map(axis => sites[molecule * 12 + index * 4 + axis]);
      const oxygen = site(0);
      const bonds = [1, 2].map(index => site(index).map((value, axis) => value - oxygen[axis]));
      for (const bond of bonds) worstBond = Math.max(worstBond, Math.abs(Math.hypot(...bond) - LENGTH));
      const cosine = bonds[0].reduce((sum, value, axis) => sum + value * bonds[1][axis], 0) / LENGTH ** 2;
      worstAngle = Math.max(worstAngle, Math.abs(Math.acos(Math.max(-1, Math.min(1, cosine))) * 180 / Math.PI - 104.52));
      const quaternion = [0, 1, 2, 3].map(axis => state[molecule * 16 + 4 + axis]);
      worstQuaternionNorm = Math.max(worstQuaternionNorm, Math.abs(Math.hypot(...quaternion) - 1));
    }
    return { worstBond, worstAngle, worstQuaternionNorm, timePs: simulation.timePs };
  }, OH_LENGTH);
  expect(result.timePs).toBeCloseTo(10, 6);
  // `writeSites` builds bonds as quatRotate(q, BODY_H - BODY_O), so bond length and angle can
  // only deviate through the quaternion losing unit norm (or going non-finite) — these two
  // assertions are a geometry cross-check, but the quaternion-norm assertion below is the one
  // that directly tests what can actually go wrong, at a bound tight enough to catch it.
  expect(result.worstBond).toBeLessThan(0.001);
  expect(result.worstAngle).toBeLessThan(0.05);
  expect(result.worstQuaternionNorm).toBeLessThan(1e-5);
});

test('the thermostat brings the sample to its set point and holds it', async ({ page }) => {
  test.setTimeout(180000);
  await ready(page);
  const samples = await page.evaluate(async () => {
    const simulation = (window as any).waterSimulation;
    simulation.setMolecules(216);
    simulation.setTemperature(300);
    // Two picoseconds of equilibration at 5 /ps friction, then twenty samples over 10 ps.
    // Twenty samples (vs. five) halve the standard error on the mean for about two extra
    // seconds of runtime, without needing to widen the 275-325 K window below.
    for (let batch = 0; batch < 10; batch++) simulation.step(100);
    const collected: number[][] = [];
    for (let sample = 0; sample < 20; sample++) {
      simulation.step(250);
      collected.push(Array.from(await simulation.readState() as Float32Array));
    }
    return { collected, molecules: simulation.molecules };
  });
  const measured = samples.collected.map(state => temperatures(state, samples.molecules));
  const mean = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / values.length;
  expect(mean(measured.map(entry => entry.translational))).toBeGreaterThan(275);
  expect(mean(measured.map(entry => entry.translational))).toBeLessThan(325);
  expect(mean(measured.map(entry => entry.rotational))).toBeGreaterThan(275);
  expect(mean(measured.map(entry => entry.rotational))).toBeLessThan(325);
});

test('the sample never acquires a net drift', async ({ page }) => {
  await ready(page);
  const drift = await page.evaluate(async () => {
    const simulation = (window as any).waterSimulation;
    simulation.setMolecules(64);
    simulation.setTemperature(450);
    for (let batch = 0; batch < 20; batch++) simulation.step(100);
    const state = Array.from(await simulation.readState() as Float32Array);
    return { state, molecules: simulation.molecules };
  });
  const { momentum, translational } = temperatures(drift.state, drift.molecules);
  // A Langevin thermostat does not conserve momentum exactly; it must stay small next
  // to the thermal momentum of a single molecule, sqrt(m kB T / 100) in amu A/ps.
  // `momentum[axis]` is a sum of `molecules` independent draws, so its standard deviation is
  // exactly `thermal * sqrt(molecules)`. A 3-sigma bound fails 0.27% of the time per axis, about
  // 0.81% across all three axes on the random seed drawn each page load — flaky enough to be
  // worth tightening. 4 sigma drops that to 0.006% per axis while staying far below the signal
  // this test actually guards against: correlated noise across molecules, which would produce a
  // momentum of order `molecules * thermal`, about 8x the bound below at 64 molecules.
  const thermal = Math.sqrt(MOLECULE_MASS * BOLTZMANN * translational * FORCE_TO_ACCELERATION);
  for (const axis of [0, 1, 2]) {
    expect(Math.abs(momentum[axis])).toBeLessThan(4 * thermal * Math.sqrt(drift.molecules));
  }
  expect(Number.isFinite(translational)).toBe(true);
});
