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
  const error = (actual: number[], expected: number[]) => {
    const scale = Math.max(...expected.map(Math.abs));
    const worst = Math.max(...expected.map((value, index) => Math.abs(value - actual[index])));
    return worst / scale;
  };
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
  // Count oxygen coordinates that must wrap around the box edge once shifted: the wrapped
  // oxygen (used for lever arms before the fix) then differs from a naive unwrapped shift by a
  // whole box vector, exactly the scenario the fix must handle.
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
  const error = (actual: number[], expected: number[]) => {
    const scale = Math.max(...expected.map(Math.abs));
    const worst = Math.max(...expected.map((value, index) => Math.abs(value - actual[index])));
    return worst / scale;
  };
  expect(error(result.forces, fixture.forces)).toBeLessThan(1e-3);
  expect(error(result.torques, fixture.torques)).toBeLessThan(1e-3);
});
