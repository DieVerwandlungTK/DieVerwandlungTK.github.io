import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';

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
