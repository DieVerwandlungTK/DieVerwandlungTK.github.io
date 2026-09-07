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
