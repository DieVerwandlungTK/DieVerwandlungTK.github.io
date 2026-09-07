import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { BOLTZMANN, FORCE_TO_ACCELERATION, MOLECULE_MASS, OH_LENGTH, principalMoments } from '../../src/water-model';
import { needsRestart } from '../../src/simulation-health';

interface Fixture {
  box: number; cutoff: number; molecules: number;
  sites: number[]; forces: number[]; torques: number[]; potentialEnergy: number;
}

async function ready(page: import('@playwright/test').Page) {
  await page.goto('/');
  await expect(page.locator('.molecular-stage')).toHaveClass(/ready/, { timeout: 20000 });
  await page.waitForFunction(() => Boolean((window as any).waterSimulation));
  // The panel drives its own render loop over this same simulation, so pause it: otherwise the
  // page's steps interleave with the ones each test takes, and `timePs` and `readStats` drift.
  await page.getByRole('button', { name: '計算を一時停止' }).click();
  await expect(page.locator('#playback-status')).toHaveText('一時停止中');
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
  // momentum of order `molecules * thermal`, twice the bound below at 64 molecules.
  const thermal = Math.sqrt(MOLECULE_MASS * BOLTZMANN * translational * FORCE_TO_ACCELERATION);
  for (const axis of [0, 1, 2]) {
    expect(Math.abs(momentum[axis])).toBeLessThan(4 * thermal * Math.sqrt(drift.molecules));
  }
  expect(Number.isFinite(translational)).toBe(true);
});

test('the reduction kernel reports the same temperatures as the raw state', async ({ page }) => {
  await ready(page);
  const result = await page.evaluate(async () => {
    const simulation = (window as any).waterSimulation;
    simulation.setMolecules(216);
    simulation.setTemperature(320);
    for (let batch = 0; batch < 10; batch++) simulation.step(100);
    const stats = await simulation.readStats();
    const state = Array.from(await simulation.readState() as Float32Array);
    return { stats, state, molecules: simulation.molecules };
  });
  const expected = temperatures(result.state, result.molecules);
  expect(result.stats.translationalTemperature).toBeGreaterThan(expected.translational - 1);
  expect(result.stats.translationalTemperature).toBeLessThan(expected.translational + 1);
  expect(result.stats.rotationalTemperature).toBeGreaterThan(expected.rotational - 1);
  expect(result.stats.rotationalTemperature).toBeLessThan(expected.rotational + 1);
  expect(result.stats.nonFinite).toBe(false);
  expect(result.stats.maximumForce).toBeGreaterThan(0);
  expect(result.stats.maximumForce).toBeLessThan(50000);
});

test('the divergence guard survives NaN saturation and recovers after a restart', async ({ page }) => {
  test.setTimeout(60000);
  await ready(page);
  const result = await page.evaluate(async () => {
    const simulation = (window as any).waterSimulation;
    simulation.setMolecules(64);
    // A density this extreme forces a genuine blow-up: the trajectory explodes within about
    // two steps and every molecule's velocity, angular momentum and force saturate to NaN.
    // Stepping to 20 is well past the ~3 steps that took to saturate when this test was written.
    simulation.setDensity(15);
    simulation.step(20);
    const diverged = await simulation.readStats();
    // reset() reloads the ice lattice but keeps whatever density was last set, so the extreme
    // density from above survives it; setDensity(1) afterwards is what actually restores a
    // sane box.
    simulation.reset();
    simulation.setDensity(1);
    simulation.step(50);
    const healthy = await simulation.readStats();
    return { diverged, healthy };
  });
  expect(result.diverged.nonFinite).toBe(true);
  expect(needsRestart(result.diverged)).toBe(true);
  expect(result.healthy.nonFinite).toBe(false);
  expect(needsRestart(result.healthy)).toBe(false);
});

/** Fraction of ideal tetrahedral order over the four nearest oxygen neighbours. */
function tetrahedralOrder(sites: number[], molecules: number, box: number): number {
  const oxygen = (index: number) => [0, 1, 2].map(axis => sites[index * 12 + axis]);
  let total = 0;
  for (let i = 0; i < molecules; i++) {
    const here = oxygen(i);
    const neighbours: { distance: number; direction: number[] }[] = [];
    for (let j = 0; j < molecules; j++) {
      if (i === j) continue;
      const delta = oxygen(j).map((value, axis) => {
        const raw = value - here[axis];
        return raw - box * Math.round(raw / box);
      });
      neighbours.push({ distance: Math.hypot(...delta), direction: delta });
    }
    neighbours.sort((a, b) => a.distance - b.distance);
    const nearest = neighbours.slice(0, 4).map(entry =>
      entry.direction.map(value => value / entry.distance));
    let order = 1;
    for (let a = 0; a < 4; a++) for (let b = a + 1; b < 4; b++) {
      const cosine = nearest[a].reduce((sum, value, axis) => sum + value * nearest[b][axis], 0);
      order -= 3 / 8 * (cosine + 1 / 3) ** 2;
    }
    total += order;
  }
  return total / molecules;
}

test('heating destroys the tetrahedral lattice while cold holds it', async ({ page }) => {
  test.setTimeout(300000);
  await ready(page);
  const run = (kelvin: number) => page.evaluate(async temperature => {
    const simulation = (window as any).waterSimulation;
    simulation.setMolecules(216);
    simulation.reset();
    simulation.setTemperature(temperature);
    for (let batch = 0; batch < 100; batch++) simulation.step(100);   // 20 ps
    return { sites: Array.from(await simulation.readSites() as Float32Array),
      molecules: simulation.molecules, box: simulation.box };
  }, kelvin);
  const cold = await run(180);
  const hot = await run(520);
  const coldOrder = tetrahedralOrder(cold.sites, cold.molecules, cold.box);
  const hotOrder = tetrahedralOrder(hot.sites, hot.molecules, hot.box);
  expect(coldOrder).toBeGreaterThan(0.85);
  expect(hotOrder).toBeLessThan(0.7);
  expect(coldOrder - hotOrder).toBeGreaterThan(0.2);
});

test('the first oxygen shell agrees with the OpenMM reference', async ({ page }) => {
  test.setTimeout(300000);
  const reference = JSON.parse(await readFile('tests/fixtures/reference-rdf-300k.json', 'utf8'));
  await ready(page);
  const measured = await page.evaluate(async () => {
    const simulation = (window as any).waterSimulation;
    simulation.setMolecules(216);
    simulation.reset();
    simulation.setTemperature(300);
    for (let batch = 0; batch < 150; batch++) simulation.step(100);   // 30 ps of equilibration
    const frames: number[][] = [];
    for (let sample = 0; sample < 20; sample++) {
      simulation.step(250);
      frames.push(Array.from(await simulation.readSites() as Float32Array));
    }
    return { frames, molecules: simulation.molecules, box: simulation.box };
  });
  const width = 0.05, start = 2, bins = new Array(80).fill(0);
  for (const sites of measured.frames) {
    for (let i = 0; i < measured.molecules; i++) for (let j = i + 1; j < measured.molecules; j++) {
      const distance = Math.hypot(...[0, 1, 2].map(axis => {
        const raw = sites[j * 12 + axis] - sites[i * 12 + axis];
        return raw - measured.box * Math.round(raw / measured.box);
      }));
      const bin = Math.floor((distance - start) / width);
      if (bin >= 0 && bin < bins.length) bins[bin]++;
    }
  }
  const density = measured.molecules / measured.box ** 3;
  const gr = bins.map((count, index) => {
    const inner = start + index * width, outer = inner + width;
    const shell = 4 / 3 * Math.PI * (outer ** 3 - inner ** 3);
    return count / (shell * density * measured.molecules / 2 * measured.frames.length);
  });
  const near = gr.slice(0, Math.floor((4 - start) / width));
  const peak = start + width * (near.indexOf(Math.max(...near)) + 0.5);
  expect(Math.abs(peak - reference.firstPeak)).toBeLessThan(0.15);
  expect(Math.max(...near)).toBeGreaterThan(1.8);
});

test('the largest sample runs for half a minute without diverging', async ({ page }) => {
  test.setTimeout(120000);
  await ready(page);
  const result = await page.evaluate(async () => {
    const simulation = (window as any).waterSimulation;
    simulation.setMolecules(512);
    simulation.setTemperature(400);
    const deadline = performance.now() + 30000;
    let steps = 0;
    while (performance.now() < deadline) {
      simulation.step(16);
      steps += 16;
      await new Promise(resolve => requestAnimationFrame(resolve));
    }
    return { steps, stats: await simulation.readStats() };
  });
  expect(result.stats.nonFinite).toBe(false);
  expect(result.stats.maximumForce).toBeLessThan(50000);
  expect(result.stats.translationalTemperature).toBeGreaterThan(300);
  expect(result.stats.translationalTemperature).toBeLessThan(500);
  expect(result.steps).toBeGreaterThan(1000);
});

test('the default sample sustains a usable step rate on this machine', async ({ page }) => {
  test.setTimeout(60000);
  await ready(page);
  const rate = await page.evaluate(async () => {
    const simulation = (window as any).waterSimulation;
    simulation.setMolecules(216);
    simulation.step(200);
    await simulation.readStats();
    const started = performance.now();
    let steps = 0;
    while (performance.now() - started < 3000) { simulation.step(16); steps += 16; }
    await simulation.readStats();
    return steps / ((performance.now() - started) / 1000);
  });
  // 300 steps/s is 0.6 ps/s: slow but still watchable. Local GPUs reach several times this.
  expect(rate).toBeGreaterThan(300);
});
