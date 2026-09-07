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

// Checks the energy *expression* the force kernel accumulates on a fixed configuration -- it is
// not a sentinel for a dynamics bug, since a wrong potential could still integrate stably.
test('the accumulated potential energy matches the Python reference on the fixture configuration', async ({ page }) => {
  const fixture: Fixture = JSON.parse(await readFile('tests/fixtures/reference-forces-64.json', 'utf8'));
  await ready(page);
  const potentialEnergy = await page.evaluate(async data => {
    const simulation = (window as any).waterSimulation;
    simulation.setMolecules(data.molecules);
    simulation.setDensity(1);
    const result = await simulation.evaluateForces(new Float32Array(data.sites));
    return result.potentialEnergy as number;
  }, fixture);
  expect(Math.abs(potentialEnergy - fixture.potentialEnergy) / Math.abs(fixture.potentialEnergy)).toBeLessThan(1e-3);
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

// Named for what this actually checks, not for "rigidity": `writeSites` rebuilds both hydrogens
// from the body-frame constants via quatRotate(q, BODY_H - BODY_O), and `freeRotation`
// normalises the quaternion every sub-step, so the bond length, the H-O-H angle and the
// quaternion norm are algebraically fixed by construction -- they cannot drift unless the
// quaternion itself has gone non-finite (NaN or Inf). This is a NaN/divergence detector wearing
// a rigidity label; the test below this one exercises something that can genuinely drift.
test('geometry reconstruction and the quaternion stay finite after five thousand steps', async ({ page }) => {
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

test('the body-frame angular momentum magnitude holds under torque-free rotation', async ({ page }) => {
  test.setTimeout(60000);
  await ready(page);
  const STEPS = 500;
  const result = await page.evaluate(async steps => {
    const simulation = (window as any).waterSimulation;
    simulation.setMolecules(64);
    // A density this low pushes every molecule's nearest neighbour far outside the 9 A cutoff, so
    // the force kernel finds no pairs within range and every molecule feels exactly zero torque
    // and zero force -- an exact torque-free run for all 64 molecules at once, without needing a
    // dedicated one-molecule configuration (setMolecules only accepts 64/216/512).
    simulation.setDensity(1e-9);
    // Zero temperature makes the thermostat's noise term exactly zero (it scales with sqrt(T)),
    // leaving only its deterministic friction decay -- see the expected-value comment below.
    simulation.setTemperature(0);
    const magnitudes = async () => {
      const state = Array.from(await simulation.readState() as Float32Array);
      return Array.from({ length: simulation.molecules }, (_, molecule) =>
        Math.hypot(...[0, 1, 2].map(axis => state[molecule * 16 + 12 + axis])));
    };
    const initial = await magnitudes();
    simulation.step(steps);
    const final = await magnitudes();
    return { initial, final };
  }, STEPS);
  // Mirrors src/simulation.ts's TIME_STEP and FRICTION (not exported, like OH_LENGTH elsewhere in
  // this suite). With zero torque and zero thermostat noise, every step's O-stage scales the
  // whole angular-momentum vector by exactly this factor -- a genuine physical change, not a bug
  // -- and the two torque-free drifts either side of it (freeRotation) must leave the magnitude
  // exactly unchanged. That conservation, not the friction decay, is what this test pins down.
  const TIME_STEP = 0.002, FRICTION = 5;
  const expectedRatio = Math.exp(-FRICTION * TIME_STEP) ** STEPS;
  for (let molecule = 0; molecule < result.initial.length; molecule++) {
    const actualRatio = result.final[molecule] / result.initial[molecule];
    // A pre-fix regression (see .superpowers/sdd/task-6-report.md) estimated freeRotation
    // inflating ||L_body|| by about 1.81e-5 relative per call, two calls per step, and
    // extrapolated that flat rate over 500 steps (1000 calls) to a ~1.8% excess. That estimate
    // treats the inflation as constant, but the Euler drift freeRotation accumulates each sub-step
    // is second order in ||L||, and T=0 here means the thermostat's friction alone shrinks ||L||
    // by exp(-FRICTION * TIME_STEP * STEPS) = exp(-5) over the run -- a cold set point, not a hot
    // one. The drift collapses along with ||L|| far faster than the flat-rate estimate assumes, so
    // margin has to be tight to actually catch the bug rather than generously sized around the old
    // (wrong) figure: measuring the real regression on this branch (freeRotation's magnitude
    // restore removed) gives a worst deviation of 2.54e-3 -- comfortably past a loose margin like
    // the old 0.005, but the fixed code measures 5.95e-6, so 1e-4 sits about 17x above the
    // post-fix f32 floor and about 25x below the actual regression (see this task's report).
    expect(Math.abs(actualRatio / expectedRatio - 1)).toBeLessThan(1e-4);
  }
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

// The bound above (0 < maximumForce < 50000) is a 5000x-wide ceiling next to the ~10 kJ/mol/A an
// equilibrium sample actually produces -- wide enough to hide a badly broken reduction. This pins
// the exact value the reduce kernel's max() tree should compute, cross-checked host-side against
// the same GPU-computed forces (evaluateForces already validates those elementwise against the
// Python reference in the fixture test above), so it isolates the reduction step itself.
test('the maximum-force statistic matches a host-computed max over the same GPU forces', async ({ page }) => {
  const fixture: Fixture = JSON.parse(await readFile('tests/fixtures/reference-forces-64.json', 'utf8'));
  await ready(page);
  const result = await page.evaluate(async data => {
    const simulation = (window as any).waterSimulation;
    simulation.setMolecules(data.molecules);
    simulation.setDensity(1);
    const { forces } = await simulation.evaluateForces(new Float32Array(data.sites));
    // evaluateForces leaves the forceTorque buffer holding these same forces (only state, sites
    // and charges are restored afterwards), so the reduce kernel dispatched by readStats() below
    // computes its max over exactly the values read back here.
    const stats = await simulation.readStats();
    return { forces: Array.from(forces as Float32Array), maximumForce: stats.maximumForce };
  }, fixture);
  let expected = 0;
  for (let molecule = 0; molecule < fixture.molecules; molecule++) {
    const magnitude = Math.hypot(...[0, 1, 2].map(axis => result.forces[molecule * 3 + axis]));
    expected = Math.max(expected, magnitude);
  }
  expect(expected).toBeGreaterThan(0);
  expect(Math.abs(result.maximumForce - expected) / expected).toBeLessThan(1e-3);
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

// setDensity(15) above saturates every molecule to NaN within a couple of steps, so it cannot
// tell a correct reduction from one that, say, only ORs together half of the max()-tree or drops
// the last lane's second stride pass -- every lane's every molecule is broken regardless. This
// isolates exactly one molecule, at the highest index, so only the specific lane and specific
// stride iteration that molecule falls into carries a non-finite value.
test('the divergence guard fires from a single poisoned molecule at the highest index', async ({ page }) => {
  test.setTimeout(30000);
  await ready(page);
  const result = await page.evaluate(async () => {
    const simulation = (window as any).waterSimulation;
    // 512 molecules exceeds the reduce kernel's 256 lanes, so the stride loop
    // (`for (var i = lane; i < molecules; i += 256)`) runs twice per lane, and molecule 511 is
    // reached only on a lane's *second* iteration -- the shape of loop the brief calls out.
    simulation.setMolecules(512);
    simulation.setDensity(1);
    simulation.step(5);
    const healthyBefore = await simulation.readStats();
    const last = simulation.molecules - 1;
    const state = Array.from(await simulation.readState() as Float32Array);
    const poisoned = state.slice(last * 16, last * 16 + 16);
    poisoned[8] = Number.NaN;   // velocity.x of the highest-index molecule only
    await simulation.pokeState(last, poisoned);
    const poisonedStats = await simulation.readStats();
    return { healthyBefore, poisonedStats };
  });
  expect(result.healthyBefore.nonFinite).toBe(false);
  expect(result.poisonedStats.nonFinite).toBe(true);
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
    simulation.setDensity(1);
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

/**
 * Bins raw oxygen-oxygen distances across `frames` into the O-O radial distribution function,
 * with the same bin width, range and normalisation `scripts/generate_reference_rdf.py` uses, so
 * the result is directly comparable to a fixture it produced.
 */
function radialDistribution(frames: number[][], molecules: number, box: number) {
  const width = 0.05, start = 2, bins = new Array(80).fill(0);
  for (const sites of frames) {
    for (let i = 0; i < molecules; i++) for (let j = i + 1; j < molecules; j++) {
      const distance = Math.hypot(...[0, 1, 2].map(axis => {
        const raw = sites[j * 12 + axis] - sites[i * 12 + axis];
        return raw - box * Math.round(raw / box);
      }));
      const bin = Math.floor((distance - start) / width);
      if (bin >= 0 && bin < bins.length) bins[bin]++;
    }
  }
  const density = molecules / box ** 3;
  const gr = bins.map((count, index) => {
    const inner = start + index * width, outer = inner + width;
    const shell = 4 / 3 * Math.PI * (outer ** 3 - inner ** 3);
    return count / (shell * density * molecules / 2 * frames.length);
  });
  const near = gr.slice(0, Math.floor((4 - start) / width));
  const peakIndex = near.indexOf(Math.max(...near));
  const peak = start + width * (peakIndex + 0.5);
  // First minimum: the smallest gr value between the first peak and 4 A, matching how the
  // reference fixture's own `firstMinimum` is derived. This is the diagnostic that separates
  // crystalline order (near 0.03) from a disordered fluid (near 0.85-0.95) -- the first-peak
  // *position* cannot, since both ice and liquid water put it near 2.7-2.8 A.
  const tail = near.slice(peakIndex);
  const firstMinimum = Math.min(...tail);
  return { gr, peak, peakHeight: near[peakIndex], firstMinimum };
}

test('the first oxygen shell agrees with the OpenMM reference for the same superheated ice', async ({ page }) => {
  test.setTimeout(300000);
  const reference = JSON.parse(await readFile('tests/fixtures/reference-rdf-ice-300k.json', 'utf8'));
  await ready(page);
  const measured = await page.evaluate(async () => {
    const simulation = (window as any).waterSimulation;
    simulation.setMolecules(216);
    simulation.setDensity(1);
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
  const { peak, peakHeight, firstMinimum } =
    radialDistribution(measured.frames, measured.molecules, measured.box);
  expect(Math.abs(peak - reference.firstPeak)).toBeLessThan(0.15);
  expect(peakHeight).toBeGreaterThan(1.8);
  // The 300 K sample stays crystalline (ice, not liquid water -- see the fixture's docstring
  // reference), so its first minimum sits close to zero (reference 0.026) rather than the
  // 0.8-0.9 a disordered fluid shows. A margin of 0.15 stays two-plus reference-magnitudes away
  // from any melted reading while comfortably covering run-to-run sampling noise: the reference
  // itself moved from 0.026 to 0.031 across two otherwise-identical OpenMM regenerations on this
  // machine (see the report), a spread of 0.005, thirty times smaller than this margin.
  expect(Math.abs(firstMinimum - reference.firstMinimum)).toBeLessThan(0.15);
});

test('the first oxygen shell agrees with the OpenMM reference for the same disordered fluid at 520 K', async ({ page }) => {
  test.setTimeout(300000);
  const reference = JSON.parse(await readFile('tests/fixtures/reference-rdf-fluid-520k.json', 'utf8'));
  await ready(page);
  const measured = await page.evaluate(async () => {
    const simulation = (window as any).waterSimulation;
    simulation.setMolecules(216);
    simulation.setDensity(1);
    simulation.reset();
    simulation.setTemperature(520);
    // The melting test reaches a tetrahedral order of 0.48 after 20 ps at 520 K; equilibrate for
    // 30 ps here (150 batches of 100 steps) so the sample is well past that and settled before
    // sampling its structure, then sample 20 frames over 10 ps exactly as the ice-state test does.
    for (let batch = 0; batch < 150; batch++) simulation.step(100);   // 30 ps of equilibration
    const frames: number[][] = [];
    for (let sample = 0; sample < 20; sample++) {
      simulation.step(250);
      frames.push(Array.from(await simulation.readSites() as Float32Array));
    }
    return { frames, molecules: simulation.molecules, box: simulation.box };
  });
  const { peak, firstMinimum } = radialDistribution(measured.frames, measured.molecules, measured.box);
  expect(Math.abs(peak - reference.firstPeak)).toBeLessThan(0.15);
  // The reference's first minimum (0.934) sits in the shallow, broad dip typical of a fluid at
  // this temperature and box size -- nowhere near ice's 0.03. A margin of 0.3 stays far above
  // the noise floor a small, high-temperature, 64-molecule OpenMM sample and a 216-molecule
  // browser sample can plausibly disagree by, while still failing hard (by more than an order of
  // magnitude) if the browser sample were actually still crystalline.
  expect(Math.abs(firstMinimum - reference.firstMinimum)).toBeLessThan(0.3);
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
