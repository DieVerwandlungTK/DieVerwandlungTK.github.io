import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { BOLTZMANN, FORCE_TO_ACCELERATION, MASS_H, MASS_O, MOLECULE_MASS, OH_LENGTH, HOH_ANGLE,
  boxLength, principalMoments } from '../src/water-model.ts';
import { buildInitialState, moleculeSites, orientationOf } from '../src/initial-state.ts';

const packed = (molecules: number) => {
  const bytes = readFileSync(`public/data/ice-${molecules}.bin`);
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
};
const build = (molecules: number, densityRatio = 1, temperature = 180, seed = 7) =>
  buildInitialState(packed(molecules), { molecules, densityRatio, temperature, seed });

test('orientation round-trips through the site builder', () => {
  const half = HOH_ANGLE / 2, along = OH_LENGTH * Math.cos(half), across = OH_LENGTH * Math.sin(half);
  // A molecule rotated 90 degrees about z: the bisector points along +y.
  const oxygen: [number, number, number] = [5, 5, 5];
  const first: [number, number, number] = [5 - across, 5 + along, 5];
  const second: [number, number, number] = [5 + across, 5 + along, 5];
  const quaternion = orientationOf(oxygen, first, second);
  const centre: [number, number, number] = [
    (MASS_O * oxygen[0] + MASS_H * (first[0] + second[0])) / MOLECULE_MASS,
    (MASS_O * oxygen[1] + MASS_H * (first[1] + second[1])) / MOLECULE_MASS,
    (MASS_O * oxygen[2] + MASS_H * (first[2] + second[2])) / MOLECULE_MASS];
  const out = new Float32Array(12);
  moleculeSites(centre, quaternion, 10, out, 0);
  for (const axis of [0, 1, 2]) assert.ok(Math.abs(out[axis] - oxygen[axis]) < 1e-5, `oxygen axis ${axis}`);
  const rebuilt = [[out[4], out[5], out[6]], [out[8], out[9], out[10]]];
  const matches = (a: number[], b: number[]) => a.every((value, axis) => Math.abs(value - b[axis]) < 1e-5);
  assert.ok(matches(rebuilt[0], first) || matches(rebuilt[0], second));
  assert.ok(matches(rebuilt[1], first) || matches(rebuilt[1], second));
});

test('every exported ice file loads into a rigid, wrapped initial state', () => {
  for (const molecules of [64, 216, 512]) {
    const initial = build(molecules);
    assert.equal(initial.molecules, molecules);
    assert.ok(Math.abs(initial.box - boxLength(molecules, 1)) < 1e-6);
    assert.equal(initial.state.length, molecules * 16);
    assert.equal(initial.sites.length, molecules * 12);
    assert.equal(initial.charges.length, molecules * 4);
    for (let molecule = 0; molecule < molecules; molecule++) {
      const base = molecule * 12;
      for (const axis of [0, 1, 2]) {
        assert.ok(initial.sites[base + axis] >= 0 && initial.sites[base + axis] < initial.box);
      }
      for (const hydrogen of [1, 2]) {
        const bond = [0, 1, 2].map(axis => initial.sites[base + hydrogen * 4 + axis] - initial.sites[base + axis]);
        assert.ok(Math.abs(Math.hypot(...bond) - OH_LENGTH) < 5e-3, `molecule ${molecule} bond`);
      }
      const quaternion = initial.state.subarray(molecule * 16 + 4, molecule * 16 + 8);
      assert.ok(Math.abs(Math.hypot(...quaternion) - 1) < 1e-5, 'unit quaternion');
    }
  }
});

test('velocities carry the requested temperature and no net momentum', () => {
  const temperature = 300;
  const initial = build(216, 1, temperature);
  const momentum = [0, 0, 0];
  let translational = 0, rotational = 0;
  const moments = principalMoments();
  for (let molecule = 0; molecule < initial.molecules; molecule++) {
    const velocity = initial.state.subarray(molecule * 16 + 8, molecule * 16 + 11);
    const angular = initial.state.subarray(molecule * 16 + 12, molecule * 16 + 15);
    for (const axis of [0, 1, 2]) {
      momentum[axis] += MOLECULE_MASS * velocity[axis];
      translational += MOLECULE_MASS * velocity[axis] ** 2;
      rotational += angular[axis] ** 2 / moments[axis];
    }
  }
  const degrees = 3 * initial.molecules;
  const toKelvin = (energy: number) => energy / FORCE_TO_ACCELERATION / (degrees * BOLTZMANN);
  assert.ok(Math.abs(toKelvin(translational) - temperature) < 1e-3, 'translational temperature');
  assert.ok(Math.abs(toKelvin(rotational) - temperature) < 1e-3, 'rotational temperature');
  for (const axis of [0, 1, 2]) assert.ok(Math.abs(momentum[axis]) < 1e-3, `momentum axis ${axis}`);
});

test('the same seed reproduces the state and a different seed does not', () => {
  assert.deepEqual(build(64, 1, 200, 11).state, build(64, 1, 200, 11).state);
  assert.notDeepEqual(build(64, 1, 200, 11).state, build(64, 1, 200, 12).state);
});

test('compressing the box scales oxygen positions but keeps molecules rigid', () => {
  const loose = build(64, 0.6), dense = build(64, 1.4);
  assert.ok(dense.box < loose.box);
  const nearest = (initial: { molecules: number; box: number; sites: Float32Array }) => {
    let best = Infinity;
    for (let a = 0; a < initial.molecules; a++) for (let b = a + 1; b < initial.molecules; b++) {
      const d = [0, 1, 2].map(axis => {
        const raw = initial.sites[b * 12 + axis] - initial.sites[a * 12 + axis];
        return raw - initial.box * Math.round(raw / initial.box);
      });
      best = Math.min(best, Math.hypot(...d));
    }
    return best;
  };
  const ratio = nearest(dense) / nearest(loose);
  assert.ok(Math.abs(ratio - dense.box / loose.box) < 0.02, `neighbour spacing scaled by ${ratio}`);
  const bond = [0, 1, 2].map(axis => dense.sites[4 + axis] - dense.sites[axis]);
  assert.ok(Math.abs(Math.hypot(...bond) - OH_LENGTH) < 5e-3);
});
