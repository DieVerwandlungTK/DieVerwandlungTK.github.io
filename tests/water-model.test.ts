import { test } from 'node:test';
import assert from 'node:assert/strict';
import { BOLTZMANN, MASS_H, MASS_O, MOLECULE_COUNTS, MOLECULE_MASS, OH_LENGTH, HOH_ANGLE,
  bodySites, boxLength, cellsFor, cutoffFor, principalMoments } from '../src/water-model.ts';

test('body-frame sites keep TIP4P-Ew geometry with the centre of mass at the origin', () => {
  const { oxygen, hydrogen, charge } = bodySites();
  for (const h of hydrogen) {
    const bond = h.map((value, axis) => value - oxygen[axis]);
    assert.ok(Math.abs(Math.hypot(...bond) - OH_LENGTH) < 1e-9);
  }
  const first = hydrogen[0].map((value, axis) => value - oxygen[axis]);
  const second = hydrogen[1].map((value, axis) => value - oxygen[axis]);
  const cosine = first.reduce((sum, value, axis) => sum + value * second[axis], 0) / OH_LENGTH ** 2;
  assert.ok(Math.abs(cosine - Math.cos(HOH_ANGLE)) < 1e-9);
  for (let axis = 0; axis < 3; axis++) {
    const moment = MASS_O * oxygen[axis] + MASS_H * (hydrogen[0][axis] + hydrogen[1][axis]);
    assert.ok(Math.abs(moment) < 1e-9, `centre of mass on axis ${axis}`);
  }
  // The M site sits 0.125 A from oxygen along the bisector, which is the body x axis.
  assert.ok(Math.abs(charge[0] - (oxygen[0] + 0.125)) < 1e-9);
  assert.ok(Math.abs(charge[1]) < 1e-12 && Math.abs(charge[2]) < 1e-12);
  assert.ok(Math.abs(MOLECULE_MASS - (MASS_O + 2 * MASS_H)) < 1e-12);
});

test('principal moments match the measured moments of inertia of water', () => {
  // Experimental values 1.918, 1.021, 2.939 e-40 g cm2 equal these amu A2 numbers.
  const [x, y, z] = principalMoments();
  assert.ok(Math.abs(x - 1.15509) < 1e-3, `x ${x}`);
  assert.ok(Math.abs(y - 0.61473) < 1e-3, `y ${y}`);
  assert.ok(Math.abs(z - 1.76982) < 1e-3, `z ${z}`);
  // A planar rigid body satisfies Iz = Ix + Iy.
  assert.ok(Math.abs(z - (x + y)) < 1e-9);
});

test('box length follows the molecule count and density, and the cutoff never reaches half the box', () => {
  assert.deepEqual([...MOLECULE_COUNTS], [64, 216, 512]);
  assert.deepEqual(MOLECULE_COUNTS.map(cellsFor), [2, 3, 4]);
  assert.ok(Math.abs(boxLength(64, 1) - 12.7) < 1e-9);
  assert.ok(Math.abs(boxLength(216, 1) - 19.05) < 1e-9);
  assert.ok(Math.abs(boxLength(512, 1) - 25.4) < 1e-9);
  // Compressing to 1.4 times ice density shrinks the box by the cube root.
  assert.ok(Math.abs(boxLength(216, 1.4) - 19.05 / 1.4 ** (1 / 3)) < 1e-9);
  for (const molecules of MOLECULE_COUNTS) {
    for (const ratio of [0.6, 1, 1.4]) {
      const box = boxLength(molecules, ratio);
      const cutoff = cutoffFor(box);
      assert.ok(cutoff < box / 2, `cutoff ${cutoff} must stay below half of ${box}`);
      assert.ok(cutoff <= 9);
    }
  }
  assert.ok(Math.abs(cutoffFor(25.4) - 9) < 1e-9);
  assert.ok(Math.abs(cutoffFor(12.7) - 0.49 * 12.7) < 1e-9);
});

test('Boltzmann constant is in kJ per mole per kelvin', () => {
  assert.ok(Math.abs(BOLTZMANN - 0.0083144626) < 1e-12);
});
