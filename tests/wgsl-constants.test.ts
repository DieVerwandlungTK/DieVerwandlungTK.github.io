import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { BOLTZMANN, COULOMB, EPSILON_O, FORCE_TO_ACCELERATION, MOLECULE_MASS, Q_H, Q_M, SIGMA_O,
  bodySites, principalMoments } from '../src/water-model.ts';

const shader = readFileSync('src/simulation.wgsl', 'utf8');
const constant = (name: string): number => {
  const match = shader.match(new RegExp(`const ${name}\\s*(?::\\s*f32)?\\s*=\\s*(-?[0-9.eE+-]+)`));
  assert.ok(match, `${name} missing from simulation.wgsl`);
  return Number(match![1]);
};
const vector = (name: string): number[] => {
  const match = shader.match(new RegExp(`const ${name}\\s*(?::\\s*vec3f)?\\s*=\\s*vec3f\\(([^)]*)\\)`));
  assert.ok(match, `${name} missing from simulation.wgsl`);
  return match![1].split(',').map(part => Number(part.trim()));
};

test('shader constants match the TypeScript water model', () => {
  const close = (actual: number, expected: number, label: string) =>
    assert.ok(Math.abs(actual - expected) < 1e-6, `${label}: ${actual} vs ${expected}`);
  close(constant('COULOMB'), COULOMB, 'COULOMB');
  close(constant('BOLTZMANN'), BOLTZMANN, 'BOLTZMANN');
  close(constant('FORCE_TO_ACCELERATION'), FORCE_TO_ACCELERATION, 'FORCE_TO_ACCELERATION');
  close(constant('SIGMA_O'), SIGMA_O, 'SIGMA_O');
  close(constant('EPSILON_O'), EPSILON_O, 'EPSILON_O');
  close(constant('Q_H'), Q_H, 'Q_H');
  close(constant('Q_M'), Q_M, 'Q_M');
  close(constant('MOLECULE_MASS'), MOLECULE_MASS, 'MOLECULE_MASS');
  const moments = principalMoments();
  vector('INERTIA').forEach((value, axis) => close(value, moments[axis], `INERTIA ${axis}`));
  const body = bodySites();
  vector('BODY_OXYGEN').forEach((value, axis) => close(value, body.oxygen[axis], `BODY_OXYGEN ${axis}`));
  vector('BODY_HYDROGEN_A').forEach((value, axis) => close(value, body.hydrogen[0][axis], `BODY_HYDROGEN_A ${axis}`));
  vector('BODY_HYDROGEN_B').forEach((value, axis) => close(value, body.hydrogen[1][axis], `BODY_HYDROGEN_B ${axis}`));
  vector('BODY_CHARGE').forEach((value, axis) => close(value, body.charge[axis], `BODY_CHARGE ${axis}`));
});
