import { test } from 'node:test';
import assert from 'node:assert/strict';
import { FORCE_LIMIT, needsRestart } from '../src/simulation-health.ts';

const stats = (overrides: Partial<Parameters<typeof needsRestart>[0]> = {}) => ({
  translationalTemperature: 300, rotationalTemperature: 300, maximumForce: 900, nonFinite: false, ...overrides,
});

test('a healthy sample keeps running', () => {
  assert.equal(needsRestart(stats()), false);
  assert.equal(needsRestart(stats({ maximumForce: FORCE_LIMIT - 1 })), false);
});

test('non-finite state or a runaway force forces a restart', () => {
  assert.equal(needsRestart(stats({ nonFinite: true })), true);
  assert.equal(needsRestart(stats({ maximumForce: FORCE_LIMIT + 1 })), true);
  assert.equal(needsRestart(stats({ maximumForce: Number.NaN })), true);
});
