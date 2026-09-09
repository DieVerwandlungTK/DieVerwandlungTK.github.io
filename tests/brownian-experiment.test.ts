import { test } from 'node:test';
import assert from 'node:assert/strict';
import { BrownianExperiment } from '../src/brownian-experiment';

test('calibration precedes independent comparison and D stays frozen', () => {
  const run = new BrownianExperiment(1,0.6,42);
  run.step(2000);
  assert.equal(run.phase,'calibration');
  assert.equal(run.diffusion,null);
  run.step(8000);
  assert.equal(run.phase,'comparison');
  assert.ok(run.diffusion!>0);
  assert.equal(run.time,0);
  assert.ok(run.sde.every(p=>p.x===0&&p.y===0));
  const d=run.diffusion;
  run.step(2000);
  assert.equal(run.diffusion,d);
  assert.ok(run.records.at(-1)!.sde>0);
  assert.ok(run.records.at(-1)!.gas>0);
  assert.ok(run.gasDisplacements.every(p=>Number.isFinite(p.x)&&Number.isFinite(p.y)));
  run.step(5000);
  assert.equal(run.phase,'complete');
  assert.equal(run.time,60);
  assert.ok(run.records.length<=301);
  const end=JSON.stringify(run.records);
  run.step(100);
  assert.equal(JSON.stringify(run.records),end);
});

test('new conditions discard previous calibration and reproduce the same seed', () => {
  const run = new BrownianExperiment(2,0.8,42);
  run.step(100);
  const reset = new BrownianExperiment(2,0.8,42);
  const twin = new BrownianExperiment(2,0.8,42);
  twin.step(100);
  assert.deepEqual(run.gases[0],twin.gases[0]);
  assert.equal(reset.diffusion,null);
  assert.equal(reset.time,0);
  assert.equal(reset.phase,'warmup');
  assert.equal(reset.records.length,0);
});
