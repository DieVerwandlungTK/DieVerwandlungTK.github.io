import { test } from 'node:test';
import assert from 'node:assert/strict';
import { maxwellBoltzmann, mostProbableSpeed, speedHistogram, smoothHistogram,
  SPEED_BINS, SPEED_AXIS_MAX } from '../src/speed-distribution';

function integral(low: number, high: number, temperature: number, steps = 10000) {
  const width = (high - low) / steps;
  let sum = 0;
  for (let i = 0; i < steps; i++) sum += maxwellBoltzmann(low + (i + .5) * width, temperature) * width;
  return sum;
}

test('Maxwell speed density integrates to unity at 150, 300 and 500 K', () => {
  for (const temperature of [150, 300, 500]) assert.ok(Math.abs(integral(0, 60, temperature) - 1) < 1e-3);
});

test('speed and height of the peak match independent closed-form values', () => {
  for (const [temperature, speed, peak] of [[150, 3.721007, .223118], [300, 5.262299, .157769], [500, 6.793603, .122208]]) {
    assert.ok(Math.abs(mostProbableSpeed(temperature) - speed) < 1e-4);
    assert.ok(Math.abs(maxwellBoltzmann(speed, temperature) - peak) < 1e-5);
    assert.ok(maxwellBoltzmann(speed, temperature) > maxwellBoltzmann(speed - .1, temperature));
    assert.ok(maxwellBoltzmann(speed, temperature) > maxwellBoltzmann(speed + .1, temperature));
  }
});

test('histogram uses translational velocity, preserves all probability and folds overflow into last bin', () => {
  const state = new Float32Array(4 * 16).fill(1000);
  [[0,0,0], [3,4,0], [18,0,0], [30,40,0]].forEach((v,i) => state.set(v, i*16+8));
  const bins = new Float64Array(12).fill(99);
  speedHistogram(state, 4, bins);
  assert.equal(bins[0], 1/6);
  assert.equal(bins[3], 1/6);
  assert.equal(bins[11], 1/3);
  assert.ok(Math.abs(bins.reduce((a,b)=>a+b,0) * 1.5 - 1) < 1e-12);
});

test('seeded Gaussian velocities agree with bin-integrated Maxwell density within five percent', () => {
  // Independent LCG + Box-Muller sampling; distribution code must not generate its own oracle.
  let seed = 22039;
  const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return (seed+.5)/4294967296; };
  const normal = () => Math.sqrt(-2*Math.log(random())) * Math.cos(2*Math.PI*random());
  const count = 800000, temperature = 500;
  const state = new Float32Array(count*16);
  const sigma = Math.sqrt(100*.0083144626*temperature/18.015324);
  for (let i=0;i<count;i++) for (let axis=0;axis<3;axis++) state[i*16+8+axis] = sigma*normal();
  const bins = new Float64Array(SPEED_BINS);
  speedHistogram(state,count,bins);
  const width = SPEED_AXIS_MAX/SPEED_BINS;
  for(let i=0;i<SPEED_BINS;i++) {
    const expected = integral(i*width,i===SPEED_BINS-1?60:(i+1)*width,temperature)/width;
    assert.ok(Math.abs(bins[i]-expected)/expected < .05, `bin ${i}: ${bins[i]} vs ${expected}`);
  }
});

test('EMA covers at least one time constant in three samples without overshooting', () => {
  const current = new Float64Array(12), target = new Float64Array(12);
  current[0] = 2/3; target[11] = 2/3;
  for (let i=0;i<3;i++) {
    smoothHistogram(current,target);
    assert.ok(current[0]>=0 && current[11]<=2/3);
    assert.ok(Math.abs(current.reduce((a,b)=>a+b,0)*1.5-1)<1e-12);
  }
  assert.ok(current[11]/target[11]>=1-1/Math.E);
});

test('invalid input cannot turn chart geometry into NaN', () => {
  assert.equal(maxwellBoltzmann(0,0),0);
  assert.equal(maxwellBoltzmann(-1,300),0);
  assert.equal(maxwellBoltzmann(Infinity,300),0);
  const state = new Float32Array(16); state[8] = NaN;
  assert.throws(()=>speedHistogram(state,1,new Float64Array(12)));
  assert.throws(()=>speedHistogram(new Float32Array(16),2,new Float64Array(12)));
});
