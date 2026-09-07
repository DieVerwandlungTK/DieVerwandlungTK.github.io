import { test } from 'node:test';
import assert from 'node:assert/strict';
import { decodeTrajectory, validateTrajectory, sampleFrame } from '../src/trajectory.ts';
import { HOH_ANGLE, OH_LENGTH } from '../src/water-geometry.ts';

const half = HOH_ANGLE / 2, c = OH_LENGTH * Math.cos(half), s = OH_LENGTH * Math.sin(half);
const header = () => ({ version: 2, box: 10, particles: 1, atomOrder: ['O', 'H', 'H'], coordinateFile: 'water.bin',
  frames: [{ temperature: 180, timePs: 0 }, { temperature: 300, timePs: 1 }] });
const oxygens = [[9.8, 1, 2], [.2, 3, 4]];
const coordinates = () => new Float32Array(oxygens.flatMap(([x, y, z]) => [x, y, z, x + c, y + s, z, x + c, y - s, z]));
const fixture = () => decodeTrajectory(header(), coordinates().buffer);

test('periodic interpolation crosses the box edge instead of the entire box', () => {
  const out = new Float32Array(12);
  sampleFrame(fixture(), .5, out);
  assert.ok(Math.min(Math.abs(out[0]), Math.abs(out[0] - 10)) < 1e-5);
  assert.equal(out[1], 2);
  assert.equal(out[2], 3);
});

test('playback clamps to actual first and final frames without invented reverse motion', () => {
  const out = new Float32Array(12);
  const data = fixture();
  sampleFrame(data, -1, out);
  assert.ok(Math.abs(out[0] - 9.8) < 1e-5);
  sampleFrame(data, 2, out);
  assert.ok(Math.abs(out[0] - .2) < 1e-5);
});

test('rejects truncated, nonfinite, and empty trajectory data before GPU upload', () => {
  const frames = () => header().frames.map((frame, index) => ({ ...frame, positions: [...coordinates().subarray(index * 9, index * 9 + 9)] }));
  const data = () => ({ ...header(), frames: frames() });
  assert.ok(validateTrajectory(data()));
  const short = data(); short.frames[0].positions.pop();
  assert.throws(() => validateTrajectory(short));
  const nan = data(); nan.frames[0].positions[0] = NaN;
  assert.throws(() => validateTrajectory(nan));
  const outside = data(); outside.frames[0].positions[0] = 11;
  assert.throws(() => validateTrajectory(outside));
  assert.throws(() => validateTrajectory({ ...data(), frames: [] }));
  assert.throws(() => validateTrajectory({ ...data(), box: 0 }));
  assert.throws(() => validateTrajectory({ ...data(), version: 1 }));
});

test('rejects coordinate files whose atom order is not oxygen and two hydrogens', () => {
  assert.throws(() => decodeTrajectory({ ...header(), atomOrder: ['O', 'H'] }, coordinates().buffer));
});
