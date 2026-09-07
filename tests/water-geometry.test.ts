import { test } from 'node:test';
import assert from 'node:assert/strict';
import { findHydrogenBonds, molecularOpacity, periodicMolecules } from '../src/water-geometry.ts';
import { decodeTrajectory, sampleFrame } from '../src/trajectory.ts';

// O-H...O is linear for the first donor; a close acceptor in the wrong
// direction must not count just because its O-O distance is small.
const molecule = (x: number, y: number, h: number) => [x,y,0,1,x+h,y,0,1,x-.24,y+.92,0,1];
test('hydrogen bonds require donor hydrogen orientation as well as O-O proximity', () => {
  const sites = new Float32Array([...molecule(1,1,.9572), ...molecule(3.8,1,.9572)]);
  assert.ok(findHydrogenBonds(sites,10).some(b=>b.donor===0&&b.acceptor===1&&b.hydrogen===1));
  sites[4] = .0428;
  assert.ok(!findHydrogenBonds(sites,10).some(b=>b.donor===0&&b.acceptor===1&&b.hydrogen===1));
  sites[4] = 1.9572;
  assert.ok(findHydrogenBonds(sites,10).some(b=>b.donor===0&&b.acceptor===1&&b.hydrogen===1));
});

test('hydrogen-bond geometry uses periodic neighbors', () => {
  const sites = new Float32Array([...molecule(9,1,.9572), ...molecule(1.8,1,.9572)]);
  const bond = findHydrogenBonds(sites,10).find(b=>b.donor===0&&b.acceptor===1&&b.hydrogen===1);
  assert.ok(bond);
  assert.deepEqual(bond.shift, [10,0,0]);
});

test('replicated observation window is continuous when a molecule wraps', () => {
  const before = new Float32Array(molecule(9.999,5,.9572));
  const after = new Float32Array(molecule(.001,5,.9572));
  // Keep H coordinates whole on either side of the wrap.
  before[2]=before[6]=before[10]=5;
  after[2]=after[6]=after[10]=5;
  const a = periodicMolecules(before,10).sort((x,y)=>x.center[0]-y.center[0]);
  const b = periodicMolecules(after,10).sort((x,y)=>x.center[0]-y.center[0]);
  assert.equal(a.length,b.length);
  for(let i=0;i<a.length;i++) {
    assert.ok(Math.abs(a[i].center[0]-b[i].center[0]) < .003);
    assert.ok(Math.abs(a[i].opacity-b[i].opacity) < .003);
  }
  assert.equal(molecularOpacity([6.6,0,0],10),0);
});

test('binary decode rejects truncated data', () => {
  const header = { version:2,particles:1,box:10,atomOrder:['O','H','H'],coordinateFile:'water.bin',frames:[{temperature:180,timePs:0},{temperature:200,timePs:.02}] };
  assert.throws(()=>decodeTrajectory(header,new ArrayBuffer(4)));
});

test('water interpolation keeps O-H bonds rigid and molecules whole at the boundary', () => {
  const half = 104.52*Math.PI/360, c = .9572*Math.cos(half), s = .9572*Math.sin(half);
  const header = { version:2,particles:1,box:10,atomOrder:['O','H','H'],coordinateFile:'water.bin',frames:[{temperature:180,timePs:0},{temperature:200,timePs:.02}] };
  // Rotates 180 degrees around z as oxygen crosses the periodic boundary.
  const binary = new Float32Array([9.8,5,5,9.8+c,5+s,5,9.8+c,5-s,5,.2,5,5,.2-c,5-s,5,.2-c,5+s,5]);
  const data = decodeTrajectory(header,binary.buffer);
  const out = new Float32Array(12);
  sampleFrame(data,.5,out);
  assert.ok(out[0]<1e-5 || Math.abs(out[0]-10)<1e-5);
  for(const h of [4,8]) assert.ok(Math.abs(Math.hypot(out[h]-out[0],out[h+1]-out[1],out[h+2]-out[2])-.9572)<1e-5);
  const dot=(out[4]-out[0])*(out[8]-out[0])+(out[5]-out[1])*(out[9]-out[1])+(out[6]-out[2])*(out[10]-out[2]);
  assert.ok(Math.abs(dot/(.9572**2)-Math.cos(104.52*Math.PI/180))<1e-5);
});
