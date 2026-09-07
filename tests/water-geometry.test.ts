import { test } from 'node:test';
import assert from 'node:assert/strict';
import { findHydrogenBonds, molecularOpacity, periodicMolecules } from '../src/water-geometry.ts';

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
