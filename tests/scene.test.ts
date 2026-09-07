import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ATOM_STRIDE, BOND_STRIDE, createScene, viewScale } from '../src/scene.ts';
import { OH_LENGTH, HOH_ANGLE } from '../src/water-geometry.ts';

const half = HOH_ANGLE / 2, c = OH_LENGTH * Math.cos(half), s = OH_LENGTH * Math.sin(half);

/** One molecule as GPU-style vec4 sites: O, H1, H2. */
function molecule(x: number, y: number, z: number, direction = 1): number[] {
  return [x,y,z,1, x+direction*c,y+s,z,1, x+direction*c,y-s,z,1];
}
const atomsOf = (scene: ReturnType<typeof createScene>, count: number) =>
  Array.from({length: count}, (_, i) => scene.atoms.subarray(i*ATOM_STRIDE, (i+1)*ATOM_STRIDE));
const bondsOf = (scene: ReturnType<typeof createScene>, vertices: number) =>
  Array.from({length: vertices/2}, (_, i) => [scene.bonds.subarray(i*2*BOND_STRIDE,(i*2+1)*BOND_STRIDE), scene.bonds.subarray((i*2+1)*BOND_STRIDE,(i*2+2)*BOND_STRIDE)]);

test('a single central molecule renders three atoms and two rigid covalent bonds', () => {
  const box = 12.7;
  const scene = createScene(1, box);
  const counts = scene.update(new Float32Array(molecule(box/2,box/2,box/2)));
  // Periodic images sit a whole box away, outside the spherical window.
  assert.equal(counts.atomCount, 3);
  const atoms = atomsOf(scene, counts.atomCount);
  assert.equal(atoms.filter(a => a[5] === 0).length, 1);
  assert.equal(atoms.filter(a => a[5] === 1).length, 2);
  const oxygen = atoms.find(a => a[5] === 0)!;
  assert.ok(oxygen[3] > atoms.find(a => a[5] === 1)![3], 'oxygen is drawn larger than hydrogen');
  for (const atom of atoms) assert.ok(Math.abs(atom[4] - 1) < 1e-6, 'central molecule is fully opaque');
  assert.equal(counts.bondVertices, 4);
  for (const [from, to] of bondsOf(scene, counts.bondVertices)) {
    assert.equal(from[5], 0, 'covalent bonds are solid');
    const clip = Math.hypot(to[0]-from[0], to[1]-from[1], to[2]-from[2]);
    assert.ok(Math.abs(clip - OH_LENGTH*viewScale(box)) < 1e-6);
  }
});

test('hydrogen bonds appear as dashed segments from donor hydrogen to acceptor oxygen', () => {
  const box = 12.7, centre = box/2;
  const scene = createScene(2, box);
  // The second oxygen sits 2.8 A straight along the first donor O-H direction, with
  // its own hydrogens turned away so only one donor satisfies the criteria.
  const reach = 2.8 / OH_LENGTH;
  const counts = scene.update(new Float32Array([...molecule(centre,centre,centre),
    ...molecule(centre+c*reach, centre+s*reach, centre)]));
  const dashed = bondsOf(scene, counts.bondVertices).filter(([from]) => from[5] === 1);
  assert.equal(dashed.length, 1);
  const [from, to] = dashed[0];
  assert.ok(to[3] > from[3], 'dash phase advances along the segment');
  const scale = viewScale(box);
  const expected = (2.8 - OH_LENGTH) * scale;
  assert.ok(Math.abs(Math.hypot(to[0]-from[0],to[1]-from[1],to[2]-from[2]) - expected) < 1e-6);
  assert.ok(from[4] > 0 && from[4] <= 1);
});

test('crossing a periodic face only relabels images, keeping the scene continuous', () => {
  const box = 12.7;
  const scene = createScene(1, box);
  const before = scene.update(new Float32Array(molecule(box-0.001, box/2, box/2)));
  const opacity = (count: number) => atomsOf(scene, count).reduce((sum, atom) => sum + atom[4], 0);
  const beforeOpacity = opacity(before.atomCount);
  const after = scene.update(new Float32Array(molecule(0.001, box/2, box/2)));
  assert.equal(after.atomCount, before.atomCount);
  assert.ok(Math.abs(opacity(after.atomCount) - beforeOpacity) < 0.01);
});

test('a dense periodic cell stays within the preallocated GPU capacity', () => {
  const box = 12.7, side = 4, count = side**3;
  const scene = createScene(count, box);
  const sites: number[] = [];
  for (let x = 0; x < side; x++) for (let y = 0; y < side; y++) for (let z = 0; z < side; z++) {
    sites.push(...molecule((x+.5)*box/side, (y+.5)*box/side, (z+.5)*box/side));
  }
  const counts = scene.update(new Float32Array(sites));
  assert.ok(counts.atomCount > 3*count, 'periodic images enlarge the visible window');
  assert.ok(counts.atomCount * ATOM_STRIDE <= scene.atoms.length);
  assert.ok(counts.bondVertices * BOND_STRIDE <= scene.bonds.length);
});
