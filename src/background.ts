import { draw, frame, storage, surface, type Gpu } from 'vgpu';
import atomShader from './particles.wgsl?raw';
import bondShader from './bonds.wgsl?raw';
import { ATOM_STRIDE, BOND_STRIDE, createScene } from './scene';
import { boxLength } from './water-model';

/** Temperature is reported in the panel now, so the shaders' tint stays constant. */
const WARMTH = 0;

export async function createBackground(gpu: Gpu, canvas: HTMLCanvasElement,
    molecules: number, onFailure: () => void) {
  let failed = false;
  let observer: ResizeObserver | undefined;
  const fail = () => {
    if (failed || gpu.disposed) return;
    failed = true;
    observer?.disconnect();
    onFailure();
  };
  gpu.onError(error => { console.error('Molecular background:', error); fail(); });
  void gpu.gpu.lost.then(info => { if (info.reason !== 'destroyed') fail(); });
  try {
    const screen = surface(gpu, canvas, { dpr: [1, 1.5], alphaMode: 'premultiplied', clearColor: [0, 0, 0, 0] });
    // Task 8 moves the box into scene.update; until then the scene is built at ice density.
    const scene = createScene(molecules, boxLength(molecules, 1));
    const atoms = draw(gpu, { shader: atomShader, vertices: 6, instances: scene.atomCapacity, blend: 'alpha' });
    const bonds = draw(gpu, { shader: bondShader, geometry: { topology: 'line-list' }, blend: 'alpha' });
    // Worst-case capacity keeps the storage binding size stable as molecules move.
    const atomBuffer = storage(gpu, scene.atoms.byteLength, 'read');
    const bondBuffer = storage(gpu, scene.bonds.byteLength, 'read');
    atoms.set({ points: atomBuffer });
    bonds.set({ points: bondBuffer });
    let current: Float32Array | undefined;
    function paint() {
      if (gpu.disposed) return;
      const counts = current ? scene.update(current) : { atomCount: 0, bondVertices: 0 };
      frame(gpu, active => {
        const params = { aspect: screen.size[0] / screen.size[1], warmth: WARMTH };
        if (counts.atomCount > 0) atomBuffer.write(scene.atoms.subarray(0, counts.atomCount * ATOM_STRIDE));
        if (counts.bondVertices > 0) bondBuffer.write(scene.bonds.subarray(0, counts.bondVertices * BOND_STRIDE));
        atoms.set({ params });
        bonds.set({ params });
        active.pass({ target: screen, clear: [0, 0, 0, 0] }, pass => {
          // The first frame runs before any sites arrive, and empty draws are a validation warning.
          if (counts.bondVertices > 0) pass.draw(bonds, { vertices: counts.bondVertices });
          if (counts.atomCount > 0) pass.draw(atoms, { instances: counts.atomCount });
        });
      });
    }
    /** Task 8 threads `box` into scene.update; the scene ignores it for now. */
    function render(sites: Float32Array, _box: number) {
      current = sites;
      paint();
    }
    // Resize redraws the last configuration; the stepping loop owns the animation.
    observer = new ResizeObserver(() => { try { paint(); } catch { fail(); } });
    observer.observe(canvas);
    paint();
    await gpu.settled();
    if (failed) throw new Error('GPU initialization failed');
    return { render, dispose: () => { observer?.disconnect(); } };
  } catch (error) {
    observer?.disconnect();
    throw error;
  }
}
