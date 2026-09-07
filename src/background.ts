import { draw, frame, storage, surface, type Gpu, type StorageBuffer } from 'vgpu';
import atomShader from './particles.wgsl?raw';
import bondShader from './bonds.wgsl?raw';
import { ATOM_STRIDE, BOND_STRIDE, createScene, type SceneCounts } from './scene';

/** Temperature is shown in the panel rather than by tinting, so the shaders' warmth stays off. */
const WARMTH = 0;
/** vgpu frees storage with gpu.dispose(); an early destroy keeps a resize from leaking. */
const release = (buffer: StorageBuffer) => (buffer as { destroy?: () => void }).destroy?.();

export async function createBackground(gpu: Gpu, canvas: HTMLCanvasElement, molecules: number,
    onFailure: () => void) {
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
    let scene = createScene(molecules);
    // Worst-case capacity keeps the storage binding size stable while the sample runs.
    let atomBuffer = storage(gpu, scene.atoms.byteLength, 'read');
    let bondBuffer = storage(gpu, scene.bonds.byteLength, 'read');
    const atoms = draw(gpu, { shader: atomShader, vertices: 6, instances: scene.atomCapacity, blend: 'alpha' });
    const bonds = draw(gpu, { shader: bondShader, geometry: { topology: 'line-list' }, blend: 'alpha' });
    const bind = () => { atoms.set({ points: atomBuffer }); bonds.set({ points: bondBuffer }); };
    bind();
    let latest: { sites: Float32Array; box: number } | undefined;

    function render(sites: Float32Array, box: number): SceneCounts {
      latest = { sites, box };
      const counts = scene.update(sites, box);
      frame(gpu, current => {
        const params = { aspect: screen.size[0] / screen.size[1], warmth: WARMTH };
        if (counts.atomCount > 0) atomBuffer.write(scene.atoms.subarray(0, counts.atomCount * ATOM_STRIDE));
        if (counts.bondVertices > 0) bondBuffer.write(scene.bonds.subarray(0, counts.bondVertices * BOND_STRIDE));
        atoms.set({ params });
        bonds.set({ params });
        current.pass({ target: screen, clear: [0, 0, 0, 0] }, pass => {
          // Empty draws are a WebGPU validation warning, so an empty window just clears.
          if (counts.bondVertices > 0) pass.draw(bonds, { vertices: counts.bondVertices });
          if (counts.atomCount > 0) pass.draw(atoms, { instances: counts.atomCount });
        });
      });
      return counts;
    }

    /** A new molecule count needs a new scene and matching buffer capacity. */
    function resize(count: number) {
      const stale = [atomBuffer, bondBuffer];
      scene = createScene(count);
      atomBuffer = storage(gpu, scene.atoms.byteLength, 'read');
      bondBuffer = storage(gpu, scene.bonds.byteLength, 'read');
      bind();
      for (const buffer of stale) release(buffer);
      latest = undefined;
    }

    // Resizing the canvas redraws the last state even while the simulation is paused.
    observer = new ResizeObserver(() => {
      if (!latest) return;
      try { render(latest.sites, latest.box); } catch { fail(); }
    });
    observer.observe(canvas);
    await gpu.settled();
    if (failed) throw new Error('GPU initialization failed');
    return { render, resize, dispose: () => observer?.disconnect() };
  } catch (error) {
    observer?.disconnect();
    throw error;
  }
}
