import { draw, frame, init, storage, surface } from 'vgpu';
import atomShader from './particles.wgsl?raw';
import bondShader from './bonds.wgsl?raw';
import { sampleFrame, type Trajectory } from './trajectory';
import { ATOM_STRIDE, BOND_STRIDE, createScene } from './scene';

export async function createBackground(canvas: HTMLCanvasElement, data: Trajectory, onFailure: () => void) {
  const gpu = await init({ powerPreference: 'low-power' });
  let failed = false;
  let observer: ResizeObserver | undefined;
  const fail = () => {
    if (failed || gpu.disposed) return;
    failed = true;
    observer?.disconnect();
    onFailure();
    gpu.dispose();
  };
  gpu.onError(error => { console.error('Molecular background:', error); fail(); });
  void gpu.gpu.lost.then(info => { if (info.reason !== 'destroyed') fail(); });
  try {
    const screen = surface(gpu, canvas, { dpr: [1, 1.5], alphaMode: 'premultiplied', clearColor: [0, 0, 0, 0] });
    const scene = createScene(data.particles, data.box);
    const sites = new Float32Array(data.particles * 12);
    const atoms = draw(gpu, { shader: atomShader, vertices: 6, instances: scene.atomCapacity, blend: 'alpha' });
    const bonds = draw(gpu, { shader: bondShader, geometry: { topology: 'line-list' }, blend: 'alpha' });
    // Worst-case capacity keeps the storage binding size stable while scrubbing.
    const atomBuffer = storage(gpu, scene.atoms.byteLength, 'read');
    const bondBuffer = storage(gpu, scene.bonds.byteLength, 'read');
    atoms.set({ points: atomBuffer });
    bonds.set({ points: bondBuffer });
    let progress = 0;
    function render(nextProgress: number) {
      if (gpu.disposed) return;
      progress = nextProgress;
      sampleFrame(data, progress, sites);
      const counts = scene.update(sites);
      frame(gpu, current => {
        const params = { aspect: screen.size[0] / screen.size[1], warmth: progress };
        atomBuffer.write(scene.atoms.subarray(0, counts.atomCount * ATOM_STRIDE));
        bondBuffer.write(scene.bonds.subarray(0, counts.bondVertices * BOND_STRIDE));
        atoms.set({ params });
        bonds.set({ params });
        current.pass({ target: screen, clear: [0, 0, 0, 0] }, pass => {
          pass.draw(bonds, { vertices: counts.bondVertices });
          pass.draw(atoms, { instances: counts.atomCount });
        });
      });
    }
    // Resize draws one frame even if playback is paused; no idle animation loop.
    observer = new ResizeObserver(() => { try { render(progress); } catch { fail(); } });
    observer.observe(canvas);
    render(0);
    await gpu.settled();
    if (failed) throw new Error('GPU initialization failed');
    return { render, dispose: () => { observer?.disconnect(); gpu.dispose(); } };
  } catch (error) {
    observer?.disconnect();
    gpu.dispose();
    throw error;
  }
}
