import './style.css';
import { init } from 'vgpu';
import { createSimulation } from './simulation';
import { MOLECULE_COUNTS } from './water-model';

const get = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const play = get<HTMLButtonElement>('play-pause');
const replay = get<HTMLButtonElement>('replay');
const timeline = get<HTMLInputElement>('timeline');
const rate = get<HTMLSelectElement>('rate');
const status = get<HTMLElement>('playback-status');
const stage = document.querySelector('.molecular-stage')!;
let renderer: Awaited<ReturnType<typeof import('./background').createBackground>> | undefined;
let gpu: Awaited<ReturnType<typeof init>> | undefined;

get('year').textContent = String(new Date().getFullYear());
// The precomputed playback controls have nothing to drive; Task 8 replaces the whole panel.
play.disabled = replay.disabled = timeline.disabled = rate.disabled = true;

function fallback() {
  renderer?.dispose();
  renderer = undefined;
  gpu?.dispose();
  gpu = undefined;
  stage.classList.remove('ready');
  play.disabled = replay.disabled = timeline.disabled = rate.disabled = true;
  get('temperature').textContent = '180';
  get('phase-label').textContent = 'ICE Ic';
  get('sim-time').textContent = '0.0 ps';
  timeline.value = '0';
  status.textContent = '静止画を表示中';
}

async function initialize() {
  if (!navigator.gpu) return;
  try {
    const signal = AbortSignal.timeout(20000);
    const base = import.meta.env.BASE_URL;
    const ice: Record<number, Float32Array> = {};
    for (const count of MOLECULE_COUNTS) {
      const response = await fetch(`${base}data/ice-${count}.bin`, { signal });
      if (!response.ok) throw new Error(`Missing ice configuration for ${count} molecules`);
      ice[count] = new Float32Array(await response.arrayBuffer());
    }
    // One gpu is shared by the simulation and the background, and owned here.
    gpu = await init({ powerPreference: 'low-power' });
    const simulation = createSimulation(gpu, { molecules: 216, densityRatio: 1, temperature: 180, ice });
    const { createBackground } = await import('./background');
    renderer = await createBackground(gpu, get<HTMLCanvasElement>('molecules'), simulation.molecules, fallback);
    (window as unknown as { waterSimulation: unknown }).waterSimulation = simulation;
    stage.classList.add('ready');
    // Nothing integrates yet (Task 6 adds stepping), so draw the ice lattice once.
    renderer.render(await simulation.readSites(), simulation.box);
  } catch (error) {
    console.warn('Using static molecular background:', error);
    fallback();
  }
}
void initialize();
