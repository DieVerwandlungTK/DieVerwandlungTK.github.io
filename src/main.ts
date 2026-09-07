import './style.css';
import { init, type Gpu } from 'vgpu';
import { createSimulation, needsRestart } from './simulation';
import { MOLECULE_COUNTS, boxLength } from './water-model';

const get = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const play = get<HTMLButtonElement>('play-pause');
const reset = get<HTMLButtonElement>('reset');
const temperatureInput = get<HTMLInputElement>('temperature-input');
const densityInput = get<HTMLInputElement>('density-input');
const moleculeSelect = get<HTMLSelectElement>('molecules');
const rate = get<HTMLSelectElement>('rate');
const status = get<HTMLElement>('playback-status');
const stage = document.querySelector('.molecular-stage')!;
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const controls = [play, reset, temperatureInput, densityInput, moleculeSelect, rate];

let renderer: Awaited<ReturnType<typeof import('./background').createBackground>> | undefined;
let simulation: ReturnType<typeof createSimulation> | undefined;
/** The simulation and the background share one device, owned here rather than by either. */
let gpu: Gpu | undefined;
let playing = false;
let request = 0;
let drawing = false;
/** A draw was requested while a readback was already in flight, so it was skipped; catch up. */
let drawPending = false;
let lastStatsAt = 0;
let stepsSince = 0;
let throughputAt = 0;

get('year').textContent = String(new Date().getFullYear());

function updateControls() {
  play.textContent = playing ? '停止' : '再開';
  play.setAttribute('aria-label', playing ? '計算を一時停止' : '計算を再開');
  status.textContent = playing ? '計算中' : '一時停止中';
}
function stop() {
  playing = false;
  cancelAnimationFrame(request);
  updateControls();
}
function fallback() {
  stop();
  renderer?.dispose();
  renderer = undefined;
  simulation?.dispose();
  simulation = undefined;
  gpu?.dispose();
  gpu = undefined;
  stage.classList.remove('ready');
  for (const control of controls) control.disabled = true;
  status.textContent = '静止画を表示中';
}
function showTime() {
  if (!simulation) return;
  get('sim-time').textContent = `${simulation.timePs.toFixed(1)} ps`;
  get('box-length').textContent = `${simulation.box.toFixed(1)} Å`;
}

/** One animation frame: step, draw the latest sites, and refresh the readouts twice a second. */
function tick(time: number) {
  if (!simulation || !renderer) return;
  const steps = Number(rate.value);
  if (playing && !document.hidden) {
    simulation.step(steps);
    stepsSince += steps;
  }
  if (!drawing) {
    drawing = true;
    const molecules = simulation.molecules;
    void simulation.readSites().then(sites => {
      drawing = false;
      // A size change mid-read resolves sites the resized scene has no room for; the next
      // frame reads the new sample, so this one is simply dropped.
      if (!renderer || simulation?.molecules !== molecules) return;
      const counts = renderer.render(sites, simulation.box);
      get('bond-count').textContent = String(counts.hydrogenBonds);
      // A draw wanted while this readback was in flight (below) was skipped; catch up now
      // rather than leaving the canvas stale. While playing the loop already reschedules every
      // frame regardless, so only re-arm here when paused.
      if (drawPending) { drawPending = false; if (!playing) requestAnimationFrame(tick); }
    }).catch(error => { console.warn('Background unavailable', error); fallback(); });
  } else {
    drawPending = true;
  }
  showTime();
  if (time - lastStatsAt > 500) {
    lastStatsAt = time;
    const elapsed = (time - throughputAt) / 1000;
    if (elapsed > 0) {
      get('throughput').textContent = `${(stepsSince * 0.002 / elapsed).toFixed(1)} ps/s`;
      stepsSince = 0;
      throughputAt = time;
    }
    void simulation.readStats().then(stats => {
      get('kinetic-temperature').textContent = `${Math.round(stats.translationalTemperature)} K`;
      // A diverged sample is restarted rather than left to fill the screen with artefacts. The
      // notice is transient: restore the normal status text shortly after, via the same
      // function the play/pause button uses, so it doesn't keep claiming a restart forever.
      if (needsRestart(stats)) {
        simulation?.reset();
        status.textContent = '氷から再開しました';
        setTimeout(updateControls, 2000);
      }
    });
  }
  if (playing) request = requestAnimationFrame(tick);
}
function start() {
  if (!simulation) return;
  playing = true;
  updateControls();
  throughputAt = performance.now();
  stepsSince = 0;
  cancelAnimationFrame(request);
  request = requestAnimationFrame(tick);
}

play.addEventListener('click', () => playing ? stop() : start());
reset.addEventListener('click', () => {
  simulation?.reset();
  showTime();
  if (!playing) requestAnimationFrame(tick);
});
temperatureInput.addEventListener('input', () => {
  get('temperature').textContent = temperatureInput.value;
  const kelvin = Number(temperatureInput.value);
  simulation?.setTemperature(kelvin);
  get('phase-label').textContent = kelvin <= 240 ? 'ICE Ic' : kelvin <= 380 ? 'WARMING' : 'HOT';
});
densityInput.addEventListener('input', () => {
  const ratio = Number(densityInput.value) / 100;
  get('density-ratio').textContent = ratio.toFixed(2);
  simulation?.setDensity(ratio);
  showTime();
  if (!playing) requestAnimationFrame(tick);
});
moleculeSelect.addEventListener('change', () => {
  const molecules = Number(moleculeSelect.value);
  simulation?.setMolecules(molecules);
  simulation?.setDensity(Number(densityInput.value) / 100);
  renderer?.resize(molecules);
  showTime();
  if (!playing) requestAnimationFrame(tick);
});
document.addEventListener('visibilitychange', () => {
  if (!document.hidden && playing) {
    cancelAnimationFrame(request);
    // Steps taken while hidden are not counted (tick skips stepping when document.hidden), but
    // the throughput window spans the hidden time regardless; reset it so the next sample covers
    // only time actually spent computing, instead of reading 0.0 ps/s for one window.
    stepsSince = 0;
    throughputAt = performance.now();
    request = requestAnimationFrame(tick);
  }
});
reducedMotion.addEventListener('change', () => { if (reducedMotion.matches) stop(); });
window.addEventListener('pagehide', () => cancelAnimationFrame(request));
window.addEventListener('pageshow', () => { if (playing) { cancelAnimationFrame(request); request = requestAnimationFrame(tick); } });

async function initialize() {
  if (!navigator.gpu) return;
  try {
    const signal = AbortSignal.timeout(20000);
    const base = import.meta.env.BASE_URL;
    const ice: Record<number, Float32Array> = {};
    await Promise.all(MOLECULE_COUNTS.map(async count => {
      const response = await fetch(`${base}data/ice-${count}.bin`, { signal });
      if (!response.ok) throw new Error(`Missing ice configuration for ${count} molecules`);
      ice[count] = new Float32Array(await response.arrayBuffer());
    }));
    // Narrow screens are usually phones; start them on the smallest sample.
    const molecules = innerWidth <= 720 ? 64 : 216;
    moleculeSelect.value = String(molecules);
    gpu = await init({ powerPreference: 'low-power' });
    simulation = createSimulation(gpu, { molecules, densityRatio: 1, temperature: 180, ice });
    const { createBackground } = await import('./background');
    renderer = await createBackground(gpu, get<HTMLCanvasElement>('molecule-canvas'), molecules, fallback);
    (window as unknown as { waterSimulation: unknown }).waterSimulation = simulation;
    stage.classList.add('ready');
    showTime();
    get('box-length').textContent = `${boxLength(molecules, 1).toFixed(1)} Å`;
    renderer.render(await simulation.readSites(), simulation.box);
    updateControls();
    if (!reducedMotion.matches) start(); else requestAnimationFrame(tick);
    // Enable the controls only now, after the loop is already running (or, under reduced
    // motion, after the single frame above is queued). Enabling them any earlier — even just
    // before this point, across the `await simulation.readSites()` above — leaves a real idle
    // window in which the pause button already reads 計算を一時停止 while `playing` is still
    // false, so the first click on it starts the loop instead of stopping it, and this function
    // then starts it a second time. Order matters here even though it looks reorderable.
    for (const control of controls) control.disabled = false;
  } catch (error) {
    console.warn('Using the static molecular background:', error);
    fallback();
  }
}
void initialize();
