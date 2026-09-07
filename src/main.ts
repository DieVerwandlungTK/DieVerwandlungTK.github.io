import './style.css';
import { decodeTrajectory, type Trajectory } from './trajectory';

const get = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const play = get<HTMLButtonElement>('play-pause');
const replay = get<HTMLButtonElement>('replay');
const timeline = get<HTMLInputElement>('timeline');
const rate = get<HTMLSelectElement>('rate');
const status = get<HTMLElement>('playback-status');
const stage = document.querySelector('.molecular-stage')!;
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
let renderer: Awaited<ReturnType<typeof import('./background').createBackground>> | undefined;
let data: Trajectory;
let progress = 0;
let playing = false;
let request = 0;
let previousTime = 0;
let lastDraw = 0;
/** Wall-clock milliseconds for the whole trajectory at 1x. */
const BASE_DURATION = 30_000;
const duration = () => BASE_DURATION / Number(rate.value);

get('year').textContent = String(new Date().getFullYear());

function updateControls() {
  play.textContent = playing ? '停止' : '再生';
  play.setAttribute('aria-label', playing ? '軌跡を一時停止' : '軌跡を再生');
  status.textContent = playing ? '事前計算軌跡を再生中' : progress >= 1 ? '再生終了' : '一時停止中';
}
function stop() {
  playing = false;
  cancelAnimationFrame(request);
  previousTime = 0;
  updateControls();
}
function fallback() {
  stop();
  renderer?.dispose();
  renderer = undefined;
  stage.classList.remove('ready');
  play.disabled = replay.disabled = timeline.disabled = rate.disabled = true;
  get('temperature').textContent = '180';
  get('phase-label').textContent = 'ICE Ic';
  get('sim-time').textContent = '0.0 ps';
  timeline.value = '0';
  status.textContent = '静止画を表示中';
}
function drawCurrent() {
  if (!renderer) return;
  try { renderer.render(progress); } catch (error) { console.warn('Background unavailable', error); fallback(); return; }
  const frameIndex = progress * (data.frames.length - 1);
  const first = data.frames[Math.floor(frameIndex)];
  const second = data.frames[Math.min(Math.floor(frameIndex) + 1, data.frames.length - 1)];
  const blend = (a: number, b: number) => a + (b - a) * (frameIndex % 1);
  const timePs = blend(first.timePs, second.timePs);
  get('temperature').textContent = String(Math.round(blend(first.temperature, second.temperature)));
  get('sim-time').textContent = `${timePs.toFixed(1)} ps`;
  get('phase-label').textContent = timePs < 5 ? 'ICE Ic' : timePs < 25 ? 'HEATING' : '450 K';
  timeline.value = String(Math.round(progress * 1000));
  timeline.setAttribute('aria-valuetext', `${timePs.toFixed(1)} ピコ秒`);
}
function tick(time: number) {
  if (!playing || document.hidden || !renderer) return;
  if (!previousTime) previousTime = time;
  progress = Math.min(1, progress + Math.min(time - previousTime, 100) / duration());
  previousTime = time;
  if (time - lastDraw >= 1000 / 30 || progress >= 1) { drawCurrent(); lastDraw = time; }
  if (!renderer) return;
  if (progress >= 1) { stop(); return; }
  request = requestAnimationFrame(tick);
}
function start() {
  if (!renderer) return;
  if (progress >= 1) progress = 0;
  playing = true;
  previousTime = 0;
  updateControls();
  cancelAnimationFrame(request);
  request = requestAnimationFrame(tick);
}
play.addEventListener('click', () => playing ? stop() : start());
replay.addEventListener('click', () => { stop(); progress = 0; drawCurrent(); start(); });
timeline.addEventListener('input', () => { stop(); progress = Number(timeline.value) / 1000; drawCurrent(); updateControls(); });
// Changing speed keeps the current position; playback rate is wall-clock only.
rate.addEventListener('change', () => { previousTime = 0; });
document.addEventListener('visibilitychange', () => {
  cancelAnimationFrame(request);
  previousTime = 0;
  if (!document.hidden && playing) request = requestAnimationFrame(tick);
});
reducedMotion.addEventListener('change', () => { if (reducedMotion.matches) stop(); });
window.addEventListener('pagehide', () => { cancelAnimationFrame(request); });
window.addEventListener('pageshow', () => { if (playing && renderer) { previousTime = 0; request = requestAnimationFrame(tick); } });

async function initialize() {
  if (!navigator.gpu) return;
  try {
    const signal = AbortSignal.timeout(20000);
    const base = import.meta.env.BASE_URL;
    const [header, coordinates] = await Promise.all([
      fetch(`${base}data/water.json`, { signal }).then(response => {
        if (!response.ok) throw new Error('Trajectory metadata request failed');
        return response.json();
      }),
      fetch(`${base}data/water.bin`, { signal }).then(response => {
        if (!response.ok) throw new Error('Trajectory coordinate request failed');
        return response.arrayBuffer();
      }),
    ]);
    data = decodeTrajectory(header, coordinates);
    const { createBackground } = await import('./background');
    renderer = await createBackground(get<HTMLCanvasElement>('molecules'), data, fallback);
    stage.classList.add('ready');
    play.disabled = replay.disabled = timeline.disabled = rate.disabled = false;
    drawCurrent();
    updateControls();
    if (!reducedMotion.matches) start();
  } catch (error) {
    console.warn('Using static molecular background:', error);
    fallback();
  }
}
void initialize();
