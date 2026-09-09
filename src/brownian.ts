import './style.css';
import './playground.css';
import './brownian.css';
import { BrownianExperiment, DT, REPLICAS } from './brownian-experiment';
import { drawStatistics, drawTrajectory } from './brownian-charts';
const get=<T extends HTMLElement>(id:string)=>document.getElementById(id) as T;
const temperature=get<HTMLInputElement>('brownian-temperature');
const density=get<HTMLInputElement>('brownian-density');
const rate=get<HTMLSelectElement>('brownian-rate');
const play=get<HTMLButtonElement>('toggle-run');
const download=get<HTMLButtonElement>('download-data');
const reduced=matchMedia('(prefers-reduced-motion: reduce)');
const msd=document.getElementById('msd-chart') as unknown as SVGSVGElement;
const histogram=document.getElementById('histogram-chart') as unknown as SVGSVGElement;
let run=new BrownianExperiment(Number(temperature.value),Number(density.value),42);
let playing=!reduced.matches;
let request=0,last=0,budget=0,lastChart=0;
const names={warmup:'準備運転',calibration:'係数推定',comparison:'比較',complete:'比較完了',failed:'推定できませんでした'};
function render(charts=true) {
  get('temperature-value').textContent=Number(temperature.value).toFixed(1);
  get('density-value').textContent=Number(density.value).toFixed(2);
  get('phase').textContent=names[run.phase];
  get('run-time').textContent=run.time.toFixed(1);
  const duration=run.phase==='warmup'?20:run.phase==='calibration'||run.phase==='failed'?80:60;
  get('phase-duration').textContent=String(duration);
  const progress=get<HTMLProgressElement>('phase-progress');progress.max=duration;progress.value=run.time;
  get('run-status').textContent=run.phase==='failed'?'正の傾きを推定できません。条件を変更してください。':run.phase==='complete'?'比較完了':playing?'計算中':'一時停止中';
  play.textContent=playing?'一時停止':'再生';
  play.disabled=run.phase==='failed'||run.phase==='complete';
  get('diffusion-value').textContent=run.diffusion?.toFixed(4)??'未推定';
  get('friction-value').textContent=run.diffusion?(run.temperature/run.diffusion).toFixed(4):'—';
  get('noise-value').textContent=run.diffusion?Math.sqrt(2*run.diffusion).toFixed(4):'—';
  get('msd-caption').textContent=run.diffusion?'比較区間：D は固定。破線は調整した係数から得る 4Dt で、独立した予測ではありません。':run.phase==='warmup'?'準備運転の後、推定区間の MSD を表示します。':'推定区間：後半 t = 40–80 の傾きから D を推定します。';
  download.disabled=run.records.length<2;
  drawTrajectory(get<HTMLCanvasElement>('gas-canvas'),run,'gas');
  drawTrajectory(get<HTMLCanvasElement>('sde-canvas'),run,'sde');
  if(charts) drawStatistics(run,msd,histogram);
}
function tick(now:number) {
  request=0;
  if(!playing||document.hidden) { last=0;return; }
  if(last) budget+=Math.min((now-last)/1000,0.05)*8*Number(rate.value)/DT;
  last=now;
  // Limit CPU work per frame. Unconsumed steps stay bounded so slow devices run slower.
  const start=performance.now();
  budget=Math.min(budget,160);
  while(budget>=1&&performance.now()-start<10) {
    const steps=Math.min(10,Math.floor(budget));run.step(steps);budget-=steps;
    if(run.phase==='complete'||run.phase==='failed') { playing=false;budget=0;break; }
  }
  const charts=now-lastChart>150||!playing;
  render(charts);if(charts) lastChart=now;
  if(playing) request=requestAnimationFrame(tick);
}
function schedule() {
  cancelAnimationFrame(request);last=0;budget=0;
  if(playing&&!document.hidden) request=requestAnimationFrame(tick);
}
play.addEventListener('click',()=>{playing=!playing;schedule();render();});
function reset() {
  run=new BrownianExperiment(Number(temperature.value),Number(density.value),42);
  schedule();render();
}
get('reset-run').addEventListener('click',reset);
temperature.addEventListener('input',reset);
density.addEventListener('input',reset);
rate.addEventListener('change',()=>{budget=0;last=0;});
reduced.addEventListener('change',()=>{if(reduced.matches){playing=false;schedule();render();}});
document.addEventListener('visibilitychange',schedule);
window.addEventListener('pagehide',()=>{cancelAnimationFrame(request);last=0;});
window.addEventListener('pageshow',schedule);
new ResizeObserver(()=>render()).observe(get('gas-canvas'));
download.addEventListener('click',()=>{
  const data={model:'2D ideal gas bath with elastic tracer collisions; overdamped free-diffusion SDE',units:'reduced: kB=1, bath mass=1',temperature:run.temperature,density:run.density,seed:run.seed,dt:DT,replicas:REPLICAS,bathParticles:64,tracerMass:8,bathRadius:0.12,tracerRadius:0.6,warmupDuration:20,calibrationDuration:80,fitWindow:[40,80],estimator:'free-intercept MSD slope / 4; finite-window estimate, no confidence interval',diffusion:run.diffusion,effectiveFriction:run.temperature/run.diffusion!,calibration:run.calibration,comparison:run.records,displacementTime:run.time,gasDisplacements:run.gasDisplacements,sdeDisplacements:run.sde};
  const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
  const a=document.createElement('a');a.href=url;a.download=`brownian-T${run.temperature}-rho${run.density}.json`;a.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
});
get('year').textContent=String(new Date().getFullYear());
render();schedule();
