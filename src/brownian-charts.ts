import type { BrownianExperiment } from './brownian-experiment';
import { BATH_RADIUS, TRACER_RADIUS, type Point } from './brownian-model';
const GAS = '#287e89', SDE = '#b36a45';
const number = (x: number) => Number(x.toPrecision(3)).toString();
const label = (x:number,y:number,text:string,anchor='middle') => `<text x="${x}" y="${y}" text-anchor="${anchor}" fill="#667c84" font-size="11">${text}</text>`;
function axes(xMax:number,yMax:number,xTitle:string,yTitle:string,xMin=0) {
  let svg='';
  for(let i=0;i<=4;i++) {
    const x=56+i*114,y=230-i*48;
    svg+=`<path d="M56 ${y}H512" stroke="#e2e9eb"/>`;
    svg+=label(x,249,number(xMin+(xMax-xMin)*i/4));
    svg+=label(48,y+4,number(yMax*i/4),'end');
  }
  return svg+label(284,274,xTitle)+label(56,22,yTitle,'start');
}
export function drawStatistics(run:BrownianExperiment, msd:SVGSVGElement, histogram:SVGSVGElement) {
  const comparing=run.phase==='comparison'||run.phase==='complete';
  const rows=comparing ? run.records : run.calibration.map(p=>({t:p.t,gas:p.msd,sde:0}));
  const maxT=Math.max(1,rows.at(-1)?.t ?? 1);
  const maxY=Math.max(0.1,...rows.flatMap(p=>[p.gas,p.sde]),comparing?4*run.diffusion!*maxT:0)*1.1;
  let svg=axes(maxT,maxY,'経過時間 t','MSD');
  const line=(values:{t:number;y:number}[],color:string,name:string,dash='') => `<path data-series="${name}" d="${values.map((p,i)=>`${i?'L':'M'}${(56+456*p.t/maxT).toFixed(2)},${(230-192*p.y/maxY).toFixed(2)}`).join(' ')}" fill="none" stroke="${color}" stroke-width="2" ${dash}/>`;
  if(comparing) svg+=line([{t:0,y:0},{t:maxT,y:4*run.diffusion!*maxT}],'#89959c','theory','stroke-dasharray="5 5"');
  svg+=line(rows.map(p=>({t:p.t,y:p.gas})),GAS,'gas');
  if(comparing) svg+=line(rows.map(p=>({t:p.t,y:p.sde})),SDE,'sde');
  if(!rows.length) svg+=label(284,130,'準備運転後に表示');
  msd.innerHTML=svg;
  if(!comparing||run.time===0) {
    histogram.innerHTML=axes(1,1,'x 方向の変位 Δx','確率密度',-1)+label(284,130,'比較区間で表示');
    return;
  }
  const gas=run.gasDisplacements;
  const extent=Math.max(0.1,...gas.map(p=>Math.abs(p.x)),...run.sde.map(p=>Math.abs(p.x)))*1.05;
  const bins=16, width=2*extent/bins;
  const counts=(points:Point[])=>{
    const values=new Array<number>(bins).fill(0);
    for(const p of points) values[Math.min(bins-1,Math.max(0,Math.floor((p.x+extent)/width)))] += 1/(points.length*width);
    return values;
  };
  const a=counts(gas),b=counts(run.sde),peak=Math.max(...a,...b)*1.1;
  svg=axes(extent,peak,'x 方向の変位 Δx','確率密度',-extent);
  for(let i=0;i<bins;i++) {
    const x=56+i*456/bins;
    svg+=`<rect data-series="gas" x="${x}" y="${230-192*a[i]/peak}" width="${456/bins}" height="${192*a[i]/peak}" fill="${GAS}" fill-opacity=".24"/>`;
    svg+=`<rect data-series="sde" x="${x}" y="${230-192*b[i]/peak}" width="${456/bins}" height="${192*b[i]/peak}" fill="none" stroke="${SDE}" stroke-width="1.5"/>`;
  }
  histogram.innerHTML=svg;
}
export function drawTrajectory(canvas:HTMLCanvasElement,run:BrownianExperiment,kind:'gas'|'sde') {
  const context=canvas.getContext('2d');
  if(!context) return;
  const dpr=Math.min(devicePixelRatio||1,2);
  const width=Math.max(1,canvas.clientWidth),height=width*330/540;
  if(canvas.width!==Math.round(width*dpr)||canvas.height!==Math.round(height*dpr)) {
    canvas.width=Math.round(width*dpr); canvas.height=Math.round(height*dpr);
  }
  context.setTransform(canvas.width/540,0,0,canvas.height/330,0,0);
  context.clearRect(0,0,540,330);
  const box=run.gases[0].box,side=284,left=128,top=20;
  const wrap=(x:number)=>((x%box)+box)%box;
  const map=(p:Point)=>({x:left+wrap(p.x)*side/box,y:top+wrap(p.y)*side/box});
  context.fillStyle='#f1f6f7';context.fillRect(left,top,side,side);
  context.strokeStyle='#d9e5e7';context.strokeRect(left,top,side,side);
  const comparing=run.phase==='comparison'||run.phase==='complete';
  if(kind==='sde'&&!comparing) {
    context.fillStyle='#71858b';context.font='14px sans-serif';context.textAlign='center';
    context.fillText('拡散係数の推定を待っています',270,160);return;
  }
  const gas=run.gases[0];
  const delta=run.gasDisplacements[0];
  const origin=kind==='gas'?{x:gas.tracer.x-delta.x,y:gas.tracer.y-delta.y}:{x:box/2,y:box/2};
  const trail=kind==='gas'?run.gasTrail:run.sdeTrail;
  const color=kind==='gas'?GAS:SDE;
  context.save();context.beginPath();context.rect(left,top,side,side);context.clip();
  if(kind==='gas') {
    context.fillStyle='#acbec5';
    for(const b of gas.bath) {
      const p=map(b);context.beginPath();context.arc(p.x,p.y,Math.max(1.5,BATH_RADIUS*side/box),0,2*Math.PI);context.fill();
    }
  }
  context.strokeStyle=color;context.lineWidth=1.6;context.beginPath();
  let previous:Point|undefined;
  for(const delta of trail) {
    const p=map({x:origin.x+delta.x,y:origin.y+delta.y});
    if(!previous||Math.abs(p.x-previous.x)>side/2||Math.abs(p.y-previous.y)>side/2) context.moveTo(p.x,p.y);
    else context.lineTo(p.x,p.y);
    previous=p;
  }
  context.stroke();
  const p=map(kind==='gas'?gas.tracer:{x:origin.x+run.sde[0].x,y:origin.y+run.sde[0].y});
  context.fillStyle=color;context.beginPath();context.arc(p.x,p.y,TRACER_RADIUS*side/box,0,2*Math.PI);context.fill();
  context.restore();context.fillStyle='#71858b';context.font='11px sans-serif';context.textAlign='center';
  context.fillText(`L = ${box.toFixed(2)} · 周期境界`,270,322);
}
