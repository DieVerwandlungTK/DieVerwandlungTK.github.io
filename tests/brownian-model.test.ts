import { test } from 'node:test';
import assert from 'node:assert/strict';
import { collide, createGas, gasStep, kineticTemperature, gaussian, seededRandom, advanceWiener, fitDiffusion, displacements } from '../src/brownian-model';

test('elastic collision conserves momentum and kinetic energy and reverses approach', () => {
  const a = { x: 0, y: 0, vx: 1, vy: 0.3, mass: 8 };
  const b = { x: 1, y: 0, vx: -2, vy: -0.7, mass: 1 };
  collide(a, b, 1, 0);
  assert.ok(Math.abs(8*a.vx+b.vx-6)<1e-12);
  assert.ok(Math.abs(8*a.vy+b.vy-1.7)<1e-12);
  assert.ok(Math.abs(8*(a.vx*a.vx+a.vy*a.vy)+b.vx*b.vx+b.vy*b.vy-13.21)<1e-12);
  assert.ok(b.vx>a.vx);
});

test('gas starts at the selected temperature and conserves it through collisions', () => {
  const gas = createGas(2, 0.6, seededRandom(42));
  assert.ok(Math.abs(kineticTemperature(gas)-2)<1e-12);
  for(let i=0;i<10000;i++) gasStep(gas, 0.01);
  assert.ok(Math.abs(kineticTemperature(gas)-2)<1e-9);
  assert.ok(gas.collisions>20);
  assert.ok(Number.isFinite(gas.tracer.x));
});

test('density sets box area and seeded initialization is reproducible', () => {
  const a=createGas(1,0.25,seededRandom(1));
  const b=createGas(1,1,seededRandom(1));
  assert.equal(a.box/b.box,2);
  assert.deepEqual(a,createGas(1,0.25,seededRandom(1)));
});

test('displacements retain multiple periodic box crossings', () => {
  const tracer={x:9.9,y:2,vx:2,vy:0,mass:8};
  const origin={x:9.9,y:2};
  const gas={tracer,bath:[],box:10,collisions:0};
  for(let i=0;i<1000;i++) gasStep(gas,0.01);
  assert.ok(Math.abs(displacements([tracer],[origin])[0].x-20)<1e-10);
  assert.ok(tracer.x>2*gas.box);
});

test('Wiener ensemble has zero mean and variance 2Dt per coordinate', () => {
  const random=seededRandom(123);
  const points=Array.from({length:30000},()=>({x:0,y:0}));
  advanceWiener(points,0.75,2,()=>gaussian(random));
  const mx=points.reduce((s,p)=>s+p.x,0)/points.length;
  const vx=points.reduce((s,p)=>s+p.x*p.x,0)/points.length;
  const vy=points.reduce((s,p)=>s+p.y*p.y,0)/points.length;
  assert.ok(Math.abs(mx)<0.04);
  assert.ok(Math.abs(vx-3)<0.08);
  assert.ok(Math.abs(vy-3)<0.08);
});

test('diffusion regression allows a nonzero intercept and rejects invalid fits', () => {
  assert.equal(fitDiffusion([{t:1,msd:10},{t:2,msd:18},{t:3,msd:26}]),2);
  assert.equal(fitDiffusion([{t:1,msd:3},{t:2,msd:2},{t:3,msd:1}]),null);
  assert.equal(fitDiffusion([]),null);
});
