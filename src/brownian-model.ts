/** Reduced units: kB = bath mass = 1. Bath particles interact only with the tracer. */
export interface Point { x: number; y: number }
export interface Body extends Point { vx: number; vy: number; mass: number }
export interface Gas { tracer: Body; bath: Body[]; box: number; collisions: number }
export const TRACER_RADIUS = 0.6;
export const BATH_RADIUS = 0.12;
const CONTACT = TRACER_RADIUS + BATH_RADIUS;
export function seededRandom(seed: number): () => number {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
export function gaussian(random: () => number): number {
  return Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
}
export function kineticTemperature(gas: Gas): number {
  const bodies = [gas.tracer, ...gas.bath];
  return bodies.reduce((e, p) => e + p.mass * (p.vx ** 2 + p.vy ** 2), 0) / (2 * (bodies.length - 1));
}
export function createGas(temperature: number, density: number, random: () => number): Gas {
  if (!(temperature > 0 && density > 0 && temperature <= 3 && density <= 1)) throw new RangeError('Invalid gas conditions');
  const box = Math.sqrt(64 / density);
  const tracer: Body = { x: box/2, y: box/2, vx: gaussian(random)/Math.sqrt(8), vy: gaussian(random)/Math.sqrt(8), mass: 8 };
  const bath: Body[] = [];
  for (let i = 0; i < 64; i++) {
    let x: number, y: number;
    do { x = random()*box; y = random()*box; } while (Math.hypot(x-tracer.x,y-tracer.y) < CONTACT);
    bath.push({ x, y, vx: gaussian(random), vy: gaussian(random), mass: 1 });
  }
  const bodies = [tracer, ...bath];
  const mass = 72;
  const ux = bodies.reduce((s,p)=>s+p.mass*p.vx,0)/mass;
  const uy = bodies.reduce((s,p)=>s+p.mass*p.vy,0)/mass;
  for (const p of bodies) { p.vx -= ux; p.vy -= uy; }
  const gas = { tracer, bath, box, collisions: 0 };
  const scale = Math.sqrt(temperature/kineticTemperature(gas));
  for (const p of bodies) { p.vx *= scale; p.vy *= scale; }
  return gas;
}
/** Normal points from a to b. Only approaching bodies exchange momentum. */
export function collide(a: Body, b: Body, nx: number, ny: number): boolean {
  const approach = (b.vx-a.vx)*nx + (b.vy-a.vy)*ny;
  if (approach >= 0) return false;
  const impulse = -2*approach/(1/a.mass + 1/b.mass);
  a.vx -= impulse*nx/a.mass; a.vy -= impulse*ny/a.mass;
  b.vx += impulse*nx/b.mass; b.vy += impulse*ny/b.mass;
  return true;
}
export function gasStep(gas: Gas, dt: number) {
  const a = gas.tracer;
  a.x += a.vx*dt; a.y += a.vy*dt;
  for (const b of gas.bath) {
    b.x += b.vx*dt; b.y += b.vy*dt;
    // Keep bath coordinates bounded. Tracer coordinates remain unwrapped for statistics.
    b.x = ((b.x % gas.box) + gas.box) % gas.box;
    b.y = ((b.y % gas.box) + gas.box) % gas.box;
    let dx = b.x-a.x, dy = b.y-a.y;
    dx -= gas.box*Math.round(dx/gas.box); dy -= gas.box*Math.round(dy/gas.box);
    const distance = Math.hypot(dx,dy);
    if (distance >= CONTACT || distance === 0) continue;
    const nx = dx/distance, ny = dy/distance;
    if (collide(a,b,nx,ny)) gas.collisions++;
    // Small-step hard-disk approximation; mass-weighted separation preserves COM.
    const overlap = CONTACT-distance;
    a.x -= nx*overlap/9; a.y -= ny*overlap/9;
    b.x += nx*overlap*8/9; b.y += ny*overlap*8/9;
  }
}
export function displacements(points: Point[], origins: Point[]): Point[] {
  return points.map((p,i)=>({x:p.x-origins[i].x,y:p.y-origins[i].y}));
}
export function meanSquare(points: Point[]): number {
  return points.reduce((s,p)=>s+p.x*p.x+p.y*p.y,0)/points.length;
}
export function advanceWiener(points: Point[], diffusion: number, dt: number, normal: () => number) {
  const scale = Math.sqrt(2*diffusion*dt);
  for (const p of points) { p.x += scale*normal(); p.y += scale*normal(); }
}
/** Free-intercept least squares; MSD = 4Dt + c in two dimensions. */
export function fitDiffusion(samples: {t: number; msd: number}[]): number | null {
  if (samples.length < 3) return null;
  const mt = samples.reduce((s,p)=>s+p.t,0)/samples.length;
  const my = samples.reduce((s,p)=>s+p.msd,0)/samples.length;
  const covariance = samples.reduce((s,p)=>s+(p.t-mt)*(p.msd-my),0);
  const variance = samples.reduce((s,p)=>s+(p.t-mt)**2,0);
  const d = covariance/variance/4;
  return Number.isFinite(d) && d > 0 ? d : null;
}
