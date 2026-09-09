import { advanceWiener, createGas, displacements, fitDiffusion, gasStep, gaussian, meanSquare, seededRandom, type Point } from './brownian-model';
export const DT = 0.01;
export const REPLICAS = 64;
export type Phase = 'warmup' | 'calibration' | 'comparison' | 'complete' | 'failed';
export interface Record { t: number; gas: number; sde: number }
export class BrownianExperiment {
  readonly gases;
  readonly sde: Point[] = Array.from({length:REPLICAS},()=>({x:0,y:0}));
  readonly records: Record[] = [];
  readonly calibration: {t:number;msd:number}[] = [];
  readonly gasTrail: Point[] = [];
  readonly sdeTrail: Point[] = [];
  phase: Phase = 'warmup';
  diffusion: number | null = null;
  private phaseSteps = 0;
  private origins: Point[];
  private normal: () => number;
  constructor(readonly temperature: number, readonly density: number, readonly seed: number) {
    const random = seededRandom(seed);
    this.gases = Array.from({length:REPLICAS},()=>createGas(temperature,density,random));
    this.origins = this.positions();
    const sdeRandom = seededRandom(seed ^ 0x51deabc);
    this.normal = () => gaussian(sdeRandom);
  }
  get time() { return this.phaseSteps*DT; }
  get gasDisplacements() { return displacements(this.positions(),this.origins); }
  private positions() { return this.gases.map(g=>({x:g.tracer.x,y:g.tracer.y})); }
  step(count: number) {
    for(let i=0;i<count;i++) {
      if(this.phase==='complete'||this.phase==='failed') break;
      for(const gas of this.gases) gasStep(gas,DT);
      if(this.phase==='comparison') advanceWiener(this.sde,this.diffusion!,DT,this.normal);
      this.phaseSteps++;
      if(this.phase==='warmup'&&this.phaseSteps===2000) {
        this.phase='calibration'; this.phaseSteps=0; this.origins=this.positions();
      } else if(this.phase==='calibration') {
        if(this.phaseSteps%20===0) this.calibration.push({t:this.time,msd:meanSquare(this.gasDisplacements)});
        if(this.phaseSteps===8000) {
          this.diffusion=fitDiffusion(this.calibration.filter(p=>p.t>=40));
          if(this.diffusion===null) { this.phase='failed'; break; }
          this.phase='comparison'; this.phaseSteps=0; this.origins=this.positions();
          this.records.push({t:0,gas:0,sde:0});
          this.gasTrail.push({x:0,y:0}); this.sdeTrail.push({x:0,y:0});
        }
      } else if(this.phase==='comparison') {
        if(this.phaseSteps%20===0) {
          const delta=this.gasDisplacements;
          this.records.push({t:this.time,gas:meanSquare(delta),sde:meanSquare(this.sde)});
          this.gasTrail.push({...delta[0]}); this.sdeTrail.push({...this.sde[0]});
        }
        if(this.phaseSteps===6000) this.phase='complete';
      }
    }
  }
}
