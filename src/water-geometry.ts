export type Vector3 = [number, number, number];
type Quaternion = [number, number, number, number];
export const OH_LENGTH = .9572;
export const HOH_ANGLE = 104.52 * Math.PI / 180;
const length = (v: Vector3) => Math.hypot(...v);
const unit = (v: Vector3): Vector3 => { const n=length(v); if(n<1e-10) throw new Error('Degenerate water orientation'); return v.map(x=>x/n) as Vector3; };
const cross = (a: Vector3,b: Vector3): Vector3 => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const dot = (a: Vector3,b: Vector3) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];

function orientation(h1: Vector3, h2: Vector3): Quaternion {
  const u=unit(h1.map((x,i)=>x+h2[i]) as Vector3);
  const w=unit(cross(h1,h2));
  const v=cross(w,u);
  // Columns form a right-handed water-fixed frame. H1 points toward -v.
  const m=[u[0],v[0],w[0],u[1],v[1],w[1],u[2],v[2],w[2]];
  const trace=m[0]+m[4]+m[8];
  let q: Quaternion;
  if(trace>0){const s=2*Math.sqrt(trace+1); q=[(m[7]-m[5])/s,(m[2]-m[6])/s,(m[3]-m[1])/s,s/4];}
  else if(m[0]>m[4]&&m[0]>m[8]){const s=2*Math.sqrt(1+m[0]-m[4]-m[8]);q=[s/4,(m[1]+m[3])/s,(m[2]+m[6])/s,(m[7]-m[5])/s];}
  else if(m[4]>m[8]){const s=2*Math.sqrt(1+m[4]-m[0]-m[8]);q=[(m[1]+m[3])/s,s/4,(m[5]+m[7])/s,(m[2]-m[6])/s];}
  else{const s=2*Math.sqrt(1+m[8]-m[0]-m[4]);q=[(m[2]+m[6])/s,(m[5]+m[7])/s,s/4,(m[3]-m[1])/s];}
  const norm=Math.hypot(...q);
  return q.map(x=>x/norm) as Quaternion;
}

function rotate(q: Quaternion,v: Vector3): Vector3 {
  const xyz: Vector3=[q[0],q[1],q[2]], c=cross(xyz,v), cc=cross(xyz,c);
  return v.map((x,i)=>x+2*(q[3]*c[i]+cc[i])) as Vector3;
}

/** SLERP preserves rigid water geometry, including a near-180-degree rotation. */
export function interpolateHydrogens(a1: Vector3,a2: Vector3,b1: Vector3,b2: Vector3,t: number): [Vector3,Vector3] {
  const a=orientation(a1,a2), b=orientation(b1,b2);
  let cosine=a.reduce((sum,x,i)=>sum+x*b[i],0);
  if(cosine<0){for(let i=0;i<4;i++) b[i]*=-1;cosine=-cosine;}
  let q: Quaternion;
  if(cosine>.9995){q=a.map((x,i)=>x+(b[i]-x)*t) as Quaternion;const n=Math.hypot(...q);q=q.map(x=>x/n) as Quaternion;}
  else{const angle=Math.acos(Math.min(1,cosine)),den=Math.sin(angle);q=a.map((x,i)=>(Math.sin((1-t)*angle)*x+Math.sin(t*angle)*b[i])/den) as Quaternion;}
  const c=OH_LENGTH*Math.cos(HOH_ANGLE/2),s=OH_LENGTH*Math.sin(HOH_ANGLE/2);
  return [rotate(q,[c,-s,0]),rotate(q,[c,s,0])];
}

export interface HydrogenBond {
  key: string;
  donor: number;
  hydrogen: number;
  acceptor: number;
  shift: Vector3;
  strength: number;
}

/** Geometric criterion, not a bond force: O-O <=3.5 A and O-H...O >=150 deg. */
export function findHydrogenBonds(sites: Float32Array,box: number): HydrogenBond[] {
  const count=sites.length/12,result: HydrogenBond[]=[];
  for(let donor=0;donor<count;donor++) for(let acceptor=0;acceptor<count;acceptor++) {
    if(donor===acceptor) continue;
    const oi=donor*12,oj=acceptor*12;
    const d: Vector3=[sites[oj]-sites[oi],sites[oj+1]-sites[oi+1],sites[oj+2]-sites[oi+2]];
    const shift=d.map(x=>-box*Math.round(x/box)||0) as Vector3;
    for(let axis=0;axis<3;axis++) d[axis]+=shift[axis];
    const distance=length(d);
    if(distance>3.5 || distance<.1) continue;
    for(const hydrogen of [1,2]) {
      const hi=oi+hydrogen*4;
      const oh: Vector3=[sites[hi]-sites[oi],sites[hi+1]-sites[oi+1],sites[hi+2]-sites[oi+2]];
      const ha=d.map((x,i)=>x-oh[i]) as Vector3;
      const cosine=-dot(oh,ha)/(length(oh)*length(ha));
      if(cosine>Math.cos(150*Math.PI/180)) continue;
      const angle=Math.acos(Math.max(-1,cosine))*180/Math.PI;
      // Fade just inside the geometric cutoffs; zero outside them.
      const strength=Math.min(1,(3.5-distance)/.18,(angle-150)/8);
      result.push({key:`${donor}:${hydrogen}:${acceptor}`,donor,hydrogen,acceptor,shift,strength:Math.max(0,strength)});
    }
  }
  return result;
}

export function molecularOpacity(center: Vector3,box: number): number {
  const t=Math.max(0,Math.min(1,(length(center)/box-.42)/.24));
  return 1-t*t*(3-2*t);
}
export interface MolecularImage { molecule: number; shift: Vector3; center: Vector3; opacity: number }

/** Includes periodic copies so crossing a box face only relabels an image. */
export function periodicMolecules(sites: Float32Array,box: number): MolecularImage[] {
  const images: MolecularImage[]=[];
  for(let molecule=0;molecule<sites.length/12;molecule++) {
    for(let x=-1;x<=1;x++) for(let y=-1;y<=1;y++) for(let z=-1;z<=1;z++) {
      const shift: Vector3=[x*box,y*box,z*box];
      const center=shift.map((s,i)=>sites[molecule*12+i]+s-box/2) as Vector3;
      const opacity=molecularOpacity(center,box);
      if(opacity>0) images.push({molecule,shift,center,opacity});
    }
  }
  return images;
}
