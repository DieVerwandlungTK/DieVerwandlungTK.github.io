export type Vector3 = [number, number, number];
export const OH_LENGTH = .9572;
export const HOH_ANGLE = 104.52 * Math.PI / 180;
const length = (v: Vector3) => Math.hypot(...v);
const dot = (a: Vector3,b: Vector3) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];

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
