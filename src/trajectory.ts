import { interpolateHydrogens, type Vector3 } from './water-geometry';

export interface TrajectoryFrame {
  positions: number[] | Float32Array;
  temperature: number;
  timePs: number;
}
export interface Trajectory {
  version: number;
  box: number;
  particles: number;
  frames: TrajectoryFrame[];
  atomOrder?: string[];
  coordinateFile?: string;
}

/** Binary file is little-endian float32, O,H1,H2 xyz per molecule per frame. */
export function decodeTrajectory(value: unknown, binary: ArrayBuffer): Trajectory {
  const header=value as Trajectory;
  if(!header || header.version!==2 || !Number.isInteger(header.particles) || header.particles<1 || header.particles>512 ||
    !Array.isArray(header.frames) || header.frames.length<2 || header.atomOrder?.join(',')!=='O,H,H') throw new Error('Invalid explicit water header');
  const stride=header.particles*9;
  if(binary.byteLength!==header.frames.length*stride*4) throw new Error('Truncated or oversized coordinate file');
  const view=new DataView(binary),coords=new Float32Array(binary.byteLength/4);
  for(let i=0;i<coords.length;i++) coords[i]=view.getFloat32(i*4,true);
  return validateTrajectory({...header,frames:header.frames.map((frame,i)=>({...frame,positions:coords.subarray(i*stride,(i+1)*stride)}))});
}

/** Oxygen is wrapped into the box; its hydrogens stay whole and may sit just outside. */
export function validateTrajectory(value: unknown): Trajectory {
  const data = value as Trajectory;
  if (!data || data.version !== 2 || !Number.isFinite(data.box) || data.box <= 0 ||
      !Number.isInteger(data.particles) || data.particles < 1 || data.particles > 512 ||
      !Array.isArray(data.frames) || data.frames.length < 2) {
    throw new Error('Invalid trajectory header');
  }
  for (const [index, frame] of data.frames.entries()) {
    if (!(Array.isArray(frame.positions) || frame.positions instanceof Float32Array) ||
        frame.positions.length !== data.particles * 9 ||
        !frame.positions.every((p, site) => Number.isFinite(p) &&
          (site % 9 >= 3 ? p >= -1.1 && p <= data.box + 1.1 : p >= 0 && p <= data.box)) ||
        !Number.isFinite(frame.temperature) || !Number.isFinite(frame.timePs) ||
        (index > 0 && frame.timePs <= data.frames[index - 1].timePs)) {
      throw new Error('Invalid trajectory frame');
    }
  }
  return data;
}

/** Writes vec4-aligned GPU storage positions with minimum-image interpolation. */
export function sampleFrame(data: Trajectory, progress: number, output: Float32Array): void {
  const position = Math.min(1, Math.max(0, progress)) * (data.frames.length - 1);
  const first = Math.floor(position);
  const second = Math.min(first + 1, data.frames.length - 1);
  const alpha = position - first;
  for(let molecule=0;molecule<data.particles;molecule++) {
    const offset=molecule*9, target=molecule*12;
    const a=data.frames[first].positions,b=data.frames[second].positions;
    for(let axis=0;axis<3;axis++) {
      let delta=b[offset+axis]-a[offset+axis];
      delta-=data.box*Math.round(delta/data.box);
      const value=a[offset+axis]+delta*alpha;
      output[target+axis]=((value%data.box)+data.box)%data.box;
    }
    const relative=(frame: number[]|Float32Array,h: number): Vector3 => [frame[offset+h*3]-frame[offset],frame[offset+h*3+1]-frame[offset+1],frame[offset+h*3+2]-frame[offset+2]];
    const hydrogens=interpolateHydrogens(relative(a,1),relative(a,2),relative(b,1),relative(b,2),alpha);
    for(let h=0;h<2;h++) for(let axis=0;axis<3;axis++) output[target+(h+1)*4+axis]=output[target+axis]+hydrogens[h][axis];
    output[target+3]=output[target+7]=output[target+11]=1;
  }
}
