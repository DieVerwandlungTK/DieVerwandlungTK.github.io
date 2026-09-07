import { compute, storage, uniforms, type Gpu } from 'vgpu';
import shader from './simulation.wgsl?raw';
import { MASS_H, MASS_O, MOLECULE_COUNTS, MOLECULE_MASS, boxLength, cutoffFor, type Vector3 } from './water-model';
import { buildInitialState, chargeSite, moleculeSites, orientationOf,
  type InitialState, type Quaternion } from './initial-state';
import type { SimulationStats } from './simulation-health';

export { FORCE_LIMIT, needsRestart, type SimulationStats } from './simulation-health';

export interface SimulationOptions {
  molecules: number;
  densityRatio: number;
  temperature: number;
  /** Packed ice configurations by molecule count, from public/data/ice-<N>.bin. */
  ice: Record<number, Float32Array>;
  seed?: number;
}

/** Float32Array's default buffer type admits SharedArrayBuffer, which write() rejects. */
const source = (data: Float32Array): BufferSource => data as BufferSource;

const TIME_STEP = 0.002;      // ps
const FRICTION = 5;           // 1/ps
const WORKGROUP = 64;

export function createSimulation(gpu: Gpu, options: SimulationOptions) {
  const params = uniforms(gpu, {
    box: 0, cutoff: 0, dt: TIME_STEP, temperature: options.temperature,
    friction: FRICTION, molecules: options.molecules, step: 0, seed: 0, scale: 1,
  });
  const forceKernel = compute(gpu, shader, { entry: 'forces' });
  const rescaleKernel = compute(gpu, shader, { entry: 'rescale' });
  const integrateKernel = compute(gpu, shader, { entry: 'integrate' });
  const reduceKernel = compute(gpu, shader, { entry: 'reduce' });
  const statsBuffer = storage(gpu, 4 * 4, 'read-write');

  let molecules = options.molecules;
  let densityRatio = options.densityRatio;
  let temperature = options.temperature;
  let seed = options.seed ?? (Math.random() * 0xffffffff) >>> 0;
  let stepCount = 0;
  let buffers = allocate();
  let simulation = { box: 0, cutoff: 0 };
  /** The box length the uploaded coordinates belong to, so density changes can scale them. */
  let currentBox = 0;
  load();

  function allocate() {
    return {
      state: storage(gpu, molecules * 16 * 4, 'read-write'),
      sites: storage(gpu, molecules * 12 * 4, 'read-write'),
      charges: storage(gpu, molecules * 4 * 4, 'read-write'),
      forceTorque: storage(gpu, molecules * 8 * 4, 'read-write'),
    };
  }

  function bind() {
    const bag = { params, state: buffers.state, sites: buffers.sites,
      charges: buffers.charges, forceTorque: buffers.forceTorque, stats: statsBuffer };
    forceKernel.set(bag);
    rescaleKernel.set(bag);
    integrateKernel.set(bag);
    reduceKernel.set(bag);
  }

  /** Uploads a fresh ice configuration and the matching uniforms. */
  function load(initial?: InitialState) {
    const state = initial ?? buildInitialState(options.ice[molecules],
      { molecules, densityRatio, temperature, seed });
    simulation = { box: state.box, cutoff: cutoffFor(state.box) };
    currentBox = state.box;
    buffers.state.write(source(state.state));
    buffers.sites.write(source(state.sites));
    buffers.charges.write(source(state.charges));
    statsBuffer.write(new Float32Array([temperature, temperature, 0, 0]));
    params.set({ box: simulation.box, cutoff: simulation.cutoff, molecules, temperature,
      step: stepCount, seed, scale: 1 });
    bind();
  }

  const groups = () => Math.ceil(molecules / WORKGROUP);

  async function readSites(): Promise<Float32Array> {
    return new Float32Array(await buffers.sites.read());
  }

  async function readState(): Promise<Float32Array> {
    return new Float32Array(await buffers.state.read());
  }

  async function readStats(): Promise<SimulationStats> {
    reduceKernel.dispatch(1);
    const values = new Float32Array(await statsBuffer.read());
    return {
      translationalTemperature: values[0],
      rotationalTemperature: values[1],
      maximumForce: values[2],
      nonFinite: values[3] > 0,
    };
  }

  /** Advances the sample by `count` BAOAB steps: one force evaluation each. */
  function step(count: number) {
    for (let index = 0; index < count; index++) {
      params.set({ step: stepCount });
      forceKernel.dispatch(groups());
      integrateKernel.dispatch(groups());
      stepCount++;
    }
  }

  /**
   * Test-only entry point: uploads a packed configuration and evaluates forces once, without
   * integrating. The `state`, `sites` and `charges` buffers are restored to whatever they held
   * before the call once it finishes, so the sample evaluated here is not the sample that keeps
   * running — the live simulation is left exactly as it was.
   */
  async function evaluateForces(packed: Float32Array) {
    if (packed.length !== molecules * 9) throw new Error('Packed configuration does not match the molecule count');
    const [savedState, savedSites, savedCharges] = await Promise.all(
      [buffers.state, buffers.sites, buffers.charges].map(async buffer => new Float32Array(await buffer.read())));
    const sites = new Float32Array(molecules * 12);
    const charges = new Float32Array(molecules * 4);
    const state = new Float32Array(molecules * 16);
    for (let molecule = 0; molecule < molecules; molecule++) {
      const read = (site: number): Vector3 => [packed[molecule * 9 + site * 3],
        packed[molecule * 9 + site * 3 + 1], packed[molecule * 9 + site * 3 + 2]];
      const oxygen = read(0), first = read(1), second = read(2);
      const centre = [0, 1, 2].map(axis =>
        (MASS_O * oxygen[axis] + MASS_H * (first[axis] + second[axis])) / MOLECULE_MASS) as Vector3;
      const quaternion: Quaternion = orientationOf(oxygen, first, second);
      moleculeSites(centre, quaternion, simulation.box, sites, molecule * 12);
      const wrapped: Vector3 = [sites[molecule * 12], sites[molecule * 12 + 1], sites[molecule * 12 + 2]];
      const charge = chargeSite(centre, quaternion, wrapped);
      for (const axis of [0, 1, 2]) {
        state[molecule * 16 + axis] = centre[axis];
        charges[molecule * 4 + axis] = charge[axis];
      }
      state[molecule * 16 + 3] = 1;
      charges[molecule * 4 + 3] = 1;
      for (let component = 0; component < 4; component++) {
        state[molecule * 16 + 4 + component] = quaternion[component];
      }
    }
    buffers.state.write(state);
    buffers.sites.write(sites);
    buffers.charges.write(charges);
    forceKernel.dispatch(groups());
    const raw = new Float32Array(await buffers.forceTorque.read());
    const forces = new Float32Array(molecules * 3);
    const torques = new Float32Array(molecules * 3);
    for (let molecule = 0; molecule < molecules; molecule++) {
      for (const axis of [0, 1, 2]) {
        forces[molecule * 3 + axis] = raw[molecule * 8 + axis];
        torques[molecule * 3 + axis] = raw[molecule * 8 + 4 + axis];
      }
    }
    buffers.state.write(source(savedState));
    buffers.sites.write(source(savedSites));
    buffers.charges.write(source(savedCharges));
    return { forces, torques };
  }

  const api = {
    get molecules() { return molecules; },
    get box() { return simulation.box; },
    get cutoff() { return simulation.cutoff; },
    get timePs() { return stepCount * TIME_STEP; },
    readSites,
    readState,
    readStats,
    step,
    evaluateForces,
    setTemperature(kelvin: number) { temperature = kelvin; params.set({ temperature }); },
    /** Scales every centre of mass on the GPU, then rebuilds the sites for the new box. */
    setDensity(ratio: number) {
      const box = boxLength(molecules, ratio);
      simulation = { box, cutoff: cutoffFor(box) };
      params.set({ box, cutoff: simulation.cutoff, scale: box / currentBox });
      rescaleKernel.dispatch(groups());
      params.set({ scale: 1 });
      densityRatio = ratio;
      currentBox = box;
    },
    setMolecules(count: number) {
      if (!(MOLECULE_COUNTS as readonly number[]).includes(count)) {
        throw new Error(`Unsupported molecule count: ${count}. Expected one of ${MOLECULE_COUNTS.join(', ')}`);
      }
      molecules = count;
      stepCount = 0;
      buffers = allocate();
      load();
    },
    reset() {
      seed = (Math.random() * 0xffffffff) >>> 0;
      stepCount = 0;
      load();
    },
    dispose() { /* buffers belong to the gpu and are freed by gpu.dispose() */ },
  };

  return api;
}

export type Simulation = ReturnType<typeof createSimulation>;
