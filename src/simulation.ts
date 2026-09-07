import { compute, storage, uniforms, type Gpu, type StorageBuffer } from 'vgpu';
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

/** vgpu frees storage with gpu.dispose(); an early destroy keeps setMolecules/dispose from
 * leaking the previous buffer set. Mirrors src/background.ts's `release`. */
const release = (buffer: StorageBuffer) => (buffer as { destroy?: () => void }).destroy?.();

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

  /**
   * Reads issued against the live buffer set (readSites/readState/readStats/evaluateForces/
   * pokeState all funnel through `tracked`). A later setMolecules() or dispose() snapshots this
   * set and waits for whatever was already in flight to settle before destroying the buffers it
   * read from -- destroying a buffer while its own read (a GPU copy into a staging buffer, then
   * mapAsync) hasn't finished is a genuine WebGPU hazard, not merely wasted work: the resulting
   * device error reaches src/background.ts's `gpu.onError` handler and tears down the whole GPU.
   * src/main.ts's render loop keeps exactly such a read in flight on every frame, independently of
   * whatever a test or the molecule-count control does, so this is not a hypothetical race.
   *
   * Scoped to this call's closure, not the module: a second createSimulation() must not wait on
   * reads that belong to a different instance's buffers.
   */
  const pendingReads = new Set<Promise<unknown>>();
  function tracked<T>(promise: Promise<T>): Promise<T> {
    pendingReads.add(promise);
    const forget = () => pendingReads.delete(promise);
    promise.then(forget, forget);
    return promise;
  }

  /** Set once dispose() has run; every method that touches the GPU checks this first so a call
   * against a disposed simulation fails with a clear message instead of a raw WebGPU error
   * ("Buffer is destroyed") once the released buffers are actually gone. */
  let disposed = false;
  function ensureLive() {
    if (disposed) throw new Error('Simulation has been disposed and can no longer be used.');
  }

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
    // A paused reset may read diagnostics before stepping. Discard force/torque values
    // from the previous configuration (possibly divergent); step() recomputes them.
    buffers.forceTorque.write(new Float32Array(molecules * 8));
    statsBuffer.write(new Float32Array([temperature, temperature, 0, 0]));
    params.set({ box: simulation.box, cutoff: simulation.cutoff, molecules, temperature,
      step: stepCount, seed, scale: 1 });
    bind();
  }

  const groups = () => Math.ceil(molecules / WORKGROUP);

  async function readSites(): Promise<Float32Array> {
    ensureLive();
    return new Float32Array(await tracked(buffers.sites.read()));
  }

  async function readState(): Promise<Float32Array> {
    ensureLive();
    return new Float32Array(await tracked(buffers.state.read()));
  }

  async function readStats(): Promise<SimulationStats> {
    ensureLive();
    reduceKernel.dispatch(1);
    const values = new Float32Array(await tracked(statsBuffer.read()));
    return {
      translationalTemperature: values[0],
      rotationalTemperature: values[1],
      maximumForce: values[2],
      nonFinite: values[3] > 0,
    };
  }

  /** Advances the sample by `count` BAOAB steps: one force evaluation each. */
  function step(count: number) {
    ensureLive();
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
   *
   * Captures `buffers` and `molecules` up front and re-checks them after every await: a
   * `setMolecules()` landing while this function is suspended reallocates `buffers` (and rebinds
   * every kernel to the new set), so resuming against the names `buffers`/`molecules` would read
   * or write a stale-length array into buffers sized for a different molecule count, and dispatch
   * against a kernel binding that no longer points at the buffers this call just wrote. Safer to
   * throw than to silently operate on a mismatched pair.
   */
  async function evaluateForces(packed: Float32Array) {
    ensureLive();
    const target = buffers;
    const targetMolecules = molecules;
    if (packed.length !== targetMolecules * 9) throw new Error('Packed configuration does not match the molecule count');
    const [savedState, savedSites, savedCharges] = await Promise.all(
      [target.state, target.sites, target.charges].map(async buffer => new Float32Array(await tracked(buffer.read()))));
    if (buffers !== target || molecules !== targetMolecules) {
      throw new Error('setMolecules() ran while evaluateForces() was awaiting a read; discarding this call.');
    }
    const sites = new Float32Array(targetMolecules * 12);
    const charges = new Float32Array(targetMolecules * 4);
    const state = new Float32Array(targetMolecules * 16);
    for (let molecule = 0; molecule < targetMolecules; molecule++) {
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
    target.state.write(state);
    target.sites.write(sites);
    target.charges.write(charges);
    forceKernel.dispatch(Math.ceil(targetMolecules / WORKGROUP));
    const raw = new Float32Array(await tracked(target.forceTorque.read()));
    if (buffers !== target || molecules !== targetMolecules) {
      throw new Error('setMolecules() ran while evaluateForces() was awaiting a read; discarding this call.');
    }
    const forces = new Float32Array(targetMolecules * 3);
    const torques = new Float32Array(targetMolecules * 3);
    // The forces kernel stows each molecule's accumulated pairwise potential energy in the
    // otherwise-unused w component of its torque vec4 (forceTorque[i*2+1].w); see the comment on
    // `energy` there. Summing it over every molecule double counts each pair, since both
    // molecules in a pair independently accumulate the same value into their own w component, so
    // halve the sum to match scripts/reference_forces.py's potential(), which sums each pair once.
    let doubledEnergy = 0;
    for (let molecule = 0; molecule < targetMolecules; molecule++) {
      for (const axis of [0, 1, 2]) {
        forces[molecule * 3 + axis] = raw[molecule * 8 + axis];
        torques[molecule * 3 + axis] = raw[molecule * 8 + 4 + axis];
      }
      doubledEnergy += raw[molecule * 8 + 7];
    }
    target.state.write(source(savedState));
    target.sites.write(source(savedSites));
    target.charges.write(source(savedCharges));
    return { forces, torques, potentialEnergy: doubledEnergy / 2 };
  }

  /**
   * Test-only entry point: overwrites one molecule's raw state vector (the 16-float centre /
   * quaternion / velocity / angular-momentum layout `readState()` returns), leaving every other
   * molecule and every other buffer untouched. `evaluateForces` can only inject a value by
   * uploading a full configuration and re-running the force kernel, which would spread a
   * non-finite value into every neighbouring molecule's own force sum through the pairwise
   * interaction loop -- exactly what the divergence-guard test needs to avoid when it poisons a
   * single molecule. This bypasses the force kernel entirely, so only the named molecule changes.
   *
   * Captures `buffers` up front and re-checks after the await for the same reason
   * `evaluateForces` does: a `setMolecules()` landing during the read would otherwise resume with
   * a stale-length array read from (and about to be written back into) buffers sized for a
   * different molecule count.
   */
  async function pokeState(molecule: number, values: ArrayLike<number>) {
    ensureLive();
    if (values.length !== 16) throw new Error('Expected 16 floats: centre, quaternion, velocity, angular momentum');
    const target = buffers;
    const targetMolecules = molecules;
    const state = new Float32Array(await tracked(target.state.read()));
    if (buffers !== target || molecules !== targetMolecules) {
      throw new Error('setMolecules() ran while pokeState() was awaiting a read; discarding this call.');
    }
    state.set(values, molecule * 16);
    target.state.write(state);
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
    pokeState,
    setTemperature(kelvin: number) {
      ensureLive();
      temperature = kelvin; params.set({ temperature });
    },
    /** Scales every centre of mass on the GPU, then rebuilds the sites for the new box. */
    setDensity(ratio: number) {
      ensureLive();
      const box = boxLength(molecules, ratio);
      simulation = { box, cutoff: cutoffFor(box) };
      params.set({ box, cutoff: simulation.cutoff, scale: box / currentBox });
      rescaleKernel.dispatch(groups());
      params.set({ scale: 1 });
      densityRatio = ratio;
      currentBox = box;
    },
    setMolecules(count: number) {
      ensureLive();
      if (!(MOLECULE_COUNTS as readonly number[]).includes(count)) {
        throw new Error(`Unsupported molecule count: ${count}. Expected one of ${MOLECULE_COUNTS.join(', ')}`);
      }
      const stale = buffers;
      // Snapshot before reassigning `buffers`: anything in here was necessarily reading from
      // `stale` (readSites/readState/readStats/evaluateForces/pokeState all read the buffers
      // current *at the time they were called*), and nothing added after this line can be, since
      // every read call site above resolves `buffers` fresh.
      const staleReads = Array.from(pendingReads);
      molecules = count;
      stepCount = 0;
      buffers = allocate();
      load();   // Binds the new buffers to every kernel before the old ones are released below.
      void Promise.allSettled(staleReads).then(() => {
        for (const buffer of Object.values(stale)) release(buffer);
      });
    },
    reset() {
      ensureLive();
      seed = (Math.random() * 0xffffffff) >>> 0;
      stepCount = 0;
      load();
    },
    /**
     * Releases every buffer this instance owns, including `statsBuffer` (the one buffer
     * `allocate()` doesn't cover, since it is shared across the whole molecule-count lifetime
     * rather than reallocated by setMolecules()). Idempotent -- a second call is a no-op, not a
     * double-free -- and marks the instance disposed so every other method throws a clear error
     * instead of eventually hitting a raw "Buffer is destroyed" once the released buffers are
     * actually gone. Mirrors setMolecules()'s wait for in-flight reads to settle before
     * destroying the buffers they were reading from.
     */
    dispose() {
      if (disposed) return;
      disposed = true;
      const stale = buffers;
      const staleReads = Array.from(pendingReads);
      void Promise.allSettled(staleReads).then(() => {
        for (const buffer of Object.values(stale)) release(buffer);
        release(statsBuffer);
      });
    },
  };

  return api;
}

export type Simulation = ReturnType<typeof createSimulation>;
