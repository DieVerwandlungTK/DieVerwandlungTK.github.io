# Interactive real-time water simulation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the precomputed trajectory background with a rigid TIP4P-Ew molecular dynamics simulation that runs in the visitor's browser, controlled by temperature, density, molecule count, pause and reset.

**Architecture:** Two WebGPU compute kernels per step (forces and torques; BAOAB rigid-body integration) plus a statistics reduction, all owned by `src/simulation.ts`. Positions are read back asynchronously once per rendered frame and fed to the existing tested CPU scene builder, which already handles periodic images, the observation window and hydrogen-bond geometry. The recorded OpenMM trajectory becomes the reference solution used by tests, not display data.

**Tech Stack:** TypeScript, vgpu 0.4.0 (WebGPU), WGSL compute shaders, Vite, node:test with tsx, Playwright with Chromium WebGPU, Python 3.13 with NumPy and OpenMM for offline reference data.

## Global Constraints

- Units everywhere: angstrom, kJ/mol, elementary charge, amu, picoseconds. Kelvin for temperature.
- `COULOMB = 1389.35456` kJ/mol·Å/e², `BOLTZMANN = 0.0083144626` kJ/mol/K.
- `FORCE_TO_ACCELERATION = 100`: acceleration in Å/ps² equals 100 × force[kJ/mol/Å] / mass[amu].
- TIP4P-Ew: O-H 0.9572 Å, H-O-H 104.52°, M site 0.125 Å from O along the bisector, `q_H = 0.52422`, `q_M = -1.04844`, `sigma_O = 3.16435` Å, `epsilon_O = 0.680946` kJ/mol, `mass_O = 15.99943`, `mass_H = 1.007947` amu.
- Molecule counts are exactly 64, 216, 512 (ice Ic cells 2, 3, 4; 8 molecules per 6.35 Å cell). Default 216, and 64 on viewports of 720 px or less.
- Box length: `6.35 * cells / densityRatio ** (1/3)`, density ratio from 0.6 to 1.4, default 1.0.
- Cutoff: `min(9, 0.49 * box)` — never half the box or more, so the minimum image convention holds at every density. Small systems therefore get a short cutoff (6.2 Å at 64 molecules and ice density, matching the 6 Å the OpenMM reference used).
- Electrostatics: Onsager reaction field with conducting boundary. `U(r) = COULOMB * qa * qb * (1/r + r²/(2 rc³) - 3/(2 rc))`, radial force magnitude `COULOMB * qa * qb * (1/r² - r/rc³)`. Both vanish at `rc`.
- Lennard-Jones on O-O only, force-shifted: force `-U_lj'(r) + U_lj'(rc)`, whose integral is `U(r) = U_lj(r) - U_lj(rc) - (r - rc) * U_lj'(rc)` — that is, plus `(r - rc)` times the *force* at the cutoff, which is how the code writes it.
- Pair inclusion is decided per molecule by the minimum-image O-O distance; all site pairs of an included molecule pair use that same molecular shift, so molecules are never split.
- Integration: 2 fs time step, BAOAB Langevin on translation and rotation, friction 5 ps⁻¹.
- The M site needs no force redistribution: rigid-body integration consumes only the net force and the net torque about the centre of mass, and the M site belongs to that rigid body.
- Body frame: x along the H-O-H bisector pointing at the hydrogens, y in the molecular plane with H1 at +y, z out of plane, origin at the centre of mass. Quaternions are `vec4(x, y, z, w)` mapping body to space.
- Every new TypeScript file passes `npx tsc --noEmit`. Node tests live in `tests/*.test.ts` (run by `npm test`); browser tests in `tests/browser/*.spec.ts` (run by `npm run test:browser` after `npm run build`).
- Japanese is the language of all visitor-facing copy; code, comments, commit messages and documents in `docs/` are English.
- Never commit a red test. Each task ends with its tests passing and a commit.
- Browser specs run in node, so they import constants from `src/water-model.ts` rather than restating them; only code inside `page.evaluate` needs values passed in as arguments.
- Tests that step the simulation for seconds call `test.setTimeout(...)` explicitly, since Playwright's default is 30 s.

## File Structure

**Created**

- `src/water-model.ts` — TIP4P-Ew constants, body-frame sites, principal moments of inertia, box/density/cutoff relations. Pure functions, no GPU.
- `src/initial-state.ts` — decodes a packed ice configuration, scales it to a density, and draws seeded Maxwell-Boltzmann velocities and angular momenta with zero net momentum.
- `src/simulation.wgsl` — `forces`, `integrate` and `reduce` compute entry points.
- `src/simulation.ts` — owns GPU buffers and kernel dispatch; exposes `step`, `readSites`, `stats`, `setTemperature`, `setDensity`, `reset`, and the `window.waterSimulation` test interface.
- `scripts/reference_forces.py` — Python implementation of the same force field; writes the force fixture.
- `scripts/test_reference_forces.py` — validates that implementation against numerical derivatives.
- `tests/water-model.test.ts`, `tests/initial-state.test.ts` — node unit tests.
- `tests/browser/simulation.spec.ts` — GPU physics tests (forces, rigid geometry, thermostat, momentum, melting, structure, stability).
- `public/data/ice-64.bin`, `ice-216.bin`, `ice-512.bin` — energy-minimized proton-disordered ice Ic configurations (float32 LE, molecule-major, O then H then H, xyz).
- `tests/fixtures/reference-forces-64.json` — configuration, forces, torques and potential energy from the Python reference.
- `tests/fixtures/reference-rdf-300k.json` — oxygen-oxygen radial distribution function from the OpenMM reference trajectory.

**Modified**

- `scripts/generate_explicit_water.py` — generalize `ice_configuration` to any cell count; export the three initial states; write the recorded trajectory to `reference/` instead of `public/data/`.
- `scripts/test_explicit_water.py` — ice rules and rigid geometry for all three cell counts.
- `scripts/generate_static.py` — build the fallback SVG from `public/data/ice-216.bin`.
- `src/scene.ts` — take the box length per `update` call instead of freezing it at construction.
- `src/water-geometry.ts` — delete `interpolateHydrogens` and its quaternion helpers.
- `src/main.ts` — drive the simulation and the new controls.
- `src/background.ts` — accept sites directly instead of a trajectory.
- `index.html`, `src/style.css` — new control panel and caption.
- `tests/scene.test.ts` — box passed to `update`.
- `tests/browser/homepage.spec.ts` — controls instead of the timeline.
- `README.md`, `docs/explicit-water-simulation.md` — describe the live simulation and the reference role of the recorded trajectory.

**Deleted**

- `src/trajectory.ts`, `tests/trajectory.test.ts`, `tests/water-geometry.test.ts`'s interpolation test, `public/data/water.json`, `public/data/water.bin`.

---

### Task 1: Water model constants and geometry

**Files:**
- Create: `src/water-model.ts`
- Test: `tests/water-model.test.ts`

**Interfaces:**
- Consumes: nothing.
- Produces: `OH_LENGTH`, `HOH_ANGLE`, `Q_H`, `Q_M`, `M_OFFSET`, `SIGMA_O`, `EPSILON_O`, `MASS_O`, `MASS_H`, `MOLECULE_MASS`, `COULOMB`, `BOLTZMANN`, `FORCE_TO_ACCELERATION`, `ICE_CELL`, `MOLECULE_COUNTS: readonly [64, 216, 512]`, `bodySites(): { oxygen: Vector3; hydrogen: [Vector3, Vector3]; charge: Vector3 }`, `principalMoments(): Vector3`, `cellsFor(molecules: number): number`, `boxLength(molecules: number, densityRatio: number): number`, `cutoffFor(box: number): number`.

- [ ] **Step 1: Write the failing test**

Create `tests/water-model.test.ts`:

```typescript
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { BOLTZMANN, MASS_H, MASS_O, MOLECULE_COUNTS, MOLECULE_MASS, OH_LENGTH, HOH_ANGLE,
  bodySites, boxLength, cellsFor, cutoffFor, principalMoments } from '../src/water-model.ts';

test('body-frame sites keep TIP4P-Ew geometry with the centre of mass at the origin', () => {
  const { oxygen, hydrogen, charge } = bodySites();
  for (const h of hydrogen) {
    const bond = h.map((value, axis) => value - oxygen[axis]);
    assert.ok(Math.abs(Math.hypot(...bond) - OH_LENGTH) < 1e-9);
  }
  const first = hydrogen[0].map((value, axis) => value - oxygen[axis]);
  const second = hydrogen[1].map((value, axis) => value - oxygen[axis]);
  const cosine = first.reduce((sum, value, axis) => sum + value * second[axis], 0) / OH_LENGTH ** 2;
  assert.ok(Math.abs(cosine - Math.cos(HOH_ANGLE)) < 1e-9);
  for (let axis = 0; axis < 3; axis++) {
    const moment = MASS_O * oxygen[axis] + MASS_H * (hydrogen[0][axis] + hydrogen[1][axis]);
    assert.ok(Math.abs(moment) < 1e-9, `centre of mass on axis ${axis}`);
  }
  // The M site sits 0.125 A from oxygen along the bisector, which is the body x axis.
  assert.ok(Math.abs(charge[0] - (oxygen[0] + 0.125)) < 1e-9);
  assert.ok(Math.abs(charge[1]) < 1e-12 && Math.abs(charge[2]) < 1e-12);
  assert.ok(Math.abs(MOLECULE_MASS - (MASS_O + 2 * MASS_H)) < 1e-12);
});

test('principal moments match the measured moments of inertia of water', () => {
  // Experimental values 1.918, 1.021, 2.939 e-40 g cm2 equal these amu A2 numbers.
  const [x, y, z] = principalMoments();
  assert.ok(Math.abs(x - 1.15509) < 1e-3, `x ${x}`);
  assert.ok(Math.abs(y - 0.61473) < 1e-3, `y ${y}`);
  assert.ok(Math.abs(z - 1.76982) < 1e-3, `z ${z}`);
  // A planar rigid body satisfies Iz = Ix + Iy.
  assert.ok(Math.abs(z - (x + y)) < 1e-9);
});

test('box length follows the molecule count and density, and the cutoff never reaches half the box', () => {
  assert.deepEqual([...MOLECULE_COUNTS], [64, 216, 512]);
  assert.deepEqual(MOLECULE_COUNTS.map(cellsFor), [2, 3, 4]);
  assert.ok(Math.abs(boxLength(64, 1) - 12.7) < 1e-9);
  assert.ok(Math.abs(boxLength(216, 1) - 19.05) < 1e-9);
  assert.ok(Math.abs(boxLength(512, 1) - 25.4) < 1e-9);
  // Compressing to 1.4 times ice density shrinks the box by the cube root.
  assert.ok(Math.abs(boxLength(216, 1.4) - 19.05 / 1.4 ** (1 / 3)) < 1e-9);
  for (const molecules of MOLECULE_COUNTS) {
    for (const ratio of [0.6, 1, 1.4]) {
      const box = boxLength(molecules, ratio);
      const cutoff = cutoffFor(box);
      assert.ok(cutoff < box / 2, `cutoff ${cutoff} must stay below half of ${box}`);
      assert.ok(cutoff <= 9);
    }
  }
  assert.ok(Math.abs(cutoffFor(25.4) - 9) < 1e-9);
  assert.ok(Math.abs(cutoffFor(12.7) - 0.49 * 12.7) < 1e-9);
});

test('Boltzmann constant is in kJ per mole per kelvin', () => {
  assert.ok(Math.abs(BOLTZMANN - 0.0083144626) < 1e-12);
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npm test`
Expected: FAIL — `Cannot find module '../src/water-model.ts'`.

- [ ] **Step 3: Write the implementation**

Create `src/water-model.ts`:

```typescript
export type Vector3 = [number, number, number];

/** TIP4P-Ew (Horn et al. 2004) in angstrom, kJ/mol, e, amu, ps. */
export const OH_LENGTH = 0.9572;
export const HOH_ANGLE = 104.52 * Math.PI / 180;
export const M_OFFSET = 0.125;
export const Q_H = 0.52422;
export const Q_M = -1.04844;
export const SIGMA_O = 3.16435;
export const EPSILON_O = 0.680946;
export const MASS_O = 15.99943;
export const MASS_H = 1.007947;
export const MOLECULE_MASS = MASS_O + 2 * MASS_H;
export const COULOMB = 1389.35456;
export const BOLTZMANN = 0.0083144626;
/** Acceleration in A/ps^2 per (kJ/mol/A) of force and amu of mass. */
export const FORCE_TO_ACCELERATION = 100;
/** Conventional ice Ic cell holding eight molecules. */
export const ICE_CELL = 6.35;
export const MOLECULE_COUNTS = [64, 216, 512] as const;

/**
 * Sites in the body frame: x along the bisector towards the hydrogens, y in the
 * molecular plane, z out of plane, origin at the centre of mass.
 */
export function bodySites(): { oxygen: Vector3; hydrogen: [Vector3, Vector3]; charge: Vector3 } {
  const half = HOH_ANGLE / 2;
  const along = OH_LENGTH * Math.cos(half), across = OH_LENGTH * Math.sin(half);
  const shift = 2 * MASS_H * along / MOLECULE_MASS;
  return {
    oxygen: [-shift, 0, 0],
    hydrogen: [[along - shift, across, 0], [along - shift, -across, 0]],
    charge: [M_OFFSET - shift, 0, 0],
  };
}

/** Diagonal inertia tensor of the rigid molecule in amu A^2. */
export function principalMoments(): Vector3 {
  const { oxygen, hydrogen } = bodySites();
  const sites: [number, Vector3][] = [[MASS_O, oxygen], [MASS_H, hydrogen[0]], [MASS_H, hydrogen[1]]];
  const moment = (first: number, second: number) =>
    sites.reduce((sum, [mass, site]) => sum + mass * (site[first] ** 2 + site[second] ** 2), 0);
  return [moment(1, 2), moment(0, 2), moment(0, 1)];
}

export const cellsFor = (molecules: number): number => Math.round((molecules / 8) ** (1 / 3));

/** Ice density is eight molecules per cell; the ratio compresses that isotropically. */
export const boxLength = (molecules: number, densityRatio: number): number =>
  ICE_CELL * cellsFor(molecules) / densityRatio ** (1 / 3);

/** Stays under half the box so the minimum image convention holds. */
export const cutoffFor = (box: number): number => Math.min(9, 0.49 * box);
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npm test`
Expected: PASS — the four new tests plus the existing scene, trajectory and geometry tests.

Run: `npx tsc --noEmit`
Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add src/water-model.ts tests/water-model.test.ts
git commit -m "Add TIP4P-Ew model constants, body-frame sites and box relations"
```

---

### Task 2: Export ice initial states for 64, 216 and 512 molecules

**Files:**
- Modify: `scripts/generate_explicit_water.py`
- Modify: `scripts/test_explicit_water.py`
- Create: `public/data/ice-64.bin`, `public/data/ice-216.bin`, `public/data/ice-512.bin` (generated)
- Move: `public/data/water.json` and `water.bin` to `reference/`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `ice_configuration(cells=2, seed=SEED)` returning `(xyz, directed, adjacent)` for `8 * cells ** 3` molecules with box `6.35 * cells`; `export_initial_states()` writing `public/data/ice-<N>.bin` as little-endian float32, molecule-major, O then H1 then H2, xyz, angstrom, oxygen wrapped into `[0, box)` with hydrogens whole.

- [ ] **Step 1: Write the failing test**

Add to `scripts/test_explicit_water.py`:

```python
class InitialStates(unittest.TestCase):
    def test_ice_rules_hold_for_every_exported_cell_count(self):
        for cells in (2, 3, 4):
            count = 8 * cells ** 3
            xyz, directed, adjacent = ice_configuration(cells=cells)
            self.assertEqual(xyz.shape, (count, 3, 3))
            self.assertTrue(np.all(adjacent.sum(axis=1) == 4))
            # Each oxygen donates two protons and accepts two.
            self.assertTrue(np.all(directed.sum(axis=1) == 2))
            self.assertTrue(np.all(directed.sum(axis=0) == 2))
            self.assertFalse(np.any(directed & directed.T))

    def test_exported_files_are_rigid_wrapped_and_the_right_size(self):
        for cells in (2, 3, 4):
            count = 8 * cells ** 3
            box = 6.35 * cells
            path = ROOT / f'public/data/ice-{count}.bin'
            raw = np.fromfile(path, dtype='<f4')
            self.assertEqual(raw.size, count * 9)
            xyz = raw.reshape(count, 3, 3).astype(float)
            self.assertTrue(np.isfinite(xyz).all())
            self.assertTrue(np.all((xyz[:, 0] >= 0) & (xyz[:, 0] < box)))
            oh = xyz[:, 1:] - xyz[:, :1]
            lengths = np.linalg.norm(oh, axis=-1)
            self.assertLess(np.max(np.abs(lengths - .9572)), 2e-3)
            cosines = np.sum(oh[:, 0] * oh[:, 1], axis=-1) / np.prod(lengths, axis=-1)
            angles = np.rad2deg(np.arccos(np.clip(cosines, -1, 1)))
            self.assertLess(np.max(np.abs(angles - 104.52)), .1)
            # Nearest oxygen neighbours stay at the lattice spacing after minimisation.
            d = minimum_image(xyz[:, 0][None] - xyz[:, 0][:, None], box)
            r = np.linalg.norm(d, axis=-1)
            np.fill_diagonal(r, np.inf)
            self.assertGreater(r.min(), 2.3)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m unittest discover -s scripts -p 'test_*.py'`
Expected: FAIL — `ice_configuration() got an unexpected keyword argument 'cells'` and missing `public/data/ice-64.bin`.

- [ ] **Step 3: Generalize the generator and export the initial states**

In `scripts/generate_explicit_water.py`, replace `minimum_image` and `ice_configuration` with cell-count-aware versions and add the exporter. The existing `generate()` keeps working by calling `ice_configuration(cells=2)`.

```python
def minimum_image(d, box=BOX):
    return d - box * np.rint(d / box)


def ice_configuration(cells=2, seed=SEED):
    """Proton-disordered ice Ic for 8 * cells**3 molecules, satisfying the ice rules."""
    count = 8 * cells ** 3
    box = ICE_CELL * cells
    rng = np.random.default_rng(seed)
    fcc = np.array([[0,0,0], [0,.5,.5], [.5,0,.5], [.5,.5,0]])
    basis = np.concatenate([fcc, fcc + .25])
    lattice = np.indices((cells,)*3).reshape(3,-1).T
    oxygen = ((lattice[:,None,:] + basis).reshape(-1,3) + .125) * ICE_CELL
    d = minimum_image(oxygen[None] - oxygen[:,None], box)
    distance = np.linalg.norm(d, axis=-1)
    adjacent = (distance < 3) & (distance > 0)
    graph = [set(np.flatnonzero(row)) for row in adjacent]
    stack, circuit = [0], []
    while stack:
        i = stack[-1]
        if graph[i]:
            j = int(rng.choice(sorted(graph[i])))
            graph[i].remove(j)
            graph[j].remove(i)
            stack.append(j)
        else:
            circuit.append(stack.pop())
    directed = np.zeros((count,count), bool)
    for i, j in zip(circuit[:-1], circuit[1:]):
        directed[i,j] = True
    xyz = np.zeros((count,3,3))
    xyz[:,0] = oxygen
    theta = np.deg2rad(104.52/2)
    for i in range(count):
        v = d[i,np.flatnonzero(directed[i])]
        v /= np.linalg.norm(v,axis=1)[:,None]
        bisector = v[0]+v[1]
        bisector /= np.linalg.norm(bisector)
        tangent = v[0]-v[1]
        tangent /= np.linalg.norm(tangent)
        xyz[i,1:] = oxygen[i] + .9572 * np.array([
            np.cos(theta)*bisector + np.sin(theta)*tangent,
            np.cos(theta)*bisector - np.sin(theta)*tangent])
    return xyz, directed, adjacent
```

Add `ICE_CELL = 6.35` next to `BOX` (and set `BOX = ICE_CELL * 2` so the two never drift apart).

Add the exporter, which minimizes each configuration with the same force field as the recorded trajectory so the browser starts from a relaxed lattice:

```python
def minimized(cells):
    """Energy-minimized ice configuration for the browser, in the same units as water.bin."""
    xyz, _, _ = ice_configuration(cells=cells)
    box = ICE_CELL * cells
    topology, system, positions = build_system(xyz, box)
    integrator = mm.LangevinMiddleIntegrator(180*unit.kelvin, 1/unit.picosecond, 2*unit.femtosecond)
    simulation = app.Simulation(topology, system, integrator, PLATFORM, PLATFORM_PROPERTIES)
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy()
    state = simulation.context.getState(getPositions=True)
    drawn = np.array(state.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
    drawn = drawn.reshape(-1, 4, 3)[:, :3]      # drop the virtual site
    drawn[:, 1:] -= ICE_CELL * cells * np.rint((drawn[:, 1:] - drawn[:, :1]) / box)
    drawn[:, 0] -= box * np.floor(drawn[:, 0] / box)
    return drawn


def export_initial_states():
    for cells in (2, 3, 4):
        drawn = minimized(cells)
        destination = ROOT / f'public/data/ice-{8 * cells ** 3}.bin'
        drawn.astype('<f4').tofile(destination)
        print(f'Wrote {destination.name}: {drawn.shape[0]} molecules')
```

The existing `generate()` already builds a system from a configuration; extract that construction into `build_system(xyz, box)` returning `(topology, system, positions)` and call it from both `generate()` and `minimized()`, so the force field, virtual sites and PME settings are defined once. Extract the platform choice into `PLATFORM` and `PLATFORM_PROPERTIES` module constants at the same time.

Change `generate()` to write into `reference/` instead of `public/data/`:

```python
    destination = ROOT / 'reference'
    destination.mkdir(exist_ok=True)
```

At the bottom of the file, run both:

```python
if __name__ == '__main__':
    export_initial_states()
    generate()
```

- [ ] **Step 4: Generate the data and run the tests**

```bash
mkdir -p reference && git mv public/data/water.json public/data/water.bin reference/
.venv/bin/python scripts/generate_explicit_water.py
.venv/bin/python -m unittest discover -s scripts -p 'test_*.py'
```

Expected: `Wrote ice-64.bin` / `ice-216.bin` / `ice-512.bin`, then `OK`. The three files total about 26 KB (`ls -l public/data`).

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_explicit_water.py scripts/test_explicit_water.py public/data reference
git commit -m "Export minimized ice Ic initial states for 64, 216 and 512 molecules"
```

---

### Task 3: Python reference forces for the reaction-field model

**Files:**
- Create: `scripts/reference_forces.py`
- Create: `scripts/test_reference_forces.py`
- Create: `tests/fixtures/reference-forces-64.json` (generated)

**Interfaces:**
- Consumes: `ice_configuration` and `ICE_CELL` from `scripts/generate_explicit_water.py`.
- Produces: `potential(xyz, box, cutoff) -> float`, `forces_and_torques(xyz, box, cutoff) -> (forces, torques)` with shapes `(N, 3)` in kJ/mol/Å and kJ/mol, torques taken about each molecule's centre of mass in the space frame; the fixture file `tests/fixtures/reference-forces-64.json` with keys `box`, `cutoff`, `molecules`, `sites` (flat `N*9`), `forces` (flat `N*3`), `torques` (flat `N*3`), `potentialEnergy`.

- [ ] **Step 1: Write the failing test**

Create `scripts/test_reference_forces.py`:

```python
"""The reference force field must be the gradient of its own potential."""
import unittest
import numpy as np
from reference_forces import BODY, centre_of_mass, forces_and_torques, potential, rotate_molecule
from generate_explicit_water import ice_configuration, ICE_CELL

BOX = ICE_CELL * 2
CUTOFF = .49 * BOX


class ReferenceForces(unittest.TestCase):
    def setUp(self):
        self.xyz, _, _ = ice_configuration(cells=2)

    def test_force_matches_the_numerical_gradient_of_the_potential(self):
        forces, _ = forces_and_torques(self.xyz, BOX, CUTOFF)
        step = 1e-4
        for molecule in (0, 7, 31):
            for axis in range(3):
                shift = np.zeros(3)
                shift[axis] = step
                plus = self.xyz.copy(); plus[molecule] += shift
                minus = self.xyz.copy(); minus[molecule] -= shift
                gradient = (potential(plus, BOX, CUTOFF) - potential(minus, BOX, CUTOFF)) / (2 * step)
                self.assertLess(abs(-gradient - forces[molecule, axis]),
                                1e-5 * max(1.0, abs(forces[molecule, axis])),
                                f'molecule {molecule} axis {axis}')

    def test_torque_matches_the_numerical_derivative_under_rigid_rotation(self):
        _, torques = forces_and_torques(self.xyz, BOX, CUTOFF)
        angle = 1e-5
        for molecule in (0, 12, 63):
            for axis in range(3):
                direction = np.eye(3)[axis]
                plus = self.xyz.copy()
                minus = self.xyz.copy()
                plus[molecule] = rotate_molecule(self.xyz[molecule], direction, angle)
                minus[molecule] = rotate_molecule(self.xyz[molecule], direction, -angle)
                gradient = (potential(plus, BOX, CUTOFF) - potential(minus, BOX, CUTOFF)) / (2 * angle)
                self.assertLess(abs(-gradient - torques[molecule, axis]),
                                1e-4 * max(1.0, abs(torques[molecule, axis])),
                                f'molecule {molecule} axis {axis}')

    def test_potential_and_forces_vanish_beyond_the_cutoff(self):
        far = np.array([self.xyz[0], self.xyz[0] + np.array([[CUTOFF + .5, 0, 0]])])
        forces, torques = forces_and_torques(far, 10 * BOX, CUTOFF)
        self.assertAlmostEqual(potential(far, 10 * BOX, CUTOFF), 0.0, places=9)
        self.assertLess(np.abs(forces).max(), 1e-9)
        self.assertLess(np.abs(torques).max(), 1e-9)

    def test_body_frame_matches_the_typescript_model(self):
        centre = centre_of_mass(self.xyz[0])
        self.assertLess(np.abs(BODY['oxygen'][1:]).max(), 1e-12)
        self.assertAlmostEqual(float(np.linalg.norm(BODY['hydrogen'][0] - BODY['oxygen'])), .9572, places=9)
        self.assertEqual(centre.shape, (3,))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m unittest discover -s scripts -p 'test_reference_forces.py'`
Expected: FAIL — `ModuleNotFoundError: No module named 'reference_forces'`.

- [ ] **Step 3: Write the reference implementation**

Create `scripts/reference_forces.py`:

```python
"""Reference reaction-field TIP4P-Ew forces, and the fixture the GPU is tested against.

This is the definition of the model the browser simulation implements. It is written
for clarity, not speed: 64 molecules and explicit loops.
"""
import json
from pathlib import Path
import numpy as np
from generate_explicit_water import ICE_CELL, ice_configuration

ROOT = Path(__file__).resolve().parents[1]
COULOMB = 1389.35456
Q_H, Q_M = .52422, -1.04844
SIGMA, EPSILON = 3.16435, .680946
MASS_O, MASS_H = 15.99943, 1.007947
M_OFFSET = .125
CHARGES = np.array([Q_H, Q_H, Q_M])
BODY = {'oxygen': np.array([-2 * MASS_H * .9572 * np.cos(np.deg2rad(52.26)) / (MASS_O + 2 * MASS_H), 0, 0]),
        'hydrogen': None}


def centre_of_mass(molecule):
    return (MASS_O * molecule[0] + MASS_H * (molecule[1] + molecule[2])) / (MASS_O + 2 * MASS_H)


def charge_sites(molecule):
    """The two hydrogens and the M site, 0.125 A from oxygen along the bisector."""
    bisector = (molecule[1] - molecule[0]) + (molecule[2] - molecule[0])
    bisector = bisector / np.linalg.norm(bisector)
    return np.array([molecule[1], molecule[2], molecule[0] + M_OFFSET * bisector])


def rotate_molecule(molecule, axis, angle):
    """Rotates one molecule rigidly about its centre of mass (Rodrigues)."""
    centre = centre_of_mass(molecule)
    k = axis / np.linalg.norm(axis)
    local = molecule - centre
    rotated = (local * np.cos(angle)
               + np.cross(k, local) * np.sin(angle)
               + np.outer(local @ k, k) * (1 - np.cos(angle)))
    return rotated + centre


def _pairs(xyz, box, cutoff):
    """Yields molecule pairs within the cutoff together with the shift applied to j."""
    for i in range(len(xyz)):
        for j in range(i + 1, len(xyz)):
            delta = xyz[j, 0] - xyz[i, 0]
            shift = -box * np.rint(delta / box)
            if np.linalg.norm(delta + shift) < cutoff:
                yield i, j, shift


def _lennard_jones(r, cutoff):
    """Force-shifted LJ energy and radial force magnitude (positive is repulsive)."""
    def raw(x):
        s6 = (SIGMA / x) ** 6
        return 4 * EPSILON * (s6 * s6 - s6), 24 * EPSILON * (2 * s6 * s6 - s6) / x
    energy, force = raw(r)
    edge_energy, edge_force = raw(cutoff)
    return energy - edge_energy + (r - cutoff) * edge_force, force - edge_force


def _reaction_field(r, product, cutoff):
    """Onsager reaction field with conducting boundary; energy and force both vanish at the cutoff."""
    energy = COULOMB * product * (1 / r + r * r / (2 * cutoff ** 3) - 3 / (2 * cutoff))
    force = COULOMB * product * (1 / (r * r) - r / cutoff ** 3)
    return energy, force


def potential(xyz, box, cutoff):
    total = 0.0
    for i, j, shift in _pairs(xyz, box, cutoff):
        d = (xyz[j, 0] + shift) - xyz[i, 0]
        total += _lennard_jones(float(np.linalg.norm(d)), cutoff)[0]
        first, second = charge_sites(xyz[i]), charge_sites(xyz[j]) + shift
        for a, qa in zip(first, CHARGES):
            for b, qb in zip(second, CHARGES):
                total += _reaction_field(float(np.linalg.norm(b - a)), qa * qb, cutoff)[0]
    return total


def forces_and_torques(xyz, box, cutoff):
    forces = np.zeros((len(xyz), 3))
    torques = np.zeros((len(xyz), 3))
    centres = np.array([centre_of_mass(molecule) for molecule in xyz])

    def apply(index, point, force):
        forces[index] += force
        torques[index] += np.cross(point - centres[index], force)

    for i, j, shift in _pairs(xyz, box, cutoff):
        d = (xyz[j, 0] + shift) - xyz[i, 0]
        r = float(np.linalg.norm(d))
        magnitude = _lennard_jones(r, cutoff)[1]
        force = magnitude * d / r                      # on j, away from i
        # The lever arm for j uses its real (unshifted) site position: translating the
        # whole molecule by `shift` would translate its centre of mass by the same
        # amount, so the shift cancels out of (site - centre). The shift belongs only in
        # the minimum-image direction of the force.
        apply(j, xyz[j, 0], force)
        apply(i, xyz[i, 0], -force)
        first, second = charge_sites(xyz[i]), charge_sites(xyz[j])
        for a, qa in zip(first, CHARGES):
            for b, qb in zip(second, CHARGES):
                delta = (b + shift) - a
                distance = float(np.linalg.norm(delta))
                magnitude = _reaction_field(distance, qa * qb, cutoff)[1]
                force = magnitude * delta / distance
                apply(j, b, force)
                apply(i, a, -force)
    return forces, torques


def write_fixture():
    box = ICE_CELL * 2
    cutoff = .49 * box
    xyz, _, _ = ice_configuration(cells=2)
    forces, torques = forces_and_torques(xyz, box, cutoff)
    fixture = {'box': box, 'cutoff': cutoff, 'molecules': len(xyz),
               'sites': xyz.reshape(-1).tolist(),
               'forces': forces.reshape(-1).tolist(),
               'torques': torques.reshape(-1).tolist(),
               'potentialEnergy': potential(xyz, box, cutoff)}
    destination = ROOT / 'tests/fixtures/reference-forces-64.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(fixture) + '\n')
    print(f'Wrote {destination.relative_to(ROOT)}: '
          f'max force {np.abs(forces).max():.3f} kJ/mol/A, energy {fixture["potentialEnergy"]:.1f} kJ/mol')


if __name__ == '__main__':
    write_fixture()
```

Fill in `BODY['hydrogen']` with the two body-frame hydrogen positions so the last test compares against the same frame as `src/water-model.ts`:

```python
_HALF = np.deg2rad(104.52 / 2)
_ALONG, _ACROSS = .9572 * np.cos(_HALF), .9572 * np.sin(_HALF)
_SHIFT = 2 * MASS_H * _ALONG / (MASS_O + 2 * MASS_H)
BODY = {'oxygen': np.array([-_SHIFT, 0, 0]),
        'hydrogen': np.array([[_ALONG - _SHIFT, _ACROSS, 0], [_ALONG - _SHIFT, -_ACROSS, 0]]),
        'charge': np.array([M_OFFSET - _SHIFT, 0, 0])}
```

- [ ] **Step 4: Run the tests and write the fixture**

Run: `.venv/bin/python -m unittest discover -s scripts -p 'test_reference_forces.py'`
Expected: PASS — 4 tests.

Run: `.venv/bin/python scripts/reference_forces.py`
Expected: `Wrote tests/fixtures/reference-forces-64.json: max force … kJ/mol/A, energy … kJ/mol`.

- [ ] **Step 5: Commit**

```bash
git add scripts/reference_forces.py scripts/test_reference_forces.py tests/fixtures/reference-forces-64.json
git commit -m "Add reference reaction-field TIP4P-Ew forces validated against numerical gradients"
```

---

### Task 4: Initial state from the packed ice configuration

**Files:**
- Create: `src/initial-state.ts`
- Test: `tests/initial-state.test.ts`

**Interfaces:**
- Consumes: `bodySites`, `principalMoments`, `boxLength`, `BOLTZMANN`, `FORCE_TO_ACCELERATION`, `MASS_H`, `MASS_O`, `MOLECULE_MASS`, `OH_LENGTH` from `src/water-model.ts`.
- Produces:
  - `orientationOf(oxygen: Vector3, first: Vector3, second: Vector3): Quaternion` — body-to-space quaternion `[x, y, z, w]`.
  - `moleculeSites(centre: Vector3, quaternion: Quaternion, box: number, out: Float32Array, offset: number): void` — writes O, H1, H2 as three `vec4` (12 floats) with the oxygen wrapped into `[0, box)` and hydrogens whole relative to it.
  - `chargeSite(centre: Vector3, quaternion: Quaternion, oxygen: Vector3): Vector3` — the M site in space, positioned relative to the same wrapped oxygen.
  - `buildInitialState(packed: Float32Array, options: { molecules: number; densityRatio: number; temperature: number; seed: number }): InitialState` where `InitialState` is `{ molecules: number; box: number; state: Float32Array; sites: Float32Array; charges: Float32Array }`. `state` holds four `vec4` per molecule — centre of mass, quaternion, velocity, body-frame angular momentum — matching the GPU layout. `sites` holds 12 floats per molecule; `charges` holds 4 per molecule.
  - `mulberry32(seed: number): () => number`.

- [ ] **Step 1: Write the failing test**

Create `tests/initial-state.test.ts`:

```typescript
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { BOLTZMANN, FORCE_TO_ACCELERATION, MASS_H, MASS_O, MOLECULE_MASS, OH_LENGTH, HOH_ANGLE,
  boxLength, principalMoments } from '../src/water-model.ts';
import { buildInitialState, moleculeSites, orientationOf } from '../src/initial-state.ts';

const packed = (molecules: number) => {
  const bytes = readFileSync(`public/data/ice-${molecules}.bin`);
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
};
const build = (molecules: number, densityRatio = 1, temperature = 180, seed = 7) =>
  buildInitialState(packed(molecules), { molecules, densityRatio, temperature, seed });

test('orientation round-trips through the site builder', () => {
  const half = HOH_ANGLE / 2, along = OH_LENGTH * Math.cos(half), across = OH_LENGTH * Math.sin(half);
  // A molecule rotated 90 degrees about z: the bisector points along +y.
  const oxygen: [number, number, number] = [5, 5, 5];
  const first: [number, number, number] = [5 - across, 5 + along, 5];
  const second: [number, number, number] = [5 + across, 5 + along, 5];
  const quaternion = orientationOf(oxygen, first, second);
  const centre: [number, number, number] = [
    (MASS_O * oxygen[0] + MASS_H * (first[0] + second[0])) / MOLECULE_MASS,
    (MASS_O * oxygen[1] + MASS_H * (first[1] + second[1])) / MOLECULE_MASS,
    (MASS_O * oxygen[2] + MASS_H * (first[2] + second[2])) / MOLECULE_MASS];
  const out = new Float32Array(12);
  moleculeSites(centre, quaternion, 10, out, 0);
  for (const axis of [0, 1, 2]) assert.ok(Math.abs(out[axis] - oxygen[axis]) < 1e-5, `oxygen axis ${axis}`);
  const rebuilt = [[out[4], out[5], out[6]], [out[8], out[9], out[10]]];
  const matches = (a: number[], b: number[]) => a.every((value, axis) => Math.abs(value - b[axis]) < 1e-5);
  assert.ok(matches(rebuilt[0], first) || matches(rebuilt[0], second));
  assert.ok(matches(rebuilt[1], first) || matches(rebuilt[1], second));
});

test('every exported ice file loads into a rigid, wrapped initial state', () => {
  for (const molecules of [64, 216, 512]) {
    const initial = build(molecules);
    assert.equal(initial.molecules, molecules);
    assert.ok(Math.abs(initial.box - boxLength(molecules, 1)) < 1e-6);
    assert.equal(initial.state.length, molecules * 16);
    assert.equal(initial.sites.length, molecules * 12);
    assert.equal(initial.charges.length, molecules * 4);
    for (let molecule = 0; molecule < molecules; molecule++) {
      const base = molecule * 12;
      for (const axis of [0, 1, 2]) {
        assert.ok(initial.sites[base + axis] >= 0 && initial.sites[base + axis] < initial.box);
      }
      for (const hydrogen of [1, 2]) {
        const bond = [0, 1, 2].map(axis => initial.sites[base + hydrogen * 4 + axis] - initial.sites[base + axis]);
        assert.ok(Math.abs(Math.hypot(...bond) - OH_LENGTH) < 5e-3, `molecule ${molecule} bond`);
      }
      const quaternion = initial.state.subarray(molecule * 16 + 4, molecule * 16 + 8);
      assert.ok(Math.abs(Math.hypot(...quaternion) - 1) < 1e-5, 'unit quaternion');
    }
  }
});

test('velocities carry the requested temperature and no net momentum', () => {
  const temperature = 300;
  const initial = build(216, 1, temperature);
  const momentum = [0, 0, 0];
  let translational = 0, rotational = 0;
  const moments = principalMoments();
  for (let molecule = 0; molecule < initial.molecules; molecule++) {
    const velocity = initial.state.subarray(molecule * 16 + 8, molecule * 16 + 11);
    const angular = initial.state.subarray(molecule * 16 + 12, molecule * 16 + 15);
    for (const axis of [0, 1, 2]) {
      momentum[axis] += MOLECULE_MASS * velocity[axis];
      translational += MOLECULE_MASS * velocity[axis] ** 2;
      rotational += angular[axis] ** 2 / moments[axis];
    }
  }
  const degrees = 3 * initial.molecules;
  const toKelvin = (energy: number) => energy / FORCE_TO_ACCELERATION / (degrees * BOLTZMANN);
  assert.ok(Math.abs(toKelvin(translational) - temperature) < 1e-3, 'translational temperature');
  assert.ok(Math.abs(toKelvin(rotational) - temperature) < 1e-3, 'rotational temperature');
  for (const axis of [0, 1, 2]) assert.ok(Math.abs(momentum[axis]) < 1e-3, `momentum axis ${axis}`);
});

test('the same seed reproduces the state and a different seed does not', () => {
  assert.deepEqual(build(64, 1, 200, 11).state, build(64, 1, 200, 11).state);
  assert.notDeepEqual(build(64, 1, 200, 11).state, build(64, 1, 200, 12).state);
});

test('compressing the box scales oxygen positions but keeps molecules rigid', () => {
  const loose = build(64, 0.6), dense = build(64, 1.4);
  assert.ok(dense.box < loose.box);
  const nearest = (initial: { molecules: number; box: number; sites: Float32Array }) => {
    let best = Infinity;
    for (let a = 0; a < initial.molecules; a++) for (let b = a + 1; b < initial.molecules; b++) {
      const d = [0, 1, 2].map(axis => {
        const raw = initial.sites[b * 12 + axis] - initial.sites[a * 12 + axis];
        return raw - initial.box * Math.round(raw / initial.box);
      });
      best = Math.min(best, Math.hypot(...d));
    }
    return best;
  };
  const ratio = nearest(dense) / nearest(loose);
  assert.ok(Math.abs(ratio - dense.box / loose.box) < 0.02, `neighbour spacing scaled by ${ratio}`);
  const bond = [0, 1, 2].map(axis => dense.sites[4 + axis] - dense.sites[axis]);
  assert.ok(Math.abs(Math.hypot(...bond) - OH_LENGTH) < 5e-3);
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npm test`
Expected: FAIL — `Cannot find module '../src/initial-state.ts'`.

- [ ] **Step 3: Write the implementation**

Create `src/initial-state.ts`:

```typescript
import { BOLTZMANN, FORCE_TO_ACCELERATION, ICE_CELL, MASS_H, MASS_O, MOLECULE_MASS,
  bodySites, boxLength, cellsFor, principalMoments, type Vector3 } from './water-model';

export type Quaternion = [number, number, number, number];
export interface InitialState {
  molecules: number;
  box: number;
  /** Four vec4 per molecule: centre, quaternion, velocity, body angular momentum. */
  state: Float32Array;
  /** Three vec4 per molecule: O, H1, H2. */
  sites: Float32Array;
  /** One vec4 per molecule: the M charge site. */
  charges: Float32Array;
}

const cross = (a: Vector3, b: Vector3): Vector3 =>
  [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = (v: Vector3): Vector3 => {
  const norm = Math.hypot(...v);
  if (norm < 1e-10) throw new Error('Degenerate water orientation');
  return [v[0] / norm, v[1] / norm, v[2] / norm];
};

/** Deterministic 32-bit generator, so a seed reproduces a run exactly. */
export function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const gaussian = (random: () => number): number =>
  Math.sqrt(-2 * Math.log(Math.max(random(), 1e-12))) * Math.cos(2 * Math.PI * random());

/** Body-to-space quaternion: x along the bisector, y from the second hydrogen to the first. */
export function orientationOf(oxygen: Vector3, first: Vector3, second: Vector3): Quaternion {
  const toFirst = first.map((value, axis) => value - oxygen[axis]) as Vector3;
  const toSecond = second.map((value, axis) => value - oxygen[axis]) as Vector3;
  const x = unit(toFirst.map((value, axis) => value + toSecond[axis]) as Vector3);
  const plane = unit(toFirst.map((value, axis) => value - toSecond[axis]) as Vector3);
  const z = unit(cross(x, plane));
  const y = cross(z, x);
  const m = [x, y, z];                                  // rows are the body axes in space
  const trace = m[0][0] + m[1][1] + m[2][2];
  let q: Quaternion;
  if (trace > 0) {
    const s = 2 * Math.sqrt(trace + 1);
    q = [(m[1][2] - m[2][1]) / s, (m[2][0] - m[0][2]) / s, (m[0][1] - m[1][0]) / s, s / 4];
  } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
    const s = 2 * Math.sqrt(1 + m[0][0] - m[1][1] - m[2][2]);
    q = [s / 4, (m[1][0] + m[0][1]) / s, (m[2][0] + m[0][2]) / s, (m[1][2] - m[2][1]) / s];
  } else if (m[1][1] > m[2][2]) {
    const s = 2 * Math.sqrt(1 + m[1][1] - m[0][0] - m[2][2]);
    q = [(m[1][0] + m[0][1]) / s, s / 4, (m[2][1] + m[1][2]) / s, (m[2][0] - m[0][2]) / s];
  } else {
    const s = 2 * Math.sqrt(1 + m[2][2] - m[0][0] - m[1][1]);
    q = [(m[2][0] + m[0][2]) / s, (m[2][1] + m[1][2]) / s, s / 4, (m[0][1] - m[1][0]) / s];
  }
  const norm = Math.hypot(...q);
  return q.map(value => value / norm) as Quaternion;
}

const rotate = (q: Quaternion, v: Vector3): Vector3 => {
  const t = cross([q[0], q[1], q[2]], v).map(value => 2 * value) as Vector3;
  const c = cross([q[0], q[1], q[2]], t);
  return v.map((value, axis) => value + q[3] * t[axis] + c[axis]) as Vector3;
};

const BODY = bodySites();

/** Writes O, H1, H2 as vec4 with the oxygen wrapped and the hydrogens whole. */
export function moleculeSites(centre: Vector3, quaternion: Quaternion, box: number,
    out: Float32Array, offset: number): void {
  const oxygen = rotate(quaternion, BODY.oxygen).map((value, axis) => centre[axis] + value) as Vector3;
  const wrapped = oxygen.map(value => ((value % box) + box) % box) as Vector3;
  for (const axis of [0, 1, 2]) out[offset + axis] = wrapped[axis];
  out[offset + 3] = 1;
  for (const hydrogen of [0, 1]) {
    const relative = rotate(quaternion, BODY.hydrogen[hydrogen].map((value, axis) =>
      value - BODY.oxygen[axis]) as Vector3);
    for (const axis of [0, 1, 2]) out[offset + (hydrogen + 1) * 4 + axis] = wrapped[axis] + relative[axis];
    out[offset + (hydrogen + 1) * 4 + 3] = 1;
  }
}

/** The M site, placed relative to the already wrapped oxygen. */
export function chargeSite(centre: Vector3, quaternion: Quaternion, oxygen: Vector3): Vector3 {
  const relative = rotate(quaternion, BODY.charge.map((value, axis) => value - BODY.oxygen[axis]) as Vector3);
  return oxygen.map((value, axis) => value + relative[axis]) as Vector3;
}

export function buildInitialState(packed: Float32Array,
    options: { molecules: number; densityRatio: number; temperature: number; seed: number }): InitialState {
  const { molecules, densityRatio, temperature, seed } = options;
  if (packed.length !== molecules * 9) throw new Error('Ice configuration does not match the molecule count');
  const latticeBox = ICE_CELL * cellsFor(molecules);
  const box = boxLength(molecules, densityRatio);
  const scale = box / latticeBox;
  const random = mulberry32(seed);
  const moments = principalMoments();
  const state = new Float32Array(molecules * 16);
  const sites = new Float32Array(molecules * 12);
  const charges = new Float32Array(molecules * 4);
  const centres: Vector3[] = [];
  const quaternions: Quaternion[] = [];

  for (let molecule = 0; molecule < molecules; molecule++) {
    const read = (site: number): Vector3 =>
      [packed[molecule * 9 + site * 3], packed[molecule * 9 + site * 3 + 1], packed[molecule * 9 + site * 3 + 2]];
    const oxygen = read(0), first = read(1), second = read(2);
    const centre = [0, 1, 2].map(axis =>
      (MASS_O * oxygen[axis] + MASS_H * (first[axis] + second[axis])) / MOLECULE_MASS) as Vector3;
    // Only the centres move with density; the molecule itself stays rigid.
    centres.push(centre.map(value => value * scale) as Vector3);
    quaternions.push(orientationOf(oxygen, first, second));
  }

  const velocities: Vector3[] = [], angular: Vector3[] = [];
  for (let molecule = 0; molecule < molecules; molecule++) {
    velocities.push([gaussian(random), gaussian(random), gaussian(random)]);
    angular.push([gaussian(random), gaussian(random), gaussian(random)]);
  }
  // Remove the net momentum, then scale both sets to hit the requested temperature exactly.
  const drift = [0, 1, 2].map(axis =>
    velocities.reduce((sum, velocity) => sum + velocity[axis], 0) / molecules);
  for (const velocity of velocities) for (const axis of [0, 1, 2]) velocity[axis] -= drift[axis];
  const degrees = 3 * molecules;
  const target = degrees * BOLTZMANN * temperature * FORCE_TO_ACCELERATION;
  const translational = velocities.reduce((sum, velocity) =>
    sum + MOLECULE_MASS * velocity.reduce((inner, value) => inner + value * value, 0), 0);
  const rotational = angular.reduce((sum, momentum) =>
    sum + momentum.reduce((inner, value, axis) => inner + value * value / moments[axis], 0), 0);
  const translationScale = Math.sqrt(target / translational);
  // Angular momentum carries the moment of inertia, so scale it in the same measure.
  const rotationScale = Math.sqrt(target / rotational);

  for (let molecule = 0; molecule < molecules; molecule++) {
    const base = molecule * 16;
    for (const axis of [0, 1, 2]) {
      state[base + axis] = centres[molecule][axis];
      state[base + 8 + axis] = velocities[molecule][axis] * translationScale;
      state[base + 12 + axis] = angular[molecule][axis] * rotationScale;
    }
    state[base + 3] = 1;
    for (let component = 0; component < 4; component++) state[base + 4 + component] = quaternions[molecule][component];
    moleculeSites(centres[molecule], quaternions[molecule], box, sites, molecule * 12);
    const oxygen: Vector3 = [sites[molecule * 12], sites[molecule * 12 + 1], sites[molecule * 12 + 2]];
    const charge = chargeSite(centres[molecule], quaternions[molecule], oxygen);
    for (const axis of [0, 1, 2]) charges[molecule * 4 + axis] = charge[axis];
    charges[molecule * 4 + 3] = 1;
  }
  return { molecules, box, state, sites, charges };
}
```

Note on the angular momentum scaling: each component of `L` is drawn from the same unit normal and then scaled by one common factor, so the per-axis variance is not `I_k kB T`. That is deliberate — the thermostat's O step imposes the correct per-axis distribution within the first picosecond, and the test only requires the total rotational temperature. Add that as a comment where `rotationScale` is computed.

- [ ] **Step 4: Run the test to verify it passes**

Run: `npm test`
Expected: PASS — five new tests.

Run: `npx tsc --noEmit`
Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add src/initial-state.ts tests/initial-state.test.ts
git commit -m "Build rigid-body initial states from the exported ice configurations"
```

---

### Task 5: Force kernel and the simulation object

**Files:**
- Create: `src/simulation.wgsl`
- Create: `src/simulation.ts`
- Create: `tests/wgsl-constants.test.ts`
- Create: `tests/browser/simulation.spec.ts`
- Modify: `src/background.ts` (accept a shared `gpu` and raw sites)
- Modify: `src/main.ts` (minimal wiring so the page still loads; the full panel comes in Task 8)

**Interfaces:**
- Consumes: everything produced by Tasks 1 and 4, and `tests/fixtures/reference-forces-64.json` from Task 3.
- Produces:
  - `createSimulation(gpu: Gpu, options: { molecules: number; densityRatio: number; temperature: number; ice: Record<number, Float32Array>; seed?: number }): Simulation`.
  - `interface Simulation { molecules: number; box: number; cutoff: number; timePs: number; readSites(): Promise<Float32Array>; setTemperature(kelvin: number): void; setDensity(ratio: number): void; setMolecules(count: number): void; reset(): void; evaluateForces(packed: Float32Array): Promise<{ forces: Float32Array; torques: Float32Array }>; dispose(): void }`. Task 6 adds `step(count: number)` and Task 7 adds `readStats()`.
  - `interface SimulationStats { translationalTemperature: number; rotationalTemperature: number; maximumForce: number; nonFinite: boolean }` — declared here, filled in by Task 7.
  - `createBackground(gpu: Gpu, canvas: HTMLCanvasElement, molecules: number, onFailure: () => void)` returning `{ render(sites: Float32Array, box: number): void; dispose(): void }`.
  - `window.waterSimulation` — the live `Simulation` object itself (`box` and `molecules` are getters on it). Step rate is measured by the panel and by the tests, so it is not a field here.

- [ ] **Step 1: Write the failing tests**

Create `tests/wgsl-constants.test.ts` — a guard so the shader constants can never drift from the TypeScript model:

```typescript
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { BOLTZMANN, COULOMB, EPSILON_O, FORCE_TO_ACCELERATION, MOLECULE_MASS, Q_H, Q_M, SIGMA_O,
  bodySites, principalMoments } from '../src/water-model.ts';

const shader = readFileSync('src/simulation.wgsl', 'utf8');
const constant = (name: string): number => {
  const match = shader.match(new RegExp(`const ${name}\\s*(?::\\s*f32)?\\s*=\\s*(-?[0-9.eE+-]+)`));
  assert.ok(match, `${name} missing from simulation.wgsl`);
  return Number(match![1]);
};
const vector = (name: string): number[] => {
  const match = shader.match(new RegExp(`const ${name}\\s*(?::\\s*vec3f)?\\s*=\\s*vec3f\\(([^)]*)\\)`));
  assert.ok(match, `${name} missing from simulation.wgsl`);
  return match![1].split(',').map(part => Number(part.trim()));
};

test('shader constants match the TypeScript water model', () => {
  const close = (actual: number, expected: number, label: string) =>
    assert.ok(Math.abs(actual - expected) < 1e-6, `${label}: ${actual} vs ${expected}`);
  close(constant('COULOMB'), COULOMB, 'COULOMB');
  close(constant('BOLTZMANN'), BOLTZMANN, 'BOLTZMANN');
  close(constant('FORCE_TO_ACCELERATION'), FORCE_TO_ACCELERATION, 'FORCE_TO_ACCELERATION');
  close(constant('SIGMA_O'), SIGMA_O, 'SIGMA_O');
  close(constant('EPSILON_O'), EPSILON_O, 'EPSILON_O');
  close(constant('Q_H'), Q_H, 'Q_H');
  close(constant('Q_M'), Q_M, 'Q_M');
  close(constant('MOLECULE_MASS'), MOLECULE_MASS, 'MOLECULE_MASS');
  const moments = principalMoments();
  vector('INERTIA').forEach((value, axis) => close(value, moments[axis], `INERTIA ${axis}`));
  const body = bodySites();
  vector('BODY_OXYGEN').forEach((value, axis) => close(value, body.oxygen[axis], `BODY_OXYGEN ${axis}`));
  vector('BODY_HYDROGEN_A').forEach((value, axis) => close(value, body.hydrogen[0][axis], `BODY_HYDROGEN_A ${axis}`));
  vector('BODY_HYDROGEN_B').forEach((value, axis) => close(value, body.hydrogen[1][axis], `BODY_HYDROGEN_B ${axis}`));
  vector('BODY_CHARGE').forEach((value, axis) => close(value, body.charge[axis], `BODY_CHARGE ${axis}`));
});
```

Create `tests/browser/simulation.spec.ts` with the force comparison only (later tasks append to this file):

```typescript
import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { BOLTZMANN, FORCE_TO_ACCELERATION, MOLECULE_MASS, OH_LENGTH, principalMoments } from '../../src/water-model';

interface Fixture {
  box: number; cutoff: number; molecules: number;
  sites: number[]; forces: number[]; torques: number[]; potentialEnergy: number;
}

async function ready(page: import('@playwright/test').Page) {
  await page.goto('/');
  await expect(page.locator('.molecular-stage')).toHaveClass(/ready/, { timeout: 20000 });
  await page.waitForFunction(() => Boolean((window as any).waterSimulation));
}

test('GPU forces and torques match the Python reference within f32 precision', async ({ page }) => {
  const fixture: Fixture = JSON.parse(await readFile('tests/fixtures/reference-forces-64.json', 'utf8'));
  await ready(page);
  const result = await page.evaluate(async data => {
    const simulation = (window as any).waterSimulation;
    simulation.setMolecules(data.molecules);
    simulation.setDensity(1);
    const { forces, torques } = await simulation.evaluateForces(new Float32Array(data.sites));
    return { forces: Array.from(forces as Float32Array), torques: Array.from(torques as Float32Array) };
  }, fixture);
  const error = (actual: number[], expected: number[]) => {
    const scale = Math.max(...expected.map(Math.abs));
    const worst = Math.max(...expected.map((value, index) => Math.abs(value - actual[index])));
    return worst / scale;
  };
  expect(result.forces.length).toBe(fixture.molecules * 3);
  expect(error(result.forces, fixture.forces)).toBeLessThan(1e-3);
  expect(error(result.torques, fixture.torques)).toBeLessThan(1e-3);
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npm test`
Expected: FAIL — `src/simulation.wgsl` does not exist.

Run: `npm run build && npx playwright test tests/browser/simulation.spec.ts`
Expected: FAIL — `window.waterSimulation` never appears.

- [ ] **Step 3: Write the shader's constants, helpers and force kernel**

Create `src/simulation.wgsl`. This step adds everything except the `integrate` and `reduce` entry points, which Tasks 6 and 7 append.

```wgsl
// Rigid TIP4P-Ew water with a reaction-field cutoff. Angstrom, kJ/mol, amu, ps, e.
// Constants are mirrored from src/water-model.ts and guarded by tests/wgsl-constants.test.ts.
const COULOMB = 1389.35456;
const BOLTZMANN = 0.0083144626;
const FORCE_TO_ACCELERATION = 100.0;
const SIGMA_O = 3.16435;
const EPSILON_O = 0.680946;
const Q_H = 0.52422;
const Q_M = -1.04844;
const MOLECULE_MASS = 18.015324;
const INERTIA = vec3f(1.1550544, 0.6145410, 1.7695954);
const BODY_OXYGEN = vec3f(-0.0655596, 0.0, 0.0);
const BODY_HYDROGEN_A = vec3f(0.5203227, 0.7569503, 0.0);
const BODY_HYDROGEN_B = vec3f(0.5203227, -0.7569503, 0.0);
const BODY_CHARGE = vec3f(0.0594404, 0.0, 0.0);

struct Params {
  box: f32,
  cutoff: f32,
  dt: f32,
  temperature: f32,
  friction: f32,
  molecules: u32,
  step: u32,
  seed: u32,
  /** Factor applied to every centre of mass by the rescale kernel. */
  scale: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Four vec4 per molecule: centre, quaternion, velocity, body angular momentum.
@group(0) @binding(1) var<storage, read_write> state: array<vec4f>;
// Three vec4 per molecule: O, H1, H2. Oxygen is wrapped; hydrogens stay whole.
@group(0) @binding(2) var<storage, read_write> sites: array<vec4f>;
// One vec4 per molecule: the M charge site, placed relative to the wrapped oxygen.
@group(0) @binding(3) var<storage, read_write> charges: array<vec4f>;
// Two vec4 per molecule: net force, then net torque about the centre of mass.
@group(0) @binding(4) var<storage, read_write> forceTorque: array<vec4f>;
// translational temperature, rotational temperature, maximum force, non-finite flag.
@group(0) @binding(5) var<storage, read_write> stats: array<f32>;

fn quatRotate(q: vec4f, v: vec3f) -> vec3f {
  let t = 2.0 * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
}

fn quatInverseRotate(q: vec4f, v: vec3f) -> vec3f {
  return quatRotate(vec4f(-q.xyz, q.w), v);
}

/** Charge site a of molecule i: 0 and 1 are hydrogens, 2 is the M site. */
fn chargeSiteOf(molecule: u32, index: u32) -> vec3f {
  if (index == 2u) { return charges[molecule].xyz; }
  return sites[molecule * 3u + index + 1u].xyz;
}

fn chargeOf(index: u32) -> f32 {
  if (index == 2u) { return Q_M; }
  return Q_H;
}

/** Body-frame position of charge site a, used as its torque lever arm. */
fn bodyArmOf(index: u32) -> vec3f {
  if (index == 0u) { return BODY_HYDROGEN_A; }
  if (index == 1u) { return BODY_HYDROGEN_B; }
  return BODY_CHARGE;
}

/** Force-shifted Lennard-Jones magnitude; positive is repulsive. */
fn lennardJones(r: f32, cutoff: f32) -> f32 {
  let s6 = pow(SIGMA_O / r, 6.0);
  let edge = pow(SIGMA_O / cutoff, 6.0);
  let force = 24.0 * EPSILON_O * (2.0 * s6 * s6 - s6) / r;
  let edgeForce = 24.0 * EPSILON_O * (2.0 * edge * edge - edge) / cutoff;
  return force - edgeForce;
}

/** Reaction field with conducting boundary; magnitude vanishes at the cutoff. */
fn reactionField(r: f32, product: f32, cutoff: f32) -> f32 {
  return COULOMB * product * (1.0 / (r * r) - r / (cutoff * cutoff * cutoff));
}

@compute @workgroup_size(64)
fn forces(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= params.molecules) { return; }
  let q = state[i * 4u + 1u];
  let oxygen = sites[i * 3u].xyz;
  let cutoff = params.cutoff;
  var force = vec3f(0.0);
  var torque = vec3f(0.0);
  let oxygenArm = quatRotate(q, BODY_OXYGEN);
  for (var j = 0u; j < params.molecules; j++) {
    if (j == i) { continue; }
    let raw = sites[j * 3u].xyz - oxygen;
    let shift = -params.box * round(raw / params.box);
    let delta = raw + shift;
    let r2 = dot(delta, delta);
    if (r2 >= cutoff * cutoff) { continue; }
    let r = sqrt(r2);
    // Lennard-Jones acts between the oxygens only.
    let pull = -lennardJones(r, cutoff) * delta / r;
    force += pull;
    // Lever arms come from the orientation, not from the site coordinates: the sites
    // are placed relative to the wrapped oxygen, so `site - centre` is off by a box
    // vector whenever the two sit in different periodic images.
    torque += cross(oxygenArm, pull);
    for (var a = 0u; a < 3u; a++) {
      let here = chargeSiteOf(i, a);
      let arm = quatRotate(q, bodyArmOf(a));
      let qa = chargeOf(a);
      for (var b = 0u; b < 3u; b++) {
        let there = chargeSiteOf(j, b) + shift;
        let separation = there - here;
        let distance = length(separation);
        let magnitude = reactionField(distance, qa * chargeOf(b), cutoff);
        let contribution = -magnitude * separation / distance;
        force += contribution;
        torque += cross(arm, contribution);
      }
    }
  }
  forceTorque[i * 2u] = vec4f(force, 0.0);
  forceTorque[i * 2u + 1u] = vec4f(torque, 0.0);
}

/** Rebuilds the drawable sites and the M site from the rigid-body state. */
fn writeSites(i: u32) {
  let centre = state[i * 4u].xyz;
  let q = state[i * 4u + 1u];
  let oxygen = centre + quatRotate(q, BODY_OXYGEN);
  let wrapped = oxygen - params.box * floor(oxygen / params.box);
  sites[i * 3u] = vec4f(wrapped, 1.0);
  sites[i * 3u + 1u] = vec4f(wrapped + quatRotate(q, BODY_HYDROGEN_A - BODY_OXYGEN), 1.0);
  sites[i * 3u + 2u] = vec4f(wrapped + quatRotate(q, BODY_HYDROGEN_B - BODY_OXYGEN), 1.0);
  charges[i] = vec4f(wrapped + quatRotate(q, BODY_CHARGE - BODY_OXYGEN), 1.0);
}

/** Density changes move centres of mass; molecules stay rigid. */
@compute @workgroup_size(64)
fn rescale(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= params.molecules) { return; }
  state[i * 4u] = vec4f(state[i * 4u].xyz * params.scale, 1.0);
  writeSites(i);
}
```

- [ ] **Step 4: Write the simulation object**

Create `src/simulation.ts`:

```typescript
import { compute, storage, uniforms, type Gpu } from 'vgpu';
import shader from './simulation.wgsl?raw';
import { MASS_H, MASS_O, MOLECULE_MASS, boxLength, cutoffFor, type Vector3 } from './water-model';
import { buildInitialState, chargeSite, moleculeSites, orientationOf,
  type InitialState, type Quaternion } from './initial-state';

export interface SimulationStats {
  translationalTemperature: number;
  rotationalTemperature: number;
  maximumForce: number;
  nonFinite: boolean;
}
export interface SimulationOptions {
  molecules: number;
  densityRatio: number;
  temperature: number;
  /** Packed ice configurations by molecule count, from public/data/ice-<N>.bin. */
  ice: Record<number, Float32Array>;
  seed?: number;
}

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
  }

  /** Uploads a fresh ice configuration and the matching uniforms. */
  function load(initial?: InitialState) {
    const state = initial ?? buildInitialState(options.ice[molecules],
      { molecules, densityRatio, temperature, seed });
    simulation = { box: state.box, cutoff: cutoffFor(state.box) };
    currentBox = state.box;
    buffers.state.write(state.state);
    buffers.sites.write(state.sites);
    buffers.charges.write(state.charges);
    statsBuffer.write(new Float32Array([temperature, temperature, 0, 0]));
    params.set({ box: simulation.box, cutoff: simulation.cutoff, molecules, temperature,
      step: stepCount, seed, scale: 1 });
    bind();
  }

  const groups = () => Math.ceil(molecules / WORKGROUP);

  async function readSites(): Promise<Float32Array> {
    return new Float32Array(await buffers.sites.read());
  }

  /** Test path: upload a packed configuration and evaluate forces once, without integrating. */
  async function evaluateForces(packed: Float32Array) {
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
    return { forces, torques };
  }

  const api = {
    get molecules() { return molecules; },
    get box() { return simulation.box; },
    get cutoff() { return simulation.cutoff; },
    get timePs() { return stepCount * TIME_STEP; },
    readSites,
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
```

- [ ] **Step 5: Wire the simulation into the page**

Modify `src/background.ts`: replace the `data: Trajectory` parameter with `molecules: number`, drop the `sampleFrame` import and call, and change `render` to `render(sites: Float32Array, box: number)`, passing `box` into `scene.update(sites, box)` (Task 8 changes `scene.ts` to accept it; until then keep `createScene(molecules, box)` and ignore the second argument).

Modify `src/main.ts` `initialize()` to fetch the three ice files, create one shared `gpu`, and start a loop:

```typescript
import { init } from 'vgpu';
import { createSimulation } from './simulation';
import { MOLECULE_COUNTS } from './water-model';

const ice: Record<number, Float32Array> = {};
for (const count of MOLECULE_COUNTS) {
  const response = await fetch(`${import.meta.env.BASE_URL}data/ice-${count}.bin`, { signal });
  if (!response.ok) throw new Error(`Missing ice configuration for ${count} molecules`);
  ice[count] = new Float32Array(await response.arrayBuffer());
}
const gpu = await init({ powerPreference: 'low-power' });
const simulation = createSimulation(gpu, { molecules: 216, densityRatio: 1, temperature: 180, ice });
renderer = await createBackground(gpu, get<HTMLCanvasElement>('molecule-canvas'), simulation.molecules, fallback);
(window as unknown as { waterSimulation: unknown }).waterSimulation = simulation;
stage.classList.add('ready');
```

then draw the initial configuration once, since nothing steps yet:

```typescript
renderer.render(await simulation.readSites(), simulation.box);
```

Leave the existing playback controls disabled for now: `play.disabled = replay.disabled = timeline.disabled = rate.disabled = true`. Task 6 adds stepping and Task 8 replaces the panel. This step only has to make the page load, draw the ice lattice and expose `window.waterSimulation`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `npm test`
Expected: PASS — including `shader constants match the TypeScript water model`.

Run: `npm run build && npx playwright test tests/browser/simulation.spec.ts`
Expected: PASS — `GPU forces and torques match the Python reference within f32 precision`.

If the force test fails, compare a single pair first: run `evaluateForces` on the first two molecules only by slicing the fixture, and print both force vectors. The usual causes are a sign error on `pull`, a missing molecular `shift` on the charge sites of `j`, or a torque taken about the oxygen instead of the centre of mass.

- [ ] **Step 7: Commit**

```bash
git add src/simulation.wgsl src/simulation.ts src/background.ts src/main.ts tests/wgsl-constants.test.ts tests/browser/simulation.spec.ts
git commit -m "Add the GPU force kernel and validate it against the Python reference"
```

---

### Task 6: BAOAB rigid-body integrator

**Files:**
- Modify: `src/simulation.wgsl` (append the RNG helpers, free-rotation helper and the `integrate` entry point)
- Modify: `src/simulation.ts` (add `step`, `readState`, `timePs`)
- Modify: `tests/browser/simulation.spec.ts` (append three tests)

**Interfaces:**
- Consumes: the shader constants, `writeSites`, `quatRotate`, `quatInverseRotate` and the buffers from Task 5.
- Produces: `step(count: number): void` advancing the simulation by `count` steps of 2 fs; `readState(): Promise<Float32Array>` returning 16 floats per molecule (centre, quaternion, velocity, body angular momentum); `timePs` reflecting the accumulated steps.

- [ ] **Step 1: Write the failing tests**

Append to `tests/browser/simulation.spec.ts`:

```typescript
/** Kinetic temperatures computed from the raw state, the way the reduction kernel will. */
function temperatures(state: number[], molecules: number) {
  const inertia = principalMoments();
  let translational = 0, rotational = 0;
  const momentum = [0, 0, 0];
  for (let molecule = 0; molecule < molecules; molecule++) {
    for (const axis of [0, 1, 2]) {
      const velocity = state[molecule * 16 + 8 + axis];
      const angular = state[molecule * 16 + 12 + axis];
      translational += MOLECULE_MASS * velocity * velocity;
      rotational += angular * angular / inertia[axis];
      momentum[axis] += MOLECULE_MASS * velocity;
    }
  }
  const degrees = 3 * molecules * BOLTZMANN * FORCE_TO_ACCELERATION;
  return { translational: translational / degrees, rotational: rotational / degrees, momentum };
}

test('molecules stay rigid after five thousand steps', async ({ page }) => {
  test.setTimeout(120000);
  await ready(page);
  const result = await page.evaluate(async LENGTH => {
    const simulation = (window as any).waterSimulation;
    simulation.setMolecules(64);
    simulation.setTemperature(300);
    for (let batch = 0; batch < 50; batch++) simulation.step(100);
    const sites = await simulation.readSites();
    let worstBond = 0, worstAngle = 0;
    for (let molecule = 0; molecule < simulation.molecules; molecule++) {
      const site = (index: number) => [0, 1, 2].map(axis => sites[molecule * 12 + index * 4 + axis]);
      const oxygen = site(0);
      const bonds = [1, 2].map(index => site(index).map((value, axis) => value - oxygen[axis]));
      for (const bond of bonds) worstBond = Math.max(worstBond, Math.abs(Math.hypot(...bond) - LENGTH));
      const cosine = bonds[0].reduce((sum, value, axis) => sum + value * bonds[1][axis], 0) / LENGTH ** 2;
      worstAngle = Math.max(worstAngle, Math.abs(Math.acos(Math.max(-1, Math.min(1, cosine))) * 180 / Math.PI - 104.52));
    }
    return { worstBond, worstAngle, timePs: simulation.timePs };
  }, OH_LENGTH);
  expect(result.timePs).toBeCloseTo(10, 6);
  expect(result.worstBond).toBeLessThan(0.001);
  expect(result.worstAngle).toBeLessThan(0.05);
});

test('the thermostat brings the sample to its set point and holds it', async ({ page }) => {
  test.setTimeout(180000);
  await ready(page);
  const samples = await page.evaluate(async () => {
    const simulation = (window as any).waterSimulation;
    simulation.setMolecules(216);
    simulation.setTemperature(300);
    // Two picoseconds of equilibration at 5 /ps friction, then five samples over 2.5 ps.
    for (let batch = 0; batch < 10; batch++) simulation.step(100);
    const collected: number[][] = [];
    for (let sample = 0; sample < 5; sample++) {
      simulation.step(250);
      collected.push(Array.from(await simulation.readState() as Float32Array));
    }
    return { collected, molecules: simulation.molecules };
  });
  const measured = samples.collected.map(state => temperatures(state, samples.molecules));
  const mean = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / values.length;
  expect(mean(measured.map(entry => entry.translational))).toBeGreaterThan(275);
  expect(mean(measured.map(entry => entry.translational))).toBeLessThan(325);
  expect(mean(measured.map(entry => entry.rotational))).toBeGreaterThan(275);
  expect(mean(measured.map(entry => entry.rotational))).toBeLessThan(325);
});

test('the sample never acquires a net drift', async ({ page }) => {
  await ready(page);
  const drift = await page.evaluate(async () => {
    const simulation = (window as any).waterSimulation;
    simulation.setMolecules(64);
    simulation.setTemperature(450);
    for (let batch = 0; batch < 20; batch++) simulation.step(100);
    const state = Array.from(await simulation.readState() as Float32Array);
    return { state, molecules: simulation.molecules };
  });
  const { momentum, translational } = temperatures(drift.state, drift.molecules);
  // A Langevin thermostat does not conserve momentum exactly; it must stay small next
  // to the thermal momentum of a single molecule, sqrt(m kB T / 100) in amu A/ps.
  const thermal = Math.sqrt(MOLECULE_MASS * BOLTZMANN * translational * FORCE_TO_ACCELERATION);
  for (const axis of [0, 1, 2]) {
    expect(Math.abs(momentum[axis])).toBeLessThan(3 * thermal * Math.sqrt(drift.molecules));
  }
  expect(Number.isFinite(translational)).toBe(true);
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npm run build && npx playwright test tests/browser/simulation.spec.ts`
Expected: FAIL — `simulation.step is not a function`.

- [ ] **Step 3: Append the integrator to the shader**

Add to `src/simulation.wgsl`:

```wgsl
struct Rotation {
  q: vec4f,
  l: vec3f,
}

/** PCG hash: a stateless stream indexed by molecule, step and lane. */
fn hash(value: u32) -> u32 {
  let state = value * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn uniform01(value: u32) -> f32 {
  return f32(hash(value)) * 2.3283064365386963e-10;
}

fn gaussianPair(value: u32) -> vec2f {
  let first = max(uniform01(value), 1e-7);
  let second = uniform01(value ^ 0x9e3779b9u);
  let radius = sqrt(-2.0 * log(first));
  let angle = 6.283185307179586 * second;
  return vec2f(radius * cos(angle), radius * sin(angle));
}

fn gaussian3(value: u32) -> vec3f {
  let first = gaussianPair(value);
  let second = gaussianPair(value * 2654435761u + 12345u);
  return vec3f(first.x, first.y, second.x);
}

/** dq/dt for a body-frame angular velocity. */
fn quatDerivative(q: vec4f, w: vec3f) -> vec4f {
  return 0.5 * vec4f(q.w * w + cross(q.xyz, w), -dot(q.xyz, w));
}

/**
 * Torque-free rotation. The sub-steps bound the first-order error, and the magnitude of
 * the body-frame angular momentum is restored afterwards because torque-free motion
 * conserves it exactly: explicit Euler on Euler's equations otherwise grows it by about
 * 1e-5 per half-step, which is a hidden heat source once the friction is lowered.
 */
fn freeRotation(start: vec4f, momentum: vec3f, duration: f32) -> Rotation {
  var q = start;
  var l = momentum;
  let magnitude = length(momentum);
  let h = duration / 4.0;
  for (var sub = 0u; sub < 4u; sub++) {
    let w = l / INERTIA;
    l -= h * cross(w, l);
    q = normalize(q + h * quatDerivative(q, w));
  }
  if (magnitude > 0.0) { l = magnitude * normalize(l); }
  return Rotation(q, l);
}

/** One BAOAB step. The trailing half kick is the leading one of the next step. */
@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= params.molecules) { return; }
  var centre = state[i * 4u].xyz;
  var q = state[i * 4u + 1u];
  var velocity = state[i * 4u + 2u].xyz;
  var angular = state[i * 4u + 3u].xyz;
  let dt = params.dt;
  let force = forceTorque[i * 2u].xyz;
  let torque = quatInverseRotate(q, forceTorque[i * 2u + 1u].xyz);

  velocity += 0.5 * dt * FORCE_TO_ACCELERATION * force / MOLECULE_MASS;
  angular += 0.5 * dt * FORCE_TO_ACCELERATION * torque;

  centre += 0.5 * dt * velocity;
  var rotation = freeRotation(q, angular, 0.5 * dt);
  q = rotation.q;
  angular = rotation.l;

  let decay = exp(-params.friction * dt);
  let spread = sqrt(1.0 - decay * decay);
  let energy = BOLTZMANN * params.temperature * FORCE_TO_ACCELERATION;
  // Mixed sequentially so distinct (molecule, step) pairs never share a stream.
  let stream = hash(params.seed ^ hash(i ^ hash(params.step)));
  velocity = decay * velocity + spread * sqrt(energy / MOLECULE_MASS) * gaussian3(stream);
  angular = decay * angular + spread * sqrt(energy * INERTIA) * gaussian3(stream ^ 0x5bf03635u);

  centre += 0.5 * dt * velocity;
  rotation = freeRotation(q, angular, 0.5 * dt);
  q = rotation.q;
  angular = rotation.l;

  // Wrapping the centre keeps f32 coordinates small over a long session.
  centre -= params.box * floor(centre / params.box);
  state[i * 4u] = vec4f(centre, 1.0);
  state[i * 4u + 1u] = q;
  state[i * 4u + 2u] = vec4f(velocity, 0.0);
  state[i * 4u + 3u] = vec4f(angular, 0.0);
  writeSites(i);
}
```

- [ ] **Step 4: Add stepping to the simulation object**

In `src/simulation.ts`, create the kernel next to the others, bind it, and add the two methods:

```typescript
  const integrateKernel = compute(gpu, shader, { entry: 'integrate' });
```

Add `integrateKernel.set(bag);` inside `bind()`, then:

```typescript
  function step(count: number) {
    for (let index = 0; index < count; index++) {
      params.set({ step: stepCount });
      forceKernel.dispatch(groups());
      integrateKernel.dispatch(groups());
      stepCount++;
    }
  }

  async function readState(): Promise<Float32Array> {
    return new Float32Array(await buffers.state.read());
  }
```

Expose both on the returned object (`step,` and `readState,` next to `readSites`).

- [ ] **Step 5: Run the tests to verify they pass**

Run: `npm test`
Expected: PASS (the shader constant guard still holds).

Run: `npm run build && npx playwright test tests/browser/simulation.spec.ts`
Expected: PASS — four tests.

If the thermostat test lands far above the set point, the usual cause is the noise amplitude: `spread * sqrt(energy / MOLECULE_MASS)` must multiply a unit normal, so `gaussian3` has to return variance one — check it by averaging `uniform01` over many seeds in a scratch test. If the sample heats up steadily instead, the drift is a sign error in `freeRotation`'s `l -= h * cross(w, l)`.

- [ ] **Step 6: Commit**

```bash
git add src/simulation.wgsl src/simulation.ts tests/browser/simulation.spec.ts
git commit -m "Add the BAOAB rigid-body integrator with a Langevin thermostat"
```

---

### Task 7: Statistics reduction and the blow-up guard

**Files:**
- Modify: `src/simulation.wgsl` (append the `reduce` entry point)
- Create: `src/simulation-health.ts` (`SimulationStats`, `FORCE_LIMIT`, `needsRestart`)
- Modify: `src/simulation.ts` (add `readStats`, re-export the health module)
- Modify: `tests/browser/simulation.spec.ts` (append one test)
- Create: `tests/simulation-guard.test.ts`

**Interfaces:**
- Consumes: the state buffer and constants from Tasks 5 and 6.
- Produces: `readStats(): Promise<SimulationStats>` on the simulation object; `SimulationStats`, `FORCE_LIMIT = 5e4` kJ/mol/Å and `needsRestart(stats): boolean` (true when `nonFinite` is set, `maximumForce` is not finite, or it exceeds `FORCE_LIMIT`) from `src/simulation-health.ts`, re-exported by `src/simulation.ts` so callers can import either.

  The guard lives in its own module because `src/simulation.ts` imports the shader as `./simulation.wgsl?raw`, which is Vite-only syntax that node's loader rejects — a node unit test cannot import that file at all. `src/simulation-health.ts` has no imports, so `tests/simulation-guard.test.ts` can exercise the predicate without a GPU.

- [ ] **Step 1: Write the failing tests**

Create `tests/simulation-guard.test.ts`:

```typescript
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { FORCE_LIMIT, needsRestart } from '../src/simulation-health.ts';

const stats = (overrides: Partial<Parameters<typeof needsRestart>[0]> = {}) => ({
  translationalTemperature: 300, rotationalTemperature: 300, maximumForce: 900, nonFinite: false, ...overrides,
});

test('a healthy sample keeps running', () => {
  assert.equal(needsRestart(stats()), false);
  assert.equal(needsRestart(stats({ maximumForce: FORCE_LIMIT - 1 })), false);
});

test('non-finite state or a runaway force forces a restart', () => {
  assert.equal(needsRestart(stats({ nonFinite: true })), true);
  assert.equal(needsRestart(stats({ maximumForce: FORCE_LIMIT + 1 })), true);
  assert.equal(needsRestart(stats({ maximumForce: Number.NaN })), true);
});
```

Append to `tests/browser/simulation.spec.ts`:

```typescript
test('the reduction kernel reports the same temperatures as the raw state', async ({ page }) => {
  await ready(page);
  const result = await page.evaluate(async () => {
    const simulation = (window as any).waterSimulation;
    simulation.setMolecules(216);
    simulation.setTemperature(320);
    for (let batch = 0; batch < 10; batch++) simulation.step(100);
    const stats = await simulation.readStats();
    const state = Array.from(await simulation.readState() as Float32Array);
    return { stats, state, molecules: simulation.molecules };
  });
  const expected = temperatures(result.state, result.molecules);
  expect(result.stats.translationalTemperature).toBeGreaterThan(expected.translational - 1);
  expect(result.stats.translationalTemperature).toBeLessThan(expected.translational + 1);
  expect(result.stats.rotationalTemperature).toBeGreaterThan(expected.rotational - 1);
  expect(result.stats.rotationalTemperature).toBeLessThan(expected.rotational + 1);
  expect(result.stats.nonFinite).toBe(false);
  expect(result.stats.maximumForce).toBeGreaterThan(0);
  expect(result.stats.maximumForce).toBeLessThan(50000);
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npm test`
Expected: FAIL — `needsRestart` is not exported from `src/simulation.ts`.

- [ ] **Step 3: Append the reduction to the shader**

Add to `src/simulation.wgsl`:

```wgsl
/** True for NaN and for infinity: an f32 exponent field of all ones. */
fn isNonFinite(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u;
}

const REDUCTION_LANES = 256u;
var<workgroup> laneTranslational: array<f32, REDUCTION_LANES>;
var<workgroup> laneRotational: array<f32, REDUCTION_LANES>;
var<workgroup> laneForce: array<f32, REDUCTION_LANES>;
var<workgroup> laneBroken: array<f32, REDUCTION_LANES>;

/** One workgroup reduces the whole sample; dispatch it with a single group. */
@compute @workgroup_size(256)
fn reduce(@builtin(local_invocation_id) local: vec3u) {
  let lane = local.x;
  var translational = 0.0;
  var rotational = 0.0;
  var peak = 0.0;
  var broken = 0.0;
  for (var i = lane; i < params.molecules; i += REDUCTION_LANES) {
    let velocity = state[i * 4u + 2u].xyz;
    let angular = state[i * 4u + 3u].xyz;
    let force = forceTorque[i * 2u].xyz;
    translational += MOLECULE_MASS * dot(velocity, velocity);
    rotational += dot(angular * angular / INERTIA, vec3f(1.0));
    peak = max(peak, length(force));
    // Bit inspection, not `x != x` or a magnitude test: some WebGPU backends compile
    // with fast-math semantics where NaN comparisons are false and max() discards a NaN
    // operand, which leaves a comparison-based guard dead on real hardware.
    if (isNonFinite(velocity.x) || isNonFinite(velocity.y) || isNonFinite(velocity.z) ||
        isNonFinite(angular.x) || isNonFinite(angular.y) || isNonFinite(angular.z) ||
        isNonFinite(force.x) || isNonFinite(force.y) || isNonFinite(force.z)) { broken = 1.0; }
  }
  laneTranslational[lane] = translational;
  laneRotational[lane] = rotational;
  laneForce[lane] = peak;
  laneBroken[lane] = broken;
  workgroupBarrier();
  var width = REDUCTION_LANES / 2u;
  loop {
    if (width == 0u) { break; }
    if (lane < width) {
      laneTranslational[lane] += laneTranslational[lane + width];
      laneRotational[lane] += laneRotational[lane + width];
      laneForce[lane] = max(laneForce[lane], laneForce[lane + width]);
      laneBroken[lane] = max(laneBroken[lane], laneBroken[lane + width]);
    }
    workgroupBarrier();
    width = width / 2u;
  }
  if (lane == 0u) {
    let degrees = 3.0 * f32(params.molecules) * BOLTZMANN * FORCE_TO_ACCELERATION;
    stats[0] = laneTranslational[0] / degrees;
    stats[1] = laneRotational[0] / degrees;
    stats[2] = laneForce[0];
    stats[3] = laneBroken[0];
  }
}
```

- [ ] **Step 4: Add the reader and the guard**

In `src/simulation.ts`:

```typescript
  const reduceKernel = compute(gpu, shader, { entry: 'reduce' });
```

Add `reduceKernel.set(bag);` inside `bind()`, add `readStats` to the returned object, and define:

```typescript
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
```

Create `src/simulation-health.ts` — no imports, so node can load it:

```typescript
export interface SimulationStats {
  translationalTemperature: number;
  rotationalTemperature: number;
  maximumForce: number;
  nonFinite: boolean;
}

/** Forces above this mean the f32 integration has diverged, not that water is hot. */
export const FORCE_LIMIT = 5e4;

export const needsRestart = (stats: SimulationStats): boolean =>
  stats.nonFinite || !Number.isFinite(stats.maximumForce) || stats.maximumForce > FORCE_LIMIT;
```

In `src/simulation.ts`, drop the local `SimulationStats` declaration and instead
`import { needsRestart, type SimulationStats } from './simulation-health';`, then re-export
both alongside `FORCE_LIMIT` so `src/main.ts` can keep importing them from `./simulation`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `npm test`
Expected: PASS — two new guard tests.

Run: `npm run build && npx playwright test tests/browser/simulation.spec.ts`
Expected: PASS — five tests.

- [ ] **Step 6: Commit**

```bash
git add src/simulation.wgsl src/simulation.ts tests/simulation-guard.test.ts tests/browser/simulation.spec.ts
git commit -m "Reduce kinetic temperatures and force peaks on the GPU, and guard against divergence"
```

---

### Task 8: Control panel and the render loop

**Files:**
- Modify: `src/scene.ts` (box per update, report the hydrogen-bond count)
- Modify: `tests/scene.test.ts` (pass the box to `update`)
- Modify: `src/background.ts` (`render(sites, box)` returns the scene counts)
- Modify: `src/main.ts` (full control wiring)
- Modify: `index.html`, `src/style.css`
- Modify: `tests/browser/homepage.spec.ts`

**Interfaces:**
- Consumes: `Simulation` from Tasks 5-7.
- Produces: `createScene(particles: number)` with `update(sites: Float32Array, box: number): SceneCounts` where `SceneCounts` is `{ atomCount: number; bondVertices: number; hydrogenBonds: number }`; `render(sites: Float32Array, box: number): SceneCounts` from `createBackground`.

- [ ] **Step 1: Write the failing tests**

Change every `createScene(n, box)` / `scene.update(sites)` pair in `tests/scene.test.ts` to `createScene(n)` / `scene.update(sites, box)`, and add one test:

```typescript
test('the scene follows a box that changes with density, and counts cell hydrogen bonds', () => {
  const box = 12.7, centre = box / 2;
  const scene = createScene(2);
  const reach = 2.8 / OH_LENGTH;
  const sites = new Float32Array([...molecule(centre, centre, centre),
    ...molecule(centre + c * reach, centre + s * reach, centre)]);
  const loose = scene.update(sites, box);
  assert.equal(loose.hydrogenBonds, 1);
  // A larger box scales the view down, so the same molecules land closer together.
  const wide = scene.update(sites, box * 1.5);
  const span = (counts: { atomCount: number }) => {
    let extent = 0;
    for (let atom = 0; atom < counts.atomCount; atom++) {
      extent = Math.max(extent, Math.abs(scene.atoms[atom * ATOM_STRIDE]));
    }
    return extent;
  };
  assert.ok(span(wide) < span(loose));
  assert.equal(wide.hydrogenBonds, 1);
});
```

Rewrite `tests/browser/homepage.spec.ts` around the new controls:

```typescript
import { test, expect } from '@playwright/test';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { resolve, extname } from 'node:path';

const ready = async (page: import('@playwright/test').Page) => {
  await expect(page.locator('.molecular-stage')).toHaveClass(/ready/, { timeout: 20000 });
};

test('heating the sample raises the measured temperature and advances simulation time', async ({ page }) => {
  test.setTimeout(120000);
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
  await page.goto('/');
  await ready(page);
  await page.screenshot({ path: 'test-results/cold.png' });
  await page.getByLabel('温度').fill('500');
  await expect(page.locator('#temperature')).toHaveText('500');
  await expect.poll(async () => Number((await page.locator('#kinetic-temperature').textContent())?.replace(/\D/g, '')),
    { timeout: 30000 }).toBeGreaterThan(400);
  await expect.poll(async () => Number((await page.locator('#sim-time').textContent())?.replace(/[^\d.]/g, '')),
    { timeout: 10000 }).toBeGreaterThan(2);
  await page.screenshot({ path: 'test-results/hot.png' });
  expect(errors).toEqual([]);
});

test('pausing holds the simulation and reset returns to the ice lattice', async ({ page }) => {
  await page.goto('/');
  await ready(page);
  await page.getByRole('button', { name: '計算を一時停止' }).click();
  const held = await page.locator('#sim-time').textContent();
  await page.waitForTimeout(600);
  await expect(page.locator('#sim-time')).toHaveText(held!);
  await page.getByRole('button', { name: '氷から再開' }).click();
  await expect(page.locator('#sim-time')).toHaveText('0.0 ps');
  await expect(page.getByRole('button', { name: '計算を再開' })).toBeEnabled();
});

test('molecule count and density change the cell and stay stable', async ({ page }) => {
  test.setTimeout(90000);
  await page.goto('/');
  await ready(page);
  await page.getByLabel('分子数').selectOption('512');
  await expect(page.locator('#box-length')).toHaveText('25.4 Å');
  await page.getByLabel('密度').fill('140');
  await expect(page.locator('#density-ratio')).toHaveText('1.40');
  await expect.poll(async () => Number((await page.locator('#box-length').textContent())?.replace(/[^\d.]/g, '')))
    .toBeLessThan(25.4);
  await expect.poll(async () => Number((await page.locator('#bond-count').textContent())?.replace(/\D/g, '')),
    { timeout: 20000 }).toBeGreaterThan(100);
  // Switching size and squeezing the box must not deform any molecule.
  const worstBond = await page.evaluate(async () => {
    const simulation = (window as any).waterSimulation;
    const sites = await simulation.readSites();
    let worst = 0;
    for (let molecule = 0; molecule < simulation.molecules; molecule++) {
      const oxygen = [0, 1, 2].map(axis => sites[molecule * 12 + axis]);
      for (const hydrogen of [1, 2]) {
        const bond = [0, 1, 2].map(axis => sites[molecule * 12 + hydrogen * 4 + axis] - oxygen[axis]);
        worst = Math.max(worst, Math.abs(Math.hypot(...bond) - 0.9572));   // OH_LENGTH
      }
    }
    return worst;
  });
  expect(worstBond).toBeLessThan(0.001);
});

test('reduced motion shows the ice lattice without stepping', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/');
  await ready(page);
  await expect(page.locator('#sim-time')).toHaveText('0.0 ps');
  await expect(page.locator('#playback-status')).toHaveText('一時停止中');
  await page.getByRole('button', { name: '計算を再開' }).click();
  await expect(page.locator('#playback-status')).toHaveText('計算中');
});

test('no WebGPU shows a real static molecule image and readable sections', async ({ page }) => {
  await page.addInitScript(() => Object.defineProperty(navigator, 'gpu', { value: undefined }));
  await page.goto('/');
  await expect(page.locator('#playback-status')).toHaveText('静止画を表示中');
  await expect(page.getByLabel('温度')).toBeDisabled();
  expect(await page.locator('#static-molecules').evaluate((img: HTMLImageElement) => img.complete && img.naturalWidth > 0)).toBe(true);
  await page.screenshot({ path: 'test-results/static-fallback.png' });
  for (const name of ['About', 'Research', 'Publications', 'Presentations', 'Elsewhere']) {
    await expect(page.getByRole('heading', { name: new RegExp(name) })).toBeVisible();
  }
});

test('a failed ice configuration fetch degrades to the static image', async ({ page }) => {
  await page.route('**/data/ice-216.bin', route => route.abort());
  await page.goto('/');
  await expect(page.locator('#playback-status')).toHaveText('静止画を表示中');
  await expect(page.locator('.molecular-stage')).not.toHaveClass(/ready/);
  await expect(page.getByLabel('密度')).toBeDisabled();
});

test('small mobile screens retain navigation and avoid horizontal overflow', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/');
  await ready(page);
  // Narrow viewports default to the smallest sample.
  await expect(page.getByLabel('分子数')).toHaveValue('64');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.getByRole('navigation').getByRole('link', { name: 'Publications' }).click();
  await expect(page).toHaveURL(/#publications$/);
  await page.screenshot({ path: 'test-results/mobile.png', fullPage: true });
});

test('production assets and the simulation work under a GitHub Pages project path', async ({ page }) => {
  const root = resolve('dist');
  const mime: Record<string, string> = { '.html':'text/html', '.js':'text/javascript', '.css':'text/css',
    '.json':'application/json', '.svg':'image/svg+xml', '.bin':'application/octet-stream' };
  const server = createServer(async (req, res) => {
    const url = new URL(req.url!, 'http://localhost');
    if (!url.pathname.startsWith('/research-homepage/')) { res.writeHead(404).end(); return; }
    const filename = resolve(root, url.pathname.slice('/research-homepage/'.length) || 'index.html');
    if (!filename.startsWith(root + '/')) { res.writeHead(403).end(); return; }
    try {
      const bytes = await readFile(filename);
      res.writeHead(200, { 'Content-Type': mime[extname(filename)] || 'application/octet-stream' }).end(bytes);
    } catch { res.writeHead(404).end(); }
  });
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const address = server.address() as { port: number };
    await page.goto(`http://127.0.0.1:${address.port}/research-homepage/`);
    await ready(page);
    await expect(page.getByRole('button', { name: '計算を一時停止' })).toBeEnabled();
    expect(await page.locator('#static-molecules').evaluate((img: HTMLImageElement) => img.naturalWidth > 0)).toBe(true);
  } finally {
    await page.goto('about:blank');
    await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
  }
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npm test`
Expected: FAIL — `scene.update` still ignores its second argument, and `hydrogenBonds` is undefined.

- [ ] **Step 3: Move the box into the scene update**

Replace `src/scene.ts` with this version. The only differences are that the box arrives per update, the projection scale is derived per frame, and the hydrogen-bond count is reported.

```typescript
import { findHydrogenBonds, molecularOpacity, periodicMolecules, type HydrogenBond, type Vector3 } from './water-geometry';

/** Atom: clip x,y, depth, radius | opacity, kind (0 oxygen, 1 hydrogen), 0, 0. */
export const ATOM_STRIDE = 8;
/** Bond vertex: clip x,y, depth, dash phase | opacity, kind (0 covalent, 1 hydrogen bond), 0, 0. */
export const BOND_STRIDE = 8;

const IMAGES = 27;
/** Molecules fade out by this fraction of the box; see molecularOpacity. */
const WINDOW = .66;
const ATOM_RADIUS = [.44, .26, .26];
const YAW = .57, PITCH = .32;
const DASHES_PER_CLIP_UNIT = 34;
/** Geometric criteria admit at most a few acceptors per donor molecule. */
const HYDROGEN_BONDS_PER_MOLECULE = 6;

/** Maps the observation sphere onto the vertical clip range. */
export const viewScale = (box: number) => 1 / (WINDOW * box);

export interface SceneCounts { atomCount: number; bondVertices: number; hydrogenBonds: number }

export function createScene(particles: number) {
  const atomCapacity = particles * IMAGES * 3;
  const bondCapacity = particles * IMAGES * (2 + HYDROGEN_BONDS_PER_MOLECULE);
  const atoms = new Float32Array(atomCapacity * ATOM_STRIDE);
  const scratch = new Float32Array(atoms.length);
  const bonds = new Float32Array(bondCapacity * 2 * BOND_STRIDE);
  const order = new Int32Array(atomCapacity);
  const imageOfCell = new Int32Array(particles * IMAGES);
  const donors: HydrogenBond[][] = Array.from({ length: particles }, () => []);
  const sin = [Math.sin(YAW), Math.sin(PITCH)], cos = [Math.cos(YAW), Math.cos(PITCH)];
  const projected: Vector3 = [0, 0, 0];

  /** Rotates a box-centred position in angstrom into clip space with its depth. */
  function project(x: number, y: number, z: number, scale: number): Vector3 {
    const rx = x * cos[0] + z * sin[0], rz = -x * sin[0] + z * cos[0];
    projected[0] = rx * scale;
    projected[1] = (y * cos[1] - rz * sin[1]) * scale;
    projected[2] = (y * sin[1] + rz * cos[1]) * scale;
    return projected;
  }
  const cellOfShift = (shift: Vector3, box: number) =>
    (Math.round(shift[0] / box) + 1) * 9 + (Math.round(shift[1] / box) + 1) * 3 + Math.round(shift[2] / box) + 1;

  function writeBond(target: number, position: Vector3, phase: number, opacity: number, kind: number): void {
    bonds[target] = position[0];
    bonds[target + 1] = position[1];
    bonds[target + 2] = position[2];
    bonds[target + 3] = phase;
    bonds[target + 4] = opacity;
    bonds[target + 5] = kind;
  }

  function update(sites: Float32Array, box: number): SceneCounts {
    const scale = viewScale(box);
    const images = periodicMolecules(sites, box);
    imageOfCell.fill(-1);
    for (let i = 0; i < images.length; i++) {
      imageOfCell[images[i].molecule * IMAGES + cellOfShift(images[i].shift, box)] = i;
    }
    let atomCount = 0;
    for (const image of images) {
      const base = image.molecule * 12;
      for (let atom = 0; atom < 3; atom++) {
        const clip = project(
          sites[base + atom * 4] + image.shift[0] - box / 2,
          sites[base + atom * 4 + 1] + image.shift[1] - box / 2,
          sites[base + atom * 4 + 2] + image.shift[2] - box / 2, scale);
        const target = atomCount * ATOM_STRIDE;
        scratch[target] = clip[0];
        scratch[target + 1] = clip[1];
        scratch[target + 2] = clip[2];
        scratch[target + 3] = ATOM_RADIUS[atom] * scale;
        scratch[target + 4] = image.opacity;
        scratch[target + 5] = atom === 0 ? 0 : 1;
        order[atomCount] = atomCount++;
      }
    }
    // Painter's order: distant atoms first, so nearer spheres blend over them.
    const sorted = order.subarray(0, atomCount);
    sorted.sort((a, b) => scratch[a * ATOM_STRIDE + 2] - scratch[b * ATOM_STRIDE + 2]);
    for (let i = 0; i < atomCount; i++) {
      atoms.set(scratch.subarray(sorted[i] * ATOM_STRIDE, (sorted[i] + 1) * ATOM_STRIDE), i * ATOM_STRIDE);
    }

    for (const list of donors) list.length = 0;
    let hydrogenBonds = 0;
    for (const bond of findHydrogenBonds(sites, box)) {
      if (bond.strength <= 0) continue;
      donors[bond.donor].push(bond);
      hydrogenBonds++;
    }
    let vertices = 0;
    const segment = (from: Vector3, to: Vector3, opacity: number, kind: number) => {
      if (vertices + 2 > bondCapacity * 2) return;
      const length = Math.hypot(to[0] - from[0], to[1] - from[1], to[2] - from[2]);
      writeBond(vertices++ * BOND_STRIDE, from, 0, opacity, kind);
      writeBond(vertices++ * BOND_STRIDE, to, kind === 1 ? length * DASHES_PER_CLIP_UNIT : 0, opacity, kind);
    };
    const site = (molecule: number, atom: number, shift: Vector3): Vector3 => {
      const base = molecule * 12 + atom * 4;
      return [...project(sites[base] + shift[0] - box / 2, sites[base + 1] + shift[1] - box / 2,
        sites[base + 2] + shift[2] - box / 2, scale)];
    };
    for (const image of images) {
      const oxygen = site(image.molecule, 0, image.shift);
      for (const hydrogen of [1, 2]) segment(oxygen, site(image.molecule, hydrogen, image.shift), image.opacity, 0);
      for (const bond of donors[image.molecule]) {
        const cells = [0, 1, 2].map(axis => Math.round((image.shift[axis] + bond.shift[axis]) / box));
        if (cells.some(index => Math.abs(index) > 1)) continue;
        const acceptor = imageOfCell[bond.acceptor * IMAGES + (cells[0] + 1) * 9 + (cells[1] + 1) * 3 + cells[2] + 1];
        if (acceptor < 0) continue;
        const opacity = Math.min(image.opacity, images[acceptor].opacity) * bond.strength;
        if (opacity < .02) continue;
        const shift = image.shift.map((value, axis) => value + bond.shift[axis]) as Vector3;
        segment(site(image.molecule, bond.hydrogen, image.shift), site(bond.acceptor, 0, shift), opacity, 1);
      }
    }
    return { atomCount, bondVertices: vertices, hydrogenBonds };
  }

  return { atoms, bonds, atomCapacity, bondCapacity, update };
}
```

Then replace `src/background.ts` with this version, which shares the caller's `gpu`, takes sites per frame, and can be resized when the molecule count changes:

```typescript
import { draw, frame, storage, surface, type Gpu } from 'vgpu';
import atomShader from './particles.wgsl?raw';
import bondShader from './bonds.wgsl?raw';
import { ATOM_STRIDE, BOND_STRIDE, createScene, type SceneCounts } from './scene';

export async function createBackground(gpu: Gpu, canvas: HTMLCanvasElement, molecules: number,
    onFailure: () => void) {
  let failed = false;
  let observer: ResizeObserver | undefined;
  const fail = () => {
    if (failed || gpu.disposed) return;
    failed = true;
    observer?.disconnect();
    onFailure();
  };
  gpu.onError(error => { console.error('Molecular background:', error); fail(); });
  void gpu.gpu.lost.then(info => { if (info.reason !== 'destroyed') fail(); });
  try {
    const screen = surface(gpu, canvas, { dpr: [1, 1.5], alphaMode: 'premultiplied', clearColor: [0, 0, 0, 0] });
    let scene = createScene(molecules);
    // Worst-case capacity keeps the storage binding size stable while the sample runs.
    let atomBuffer = storage(gpu, scene.atoms.byteLength, 'read');
    let bondBuffer = storage(gpu, scene.bonds.byteLength, 'read');
    const atoms = draw(gpu, { shader: atomShader, vertices: 6, instances: scene.atomCapacity, blend: 'alpha' });
    const bonds = draw(gpu, { shader: bondShader, geometry: { topology: 'line-list' }, blend: 'alpha' });
    const bind = () => { atoms.set({ points: atomBuffer }); bonds.set({ points: bondBuffer }); };
    bind();
    let latest: { sites: Float32Array; box: number } | undefined;

    function render(sites: Float32Array, box: number): SceneCounts {
      latest = { sites, box };
      const counts = scene.update(sites, box);
      frame(gpu, current => {
        const params = { aspect: screen.size[0] / screen.size[1], warmth: 0 };
        atomBuffer.write(scene.atoms.subarray(0, counts.atomCount * ATOM_STRIDE));
        bondBuffer.write(scene.bonds.subarray(0, counts.bondVertices * BOND_STRIDE));
        atoms.set({ params });
        bonds.set({ params });
        current.pass({ target: screen, clear: [0, 0, 0, 0] }, pass => {
          pass.draw(bonds, { vertices: counts.bondVertices });
          pass.draw(atoms, { instances: counts.atomCount });
        });
      });
      return counts;
    }

    /** A new molecule count needs a new scene and matching buffer capacity. */
    function resize(count: number) {
      scene = createScene(count);
      atomBuffer = storage(gpu, scene.atoms.byteLength, 'read');
      bondBuffer = storage(gpu, scene.bonds.byteLength, 'read');
      bind();
      latest = undefined;
    }

    // Resizing the canvas redraws the last state even while the simulation is paused.
    observer = new ResizeObserver(() => {
      if (!latest) return;
      try { render(latest.sites, latest.box); } catch { fail(); }
    });
    observer.observe(canvas);
    await gpu.settled();
    if (failed) throw new Error('GPU initialization failed');
    return { render, resize, dispose: () => observer?.disconnect() };
  } catch (error) {
    observer?.disconnect();
    throw error;
  }
}
```

The `warmth` uniform stays in the shaders but is now fed a constant zero, because temperature is shown in the panel rather than by tinting; leave the shader unchanged so the render path keeps working.

- [ ] **Step 4: Rewrite the panel markup**

Replace the `<aside class="simulation-panel">` block in `index.html` with:

```html
      <aside class="simulation-panel" aria-label="水の分子動力学計算">
        <div class="simulation-heading"><span class="small-label">WATER / <span id="phase-label">ICE Ic</span></span><span class="model-tag">TIP4P-Ew</span></div>
        <div class="simulation-readout"><span><strong id="temperature">180</strong><span class="unit"> K</span></span><span class="simulation-caption">実測 <span id="kinetic-temperature">180 K</span><span id="sim-time">0.0 ps</span><span id="playback-status" role="status">静止画を表示中</span></span></div>
        <div class="control-row">
          <label for="temperature-input">温度</label>
          <input id="temperature-input" type="range" min="150" max="500" step="5" value="180" disabled />
        </div>
        <div class="control-row">
          <label for="density-input">密度</label>
          <input id="density-input" type="range" min="60" max="140" step="5" value="100" disabled />
          <span class="control-value"><span id="density-ratio">1.00</span> · <span id="box-length">19.0 Å</span></span>
        </div>
        <div class="playback-controls">
          <button id="play-pause" type="button" disabled aria-label="計算を一時停止">停止</button>
          <label class="sr-only" for="molecules">分子数</label>
          <select id="molecules" disabled><option value="64">64</option><option value="216" selected>216</option><option value="512">512</option></select>
          <label class="sr-only" for="rate">計算速度</label>
          <select id="rate" disabled><option value="2">0.5×</option><option value="8" selected>1×</option><option value="16">2×</option><option value="32">4×</option></select>
          <button id="reset" type="button" disabled aria-label="氷から再開">↺</button>
        </div>
        <div class="simulation-metrics"><span>水素結合 <strong id="bond-count">0</strong></span><span id="throughput">0.0 ps/s</span></div>
        <details class="simulation-details"><summary>この計算について <span aria-hidden="true">＋</span></summary><p>剛体TIP4P-Ew水をブラウザのWebGPUで実時間計算しています。酸素と2つの水素を描き、電荷サイトは描画していません。実線は拘束された分子内O-H結合、破線は幾何学的に判定した水素結合（O···O ≤ 3.5 Å、O-H···O ≥ 150°）です。温度スライダーはサーモスタットの設定値で、実測値は瞬間の運動温度です。</p><p>近似: 長距離静電相互作用はPMEではなく反応場カットオフ、計算はf32単精度、摩擦5 ps⁻¹のLangevin熱浴で動力学は減衰しています。<strong>冷却しても氷には戻りません</strong>（結晶核形成はこの規模・時間では起こらず、非晶質に固まります）。「↺」は氷Icからの再開です。表示は周期境界のバルク水を球状に切り出した窓で、液滴や表面ではありません。</p><p><a href="https://doi.org/10.1063/1.1683075" target="_blank" rel="noopener noreferrer">水モデルの原論文 ↗</a> · <a href="./data/ice-216.bin" download>初期配置 ↓</a></p></details>
      </aside>
```

Add to `src/style.css`, right after the `.playback-controls` rules:

```css
.control-row{display:flex;align-items:center;gap:10px;margin-bottom:8px;font-size:10px;color:#5d7c89}
.control-row label{width:28px;flex:none}
.control-row input{flex:1;width:0;accent-color:#4d849d;height:24px}
.control-value{flex:none;font-size:9px;color:#718690;font-variant-numeric:tabular-nums}
.simulation-metrics{display:flex;justify-content:space-between;font-size:9px;color:#718690;margin-top:8px;font-variant-numeric:tabular-nums}
.simulation-metrics strong{font-weight:600;color:#4a6b78}
```

- [ ] **Step 5: Rewrite `src/main.ts`**

```typescript
import './style.css';
import { init } from 'vgpu';
import { createSimulation, needsRestart } from './simulation';
import { MOLECULE_COUNTS, boxLength } from './water-model';

const get = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const play = get<HTMLButtonElement>('play-pause');
const reset = get<HTMLButtonElement>('reset');
const temperatureInput = get<HTMLInputElement>('temperature-input');
const densityInput = get<HTMLInputElement>('density-input');
const moleculeSelect = get<HTMLSelectElement>('molecules');
const rate = get<HTMLSelectElement>('rate');
const status = get<HTMLElement>('playback-status');
const stage = document.querySelector('.molecular-stage')!;
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const controls = [play, reset, temperatureInput, densityInput, moleculeSelect, rate];

let renderer: Awaited<ReturnType<typeof import('./background').createBackground>> | undefined;
let simulation: ReturnType<typeof createSimulation> | undefined;
let playing = false;
let request = 0;
let drawing = false;
let lastStatsAt = 0;
let stepsSince = 0;
let throughputAt = 0;

get('year').textContent = String(new Date().getFullYear());

function updateControls() {
  play.textContent = playing ? '停止' : '再開';
  play.setAttribute('aria-label', playing ? '計算を一時停止' : '計算を再開');
  status.textContent = playing ? '計算中' : '一時停止中';
}
function stop() {
  playing = false;
  cancelAnimationFrame(request);
  updateControls();
}
function fallback() {
  stop();
  renderer?.dispose();
  renderer = undefined;
  simulation = undefined;
  stage.classList.remove('ready');
  for (const control of controls) control.disabled = true;
  status.textContent = '静止画を表示中';
}
function showTime() {
  if (!simulation) return;
  get('sim-time').textContent = `${simulation.timePs.toFixed(1)} ps`;
  get('box-length').textContent = `${simulation.box.toFixed(1)} Å`;
}

/** One animation frame: step, draw the latest sites, and refresh the readouts twice a second. */
function tick(time: number) {
  if (!simulation || !renderer) return;
  const steps = Number(rate.value);
  if (playing && !document.hidden) {
    simulation.step(steps);
    stepsSince += steps;
  }
  if (!drawing) {
    drawing = true;
    void simulation.readSites().then(sites => {
      const counts = renderer?.render(sites, simulation!.box);
      if (counts) get('bond-count').textContent = String(counts.hydrogenBonds);
      drawing = false;
    }).catch(error => { console.warn('Background unavailable', error); fallback(); });
  }
  showTime();
  if (time - lastStatsAt > 500) {
    lastStatsAt = time;
    const elapsed = (time - throughputAt) / 1000;
    if (elapsed > 0) {
      get('throughput').textContent = `${(stepsSince * 0.002 / elapsed).toFixed(1)} ps/s`;
      stepsSince = 0;
      throughputAt = time;
    }
    void simulation.readStats().then(stats => {
      get('kinetic-temperature').textContent = `${Math.round(stats.translationalTemperature)} K`;
      // A diverged sample is restarted rather than left to fill the screen with artefacts.
      if (needsRestart(stats)) { simulation?.reset(); status.textContent = '氷から再開しました'; }
    });
  }
  if (playing) request = requestAnimationFrame(tick);
}
function start() {
  if (!simulation) return;
  playing = true;
  updateControls();
  throughputAt = performance.now();
  stepsSince = 0;
  cancelAnimationFrame(request);
  request = requestAnimationFrame(tick);
}

play.addEventListener('click', () => playing ? stop() : start());
reset.addEventListener('click', () => {
  simulation?.reset();
  showTime();
  if (!playing) requestAnimationFrame(tick);
});
temperatureInput.addEventListener('input', () => {
  get('temperature').textContent = temperatureInput.value;
  simulation?.setTemperature(Number(temperatureInput.value));
  const kelvin = Number(temperatureInput.value);
  get('phase-label').textContent = kelvin <= 240 ? 'ICE Ic' : kelvin <= 380 ? 'WARMING' : 'HOT';
});
densityInput.addEventListener('input', () => {
  const ratio = Number(densityInput.value) / 100;
  get('density-ratio').textContent = ratio.toFixed(2);
  simulation?.setDensity(ratio);
  showTime();
  if (!playing) requestAnimationFrame(tick);
});
moleculeSelect.addEventListener('change', () => {
  const molecules = Number(moleculeSelect.value);
  simulation?.setMolecules(molecules);
  simulation?.setDensity(Number(densityInput.value) / 100);
  renderer?.resize(molecules);
  showTime();
  if (!playing) requestAnimationFrame(tick);
});
document.addEventListener('visibilitychange', () => { if (!document.hidden && playing) request = requestAnimationFrame(tick); });
reducedMotion.addEventListener('change', () => { if (reducedMotion.matches) stop(); });
window.addEventListener('pagehide', () => cancelAnimationFrame(request));
window.addEventListener('pageshow', () => { if (playing) request = requestAnimationFrame(tick); });

async function initialize() {
  if (!navigator.gpu) return;
  try {
    const signal = AbortSignal.timeout(20000);
    const base = import.meta.env.BASE_URL;
    const ice: Record<number, Float32Array> = {};
    await Promise.all(MOLECULE_COUNTS.map(async count => {
      const response = await fetch(`${base}data/ice-${count}.bin`, { signal });
      if (!response.ok) throw new Error(`Missing ice configuration for ${count} molecules`);
      ice[count] = new Float32Array(await response.arrayBuffer());
    }));
    // Narrow screens are usually phones; start them on the smallest sample.
    const molecules = innerWidth <= 720 ? 64 : 216;
    moleculeSelect.value = String(molecules);
    const gpu = await init({ powerPreference: 'low-power' });
    simulation = createSimulation(gpu, { molecules, densityRatio: 1, temperature: 180, ice });
    const { createBackground } = await import('./background');
    renderer = await createBackground(gpu, get<HTMLCanvasElement>('molecule-canvas'), molecules, fallback);
    (window as unknown as { waterSimulation: unknown }).waterSimulation = simulation;
    stage.classList.add('ready');
    showTime();
    get('box-length').textContent = `${boxLength(molecules, 1).toFixed(1)} Å`;
    renderer.render(await simulation.readSites(), simulation.box);
    updateControls();
    if (!reducedMotion.matches) start(); else requestAnimationFrame(tick);
    // Enabled last: between enabling and start() the pause button reads 停止 while
    // `playing` is still false, so a click in that window starts the loop instead.
    for (const control of controls) control.disabled = false;
  } catch (error) {
    console.warn('Using the static molecular background:', error);
    fallback();
  }
}
void initialize();
```

`fallback()` in `main.ts` must now dispose the shared `gpu` as well, since `createBackground` no longer owns it: keep a module-level `let gpu: Gpu | undefined`, assign it in `initialize()`, and call `gpu?.dispose()` inside `fallback()` after `renderer?.dispose()`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `npm test`
Expected: PASS.

Run: `npm run build && npx playwright test`
Expected: PASS — the simulation suite plus the eight homepage tests.

- [ ] **Step 7: Commit**

```bash
git add src/scene.ts src/background.ts src/main.ts index.html src/style.css tests/scene.test.ts tests/browser/homepage.spec.ts
git commit -m "Drive the live simulation from temperature, density, size and speed controls"
```

---

### Task 9: Remove the playback path and update the documents

**Files:**
- Delete: `src/trajectory.ts`, `tests/trajectory.test.ts`
- Modify: `src/water-geometry.ts` (drop `interpolateHydrogens`, `orientation`, `rotate`)
- Modify: `tests/water-geometry.test.ts` (drop the interpolation test and the decoder tests)
- Modify: `scripts/generate_static.py` (read `public/data/ice-216.bin`)
- Modify: `README.md`, `docs/explicit-water-simulation.md`
- Regenerate: `public/molecules.svg`

**Interfaces:**
- Consumes: nothing new.
- Produces: no exported API; `public/molecules.svg` now shows the 216-molecule initial state.

- [ ] **Step 1: Delete the dead code and its tests**

```bash
git rm src/trajectory.ts tests/trajectory.test.ts
```

In `src/water-geometry.ts` remove `interpolateHydrogens`, `orientation`, `rotate` and the now-unused `Quaternion` type; keep `findHydrogenBonds`, `periodicMolecules`, `molecularOpacity`, `OH_LENGTH`, `HOH_ANGLE` and the vector helpers they use. In `tests/water-geometry.test.ts` remove `water interpolation keeps O-H bonds rigid…` and `binary decode rejects truncated data`, and drop the `decodeTrajectory`/`sampleFrame` imports.

Run: `npm test`
Expected: PASS. Run `npx tsc --noEmit`; expected: no output. If it reports unused imports in `src/scene.ts`, remove them.

- [ ] **Step 2: Rebuild the static fallback from the initial configuration**

In `scripts/generate_static.py`, replace the metadata and coordinate loading with the ice file, keeping the rest of the drawing code unchanged:

```python
MOLECULES = 216
CELLS = 3
box = 6.35 * CELLS
molecules = MOLECULES
raw = np.fromfile(root / f'public/data/ice-{MOLECULES}.bin', dtype='<f4', count=MOLECULES * 9)
sites = raw.reshape(MOLECULES, 3, 3).astype(float)
scale = 1 / (WINDOW * box)
```

Delete the `meta = json.loads(...)` lines and the `molecules, box = meta['particles'], meta['box']` line.

Run: `.venv/bin/python scripts/generate_static.py`
Expected: `Generated public/molecules.svg: … molecule images, … hydrogen bonds` with roughly 270 images.

- [ ] **Step 3: Update the documents**

In `README.md`, replace the「サンプル計算の内容と再生成」section so it describes the live simulation: the model and approximations (reaction field instead of PME, f32, 5 ps⁻¹ friction, no refreezing on cooling), the three sample sizes with their boxes and cutoffs, the controls, and the commands

```sh
.venv/bin/python scripts/generate_explicit_water.py     # ice initial states and the reference trajectory
.venv/bin/python scripts/reference_forces.py            # force fixture for the GPU test
.venv/bin/python scripts/generate_static.py             # static fallback image
.venv/bin/python -m unittest discover -s scripts -p 'test_*.py'
```

State plainly that the OpenMM trajectory in `reference/` is no longer displayed and exists to validate the browser simulation.

In `docs/explicit-water-simulation.md`, retitle the document as the reference-solution report, keep the protocol and validation tables, replace the "Data contract" section with the initial-state format (`public/data/ice-<N>.bin`, float32 LE, molecule-major, O then H then H, oxygen wrapped) and add a short section describing the browser model: reaction field with conducting boundary, cutoff `min(9, 0.49 L)`, 2 fs BAOAB Langevin at 5 ps⁻¹, and the fact that forces are tested against `scripts/reference_forces.py`.

- [ ] **Step 4: Verify the whole suite**

Run: `npm test && npm run build && npx playwright test && .venv/bin/python -m unittest discover -s scripts -p 'test_*.py'`
Expected: all green, and `ls -l dist/data` shows only the three ice files.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Remove the recorded-playback path and document the live simulation"
```

---

### Task 10: Physical validation and performance

**Files:**
- Create: `scripts/generate_reference_rdf.py`
- Create: `tests/fixtures/reference-rdf-300k.json` (generated)
- Modify: `tests/browser/simulation.spec.ts` (append four tests)

**Interfaces:**
- Consumes: `build_system` from `scripts/generate_explicit_water.py`, the `Simulation` API, and `tests/fixtures/reference-rdf-300k.json`.
- Produces: the fixture with keys `bins` (bin centres in Å), `gr` (oxygen-oxygen g(r)), `temperature`, `molecules`, `box`, `firstPeak` (position of the highest maximum below 4 Å).

- [ ] **Step 1: Generate the OpenMM reference structure at 300 K**

Create `scripts/generate_reference_rdf.py`:

```python
"""Equilibrium O-O radial distribution function at 300 K from OpenMM with PME.

This is the structural reference for the browser simulation, which uses a reaction
field instead. Run it once; the fixture is committed.
"""
import json
from pathlib import Path
import numpy as np
import openmm as mm
from openmm import app, unit
from generate_explicit_water import ICE_CELL, PLATFORM, PLATFORM_PROPERTIES, build_system, ice_configuration

ROOT = Path(__file__).resolve().parents[1]
CELLS, TEMPERATURE, SEED = 2, 300.0, 22039
EQUILIBRATION_PS, SAMPLING_PS, INTERVAL_PS = 20.0, 40.0, 0.5
BINS = np.arange(2.0, 6.0001, 0.05)


def sample_oxygens():
    box = ICE_CELL * CELLS
    xyz, _, _ = ice_configuration(cells=CELLS)
    topology, system, positions = build_system(xyz, box)
    integrator = mm.LangevinMiddleIntegrator(TEMPERATURE*unit.kelvin, 1/unit.picosecond, 2*unit.femtosecond)
    integrator.setRandomNumberSeed(SEED)
    simulation = app.Simulation(topology, system, integrator, PLATFORM, PLATFORM_PROPERTIES)
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(TEMPERATURE*unit.kelvin, SEED)
    simulation.step(int(EQUILIBRATION_PS / 0.002))
    frames = []
    for _ in range(int(SAMPLING_PS / INTERVAL_PS)):
        simulation.step(int(INTERVAL_PS / 0.002))
        state = simulation.context.getState(getPositions=True)
        drawn = np.array(state.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
        frames.append(drawn.reshape(-1, 4, 3)[:, 0])
    return np.array(frames), box


def radial_distribution(frames, box):
    molecules = frames.shape[1]
    counts = np.zeros(len(BINS) - 1)
    for oxygen in frames:
        d = oxygen[None] - oxygen[:, None]
        d -= box * np.rint(d / box)
        r = np.linalg.norm(d, axis=-1)
        counts += np.histogram(r[np.triu_indices(molecules, 1)], bins=BINS)[0]
    shells = 4 / 3 * np.pi * (BINS[1:] ** 3 - BINS[:-1] ** 3)
    density = molecules / box ** 3
    ideal = shells * density * molecules / 2 * len(frames)
    return counts / ideal


if __name__ == '__main__':
    frames, box = sample_oxygens()
    gr = radial_distribution(frames, box)
    centres = (BINS[1:] + BINS[:-1]) / 2
    near = centres < 4
    fixture = {'bins': centres.tolist(), 'gr': gr.tolist(), 'temperature': TEMPERATURE,
               'molecules': int(frames.shape[1]), 'box': box,
               'firstPeak': float(centres[near][np.argmax(gr[near])])}
    destination = ROOT / 'tests/fixtures/reference-rdf-300k.json'
    destination.write_text(json.dumps(fixture) + '\n')
    print(f'Wrote {destination.name}: first peak at {fixture["firstPeak"]:.2f} A')
```

Run: `.venv/bin/python scripts/generate_reference_rdf.py`
Expected: `first peak at 2.75 A` or 2.80 Å (the bin width is 0.05 Å). If it prints anything outside 2.6-2.9 Å, stop and investigate before trusting it as a reference.

- [ ] **Step 2: Write the failing tests**

Append to `tests/browser/simulation.spec.ts`:

```typescript
/** Fraction of ideal tetrahedral order over the four nearest oxygen neighbours. */
function tetrahedralOrder(sites: number[], molecules: number, box: number): number {
  const oxygen = (index: number) => [0, 1, 2].map(axis => sites[index * 12 + axis]);
  let total = 0;
  for (let i = 0; i < molecules; i++) {
    const here = oxygen(i);
    const neighbours: { distance: number; direction: number[] }[] = [];
    for (let j = 0; j < molecules; j++) {
      if (i === j) continue;
      const delta = oxygen(j).map((value, axis) => {
        const raw = value - here[axis];
        return raw - box * Math.round(raw / box);
      });
      neighbours.push({ distance: Math.hypot(...delta), direction: delta });
    }
    neighbours.sort((a, b) => a.distance - b.distance);
    const nearest = neighbours.slice(0, 4).map(entry =>
      entry.direction.map(value => value / entry.distance));
    let order = 1;
    for (let a = 0; a < 4; a++) for (let b = a + 1; b < 4; b++) {
      const cosine = nearest[a].reduce((sum, value, axis) => sum + value * nearest[b][axis], 0);
      order -= 3 / 8 * (cosine + 1 / 3) ** 2;
    }
    total += order;
  }
  return total / molecules;
}

test('heating destroys the tetrahedral lattice while cold holds it', async ({ page }) => {
  test.setTimeout(300000);
  await ready(page);
  const run = (kelvin: number) => page.evaluate(async temperature => {
    const simulation = (window as any).waterSimulation;
    simulation.setMolecules(216);
    simulation.reset();
    simulation.setTemperature(temperature);
    for (let batch = 0; batch < 100; batch++) simulation.step(100);   // 20 ps
    return { sites: Array.from(await simulation.readSites() as Float32Array),
      molecules: simulation.molecules, box: simulation.box };
  }, kelvin);
  const cold = await run(180);
  const hot = await run(520);
  const coldOrder = tetrahedralOrder(cold.sites, cold.molecules, cold.box);
  const hotOrder = tetrahedralOrder(hot.sites, hot.molecules, hot.box);
  expect(coldOrder).toBeGreaterThan(0.85);
  expect(hotOrder).toBeLessThan(0.7);
  expect(coldOrder - hotOrder).toBeGreaterThan(0.2);
});

test('the first oxygen shell agrees with the OpenMM reference', async ({ page }) => {
  test.setTimeout(300000);
  const reference = JSON.parse(await readFile('tests/fixtures/reference-rdf-300k.json', 'utf8'));
  await ready(page);
  const measured = await page.evaluate(async () => {
    const simulation = (window as any).waterSimulation;
    simulation.setMolecules(216);
    simulation.reset();
    simulation.setTemperature(300);
    for (let batch = 0; batch < 150; batch++) simulation.step(100);   // 30 ps of equilibration
    const frames: number[][] = [];
    for (let sample = 0; sample < 20; sample++) {
      simulation.step(250);
      frames.push(Array.from(await simulation.readSites() as Float32Array));
    }
    return { frames, molecules: simulation.molecules, box: simulation.box };
  });
  const width = 0.05, start = 2, bins = new Array(80).fill(0);
  for (const sites of measured.frames) {
    for (let i = 0; i < measured.molecules; i++) for (let j = i + 1; j < measured.molecules; j++) {
      const distance = Math.hypot(...[0, 1, 2].map(axis => {
        const raw = sites[j * 12 + axis] - sites[i * 12 + axis];
        return raw - measured.box * Math.round(raw / measured.box);
      }));
      const bin = Math.floor((distance - start) / width);
      if (bin >= 0 && bin < bins.length) bins[bin]++;
    }
  }
  const density = measured.molecules / measured.box ** 3;
  const gr = bins.map((count, index) => {
    const inner = start + index * width, outer = inner + width;
    const shell = 4 / 3 * Math.PI * (outer ** 3 - inner ** 3);
    return count / (shell * density * measured.molecules / 2 * measured.frames.length);
  });
  const near = gr.slice(0, Math.floor((4 - start) / width));
  const peak = start + width * (near.indexOf(Math.max(...near)) + 0.5);
  expect(Math.abs(peak - reference.firstPeak)).toBeLessThan(0.15);
  expect(Math.max(...near)).toBeGreaterThan(1.8);
});

test('the largest sample runs for half a minute without diverging', async ({ page }) => {
  test.setTimeout(120000);
  await ready(page);
  const result = await page.evaluate(async () => {
    const simulation = (window as any).waterSimulation;
    simulation.setMolecules(512);
    simulation.setTemperature(400);
    const deadline = performance.now() + 30000;
    let steps = 0;
    while (performance.now() < deadline) {
      simulation.step(16);
      steps += 16;
      await new Promise(resolve => requestAnimationFrame(resolve));
    }
    return { steps, stats: await simulation.readStats() };
  });
  expect(result.stats.nonFinite).toBe(false);
  expect(result.stats.maximumForce).toBeLessThan(50000);
  expect(result.stats.translationalTemperature).toBeGreaterThan(300);
  expect(result.stats.translationalTemperature).toBeLessThan(500);
  expect(result.steps).toBeGreaterThan(1000);
});

test('the default sample sustains a usable step rate on this machine', async ({ page }) => {
  test.setTimeout(60000);
  await ready(page);
  const rate = await page.evaluate(async () => {
    const simulation = (window as any).waterSimulation;
    simulation.setMolecules(216);
    simulation.step(200);
    await simulation.readStats();
    const started = performance.now();
    let steps = 0;
    while (performance.now() - started < 3000) { simulation.step(16); steps += 16; }
    await simulation.readStats();
    return steps / ((performance.now() - started) / 1000);
  });
  // 300 steps/s is 0.6 ps/s: slow but still watchable. Local GPUs reach several times this.
  expect(rate).toBeGreaterThan(300);
});
```

- [ ] **Step 3: Run the tests**

Run: `npm run build && npx playwright test tests/browser/simulation.spec.ts`
Expected: PASS — nine tests.

If the melting test shows a cold order below 0.85, the lattice is being disturbed by the thermostat: check that `simulation.reset()` reloads the minimized configuration and that the friction is 5 ps⁻¹, not larger. If the RDF peak sits below 2.6 Å, the reaction-field sign is wrong and molecules are over-attracting; re-run the Task 5 force test first.

- [ ] **Step 4: Full verification and screenshots**

```bash
npm test
npm run build
npx playwright test
.venv/bin/python -m unittest discover -s scripts -p 'test_*.py'
```

Expected: all green. Inspect `test-results/cold.png` and `test-results/hot.png`: the cold frame shows the ice lattice with dashed hydrogen bonds, the hot frame a disordered sample with fewer bonds.

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_reference_rdf.py tests/fixtures/reference-rdf-300k.json tests/browser/simulation.spec.ts
git commit -m "Validate melting, structure, stability and step rate against the OpenMM reference"
```

---

## Deployment

The GitHub Actions workflow runs `npm test` and `npm run build` only; the browser suite needs a real GPU and stays local. After Task 10, push to `main` and confirm on https://dieverwandlungtk.github.io/ that the sample runs, the temperature slider responds, and the console is clean.
