# Interactive real-time water simulation

Replace the precomputed trajectory background with a molecular dynamics simulation
that runs in the visitor's browser, so visitors can heat, compress and restart the
water themselves. The recorded OpenMM trajectory stops being display data and becomes
the reference solution that the browser simulation is tested against.

## Goal and non-goals

The motivation is interactivity: a visitor moves a temperature slider and watches ice
lose its lattice, hydrogen bonds break and reform, and thermal motion change.

Cooling does not refreeze the sample. Crystal nucleation takes microseconds to
milliseconds; a few hundred molecules over tens of picoseconds gives an amorphous
solid instead. The reset button returns to the ice lattice and is labelled as a
restart, not as freezing. This limitation is stated in the page caption.

Reproducibility of a specific trajectory is also given up: velocities are drawn from a
fresh random seed on every visit, so each visitor sees a different run.

## Physics

Rigid TIP4P-Ew water. Lennard-Jones on oxygen; charges on the two hydrogens and on
the massless M site, which sits 0.125 A from oxygen along the H-O-H bisector.

The M-site force needs no redistribution onto O and H. Rigid-body integration consumes
only the net force and the net torque about the centre of mass, and the M site is part
of the same rigid body, so its force contributes to both sums directly. This removes
the redistribution step and its test.

All pairs interact (O(N^2), GPU compute) under the minimum image convention with
cutoff `min(9, 0.49 L)` angstrom. There is deliberately no lower clamp: a 6 A floor would
exceed half the box for 64 molecules compressed to 1.4 times ice density (box 11.35 A) and
break the minimum image convention. Small samples therefore get a short cutoff. Electrostatics use the Onsager reaction field
with conducting boundary (eps_RF = infinity); both Coulomb and Lennard-Jones terms are
shifted so forces vanish continuously at the cutoff (the Lennard-Jones energy shift is
`- (r - rc) U'(rc)`, i.e. plus `(r - rc)` times the force at the cutoff). No PME, no
dispersion correction.

Integration is rigid-body: centre-of-mass translation plus quaternion rotation with
angular momentum in the principal frame, 2 fs time step, BAOAB Langevin thermostat on
translation and rotation with 5 ps^-1 friction. The friction is stronger than the usual
1 ps^-1 so the sample follows the temperature slider quickly.

| Molecules | Box at ice density | Cutoff |
| ---: | ---: | ---: |
| 64 | 12.7 A | 6.22 A |
| 216 (default) | 19.05 A | 9 A |
| 512 | 25.4 A | 9 A |

The density slider scales the box and all oxygen positions between 0.6 and 1.4 times
ice density, recomputing the cutoff. Hydrogens stay whole with their oxygen.

Initial states are not generated in the browser. `scripts/generate_explicit_water.py`
exports energy-minimized, proton-disordered ice Ic configurations for 64, 216 and 512
molecules (about 26 KB in total). Its Euler-circuit proton placement already has tests
for the ice rules, so that algorithm is not reimplemented in TypeScript.

WebGPU offers f32 only. Energy is not conserved exactly; the Langevin thermostat
absorbs the error. The statistics buffer carries the maximum force and a non-finite
flag, and the page restarts from the ice lattice when either indicates a blow-up.

The caption states the approximations: reaction-field cutoff rather than PME, f32
arithmetic, strong friction damping the dynamics, and no refreezing on cooling.

## Modules

New:

- `src/water-model.ts` — TIP4P-Ew geometry, charges, Lennard-Jones parameters, masses,
  principal moments of inertia, and the box/density/cutoff relations. Pure functions.
- `src/simulation.wgsl` — four kernels: build M sites; forces and torques with
  workgroup tiling; BAOAB integration of translation and quaternion rotation; reduce
  statistics (translational and rotational kinetic temperature, maximum force,
  non-finite flag).
- `src/simulation.ts` — owns the GPU buffers, dispatches kernels, updates uniforms,
  applies the steps-per-frame budget, reads statistics back asynchronously, and
  handles reset, molecule-count changes and density changes.
- `src/initial-state.ts` — loads the packed ice configuration for a molecule count,
  scales it to the requested density, and assigns Maxwell-Boltzmann velocities and
  angular momenta from a seeded generator with zero total momentum.

Reused: `src/scene.ts` (periodic images, observation window, hydrogen-bond geometry,
projection), `findHydrogenBonds` / `periodicMolecules` / `molecularOpacity` in
`src/water-geometry.ts`, `src/background.ts`, and both render shaders. Their unit tests
keep their meaning.

One change is needed there: `createScene(particles, box)` currently freezes the box
length at construction, but the density slider changes it while running. `update` takes
the current box length instead, and the view scale is recomputed per frame; capacity
still comes from the molecule count. Changing the molecule count recreates the scene
and the GPU buffers.

Removed: `src/trajectory.ts` and `tests/trajectory.test.ts`; `interpolateHydrogens`
(and its quaternion helpers and test), which existed only to interpolate between
recorded frames; `public/data/water.json` and `water.bin`. The OpenMM generator and
its report stay as the reference solution, with its outputs moved to `reference/` so
they are no longer served — `dist` loses 4.6 MB.

## Data flow per animation frame

1. JavaScript updates uniforms: target temperature, box length, cutoff, random counter.
2. It dispatches the M-site, force and integration kernels `stepsPerFrame` times, and
   the statistics reduction every few frames.
3. It copies positions into a staging buffer, maps it asynchronously, and passes the
   O, H, H sites to `scene.update` and then `background.render`.

The readback is 18 KB at 512 molecules. Rendering lags the simulation by about one
frame, which is not perceptible. Keeping the hydrogen-bond analysis in tested
JavaScript is the reason for reading back at all; the compute shaders stay limited to
physics. If JavaScript becomes the bottleneck at 512 molecules, scene assembly moves
to a compute shader afterwards — measured first, not assumed.

## Screen and controls

The timeline scrubber is removed; a live simulation has no seekable timeline.

- Temperature: 150-500 K slider, default 180 K so the first impression is the ice
  lattice, showing the set point and the measured kinetic temperature separately.
- Density: slider from 0.6 to 1.4 times ice density, default 1.0, showing the box
  length in angstrom and the ratio.
- Molecules: 64 / 216 / 512, defaulting to 216, and to 64 on viewports of 720 px or
  less.
- Pause and resume; reset, which restarts from ice Ic.
- Speed: steps per frame (2 / 8 / 16 / 32, default 8), with the measured picoseconds
  per second displayed.
- Readouts: elapsed simulation time in picoseconds and the hydrogen-bond count in the
  cell.
- Reduced-motion settings show the initial ice configuration, paused, without stepping.
- Without WebGPU the static SVG remains the fallback, regenerated from the 216-molecule
  initial configuration.

## Test interface

The browser tests need the simulation state, not pixels. `src/simulation.ts` exposes the live
simulation object on `window.waterSimulation` in every build: the current sites and the
raw rigid-body state as `Float32Array`s, the latest statistics (kinetic temperatures,
maximum force, non-finite flag), the elapsed simulation time, the controls the panel
uses, and one method used only by tests, which loads a given configuration and evaluates
forces once without integrating. Step rate is measured by the tests themselves, and by
the panel for its own readout. It is a few
dozen lines, carries no secrets, and keeping it in the production build means the tests
exercise exactly what visitors run.

Derived quantities the tests need (rigid geometry, total momentum, tetrahedral order,
the radial distribution function) are computed inside the tests from those sites, not
in application code.

## Verification

The physics is validated against a reference solution rather than by appearance, and
is not implemented twice.

Offline (Python):

- `scripts/reference_forces.py` computes forces, torques and potential energy for the
  reaction-field TIP4P-Ew model on a fixed 64-molecule configuration and writes
  `tests/fixtures/reference-forces-64.json`.
- `scripts/test_reference_forces.py` checks those forces and torques against
  central-difference derivatives of the potential, taken along rigid translations and
  rigid rotations of one molecule (relative error below 1e-5). This is the foundation
  everything else rests on.
- The 300 K oxygen-oxygen radial distribution function from the existing OpenMM
  reference trajectory is exported into the same fixture directory.

Node unit tests (no GPU):

- `water-model.ts`: moments of inertia, M-site construction, and the
  density/box/cutoff relations.
- `initial-state.ts`: loading each molecule count, rigid geometry, density scaling,
  zero total momentum, kinetic energy matching the requested temperature, and
  reproducibility for a fixed seed.

Browser tests (Playwright with WebGPU):

1. Forces from one GPU evaluation of the fixture configuration match the Python
   reference within 1e-3 relative error.
2. After 5,000 steps every O-H length is 0.9572 +/- 0.001 A and every H-O-H angle is
   104.52 +/- 0.05 degrees.
3. At a 300 K set point over 10 ps the mean kinetic temperature is within 25 K.
4. Total translational momentum stays near zero.
5. At 500 K the tetrahedral order parameter falls while at 180 K it is retained over
   the same interval, showing melting happens as physics.
6. The first O-O peak at 300 K agrees with the OpenMM reference within 0.15 A.
7. 512 molecules stepped continuously for 30 s never raise the non-finite flag.
8. Pause, reset, molecule-count changes and density changes raise no errors and leave
   rigid geometry intact.
9. Steps per second at 216 molecules stays above a lenient threshold. This runs
   locally only; CI runs `npm test` and the build, not the browser suite.

Acceptance: all of the above pass, `npm test`, `npm run build` and
`npm run test:browser` are green, and screenshots of the published site show the
180 K lattice and the melted 500 K sample.
