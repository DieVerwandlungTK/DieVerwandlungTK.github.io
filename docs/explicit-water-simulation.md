# Explicit water heating sample: reference-solution report

The homepage no longer plays this trajectory. It runs a live WebGPU simulation of
rigid TIP4P-Ew water in the browser instead (see "Browser model" below). This
document is now a report on the offline OpenMM reference solution: a
molecular-dynamics heating trajectory of 64 rigid TIP4P-Ew water molecules in a
12.7 Å periodic cube, kept in `reference/` to validate the browser simulation's
force field and integrator. Oxygen and the two hydrogens are drawn in both the
reference trajectory and the live simulation; the massless fourth charge site
participates in the physics but is not drawn. Intramolecular O–H bonds are
constrained and never dissociate. Dashed hydrogen bonds are a geometric analysis,
not additional spring forces.

## Reproduce

Using Python with NumPy and OpenMM installed:

```sh
.venv/bin/python scripts/generate_explicit_water.py
.venv/bin/python -m unittest discover -s scripts -p test_explicit_water.py
```

The generated sample uses OpenMM 8.6.0, NumPy 2.5.3, the CPU platform with two
threads and deterministic forces, and random seed 22039 for both proton placement
and the thermostat. Floating-point trajectories may differ across platforms and
versions despite identical seeds.

The initial oxygen lattice is a 2×2×2 conventional diamond lattice (ice Ic).
Randomized Euler-circuit orientation of its periodic four-neighbor graph puts one
proton on each edge, with two donors and two acceptors at every oxygen. Each
molecule is projected to TIP4P-Ew geometry: 0.9572 Å O–H lengths and 104.52° H–O–H
angle. This gives proton disorder subject to the ice rules; it is not an ensemble
sampling algorithm for zero-polarization ice configurations.

OpenMM's built-in `tip4pew.xml` and `Modeller.addExtraParticles()` provide the
virtual sites and force-field parameters. The simulation uses PME with 6 Å real
space cutoff (less than half the box), Ewald error tolerance 10⁻⁵, rigid water,
constraint tolerance 10⁻⁷, and a Langevin-middle integrator with 2 fs time step and
1 ps⁻¹ friction. OpenMM's default dispersion correction and center-of-mass motion
removal remain enabled. Energy minimization is followed by 10 ps at 180 K before
recording. The 40 ps recording contains a 5 ps 180 K hold, a 20 ps linear thermostat
ramp to 450 K, and a 15 ps 450 K hold, at fixed volume. Positions are sampled every
20 fs; instantaneous temperature is calculated from kinetic energy with 381
unconstrained degrees of freedom. Oxygen RMS speed is calculated from the saved
state's velocities (rather than finite differences between frames).

This small, rapidly heated periodic sample illustrates lattice disorder and
hydrogen-bond turnover. It is not an equilibrium melting-point measurement,
experimental water, a droplet, or a surface simulation. Thermostat set point and
instantaneous kinetic temperature are distinct quantities. The geometric bond
criterion is O···O ≤3.5 Å and O–H···O ≥150°, using minimum-image distances. Fast
threshold recrossings count as break/reformation events; these counts are not
chemical reaction rates or hydrogen-bond lifetimes.

## Data contract

The browser simulation loads its initial configuration from
`public/data/ice-<N>.bin`, one file per molecule count `N` in `{64, 216, 512}`. Each
file is little-endian float32, molecule-major, three sites per molecule in the order
O, H1, H2, xyz in Å (`N × 3 × 3` values; `N × 36` bytes). Oxygen is wrapped into
`[0, box)` for the corresponding ice-Ic box (`6.35 Å × cells`, `cells = (N/8)^(1/3)`);
hydrogens are whole relative to their oxygen and may lie just outside the box. No
virtual-site (M-site) coordinates are stored; `src/initial-state.ts` derives the
charge site, velocities and rigid-body state from these three sites at load time.

The generator (`scripts/generate_explicit_water.py`) checks all coordinates, rigid
geometry, periodic interframe displacements, thermal change, structural change and
hydrogen-bond turnover before replacing the committed reference sample and the
committed ice files. The test also verifies the initial ice rules.

## Browser model

`src/simulation.wgsl` is a from-scratch rigid TIP4P-Ew integrator that runs entirely
on the GPU. It differs from the OpenMM reference protocol above in the approximations
it makes to run in real time:

- Electrostatics use an Onsager reaction field with a conducting boundary
  (`ε_rf = ∞`) instead of Ewald summation (PME), cut off at `min(9, 0.49 × box)` Å —
  never more than 9 Å, and never past half the box so the minimum-image convention
  holds.
- Arithmetic is 32-bit float throughout, so energy is not conserved exactly the way
  a double-precision reference integrator's is.
- Integration is 2 fs BAOAB Langevin at 5 ps⁻¹ friction; that friction measurably
  damps the dynamics, it is not a light thermostat coupling.
- Cooling does not refreeze the sample. Crystal nucleation is far beyond what a
  system of a few hundred molecules can show on a real-time timescale, so cooling
  produces an amorphous solid, not a return to ice. The reset control restarts the
  sample from the ice-Ic initial configuration; it does not freeze anything.

The WGSL forces are tested against `scripts/reference_forces.py`, an independent,
loop-based (not vectorized) Python reference implementation of the same reaction-field
TIP4P-Ew force law, so the two must agree to floating-point precision on the same
input configuration. The OpenMM trajectory in `reference/` is not used by the
browser at all; it is a separate, independent validation of the physics.

## Display

`src/scene.ts` replicates each molecule into the 27 neighbouring periodic cells and
keeps only whole molecules inside a spherical window of 0.66 box lengths, fading them
to zero opacity before culling. Oxygen is drawn at 0.44 Å and hydrogen at 0.26 Å in a
ball-and-stick style, solid lines for the constrained O-H bonds and dashed lines for
geometric hydrogen bonds. `src/main.ts` runs the live simulation continuously rather
than playing back a fixed-length recording; the visitor controls temperature
(150–500 K), density (60–140% of ice density), molecule count (64, 216 or 512) and
computation speed (0.5x, 1x, 2x or 4x steps per animation frame), plus
play/pause and a reset to the ice-Ic initial state. The readout shows the
thermostat set point alongside the measured kinetic temperature and the simulation
time; the two are not the same quantity.

## Sources

- [Horn et al., TIP4P-Ew (2004)](https://doi.org/10.1063/1.1683075).
- [Official OpenMM multisite-water tutorial](https://openmm.github.io/openmm-cookbook/latest/notebooks/tutorials/Histone_methyltransferase_simulation_with_a_multisite_water_model_TIP4P-Ew.html).

## Validation of the exported sample

Both offline tests pass. Means over 0–5 ps and 30–40 ps, respectively:

| Quantity | Initial cold hold | Final hot hold |
| --- | ---: | ---: |
| Kinetic temperature | 176.27 K | 455.45 K |
| Oxygen RMS speed | 5.083 Å/ps | 8.056 Å/ps |
| Four-neighbor tetrahedral order | 0.96841 | 0.51368 |
| Geometric hydrogen bonds in the cell | 127.61 | 68.36 |

Across consecutive recorded frames there are 23,182 bond losses and 23,121 bond
formations. Of 2,669 distinct donor-hydrogen-acceptor bonds seen, 2,107 disappear
and subsequently reappear. These include rapid threshold recrossings. The initial
frame has 128 bonds and the final frame has 67. The largest minimum-image oxygen
displacement between 20 fs samples is 0.4286 Å. All exported O–H lengths agree with
0.9572 Å within 0.0002 Å, H–O–H angles agree with 104.52° within 0.01°, and every
coordinate is finite. Lattice order persists through the cold hold and is lost
by the hot hold without coordinate morphing or artificially amplified velocities.

Oxygen mean-square displacement, reconstructed by minimum-image unwrapping and
subtracting center-of-mass translation, is 0.194 Å² over the initial 5 ps versus
60.878 Å² over the final 10 ps. The combination of bounded cold motion, large hot
displacements and lost tetrahedral order supports a solid-to-disordered-fluid
interpretation for this particular nonequilibrium sample.
