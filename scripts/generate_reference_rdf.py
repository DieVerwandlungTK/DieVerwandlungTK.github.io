"""O-O radial distribution function from OpenMM with PME, in two states used to validate the
browser's reaction-field structure.

The "ice" state is superheated crystalline ice Ic at 300 K, not equilibrium liquid water: TIP4P-Ew
ice Ic at ice density with no free surface does not melt within tens of picoseconds at 300 K in
this periodic box, and the run shows the crystalline signature directly, first peak 4.4 at 2.725 A
and first minimum 0.03 at 3.475 A (liquid water's first peak sits near 2.8 with a first minimum
near 0.85). The first-peak position alone cannot tell ice from liquid apart -- both put the O-O
first peak near 2.73 A -- so a 2.6-2.9 A sanity band on peak position is not a liquid-vs-ice test;
only the first-minimum height is. The "fluid" state runs the same protocol at 520 K, where the
sample does disorder within the run, giving a genuine disordered-fluid comparison.

This is the structural reference for the browser simulation, which uses a reaction field instead
of PME. Run it once per state; both fixtures are committed.
"""
import json
import sys
from pathlib import Path
import numpy as np
import openmm as mm
from openmm import app, unit
from generate_explicit_water import ICE_CELL, PLATFORM, PLATFORM_PROPERTIES, build_system, ice_configuration

ROOT = Path(__file__).resolve().parents[1]
CELLS, SEED = 2, 22039
EQUILIBRATION_PS, SAMPLING_PS, INTERVAL_PS = 20.0, 40.0, 0.5
BINS = np.linspace(2.0, 6.0, 81)

STATES = {
    'ice': {'temperature': 300.0, 'destination': 'tests/fixtures/reference-rdf-ice-300k.json'},
    'fluid': {'temperature': 520.0, 'destination': 'tests/fixtures/reference-rdf-fluid-520k.json'},
}


def sample_oxygens(temperature):
    box = ICE_CELL * CELLS
    xyz, _, _ = ice_configuration(cells=CELLS)
    topology, system, positions = build_system(xyz, box)
    integrator = mm.LangevinMiddleIntegrator(temperature*unit.kelvin, 1/unit.picosecond, 2*unit.femtosecond)
    integrator.setRandomNumberSeed(SEED)
    simulation = app.Simulation(topology, system, integrator, PLATFORM, PLATFORM_PROPERTIES)
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(temperature*unit.kelvin, SEED)
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


def generate(state):
    temperature = STATES[state]['temperature']
    frames, box = sample_oxygens(temperature)
    gr = radial_distribution(frames, box)
    centres = (BINS[1:] + BINS[:-1]) / 2
    near = centres < 4
    peak_index = np.argmax(gr[near])
    first_peak = float(centres[near][peak_index])
    # First minimum: the smallest gr value between the first peak and 4 A. This is the
    # diagnostic that separates crystalline order (near 0.03) from a disordered fluid (near
    # 0.85) -- the first-peak position alone cannot, since both land near 2.73 A.
    tail = near & (centres >= first_peak)
    minimum_index = np.argmin(gr[tail])
    first_minimum = float(gr[tail][minimum_index])
    fixture = {'bins': centres.tolist(), 'gr': gr.tolist(), 'temperature': temperature,
               'molecules': int(frames.shape[1]), 'box': box,
               'firstPeak': first_peak, 'firstMinimum': first_minimum}
    destination = ROOT / STATES[state]['destination']
    destination.write_text(json.dumps(fixture) + '\n')
    print(f'Wrote {destination.name}: first peak {fixture["firstPeak"]:.3f} at height '
          f'{gr[near][peak_index]:.3f}, first minimum {fixture["firstMinimum"]:.3f}')


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in STATES:
        sys.exit(f'usage: {sys.argv[0]} {{{"|".join(STATES)}}}')
    generate(sys.argv[1])
