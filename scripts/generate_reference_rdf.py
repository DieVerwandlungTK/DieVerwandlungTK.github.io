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
