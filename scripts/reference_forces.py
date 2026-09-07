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

_HALF = np.deg2rad(104.52 / 2)
_ALONG, _ACROSS = .9572 * np.cos(_HALF), .9572 * np.sin(_HALF)
_SHIFT = 2 * MASS_H * _ALONG / (MASS_O + 2 * MASS_H)
BODY = {'oxygen': np.array([-_SHIFT, 0, 0]),
        'hydrogen': np.array([[_ALONG - _SHIFT, _ACROSS, 0], [_ALONG - _SHIFT, -_ACROSS, 0]]),
        'charge': np.array([M_OFFSET - _SHIFT, 0, 0])}


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
    """Yields molecule pairs within the cutoff, together with the minimum-image offset
    of j relative to i (added to j's site positions when computing a distance; j's
    stored coordinates are never shifted).

    Inclusion is decided once per pair, by O-O distance alone. Individual site-pair
    terms in `_reaction_field`/`_lennard_jones` vanish at their own r = cutoff, but
    since sites sit off the O-O axis, a pair admitted by that O-O test generally has
    site separations a little short of or past cutoff -- so site-pair energies and
    forces are generally nonzero right at the molecular boundary. That discontinuity
    is the intended, molecule-based approximation.
    """
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
    # Mirrors the pair/site loop structure of forces_and_torques below -- the two must
    # stay in lockstep. This is exactly where the periodic lever-arm bug lived.
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
    # Mirrors the pair/site loop structure of potential above -- the two must stay in
    # lockstep. This is exactly where the periodic lever-arm bug lived.
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
        # amount, so the shift cancels out of (site - centre) and can be omitted here;
        # `shift` is only needed above, to get the minimum-image direction of the force.
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
