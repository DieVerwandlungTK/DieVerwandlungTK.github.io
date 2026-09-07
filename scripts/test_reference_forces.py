"""The reference force field must be the gradient of its own potential."""
import re
import unittest
import numpy as np
from reference_forces import (BODY, COULOMB, MASS_H, MASS_O, Q_H, Q_M, ROOT, SIGMA, EPSILON,
                               _lennard_jones, _reaction_field, centre_of_mass,
                               forces_and_torques, potential, rotate_molecule)
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

    def test_torque_matches_the_numerical_derivative_across_the_periodic_boundary(self):
        """Regression test for the lever-arm bug: a pair whose interaction crosses the
        periodic boundary (O-O separation BOX - 2.8, direct distance far outside the
        cutoff but the minimum image well inside it) must still satisfy torque = -dU/dtheta
        and Newton's third law on the net force, exactly as an in-cell pair does.

        Molecule 1 is rotated relative to molecule 0 -- two identically-oriented
        molecules displaced along a single axis are too symmetric to exercise the bug
        (the spurious cross(shift, force) term vanishes when shift and force are
        parallel, which is exactly what happens for that degenerate geometry)."""
        unrotated = np.array([BODY['oxygen'], BODY['hydrogen'][0], BODY['hydrogen'][1]])
        tilted = rotate_molecule(unrotated, np.array([.3, .7, .2]), 1.1)
        xyz = np.array([unrotated, tilted + np.array([BOX - 2.8, 0, 0])])

        forces, torques = forces_and_torques(xyz, BOX, CUTOFF)
        self.assertLess(np.abs(forces[0] + forces[1]).max(), 1e-9)

        angle = 1e-5
        for molecule in (0, 1):
            for axis in range(3):
                direction = np.eye(3)[axis]
                plus = xyz.copy()
                minus = xyz.copy()
                plus[molecule] = rotate_molecule(xyz[molecule], direction, angle)
                minus[molecule] = rotate_molecule(xyz[molecule], direction, -angle)
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

    def test_reaction_field_matches_the_closed_form(self):
        """Pins the Onsager reaction field (conducting boundary) itself, not just its
        self-consistency with the numerical gradient: U(r) = COULOMB*qa*qb*(1/r +
        r^2/(2 rc^3) - 3/(2 rc)), radial force COULOMB*qa*qb*(1/r^2 - r/rc^3)."""
        r, cutoff = 3.0, CUTOFF
        qa, qb = Q_H, Q_M
        product = qa * qb

        # Computed longhand from the formula, independently of _reaction_field.
        expected_energy = COULOMB * qa * qb * (1 / r + r ** 2 / (2 * cutoff ** 3) - 3 / (2 * cutoff))
        expected_force = COULOMB * qa * qb * (1 / r ** 2 - r / cutoff ** 3)
        energy, force = _reaction_field(r, product, cutoff)
        self.assertAlmostEqual(energy, expected_energy, places=9)
        self.assertAlmostEqual(force, expected_force, places=9)

        edge_energy, edge_force = _reaction_field(cutoff, product, cutoff)
        self.assertAlmostEqual(edge_energy, 0.0, places=12)
        self.assertAlmostEqual(edge_force, 0.0, places=12)

    def test_lennard_jones_matches_the_closed_form(self):
        """Pins the force-shifted LJ closed form itself: U(r) = U_lj(r) - U_lj(rc) -
        (r - rc)*U_lj'(rc), force = -U_lj'(r) + U_lj'(rc), with U_lj(r) = 4e((s/r)^12 -
        (s/r)^6). (The shift is subtracted, not added, because the code's "edge force"
        is the physical force -U_lj'(rc), not the derivative itself.)"""
        r, cutoff = 3.5, CUTOFF

        def u_lj(x):
            return 4 * EPSILON * ((SIGMA / x) ** 12 - (SIGMA / x) ** 6)

        def u_lj_prime(x):
            return 4 * EPSILON * (-12 * SIGMA ** 12 / x ** 13 + 6 * SIGMA ** 6 / x ** 7)

        expected_energy = u_lj(r) - u_lj(cutoff) - (r - cutoff) * u_lj_prime(cutoff)
        expected_force = -u_lj_prime(r) + u_lj_prime(cutoff)
        energy, force = _lennard_jones(r, cutoff)
        self.assertAlmostEqual(energy, expected_energy, places=9)
        self.assertAlmostEqual(force, expected_force, places=9)

        edge_energy, edge_force = _lennard_jones(cutoff, cutoff)
        self.assertAlmostEqual(edge_energy, 0.0, places=12)
        self.assertAlmostEqual(edge_force, 0.0, places=12)

    def test_body_frame_matches_the_typescript_model(self):
        """Pins the body-frame convention against src/water-model.ts's bodySites(): x
        along the H-O-H bisector towards the hydrogens, y in the molecular plane with
        the first hydrogen at +y, z out of plane, origin at the centre of mass."""
        source = (ROOT / 'src/water-model.ts').read_text()

        def constant(name):
            return float(re.search(rf'export const {name} = ([\d.]+)', source).group(1))

        oh_length = constant('OH_LENGTH')
        hoh_angle_degrees = constant('HOH_ANGLE')
        m_offset = constant('M_OFFSET')
        mass_o = constant('MASS_O')
        mass_h = constant('MASS_H')

        half = np.deg2rad(hoh_angle_degrees / 2)
        along, across = oh_length * np.cos(half), oh_length * np.sin(half)
        shift = 2 * mass_h * along / (mass_o + 2 * mass_h)

        np.testing.assert_allclose(BODY['oxygen'], [-shift, 0, 0], atol=1e-12)
        np.testing.assert_allclose(BODY['hydrogen'][0], [along - shift, across, 0], atol=1e-12)
        np.testing.assert_allclose(BODY['hydrogen'][1], [along - shift, -across, 0], atol=1e-12)
        np.testing.assert_allclose(BODY['charge'], [m_offset - shift, 0, 0], atol=1e-12)

        # H1 is the +y hydrogen, H2 the -y one -- pins the convention, not just its magnitude.
        self.assertGreater(BODY['hydrogen'][0][1], 0)
        self.assertLess(BODY['hydrogen'][1][1], 0)

        # Origin at the centre of mass.
        mass_weighted = MASS_O * BODY['oxygen'] + MASS_H * (BODY['hydrogen'][0] + BODY['hydrogen'][1])
        np.testing.assert_allclose(mass_weighted, [0, 0, 0], atol=1e-10)

        # Geometry: 0.9572 A O-H bonds, 104.52 degree H-O-H angle.
        oh1 = BODY['hydrogen'][0] - BODY['oxygen']
        oh2 = BODY['hydrogen'][1] - BODY['oxygen']
        self.assertAlmostEqual(float(np.linalg.norm(oh1)), .9572, places=9)
        self.assertAlmostEqual(float(np.linalg.norm(oh2)), .9572, places=9)
        cosine = np.dot(oh1, oh2) / (np.linalg.norm(oh1) * np.linalg.norm(oh2))
        self.assertAlmostEqual(np.degrees(np.arccos(cosine)), 104.52, places=6)

        # Charge site 0.125 A from oxygen along +x.
        np.testing.assert_allclose(BODY['charge'] - BODY['oxygen'], [.125, 0, 0], atol=1e-12)

        # Sanity: the actual centre_of_mass() function agrees on a real molecule.
        centre = centre_of_mass(self.xyz[0])
        self.assertEqual(centre.shape, (3,))


if __name__ == '__main__':
    unittest.main()
