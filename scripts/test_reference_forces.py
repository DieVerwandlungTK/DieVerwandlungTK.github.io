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
