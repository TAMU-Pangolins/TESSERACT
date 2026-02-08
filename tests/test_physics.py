import unittest

import numpy as np

from nucres.physics import energy_grid, spin_stat_factor


class PhysicsHelpersTest(unittest.TestCase):
    def test_energy_grid_bounds(self):
        grid = energy_grid(center_eV=100.0, half_width_eV=10.0, n=5)
        self.assertEqual(grid.shape[0], 5)
        self.assertAlmostEqual(grid[0], 90.0)
        self.assertAlmostEqual(grid[-1], 110.0)
        self.assertTrue(np.all(np.diff(grid) > 0))

    def test_spin_stat_factor(self):
        s = spin_stat_factor(J=1.0, s1=0.5, s2=0.5)
        self.assertAlmostEqual(s, 3.0 / 4.0)


if __name__ == "__main__":
    unittest.main()
