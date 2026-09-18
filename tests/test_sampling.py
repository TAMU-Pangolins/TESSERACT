import unittest

import numpy as np

from nucres.sampling import (
    porter_thomas_factors,
    stationary_wigner_placements,
    unfolded_wigner_placements,
    wigner_surmise_spacings,
)


class SamplingTest(unittest.TestCase):
    def test_porter_thomas_mean_is_one(self):
        rng = np.random.default_rng(1234)
        samples = porter_thomas_factors(50000, mu=1.0, df=1, rng=rng)
        self.assertGreater(samples.min(), 0.0)
        self.assertAlmostEqual(float(samples.mean()), 1.0, delta=0.05)

    def test_wigner_surmise_has_unit_mean_and_repels(self):
        rng = np.random.default_rng(4321)
        samples = wigner_surmise_spacings(100000, rng=rng)
        self.assertAlmostEqual(float(samples.mean()), 1.0, delta=0.015)
        self.assertLess(float(np.mean(samples < 0.2)), 0.05)

    def test_stationary_wigner_placements_are_bounded_and_reproducible(self):
        first = stationary_wigner_placements(
            2.0, 102.0, rng=np.random.default_rng(123)
        )
        second = stationary_wigner_placements(
            2.0, 102.0, rng=np.random.default_rng(123)
        )
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(np.diff(first) > 0.0))
        self.assertTrue(np.all((first >= 2.0) & (first <= 102.0)))

    def test_stationary_wigner_placements_preserve_expected_count(self):
        rng = np.random.default_rng(2468)
        counts = [
            stationary_wigner_placements(0.0, 20.0, rng=rng).size
            for _ in range(2000)
        ]
        self.assertAlmostEqual(float(np.mean(counts)), 20.0, delta=0.2)

    def test_unfolded_wigner_placements_follow_density_support(self):
        energy = np.linspace(0.0, 10.0, 1001)
        density = np.where(energy <= 5.0, 0.0, 20.0)
        placements = unfolded_wigner_placements(
            energy, density, rng=np.random.default_rng(99)
        )
        self.assertGreater(placements.size, 50)
        self.assertTrue(np.all(placements >= 5.0))
        self.assertTrue(np.all(placements <= 10.0))


if __name__ == "__main__":
    unittest.main()
