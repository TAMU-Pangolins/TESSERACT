import unittest

import numpy as np

from nucres.sampling import porter_thomas_factors


class SamplingTest(unittest.TestCase):
    def test_porter_thomas_mean_is_one(self):
        rng = np.random.default_rng(1234)
        samples = porter_thomas_factors(50000, df=1, rng=rng)
        self.assertGreater(samples.min(), 0.0)
        self.assertAlmostEqual(float(samples.mean()), 1.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
