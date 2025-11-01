import sys
import unittest
from pathlib import Path

import numpy as np

from nucres.model import HFBRateRequest, generate_spectrum, compute_rate_table


class ModelAPITest(unittest.TestCase):
    def setUp(self):
        self.data_root = Path(__file__).resolve().parent.parent / "data" / "densities" / "level-densities-hfb"
        if not self.data_root.exists():
            self.skipTest("Density tables not available for model tests.")

    def test_generate_spectrum_and_rates(self):
        request = HFBRateRequest(
            Z=9,
            A=18,
            J=1.0,
            pi=1,
            Gamma_i_mean_eV=0.5,
            Gamma_o_mean_eV=0.5,
            E_min_mev=0.2,
            E_max_mev=0.6,
            n_sigma_points=128,
            n_density_points=201,
            data_root=self.data_root,
            seed=123,
        )

        spectrum = generate_spectrum(request)
        self.assertEqual(spectrum.energy_MeV.shape[0], request.n_sigma_points)
        self.assertEqual(spectrum.sigma_barns.shape[0], request.n_sigma_points)

        temps = [0.1, 0.2, 0.3]
        rate = compute_rate_table(request, temps)
        self.assertEqual(len(rate.temperature), len(temps))
        self.assertTrue(np.all(np.asarray(rate.na_sigma_v) >= 0.0))


if __name__ == "__main__":
    unittest.main()
