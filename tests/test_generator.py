import unittest
from pathlib import Path

from nucres.generator import HFBSamplerConfig, synthesize_sigma_from_hfb


DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "densities" / "level-densities-hfb"


class HFBSynthesisTest(unittest.TestCase):
    def test_generate_sigma_basic(self):
        cfg = HFBSamplerConfig(
            Z=9,
            data_root=DATA_ROOT,
            A=18,
            J=1.0,
            pi=1,
            Gamma_i_mean_eV=0.5,
            Gamma_o_mean_eV=0.5,
            delta_E_mev=0.05,
            E_min_mev=0.2,
            E_max_mev=0.6,
            n_density_points=201,
            n_sigma_points=256,
            U_offset_mev=8.0,
            seed=42,
        )
        spectrum = synthesize_sigma_from_hfb(cfg)

        self.assertEqual(spectrum.energy_MeV.shape[0], cfg.n_sigma_points)
        self.assertEqual(spectrum.sigma_barns.shape[0], cfg.n_sigma_points)
        self.assertEqual(len(spectrum.resonances), spectrum.metadata["n_drawn"])
        self.assertGreaterEqual(spectrum.metadata["n_bins"], 1)
        self.assertEqual(spectrum.metadata["spacing_model"], "poisson")

        rate = spectrum.compute_rate([0.1], temperature_unit="GK")
        self.assertEqual(rate.na_sigma_v.shape[0], 1)

    def test_generate_wigner_spaced_sigma_basic(self):
        cfg = HFBSamplerConfig(
            Z=9,
            data_root=DATA_ROOT,
            A=18,
            J=1.0,
            pi=1,
            Gamma_i_mean_eV=0.5,
            Gamma_o_mean_eV=0.5,
            delta_E_mev=0.05,
            E_min_mev=0.2,
            E_max_mev=5.0,
            n_density_points=201,
            n_sigma_points=128,
            U_offset_mev=8.0,
            seed=42,
            spacing_model="wigner",
        )
        spectrum = synthesize_sigma_from_hfb(cfg)

        self.assertEqual(spectrum.metadata["spacing_model"], "wigner")
        self.assertEqual(len(spectrum.resonances), spectrum.metadata["n_drawn"])
        self.assertGreaterEqual(spectrum.metadata["n_spacing_ladders"], 1)
        self.assertTrue(
            all(
                cfg.E_min_mev <= resonance.E_r * 1e-6 <= cfg.E_max_mev
                for resonance in spectrum.resonances
            )
        )


if __name__ == "__main__":
    unittest.main()
