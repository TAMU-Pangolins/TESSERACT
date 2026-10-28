import unittest
from pathlib import Path

from nucres.generator import (
    HFBSamplerConfig,
    _allowed_L_values,
    _exit_L_values,
    _selected_parities,
    synthesize_sigma_from_hfb,
)


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


class ReducedWidthSpectrumTest(unittest.TestCase):
    def test_reduced_width_spectrum_has_no_cross_section(self):
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
            widths_are_reduced=True,
        )
        spectrum = synthesize_sigma_from_hfb(cfg)
        self.assertIsNone(spectrum.sigma_barns)
        self.assertGreater(len(spectrum.resonances), 0)
        self.assertTrue(spectrum.metadata["widths_are_reduced"])
        with self.assertRaises(ValueError):
            spectrum.compute_rate([0.1])


class HfbCorrectionsConfigTest(unittest.TestCase):
    """z009.cor has a real (nonzero ptable) entry for A=17 (9F: see test_hfb_corrections.py)."""

    def _cfg(self, **overrides):
        kwargs = dict(
            Z=9, data_root=DATA_ROOT, A=17, J=0.5, pi=1, s1=0.5, s2=0.0,
            Gamma_i_mean_eV=0.5, Gamma_o_mean_eV=0.5, delta_E_mev=0.2,
            E_min_mev=0.5, E_max_mev=3.0, n_density_points=201,
            n_sigma_points=64, U_offset_mev=4.0, seed=1, sample_J=False,
        )
        kwargs.update(overrides)
        return HFBSamplerConfig(**kwargs)

    def test_metadata_reports_corrections_applied(self):
        spec_on = synthesize_sigma_from_hfb(self._cfg(use_hfb_corrections=True))
        spec_off = synthesize_sigma_from_hfb(self._cfg(use_hfb_corrections=False))
        self.assertTrue(spec_on.metadata["hfb_corrections_applied"])
        self.assertFalse(spec_off.metadata["hfb_corrections_applied"])

    def test_corrections_change_the_level_density_grid(self):
        import numpy as np

        spec_on = synthesize_sigma_from_hfb(self._cfg(use_hfb_corrections=True))
        spec_off = synthesize_sigma_from_hfb(self._cfg(use_hfb_corrections=False))
        self.assertGreater(
            np.max(np.abs(spec_on.rho_levels_per_MeV - spec_off.rho_levels_per_MeV)), 0.0
        )


class SelectedParitiesTest(unittest.TestCase):
    def test_zero_means_both(self):
        self.assertEqual(_selected_parities(0), (+1, -1))

    def test_explicit_single_parity(self):
        self.assertEqual(_selected_parities(1), (1,))
        self.assertEqual(_selected_parities(-1), (-1,))

    def test_invalid_value_raises(self):
        with self.assertRaises(ValueError):
            _selected_parities(2)


class AllowedLValuesIntrinsicParityTest(unittest.TestCase):
    """alpha (0+) on a 0+ target: I=0 only, so L=J exactly and parity = (-1)^L
    is forced -- the textbook case from the commit this was ported from."""

    def test_natural_parity_states_are_allowed(self):
        for J in (0.0, 2.0, 4.0, 6.0):
            self.assertEqual(_allowed_L_values(J, 0.0, 0.0, +1, 1), [int(J)])
        for J in (1.0, 3.0, 5.0):
            self.assertEqual(_allowed_L_values(J, 0.0, 0.0, -1, 1), [int(J)])

    def test_unnatural_parity_states_are_forbidden(self):
        for J in (1.0, 3.0, 5.0):
            self.assertEqual(_allowed_L_values(J, 0.0, 0.0, +1, 1), [])
        for J in (0.0, 2.0, 4.0):
            self.assertEqual(_allowed_L_values(J, 0.0, 0.0, -1, 1), [])

    def test_negative_intrinsic_parity_flips_natural_parity(self):
        # pi_intrinsic=-1 (e.g. a negative-parity target): now ODD J is
        # natural for pi_res=+1, matching wanted = pi_res * pi_intrinsic.
        self.assertEqual(_allowed_L_values(1.0, 0.0, 0.0, +1, -1), [1])
        self.assertEqual(_allowed_L_values(2.0, 0.0, 0.0, +1, -1), [])


class ExitLValuesTest(unittest.TestCase):
    def test_gamma_excludes_monopole(self):
        # 0 -> 0 gamma: L_min = max(1, 0) = 1, L_max = 0 -> no allowed L.
        self.assertEqual(_exit_L_values(0.0, +1, "gamma", 0.0, 1, 0.5, 1), [])

    def test_gamma_multipolarity_matches_triangle_rule(self):
        # J=0.5 -> J_f=2.5: L_min=max(1,2)=2, L_max=3.
        self.assertEqual(_exit_L_values(0.5, +1, "gamma", 2.5, 1, 0.5, 1), [2, 3])

    def test_particle_exit_uses_ejectile_and_final_parity(self):
        # proton (s_e=1/2, +) to a 5/2+ final state, from a 2+ resonance:
        # S = |1/2-5/2|..1/2+5/2 = 2..3; wanted = pi_res*pi_e*pi_f = +1*1*1=+1
        # -> even L only.
        allowed = _exit_L_values(2.0, +1, "particle", 2.5, +1, 0.5, 1)
        self.assertTrue(all(l % 2 == 0 for l in allowed))
        self.assertTrue(len(allowed) > 0)

    def test_invalid_exit_kind_raises(self):
        with self.assertRaises(ValueError):
            _exit_L_values(1.0, +1, "neutrino", 1.0, 1)


class DropForbiddenEndToEndTest(unittest.TestCase):
    """24Mg (Z=12, A=24): alpha (0+) on a 0+-like fixed J=0 test slice."""

    def _cfg(self, **overrides):
        kwargs = dict(
            Z=12, data_root=DATA_ROOT, A=24, J=0.0, pi=1, s1=0.0, s2=0.0,
            projectile_parity=1, target_parity=1,
            Gamma_i_mean_eV=0.5, Gamma_o_mean_eV=0.5, delta_E_mev=0.2,
            E_min_mev=0.5, E_max_mev=3.0, n_density_points=201,
            n_sigma_points=64, U_offset_mev=4.0, seed=1, sample_J=True,
        )
        kwargs.update(overrides)
        return HFBSamplerConfig(**kwargs)

    def test_unnatural_parity_sequences_are_dropped_by_default(self):
        spec = synthesize_sigma_from_hfb(self._cfg(pi=1))
        dropped_J = {J for J, p in spec.metadata["dropped_Jpi"]}
        self.assertIn(1.0, dropped_J)  # odd J at pi_res=+1 is unnatural here
        self.assertTrue(all(r.L1 is not None for r in spec.resonances))
        self.assertTrue(all(J % 2 == 0 for J in (r.J for r in spec.resonances)))

    def test_keep_forbidden_restores_old_behaviour(self):
        spec_drop = synthesize_sigma_from_hfb(self._cfg(pi=1, drop_forbidden=True))
        spec_keep = synthesize_sigma_from_hfb(self._cfg(pi=1, drop_forbidden=False))
        self.assertGreater(
            spec_keep.metadata["expected_levels"], spec_drop.metadata["expected_levels"]
        )

    def test_both_parities_sampled_by_default(self):
        cfg = self._cfg()
        self.assertEqual(cfg.pi, 1)  # this test's own fixture still pins pi=1
        default_cfg = HFBSamplerConfig(Z=12, data_root=DATA_ROOT, A=24)
        self.assertEqual(default_cfg.pi, 0)


class ExitChannelL2Test(unittest.TestCase):
    def _cfg(self, **overrides):
        kwargs = dict(
            Z=12, data_root=DATA_ROOT, A=24, s1=0.0, s2=0.0,
            projectile_parity=1, target_parity=1,
            Gamma_i_mean_eV=0.5, Gamma_o_mean_eV=0.5, delta_E_mev=0.2,
            E_min_mev=0.5, E_max_mev=3.0, n_density_points=201,
            n_sigma_points=64, U_offset_mev=4.0, seed=1, sample_J=True,
            exit_kind="particle", ejectile_spin=0.0, ejectile_parity=1,
            final_spin=0.0, final_parity=1,
        )
        kwargs.update(overrides)
        return HFBSamplerConfig(**kwargs)

    def test_l2_is_set_when_exit_channel_is_configured(self):
        spec = synthesize_sigma_from_hfb(self._cfg())
        self.assertGreater(len(spec.resonances), 0)
        self.assertTrue(all(r.L2 is not None for r in spec.resonances))

    def test_l2_unset_without_exit_kind(self):
        spec = synthesize_sigma_from_hfb(self._cfg(exit_kind=None))
        self.assertTrue(all(r.L2 is None for r in spec.resonances))

    def test_gamma_zero_to_zero_drops_every_resonance(self):
        # 0+ -> 0+ gamma has no allowed multipolarity (no monopole radiation).
        # Pin J=0 (sample_J=False) so every entrance-allowed resonance is a
        # 0+ state that must then be dropped on exit; with sample_J=True
        # other J values would have allowed gamma transitions to J_f=0.
        spec = synthesize_sigma_from_hfb(
            self._cfg(exit_kind="gamma", final_spin=0.0, final_parity=1,
                      sample_J=False, J=0.0)
        )
        self.assertEqual(len(spec.resonances), 0)
        self.assertGreater(spec.metadata["n_exit_forbidden"], 0)


if __name__ == "__main__":
    unittest.main()
