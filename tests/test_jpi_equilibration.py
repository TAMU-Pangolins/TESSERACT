"""Parity-equilibrated spin-cutoff J^pi split of the HFB level densities."""

import unittest
from pathlib import Path

import numpy as np

from common.densities_retrieval import (
    apply_hfb_corrections,
    apply_level_density_jpi,
    equilibrate_jpi,
    fermi_gas_a,
    read_hfb_cor,
    read_hfb_tab,
    spin_cutoff_sigma2,
    spin_distribution,
    spin_grid,
)
from nucres.final_states import hfb_continuum
from nucres.generator import HFBSamplerConfig, synthesize_sigma_from_hfb
from nucres.hfb_adapter import load_hfb_record, resolve_density_paths

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "densities" / "level-densities-hfb"


def _record(Z, A, corrected=True):
    tab, cor = resolve_density_paths(Z, data_root=DATA_ROOT)
    rec = read_hfb_tab(str(tab), A=A)
    if corrected:
        rec = apply_hfb_corrections(rec, read_hfb_cor(str(cor)))
    return rec


def _total(rec):
    return rec.positive.rho_J.sum(axis=1) + rec.negative.rho_J.sum(axis=1)


class SystematicsTest(unittest.TestCase):
    def test_sigma2_formula(self):
        # 28Si: a = 0.0722396*28 + 0.195267*28^(2/3), U* = U - 2*12/sqrt(28).
        a = 0.0722396 * 28 + 0.195267 * 28 ** (2 / 3)
        self.assertAlmostEqual(fermi_gas_a(28), a)
        U = 10.0
        Ueff = U - 24.0 / np.sqrt(28.0)
        expected = 0.0146 * 28 ** (5 / 3) * (1 + np.sqrt(1 + 4 * a * Ueff)) / (2 * a)
        self.assertAlmostEqual(float(spin_cutoff_sigma2(U, 14, 28)), expected)
        # Below the pairing shift U* is clamped to 0.
        self.assertAlmostEqual(float(spin_cutoff_sigma2(0.0, 14, 28)),
                               0.0146 * 28 ** (5 / 3) / a)

    def test_spin_distribution_is_normalised(self):
        for J in (spin_grid(28), spin_grid(27)):
            P = spin_distribution(J, np.array([2.0, 5.0, 9.0]))
            np.testing.assert_allclose(P.sum(axis=1), 1.0)


class EquilibrateRecordTest(unittest.TestCase):
    def setUp(self):
        if not (DATA_ROOT / "z014.tab").exists():
            self.skipTest("HFB tables not available.")

    def test_total_kept_parities_equal_J_follows_P(self):
        for Z, A in ((14, 28), (12, 25)):  # even and odd A
            rec = _record(Z, A)
            eq = equilibrate_jpi(rec)
            np.testing.assert_allclose(_total(eq), _total(rec), rtol=1e-12)
            np.testing.assert_array_equal(eq.positive.rho_J, eq.negative.rho_J)
            s2 = spin_cutoff_sigma2(rec.positive.U, Z, A)
            tot = _total(eq)
            ok = tot > 0
            np.testing.assert_allclose(
                eq.positive.rho_J[ok] / (0.5 * tot[ok, None]),
                spin_distribution(spin_grid(A), s2[ok]), rtol=1e-10, atol=1e-300)
            np.testing.assert_allclose(eq.positive.Rho_level, 0.5 * tot)

    def test_sigma2_override(self):
        rec = _record(14, 28)
        eq = equilibrate_jpi(rec, sigma2=3.0)
        tot = _total(eq)
        k = int(np.argmax(tot))
        np.testing.assert_allclose(eq.positive.rho_J[k] / (0.5 * tot[k]),
                                   spin_distribution(spin_grid(28), 3.0))
        self.assertFalse(np.allclose(eq.positive.rho_J, equilibrate_jpi(rec).positive.rho_J))

    def test_hfb_model_is_the_identity_and_unknown_model_fails(self):
        rec = _record(14, 28)
        self.assertIs(apply_level_density_jpi(rec, "hfb"), rec)
        with self.assertRaises(ValueError):
            apply_level_density_jpi(rec, "uniform")

    def test_load_hfb_record_default_unchanged(self):
        tab, cor = resolve_density_paths(14, data_root=DATA_ROOT)
        raw = read_hfb_tab(str(tab), A=28)
        loaded = load_hfb_record(str(tab), str(cor), A=28, warn_if_ignored=False)
        np.testing.assert_array_equal(loaded.positive.rho_J, raw.positive.rho_J)
        np.testing.assert_array_equal(loaded.negative.rho_J, raw.negative.rho_J)
        eq = load_hfb_record(str(tab), str(cor), A=28, use_corrections=True,
                             level_density_jpi="equilibrated")
        np.testing.assert_allclose(_total(eq), _total(_record(14, 28)))

    def test_continuum_keeps_the_total(self):
        rec = _record(12, 25)
        a = hfb_continuum(rec, 25, 2.0, 8.0, 0.1)
        b = hfb_continuum(equilibrate_jpi(rec), 25, 2.0, 8.0, 0.1)
        np.testing.assert_allclose(b.N_pos.sum(axis=1) + b.N_neg.sum(axis=1),
                                   a.N_pos.sum(axis=1) + a.N_neg.sum(axis=1), rtol=1e-10)
        np.testing.assert_allclose(b.N_pos, b.N_neg)


class GeneratorTest(unittest.TestCase):
    """alpha + 24Mg -> 28Si; drop_forbidden off so every J^pi is counted."""

    def setUp(self):
        if not (DATA_ROOT / "z014.tab").exists():
            self.skipTest("HFB tables not available.")

    def _cfg(self, **overrides):
        kw = dict(Z=14, data_root=DATA_ROOT, A=28, pi=0, s1=0.0, s2=0.0,
                  Gamma_i_mean_eV=1.0, Gamma_o_mean_eV=1.0, delta_E_mev=0.05,
                  E_min_mev=0.05, E_max_mev=3.0, n_density_points=601, n_sigma_points=16,
                  U_offset_mev=9.984, seed=3, widths_are_reduced=True,
                  use_hfb_corrections=True, drop_forbidden=False)
        kw.update(overrides)
        return HFBSamplerConfig(**kw)

    @staticmethod
    def _sig(spec):
        return [(r.E_r, r.J, r.parity, r.Gamma_i, r.Gamma_o) for r in spec.resonances]

    def test_default_is_hfb_and_unchanged(self):
        self.assertEqual(HFBSamplerConfig(Z=14).level_density_jpi, "hfb")
        self.assertIsNone(HFBSamplerConfig(Z=14).spin_cutoff_sigma2)
        a = synthesize_sigma_from_hfb(self._cfg())
        b = synthesize_sigma_from_hfb(self._cfg(level_density_jpi="hfb"))
        self.assertEqual(self._sig(a), self._sig(b))
        self.assertEqual(a.metadata["level_density_jpi"], "hfb")

    def test_equilibrated_keeps_the_count_and_composes_with_corrections(self):
        for extra in ({}, {"hfb_correction": (0.0, 2.0)},
                      {"use_hfb_corrections": False}):
            hfb = synthesize_sigma_from_hfb(self._cfg(**extra))
            eq = synthesize_sigma_from_hfb(self._cfg(level_density_jpi="equilibrated", **extra))
            self.assertAlmostEqual(eq.metadata["expected_levels"] / hfb.metadata["expected_levels"],
                                   1.0, places=6)
            np.testing.assert_allclose(eq.rho_levels_per_MeV, hfb.rho_levels_per_MeV, rtol=1e-9)
            for J in (0.0, 1.0, 3.0):
                np.testing.assert_allclose(eq.channel_rho[(J, 1)], eq.channel_rho[(J, -1)])

    def test_channel_densities_follow_P_and_override(self):
        eq = synthesize_sigma_from_hfb(self._cfg(level_density_jpi="equilibrated",
                                                 spin_cutoff_sigma2=4.0))
        total = eq.rho_levels_per_MeV
        P = spin_distribution(spin_grid(28), 4.0)
        k = len(total) // 2
        for J in range(6):
            self.assertAlmostEqual(eq.channel_rho[(float(J), 1)][k] / (0.5 * total[k]), P[J])
        self.assertEqual(eq.metadata["spin_cutoff_sigma2"][0]["systematic_or_override"], 4.0)

    def test_sampled_levels_have_equal_parities(self):
        n = {+1: 0, -1: 0}
        for seed in range(10):
            spec = synthesize_sigma_from_hfb(self._cfg(level_density_jpi="equilibrated", seed=seed))
            for r in spec.resonances:
                n[r.parity] += 1
        tot = n[1] + n[-1]
        self.assertLess(abs(n[1] - n[-1]), 4 * np.sqrt(tot))

    def test_clusters_compose(self):
        spec = synthesize_sigma_from_hfb(self._cfg(
            level_density_jpi="equilibrated", drop_forbidden=True,
            cluster_density=2.0, cluster_wigner_eV=1e6, cluster_l_weights=(1, 1)))
        st = spec.metadata["cluster_states"]
        self.assertGreater(st["n_requested"], 0)
        self.assertEqual(st["n_unplaced"], 0)


class PlumbingTest(unittest.TestCase):
    def test_tesseract_passes_the_options(self):
        import tempfile
        from unittest.mock import patch

        import tesseract

        calls = []

        class Done:
            returncode = 0

        with tempfile.TemporaryDirectory() as d, \
                patch.object(tesseract.subprocess, "run", lambda cmd: calls.append(cmd) or Done()), \
                patch.object(tesseract, "_merge_stage_run_meta", lambda *a, **k: None):
            root = Path(__file__).resolve().parent.parent
            tesseract.step1_build_ratesmc(
                {"reaction": "22Ne(a,n)25Mg"},
                {"input_dir": str(root / "input"), "output_dir": d, "seed": "5",
                 "level_density_jpi": "equilibrated", "spin_cutoff_sigma2": "4.5"})
        cmd = calls[0]
        self.assertEqual(cmd[cmd.index("--level-density-jpi") + 1], "equilibrated")
        self.assertEqual(cmd[cmd.index("--spin-cutoff-sigma2") + 1], "4.5")


if __name__ == "__main__":
    unittest.main()
