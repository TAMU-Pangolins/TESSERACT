"""Wigner-limit convention of the mean reduced widths in tesseract.py."""

import unittest

import tesseract


class WignerLimitTest(unittest.TestCase):
    def test_matches_the_ratesmc_and_pogrebnyak_convention(self):
        # RatesMC: Gamma = 2 * 41.80 * P * theta^2 / (mu R^2), mu in u, R in fm,
        # Gamma in MeV, i.e. gamma_W^2 = hbar^2 / (mu R^2) (Pogrebnyak eq. 1).
        m1, m2, r0 = 4.0026, 23.985, 1.25
        mu = m1 * m2 / (m1 + m2)
        R = r0 * (m1 ** (1 / 3) + m2 ** (1 / 3))
        expected_eV = 41.80 / (mu * R ** 2) * 1e6
        self.assertAlmostEqual(tesseract._wigner_limit_ev(m1, m2, r0) / expected_eV, 1.0, places=3)

    def test_compute_gamma_scales_both_channels(self):
        class Rxn:
            M_proj, M_targ, M_exit, Z_exit, R0_fm = 4.0026, 21.991, 1.0087, 0, 1.25
        gi, go = tesseract.compute_gamma(Rxn, 0.018, 0.0045)
        self.assertAlmostEqual(gi, 0.018 * tesseract._wigner_limit_ev(4.0026, 21.991, 1.25))
        m_res = 4.0026 + 21.991 - 1.0087
        self.assertAlmostEqual(go, 0.0045 * tesseract._wigner_limit_ev(1.0087, m_res, 1.25))


if __name__ == "__main__":
    unittest.main()


class ClusterPlumbingTest(unittest.TestCase):
    def _cmd(self, resonance):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        calls = []

        class Done:
            returncode = 0

        with tempfile.TemporaryDirectory() as d, \
                patch.object(tesseract.subprocess, "run", lambda cmd: calls.append(cmd) or Done()), \
                patch.object(tesseract, "_merge_stage_run_meta", lambda *a, **k: None):
            root = Path(__file__).resolve().parent.parent
            tesseract.step1_build_ratesmc(
                {"reaction": "22Ne(a,n)25Mg"},
                {"input_dir": str(root / "input"), "output_dir": d, "seed": "5", **resonance},
            )
        return calls[0]

    def test_cluster_options_reach_build_ratesmc_input(self):
        cmd = self._cmd({"cluster_density": "2", "cluster_l_weights": "1, 1, 0.3",
                         "cluster_sigma_ln": "0.8"})
        rxn = tesseract.load_reaction_params(
            tesseract.Path(__file__).resolve().parent.parent / "input" / "22Ne(a,n)25Mg.txt")
        w = float(cmd[cmd.index("--cluster-wigner-eV") + 1])
        self.assertAlmostEqual(w, tesseract._wigner_limit_ev(rxn.M_proj, rxn.M_targ, rxn.R0_fm))
        self.assertEqual(cmd[cmd.index("--cluster-density") + 1], "2")
        i = cmd.index("--cluster-l-weights")
        self.assertEqual(cmd[i + 1:i + 4], ["1", "1", "0.3"])
        self.assertEqual(cmd[cmd.index("--cluster-sigma-ln") + 1], "0.8")

    def test_cluster_options_are_off_by_default(self):
        cmd = self._cmd({})
        self.assertFalse(any(str(a).startswith("--cluster") for a in cmd))
