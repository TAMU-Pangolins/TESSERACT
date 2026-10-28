import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import mpmath as mp
import numpy as np

from nucres.resonance_sum import ReactionParams, ResonanceRow, ResonanceSum


class TransmissionUnderflowTest(unittest.TestCase):
    def setUp(self):
        self.reaction = ReactionParams(
            Z_proj=2, Z_targ=12, Z_exit=2, M_proj=4.0, M_targ=22.0, M_exit=4.0,
            J_proj=0.0, J_targ=0.0, S_proj_mev=9.16, S_exit_mev=9.16,
            R0_fm=1.25, gamma_channel=3,
        )
        self.row = ResonanceRow(
            E_r=0.001, J=20, G0=1.0, L0=20, G1=0.2, L1=20,
            G2=0.0, L2=1, Exf=0.0,
        )

    def _direct_P(self, E):
        # Preserve mpmath values throughout this independent reference.
        mu = 4.0 * 22.0 / 26.0
        radius = 1.25 * (4.0 ** (1 / 3) + 22.0 ** (1 / 3))
        rho = 0.218735 * radius * (mu * E) ** 0.5
        eta = 0.157489 * 24 * (mu / E) ** 0.5
        return rho / (mp.coulombf(20, eta, rho) ** 2 + mp.coulombg(20, eta, rho) ** 2)

    def test_coulomb_entrance_and_exit_underflow_match_mpmath(self):
        from nucres.resonance import make_penetrability_interp

        energies = np.array([0.00099, self.row.E_r, 0.00101])
        direct_r = self._direct_P(self.row.E_r)
        self.assertGreater(direct_r, 0)
        self.assertEqual(float(direct_r), 0.0)
        P = make_penetrability_interp(20, 2, 12, 4, 22,
                                      Emin_mev=0.0005, Emax_mev=0.00165, npts=64)
        self.assertEqual(float(P(self.row.E_r)), 0.0)
        rs = ResonanceSum(self.reaction, [self.row], 0.001, 0.0011, npts=64)
        expected = []
        for E in energies:
            ratio = self._direct_P(float(E)) / direct_r
            g0, g1 = self.row.G0 * ratio * 1e-6, self.row.G1 * ratio * 1e-6
            bw = g0 * g1 / ((E - self.row.E_r) ** 2 + (g0 + g1) ** 2 / 4)
            expected.append(float((2 * self.row.J + 1) * rs.pi_lambda2_b_mev / E * bw))
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            actual = rs.sigma(energies)
        np.testing.assert_allclose(actual, expected, rtol=1e-3)
        self.assertAlmostEqual(float(rs.sigma(self.row.E_r)), expected[1], places=7)

    def test_jwkb_underflow_and_extreme_ratios_remain_finite(self):
        # Log transmissions span 1600 at the query points: even forming
        # exp(logT(E) - logT(Er)) would overflow at the high-energy point.
        def logT(E):
            return -1000.0 + (np.asarray(E) - self.row.E_r) * 8e6

        rx = replace(self.reaction, gamma_channel=2, Z_exit=0, M_exit=0.0)
        with patch("nucres.resonance_sum.make_jwkb_log_transmission_interp", return_value=logT):
            rs = ResonanceSum(rx, [self.row], 0.0009, 0.0011,
                              penetrability_model="jwkb_real_omp")
            with np.errstate(divide="raise", invalid="raise", over="raise"):
                result = rs.sigma(np.array([-1.0, 0.0, 0.0009, 0.001, 0.0011]))
                integrals = rs.bin_integrals([0.0009, 0.001, 0.0011], n_base=32, n_peak=32)
                rates = rs.rate([0.01, 0.1], E_lo=0.0009, E_hi=0.0011,
                                n_base=32, n_peak=32)
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertTrue(np.all(result >= 0))
        np.testing.assert_array_equal(result[:2], 0.0)
        peak_bw = 4 * self.row.G0 * self.row.G1 / (self.row.G0 + self.row.G1) ** 2
        peak = (2 * self.row.J + 1) * rs.pi_lambda2_b_mev / self.row.E_r * peak_bw
        self.assertAlmostEqual(result[3] / peak, 1.0, places=10)
        self.assertTrue(np.all(np.isfinite(integrals)) and np.all(integrals > 0))
        self.assertTrue(np.all(np.isfinite(rates)) and np.all(rates > 0))

    def test_real_jwkb_underflow_preserves_resonance_peak(self):
        rx = replace(self.reaction, gamma_channel=2, Z_exit=0, M_exit=0.0)
        rs = ResonanceSum(rx, [self.row], 0.0009, 0.0011,
                          penetrability_model="jwkb_real_omp", jwkb_npts=8)
        self.assertLess(float(rs._entrance_log(20)(self.row.E_r)), -745.0)
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            peak = float(rs.sigma(self.row.E_r))
        expected_bw = 4 * self.row.G0 * self.row.G1 / (self.row.G0 + self.row.G1) ** 2
        expected = (2 * self.row.J + 1) * rs.pi_lambda2_b_mev / self.row.E_r * expected_bw
        self.assertAlmostEqual(peak / expected, 1.0, places=7)

    def test_particle_width_overflow_matches_mpmath(self):
        def logP(E):
            return -1000.0 + (np.asarray(E) - self.row.E_r) * 8e6

        rx = replace(self.reaction, gamma_channel=0)
        row = replace(self.row, G2=0.1, L2=20)
        energies = np.array([0.0009, row.E_r, 0.0011])
        with patch("nucres.resonance_sum.make_log_penetrability_interp", return_value=logP):
            rs = ResonanceSum(rx, [row], 0.0009, 0.0011)
            with np.errstate(divide="raise", invalid="raise", over="raise"):
                actual = rs.sigma(energies)
        expected = []
        for E in energies:
            scale = mp.exp(float(logP(E) - logP(row.E_r))) * 1e-6
            g0, g1, g2 = row.G0 * scale, row.G1 * scale, row.G2 * scale
            bw = g0 * g1 / ((E - row.E_r) ** 2 + (g0 + g1 + g2) ** 2 / 4)
            expected.append(float((2 * row.J + 1) * rs.pi_lambda2_b_mev / E * bw))
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=0.0)

    def test_exit_and_spectator_thresholds(self):
        rx = replace(self.reaction, S_exit_mev=10.16, gamma_channel=0)
        row = replace(self.row, E_r=2.0, G2=0.1)
        rs = ResonanceSum(rx, [row], 2.1, 3.0, npts=16)
        energies = np.array([0.5, 1.0, row.E_r])
        for channel in (1, 2):
            widths = rs._log_scaled(channel, 0.2, 20, 0.0, energies, row.E_r)
            self.assertTrue(np.all(np.isneginf(widths[:2])))
            self.assertAlmostEqual(widths[2], np.log(0.2), places=10)

    def test_nonzero_width_at_closed_reference_is_an_error(self):
        for gamma_channel in (2, 3):
            rx = replace(self.reaction, gamma_channel=gamma_channel,
                         S_proj_mev=0.0, S_exit_mev=0.002)
            rs = ResonanceSum(rx, [self.row], 0.0005, 0.0011)
            with self.assertRaisesRegex(ValueError, "closed channel"):
                rs._log_scaled(1, 0.2, 1, 0.002, np.array([0.003]), self.row.E_r)

class ReactionNamesTest(unittest.TestCase):
    def test_file_stem_matches_old_ap_convention(self):
        from nucres.reaction_names import file_stem

        self.assertEqual(file_stem("22Mg(a,p)25Al"), "22Mg_ap_25Al")

    def test_file_stem_is_reaction_agnostic(self):
        from nucres.reaction_names import file_stem

        self.assertEqual(file_stem("24Mg(p,g)25Al"), "24Mg_pg_25Al")


class ExtractDataNonResonantTest(unittest.TestCase):
    def test_non_resonant_contribution_is_not_mistaken_for_resonant(self):
        from extract_resonance_data import extract_data

        repo = Path(__file__).resolve().parent.parent
        template = repo / "input" / "16O(a,g)20Ne.txt"
        self.assertIn("Non-Resonant Contribution", template.read_text())
        df = extract_data(str(template))
        self.assertGreater(len(df), 0)
        self.assertIn("Ecm", df.columns)


class StepTwoIntegrationTest(unittest.TestCase):
    """Step 2 bin averages are exact per-resonance integrals of RatesMC's integrand."""

    def _sum(self, rows, E_lo=0.5, E_hi=3.0):
        from nucres.resonance_sum import ReactionParams, ResonanceSum

        # p + 24Mg -> 25Al + gamma, numbers as in input/24Mg(p,g)25Al.txt
        rxn = ReactionParams(
            Z_proj=1, Z_targ=12, Z_exit=0, M_proj=1.0078, M_targ=23.985, M_exit=0.0,
            J_proj=0.5, J_targ=0.0, S_proj_mev=2.2716, S_exit_mev=0.0, R0_fm=1.25,
            gamma_channel=2,
        )
        return rxn, ResonanceSum(rxn, rows, E_lo, E_hi, npts=1500)

    def test_bin_edges_are_exactly_dE_wide(self):
        from generate_cross_sections_vectorized import bin_edges

        edges = bin_edges(0.1, 10.0, 0.05)
        np.testing.assert_allclose(np.diff(edges), 0.05, rtol=1e-12)
        self.assertAlmostEqual(edges[0], 0.1)
        self.assertLessEqual(edges[-1], 10.0 + 1e-12)
        self.assertEqual(len(edges) - 1, 198)

    def test_narrow_resonance_area_is_captured_exactly(self):
        from nucres.resonance_sum import ResonanceRow

        # 1.5 MeV, Gamma ~ 1 eV: far narrower than any uniform grid.
        row = ResonanceRow(E_r=1.5, J=1.5, G0=1.0, L0=1, G1=0.2, L1=1, G2=0.0, L2=0, Exf=0.0)
        rxn, rs = self._sum([row])
        edges = np.array([0.5, 1.45, 1.55, 3.0])
        integ = rs.bin_integrals(edges)
        # narrow-resonance area: 2 pi^2 lambda-bar^2 omega gamma (b MeV)
        omega_gamma = (2 * 1.5 + 1) / 2 * (1.0 * 0.2 / 1.2) * 1e-6
        area = 2 * np.pi * rs.pi_lambda2_b_mev / row.E_r * omega_gamma
        self.assertAlmostEqual(integ[1] / area, 1.0, delta=2e-3)
        self.assertLess(integ[0] + integ[2], 1e-3 * integ[1])  # wings are negligible here

    def test_bin_integrals_are_additive(self):
        from nucres.resonance_sum import ResonanceRow

        rows = [ResonanceRow(E_r=e, J=0.5, G0=g, L0=0, G1=0.1, L1=1, G2=0.0, L2=0, Exf=0.0)
                for e, g in ((0.9, 50.0), (1.21, 3e3), (2.3, 0.4))]
        _, rs = self._sum(rows)
        fine = rs.bin_integrals(np.linspace(0.5, 3.0, 51))
        coarse = rs.bin_integrals(np.array([0.5, 3.0]))
        self.assertAlmostEqual(fine.sum() / coarse[0], 1.0, delta=1e-3)

    def test_rate_matches_narrow_resonance_formula(self):
        from nucres.resonance_sum import ResonanceRow

        row = ResonanceRow(E_r=1.5, J=1.5, G0=1.0, L0=1, G1=0.2, L1=1, G2=0.0, L2=0, Exf=0.0)
        rxn, rs = self._sum([row])
        T9 = np.array([1.0, 2.0, 5.0])
        omega_gamma = (2 * 1.5 + 1) / 2 * (1.0 * 0.2 / 1.2) * 1e-6
        narrow = 1.5394e11 * (rxn.mu_u * T9) ** -1.5 * omega_gamma * np.exp(-11.605 * 1.5 / T9)
        np.testing.assert_allclose(rs.rate(T9), narrow, rtol=5e-3)


if __name__ == "__main__":
    unittest.main()
