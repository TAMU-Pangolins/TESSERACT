import unittest
from pathlib import Path

import numpy as np


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
