import unittest

from nucres.physics import MASS_PROTON
from nucres.ratesmc_export import RatesMCExportOptions, render_rows, resonance_to_row
from nucres.resonance import Resonance


class SpinExportTest(unittest.TestCase):
    def test_half_integer_spins_are_not_rounded(self):
        spins = (0.0, 0.5, 1.0, 1.5, 2.5, 3.0, 3.5)
        res = [
            Resonance(E_r=5e5, J=J, s1=0.5, s2=0.0, m1=MASS_PROTON, m2=MASS_PROTON,
                      Gamma_i=1.0, Gamma_o=1.0)
            for J in spins
        ]
        rows, _ = render_rows(res, RatesMCExportOptions())
        written = [row.split()[4] for row in rows]
        self.assertEqual(written, ["0", "0.5", "1", "1.5", "2.5", "3", "3.5"])
        self.assertEqual([float(w) for w in written], list(spins))


class ExitEnergyUncertaintyTest(unittest.TestCase):
    def _row(self, E_r_keV, Exf_keV, exit_q_keV):
        res = Resonance(E_r=E_r_keV * 1e3, J=2.0, s1=0.0, s2=0.0, m1=4.0, m2=22.0,
                        Gamma_i=1.0, Gamma_o=1.0, Exf_keV=Exf_keV)
        opts = RatesMCExportOptions(default_frac_unc=1e-3, exit_q_keV=exit_q_keV)
        return resonance_to_row(res, opts)

    def test_decm_keeps_a_near_threshold_exit_open(self):
        # 22Ne(a,n): Q = -478.30 keV; a final state 0.417 keV below E_r + Q.
        row = self._row(3280.177, 2801.46, -478.30)
        exit_keV = row.Ecm - 478.30 - row.Exf
        self.assertAlmostEqual(exit_keV, 0.417, places=6)
        self.assertLessEqual(row.DEcm * 10.0, exit_keV + 1e-12)

    def test_decm_is_the_fractional_uncertainty_far_from_threshold(self):
        row = self._row(3280.177, 0.0, -478.30)
        self.assertAlmostEqual(row.DEcm, 3.280177)
        self.assertAlmostEqual(self._row(3280.177, 0.0, None).DEcm, 3.280177)


if __name__ == "__main__":
    unittest.main()
