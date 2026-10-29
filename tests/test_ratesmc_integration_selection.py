"""Verify dimensionless per-resonance integration selection and fixed defaults."""

import unittest

class IntWidthCutoffTest(unittest.TestCase):
    def test_nonfinite_and_negative_cutoffs_are_rejected(self):
        from nucres.ratesmc_export import RatesMCExportOptions
        for cutoff in [-0.1, float('nan'), float('inf')]:
            with self.subTest(cutoff=cutoff), self.assertRaisesRegex(ValueError, 'cutoff'):
                RatesMCExportOptions(int_width_cutoff=cutoff)

    def test_exact_cutoff_uses_analytical_integration(self):
        self.assertEqual(self._row(9999.0, int_width_cutoff=0.01).Int, 0)

    def _row(self, Gamma_o, **opts):
        from nucres.ratesmc_export import RatesMCExportOptions, resonance_to_row
        from nucres.resonance import Resonance

        res = Resonance(E_r=1.0e6, J=1.0, s1=0.0, s2=0.0, m1=1.0, m2=1.0,
                        Gamma_i=1.0, Gamma_o=Gamma_o)
        return resonance_to_row(res, RatesMCExportOptions(**opts))

    def test_flag_set_by_total_width_over_energy(self):
        # E_r = 1 MeV; Gamma_tot/E_r = (1 eV + Gamma_o) / 1e6 eV
        self.assertEqual(self._row(2.0e4, int_width_cutoff=0.01).Int, 1)   # ~0.02 > 0.01
        self.assertEqual(self._row(5.0e3, int_width_cutoff=0.01).Int, 0)   # ~0.005 < 0.01

    def test_cutoff_overrides_int_flag_and_default_is_unchanged(self):
        self.assertEqual(self._row(5.0e3, int_flag=1, int_width_cutoff=0.01).Int, 0)
        self.assertEqual(self._row(5.0e3, int_flag=1).Int, 1)
        self.assertEqual(self._row(2.0e4, int_flag=0).Int, 0)
