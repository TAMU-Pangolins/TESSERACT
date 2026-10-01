import unittest

from nucres.physics import MASS_PROTON
from nucres.ratesmc_export import RatesMCExportOptions, render_rows
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


if __name__ == "__main__":
    unittest.main()
