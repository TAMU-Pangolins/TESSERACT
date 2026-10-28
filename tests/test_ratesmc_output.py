import tempfile
import unittest
from pathlib import Path

import numpy as np

RATESMC_211_OUT = """22Mg(a,p)25Al
Samples = 1
 T9     RRate_low       Classical Rate  Median Rate     Mean Rate       RRate_high      Log-Normal mu      Log-Normal sigma A-D Statistic
 0.0100   1.000e-87       9.000e-87       2.000e-87       0.000e+00       3.000e-87       0.0000e+00        0.0000e+00       0.000e+00
 1.0000   1.000e-06       9.000e-06       2.000e-06       0.000e+00       3.000e-06       0.0000e+00        0.0000e+00       0.000e+00
"""

RATESMC_23_OUT = """22Mg(a,p)25Al
Calculated with RatesMC 2.3.0 (git hash:3649f24baa) on Wed Sep 30 22:51:06 2026
Samples = 1
 T9      RRate_low       Median Rate     RRate_high     f.u.
 0.010   1.000e-87       2.000e-87       3.000e-87        1.000e+00
 1.000   1.000e-06       2.000e-06       3.000e-06        1.500e+00
"""


class RatesMCOutputReaderTest(unittest.TestCase):
    def _write(self, d, rel, text):
        path = Path(d) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def test_both_lineages_give_the_same_columns(self):
        from nucres.ratesmc_output import read_ratesmc_out

        with tempfile.TemporaryDirectory() as d:
            for text, has_fu in ((RATESMC_211_OUT, False), (RATESMC_23_OUT, True)):
                table = read_ratesmc_out(self._write(d, "x.out", text))
                np.testing.assert_allclose(table["T9"], [0.01, 1.0])
                np.testing.assert_allclose(table["low"], [1e-87, 1e-6])
                np.testing.assert_allclose(table["median"], [2e-87, 2e-6])
                np.testing.assert_allclose(table["high"], [3e-87, 3e-6])
                self.assertEqual("fu" in table, has_fu)

    def test_finds_both_run_layouts(self):
        from nucres.ratesmc_output import find_ratesmc_outputs

        with tempfile.TemporaryDirectory() as d:
            rxn = Path(d) / "22Mg(a,p)25Al"
            self._write(rxn, "RUN_0/22Mg(a,p)25Al.out", RATESMC_23_OUT)
            self._write(rxn, "RUN_0/RatesMC.out", RATESMC_23_OUT)  # same run: not twice
            self._write(rxn, "RUN_10/RatesMC.out", RATESMC_23_OUT)
            self._write(rxn, "Run_02/RatesMC.out", RATESMC_211_OUT)
            self._write(rxn, "notes/RatesMC.out", RATESMC_211_OUT)
            found = find_ratesmc_outputs(rxn)
            self.assertEqual([name for name, _ in found], ["RUN_0", "Run_02", "RUN_10"])
            self.assertEqual(found[0][1].name, "22Mg(a,p)25Al.out")


if __name__ == "__main__":
    unittest.main()
