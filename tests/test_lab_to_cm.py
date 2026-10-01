import tempfile
import unittest
from pathlib import Path

import numpy as np

from nucres.lab_to_cm import cm_to_lab, lab_to_cm


class LabToCMTest(unittest.TestCase):
    def test_alpha_on_22Ne(self):
        # m(22Ne)/(m(22Ne)+m(4He)) = 0.84602
        self.assertAlmostEqual(lab_to_cm(5.0, 22, "Ne", 4, "He"), 4.2301, places=3)

    def test_round_trip(self):
        e_cm = lab_to_cm(3.0, 22, "Mg", 4, "He")
        self.assertAlmostEqual(cm_to_lab(e_cm, 22, "Mg", 4, "He"), 3.0, places=12)


class TalysEnergyFrameTest(unittest.TestCase):
    CONFIG = """[basics]
reaction = 22Mg(a,p)25Al
projectile_Z = 2
projectile_A = 4
target_Z = 12
target_A = 22
ejectile_Z = 1
ejectile_A = 1
residual_Z = 13
residual_A = 25
[talys]
{extra}
\\opt
rvadjust a 1.0 0.8 1.2
\\opt
"""

    def _energies_written(self, extra=""):
        import talys_opt_with_unc as topt

        with tempfile.TemporaryDirectory() as d:
            cfg_path = Path(d) / "tesseract.in"
            cfg_path.write_text(self.CONFIG.format(extra=extra), encoding="utf-8")
            cfg = topt.load_config(str(cfg_path))
            cfg["_x_exp"] = np.array([2.0, 3.0])
            topt.write_talys_files(d, np.array([1.0]), cfg)
            energies = np.loadtxt(Path(d) / "energies.txt")
            back = topt._ap_tot_in_exp_frame((energies, np.ones(2)), cfg)[0]
        return energies, back

    def test_cm_exp_energies_are_converted_to_lab_for_talys(self):
        energies, back = self._energies_written()
        expected = np.array([2.0, 3.0]) * cm_to_lab(1.0, 22, "Mg", 4, "He")
        np.testing.assert_allclose(energies, expected, rtol=1e-9)
        np.testing.assert_allclose(back, [2.0, 3.0], rtol=1e-9)

    def test_lab_frame_option_passes_energies_through(self):
        energies, _ = self._energies_written(extra="exp_energy_frame = lab")
        np.testing.assert_allclose(energies, [2.0, 3.0], rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
