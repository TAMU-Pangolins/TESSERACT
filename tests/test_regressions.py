import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from common.densities_retrieval import read_hfb_tab
from nucres.lab_to_cm import cm_to_lab, lab_to_cm
from nucres.resonance import make_penetrability_interp, penetrability_P_l_mev


def _write_multi_isotope_tab(path: Path, Z: int, A_values, rho_by_A) -> None:
    """Write a minimal RIPL-3-style zXXX.tab holding several isotopes."""
    lines = []
    for A in A_values:
        for label in ("Positive-Parity", "Negative-parity"):
            lines.append(" " * 20 + "*" * 40)
            lines.append(
                " " * 20
                + f"*  Z={Z:3d} A={A:3d}: {label} Spin-dependent Level Density [MeV-1]  *"
            )
            lines.append(" " * 20 + "*" * 40)
            lines.append(" U[MeV]  T[MeV]  NCUMUL   RHOOBS   RHOTOT     J=00")
            for k in range(60):
                vals = [rho_by_A[A]] * 53
                lines.append(
                    f"{0.25 * (k + 1):7.2f}{0.5:7.3f} "
                    + " ".join(f"{v:8.2E}" for v in vals)
                )
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class HFBIsotopeSelectionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tab = Path(self.tmp.name) / "z014.tab"
        _write_multi_isotope_tab(self.tab, 14, (22, 28), {22: 13.0, 28: 150.0})

    def tearDown(self):
        self.tmp.cleanup()

    def test_selects_requested_isotope(self):
        rec = read_hfb_tab(str(self.tab), A=28)
        self.assertEqual((rec.header.Z, rec.header.A), (14, 28))
        self.assertTrue(np.allclose(rec.positive.Rho_level, 150.0))
        self.assertTrue(np.allclose(rec.negative.Rho_level, 150.0))
        self.assertEqual(rec.positive.U.shape[0], 60)

    def test_missing_A_is_an_error_for_multi_isotope_file(self):
        with self.assertRaises(ValueError):
            read_hfb_tab(str(self.tab))

    def test_unknown_A_is_an_error(self):
        with self.assertRaises(ValueError):
            read_hfb_tab(str(self.tab), A=30)


class PenetrabilityInterpolationTest(unittest.TestCase):
    def test_sub_coulomb_alpha_interpolation_is_accurate(self):
        # alpha + 22Mg on the grid tesseract.py uses (0.1-10 MeV, 600 points).
        # Linear-in-P interpolation was 25-50% high here.
        for l in (0, 2):
            P = make_penetrability_interp(l, 2, 12, 4, 22, Emin_mev=0.1, Emax_mev=10.0, npts=600)
            for E in (0.105, 0.3083, 0.5083, 1.2083):
                exact = penetrability_P_l_mev(l, 2, 12, 4, 22, E)
                self.assertAlmostEqual(float(P(E)) / exact, 1.0, delta=0.01)

    def test_nonpositive_energy_gives_zero(self):
        P = make_penetrability_interp(0, 2, 12, 4, 22, Emin_mev=0.1, Emax_mev=10.0, npts=50)
        self.assertTrue(np.all(P(np.array([0.0, -1.0])) == 0.0))


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


class PerRunSeedTest(unittest.TestCase):
    def test_each_run_gets_a_distinct_seed(self):
        import tesseract

        repo = Path(__file__).resolve().parent.parent
        captured = []

        def fake_run(cmd, *args, **kwargs):
            captured.append(cmd)
            return mock.Mock(returncode=0)

        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(tesseract, "_PYTHON", "python"), \
                mock.patch.object(tesseract.subprocess, "run", side_effect=fake_run):
            tesseract.step1_build_ratesmc(
                {"reaction": "22Mg(a,p)25Al"},
                {"input_dir": str(repo / "input"), "output_dir": d, "runs": "3", "seed": "100"},
            )

        seeds = [cmd[cmd.index("--seed") + 1] for cmd in captured]
        self.assertEqual(seeds, ["100", "101", "102"])


if __name__ == "__main__":
    unittest.main()
