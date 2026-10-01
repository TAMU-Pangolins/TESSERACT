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


class ParitySelectionTest(unittest.TestCase):
    def test_alpha_on_0plus_target_forbids_unnatural_parity(self):
        from nucres.generator import _allowed_L_values

        for J in range(6):
            natural = (-1) ** J
            self.assertEqual(_allowed_L_values(J, 0.0, 0.0, natural, 1), [J])
            self.assertEqual(_allowed_L_values(J, 0.0, 0.0, -natural, 1), [])

    def test_negative_parity_target(self):
        from nucres.generator import _allowed_L_values

        # 15N (1/2-) + alpha: a 1/2+ resonance needs l = 1, a 1/2- one l = 0.
        self.assertEqual(_allowed_L_values(0.5, 0.0, 0.5, +1, -1), [1])
        self.assertEqual(_allowed_L_values(0.5, 0.0, 0.5, -1, -1), [0])

    def test_generator_samples_both_parities_and_drops_forbidden(self):
        from nucres.generator import HFBSamplerConfig, synthesize_sigma_from_hfb

        data_root = (
            Path(__file__).resolve().parent.parent
            / "data" / "densities" / "level-densities-hfb"
        )
        if not (data_root / "z010.tab").exists():
            self.skipTest("HFB tables not available.")
        # 16O(a,g)20Ne: compound 20Ne, alpha + 0+ target.
        cfg = HFBSamplerConfig(
            Z=10, A=20, data_root=data_root, s1=0.0, s2=0.0,
            E_min_mev=0.5, E_max_mev=6.0, U_offset_mev=4.73, seed=1,
            n_density_points=400, n_sigma_points=64,
        )
        spec = synthesize_sigma_from_hfb(cfg)
        parities = {r.parity for r in spec.resonances}
        self.assertEqual(parities, {+1, -1})
        for r in spec.resonances:
            self.assertEqual(r.parity, (-1) ** int(r.J))
            self.assertEqual(r.L1, int(r.J))
        self.assertIn((1.0, +1), spec.metadata["dropped_Jpi"])
        self.assertGreater(spec.metadata["expected_levels_dropped"], 0.0)


class NubaseLookupTest(unittest.TestCase):
    def test_ground_state_jpi(self):
        from nucres.read_qvals import ground_state_jpi

        self.assertEqual(ground_state_jpi(2, 4), (0.0, 1))
        self.assertEqual(ground_state_jpi(7, 15), (0.5, -1))
        self.assertEqual(ground_state_jpi(13, 27), (2.5, 1))


class TemplateHandlingTest(unittest.TestCase):
    def test_mass_number_is_rounded_not_truncated(self):
        from build_ratesmc_input import _mass_number_from_token

        self.assertEqual(_mass_number_from_token("26.9815"), 27)
        self.assertEqual(_mass_number_from_token("4.0026"), 4)
        self.assertEqual(_mass_number_from_token("22"), 22)

    def test_upper_limit_rows_are_cleared(self):
        from build_ratesmc_input import _clear_upper_limits

        repo = Path(__file__).resolve().parent.parent
        lines = (repo / "input" / "22Mg(a,p)25Al.txt").read_text().splitlines()
        n_before = len(lines)
        removed = _clear_upper_limits(lines)
        self.assertEqual(removed, 19)
        start = next(i for i, ln in enumerate(lines) if ln.startswith("Upper Limits"))
        header = next(j for j in range(start, len(lines)) if lines[j].startswith("Ecm"))
        self.assertTrue(lines[header + 1].startswith("!"))
        self.assertTrue(lines[header + 2].startswith("*"))
        self.assertEqual(len(lines), n_before - 19 + 1)
        self.assertTrue(any(ln.startswith("Interference") for ln in lines))


class RatesMCStepTest(unittest.TestCase):
    FAKE = """#!/bin/sh
# Mimics RatesMC: refuses arguments, reads ./RatesMC.in, needs ./mass_1.mas20,
# writes ./RatesMC.out and exits 1 even on success (as the 2.2+ rewrite does).
[ "$#" -eq 0 ] || { echo "Custom filenames not yet implemented!"; exit 0; }
[ -f RatesMC.in ] && [ -f mass_1.mas20 ] || exit 2
head -1 RatesMC.in > RatesMC.out
echo " T9      RRate_low       Median Rate     RRate_high     f.u." >> RatesMC.out
echo " 0.010   1.0e-90   1.0e-90   1.0e-90   1.0e+00" >> RatesMC.out
echo " 0.020   1.0e-60   1.0e-60   1.0e-60   1.0e+00" >> RatesMC.out
exit 1
"""

    def test_runs_in_run_dir_without_arguments_and_ignores_exit_code(self):
        import os
        import tesseract

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            install = d / "RatesMC-src"
            (install / "build").mkdir(parents=True)
            fake = install / "build" / "RatesMC"
            fake.write_text(self.FAKE)
            os.chmod(fake, 0o755)
            for name in ("mass_1.mas20", "nubase_3.mas20"):
                (install / name).write_text("table\n")

            reaction = "22Mg(a,p)25Al"
            run_dir = d / "out" / reaction / "RUN_0"
            run_dir.mkdir(parents=True)
            infile = run_dir / f"{reaction}.in"
            infile.write_text(f"{reaction}\nbody\n")

            tesseract.step_run_ratesmc(
                {"reaction": reaction},
                {"output_dir": str(d / "out"), "runs": "1"},
                {"ratesmc_bin": str(fake)},
            )

            out = run_dir / f"{reaction}.out"
            self.assertTrue(out.exists())
            self.assertEqual(tesseract._ratesmc_rate_rows(out), 2)
            self.assertEqual(infile.read_text(), f"{reaction}\nbody\n")  # input untouched
            self.assertTrue((run_dir / "mass_1.mas20").is_symlink())


class SpinExportTest(unittest.TestCase):
    def test_half_integer_spins_are_not_rounded(self):
        from nucres.physics import MASS_PROTON
        from nucres.ratesmc_export import RatesMCExportOptions, render_rows
        from nucres.resonance import Resonance

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


class ReducedWidthSpectrumTest(unittest.TestCase):
    def test_reduced_width_spectrum_has_no_cross_section(self):
        from nucres.generator import HFBSamplerConfig, synthesize_sigma_from_hfb

        data_root = (
            Path(__file__).resolve().parent.parent
            / "data" / "densities" / "level-densities-hfb"
        )
        if not (data_root / "z010.tab").exists():
            self.skipTest("HFB tables not available.")
        cfg = HFBSamplerConfig(
            Z=10, A=20, data_root=data_root, s1=0.0, s2=0.0,
            E_min_mev=0.5, E_max_mev=3.0, U_offset_mev=4.73, seed=1,
            n_density_points=200, n_sigma_points=32, widths_are_reduced=True,
        )
        spec = synthesize_sigma_from_hfb(cfg)
        self.assertIsNone(spec.sigma_barns)
        self.assertGreater(len(spec.resonances), 0)
        with self.assertRaises(ValueError):
            spec.compute_rate([1.0])


class BinAverageTest(unittest.TestCase):
    def test_constant_cross_section_averages_to_itself(self):
        from generate_cross_sections_vectorized import bin_average

        E = np.linspace(0.1, 10.0, 10000)
        dE_grid = E[1] - E[0]
        for dE in (0.05, 0.15, 0.25):  # rounding of points/bin differs per dE
            ppb = int(round(dE / dE_grid))
            _, avg = bin_average(E, np.full_like(E, 2.0), ppb)
            np.testing.assert_allclose(avg, 2.0, rtol=1e-12)

    def test_linear_cross_section_averages_to_value_at_centre(self):
        from generate_cross_sections_vectorized import bin_average

        E = np.linspace(0.0, 1.0, 1001)
        centres, avg = bin_average(E, 3.0 * E, 100)
        np.testing.assert_allclose(avg, 3.0 * centres, rtol=1e-12)


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
