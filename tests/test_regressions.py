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
        lines = (repo / "input" / "22Ne(a,n)25Mg.txt").read_text().splitlines()
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


class HFBCorrectionTest(unittest.TestCase):
    def test_ripl_cor_normalisation(self):
        from common.densities_retrieval import apply_hfb_corrections, read_hfb_cor

        with tempfile.TemporaryDirectory() as d:
            tab = Path(d) / "z014.tab"
            _write_multi_isotope_tab(tab, 14, (28,), {28: 10.0})
            cor_path = Path(d) / "z014.cor"
            cor_path.write_text("  14  28   7  15     0.50000     1.00000                                    28Si\n")
            cor = read_hfb_cor(str(cor_path))
            self.assertEqual(cor, {(14, 28): (0.5, 1.0)})
            rec = apply_hfb_corrections(read_hfb_tab(str(tab), A=28), cor)
            U = rec.positive.U
            expected = np.where(U - 1.0 >= U[0], 10.0 * np.exp(0.5 * np.sqrt(np.clip(U - 1.0, 0, None))), 0.0)
            np.testing.assert_allclose(rec.positive.Rho_level, expected, rtol=1e-12)
            with self.assertRaises(KeyError):
                apply_hfb_corrections(read_hfb_tab(str(tab), A=28), {})


class ReactionNamingTest(unittest.TestCase):
    def test_file_stem(self):
        from nucres.reaction_names import file_stem, parse_reaction

        self.assertEqual(file_stem("22Mg(a,p)25Al"), "22Mg_ap_25Al")  # unchanged for (a,p)
        self.assertEqual(file_stem("24Mg(p,g)25Al"), "24Mg_pg_25Al")
        self.assertEqual(parse_reaction("17O(p,a)14N").ejectile, "a")
        with self.assertRaises(ValueError):
            parse_reaction("22Mg-a-p")

    def test_compute_gamma_any_reaction(self):
        import tesseract
        from extract_resonance_data import load_reaction_params

        repo = Path(__file__).resolve().parent.parent
        rx = load_reaction_params(repo / "input" / "22Mg(a,p)25Al.txt")
        g_i, g_o = tesseract.compute_gamma(rx, 0.01, 0.0045)
        # Wigner limit for a + 22Mg (R = 1.25 (4.003^1/3 + 22^1/3) fm) is 0.615 MeV
        self.assertAlmostEqual(g_i / 0.01 / 1e6, 0.615, delta=0.002)
        rx_g = load_reaction_params(repo / "input" / "24Mg(p,g)25Al.txt")
        _, g_o_gamma = tesseract.compute_gamma(rx_g, 0.01, 0.3)
        self.assertEqual(g_o_gamma, 0.3)  # gamma exit: mean_o is the mean gamma width in eV


class TalysFitTest(unittest.TestCase):
    def test_bin_grid_and_average(self):
        import talys_opt_with_unc as topt

        centres = np.array([1.0, 1.1, 1.2])
        E, w = topt.talys_bin_grid(centres, 0.1, 3)
        self.assertEqual(len(E), 9)
        self.assertEqual(len(np.unique(E)), 9)               # no shared edge energies
        self.assertTrue(np.all(abs(E.reshape(3, 3) - centres[:, None]) < 0.05))
        self.assertAlmostEqual(w.sum(), 1.0)
        np.testing.assert_allclose(topt.talys_bin_average(3.0 * E + 1.0, w), 3.0 * centres + 1.0)
        # convex sigma(E): the bin average exceeds the centre value
        f = lambda x: np.exp(-30.0 / np.sqrt(x))
        exact = np.array([quad_avg(f, c - 0.05, c + 0.05) for c in centres])
        np.testing.assert_allclose(topt.talys_bin_average(f(E), w), exact, rtol=1e-6)
        self.assertTrue(np.all(exact > f(centres)))

    def test_bin_width_from_header(self):
        import talys_opt_with_unc as topt

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "22Mg_ap_25Al_integrated_xs_dE_0.2_run0.csv"
            path.write_text("# E (MeV),sigma (mb)| dE=0.05\n1.0,2.0\n")
            self.assertEqual(topt.exp_bin_width(str(path), np.array([1.0]), {}), 0.05)
            path.write_text("1.0,2.0\n")
            self.assertEqual(topt.exp_bin_width(str(path), np.array([1.0]), {}), 0.2)
            self.assertEqual(topt.exp_bin_width(str(path), np.array([1.0]), {"exp_bin_width": "0.1"}), 0.1)

    def _cfg(self, d, ejectile=(1, 1), residual=(13, 25), extra=""):
        import talys_opt_with_unc as topt

        text = f"""[basics]
reaction = x
projectile_Z = 2
projectile_A = 4
target_Z = 12
target_A = 22
ejectile_Z = {ejectile[0]}
ejectile_A = {ejectile[1]}
residual_Z = {residual[0]}
residual_A = {residual[1]}
[talys]
{extra}
\\opt
rvadjust a 1.0 0.5 1.5
\\opt
"""
        path = Path(d) / "tesseract.in"
        path.write_text(text)
        return topt.load_config(str(path))

    def test_first_data_row_is_kept(self):
        import talys_opt_with_unc as topt

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.csv"
            path.write_text("# E (MeV),sigma (mb)| dE=0.1\n1.5e-01,7.3e-32\n2.5e-01,1.2e-22\n")
            np.testing.assert_allclose(topt.read_exp_table(str(path))[:, 0], [0.15, 0.25])
            path.write_text("E,sigma,err\n1.0,2.0,0.1\n")
            np.testing.assert_allclose(topt.read_exp_table(str(path)), [[1.0, 2.0, 0.1]])

    def test_channel_file_names(self):
        import talys_opt_with_unc as topt

        with tempfile.TemporaryDirectory() as d:
            cfg = self._cfg(d)
            self.assertEqual(topt.talys_xs_candidates(cfg), ["ap.tot", "xs010000.tot"])
            cfg_g = self._cfg(d, ejectile=(0, 0), residual=(14, 26))
            self.assertEqual(topt.talys_xs_candidates(cfg_g), ["ag.tot", "xs000000.tot"])
            (Path(d) / "xs010000.tot").write_text("# E xs\n 1.0 5.0\n 2.0 6.0\n")
            x, y = topt.read_ap_tot(d, cfg)
            np.testing.assert_allclose(y, [5.0, 6.0])

    def test_objective_is_chi2_plus_prior_on_bin_averages(self):
        import talys_opt_with_unc as topt

        with tempfile.TemporaryDirectory() as d:
            cfg = self._cfg(d, extra="prior_rel_std = 0.1\nexp_rel_err = 0.1")
        x = np.array([2.0, 2.2, 2.4, 2.6])
        E, w = topt.talys_bin_grid(x, 0.2, 3)
        cfg["_x_exp"], cfg["_talys_E_cm"], cfg["_bin_weights"] = x, E, w
        model = lambda p: np.exp(-30.0 / np.sqrt(E)) * p        # sigma at sub-grid points
        y_bin_true = topt.talys_bin_average(model(1.0), w)
        y_exp = y_bin_true * np.array([1.0, 1.0, 1.0, 1.2])
        y_err = y_exp * np.array([0.1, 0.1, 0.1, 0.05])           # last point: 5%
        mask = np.ones(4, dtype=bool)
        with mock.patch.object(topt, "run_talys_get_xs", lambda p, c: (E, model(p[0]))):
            obj = topt.make_objective(cfg, x, y_exp, y_err, mask)
            total = obj(np.array([1.0]))
        chi2 = (np.log(1.2) / 0.05) ** 2                          # bin-averaged model, per-point sigma
        self.assertAlmostEqual(total, chi2, places=6)              # prior = 0 at x0
        self.assertAlmostEqual(obj.state["best_parts"]["chi2_red"], chi2 / 3, places=6)


def quad_avg(f, a, b):
    from scipy.integrate import quad
    return quad(f, a, b, epsabs=0, epsrel=1e-12)[0] / (b - a)


def _opt_block(input_file, run, dE, chi2, param):
    return (
        "=" * 72 + "\n"
        f"Date/Time   : 2026-10-01 00:00:00\nInput file  : {input_file}\n"
        f"Reaction    : 22Mg(α,p)25Al\nBin width   : dE = {dE} MeV\nRun index   : {run}\n"
        + "-" * 72 + "\nOptimizer   : Powell  (converged=True, nfev=10)\n"
        f"Min reduced chi-square: {chi2}\n" + "-" * 72 + "\nOptimized parameters:\n"
        f"  rvadjust_a = {param:+.6g}\n" + "-" * 72 + "\nCross sections  E[MeV]  sigma[mb]:\n"
        "  2.0e+00  1.0e-03\n" + "-" * 72 + "\nReaction rates  T9[GK]  rate[cm3/s/mol]:\n"
        + "".join(f"  {t:.6e}  {1e-5 * t ** 3:.6e}\n" for t in (0.5, 1.0, 1.5, 2.0, 3.0))
        + "=" * 72 + "\n\n"
    )


class FitRecordDedupTest(unittest.TestCase):
    def test_reruns_are_counted_once(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        import plot_talys_param_grid as grid
        import plot_rate_ratio_wide as ratio
        import plot_talys_results_new as results

        a = "out/22Mg_ap_25Al_integrated_xs_dE_0.2_run0.csv"
        b = "out/22Mg_ap_25Al_integrated_xs_dE_0.5_run0.csv"
        text = (_opt_block(a, 0, 0.2, 5.0, 1.1) + _opt_block(b, 0, 0.5, 4.0, 1.2)
                + _opt_block(a, 0, 0.2, 3.0, 1.3))      # re-run of fit a
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "talys_optimization.out"
            path.write_text(text)
            g = grid.parse_talys_optimization(path)
            self.assertEqual(len(g), 2)
            self.assertEqual([r["params"]["rvadjust_a"] for r in g], [1.3, 1.2])  # newest kept
            self.assertEqual(len(ratio.parse_talys_rate_records(path)), 2)
            self.assertEqual(len(results.parse_talys_opt_out(str(path))), 2)


class CheckpointTest(unittest.TestCase):
    def test_entries_per_fit_in_one_file(self):
        import json
        import talys_opt_with_unc as topt

        names = ["rvadjust_a"]
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "checkpoint.json")
            entry = lambda x: {"params": [x], "param_names": names, "loss": 1.0, "nfev": 3, "timestamp": "t"}
            topt.checkpoint_save(path, "fit_dE_0.2", entry(1.1))
            topt.checkpoint_save(path, "fit_dE_0.5", entry(1.2))
            self.assertEqual(topt.checkpoint_load(path, "fit_dE_0.2", names)["params"], [1.1])
            self.assertEqual(topt.checkpoint_load(path, "fit_dE_0.5", names)["params"], [1.2])
            self.assertIsNone(topt.checkpoint_load(path, "fit_dE_0.5", ["other"]))  # \\opt changed
            topt.checkpoint_clear(path, "fit_dE_0.2")
            self.assertIsNone(topt.checkpoint_load(path, "fit_dE_0.2", names))
            self.assertEqual(sorted(json.loads(Path(path).read_text())), ["fit_dE_0.5"])
            topt.checkpoint_clear(path, "fit_dE_0.5")
            self.assertFalse(Path(path).exists())                 # removed when empty

    def test_old_single_fit_checkpoint_is_not_resumed(self):
        import json
        import talys_opt_with_unc as topt

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "checkpoint.json"
            path.write_text(json.dumps({"params": [1.4], "loss": 2.0, "nfev": 5, "timestamp": "t"}))
            self.assertIsNone(topt.checkpoint_load(str(path), "fit_dE_0.2", ["rvadjust_a"]))


_LEV_25AL = """  13  25    7    5                                                          25Al
   0   0.000000   2.5    1  0                   7.183E+00                 5/2+
   1   0.451700   0.5    1  1                   2.290E-09                 1/2+
                               0  1.000000 0.000E+00
   2   0.944900   1.5    1  0                   4.300E-12                 3/2+
   3   1.612500   3.5    1  0                   1.200E-14               (7/2)+
   4   1.789500   2.5    1  0                   3.930E-13                 5/2+
   5   5.045000   1.5   -1  0                   0.000E+00 JP
"""


class FinalStatesTest(unittest.TestCase):
    def _levels_dir(self, d):
        (Path(d) / "Al.lev").write_text(_LEV_25AL)
        return d

    def test_level_file_and_cutoff(self):
        from nucres.final_states import discrete_final_states, read_levels

        with tempfile.TemporaryDirectory() as d:
            self._levels_dir(d)
            read_levels.cache_clear()
            lv = read_levels(d, 13, 25)
            self.assertEqual([round(x.E, 4) for x in lv], [0.0, 0.4517, 0.9449, 1.6125, 1.7895, 5.045])
            self.assertEqual((lv[1].J, lv[1].parity, lv[1].measured), (0.5, 1, True))
            self.assertFalse(lv[5].measured)
            # no .cor: complete up to the last measured level from the ground state
            keep, E_cut = discrete_final_states(d, 13, 25, None)
            self.assertEqual((len(keep), E_cut), (5, 1.7895))
            cor = Path(d) / "z013.cor"
            cor.write_text("  13  25   2   2     0.00000    -0.07947                                    25Al\n")
            keep, E_cut = discrete_final_states(d, 13, 25, str(cor))
            self.assertEqual((len(keep), E_cut), (3, 0.9449))

    def test_gamma_multipoles_and_strengths(self):
        from nucres.final_states import gamma_multipole, strength

        self.assertEqual(gamma_multipole(1, -1, 0, 1), "E1")
        self.assertEqual(gamma_multipole(1, 1, 0, 1), "M1")
        self.assertEqual(gamma_multipole(2, 1, 0, 1), "E2")
        self.assertIsNone(gamma_multipole(0, 1, 0, 1))      # no 0 -> 0
        self.assertIsNone(gamma_multipole(3, 1, 1, -1))     # M2: beyond the model
        Eg = np.linspace(1, 30, 300)
        fe1 = strength("E1", Eg, 28, 14)
        self.assertTrue(np.all(fe1 > 0))
        self.assertTrue(15.0 < Eg[np.argmax(fe1 * Eg ** 3)] < 25.0)  # GDR region
        self.assertEqual(len(set(np.round(strength("M1", Eg, 28, 14), 20))), 1)

    def test_summed_particle_width_mean(self):
        from nucres.final_states import Continuum, FinalStateModel, Level

        levels = [Level(0.0, 2.5, 1, True), Level(0.4517, 0.5, 1, True)]
        J = np.array([0.5, 1.5, 2.5])
        cont = Continuum(U=np.array([1.0]), J=J,
                         N_pos=np.array([[0.0, 4.0, 0.0]]), N_neg=np.zeros((1, 3)))
        P = {l: (lambda E, l=l: np.full_like(np.asarray(E, dtype=float), 10.0 ** -(l + 2))) for l in range(9)}
        m = FinalStateModel("particle", 13, 25, levels, cont, S_proj=9.166, S_exit=5.514,
                            ejectile_spin=0.5, ejectile_parity=1, pen=lambda l: P[l],
                            gamma2_mean_eV=100.0)
        # 2+ resonance: l=0 to 5/2+ (S=2,3), l=2 to 1/2+, l=0 to 3/2+ (continuum, 4 states)
        expect = 2 * 100.0 * (1e-2 + 1e-4 + 4 * 1e-2)
        rng = np.random.default_rng(0)
        draws = [m.exit_width(1.0, 2.0, 1, rng).Gamma_eV for _ in range(20000)]
        self.assertAlmostEqual(np.mean(draws) / expect, 1.0, delta=0.03)
        # a state that is energetically closed contributes nothing
        far = FinalStateModel("particle", 13, 25, [Level(9.0, 0.5, 1, True)], Continuum(np.zeros(0), J, np.zeros((0, 3)), np.zeros((0, 3))),
                              S_proj=9.166, S_exit=5.514, pen=lambda l: P[l], gamma2_mean_eV=100.0)
        self.assertEqual(far.exit_width(1.0, 2.0, 1, rng).Gamma_eV, 0.0)


if __name__ == "__main__":
    unittest.main()
