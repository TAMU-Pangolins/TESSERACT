import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from build_ratesmc_input import (
    _clear_upper_limits,
    _convert_reduced_to_partial_widths,
    _exit_channel_settings,
    _infer_compound_nucleus,
    _mass_number_from_token,
    _parse_template_metadata,
    _resolve_intrinsic_parity,
    build_ratesmc_input,
    inject_rows,
    projectile_separation_energy_mev,
)
from nucres.resonance import Resonance, penetrability_P_l_mev


TEMPLATE_TEXT = """22Mg(a,p)25Al
12    ! Ztarget
2     ! Zproj
0.0   ! Jproj
0.0   ! Jtarget
4     ! Aproj
22    ! Atarget
****************************************************************************************************************
Resonant Contribution
Ecm     DEcm    wg      Dwg     J     G1        DG1        L1    G2      DG2      L2   G3  DG3  L3  Exf   Int
****************************************************************************************************************
Upper Limits of Resonances
****************************************************************************************************************
1000 ! Number of random samples
"""


class BuildRatesMCInputTest(unittest.TestCase):
    def test_mass_number_from_token_rounds_not_truncates(self):
        # Templates store Aproj/Atarget as AME atomic masses in u (e.g.
        # 23.985 for 24Mg), which are almost always just under the true
        # integer mass number. Truncating instead of rounding silently
        # picked A-1 for the large majority of real templates.
        self.assertEqual(_mass_number_from_token("23.985"), 24)
        self.assertEqual(_mass_number_from_token("1.0078"), 1)
        self.assertEqual(_mass_number_from_token("26.9815"), 27)

    def test_infer_compound_nucleus_uses_rounded_mass_numbers(self):
        # 24Mg(p,g)25Al: target 24Mg (Z=12, Atoken=23.985) + proton
        # (Z=1, Atoken=1.0078) must give the compound nucleus 25Al.
        lines = [
            "24Mg(p,g)25Al",
            "12    ! Ztarget",
            "1     ! Zproj",
            "0.5   ! Jproj",
            "0.0   ! Jtarget",
            "1.0078 ! Aproj",
            "23.985 ! Atarget",
            "*" * 50,
            "Resonant Contribution",
            "Ecm     DEcm    wg      Dwg     J     G1        DG1        L1    G2      DG2      L2   G3  DG3  L3  Exf   Int",
            "*" * 50,
            "Upper Limits of Resonances",
            "*" * 50,
            "1000 ! Number of random samples",
        ]
        meta = _parse_template_metadata(lines)
        self.assertEqual(_infer_compound_nucleus(meta), (13, 25))

    def test_projectile_separation_energy_uses_compound_mass(self):
        offset = projectile_separation_energy_mev(
            z_proj=2,
            a_proj=4,
            z_targ=12,
            a_targ=22,
        )
        self.assertAlmostEqual(offset, 9.16593287, places=8)

    def test_build_ratesmc_input_infers_u_offset_when_omitted(self):
        captured = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            template = Path(tmpdir) / "RatesMC.in"
            output = Path(tmpdir) / "out.in"
            template.write_text(TEMPLATE_TEXT, encoding="utf-8")

            args = Namespace(
                template=template,
                output=output,
                output_dir=None,
                use_strength=False,
                default_frac_unc=0.001,
                l1=0,
                l2=1,
                l3=0,
                exf_kev=0.0,
                int_flag=1,
                include_g3=False,
                j_proj=None,
                j_targ=None,
                precision=3,
                n_random_samples=1,
                Z=None,
                data_root=None,
                A=None,
                J=1.0,
                pi=1,
                s1=0.0,
                s2=0.0,
                m1=1.0,
                m2=1.0,
                Gamma_i_mean_eV=1.0,
                Gamma_o_mean_eV=1.0,
                delta_E_mev=0.05,
                spacing_model="poisson",
                E_min_mev=0.1,
                E_max_mev=2.0,
                n_density_points=101,
                n_sigma_points=128,
                U_offset_mev=None,
                seed=123,
                sample_J=False,
                auto_l1=False,
            )

            def fake_synthesize(cfg):
                captured["U_offset_mev"] = cfg.U_offset_mev
                captured["Z"] = cfg.Z
                captured["A"] = cfg.A
                return Namespace(resonances=[], metadata={})

            with patch("build_ratesmc_input.synthesize_sigma_from_hfb", side_effect=fake_synthesize):
                build_ratesmc_input(args)

        self.assertAlmostEqual(captured["U_offset_mev"], 9.16593287, places=8)
        self.assertEqual(captured["Z"], 14)
        self.assertEqual(captured["A"], 26)

    def test_build_ratesmc_input_passes_partial_widths_through(self):
        resonance = Resonance(
            E_r=1.0e6,
            J=1.0,
            s1=0.0,
            s2=0.0,
            m1=1.0,
            m2=1.0,
            Gamma_i=3.25,
            Gamma_o=7.5,
            L1=0,
            L2=1,
        )
        captured = {}

        def fake_render_rows(resonances, opts):
            captured["Gamma_i"] = resonances[0].Gamma_i
            captured["Gamma_o"] = resonances[0].Gamma_o
            return ["row"], ["G1", "G2"]

        with tempfile.TemporaryDirectory() as tmpdir:
            template = Path(tmpdir) / "RatesMC.in"
            output = Path(tmpdir) / "out.in"
            template.write_text(TEMPLATE_TEXT, encoding="utf-8")

            args = Namespace(
                template=template,
                output=output,
                output_dir=None,
                use_strength=False,
                default_frac_unc=0.001,
                l1=0,
                l2=1,
                l3=0,
                exf_kev=0.0,
                int_flag=1,
                include_g3=False,
                j_proj=None,
                j_targ=None,
                precision=3,
                n_random_samples=1,
                Z=None,
                data_root=None,
                A=None,
                J=1.0,
                pi=1,
                s1=0.0,
                s2=0.0,
                m1=1.0,
                m2=1.0,
                Gamma_i_mean_eV=1.0,
                Gamma_o_mean_eV=1.0,
                delta_E_mev=0.05,
                spacing_model="poisson",
                E_min_mev=0.1,
                E_max_mev=2.0,
                n_density_points=101,
                n_sigma_points=128,
                U_offset_mev=8.0,
                seed=123,
                sample_J=False,
                auto_l1=False,
            )

            with patch("build_ratesmc_input.synthesize_sigma_from_hfb") as synth_mock:
                synth_mock.return_value = Namespace(resonances=[resonance], metadata={})
                with patch(
                    "build_ratesmc_input._convert_reduced_to_partial_widths",
                    side_effect=lambda resonances, *_, **__: resonances,
                ):
                    with patch("build_ratesmc_input.render_rows", side_effect=fake_render_rows):
                        with patch("build_ratesmc_input.resonant_header_line", return_value="header"):
                            build_ratesmc_input(args)

        self.assertEqual(captured["Gamma_i"], resonance.Gamma_i)
        self.assertEqual(captured["Gamma_o"], resonance.Gamma_o)

    def test_reduced_width_conversion_matches_penetrability_formula(self):
        metadata = _parse_template_metadata(TEMPLATE_TEXT.splitlines())
        resonance = Resonance(
            E_r=1.0e6,
            J=1.0,
            s1=0.0,
            s2=0.0,
            m1=1.0,
            m2=1.0,
            Gamma_i=2.5,
            Gamma_o=4.0,
            L1=0,
            L2=1,
        )
        q_mev = 1.2
        exf_mev = 0.1
        converted = _convert_reduced_to_partial_widths(
            [resonance],
            metadata,
            l1_default=9,
            l2_default=9,
            Q_mev=q_mev,
            Exf_mev=exf_mev,
        )[0]

        p1 = penetrability_P_l_mev(0, 2, 12, 4, 22, 1.0)
        p2 = penetrability_P_l_mev(1, 1, 13, 1, 25, 1.0 + q_mev - exf_mev)
        self.assertAlmostEqual(converted.Gamma_i, 2.0 * resonance.Gamma_i * p1)
        self.assertAlmostEqual(converted.Gamma_o, 2.0 * resonance.Gamma_o * p2)


class ClearUpperLimitsTest(unittest.TestCase):
    def test_single_section_is_cleared(self):
        repo = Path(__file__).resolve().parent.parent
        lines = (repo / "input" / "22Mg(a,p)25Al.txt").read_text().splitlines()
        n_before = len(lines)
        removed = _clear_upper_limits(lines)
        self.assertGreater(removed, 0)
        start = next(i for i, ln in enumerate(lines) if ln.startswith("Upper Limits"))
        header = next(j for j in range(start, len(lines)) if lines[j].startswith("Ecm"))
        self.assertTrue(lines[header + 1].startswith("!"))
        self.assertTrue(lines[header + 2].startswith("*"))
        self.assertEqual(len(lines), n_before - removed + 1)

    def test_multiple_sections_are_all_cleared(self):
        repo = Path(__file__).resolve().parent.parent
        for fname, expected_sections in (
            ("24Mg(a,g)28Si.txt", 4),
            ("25Al(p,g)26Si.txt", 2),
        ):
            lines = (repo / "input" / fname).read_text().splitlines()
            n_sections = sum(
                1 for ln in lines
                if ln.strip().lower().startswith("upper limits of resonances")
            )
            self.assertEqual(n_sections, expected_sections, fname)
            _clear_upper_limits(lines)

            # Walk every occurrence and confirm no real data rows remain.
            i = 0
            while i < len(lines):
                if lines[i].strip().lower().startswith("upper limits of resonances"):
                    header = next(
                        j for j in range(i + 1, len(lines))
                        if lines[j].strip().startswith("Ecm")
                    )
                    end = next(
                        (k for k in range(header + 1, len(lines))
                         if lines[k].strip().startswith("*")),
                        len(lines),
                    )
                    body = lines[header + 1:end]
                    bad = [ln for ln in body if ln.strip() and not ln.strip().startswith("!")]
                    self.assertEqual(bad, [], f"{fname}: section at line {i} not cleared")
                    i = end
                else:
                    i += 1

    def test_duplicated_template_data_is_not_carried_into_output(self):
        # input/14O(a,p)17F.txt's Upper Limits section used to be a byte-for-byte
        # copy of 22Mg(a,p)25Al's -- confirm the generated output no longer
        # contains that leftover data by default.
        repo = Path(__file__).resolve().parent.parent
        template = repo / "input" / "14O(a,p)17F.txt"
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.in"
            inject_rows(template, out, rows=["  (generated rows)"], header_line=None)
            text = out.read_text()
        self.assertNotIn("497.38", text)

    def test_keep_upper_limits_opt_out(self):
        repo = Path(__file__).resolve().parent.parent
        template = repo / "input" / "14O(a,p)17F.txt"
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.in"
            inject_rows(
                template, out, rows=["  (generated rows)"], header_line=None,
                clear_upper_limits=False,
            )
            text = out.read_text()
        self.assertIn("497.38", text)


class ResolveIntrinsicParityTest(unittest.TestCase):
    def test_explicit_override_wins_without_lookup(self):
        # Z=999999 would fail any real lookup; the override must short-circuit it.
        self.assertEqual(_resolve_intrinsic_parity("target", 999999, 1, -1, None), -1)

    def test_real_nubase_lookup_succeeds(self):
        # alpha: Z=2, A=4, ground state 0+.
        self.assertEqual(_resolve_intrinsic_parity("projectile", 2, 4, None, 0.0), 1)

    def test_missing_nuclide_raises_with_required_wording(self):
        with self.assertRaises(ValueError) as ctx:
            _resolve_intrinsic_parity("target", 999, 999, None, None)
        msg = str(ctx.exception)
        self.assertIn("NUBASE failed to retrieve the spin/parity for the target", msg)
        self.assertIn("Z=999, A=999", msg)
        self.assertIn("doesn't exist or its parity is not recorded in the NUBASE file", msg)

    def test_ambiguous_real_nuclide_raises_with_required_wording(self):
        # Z=7, A=24 is a real NUBASE entry with an undetermined ground-state parity.
        with self.assertRaises(ValueError) as ctx:
            _resolve_intrinsic_parity("projectile", 7, 24, None, None)
        msg = str(ctx.exception)
        self.assertIn("NUBASE failed to retrieve the spin/parity for the projectile", msg)
        self.assertIn("Z=7, A=24", msg)

    def test_missing_za_without_override_raises(self):
        with self.assertRaises(ValueError):
            _resolve_intrinsic_parity("target", None, None, None, None)


class ExitChannelSettingsTest(unittest.TestCase):
    PARTICLE_LINES = TEMPLATE_TEXT.splitlines()  # 22Mg(a,p)25Al

    GAMMA_LINES = [
        "24Mg(p,g)25Al",
        "12    ! Ztarget",
        "1     ! Zproj",
        "0.5   ! Jproj",
        "0.0   ! Jtarget",
        "1.0078 ! Aproj",
        "23.985 ! Atarget",
        "*" * 50,
        "Resonant Contribution",
        "Ecm     DEcm    wg      Dwg     J     G1        DG1        L1    G2      DG2      L2   G3  DG3  L3  Exf   Int",
        "*" * 50,
        "Upper Limits of Resonances",
        "*" * 50,
        "1000 ! Number of random samples",
    ]

    def test_particle_exit_resolves_ejectile_and_final_state_from_nubase(self):
        metadata = _parse_template_metadata(self.PARTICLE_LINES)
        args = Namespace(auto_l2=True, final_spin=None, final_parity=None, exf_kev=0.0)
        settings = _exit_channel_settings(metadata, args)
        self.assertEqual(settings["exit_kind"], "particle")
        # proton ejectile: 1/2+
        self.assertAlmostEqual(settings["ejectile_spin"], 0.5)
        self.assertEqual(settings["ejectile_parity"], 1)
        # 25Al residual ground state: 5/2+
        self.assertAlmostEqual(settings["final_spin"], 2.5)
        self.assertEqual(settings["final_parity"], 1)

    def test_gamma_exit_resolves_compound_ground_state_from_nubase(self):
        metadata = _parse_template_metadata(self.GAMMA_LINES)
        args = Namespace(auto_l2=True, final_spin=None, final_parity=None, exf_kev=0.0)
        settings = _exit_channel_settings(metadata, args)
        self.assertEqual(settings["exit_kind"], "gamma")
        # compound nucleus is 25Al: 5/2+
        self.assertAlmostEqual(settings["final_spin"], 2.5)

    def test_explicit_final_spin_parity_override_skips_nubase(self):
        metadata = _parse_template_metadata(self.GAMMA_LINES)
        args = Namespace(auto_l2=True, final_spin=0.5, final_parity=-1, exf_kev=500.0)
        settings = _exit_channel_settings(metadata, args)
        self.assertAlmostEqual(settings["final_spin"], 0.5)
        self.assertEqual(settings["final_parity"], -1)

    def test_excited_final_state_without_override_raises(self):
        metadata = _parse_template_metadata(self.GAMMA_LINES)
        args = Namespace(auto_l2=True, final_spin=None, final_parity=None, exf_kev=500.0)
        with self.assertRaises(ValueError) as ctx:
            _exit_channel_settings(metadata, args)
        self.assertIn("--exf-kev populates an excited final state", str(ctx.exception))

    def test_no_auto_l2_disables_exit_kind(self):
        metadata = _parse_template_metadata(self.PARTICLE_LINES)
        args = Namespace(auto_l2=False, final_spin=None, final_parity=None, exf_kev=0.0)
        settings = _exit_channel_settings(metadata, args)
        self.assertIsNone(settings["exit_kind"])


if __name__ == "__main__":
    unittest.main()
