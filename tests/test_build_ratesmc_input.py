import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from build_ratesmc_input import (
    _convert_reduced_to_partial_widths,
    _parse_template_metadata,
    build_ratesmc_input,
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
                return Namespace(resonances=[])

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
                synth_mock.return_value = Namespace(resonances=[resonance])
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


if __name__ == "__main__":
    unittest.main()
