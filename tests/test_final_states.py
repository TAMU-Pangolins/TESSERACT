"""Verify opt-in final-state sampling and exported energy/width conventions."""

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from unittest.mock import patch

import numpy as np

import build_ratesmc_input as builder
from nucres.final_states import ExitWidth, hfb_continuum
from nucres.resonance import Resonance

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


class FinalStateContinuumTest(unittest.TestCase):
    def test_partial_final_bin_preserves_all_states(self):
        block = SimpleNamespace(U=np.array([0.0, 2.0]), rho_J=np.ones((2, 50)))
        record = SimpleNamespace(positive=block, negative=block)
        continuum = hfb_continuum(record, 26, 0.0, 0.25, 0.1)
        np.testing.assert_allclose(continuum.U, [0.05, 0.15, 0.225])
        np.testing.assert_allclose(continuum.N_pos.sum(axis=0), 0.25)
        np.testing.assert_allclose(continuum.N_neg.sum(axis=0), 0.25)

    def test_invalid_continuum_width_is_rejected(self):
        for width in (0.0, -0.1, float('nan'), float('inf')):
            with self.subTest(width=width), self.assertRaisesRegex(ValueError, 'positive'):
                hfb_continuum(None, 26, 0.0, 1.0, width)

    def test_summed_exit_does_not_require_residual_ground_state(self):
        metadata = SimpleNamespace(reaction='22Mg(a,p)25Al')
        args = Namespace(exit_width_model='summed', auto_l2=True)
        with patch.object(builder, 'ground_state_jpi', return_value=(0.5, 1)) as lookup:
            settings = builder._exit_channel_settings(metadata, args)
        lookup.assert_called_once_with(1, 1)
        self.assertEqual(settings['ejectile_spin'], 0.5)
        self.assertNotIn('final_spin', settings)

    def test_summed_gamma_does_not_require_ground_state_lookup(self):
        metadata = SimpleNamespace(reaction='24Mg(p,g)25Al')
        with patch.object(builder, 'ground_state_jpi', side_effect=KeyError), \
             patch.object(builder, '_infer_compound_nucleus', return_value=(13, 25)):
            settings = builder._exit_channel_settings(
                metadata, Namespace(exit_width_model='summed', auto_l2=True))
        self.assertEqual(settings['exit_kind'], 'gamma')


class SummedWidthExportTest(unittest.TestCase):
    def test_summed_partial_width_and_final_state_survive_export_with_provenance(self):
        from nucres.resonance_sum import ResonanceSum, rows_from_dataframe
        from extract_resonance_data import load_reaction_params, extract_data

        repo = Path(__file__).resolve().parent.parent
        res = Resonance(E_r=1e6, J=2, s1=0, s2=0, m1=1, m2=1,
                        Gamma_i=100, Gamma_o=200, L1=2, parity=1)
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / 'summed.in'
            with patch('sys.argv', ['build_ratesmc_input.py', '--template',
                       str(repo / 'input/22Mg(a,p)25Al.txt'), '--output', str(output),
                       '--seed', '7', '--exit-width-model', 'summed']):
                args = builder.parse_args()
            generated = SimpleNamespace(resonances=[res], metadata={'seed': 7})
            model = SimpleNamespace(exit_width=lambda *a: ExitWidth(12.0, 0, 0.4517, 2))
            with patch.object(builder, 'synthesize_sigma_from_hfb', return_value=generated) as synth, \
                 patch.object(builder, '_final_state_model', return_value=model):
                builder.build_ratesmc_input(args)
            cfg = synth.call_args.args[0]
            self.assertTrue(cfg.widths_are_reduced)
            self.assertIsNone(cfg.exit_kind)
            rows = rows_from_dataframe(extract_data(output))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].G1, 12.0)
            self.assertAlmostEqual(rows[0].Exf, 0.4517)
            self.assertEqual(rows[0].L1, 0)
            metadata = json.loads(output.with_name('summed_meta.json').read_text())
            self.assertEqual(metadata['common']['exit_width_model'], 'summed')
            self.assertEqual(metadata['run']['n_closed_final_states'], 0)
            spectrum = ResonanceSum(load_reaction_params(output), rows, 0.5, 2.0, npts=30)
            self.assertTrue(np.isfinite(spectrum.sigma(np.array([1.0]))).all())


class WidthFluctuationValidationTest(unittest.TestCase):
    def test_infinite_degrees_of_freedom_preserve_every_mean_width(self):
        from nucres.generator import HFBSamplerConfig, synthesize_sigma_from_hfb
        configuration = HFBSamplerConfig(
            Z=9, A=18, J=1.0, pi=1,
            Gamma_i_mean_eV=0.5, Gamma_o_mean_eV=0.8,
            Gamma_i_dof=np.inf, Gamma_o_dof=np.inf,
            E_min_mev=0.2, E_max_mev=0.6, U_offset_mev=8.0,
            n_density_points=201, n_sigma_points=128, seed=42,
            widths_are_reduced=True)
        spectrum = synthesize_sigma_from_hfb(configuration)
        self.assertTrue(spectrum.resonances)
        for resonance in spectrum.resonances:
            self.assertEqual(resonance.Gamma_i, 0.5)
            self.assertEqual(resonance.Gamma_o, 0.8)

    def test_cli_preserves_the_ground_state_model_by_default(self):
        with patch('sys.argv', ['build_ratesmc_input.py', '--template', 'unused.txt',
                                '--output', 'unused.in']):
            args = builder.parse_args()
        self.assertEqual(args.exit_width_model, 'ground-state')
        self.assertEqual(args.Gamma_i_dof, 1.0)

    def test_gamma_ground_state_retains_one_degree_of_freedom(self):
        with patch.object(builder, 'ground_state_jpi', return_value=(0.5, 1)), \
             patch.object(builder, '_infer_compound_nucleus', return_value=(13, 25)):
            settings = builder._exit_channel_settings(
                SimpleNamespace(reaction='24Mg(p,g)25Al'), Namespace(auto_l2=True))
        self.assertEqual(settings['Gamma_o_dof'], 1.0)

    def test_invalid_degrees_of_freedom_fail_before_loading_data(self):
        from nucres.generator import HFBSamplerConfig, synthesize_sigma_from_hfb
        for dof in (0, -1, -float('inf'), float('nan')):
            with self.subTest(dof=dof), self.assertRaisesRegex(ValueError, 'degrees of freedom'):
                synthesize_sigma_from_hfb(HFBSamplerConfig(Z=9, Gamma_o_dof=dof))
