"""Regressions for PR #22's integration with current main."""
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import build_ratesmc_input as builder
from nucres.final_states import ExitWidth, hfb_continuum
from nucres.resonance import Resonance


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
                       '--seed', '7', '--int-width-cutoff', '0.01']):
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


class RatesMCValidationTest(unittest.TestCase):
    def test_diagnostics_and_nonfinite_values_are_not_rate_rows(self):
        import tesseract
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / 'RatesMC.out'
            output.write_text('1 diagnostic text only\n'
                              '0.1 nan 1 2\n0.2 1 inf 2\n-0.3 1 2 3\n'
                              '0.4 1 2 3\n0.5 0 0 0\n')
            self.assertEqual(tesseract._ratesmc_rate_rows(output), 2)


class WidthFluctuationValidationTest(unittest.TestCase):
    def test_invalid_degrees_of_freedom_fail_before_loading_data(self):
        from nucres.generator import HFBSamplerConfig, synthesize_sigma_from_hfb
        for dof in (0, -1, -float('inf'), float('nan')):
            with self.subTest(dof=dof), self.assertRaisesRegex(ValueError, 'degrees of freedom'):
                synthesize_sigma_from_hfb(HFBSamplerConfig(Z=9, Gamma_o_dof=dof))


class BatchRunnerPathTest(unittest.TestCase):
    def test_relative_binary_and_mass_directory_work_inside_run(self):
        import os
        import subprocess
        repo = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'install').mkdir()
            binary = root / 'install/RatesMC'
            binary.write_text('#!/bin/sh\n'
                              '[ -f mass_1.mas20 ] && [ -f nubase_3.mas20 ] || exit 2\n'
                              'echo "0.1 1 2 3" > RatesMC.out\nexit 1\n')
            binary.chmod(0o755)
            (root / 'tables').mkdir()
            for name in ('mass_1.mas20', 'nubase_3.mas20'):
                (root / 'tables' / name).write_text('table')
            generator = root / 'generate'
            generator.write_text('#!/bin/sh\nwhile [ "$#" -gt 0 ]; do\n'
                                 'case "$1" in --output) shift; touch "$1";; esac\n'
                                 'shift\ndone\n')
            generator.chmod(0o755)
            (root / 'example.txt').write_text('example')
            env = dict(os.environ, RATESMC_BIN='./install/RatesMC', MASS_DIR='tables',
                       PYTHON=str(generator), RUNS='1', OUTPUT_ROOT='outputs')
            result = subprocess.run(['bash', str(repo / 'run_ratesmc_batches.sh'), 'example.txt'],
                                    cwd=root, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(len(list(root.glob('outputs/**/RatesMC.out'))), 1)


if __name__ == '__main__':
    unittest.main()
