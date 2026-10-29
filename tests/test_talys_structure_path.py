"""Verify the runtime TALYS structure path has unambiguous precedence."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import talys_opt_with_unc as optimizer


class TalysStructurePathTest(unittest.TestCase):
    def write_input(self, environment_path):
        configuration = {'proj': 'a', 'tar': 'Mg', 'A_tar': 22,
                         '_x_exp': np.array([2.0]), '_exp_to_lab': 1.0,
                         'talys_fixed': {'strucpath': '/configured/structure/'},
                         'opt_params': []}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
            os.environ.pop('TALYS_STRUCPATH', None)
            if environment_path is not None:
                os.environ['TALYS_STRUCPATH'] = environment_path
            path = optimizer.write_talys_files(directory, np.array([]), configuration)
            lines = Path(path).read_text().splitlines()
        self.assertEqual(configuration['talys_fixed']['strucpath'], '/configured/structure/')
        return [line for line in lines if line.startswith('strucpath ')]

    def test_environment_override_is_written_exactly_once(self):
        self.assertEqual(self.write_input('/local/structure/'), ['strucpath /local/structure/'])

    def test_unset_or_empty_environment_preserves_explicit_configuration(self):
        for path in [None, '']:
            with self.subTest(environment_path=path):
                self.assertEqual(self.write_input(path), ['strucpath /configured/structure/'])
