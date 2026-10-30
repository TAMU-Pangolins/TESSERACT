"""Verify diagnostic suppression preserves existing RatesMC output files."""

import os
from pathlib import Path
import tempfile
import unittest

import tesseract


class RatesMCDiagnosticRetentionTest(unittest.TestCase):
    def test_suppression_archives_previous_output_without_overwriting_an_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'RatesMC.integ'
            path.write_text('previous diagnostic')
            (Path(directory) / 'RatesMC.integ.previous').write_text('older diagnostic')
            backup = tesseract._configure_ratesmc_diagnostic(path, False)
            self.assertEqual(backup.name, 'RatesMC.integ.previous.1')
            self.assertEqual(backup.read_text(), 'previous diagnostic')
            self.assertEqual((Path(directory) / 'RatesMC.integ.previous').read_text(),
                             'older diagnostic')
            self.assertTrue(path.is_symlink())
            self.assertEqual(path.resolve(), Path(os.devnull).resolve())

    def test_retention_removes_a_previous_suppression_link(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'RatesMC.sfact'
            path.symlink_to(os.devnull)
            self.assertIsNone(tesseract._configure_ratesmc_diagnostic(path, True))
            self.assertFalse(path.exists())
            self.assertFalse(path.is_symlink())

    def test_retention_preserves_an_existing_regular_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'RatesMC.sfact'
            path.write_text('retained diagnostic')
            self.assertIsNone(tesseract._configure_ratesmc_diagnostic(path, True))
            self.assertEqual(path.read_text(), 'retained diagnostic')
