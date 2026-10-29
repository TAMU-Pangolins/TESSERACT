"""Verify RatesMC native invocation, table validation, and batch failures."""

import tempfile
import unittest
from pathlib import Path
import shutil
import subprocess
import os

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



class RatesMCValidationTest(unittest.TestCase):
    def test_diagnostics_and_nonfinite_values_are_not_rate_rows(self):
        import tesseract
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / 'RatesMC.out'
            output.write_text('1 diagnostic text only\n'
                              '0.1 nan 1 2\n0.2 1 inf 2\n-0.3 1 2 3\n'
                              '0.4 1 2 3\n0.5 0 0 0\n')
            self.assertEqual(tesseract._ratesmc_rate_rows(output), 2)

    def test_unordered_intervals_are_not_valid_rate_rows(self):
        import tesseract
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'RatesMC.out'
            output.write_text('0.1 3 2 1\n0.2 1 2 3\n')
            self.assertEqual(tesseract._ratesmc_rate_rows(output), 1)


class RatesMCFailureHandlingTest(unittest.TestCase):
    def prepare_run(self, program):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        binary = root / 'RatesMC'
        binary.write_text(program)
        binary.chmod(0o755)
        for filename in ['mass_1.mas20', 'nubase_3.mas20']:
            (root / filename).write_text('table\n')
        reaction = '22Mg(a,p)25Al'
        run = root / 'outputs' / reaction / 'RUN_0'
        run.mkdir(parents=True)
        (run / f'{reaction}.in').write_text(reaction + '\n')
        return root, binary, reaction, run

    def test_success_exit_with_stale_native_output_is_rejected(self):
        import tesseract
        root, binary, reaction, run = self.prepare_run('#!/bin/sh\nexit 0\n')
        (run / 'RatesMC.out').write_text('0.1 1 2 3\n')
        with self.assertRaisesRegex(RuntimeError, 'no valid rates'):
            tesseract.step_run_ratesmc({'reaction': reaction},
                                      {'output_dir': str(root / 'outputs')},
                                      {'ratesmc_bin': str(binary)})
        self.assertFalse((run / f'{reaction}.out').exists())

    def test_invalid_cached_reaction_table_is_recomputed(self):
        import tesseract
        root, binary, reaction, run = self.prepare_run(RatesMCStepTest.FAKE)
        output = run / f'{reaction}.out'
        output.write_text('diagnostics without rates\n')
        tesseract.step_run_ratesmc({'reaction': reaction},
                                  {'output_dir': str(root / 'outputs')},
                                  {'ratesmc_bin': str(binary)})
        self.assertEqual(tesseract._ratesmc_rate_rows(output), 2)
        self.assertTrue((run / 'RatesMC.stdout').exists())

    def test_missing_requested_template_fails_batch(self):
        repository = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as directory:
            environment = dict(os.environ, RATESMC_BIN=shutil.which('true'),
                               PYTHON=shutil.which('true'))
            result = subprocess.run(['bash', str(repository / 'run_ratesmc_batches.sh'),
                                     'missing-template.txt'], cwd=directory,
                                    env=environment, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Missing requested template', result.stderr)



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
