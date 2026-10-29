"""Verify affected-test selection across the repository's code locations."""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parent.parent / '.github/scripts'
sys.path.insert(0, str(SCRIPTS))
SPECIFICATION = importlib.util.spec_from_file_location('code_checks', SCRIPTS / 'code_checks.py')
code_checks = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(code_checks)


class CodeTestSelectionTest(unittest.TestCase):
    AVAILABLE = ['test_build_ratesmc_input.py', 'test_data.py', 'test_lab_to_cm.py',
                 'test_ratesmc_export.py', 'test_resonance_sum.py', 'test_talys_statistics.py']

    def test_optimizer_selects_statistics_and_energy_frame_tests(self):
        selected, _ = code_checks.select_tests(['talys_opt_with_unc.py'], self.AVAILABLE)
        self.assertEqual(selected, ['test_lab_to_cm.py', 'test_talys_statistics.py'])

    def test_existing_analysis_tools_select_their_parser_regressions(self):
        for path in ['scripts/plot_talys_param_grid.py', 'plot_talys_results_new.py',
                     'post_processing/plot_talys_results_new.py']:
            with self.subTest(path=path):
                selected, _ = code_checks.select_tests([path], self.AVAILABLE)
                self.assertEqual(selected, ['test_talys_statistics.py'])

    def test_common_and_numerical_package_changes_require_the_full_suite(self):
        for path in ['common/densities_retrieval.py', 'nucres/resonance.py']:
            with self.subTest(path=path):
                selected, reason = code_checks.select_tests([path], self.AVAILABLE)
                self.assertEqual(selected, sorted(self.AVAILABLE))
                self.assertIn('shared', reason)

    def test_unmapped_examples_scripts_and_root_code_are_not_skipped(self):
        for path in ['examples/new_calculation.py', 'scripts/new_analysis.py', 'new_pipeline.py']:
            with self.subTest(path=path):
                selected, reason = code_checks.select_tests([path], self.AVAILABLE)
                self.assertEqual(selected, sorted(self.AVAILABLE))
                self.assertIn('no dedicated regression mapping', reason)

    def test_matching_new_example_test_is_selected_when_available(self):
        available = self.AVAILABLE + ['test_new_calculation.py']
        selected, _ = code_checks.select_tests(['examples/new_calculation.py'], available)
        self.assertEqual(selected, ['test_new_calculation.py'])

    def test_multiple_changes_combine_targets_without_duplicates(self):
        selected, _ = code_checks.select_tests(
            ['talys_opt_with_unc.py', 'scripts/plot_talys_param_grid.py', 'download_data.py'],
            self.AVAILABLE)
        self.assertEqual(selected, ['test_data.py', 'test_lab_to_cm.py', 'test_talys_statistics.py'])

    def test_missing_target_or_unavailable_comparison_requires_the_full_suite(self):
        for paths in [None, ['talys_opt_with_unc.py']]:
            with self.subTest(paths=paths):
                selected, _ = code_checks.select_tests(paths, ['test_data.py'])
                self.assertEqual(selected, ['test_data.py'])

    def test_dependency_reference_data_and_ci_changes_require_the_full_suite(self):
        for path in ['pyproject.toml', 'uv.lock', 'data/ame20.csv', 'input/reaction.txt',
                     '.github/workflows/tests.yml', '.github/scripts/code_checks.py']:
            with self.subTest(path=path):
                selected, _ = code_checks.select_tests([path], self.AVAILABLE)
                self.assertEqual(selected, sorted(self.AVAILABLE))

    def test_full_suite_runs_after_targeted_tests(self):
        with patch.object(code_checks.Path, 'glob', return_value=[Path(name) for name in self.AVAILABLE]), \
             patch.object(code_checks, 'run_test_files', return_value=True) as run, \
             patch.object(code_checks, 'append_summary'):
            result = code_checks.test_checks(['talys_opt_with_unc.py'])
        self.assertEqual(result, 0)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[-1].args[0], sorted(self.AVAILABLE))

    def test_shared_change_does_not_repeat_the_full_suite(self):
        with patch.object(code_checks.Path, 'glob', return_value=[Path(name) for name in self.AVAILABLE]), \
             patch.object(code_checks, 'run_test_files', return_value=True) as run, \
             patch.object(code_checks, 'append_summary'):
            result = code_checks.test_checks(['nucres/resonance.py'])
        self.assertEqual(result, 0)
        run.assert_called_once()

    def test_targeted_failure_fails_the_check(self):
        with patch.object(code_checks.Path, 'glob', return_value=[Path(name) for name in self.AVAILABLE]), \
             patch.object(code_checks, 'run_test_files', return_value=False), \
             patch.object(code_checks, 'append_summary'):
            self.assertEqual(code_checks.test_checks(['talys_opt_with_unc.py']), 1)


class CodeCheckFailureReportingTest(unittest.TestCase):
    def test_python_failure_does_not_misreport_shell_syntax(self):
        def files(pattern):
            return ['program.py'] if pattern == '*.py' else ['program.sh']

        python_result = type('Result', (), {'returncode': 1})()
        shell_result = type('Result', (), {'returncode': 0})()
        with patch.object(code_checks, 'tracked_files', side_effect=files), \
             patch.object(code_checks.subprocess, 'run', side_effect=[python_result, shell_result]), \
             patch.object(code_checks, 'append_summary') as summary:
            self.assertEqual(code_checks.syntax_check(), 1)
        self.assertIn('Python syntax: **failed**', summary.call_args_list[0].args[0])
        self.assertIn('Shell syntax: **passed**', summary.call_args_list[1].args[0])

    def test_an_empty_regression_suite_cannot_pass(self):
        with self.assertRaisesRegex(RuntimeError, 'contains no tests'):
            code_checks.run_test_files([], 'Empty selection')


if __name__ == '__main__':
    unittest.main()
