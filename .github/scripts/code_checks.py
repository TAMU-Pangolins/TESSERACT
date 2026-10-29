"""Check tracked Python code and run affected tests with a full-suite safeguard."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from documentation_changes import changed_paths


TARGETED_TESTS = {
    'talys_opt_with_unc.py': ('test_talys_statistics.py', 'test_lab_to_cm.py'),
    'plot_talys_results_new.py': ('test_talys_statistics.py',),
    'post_processing/plot_talys_results_new.py': ('test_talys_statistics.py',),
    'build_ratesmc_input.py': ('test_build_ratesmc_input.py', 'test_ratesmc_export.py'),
    'generate_cross_sections_vectorized.py': ('test_resonance_sum.py',),
    'extract_resonance_data.py': ('test_resonance_sum.py',),
    'download_data.py': ('test_data.py',),
    'scripts/plot_talys_param_grid.py': ('test_talys_statistics.py',),
    'scripts/plot_rate_ratio_wide.py': ('test_talys_statistics.py',),
}
SHARED_FILES = {'pyproject.toml', 'uv.lock', '.github/workflows/tests.yml'}
SHARED_DIRECTORIES = ('common/', 'nucres/', '.github/scripts/')


def select_tests(paths, available_tests):
    """Select existing test files; shared or unmapped code requires the full suite."""
    available_tests = set(available_tests)
    selected = set()
    if paths is None:
        return sorted(available_tests), 'Full suite: no reliable changed-file comparison.'
    for path in paths:
        if path in SHARED_FILES or path.startswith(SHARED_DIRECTORIES):
            return sorted(available_tests), f'Full suite: shared code or CI configuration changed ({path}).'
        if path.startswith('tests/'):
            candidates = (Path(path).name,)
        elif path in TARGETED_TESTS:
            candidates = TARGETED_TESTS[path]
        elif path.endswith(('.py', '.sh')) or path.startswith('examples/'):
            candidates = (f'test_{Path(path).stem}.py',)
            if not set(candidates).intersection(available_tests):
                return sorted(available_tests), f'Full suite: no dedicated regression mapping for {path}.'
        elif path.startswith(('data/', 'input/')):
            return sorted(available_tests), f'Full suite: reaction inputs or reference data changed ({path}).'
        else:
            continue
        missing = set(candidates) - available_tests
        if missing:
            return sorted(available_tests), f'Full suite: a selected test file is unavailable ({path}).'
        selected.update(candidates)
    if not selected:
        return sorted(available_tests), 'Full suite: no narrower code-specific selection applies.'
    return sorted(selected), 'Affected component tests.'


def append_summary(text):
    destination = os.environ.get('GITHUB_STEP_SUMMARY')
    if destination:
        with Path(destination).open('a') as summary:
            summary.write(text + '\n')


def tracked_files(pattern):
    output = subprocess.check_output(['git', 'ls-files', '-z', '--', pattern])
    return [path for path in output.decode().split('\0') if path and Path(path).is_file()]


def syntax_check():
    files = tracked_files('*.py')
    if not files:
        raise RuntimeError('No tracked Python files were found to verify.')
    command = [sys.executable, '-m', 'compileall', '-q', *files]
    result = subprocess.run(command)
    status = 'passed' if result.returncode == 0 else 'failed'
    append_summary(f'- Python syntax: **{status}**, {len(files)} tracked Python files.')
    returncode = result.returncode
    shell_returncode = 0
    shell_files = tracked_files('*.sh')
    for path in shell_files:
        shell_result = subprocess.run(['bash', '-n', path])
        if shell_result.returncode:
            shell_returncode = shell_result.returncode
    shell_status = 'passed' if shell_returncode == 0 else 'failed'
    append_summary(f'- Shell syntax: **{shell_status}**, {len(shell_files)} tracked shell scripts.')
    return returncode or shell_returncode


def lint_check():
    files = tracked_files('*.py')
    if not files:
        raise RuntimeError('No tracked Python files were found to verify.')
    command = [sys.executable, '-m', 'ruff', 'check', '--select', 'E9,F63,F7,F82', *files]
    result = subprocess.run(command)
    status = 'passed' if result.returncode == 0 else 'failed'
    append_summary(f'- Correctness lint: **{status}**, `ruff check --select E9,F63,F7,F82` across all tracked Python files.')
    return result.returncode


def run_test_files(files, label):
    sys.path.insert(0, str(Path.cwd()))
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for filename in files:
        suite.addTests(loader.discover('tests', pattern=filename))
    if suite.countTestCases() == 0:
        raise RuntimeError('The selected regression suite contains no tests.')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    status = 'passed' if result.wasSuccessful() else 'failed'
    append_summary(f'- {label}: **{status}**, {result.testsRun} run, '
                   f'{len(result.failures)} failures, {len(result.errors)} errors, {len(result.skipped)} skipped.')
    append_summary('  Files: ' + ', '.join(f'`tests/{filename}`' for filename in files))
    return result.wasSuccessful()


def test_checks(paths, targeted_only=False):
    available = sorted(path.name for path in Path('tests').glob('test_*.py'))
    selected, reason = select_tests(paths, available)
    append_summary('### Testing\n\n' + reason + '\n')
    if paths:
        append_summary('Changed files: ' + ', '.join(f'`{path}`' for path in paths) + '\n')
    complete = selected == available
    if not run_test_files(selected, 'Full regression suite' if complete else 'Targeted regression tests'):
        return 1
    if not complete and not targeted_only:
        return 0 if run_test_files(available, 'Full regression suite') else 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('check', choices=('syntax', 'lint', 'tests'))
    parser.add_argument('--changed-file', action='append', help='Select a changed file for local verification; repeat as needed.')
    parser.add_argument('--targeted-only', action='store_true', help='Local focused run; CI also runs the full regression suite.')
    arguments = parser.parse_args()
    if arguments.check == 'syntax':
        return syntax_check()
    if arguments.check == 'lint':
        return lint_check()
    paths = arguments.changed_file
    if paths is None and os.environ.get('GITHUB_EVENT_PATH'):
        event = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text())
        paths = changed_paths(os.environ.get('GITHUB_EVENT_NAME'), event)
    return test_checks(paths, arguments.targeted_only)


if __name__ == '__main__':
    sys.exit(main())
