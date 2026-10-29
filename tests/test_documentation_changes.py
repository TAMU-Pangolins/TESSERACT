"""Verify documentation selection against complete Git event comparisons."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parent.parent / '.github/scripts/documentation_changes.py'
SPECIFICATION = importlib.util.spec_from_file_location('documentation_changes', SCRIPT_PATH)
documentation_changes = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(documentation_changes)


class DocumentationChangesTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.repository = Path(self.temporary.name)
        self.previous_directory = Path.cwd()
        os.chdir(self.repository)
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(os.chdir, self.previous_directory)
        self._git('init', '-q', '--initial-branch=main')
        self._git('config', 'user.name', 'Workflow validation')
        self._git('config', 'user.email', 'workflow@example.invalid')
        self.base_revision = self._commit_file('README.md', 'Initial documentation\n')

    def _git(self, *arguments):
        return subprocess.check_output(['git', *arguments], stderr=subprocess.PIPE).decode().strip()

    def _commit_file(self, relative_path, contents):
        path = Path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
        self._git('add', '--', relative_path)
        self._git('commit', '-q', '-m', 'Update workflow selection fixture')
        return self._git('rev-parse', 'HEAD')

    def _push_event(self, head_revision):
        return {'before': self.base_revision, 'after': head_revision}

    def test_unrelated_code_and_tests_do_not_require_documentation(self):
        self._commit_file('talys_opt_with_unc.py', '# Optimizer implementation\n')
        head_revision = self._commit_file('tests/test_optimizer.py', '# Regression tests\n')
        self.assertFalse(documentation_changes.documentation_required(
            'push', self._push_event(head_revision)))

    def test_rendered_sources_configuration_and_workflow_are_documentation_inputs(self):
        paths = [
            'docs/api/example.md', 'nucres/resonance.py', 'common/densities_retrieval.py',
            'mkdocs.yml', 'pyproject.toml', 'uv.lock',
            '.github/workflows/documentation.yml',
            '.github/scripts/documentation_changes.py',
        ]
        for path in paths:
            with self.subTest(path=path):
                head_revision = self._commit_file(path, 'Changed documentation input\n')
                self.assertTrue(documentation_changes.documentation_required(
                    'push', self._push_event(head_revision)))

    def test_pull_request_includes_documentation_from_an_earlier_commit(self):
        self._commit_file('docs/guide.md', 'Updated guide\n')
        head_revision = self._commit_file('tests/test_optimizer.py', '# Later regression change\n')
        event = {'pull_request': {'base': {'sha': self.base_revision},
                                  'head': {'sha': head_revision}}}
        self.assertTrue(documentation_changes.documentation_required('pull_request', event))

    def test_moving_a_file_out_of_documentation_still_requires_a_build(self):
        self.base_revision = self._commit_file('docs/guide.md', 'Guide\n')
        self._git('mv', 'docs/guide.md', 'notes.txt')
        self._git('commit', '-q', '-m', 'Move documentation fixture')
        event = self._push_event(self._git('rev-parse', 'HEAD'))
        self.assertTrue(documentation_changes.documentation_required('push', event))

    def test_merge_group_uses_its_base_and_head(self):
        head_revision = self._commit_file('docs/guide.md', 'Queued documentation change\n')
        event = {'merge_group': {'base_sha': self.base_revision, 'head_sha': head_revision}}
        self.assertTrue(documentation_changes.documentation_required('merge_group', event))

    def test_manual_initial_and_unavailable_comparisons_require_a_build(self):
        cases = [
            ('workflow_dispatch', {}),
            ('push', {'before': '0' * 40, 'after': self.base_revision}),
            ('push', {'before': '1' * 40, 'after': self.base_revision}),
            ('pull_request', {}),
        ]
        for event_name, event in cases:
            with self.subTest(event_name=event_name, event=event):
                self.assertTrue(documentation_changes.documentation_required(event_name, event))

    def test_action_output_is_appended_for_an_unrelated_change(self):
        head_revision = self._commit_file('tests/test_optimizer.py', '# Regression tests\n')
        event_path = Path('event.json').resolve()
        event_path.write_text(json.dumps(self._push_event(head_revision)))
        output_path = Path('output.txt').resolve()
        output_path.write_text('existing=value\n')
        environment = {'GITHUB_EVENT_NAME': 'push', 'GITHUB_EVENT_PATH': str(event_path),
                       'GITHUB_OUTPUT': str(output_path)}
        with patch.dict(os.environ, environment):
            documentation_changes.main()
        self.assertEqual(output_path.read_text(), 'existing=value\nrequired=false\n')


if __name__ == '__main__':
    unittest.main()
