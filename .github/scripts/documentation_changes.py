"""Decide whether a GitHub event changes inputs to the documentation build."""

import json
import os
from pathlib import Path
import re
import subprocess


DOCUMENTATION_FILES = {
    'README.md',
    'mkdocs.yml',
    'pyproject.toml',
    'uv.lock',
    '.github/workflows/documentation.yml',
    '.github/scripts/documentation_changes.py',
}
DOCUMENTATION_DIRECTORIES = ('docs/', 'nucres/', 'common/')


def documentation_required(event_name, event):
    """Compare the complete event change, including removed and renamed inputs."""
    if event_name == 'workflow_dispatch':
        return True

    try:
        if event_name == 'pull_request':
            base_revision = event['pull_request']['base']['sha']
            head_revision = event['pull_request']['head']['sha']
        elif event_name == 'merge_group':
            base_revision = event['merge_group']['base_sha']
            head_revision = event['merge_group']['head_sha']
        elif event_name == 'push':
            base_revision = event['before']
            head_revision = event['after']
        else:
            return True
    except KeyError:
        return True

    # A missing comparison must cause a build rather than silently omit it.
    revisions = (base_revision, head_revision)
    if any(not isinstance(revision, str)
           or re.fullmatch(r'(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})', revision) is None
           or set(revision) == {'0'} for revision in revisions):
        return True

    if event_name == 'pull_request':
        comparison = [f'{base_revision}...{head_revision}']
    else:
        comparison = [base_revision, head_revision]
    try:
        changed = subprocess.check_output(
            ['git', 'diff', '--no-renames', '--name-only', '-z', *comparison, '--'],
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError:
        print('Could not compare documentation inputs; running the build.')
        return True

    paths = changed.decode('utf-8', errors='surrogateescape').split('\0')
    return any(path in DOCUMENTATION_FILES
               or path.startswith(DOCUMENTATION_DIRECTORIES) for path in paths)


def main():
    event_name = os.environ['GITHUB_EVENT_NAME']
    event = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text())
    required = str(documentation_required(event_name, event)).lower()
    with Path(os.environ['GITHUB_OUTPUT']).open('a') as output:
        output.write(f'required={required}\n')
    print(f'Documentation build required: {required}')


if __name__ == '__main__':
    main()
