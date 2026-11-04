"""Explicit typing-marker inclusion must not depend on newer build defaults."""
import ast
from pathlib import Path
import re
import unittest

import fibernet


def _literal_list_declaration(text, section, key):
    """Read the simple string-list declarations checked here, not full TOML.

    Python 3.9 stdlib only: neither tomllib nor an extra TOML dependency is needed.
    Literal single/double-quoted strings, multiline lists and comments are accepted.
    """
    section_match = re.search(
        r'^\[' + re.escape(section) + r'\][ \t]*(?:#[^\n]*)?\n'
        r'(.*?)(?=^\[|\Z)',
        text, re.MULTILINE | re.DOTALL,
    )
    if section_match is None:
        raise AssertionError('Missing configuration section: ' + section)
    key_pattern = r'''(?:"{0}"|'{0}'|{0})'''.format(re.escape(key))
    value_match = re.search(
        r'^[ \t]*' + key_pattern + r'[ \t]*=[ \t]*(\[[^\]]*\])',
        section_match.group(1), re.MULTILINE,
    )
    if value_match is None:
        raise AssertionError('Missing list declaration: ' + section + '.' + key)
    value = ast.literal_eval(value_match.group(1))
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise AssertionError('Expected a literal string list: ' + section + '.' + key)
    return value


class TestTypedPackageData(unittest.TestCase):
    def test_explicit_typed_marker_without_raising_build_floor(self):
        project_root = Path(__file__).resolve().parents[1]
        text = (project_root / 'pyproject.toml').read_text(encoding='utf-8')
        self.assertIn(
            'py.typed',
            _literal_list_declaration(text, 'tool.setuptools.package-data', 'fibernet'),
        )
        self.assertEqual(
            _literal_list_declaration(text, 'build-system', 'requires'),
            ['setuptools>=61.0'],
        )
        self.assertTrue((Path(fibernet.__file__).resolve().parent / 'py.typed').is_file())

    def test_requirements_file_matches_test_runner_constraints(self):
        project_root = Path(__file__).resolve().parents[1]
        dev = _literal_list_declaration(
            (project_root / 'pyproject.toml').read_text(encoding='utf-8'),
            'project.optional-dependencies', 'dev',
        )
        requirements = [
            line.strip()
            for line in (project_root / 'requirements-dev.txt')
            .read_text(encoding='utf-8').splitlines()
            if line.strip() and not line.lstrip().startswith('#')
        ]
        for name in ('pytest', 'pytest-forked'):
            with self.subTest(requirement=name):
                pattern = re.compile(
                    r'^' + re.escape(name) + r'(?=[<>=!~;\[\s]|$)'
                )
                expected = [item for item in dev if pattern.match(item)]
                actual = [item for item in requirements if pattern.match(item)]
                self.assertEqual(len(expected), 1)
                self.assertEqual(actual, expected)
