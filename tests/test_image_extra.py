"""The legacy image-feature provider is optional and explicitly installable."""
import ast
from pathlib import Path
import re
import unittest


def _requirements(text, section, key):
    # The relevant declarations are literal string lists; support Python 3.9
    # without adding a TOML parser to core or test dependencies.
    block = re.search(
        r'^\[' + re.escape(section) + r'\]\s*\n(.*?)(?=^\[|\Z)',
        text, re.MULTILINE | re.DOTALL,
    )
    if block is None:
        raise AssertionError('Missing section: ' + section)
    declaration = re.search(
        r'^' + re.escape(key) + r'\s*=\s*(\[[^\]]*\])',
        block.group(1), re.MULTILINE,
    )
    if declaration is None:
        raise AssertionError('Missing dependency group: ' + key)
    return ast.literal_eval(declaration.group(1))


class TestImageExtra(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.config = (cls.root / 'pyproject.toml').read_text(encoding='utf-8')

    def test_image_extra_declares_one_headless_provider(self):
        self.assertEqual(
            _requirements(self.config, 'project.optional-dependencies', 'image'),
            ['opencv-python-headless>=4.5'],
        )

    def test_full_extra_includes_the_same_single_provider(self):
        full = _requirements(self.config, 'project.optional-dependencies', 'full')
        self.assertEqual(
            [item for item in full if item.startswith('opencv-')],
            _requirements(self.config, 'project.optional-dependencies', 'image'),
        )

    def test_base_requirements_remain_unchanged(self):
        self.assertEqual(
            _requirements(self.config, 'project', 'dependencies'),
            ['numpy>=1.21', 'scipy>=1.7', 'networkx>=3.0', 'matplotlib>=3.5'],
        )

    def test_installation_guide_distinguishes_source_and_release(self):
        guide = (self.root / 'docs/getting_started.rst').read_text(encoding='utf-8')
        self.assertIn('python -m pip install -e ".[image]"', guide)
        self.assertIn('python -m pip install "fibernet[image]==4.4.1"', guide)
        self.assertIn('published 4.4.0 extras', guide)
        self.assertIn('dependency_missing.cv2', guide)
        self.assertIn('Use only one OpenCV distribution', guide)


if __name__ == '__main__':
    unittest.main()
