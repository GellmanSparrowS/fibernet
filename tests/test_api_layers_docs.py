"""Source-only checks for the API navigation guide; no solver imports/runs."""
import ast
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / 'docs' / 'API_LAYERS.md'


class APILayersDocumentationTests(unittest.TestCase):
    def test_relative_links_resolve(self):
        for target in re.findall(r'\]\(([^)]+)\)', GUIDE.read_text(encoding='utf-8')):
            if '://' not in target:
                self.assertTrue((GUIDE.parent / target.split('#')[0]).is_file(), target)

    def test_python_examples_parse_and_use_keyword_pattern(self):
        blocks = re.findall(r'```python\n(.*?)```', GUIDE.read_text(encoding='utf-8'), flags=re.S)
        self.assertTrue(blocks)
        for block in blocks:
            for node in ast.walk(ast.parse(block)):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id == 'pattern_2d':
                        self.assertEqual(node.args, [])
                        self.assertIn('unit', {k.arg for k in node.keywords})

    def test_facade_unset_strain_signature_matches_guide(self):
        tree = ast.parse((ROOT / 'fibernet' / 'easy.py').read_text(encoding='utf-8'))
        simulate = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'simulate')
        defaults = dict(zip([a.arg for a in simulate.args.args][-len(simulate.args.defaults):],
                            simulate.args.defaults))
        self.assertIsNone(ast.literal_eval(defaults['strain']))
        self.assertEqual(ast.literal_eval(defaults['backend']), 'spring')
        self.assertEqual(ast.literal_eval(defaults['mode']), 'stretch')

    def test_homepage_guide_links_are_discoverable(self):
        self.assertIn('(docs/API_LAYERS.md)', (ROOT / 'README.md').read_text(encoding='utf-8'))
        pypi = (ROOT / 'PYPI_README.md').read_text(encoding='utf-8')
        self.assertIn('API layers and simulation modes (development branch)', pypi)
        self.assertIn('https://github.com/GellmanSparrowS/fibernet/blob/'
                      'work/verified-fibernet-20261016/docs/API_LAYERS.md', pypi)

    def test_legacy_scope_is_explicit(self):
        paths = ['fibernet/integrations/__init__.py',
                 'fibernet/integrations/lammps_integration.py',
                 'fibernet/integrations/LAMMPS_NOTICE.md']
        for path in paths:
            text = (ROOT / path).read_text(encoding='utf-8')
            self.assertIn('lammps()', text)
            self.assertIn('unvalidated', text)
            self.assertIn('qualification', text)


if __name__ == '__main__':
    unittest.main()
