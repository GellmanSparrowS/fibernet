"""Source-only checks for the homepage's existing-graph manufacturing routes."""
from pathlib import Path
import importlib.util
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "homepage_media_check", ROOT / "scripts/check_bilingual_homepage.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class ManufacturingNavigationTests(unittest.TestCase):
    def test_both_languages_route_the_current_graph_to_existing_apis(self):
        readme = (ROOT / 'README.md').read_text(encoding='utf-8')
        methods = ('manufacturing_from_graph', 'export_mapped_route',
                   'manufacturing_trails_from_graph', 'export_mapped_trails')
        limits = {
            'docs/MANUFACTURING_GRAPH_MAPPING.md#geometry-and-ownership':
                '## Geometry and ownership',
            'docs/MULTISTROKE_MANUFACTURING.md#scope-and-input-validation':
                '## Scope and input validation',
        }
        for locale in ('zh', 'en'):
            with self.subTest(locale=locale):
                section = readme.split(f'<!-- locale:{locale}:start -->', 1)[1]
                section = section.split(f'<!-- locale:{locale}:end -->', 1)[0]
                rows = [line for line in section.splitlines()
                        if line.startswith('|') and '`manufacturing_from_graph`' in line]
                self.assertEqual(len(rows), 1)
                self.assertIn('`StructureGraph`', rows[0])
                for method in methods:
                    self.assertIn(f'`{method}`', rows[0])
                for target, heading in limits.items():
                    self.assertIn(f']({target})', section)
                    path = target.split('#', 1)[0]
                    self.assertIn(f']({path})', rows[0])
                    self.assertIn(heading, (ROOT / path).read_text(encoding='utf-8'))

    def test_native_language_defaults_and_cover_remain(self):
        readme = (ROOT / 'README.md').read_text(encoding='utf-8')
        folds = [('<details' + attrs + '>', summary)
                 for attrs, summary, _ in CHECK.top_level_disclosures(readme)]
        self.assertEqual(folds, [
            ('<details>', '中文'),
            ('<details open>', 'English'),
        ])
        self.assertIn('<img src="docs/media/ui/homepage-cover-v2.svg"', readme)
        self.assertIn('<a href="#quick-start"><img src="docs/media/ui/homepage-quickstart-en-v2.svg"', readme)


if __name__ == '__main__':
    unittest.main()
