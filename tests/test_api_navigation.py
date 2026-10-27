"""Provider-free checks for the generated, selective README navigation."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'update_api_navigation', ROOT / 'scripts/update_api_navigation.py')
NAV = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NAV)


class APINavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.overlay = json.loads((ROOT / NAV.OVERLAY).read_text(encoding='utf-8'))
        cls.data = NAV.derive(ROOT, ROOT / NAV.OVERLAY, cls.overlay['sourceCommit'])
        cls.outputs = NAV.build_outputs(ROOT)

    def test_selected_interfaces_and_handoffs(self):
        self.assertEqual(len(self.data['groups']), 7)
        self.assertEqual(len(self.data['nodes']), 49)
        self.assertEqual(sum(e['type'] == 'dataflow' for e in self.data['edges']), 50)
        self.assertEqual(self.overlay['canonicalRegistry'], 'code_registry.json')
        self.assertTrue(all(e['automatic'] is False for e in self.data['edges']
                            if e['type'] == 'dataflow'))

    def test_generated_files_are_current(self):
        self.assertEqual(set(self.outputs), {'README.md', *NAV.ASSETS})
        for name, text in self.outputs.items():
            with self.subTest(path=name):
                self.assertEqual((ROOT / name).read_bytes(), text.encode('utf-8'))

    def test_namespaces_and_export_links(self):
        for language in ('en', 'zh'):
            section = NAV.section(self.data, language)
            for node in self.data['nodes']:
                with self.subTest(language=language, api=node['id']):
                    self.assertIn('`' + node['namespace'] + '`', section)
                    self.assertIn('](' + node['publicExportUrl'] + ')', section)
        by_id = {n['id']: n for n in self.data['nodes']}
        self.assertEqual(by_id['rl_train']['namespace'], 'fibernet.rl.train_rl')
        self.assertEqual(by_id['legacy_rl']['namespace'], 'fibernet.train_rl')
        self.assertIn('NotImplementedError', by_id['legacy_rl']['output']['en'])

    def test_source_identity_and_illustrative_syntax(self):
        prefix = ('https://github.com/GellmanSparrowS/fibernet/blob/'
                  + self.overlay['sourceCommit'] + '/')
        for node in self.data['nodes']:
            raw = (ROOT / node['sourcePath']).read_bytes()
            self.assertEqual(node['sourceSha256'], hashlib.sha256(raw).hexdigest())
            for key in ('sourceUrl', 'docsUrl', 'publicExportUrl'):
                self.assertTrue(node[key].startswith(prefix))
            self.assertTrue((ROOT / node['publicExportUrl'][len(prefix):]).is_file())
            if 'example' in node:
                ast.parse(node['example'])  # Syntax only; never execute an example.

    def test_surrounding_readme_is_preserved(self):
        current = (ROOT / 'README.md').read_text(encoding='utf-8')
        self.assertEqual(NAV.strip_navigation(current),
                         NAV.strip_navigation(self.outputs['README.md']))
        for language in ('en', 'zh'):
            self.assertEqual(current.count('<!-- api-navigation:' + language + ':start -->'), 1)
            self.assertEqual(current.count('<!-- api-navigation:' + language + ':end -->'), 1)
        self.assertEqual(NAV.build_outputs(ROOT), self.outputs)

    def test_bad_markers_reject(self):
        for value in ('<!-- api-navigation:en:start -->',
                      '<!-- api-navigation:en:end -->',
                      '<!-- api-navigation:en:end --><!-- api-navigation:en:start -->',
                      '<!-- api-navigation:en:start --><!-- api-navigation:en:start -->'
                      '<!-- api-navigation:en:end --><!-- api-navigation:en:end -->'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                NAV.strip_navigation(value)

    def test_input_and_handoff_contracts(self):
        by_id = {n['id']: n for n in self.data['nodes']}
        self.assertIn('Original graph', by_id['joule']['input']['en'])
        self.assertIn('endpoint_weights', by_id['joule']['input']['en'])
        self.assertIn('maximum_bond_length_m', by_id['lammps_prepare']['input']['en'])
        self.assertIn('train(X, Y)', by_id['regressor']['input']['en'])
        edge = next(e for e in self.data['edges'] if e['source'] == 'light_features'
                    and e['target'] == 'regressor')
        self.assertIn('stack 14-element vectors into X rows', edge['label']['en'])

    def test_native_markup_and_svg(self):
        readme = self.outputs['README.md']
        self.assertEqual(readme.count('<details'), readme.count('</details>'))
        self.assertNotRegex(readme.lower(), r'<(?:script|iframe|style)\b')
        for name in NAV.ASSETS:
            root = ET.fromstring(self.outputs[name])
            self.assertEqual(root.attrib['viewBox'], '0 0 1200 748')
            self.assertFalse(any(e.tag.rsplit('}', 1)[-1] in
                                 {'script', 'foreignObject', 'a'} for e in root.iter()))

    def test_read_only_standard_library_derivation(self):
        before = (ROOT / 'code_registry.json').read_bytes()
        NAV.build_outputs(ROOT)
        self.assertEqual(before, (ROOT / 'code_registry.json').read_bytes())
        tree = ast.parse((ROOT / 'scripts/update_api_navigation.py').read_text(encoding='utf-8'))
        allowed = {'__future__', 'argparse', 'ast', 'hashlib', 'html', 'json', 're', 'pathlib'}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(n.name.split('.')[0] in allowed for n in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertIn(node.module.split('.')[0], allowed)


if __name__ == '__main__':
    unittest.main(verbosity=2)
