"""Provider-free checks for the generated, selective README navigation."""
import ast
from collections import Counter
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
        self.assertEqual(set(self.outputs), {'README.md', 'README_CN.md', *NAV.ASSETS})
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
        current = (ROOT / 'README.md').read_bytes().decode('utf-8')
        self.assertEqual(NAV.strip_navigation(current),
                         NAV.strip_navigation(self.outputs['README.md']))
        for language in ('en', 'zh'):
            self.assertEqual(current.count('<!-- api-navigation:' + language + ':start -->'), 1)
            self.assertEqual(current.count('<!-- api-navigation:' + language + ':end -->'), 1)
        self.assertEqual(NAV.build_outputs(ROOT), self.outputs)

    def test_standalone_chinese_parity_and_preservation(self):
        standalone = (ROOT / 'README_CN.md').read_bytes().decode('utf-8')
        self.assertEqual(NAV.strip_navigation(standalone),
                         NAV.strip_navigation(self.outputs['README_CN.md']))
        start, end = '<!-- locale:zh:start -->', '<!-- locale:zh:end -->'
        inline = self.outputs['README.md'].split(start, 1)[1].split(end, 1)[0].strip()
        separate = standalone.split(start, 1)[1].split(end, 1)[0].strip()
        self.assertEqual(inline, separate)

    def test_lf_and_crlf_preserve_all_non_navigation_bytes(self):
        for newline in ('\n', '\r\n'):
            original = newline.join(('prefix  \t', '<!-- locale:en:start -->',
                                      'body  ', '<!-- locale:en:end -->', 'tail'))
            actual = NAV.insert_section(original, 'en', 'one\ntwo\n')
            with self.subTest(newline=repr(newline)):
                self.assertEqual(NAV.strip_navigation(actual).encode('utf-8'),
                                 original.encode('utf-8'))
                self.assertIn('one' + newline + 'two' + newline, actual)
                if newline == '\r\n':
                    self.assertNotIn('\n', actual.replace('\r\n', ''))

    def test_other_whitespace_is_not_normalized(self):
        first = 'prefix \n<!-- locale:en:start -->\nbody\n'
        second = first.replace('prefix ', 'prefix  ')
        left = NAV.insert_section(first, 'en', 'guide\n')
        right = NAV.insert_section(second, 'en', 'guide\n')
        self.assertNotEqual(left.encode('utf-8'), right.encode('utf-8'))
        self.assertEqual(NAV.strip_navigation(right), second)

    def test_mixed_line_endings_reject(self):
        for text in ('prefix\r\n<!-- locale:en:start -->\nbody',
                     'prefix\r<!-- locale:en:start -->'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                NAV.insert_section(text, 'en', 'guide\n')

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

    def test_inverse_target_names_match_source_in_both_languages(self):
        tree = ast.parse((ROOT / 'fibernet/ml/curve_inverse.py').read_text(encoding='utf-8'))
        targets = set()
        for node in tree.body:
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
                    and any(isinstance(t, ast.Name) and t.id in ('TARGETS', 'SCALARS')
                            for t in node.targets)):
                targets.update(ast.literal_eval(key) for key in node.value.keys)
        self.assertTrue(targets)
        node = next(n for n in self.data['nodes'] if n['id'] == 'curve_inverse')
        for language in ('en', 'zh'):
            with self.subTest(language=language):
                self.assertIn('.run(target_name=..., evaluator=...)', node['input'][language])
                for name in targets:
                    self.assertIn('`' + name + '`', node['input'][language])
                for token in ('unit_keys', 'fixed_unit', 'evaluator(graph, target_name)',
                              '(objective_to_minimize, run)'):
                    self.assertIn(token, node['caveat'][language])
        self.assertIn('Target-curve arrays are not accepted', node['caveat']['en'])
        self.assertIn('不接受目标曲线数组', node['caveat']['zh'])
        for language, phrases in (
                ('en', ('must return', 'finite scalar objective', 'lower is better',
                        'Negate a custom maximization score')),
                ('zh', ('必须返回', '有限标量', '越小越好', '自定义最大化分数应先取负'))):
            for phrase in phrases:
                self.assertIn(phrase, node['caveat'][language])

    def test_geometry_handoffs_are_explicit_in_both_languages(self):
        by_id = {n['id']: n for n in self.data['nodes']}
        edges = {(e['source'], e['target']): e for e in self.data['edges']
                 if e['type'] == 'dataflow'}
        for language in ('en', 'zh'):
            with self.subTest(language=language):
                for token in ('XY', 'internal_points', 'ReducedBeamConfig'):
                    self.assertIn(token, by_id['reduced']['input'][language])
                for node_id in ('simulate', 'taichi'):
                    for token in ('reject_internal_points', 'endpoint_chord'):
                        self.assertIn(token, by_id[node_id]['caveat'][language])
                        self.assertIn(token, edges['graph', 'simulate']['label'][language])
                for token in ('XY', 'internal_points'):
                    self.assertIn(token, edges['graph', 'reduced']['label'][language])
        for language, phrase in (('en', 'graph nodes and straight edges'),
                                 ('zh', '实际图节点与直边')):
            self.assertIn(phrase, by_id['reduced']['caveat'][language])
            self.assertIn(phrase, edges['graph', 'reduced']['label'][language])

    def test_entry_anchors_and_relationship_destinations(self):
        by_id = {n['id']: n for n in self.data['nodes']}
        pattern = r'<a name="(api-(?:en|zh)-[a-z][a-z0-9_]*)"></a>'
        for language in ('en', 'zh'):
            section = NAV.section(self.data, language)
            anchors = re.findall(pattern, section)
            expected = {NAV.entry_anchor(n['id'], language) for n in self.data['nodes']}
            self.assertEqual(len(anchors), len(expected))
            self.assertEqual(set(anchors), expected)
            chunks = re.split(pattern, section)
            entries = dict(zip(chunks[1::2], chunks[2::2]))
            for node in self.data['nodes']:
                entry = entries[NAV.entry_anchor(node['id'], language)]
                expected_links = []
                for edge in self.data['edges']:
                    if edge['type'] != 'dataflow' or node['id'] not in (edge['source'], edge['target']):
                        continue
                    other_id = edge['target'] if edge['source'] == node['id'] else edge['source']
                    other = by_id[other_id]
                    href = NAV.entry_href(other_id, language)
                    expected_links.append((other['namespace'], href))
                    source_label = 'Source' if language == 'en' else '源码'
                    suffix = (f']({href}) — {edge["label"][language]} · '
                              f'[{source_label}]({other["sourceUrl"]})')
                    self.assertIn(suffix, entry)
                actual_links = re.findall(r'\[`([^`]+)`\]\((#user-content-api-[^)]+)\)', entry)
                self.assertCountEqual(actual_links, expected_links)
        for path, count in (('README.md', 98), ('README_CN.md', 49)):
            anchors = re.findall(pattern, self.outputs[path])
            self.assertEqual(len(anchors), count)
            self.assertEqual(len(set(anchors)), count)
        self.assertNotIn('name="api-en-', self.outputs['README_CN.md'])

    def test_anchor_validation_and_static_wording(self):
        self.assertEqual(NAV.entry_anchor('curve_inverse', 'en'), 'api-en-curve_inverse')
        self.assertEqual(NAV.entry_anchor('curve_inverse', 'zh'), 'api-zh-curve_inverse')
        self.assertEqual(NAV.entry_href('curve_inverse', 'en'), '#user-content-api-en-curve_inverse')
        self.assertEqual(NAV.entry_href('curve_inverse', 'zh'), '#user-content-api-zh-curve_inverse')
        for node_id, language in (('Graph', 'en'), ('x/y', 'en'), ('', 'en'), ('graph', 'fr')):
            with self.subTest(node_id=node_id, language=language), self.assertRaises(ValueError):
                NAV.entry_anchor(node_id, language)
            with self.subTest(node_id=node_id, language=language), self.assertRaises(ValueError):
                NAV.entry_href(node_id, language)
        for language, phrase in (('en', 'static directed chord summary'), ('zh', '静态有向弦图')):
            self.assertIn(phrase, NAV.section(self.data, language))

    def test_github_sanitized_relationship_targets_match_authored_names(self):
        for language in ('en', 'zh'):
            section = NAV.section(self.data, language)
            anchors = set(re.findall(r'<a name="(api-(?:en|zh)-[a-z][a-z0-9_]*)"></a>', section))
            links = re.findall(r'\]\((#user-content-api-[^)]+)\)', section)
            # 50 table rows x two endpoints + 50 handoffs x two entries +
            # two ownership rows x two endpoints. No raw or double-prefixed hrefs.
            self.assertEqual(len(links), 204)
            self.assertNotRegex(section, r'\]\(#api-(?:en|zh)-')
            self.assertNotIn('#user-content-user-content-', section)
            self.assertNotIn('<a name="user-content-', section)
            prefix = '#user-content-'
            for href in links:
                self.assertTrue(href.startswith(prefix))
                authored = href[len(prefix):]  # Strip this exact prefix once only.
                self.assertIn(authored, anchors)
                self.assertTrue(authored.startswith('api-' + language + '-'))
            self.assertIn('](#user-content-api-' + language + '-graph)', section)

    def test_chord_projection_accounts_for_all_typed_edges(self):
        nodes, handoffs, pairs, ownership = NAV.relationships(self.data)
        expected = Counter((nodes[e['source']]['group'], nodes[e['target']]['group'])
                           for e in self.data['edges'] if e['type'] == 'dataflow')
        self.assertEqual({(s, t): count for s, t, count in pairs}, dict(expected))
        self.assertEqual(len(handoffs), 50)
        self.assertEqual(sum(count for s, t, count in pairs if s != t), 36)
        self.assertEqual(sum(count for s, t, count in pairs if s == t), 14)
        self.assertEqual(sum(s != t for s, t, _ in pairs), 15)
        self.assertEqual(sum(s == t for s, t, _ in pairs), 6)
        self.assertEqual({(e['source'], e['target']) for e in ownership},
                         {('graph', 'snode'), ('graph', 'sedge')})
        memberships = [e for e in self.data['edges'] if e['type'] == 'containment'
                       and e['source'] not in nodes]
        self.assertEqual(len(memberships), 49)
        self.assertEqual({(e['source'], e['target']) for e in memberships},
                         {(n['group'], n['id']) for n in nodes.values()})
        self.assertEqual(len(self.data['edges']), len(handoffs) + len(ownership) + len(memberships))
        self.assertEqual(NAV.relationships(self.data), NAV.relationships(self.data))

    def test_chord_svg_directions_counts_and_linear_labels(self):
        _, _, pairs, _ = NAV.relationships(self.data)
        expected = {(s, t): str(count) for s, t, count in pairs}
        for language in ('en', 'zh'):
            root = ET.fromstring(NAV.overview(self.data, language))
            paths = [e for e in root.iter() if 'data-source' in e.attrib]
            self.assertEqual(len(paths), len(expected))
            self.assertEqual({(p.attrib['data-source'], p.attrib['data-target']):
                              p.attrib['data-count'] for p in paths}, expected)
            self.assertTrue(all(p.attrib['marker-end'] == 'url(#arrow)' for p in paths))
            labels = ' '.join(''.join(e.itertext()) for e in root.iter()
                              if e.tag.rsplit('}', 1)[-1] == 'text')
            for group in self.data['groups']:
                self.assertIn(group['title'][language], labels)
                within = int(expected.get((group['id'], group['id']), '0'))
                phrase = f'组内 {within} 条' if language == 'zh' else f'{within} within'
                self.assertIn(phrase, labels)
            self.assertIn('50', labels)
            self.assertIn('14', labels)

    def test_linear_handoff_list_has_one_row_per_edge_and_sources(self):
        nodes, handoffs, _, ownership = NAV.relationships(self.data)
        for language in ('en', 'zh'):
            block = NAV.handoff_list(self.data, language)
            rows = [line for line in block.splitlines() if line.startswith('| [`')]
            self.assertEqual(len(rows), 50)
            for row, edge in zip(rows, handoffs):
                for key in ('source', 'target'):
                    node_id = edge[key]
                    href = NAV.entry_href(node_id, language)
                    self.assertIn(f'[`{nodes[node_id]["namespace"]}`]({href})', row)
                    self.assertIn(f'[api-{language}-src-{node_id}]', row)
                    self.assertIn(f'[api-{language}-src-{node_id}]: {nodes[node_id]["sourceUrl"]}', block)
                self.assertIn(edge['label'][language].replace('|', r'\|').replace('\n', ' '), row)
            self.assertEqual(len([line for line in block.splitlines() if line.startswith('- [`')]),
                             len(ownership))
            self.assertIn('containment', block)
            self.assertIn('dataflow', block)
            self.assertNotIn('<a name=', block)
            self.assertNotIn('api-' + ('zh' if language == 'en' else 'en') + '-', block)

    def test_chord_payload_and_no_external_svg_resources(self):
        # Growth allowance relative to the accepted 1353 baseline; not whole-repo size.
        baselines = {'README.md': 157596, 'README_CN.md': 77775}
        for name, baseline in baselines.items():
            self.assertLessEqual(len(self.outputs[name].encode('utf-8')), baseline + 64 * 1024)
        for name in NAV.ASSETS:
            raw = self.outputs[name]
            self.assertLessEqual(len(raw.encode('utf-8')), 16 * 1024)
            self.assertNotRegex(raw, r'https?://(?!www\.w3\.org/2000/svg)')
            root = ET.fromstring(raw)
            self.assertFalse(any(e.tag.rsplit('}', 1)[-1] in
                                 {'image', 'use', 'style', 'foreignObject', 'script', 'a'}
                                 for e in root.iter()))

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
        allowed = {'__future__', 'argparse', 'ast', 'hashlib', 'html', 'json', 'math', 're', 'pathlib'}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(n.name.split('.')[0] in allowed for n in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertIn(node.module.split('.')[0], allowed)


if __name__ == '__main__':
    unittest.main(verbosity=2)
