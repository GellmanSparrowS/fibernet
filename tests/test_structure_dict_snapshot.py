"""Public StructureGraph snapshot editing and persistence regressions.

Uses existing NumPy plus unittest; no numerical solver or optional backend.
"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import fibernet
from fibernet import Material, StructureGraph

ARTIFACT_DIR = None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def graph(dimension=2):
    g = StructureGraph(dimension=dimension, box_size=(3, 2) if dimension == 2 else (3, 2, 1))
    a = g.add_node([0, 0, 0], boundary=(True, False, False, False, False, False))
    removed = g.add_node([9, 9, 0])
    g.remove_node(removed)
    b = g.add_node([1, 0, 0.3 if dimension == 3 else 0])
    isolated = g.add_node([2, 1, 0.5 if dimension == 3 else 0])
    material = Material(name='snapshot-test', youngs_modulus=1200.0, density=1000.0,
                        stiffness_tensor=np.eye(6), extra={'batch': {'revision': [1]}})
    curve = np.array([[0.2, 0.15, 0.06 if dimension == 3 else 0],
                      [0.8, 0.1, 0.24 if dimension == 3 else 0]])
    e = g.add_edge(a, b, radius=0.2, material=material, internal_points=curve, segments=3)
    g.add_edge(b, a, radius=0.1, material=copy.deepcopy(material), allow_parallel=True)
    removed_edge = g.add_edge(b, isolated)
    g.remove_edge(removed_edge)
    g.metadata.update(design={'tags': ['original']}, next_node_id={'metadata': [100]})
    g.nodes[a].metadata.update(label={'tags': ['support']}, position={'metadata': [0]})
    g.edges[e].metadata.update(route={'speeds': [5.0]}, radius={'metadata': [0.2]})
    return g


def location(g, record, level):
    if level == 'graph':
        return g.metadata['design']['tags'], record['metadata']['design']['tags']
    if level == 'node':
        return g.nodes[0].metadata['label']['tags'], record['nodes'][0]['metadata']['label']['tags']
    return g.edges[0].metadata['route']['speeds'], record['edges'][0]['metadata']['route']['speeds']


class StructureDictSnapshot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temporary = tempfile.TemporaryDirectory(prefix='fibernet-snapshot-tests-') if ARTIFACT_DIR is None else None
        cls.output = Path(cls._temporary.name) if cls._temporary else ARTIFACT_DIR
        cls.output.mkdir(parents=True, exist_ok=True)
        (cls.output/'import-origin.json').write_text(json.dumps({
            'fibernet': fibernet.__file__, 'StructureGraph': sys.modules[StructureGraph.__module__].__file__,
            'numpy_version': np.__version__, 'python': sys.version}, indent=2)+'\n')

    @classmethod
    def tearDownClass(cls):
        if cls._temporary:
            cls._temporary.cleanup()

    def test_export_metadata_is_detached_in_both_directions(self):
        for level in ('graph', 'node', 'edge'):
            for direction in ('export', 'source'):
                with self.subTest(level=level, direction=direction):
                    g = graph(); record = g.to_dict()
                    source, exported = location(g, record, level)
                    expected = copy.deepcopy(source)
                    (exported if direction == 'export' else source).append('edited')
                    self.assertEqual(source if direction == 'export' else exported, expected)

    def test_successive_exports_are_independent(self):
        g = graph(); first = g.to_dict(); second = g.to_dict()
        expected = canonical(second)
        first['metadata']['design']['tags'].append('draft')
        first['nodes'][0]['metadata']['label']['tags'].append('draft')
        first['edges'][0]['metadata']['route']['speeds'].append(99)
        self.assertEqual(canonical(second), expected)
        self.assertEqual(canonical(g.to_dict()), expected)

    def test_edit_restore_and_save_preserves_original_design(self):
        for dimension in (2, 3):
            with self.subTest(dimension=dimension):
                g = graph(dimension); before = canonical(g.to_dict())
                draft = g.to_dict()
                draft['metadata']['design']['tags'].append('export-only')
                draft['nodes'][0]['metadata']['label']['tags'].append('new-label')
                draft['edges'][0]['metadata']['route']['speeds'][0] = 9.0
                revised = StructureGraph.from_dict(draft)
                original_path = self.output/f'original-{dimension}d.json'
                draft_path = self.output/f'draft-{dimension}d.json'
                g.save_json(str(original_path)); revised.save_json(str(draft_path))
                original = StructureGraph.load_json(str(original_path))
                loaded_draft = StructureGraph.load_json(str(draft_path))
                self.assertEqual(canonical(original.to_dict()), before)
                self.assertEqual(canonical(loaded_draft.to_dict()), canonical(draft))
                self.assertEqual(list(original.nodes), [0, 2, 3])
                self.assertEqual(list(original.edges), [0, 1])
                self.assertEqual(original.to_dict()['next_node_id'], 4)
                self.assertEqual(original.to_dict()['next_edge_id'], 3)
                np.testing.assert_array_equal(original.edges[0].internal_points, g.edges[0].internal_points)
                self.assertEqual(original.edges[0].segments, 3)
                self.assertEqual(original.edges[1].node_i, 2)

    def test_unedited_canonical_values_remain_stable(self):
        for dimension in (2, 3):
            g = graph(dimension); record = g.to_dict(); encoded = canonical(record)
            (self.output/f'unedited-{dimension}d.json').write_text(encoded+'\n')
            self.assertEqual(canonical(StructureGraph.from_dict(record).to_dict()), encoded)
            self.assertEqual(record['version'], '3.1')
            self.assertEqual([n['id'] for n in record['nodes']], [0, 2, 3])
            self.assertEqual([e['id'] for e in record['edges']], [0, 1])
            self.assertEqual(record['next_node_id'], 4)
            self.assertEqual(record['next_edge_id'], 3)

    def test_array_and_tuple_metadata_not_coerced(self):
        g = graph(); original = np.array([1.0, 2.0])
        g.metadata['array'] = original
        g.metadata['tuple'] = ('kept', [3])
        record = g.to_dict()
        self.assertIsInstance(record['metadata']['array'], np.ndarray)
        self.assertIsInstance(record['metadata']['tuple'], tuple)
        record['metadata']['array'][0] = 9
        record['metadata']['tuple'][1].append(4)
        np.testing.assert_array_equal(original, [1, 2])
        self.assertEqual(g.metadata['tuple'], ('kept', [3]))

    def test_nested_aliases_and_cycles_follow_deepcopy(self):
        g = graph(); shared = []
        g.metadata['a'] = shared; g.metadata['b'] = shared
        shared.append(shared)
        record = g.to_dict()
        self.assertIs(record['metadata']['a'], record['metadata']['b'])
        self.assertIs(record['metadata']['a'][0], record['metadata']['a'])
        self.assertIsNot(record['metadata']['a'], shared)

    def test_live_metadata_mutation_remains_supported(self):
        g = graph()
        g.metadata['design']['tags'].append('live')
        g.nodes[0].metadata['label']['tags'].append('live')
        g.edges[0].metadata['route']['speeds'].append(8)
        record = g.to_dict()
        self.assertEqual(record['metadata']['design']['tags'], ['original', 'live'])
        self.assertEqual(record['nodes'][0]['metadata']['label']['tags'], ['support', 'live'])
        self.assertEqual(record['edges'][0]['metadata']['route']['speeds'], [5.0, 8])

    def test_preexisting_material_and_geometry_detachment(self):
        g = graph(); before = canonical(g.to_dict()); record = g.to_dict()
        record['nodes'][0]['position'][0] = 99
        record['nodes'][0]['boundary'][0] = False
        record['box_size'][0] = 99
        record['edges'][0]['internal_points'][0][0] = 99
        record['edges'][0]['material']['extra']['batch']['revision'].append(2)
        record['edges'][0]['material']['stiffness_tensor'][0][0] = 99
        self.assertEqual(canonical(g.to_dict()), before)

    def test_serialization_failure_keeps_destination(self):
        for failure in ('metadata', 'curve', 'material'):
            with self.subTest(failure=failure):
                g = graph()
                if failure == 'metadata': g.metadata['not_json'] = object()
                elif failure == 'curve': g.edges[0].internal_points[0, 0] = np.nan
                else: g.edges[0].material = 'invalid'
                path = self.output/f'preserve-{failure}.json'; path.write_bytes(b'KEEP\r\n')
                with self.assertRaises((TypeError, ValueError)):
                    g.save_json(str(path))
                self.assertEqual(path.read_bytes(), b'KEEP\r\n')

    def test_legacy_record_defaults_remain_compatible(self):
        old = {'nodes': [{'position': [0, 0]}, {'position': [1, 0]}],
               'edges': [{'node_i': 0, 'node_j': 1}]}
        for version in (None, '3.0', '3.1'):
            value = copy.deepcopy(old)
            if version is not None: value['version'] = version
            g = StructureGraph.from_dict(value); record = g.to_dict()
            self.assertEqual(record['version'], '3.1')
            self.assertEqual(list(g.nodes), [0, 1])
            self.assertEqual(list(g.edges), [0])
            self.assertEqual(record['edges'][0]['material']['name'], 'generic')


if __name__ == '__main__':
    if '--artifact-dir' in sys.argv:
        i = sys.argv.index('--artifact-dir'); ARTIFACT_DIR = Path(sys.argv[i+1]).resolve()
        del sys.argv[i:i+2]
    unittest.main(verbosity=2)
