"""Native result bytes cannot reenter batch resume as a new graph."""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import numpy as np

import fibernet as fn
import fibernet.easy as easy
from fibernet.sim.accelerated import SimResult


class BatchSavedResultInputTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.calls = []
        calls = self.calls

        class Engine:
            def __init__(self):
                calls.append('init')

            def stretch_test(self, graph, **kwargs):
                calls.append((graph.num_nodes, graph.num_edges))
                return SimResult(mode='dynamics', n_nodes=graph.num_nodes,
                                 n_edges=graph.num_edges, max_force=1.,
                                 displacements=np.zeros((graph.num_nodes, 3)))

        engine = patch.object(easy, '_taichi_engine_class', return_value=Engine)
        engine.start()
        self.addCleanup(engine.stop)
        graph = fn.StructureGraph(dimension=2)
        a, b = (graph.add_node(p) for p in ([0., 0.], [1., 1.]))
        graph.add_edge(a, b)
        self.source = self.root/'a.json'
        graph.save_json(self.source)
        self.run_batch()
        self.result_path = self.root/'a_result.json'
        self.assertEqual(json.loads(self.result_path.read_text())['displacements'],
                         [[0., 0., 0.], [0., 0., 0.]])
        self.calls.clear()

    def run_batch(self):
        return easy.batch_simulate_from_json(str(self.root), str(self.root))

    def check_refusal(self):
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        with self.assertRaisesRegex(ValueError, 'saved simulation result, not a StructureGraph'):
            self.run_batch()
        self.assertEqual(self.calls, [])
        self.assertEqual({p.name: p.read_bytes() for p in self.root.iterdir()}, before)

    def test_actual_saved_detailed_result_after_original_removed(self):
        self.source.unlink()
        self.check_refusal()

    def test_native_result_summary_at_same_historical_output_path(self):
        self.source.unlink()
        SimResult(mode='dynamics', n_nodes=2, n_edges=1).save(str(self.result_path))
        self.assertNotIn('displacements', json.loads(self.result_path.read_text()))
        self.check_refusal()

    def test_replacement_at_same_name_allows_explicit_and_legacy_graphs(self):
        doc = json.loads(self.source.read_text())
        self.source.unlink()
        csv_file = self.root/'simulation_results.csv'
        original_csv = csv_file.read_bytes()
        for version in ('3.0', '3.1', None):
            for empty in (False, True):
                with self.subTest(version=version, empty=empty):
                    replacement = dict(doc)
                    if version is None:
                        replacement.pop('version')
                    else:
                        replacement['version'] = version
                    if empty:
                        replacement.update(nodes=[], edges=[])
                    (self.root/'a_result_result.json').unlink(missing_ok=True)
                    csv_file.write_bytes(original_csv)
                    self.result_path.write_text(json.dumps(replacement))
                    self.calls.clear()
                    self.run_batch()
                    self.assertEqual(self.calls, ['init', (0, 0) if empty else (2, 1)])
                    with csv_file.open(newline='', encoding='utf-8') as stream:
                        self.assertEqual([r['filename'] for r in csv.DictReader(stream)],
                                         ['a.json', 'a_result.json'])

    def test_unrelated_filename_keeps_existing_batch_contract(self):
        self.source.unlink()
        self.result_path.rename(self.root/'unassociated.json')
        self.run_batch()
        self.assertEqual(self.calls, ['init', (0, 0)])
