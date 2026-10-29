"""Verify the documented public API journey and its saved artifacts."""
import csv
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

import fibernet as fn
import examples.open_source_workflow as workflow
from fibernet.ml.beam_frame_fem import BeamFrameFEM
from examples.open_source_workflow import OpenSourceWorkflow


class OpenSourceWorkflowTests(unittest.TestCase):
    def test_documented_workflow(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tmp_path = Path(temporary.name)
        artifact_dir = os.environ.get('FIBERNET_WORKFLOW_TEST_ARTIFACTS')
        if artifact_dir:
            def retain_artifacts():
                destination = Path(artifact_dir)
                destination.mkdir()
                for name in ('network.png', 'fem_result.json', 'summary.json'):
                    source = tmp_path / name
                    if source.is_file():
                        shutil.copyfile(source, destination / name)

            # Cleanups run in reverse order, before removing temporary files.
            self.addCleanup(retain_artifacts)
        observed = {}
        preparations = []
        simulate = fn.simulate
        prepare = BeamFrameFEM.graph_to_fem_input
        analyze = workflow.analyze_tensile_recruitment
        load = fn.SimResult.load
        array_fields = ('displacements', 'deformed_positions',
                        'edge_forces', 'edge_stretches')

        def capture_simulation(graph, **kwargs):
            self.assertEqual(kwargs['stretch_ratio'], 1.01)
            self.assertIs(kwargs['nonlinear'], False)
            self.assertEqual(kwargs['E'], 1e9)
            self.assertEqual(kwargs['nu'], 0.3)
            self.assertNotIn('strain', kwargs)
            self.assertEqual(kwargs['pct'], 0.10)
            self.assertEqual(kwargs['dim'], 2)
            observed['graph'] = graph
            result = simulate(graph, **kwargs)
            observed['simulation'] = result
            observed['arrays'] = {name: getattr(result, name).copy()
                                  for name in array_fields}
            return result

        def capture_reload(path):
            restored = load(path)
            observed['restored'] = restored
            # Discard the live array after loading: analysis must use
            # the saved data.
            observed['simulation'].deformed_positions = None
            return restored

        def capture_preparation(graph, **kwargs):
            self.assertEqual(kwargs, {'dim': 2, 'pct': 0.10})
            prepared = prepare(graph, **kwargs)
            preparations.append(prepared)
            return prepared

        def check_recruitment(edge_strain, edges, left, right, **kwargs):
            # The first preparation is the one used by the actual FEM solve.
            self.assertEqual(len(preparations), 2)
            for side, rows in [('left', left), ('right', right)]:
                np.testing.assert_array_equal(
                    rows, preparations[0]['boundaries'][side])
                np.testing.assert_array_equal(
                    rows, preparations[1]['boundaries'][side])
            np.testing.assert_array_equal(edges, preparations[0]['edge_index'].T)
            initial = preparations[0]['node_pos']
            final = observed['restored'].deformed_positions[:, :2]
            lengths = np.linalg.norm(
                initial[edges[:, 1]] - initial[edges[:, 0]], axis=1)
            final_lengths = np.linalg.norm(
                final[edges[:, 1]] - final[edges[:, 0]], axis=1)
            np.testing.assert_array_equal(edge_strain[0], np.zeros(len(edges)))
            np.testing.assert_allclose(edge_strain[1], final_lengths / lengths - 1.0)
            observed['recruitment_checked'] = True
            return analyze(edge_strain, edges, left, right, **kwargs)

        with patch.object(fn, 'simulate', capture_simulation), \
                patch.object(fn.SimResult, 'load', staticmethod(capture_reload)), \
                patch.object(BeamFrameFEM, 'graph_to_fem_input',
                             staticmethod(capture_preparation)), \
                patch.object(workflow, 'analyze_tensile_recruitment',
                             check_recruitment):
            summary = OpenSourceWorkflow(tmp_path).run()
        self.assertTrue(observed['recruitment_checked'])
        self.assertEqual(summary['nodes'], 18)
        self.assertEqual(summary['edges'], 22)
        self.assertGreaterEqual(summary['recruited_edge_fraction'], 0)
        self.assertLessEqual(summary['recruited_edge_fraction'], 1)
        self.assertGreater((tmp_path / 'network.png').stat().st_size, 1000)
        restored = load(tmp_path / 'fem_result.json')
        self.assertGreater(restored.max_force, 0)
        self.assertEqual((restored.n_nodes, restored.n_edges), (18, 22))
        for name in array_fields:
            self.assertIsNotNone(getattr(restored, name))
            np.testing.assert_array_equal(
                getattr(restored, name), observed['arrays'][name])
        for name, entities in [('node_ids', observed['graph'].nodes),
                               ('edge_ids', observed['graph'].edges)]:
            self.assertEqual(restored.metadata[name], list(entities))
            self.assertEqual(restored.metadata[name],
                             observed['simulation'].metadata[name])
        self.assertEqual(
            json.loads((tmp_path / 'summary.json').read_text()), summary)

    def test_batch_fem_columns_and_invalid_backend(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tmp_path = Path(temporary.name)
        graph = fn.pattern_2d(unit='honeycomb', grid=(2, 2))
        with self.assertRaisesRegex(ValueError, 'backend'):
            fn.simulate(graph, backend='missing')
        output = tmp_path / 'results.csv'
        fn.batch_simulate([{'unit': 'honeycomb', 'grid': (2, 2)}],
                          str(output), strain=1.01)
        with output.open(newline='') as file:
            row = next(csv.DictReader(file))
        self.assertGreater(float(row['max_force']), 0)
        self.assertGreater(float(row['max_displacement']), 0)
        self.assertNotIn('E_star', row)


if __name__ == '__main__':
    unittest.main()
