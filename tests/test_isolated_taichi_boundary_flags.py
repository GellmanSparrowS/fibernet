"""Provider-free proof of exact generated boundary flags and strict admission."""
import json
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from fibernet import StructureGraph, pattern_2d, pattern_3d
from fibernet.sim.isolated_taichi import WorkerRuntime, prepare_request, start_isolated_taichi


def controls(graph):
    return dict(fixed_nodes=[0], external_force=[[0.0, 0.0, 0.0] for _ in graph.nodes],
                spring_k=2.0, dt=0.001, num_steps=2, save_interval=1)


class IsolatedTaichiBoundaryFlagsTests(TestCase):
    def assert_exact_request(self, graph):
        before = [node.boundary for node in graph.nodes.values()]
        self.assertTrue(any(type(flag) is np.bool_ for row in before for flag in row))
        self.assertTrue(any(flag for row in before for flag in row))
        expected = graph.copy()
        for node in expected.nodes.values():
            node.boundary = tuple(bool(flag) for flag in node.boundary)
        args = controls(graph)
        args['external_force'][-1][0] = 0.125
        raw = prepare_request(graph, **args)
        self.assertEqual(raw, prepare_request(expected, **args))
        req = json.loads(raw)
        self.assertEqual(req['node_ids'], list(graph.nodes))
        self.assertEqual(req['edge_ids'], list(graph.edges))
        self.assertEqual(req['controls']['external_force'], args['external_force'])
        fields = [node['boundary'] for node in req['graph']['nodes']]
        self.assertEqual(fields, [list(map(bool, row)) for row in before])
        self.assertTrue(all(type(flag) is bool for row in fields for flag in row))
        self.assertEqual(raw, prepare_request(StructureGraph.from_dict(req['graph']), **req['controls']))
        for node, flags in zip(graph.nodes.values(), before):
            self.assertIs(node.boundary, flags)

    def test_real_unmodified_planar_and_spatial_tilings(self):
        self.assert_exact_request(pattern_2d(unit='honeycomb', box=(2.0, 2.0),
                                             grid=(2, 1), radius=0.05))
        self.assert_exact_request(pattern_3d(unit='cubic', box=(2.0, 2.0, 2.0),
                                             grid=(1, 1, 1), radius=0.05))

    def test_nonmonotone_stable_ids_and_parallel_edges(self):
        graph = StructureGraph.from_dict(dict(version='3.1', dimension=2, nodes=[
            dict(id=9, position=[0., 0.], boundary=[np.bool_(True), False, np.bool_(False),
                                                     True, False, False]),
            dict(id=2, position=[2., 0.]),
        ], edges=[dict(id=18, node_i=9, node_j=2), dict(id=4, node_i=2, node_j=9)]))
        self.assert_exact_request(graph)

    def test_malformed_values_and_containers_reject_before_runtime(self):
        class TruthTrap:
            def __bool__(self):
                raise AssertionError('must not coerce')

        graph = pattern_2d(unit='square', box=(2.0, 2.0), grid=(1, 1), radius=0.05)
        node = next(iter(graph.nodes.values()))
        args = controls(graph)
        bad_flags = [(bad,) + (False,) * 5 for bad in
                     (1, np.int64(1), 0.0, 'False', None, np.ma.masked, TruthTrap())]
        bad_containers = [(False,) * 5, np.zeros(6, bool),
                          np.ma.array(np.zeros(6, bool), mask=False)]
        checked_runtime = WorkerRuntime("must-not-check", "must-not-read", "0" * 64)
        with patch('fibernet.sim.isolated_taichi.WorkerRuntime.checked',
                   side_effect=AssertionError('runtime checked')) as runtime, patch(
                       'fibernet.sim.isolated_taichi.subprocess.Popen',
                       side_effect=AssertionError('worker launched')) as launch:
            for flags in bad_flags + bad_containers:
                with self.subTest(flags=type(flags), elements=len(flags)):
                    node.boundary = flags
                    with self.assertRaisesRegex(ValueError, 'six booleans'):
                        start_isolated_taichi(graph, runtime=checked_runtime, **args)
            node.boundary = (False,) * 6
            node.metadata['not_a_boundary_flag'] = np.bool_(True)
            with self.assertRaisesRegex(ValueError, 'finite built-in JSON'):
                start_isolated_taichi(graph, runtime=checked_runtime, **args)
            runtime.assert_not_called()
            launch.assert_not_called()
