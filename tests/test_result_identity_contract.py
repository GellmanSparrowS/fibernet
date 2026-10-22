"""Fabricated-array identity/persistence checks. Never run a physical solver."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from fibernet.core.structure_graph import StructureGraph
from fibernet.ml.beam_frame_fem import BeamFrameFEM
from fibernet.sim.accelerated import SimResult, TaichiEngine


class ResultIdentityContractTests(unittest.TestCase):
    def setUp(self):
        # Any accidental solve/runtime initialization is a hard test failure.
        for owner, methods in (
            (BeamFrameFEM, ('solve_2d', 'solve_3d', 'solve_2d_nonlinear', 'stretch_test', 'history')),
            (TaichiEngine, ('__init__', 'dynamics', 'stretch_test', 'compute_forces')),
        ):
            for method in methods:
                blocker = patch.object(owner, method, side_effect=AssertionError('solver execution forbidden'))
                blocker.start()
                self.addCleanup(blocker.stop)

    def graph(self, dim=2):
        graph = StructureGraph.from_dict({
            'version': '3.0', 'dimension': dim,
            'nodes': [
                {'id': 40, 'position': [0., 0., 0.]},
                {'id': 7, 'position': [2., 0., 0.]},
                {'id': 19, 'position': [2., 1., 0.]},
            ],
            'edges': [
                {'id': 90, 'node_i': 40, 'node_j': 7, 'radius': .03},
                {'id': 3, 'node_i': 7, 'node_j': 40, 'radius': .07},
                {'id': 44, 'node_i': 7, 'node_j': 19, 'radius': .05},
            ],
        })
        # Parallel edges have distinct section/material identities, not only IDs.
        graph.edges[90].material.name = 'parallel_a'
        graph.edges[90].material.youngs_modulus = 100.
        graph.edges[3].material.name = 'parallel_b'
        graph.edges[3].material.youngs_modulus = 300.
        return graph

    def result(self, graph, edge_ids=None):
        node_ids = list(graph.nodes)
        edge_ids = list(graph.edges) if edge_ids is None else list(edge_ids)
        displacement = {40: [0., 0., 0.], 7: [.2, 0., 0.], 19: [.2, .4, 0.]}
        original = np.array([graph.nodes[nid].position for nid in node_ids])
        disp = np.array([displacement[nid] for nid in node_ids])
        return SimResult(
            mode='dynamics', n_nodes=len(node_ids), n_edges=len(edge_ids),
            displacements=disp, deformed_positions=original + disp,
            edge_forces=np.arange(len(edge_ids), dtype=float) + 10.,
            edge_stretches=np.ones(len(edge_ids)), energy=123., max_force=31.,
            positions_trajectory=[original.copy(), original + disp],
            metadata={'node_ids': node_ids, 'edge_ids': edge_ids, 'sentinel': 'unchanged'},
        )

    @staticmethod
    def coefficients(edge_ids):
        by_id = {90: 3., 3: 11., 44: 7.}
        return np.array([by_id[eid] for eid in edge_ids])

    def expected(self, graph, result):
        node_ids = result.metadata.get('node_ids', list(graph.nodes))
        edge_ids = result.metadata.get('edge_ids', list(graph.edges))
        final = dict(zip(node_ids, result.deformed_positions))
        stretches, lengths = [], []
        # Resolve the physical edge, never an endpoint-pair deduplication.
        for eid in edge_ids:
            edge = graph.edges[eid]
            length = np.linalg.norm(graph.nodes[edge.node_j].position - graph.nodes[edge.node_i].position)
            lengths.append(length)
            stretches.append(np.linalg.norm(final[edge.node_j] - final[edge.node_i]) / length)
        stretches, lengths = np.array(stretches), np.array(lengths)
        k = self.coefficients(edge_ids)
        return stretches, k * (stretches - 1.), .5 * np.sum(k * (stretches - 1.) ** 2 * lengths)

    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, np.ndarray):
            self.assertIsInstance(actual, np.ndarray)
            self.assertEqual(actual.dtype, expected.dtype)
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(type(actual), type(expected))
            self.assertEqual(len(actual), len(expected))
            for a, e in zip(actual, expected):
                self.assert_nested_equal(a, e)
        else:
            self.assertEqual(actual, expected)

    def assert_rejected_unchanged(self, result, graph, stiffness=None):
        before = copy.deepcopy(vars(result))
        with self.assertRaises(ValueError):
            result.compute_detailed(graph, stiffness=stiffness)
        self.assert_nested_equal(vars(result), before)

    @staticmethod
    def reorder(graph, node_order, edge_order):
        data = graph.to_dict()
        nodes = {record['id']: record for record in data['nodes']}
        edges = {record['id']: record for record in data['edges']}
        data['nodes'] = [nodes[nid] for nid in node_order]
        data['edges'] = [edges[eid] for eid in edge_order]
        return StructureGraph.from_dict(data)

    def raw_fem(self, graph, dim=2, node_ids=None, selected=None):
        node_ids = list(graph.nodes) if node_ids is None else list(node_ids)
        selected = [0, 1, 2] if selected is None else list(selected)
        physical_ids = [list(graph.edges)[row] for row in selected]
        displacements = {40: [0., 0., 0.], 7: [.2, 0., 0.], 19: [.2, .4, 0.]}
        u = np.zeros((len(node_ids), 3 if dim == 2 else 6))
        for row, nid in enumerate(node_ids):
            u[row, :dim] = displacements[nid][:dim]
        forces = {90: [5., 7., 11.], 3: [-3., 13., 17.], 44: [2., 19., 23.]}
        payload = np.array([forces[eid] for eid in physical_ids])
        if dim == 3:
            payload[:, 1:] = 0.
        return {
            'u': u, 'n_nodes': len(node_ids), 'n_edges': len(selected),
            'edge_list': np.array(selected, dtype=int), 'edge_forces': payload,
            'sigma_axial': np.array([float(eid) for eid in physical_ids]),
            'sigma_bending': np.ones(len(selected)),
            'sigma_total': np.ones(len(selected)) * 2.,
            'reactions': np.arange(u.size, dtype=float).reshape(u.shape),
        }

    def test_reordered_graph_and_result_roundtrip_preserves_identity(self):
        graph = self.graph()
        result = self.result(graph)
        stretches, forces, energy = self.expected(graph, result)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            result.save(str(path), detailed=True)
            restored = SimResult.load(str(path))
        reordered = StructureGraph.from_dict(json.loads(json.dumps(graph.to_dict())))
        self.assertNotEqual(list(graph.nodes), list(reordered.nodes))
        self.assertNotEqual(list(graph.edges), list(reordered.edges))
        trajectories = copy.deepcopy(restored.positions_trajectory)
        restored.compute_detailed(reordered, self.coefficients(restored.metadata['edge_ids']))
        self.assertEqual(restored.metadata['node_ids'], [40, 7, 19])
        self.assertEqual(restored.metadata['edge_ids'], [90, 3, 44])
        np.testing.assert_allclose(restored.edge_stretches, stretches)
        np.testing.assert_allclose(restored.edge_forces, forces)
        self.assertAlmostEqual(restored.energy, energy)
        self.assert_nested_equal(restored.positions_trajectory, trajectories)

    def test_node_and_edge_permutations_are_independent(self):
        graph = self.graph()
        for nodes, edges in (([7, 19, 40], [90, 3, 44]),
                             ([40, 7, 19], [44, 90, 3]),
                             ([19, 40, 7], [3, 44, 90])):
            with self.subTest(nodes=nodes, edges=edges):
                result = self.result(graph)
                expected = self.expected(graph, result)
                reordered = self.reorder(graph, nodes, edges)
                result.compute_detailed(reordered, self.coefficients([90, 3, 44]))
                np.testing.assert_allclose(result.edge_stretches, expected[0])
                np.testing.assert_allclose(result.edge_forces, expected[1])
                self.assertEqual(result.metadata['edge_ids'], [90, 3, 44])

    def test_selected_physical_edges_keep_order_and_properties(self):
        graph = self.graph()
        reordered = StructureGraph.from_dict(graph.to_dict())
        result = self.result(graph, [3, 90])
        expected = self.expected(graph, result)
        result.compute_detailed(reordered, self.coefficients([3, 90]))
        self.assertEqual((result.n_nodes, result.n_edges), (3, 2))
        self.assertEqual(result.metadata['edge_ids'], [3, 90])
        np.testing.assert_allclose(result.edge_forces, expected[1])
        self.assertNotEqual(result.edge_forces[0], result.edge_forces[1])
        self.assertEqual([reordered.edges[eid].radius for eid in result.metadata['edge_ids']], [.07, .03])
        self.assertEqual([reordered.edges[eid].material.name for eid in result.metadata['edge_ids']],
                         ['parallel_b', 'parallel_a'])
        self.assertEqual([reordered.edges[eid].material.youngs_modulus for eid in result.metadata['edge_ids']],
                         [300., 100.])
        # The raw converter also uses each selected physical edge's own radius.
        raw = self.raw_fem(graph, selected=[1, 0])
        converted = BeamFrameFEM(E=100.).to_sim_result(
            raw, graph, node_ids=list(graph.nodes), edge_ids=list(graph.edges))
        expected_energy = 0.
        for sigma, eid in zip(raw['sigma_axial'], [3, 90]):
            edge = graph.edges[eid]
            length = np.linalg.norm(graph.nodes[edge.node_j].position - graph.nodes[edge.node_i].position)
            expected_energy += .5 * sigma ** 2 / 100. * np.pi * edge.radius ** 2 * length
        self.assertAlmostEqual(converted.energy, expected_energy)
        self.assertEqual(converted.metadata['edge_ids'], [3, 90])

    def test_invalid_identity_maps_do_not_mutate_result(self):
        graph = self.graph()
        invalid = [
            ('node_ids', None), ('edge_ids', None),
            ('node_ids', [40, 40, 19]), ('node_ids', [40, 7]),
            ('node_ids', [40, 7, 999]), ('edge_ids', [90, 90, 44]),
            ('edge_ids', [90, 3, 999]), ('node_ids', [True, 7, 19]),
            ('node_ids', [40., 7, 19]), ('edge_ids', [-1, 3, 44]),
            ('node_ids', np.ma.array([40, 7, 19], mask=False)),
        ]
        for name, value in invalid:
            with self.subTest(name=name, value=value):
                result = self.result(graph)
                if value is None:
                    result.metadata.pop(name)
                else:
                    result.metadata[name] = value
                self.assert_rejected_unchanged(result, graph)
        # Persistence keeps its existing policy: duplicate maps cannot load;
        # a historical partial map can load, but recomputation must not guess.
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid_maps.json'
            payload = self.result(graph).to_dict(detailed=True)
            payload['node_ids'] = [40, 40, 19]
            path.write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaises(ValueError):
                SimResult.load(str(path))
            payload['node_ids'] = [40, 7, 19]
            payload.pop('edge_ids')
            path.write_text(json.dumps(payload), encoding='utf-8')
            partial = SimResult.load(str(path))
            self.assert_rejected_unchanged(partial, graph)

    def test_invalid_arrays_counts_and_stiffness_do_not_mutate_result(self):
        graph = self.graph()
        invalid = [
            ('deformed_positions', np.zeros((2, 3))),
            ('deformed_positions', np.zeros((3, 4))),
            ('displacements', np.zeros((2, 3))),
            ('deformed_positions', np.full((3, 3), np.nan)),
            ('displacements', np.ma.array(np.zeros((3, 3)), mask=False)),
            ('edge_forces', np.zeros(2)), ('edge_stretches', np.zeros((3, 1))),
            ('n_nodes', 2), ('n_edges', 2), ('n_nodes', True),
        ]
        for name, value in invalid:
            with self.subTest(field=name, value=value):
                result = self.result(graph)
                setattr(result, name, value)
                self.assert_rejected_unchanged(result, graph)
        for stiffness in ([1., 2.], [1., -1., 2.], float('inf'), True,
                          np.ma.array([1., 2., 3.], mask=False)):
            with self.subTest(stiffness=stiffness):
                self.assert_rejected_unchanged(self.result(graph), graph, stiffness)
        # The existing spring producer populates its default zero counts here.
        result = self.result(graph)
        result.n_nodes = result.n_edges = 0
        result.compute_detailed(graph, self.coefficients([90, 3, 44]))
        self.assertEqual((result.n_nodes, result.n_edges), (3, 3))

    def test_legacy_mapless_result_and_file_keep_insertion_order(self):
        graph = self.graph()
        result = self.result(graph)
        expected = self.expected(graph, result)
        result.metadata.pop('node_ids')
        result.metadata.pop('edge_ids')
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.json'
            result.save(str(path), detailed=True)
            restored = SimResult.load(str(path))
        restored.compute_detailed(graph, self.coefficients(list(graph.edges)))
        np.testing.assert_allclose(restored.edge_stretches, expected[0])
        np.testing.assert_allclose(restored.edge_forces, expected[1])
        self.assertNotIn('node_ids', restored.metadata)
        self.assertNotIn('edge_ids', restored.metadata)

    def test_fem_explicit_stiffness_is_rejected_before_mutation(self):
        graph = self.graph()
        result = BeamFrameFEM(E=100.).to_sim_result(self.raw_fem(graph), graph)
        self.assertEqual(result.mode, 'fem_beam_frame')
        self.assertNotIn('backend', result.metadata)
        self.assert_rejected_unchanged(result, graph, 123.)
        self.assertEqual(result.metadata['energy_kind'], 'axial_stress_proxy')

    def test_fem_geometry_only_recompute_preserves_solver_channels(self):
        graph = self.graph()
        raw = self.raw_fem(graph, selected=[1, 0])
        raw['field_status'] = {'edge_forces': {'status': 'approximate', 'model': 'fabricated'}}
        raw['solver_status'] = {'equilibrium': 'unverified'}
        result = BeamFrameFEM(E=100.).to_sim_result(raw, graph,
                                                  node_ids=list(graph.nodes), edge_ids=list(graph.edges))
        expected = self.expected(graph, result)[0]
        before = copy.deepcopy(vars(result))
        result.compute_detailed(StructureGraph.from_dict(graph.to_dict()))
        np.testing.assert_allclose(result.edge_stretches, expected)
        self.assertEqual(result.n_edges, 2)
        for name in ('edge_forces', 'max_force', 'energy', 'metadata', 'displacements', 'deformed_positions'):
            self.assert_nested_equal(getattr(result, name), before[name])

    def test_raw_numpy_row_maps_survive_conversion_and_roundtrip(self):
        graph = self.graph()
        positions, edges, radii = graph.to_numpy()
        node_ids, edge_ids = sorted(graph.nodes), list(graph.edges)
        self.assertEqual(len(positions), len(node_ids))
        self.assertEqual(len(edges), len(edge_ids))
        np.testing.assert_array_equal(radii, [graph.edges[eid].radius for eid in edge_ids])
        raw = self.raw_fem(graph, node_ids=node_ids, selected=[2, 1, 0])
        result = BeamFrameFEM(E=100.).to_sim_result(raw, graph, node_ids=node_ids, edge_ids=edge_ids)
        self.assertEqual(result.metadata['node_ids'], [7, 19, 40])
        self.assertEqual(result.metadata['edge_ids'], [44, 3, 90])
        expected = self.expected(graph, result)[0]
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'mapped_fem.json'
            result.save(str(path), detailed=True)
            restored = SimResult.load(str(path))
        restored.compute_detailed(graph)
        np.testing.assert_allclose(restored.edge_stretches, expected)
        self.assertEqual(restored.metadata['edge_ids'], [44, 3, 90])

    def assert_contract_roundtrip(self, result):
        with TemporaryDirectory() as directory:
            for detailed in (False, True):
                path = Path(directory) / ('detailed.json' if detailed else 'summary.json')
                result.save(str(path), detailed=detailed)
                restored = SimResult.load(str(path))
                self.assertEqual(restored.metadata['beam_resultant_contract'],
                                 result.metadata['beam_resultant_contract'])
                np.testing.assert_array_equal(restored.metadata['beam_resultants'], result.metadata['beam_resultants'])

    def test_2d_resultant_contract_distinguishes_force_and_moment(self):
        graph = self.graph()
        raw = self.raw_fem(graph)
        result = BeamFrameFEM(E=100.).to_sim_result(raw, graph)
        np.testing.assert_array_equal(result.edge_forces, raw['edge_forces'][:, 0])
        self.assertEqual(result.max_force, 5.)
        self.assert_nested_equal(result.metadata['beam_resultants'], raw['edge_forces'])
        contract = result.metadata['beam_resultant_contract']
        self.assertEqual(contract['dimension'], 2)
        self.assertEqual(contract['unit_system'], 'not_declared')
        self.assertEqual([c['quantity_kind'] for c in contract['components']], ['force', 'force', 'moment'])
        self.assertEqual([c['name'] for c in contract['components']],
                         ['axial_N', 'shear_V', 'mean_bending_moment'])
        self.assertEqual(contract['components'][2]['definition'], '(M_i + M_j) / 2')
        self.assertEqual([c['status'] for c in contract['components']], ['available'] * 3)
        self.assert_contract_roundtrip(result)

    def test_3d_resultant_contract_marks_shear_unavailable(self):
        graph = self.graph(dim=3)
        raw = self.raw_fem(graph, dim=3)
        result = BeamFrameFEM(E=100.).to_sim_result(raw, graph)
        self.assert_nested_equal(result.metadata['beam_resultants'], raw['edge_forces'])
        contract = result.metadata['beam_resultant_contract']
        self.assertEqual(contract['dimension'], 3)
        self.assertEqual([c['quantity_kind'] for c in contract['components']], ['force'] * 3)
        self.assertEqual([c['name'] for c in contract['components']], ['axial_N', 'shear_V_y', 'shear_V_z'])
        self.assertEqual([c['status'] for c in contract['components']], ['available', 'unavailable', 'unavailable'])
        for component in contract['components'][1:]:
            self.assertEqual(component['reason'], 'not_recovered_by_3d_solver')
        self.assertNotIn('contact_forces', result.metadata)
        self.assertNotIn('moments', result.metadata)
        self.assert_contract_roundtrip(result)

    def test_nonlinear_resultant_status_remains_approximate(self):
        graph = self.graph()
        raw = self.raw_fem(graph)
        raw['field_status'] = {'edge_forces': {
            'status': 'approximate', 'approximation': 'original_geometry_total_displacement_linear_reference',
            'model': 'incremental_updated_geometry_linear_solves',
        }}
        raw['solver_status'] = {'equilibrium': 'unverified'}
        result = BeamFrameFEM(E=100.).to_sim_result(raw, graph)
        self.assertEqual([c['status'] for c in result.metadata['beam_resultant_contract']['components']],
                         ['approximate'] * 3)
        self.assertEqual(result.metadata['field_status']['beam_resultants'], raw['field_status']['edge_forces'])
        self.assertEqual(result.metadata['solver_status'], raw['solver_status'])
        self.assert_contract_roundtrip(result)

    def test_legacy_three_column_fem_file_remains_readable(self):
        payload = {'mode': 'fem_beam_frame', 'energy': 1., 'n_nodes': 2, 'n_edges': 1,
                   'edge_forces': [[-2., 3., 7.]], 'edge_stretches': []}
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'historical.json'
            path.write_text(json.dumps(payload), encoding='utf-8')
            restored = SimResult.load(str(path))
        np.testing.assert_array_equal(restored.edge_forces, [-2.])
        self.assertEqual(restored.metadata['beam_resultants'], [[-2., 3., 7.]])
        self.assertNotIn('beam_resultant_contract', restored.metadata)
        self.assertNotIn('node_ids', restored.metadata)
        self.assertNotIn('edge_ids', restored.metadata)

    def test_public_workflow_metadata_compatibility_without_solver(self):
        import examples.open_source_workflow as workflow
        graph = self.graph()
        result = BeamFrameFEM(E=100.).to_sim_result(self.raw_fem(graph), graph)
        with TemporaryDirectory() as directory, \
                patch.object(workflow.fn, 'pattern_2d', return_value=graph), \
                patch.object(workflow.fn, 'show', return_value=None), \
                patch.object(workflow.fn, 'simulate', return_value=result) as simulate, \
                patch.object(BeamFrameFEM, 'graph_to_fem_input', return_value={
                    'boundaries': {'left': [0], 'right': [1, 2]}}), \
                patch('matplotlib.pyplot.close'):
            summary = workflow.OpenSourceWorkflow(directory).run()
            simulate.assert_called_once_with(graph, backend='fem', mode='stretch', strain=1.01, dim=2, pct=.10)
            restored = SimResult.load(str(Path(directory) / 'fem_result.json'))
            self.assertEqual(restored.metadata['beam_resultant_contract'], result.metadata['beam_resultant_contract'])
            self.assertEqual(summary['nodes'], 3)
            self.assertEqual(summary['edges'], 3)
            self.assertEqual(summary['analysis'], 'positive axial edge strain threshold, not total force flow')
            self.assertEqual(json.loads((Path(directory) / 'summary.json').read_text()), summary)


if __name__ == '__main__':
    unittest.main()
