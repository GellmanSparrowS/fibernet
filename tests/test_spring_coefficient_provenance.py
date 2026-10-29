"""Fabricated-array provenance checks only; never initialize or step a solver."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from fibernet.core.structure_graph import StructureGraph
from fibernet.sim import accelerated as module
from fibernet.sim.accelerated import SimResult, TaichiEngine


KEY = 'spring_coefficient_provenance'


class SpringCoefficientProvenanceTests(unittest.TestCase):
    def setUp(self):
        for owner, names in (
            (TaichiEngine, ('__init__', 'dynamics', 'stretch_test', 'compute_forces')),
            (module, ('_ensure_taichi', '_check_taichi_owner')),
        ):
            for name in names:
                blocker = patch.object(owner, name, side_effect=AssertionError('solver execution forbidden'))
                blocker.start()
                self.addCleanup(blocker.stop)

    def graph(self):
        return StructureGraph.from_dict({
            'version': '3.0', 'dimension': 2,
            'nodes': [
                {'id': 40, 'position': [0., 0., 0.]},
                {'id': 7, 'position': [2., 0., 0.]},
                {'id': 19, 'position': [3., 0., 0.]},
            ],
            'edges': [
                {'id': 90, 'node_i': 40, 'node_j': 7, 'radius': .03},
                {'id': 3, 'node_i': 7, 'node_j': 19, 'radius': .07},
            ],
        })

    def result(self):
        return SimResult(
            mode='dynamics', n_nodes=3, n_edges=2,
            displacements=np.array([[0., 0., 0.], [.2, 0., 0.], [.5, 0., 0.]]),
            deformed_positions=np.array([[0., 0., 0.], [2.2, 0., 0.], [3.5, 0., 0.]]),
            metadata={'node_ids': [40, 7, 19], 'edge_ids': [90, 3], 'stiffness': 100000.})

    def produced(self, active=(2., 7.), source='spring_k', legacy=100000.):
        result = self.result()
        result.metadata['stiffness'] = legacy
        normalized = module._normalize_spring_stiffness(active, 2)
        scalar = float(normalized[0]) if np.ndim(active) == 0 else None
        module._populate_spring_result(result, self.graph(), normalized, source, scalar)
        return result, normalized

    def signed_case(self):
        # Preserve the native in-memory result/state ID domain. Do not pass
        # signed IDs through StructureGraph.from_dict's stricter JSON boundary.
        graph, result = self.graph(), self.result()
        graph._nodes = {(-7 if nid == 7 else nid): node for nid, node in graph.nodes.items()}
        for nid, node in graph.nodes.items():
            node.node_id = nid
        graph._edges = {(-90 if eid == 90 else eid): edge for eid, edge in graph.edges.items()}
        for eid, edge in graph.edges.items():
            edge.edge_id = eid
            if edge.node_i == 7:
                edge.node_i = -7
            if edge.node_j == 7:
                edge.node_j = -7
        result.metadata.update(node_ids=[40, -7, 19], edge_ids=[-90, 3])
        normalized = module._normalize_spring_stiffness([2., 7.], 2)
        module._populate_spring_result(result, graph, normalized, 'spring_k', None)
        return graph, result

    def roundtrip(self, result, detailed):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            result.save(str(path), detailed=detailed)
            payload = json.loads(path.read_text(encoding='utf-8'))
            restored = SimResult.load(str(path))
        return restored, payload

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
            for item, previous in zip(actual, expected):
                self.assert_nested_equal(item, previous)
        else:
            self.assertEqual(actual, expected)

    def test_scalar_inputs_preserve_source_and_legacy_argument(self):
        for active, source, legacy in ((100000., 'stiffness', 100000.),
                                       (4., 'stiffness', 4.), (2., 'spring_k', 100000.)):
            with self.subTest(source=source, active=active):
                result, normalized = self.produced(active, source, legacy)
                record = result.metadata[KEY]
                self.assertEqual(record['schema'], 'spring_coefficient_provenance/1')
                self.assertEqual(record['law'], 'geometric_rest_axial/1')
                self.assertEqual(record['quantity_kind'], 'force')
                self.assertEqual(record['simulation'], {
                    'source': source, 'representation': 'scalar', 'edge_count': 2,
                    'availability': 'available', 'value': active})
                self.assertEqual(record['current_metrics'], {'source': 'simulation'})
                self.assertEqual(result.metadata['stiffness'], legacy)
                self.assertNotIn('values', record['simulation'])
                self.assertNotIn('units', record)

    def test_per_edge_input_adopts_normalized_owned_buffer(self):
        caller = np.array([2., 7.])
        result, normalized = self.produced(caller)
        record = result.metadata[KEY]['simulation']
        self.assertIs(record['values'], normalized)
        self.assertFalse(np.shares_memory(normalized, caller))
        caller[:] = 999.
        np.testing.assert_array_equal(record['values'], [2., 7.])
        self.assertEqual(record['row_map'], 'metadata.edge_ids')
        self.assertNotIn('edge_ids', record)
        self.assertEqual(result.metadata['edge_ids'], [90, 3])
        np.testing.assert_allclose(result.edge_forces, [.2, 2.1])
        self.assertAlmostEqual(result.energy, .335)

    def test_producer_passes_exact_normalized_buffer_to_metrics(self):
        result = self.result()
        normalized = module._normalize_spring_stiffness([2., 7.], 2)
        calls = []
        def metrics(graph, *, stiffness):
            self.assertIs(stiffness, normalized)
            self.assertNotIn(KEY, result.metadata)
            calls.append(graph)
        graph = self.graph()
        with patch.object(result, 'compute_detailed', side_effect=metrics):
            module._populate_spring_result(result, graph, normalized, 'spring_k', None)
        self.assertEqual(calls, [graph])
        self.assertIs(result.metadata[KEY]['simulation']['values'], normalized)
        self.assertEqual(result.metadata[KEY]['current_metrics'], {'source': 'simulation'})

    def test_existing_accepted_coefficient_shapes_are_preserved(self):
        # The dynamics normalizer already accepts these; provenance must not
        # tighten input acceptance through an unrelated metadata change.
        for active in (2, np.array(2.), True, [2., 2.], (2., 2.),
                       np.array([2., 7.], dtype=np.float32),
                       np.ma.array([2., 7.], mask=[False, True])):
            with self.subTest(active=repr(active)):
                result, normalized = self.produced(active)
                record = result.to_dict(detailed=True)[KEY]['simulation']
                if np.ndim(active) == 0:
                    self.assertEqual(record['representation'], 'scalar')
                    self.assertEqual(record['value'], float(normalized[0]))
                else:
                    self.assertEqual(record['representation'], 'per_edge')
                    self.assertEqual(record['values'], normalized.tolist())

    def test_detailed_roundtrip_preserves_stable_order_units_and_metrics(self):
        result, normalized = self.produced()
        units = {'length': 'mm', 'time': 'ms', 'mass': 'g', 'frame': 'fixture'}
        result.metadata['native_axial_inputs'] = {
            'schema': 'native_axial_inputs/1', 'node_ids': [40, 7, 19],
            'masses': [1., 2., 3.], 'units': units}
        restored, payload = self.roundtrip(result, detailed=True)
        self.assertEqual(restored.metadata['edge_ids'], [90, 3])
        self.assertEqual(payload[KEY]['simulation']['values'], [2., 7.])
        self.assertEqual(payload[KEY]['current_metrics'], {'source': 'simulation'})
        self.assertEqual(restored.metadata['native_axial_inputs']['units'], units)
        self.assertNotIn('units', restored.metadata[KEY])
        values = restored.metadata[KEY]['simulation']['values']
        self.assertEqual(values.dtype, np.dtype(np.float64))
        self.assertFalse(np.shares_memory(values, normalized))
        # Serialization sorts graph IDs, not result coefficient rows.
        graph = StructureGraph.from_dict(self.graph().to_dict())
        self.assertNotEqual(list(graph.edges), [90, 3])
        restored.compute_detailed(graph)
        np.testing.assert_allclose(restored.edge_forces, [.2, 2.1])
        self.assertAlmostEqual(restored.energy, .335)
        # Signed sparse result IDs remain valid through result JSON and an
        # independently reordered in-memory graph, without widening graph JSON.
        graph, signed = self.signed_case()
        restored, payload = self.roundtrip(signed, detailed=True)
        self.assertEqual(payload['node_ids'], [40, -7, 19])
        self.assertEqual(payload['edge_ids'], [-90, 3])
        self.assertEqual(payload[KEY]['simulation']['values'], [2., 7.])
        graph._nodes = {nid: graph.nodes[nid] for nid in (19, -7, 40)}
        graph._edges = {eid: graph.edges[eid] for eid in (3, -90)}
        restored.compute_detailed(graph, stiffness=[11., 13.])
        self.assertEqual(restored.metadata['edge_ids'], [-90, 3])
        np.testing.assert_array_equal(restored.metadata[KEY]['simulation']['values'], [2., 7.])
        np.testing.assert_allclose(restored.edge_forces, [1.1, 3.9])
        summary, payload = self.roundtrip(restored, detailed=False)
        self.assertEqual(payload['edge_ids'], [-90, 3])
        self.assertEqual(payload[KEY]['simulation']['reason'], 'omitted_from_summary')
        self.assertEqual(payload[KEY]['current_metrics']['reason'], 'omitted_from_summary')
        self.assertEqual(summary.to_dict(detailed=True)[KEY], payload[KEY])

    def test_scalar_summary_keeps_value_and_reference(self):
        result, _ = self.produced(2.)
        restored, payload = self.roundtrip(result, detailed=False)
        self.assertEqual(payload[KEY]['simulation']['value'], 2.)
        self.assertEqual(payload[KEY]['simulation']['availability'], 'available')
        self.assertEqual(restored.metadata[KEY], result.metadata[KEY])
        self.assertNotIn('edge_forces', payload)

    def test_per_edge_summary_never_converts_or_duplicates_values(self):
        class NoMaterialization(np.ndarray):
            def tolist(self):
                raise AssertionError('summary must not materialize coefficient values')
            def copy(self, *args, **kwargs):
                raise AssertionError('summary must not copy coefficient values')
        result, _ = self.produced()
        result.metadata[KEY]['simulation']['values'] = np.array([2., 7.]).view(NoMaterialization)
        summary = result.to_dict()
        record = summary[KEY]['simulation']
        self.assertEqual(record['availability'], 'unavailable')
        self.assertEqual(record['reason'], 'omitted_from_summary')
        self.assertNotIn('values', record)
        self.assertNotIn('edge_ids', record)
        self.assertEqual(summary['edge_ids'], [90, 3])
        self.assertEqual(result.metadata[KEY]['simulation']['availability'], 'available')

    def test_summary_load_then_detailed_save_cannot_invent_values(self):
        result, _ = self.produced()
        restored, summary = self.roundtrip(result, detailed=False)
        again, detailed = self.roundtrip(restored, detailed=True)
        self.assertEqual(detailed[KEY], summary[KEY])
        self.assertNotIn('values', again.metadata[KEY]['simulation'])
        self.assertEqual(again.metadata['stiffness'], 100000.)

    def test_legacy_file_does_not_infer_simulation_coefficients(self):
        result = self.result()
        restored, payload = self.roundtrip(result, detailed=True)
        self.assertNotIn(KEY, payload)
        self.assertNotIn(KEY, restored.metadata)
        self.assertEqual(restored.metadata['stiffness'], 100000.)
        restored.compute_detailed(self.graph(), stiffness=3.)
        self.assertEqual(restored.metadata[KEY]['simulation'],
                         {'availability': 'unknown', 'reason': 'not_recorded'})
        self.assertEqual(restored.metadata[KEY]['current_metrics']['value'], 3.)

    def test_invalid_record_versions_values_and_shapes_reject(self):
        result, _ = self.produced()
        valid = result.to_dict(detailed=True)
        cases = [
            ('record', 'schema', 'spring_coefficient_provenance/99'),
            ('record', 'law', 'force_per_extension'), ('record', 'quantity_kind', 'N'),
            ('record', 'unexpected', True), ('simulation', 'source', 'material'),
            ('simulation', 'edge_count', 1), ('simulation', 'edge_count', True),
            ('simulation', 'representation', 'inferred'),
            ('simulation', 'row_map', 'graph.edges'),
            ('simulation', 'values', [2.]), ('simulation', 'values', [[2., 7.]]),
            ('simulation', 'values', [2., -7.]), ('simulation', 'values', [2., float('nan')]),
            ('simulation', 'values', [2., float('inf')]), ('simulation', 'values', [True, 7.]),
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.json'
            for role, key, value in cases:
                with self.subTest(role=role, key=key, value=value):
                    payload = copy.deepcopy(valid)
                    target = payload[KEY] if role == 'record' else payload[KEY][role]
                    target[key] = value
                    path.write_text(json.dumps(payload), encoding='utf-8')
                    with self.assertRaises(ValueError):
                        SimResult.load(str(path))
                    malformed = self.result()
                    malformed.metadata[KEY] = payload[KEY]
                    with self.assertRaises(ValueError):
                        malformed.to_dict()
        # The unavailable and reference variants are equally strict.
        for replacement in ({'availability': 'unknown', 'reason': 'not_recorded'},
                            {**valid[KEY]['simulation'], 'availability': 'unavailable'},
                            {**valid[KEY]['simulation'], 'values': np.ma.array([2., 7.], mask=False)}):
            result.metadata[KEY]['simulation'] = replacement
            with self.assertRaises(ValueError):
                result.to_dict()
        scalar, _ = self.produced(2.)
        for invalid in (-1., float('inf'), True):
            scalar.metadata[KEY]['simulation']['value'] = invalid
            with self.assertRaises(ValueError):
                scalar.to_dict()

    def test_required_row_maps_and_counts_are_validated(self):
        for ids in (None, [90, 90], [90], [90., 3], [True, 3]):
            with self.subTest(ids=ids):
                result, _ = self.produced()
                if ids is None:
                    result.metadata.pop('edge_ids')
                else:
                    result.metadata['edge_ids'] = ids
                with self.assertRaises(ValueError):
                    result.to_dict(detailed=True)
                payload = self.produced()[0].to_dict(detailed=True)
                if ids is None:
                    payload.pop('edge_ids')
                else:
                    payload['edge_ids'] = ids
                with TemporaryDirectory() as directory:
                    path = Path(directory) / 'invalid_map.json'
                    path.write_text(json.dumps(payload), encoding='utf-8')
                    with self.assertRaises(ValueError):
                        SimResult.load(str(path))
        # A negative ID is not intrinsically invalid. A graph-unknown ID is
        # invalid when recomputing, regardless of its sign, before mutation.
        for unknown in (-999, 999):
            result, _ = self.produced()
            result.metadata['edge_ids'] = [unknown, 3]
            before = copy.deepcopy(vars(result))
            with self.assertRaises(ValueError):
                result.compute_detailed(self.graph(), stiffness=[2., 7.])
            self.assert_nested_equal(vars(result), before)

    def test_recomputation_keeps_simulation_distinct_from_current_metrics(self):
        result, simulation_values = self.produced()
        caller = np.array([11., 13.])
        result.compute_detailed(StructureGraph.from_dict(self.graph().to_dict()), stiffness=caller)
        record = result.metadata[KEY]
        self.assertIs(record['simulation']['values'], simulation_values)
        np.testing.assert_array_equal(record['simulation']['values'], [2., 7.])
        self.assertEqual(record['current_metrics']['source'], 'compute_detailed.stiffness')
        current_values = record['current_metrics']['values']
        self.assertFalse(np.shares_memory(current_values, caller))
        self.assertFalse(np.shares_memory(current_values, simulation_values))
        caller[:] = 999.
        np.testing.assert_array_equal(current_values, [11., 13.])
        np.testing.assert_allclose(result.edge_forces, [1.1, 3.9])
        restored, payload = self.roundtrip(result, detailed=True)
        self.assertEqual(payload[KEY]['simulation']['values'], [2., 7.])
        self.assertEqual(payload[KEY]['current_metrics']['values'], [11., 13.])
        self.assert_nested_equal(restored.metadata[KEY], record)

    def test_geometry_only_or_missing_geometry_does_not_replace_metric_model(self):
        result, _ = self.produced()
        before = copy.deepcopy(vars(result))
        result.compute_detailed(self.graph())
        self.assert_nested_equal(vars(result), before)
        result.deformed_positions = None
        previous = copy.deepcopy(result.metadata[KEY])
        result.compute_detailed(self.graph(), stiffness=999.)
        self.assert_nested_equal(result.metadata[KEY], previous)
        np.testing.assert_allclose(result.edge_forces, [.2, 2.1])

    def test_failed_recomputation_is_atomic_for_metrics_and_provenance(self):
        result, _ = self.produced()
        for coefficient in ([1.], [1., -2.], float('nan')):
            before = copy.deepcopy(vars(result))
            with self.assertRaises(ValueError):
                result.compute_detailed(self.graph(), stiffness=coefficient)
            self.assert_nested_equal(vars(result), before)
        # Failure after arithmetic but before commit must preserve every field.
        before = copy.deepcopy(vars(result))
        with patch.object(module, '_spring_coefficient_record', return_value={'invalid': True}):
            with self.assertRaises(ValueError):
                result.compute_detailed(self.graph(), stiffness=99.)
        self.assert_nested_equal(vars(result), before)
        result.metadata[KEY]['schema'] = 'bad/2'
        before = copy.deepcopy(vars(result))
        with self.assertRaises(ValueError):
            result.compute_detailed(self.graph(), stiffness=99.)
        self.assert_nested_equal(vars(result), before)

    def test_summary_recomputation_preserves_unavailable_simulation_inputs(self):
        original, _ = self.produced()
        restored, payload = self.roundtrip(original, detailed=False)
        # Geometry is explicitly supplied; a summary does not reconstruct it.
        restored.deformed_positions = original.deformed_positions.copy()
        restored.compute_detailed(self.graph(), stiffness=5.)
        record = restored.metadata[KEY]
        self.assertEqual(record['simulation'], payload[KEY]['simulation'])
        self.assertEqual(record['simulation']['reason'], 'omitted_from_summary')
        self.assertEqual(record['current_metrics']['value'], 5.)
        self.assertNotIn('values', restored.to_dict(detailed=True)[KEY]['simulation'])

    def test_legacy_mapless_recomputation_does_not_invent_edge_identity(self):
        result = self.result()
        result.metadata.pop('node_ids')
        result.metadata.pop('edge_ids')
        result.compute_detailed(self.graph(), stiffness=[2., 7.])
        record = result.metadata[KEY]
        self.assertNotIn('node_ids', result.metadata)
        self.assertNotIn('edge_ids', result.metadata)
        self.assertEqual(record['simulation'], {'availability': 'unknown', 'reason': 'not_recorded'})
        self.assertEqual(record['current_metrics']['row_map'], None)
        self.assertEqual(record['current_metrics']['reason'], 'missing_edge_ids')
        self.assertNotIn('values', record['current_metrics'])
        restored, _ = self.roundtrip(result, detailed=True)
        self.assertEqual(restored.metadata[KEY], record)
        result.compute_detailed(self.graph(), stiffness=4.)
        self.assertEqual(result.metadata[KEY]['current_metrics']['value'], 4.)

    def test_unrelated_result_families_do_not_acquire_spring_provenance(self):
        fem = self.result()
        fem.mode = 'fem_beam_frame'
        fem.compute_detailed(self.graph())
        self.assertNotIn(KEY, fem.to_dict())
        # Delegate validation remains in the existing backend owners. These
        # synthetic markers are deliberately not represented as valid physics.
        material = self.result()
        material.metadata['native_axial_viscoelastic'] = {'synthetic': True}
        with patch('fibernet.sim._taichi_viscoelastic.validate_result'), \
                patch('fibernet.sim._taichi_viscoelastic.populate_result', return_value=material):
            material.compute_detailed(self.graph())
            self.assertNotIn(KEY, material.to_dict())
        lammps = self.result()
        lammps.metadata['backend'] = 'lammps'
        with patch('fibernet.integrations._harmonic_profile.result_guard', return_value=None):
            self.assertNotIn(KEY, lammps.to_dict())
        with self.assertRaises(ValueError):
            lammps.compute_detailed(self.graph(), stiffness=2.)
        self.assertNotIn(KEY, lammps.metadata)

    def test_invalid_provenance_save_preserves_existing_file(self):
        result, _ = self.produced()
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            result.save(str(path), detailed=True)
            before = path.read_bytes()
            result.metadata[KEY]['simulation']['values'][0] = -2.
            with self.assertRaises(ValueError):
                result.save(str(path), detailed=True)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual([p.name for p in Path(directory).iterdir()], ['result.json'])

    def test_exported_descriptors_and_values_are_detached(self):
        result, normalized = self.produced()
        detailed = result.to_dict(detailed=True)
        detailed[KEY]['simulation']['values'][0] = 999.
        detailed[KEY]['simulation']['source'] = 'changed'
        detailed[KEY]['current_metrics']['source'] = 'changed'
        np.testing.assert_array_equal(normalized, [2., 7.])
        self.assertEqual(result.metadata[KEY]['simulation']['source'], 'spring_k')
        self.assertEqual(result.metadata[KEY]['current_metrics'], {'source': 'simulation'})
        summary = result.to_dict()
        summary[KEY]['simulation']['reason'] = 'changed'
        self.assertEqual(result.metadata[KEY]['simulation']['availability'], 'available')

    def test_independent_metric_summary_preserves_compact_scalar_inputs(self):
        result, _ = self.produced(2.)
        result.compute_detailed(self.graph(), stiffness=[11., 13.])
        restored, payload = self.roundtrip(result, detailed=False)
        self.assertEqual(payload[KEY]['simulation']['value'], 2.)
        self.assertEqual(payload[KEY]['current_metrics']['reason'], 'omitted_from_summary')
        self.assertEqual(restored.to_dict(detailed=True)[KEY], payload[KEY])
        # A later scalar metric overwrite changes neither original inputs nor
        # the meaning of the legacy stiffness argument.
        restored.deformed_positions = result.deformed_positions.copy()
        restored.compute_detailed(self.graph(), stiffness=3.)
        self.assertEqual(restored.metadata[KEY]['simulation']['value'], 2.)
        self.assertEqual(restored.metadata[KEY]['current_metrics']['value'], 3.)
        self.assertEqual(restored.metadata['stiffness'], 100000.)


if __name__ == '__main__':
    unittest.main()
