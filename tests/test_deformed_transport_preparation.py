"""Bounded prepare-only contract checks, with no generated graphs or physics.

Run directly with stdlib unittest for preparation-only checks.
Every graph below is a three-node manual fixture. These tests establish API,
identity, arithmetic, admission and ownership contracts, not physical validity.
"""
from contextlib import ExitStack, contextmanager
import hashlib
import importlib
import importlib.abc
import json
import pickle
import sys
import unittest
from unittest.mock import patch

import numpy as np

from fibernet.core.material import Material
from fibernet.core.structure_graph import SEdge, SNode, StructureGraph


_FORBIDDEN_IMPORTS = (
    'taichi', 'torch', 'scipy', 'fibernet.generators', 'fibernet.ml',
    'fibernet.sim.accelerated', 'fibernet.sim.linear_transport',
    'fibernet.sim.transient_thermal', 'fibernet.sim.electrothermal',
    'fibernet.sim.joule_heating', 'fibernet.sim.thermoelastic',
    'fibernet.sim._thermoelastic', 'fibernet.sim.reduced_beam',
)


class _NoPhysicsImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.')
               for name in _FORBIDDEN_IMPORTS):
            raise AssertionError('Preparation must not import provider: ' + fullname)
        return None


def _forbidden_numerics(*args, **kwargs):
    raise AssertionError('Preparation must not call a solver or factorization')


@contextmanager
def _no_physics():
    blocker = _NoPhysicsImports()
    sys.meta_path.insert(0, blocker)
    try:
        with ExitStack() as stack:
            for name in ('solve', 'lstsq', 'inv', 'pinv', 'cholesky', 'eig',
                         'eigh', 'eigvals', 'eigvalsh', 'svd', 'qr'):
                stack.enter_context(patch.object(np.linalg, name,
                                                  side_effect=_forbidden_numerics))
            yield
    finally:
        sys.meta_path.remove(blocker)


class _GraphSubclass(StructureGraph):
    pass


class _NodeSubclass(SNode):
    pass


class _EdgeSubclass(SEdge):
    pass


class _MaterialSubclass(Material):
    pass


_SCALE = {'m': 1.0, 'mm': 1e-3, 'um': 1e-6}
_NODE_ORDER = [42, 105, 7]
_EDGE_ORDER = [307, 9, 61]
_REFERENCE_G = {61: 10.0, 307: 4.0, 9: 6.0}
_TRANSLATIONS = {105: [1.5, 0.0], 42: [0.0, 0.0], 7: [0.5, 0.0]}


def _manual_graph(length_unit='m', plane_z_m=0.0):
    """Unequal series chords (2 m, 3 m), with a reversed parallel edge."""
    scale = _SCALE[length_unit]
    positions_m = {42: [0.0, 0.0, plane_z_m],
                   105: [5.0, 0.0, plane_z_m],
                   7: [2.0, 0.0, plane_z_m]}
    material = {
        'name': 'manual-test-material', 'density': 1234.0,
        'youngs_modulus': 2e6, 'poissons_ratio': 0.25,
        'electrical_conductivity': 999.0,
        'stiffness_tensor': (np.eye(6) * 3.0).tolist(),
        'extra': {'calibration': {'values': [1.0, 2.0]}},
    }
    edges = {
        307: (7, 42, 0.03, 7, None),
        9: (7, 105, 0.02, 3, None),
        61: (42, 7, 0.01, 5, []),
    }
    return StructureGraph.from_dict({
        'version': '3.1', 'dimension': 2,
        'tolerance': 1e-8 / scale,
        'box_size': [6.0 / scale, 2.0 / scale, 0.0],
        'next_node_id': 1000, 'next_edge_id': 5000,
        'metadata': {'fixture': 'manual-only', 'coordinate_frame': 'lab_xyz',
                     'nested': {'values': [2, 3]}},
        'nodes': [{
            'id': nid, 'position': [v / scale for v in positions_m[nid]],
            'boundary': [nid == 42, nid == 105, False, False, False, False],
            'metadata': {'label': str(nid), 'nested': {'values': [nid]}},
        } for nid in _NODE_ORDER],
        'edges': [{
            'id': eid, 'node_i': edges[eid][0], 'node_j': edges[eid][1],
            'radius': edges[eid][2] / scale, 'segments': edges[eid][3],
            'internal_points': edges[eid][4], 'material': material,
            'metadata': {'label': str(eid), 'nested': {'values': [eid]}},
        } for eid in _EDGE_ORDER],
    })


def _canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def _full_hash(graph):
    return hashlib.sha256(_canonical_bytes(graph.to_dict())).hexdigest()


def _zero_translations(graph):
    return {nid: [0.0, 0.0] for nid in graph.nodes}


class DeformedTransportPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import the new module under tripwires, without importing providers to
        # monkeypatch them.
        before = set(sys.modules)
        with _no_physics():
            cls.module = importlib.import_module('fibernet.sim.deformed_transport')
        unexpected = [name for name in set(sys.modules) - before
                      if any(name == prefix or name.startswith(prefix + '.')
                             for prefix in _FORBIDDEN_IMPORTS)]
        if unexpected:
            raise AssertionError('Unexpected provider imports: ' + repr(unexpected))

    def setUp(self):
        self._guard = _no_physics()
        self._guard.__enter__()
        self.addCleanup(self._guard.__exit__, None, None, None)

    def prepare(self, graph, **kwargs):
        kwargs.setdefault('translations_m_by_id', _zero_translations(graph))
        kwargs.setdefault('reference_conductance_S', dict(_REFERENCE_G))
        kwargs.setdefault('length_unit', 'm')
        return self.module.prepare_deformed_transport(graph, **kwargs)

    def assertRejected(self, graph, **kwargs):
        # Pickle is used only as an in-memory mutation sentinel; no pickle is
        # loaded and no external fixture is read. It supports cyclic bad input.
        before = pickle.dumps(graph, protocol=5)
        with self.assertRaises((TypeError, ValueError, MemoryError)):
            self.prepare(graph, **kwargs)
        self.assertEqual(pickle.dumps(graph, protocol=5), before,
                         'Rejected preparation changed its source graph')

    def assertSpatialMembership(self, graph):
        members = []
        for key, bucket in graph._spatial_hash.items():
            self.assertIs(type(key), tuple)
            self.assertEqual(len(key), 3)
            for nid in bucket:
                self.assertIn(nid, graph.nodes)
                self.assertEqual(key, graph._pos_key(graph.nodes[nid].position))
                members.append(nid)
        self.assertCountEqual(members, list(graph.nodes))
        self.assertEqual(len(members), len(set(members)))

    def assertFiniteJson(self, value):
        serialized = _canonical_bytes(value)
        self.assertEqual(json.loads(serialized), value)
        return serialized

    def test_unequal_series_parallel_id_alignment_and_exact_arithmetic(self):
        graph = _manual_graph()
        original = pickle.dumps(graph, protocol=5)
        derived, conductance, report = self.prepare(
            graph, translations_m_by_id={k: list(v) for k, v in _TRANSLATIONS.items()})
        self.assertIs(type(derived), StructureGraph)
        self.assertIs(type(conductance), dict)
        self.assertIs(type(report), dict)
        self.assertEqual(list(derived.nodes), _NODE_ORDER)
        self.assertEqual(list(derived.edges), _EDGE_ORDER)
        self.assertEqual({eid: (e.node_i, e.node_j) for eid, e in derived.edges.items()},
                         {307: (7, 42), 9: (7, 105), 61: (42, 7)})
        self.assertEqual(set(conductance), set(_REFERENCE_G))
        self.assertTrue(all(type(eid) is int for eid in conductance))
        for eid, expected in {61: 8.0, 9: 4.5, 307: 3.2}.items():
            self.assertAlmostEqual(conductance[eid], expected, places=14)
        for nid, expected in {42: [0., 0., 0.], 7: [2.5, 0., 0.],
                              105: [6.5, 0., 0.]}.items():
            np.testing.assert_array_equal(derived.nodes[nid].position, expected)
        self.assertEqual(report['node_ids'], [7, 42, 105])
        self.assertEqual(report['edge_ids'], [9, 61, 307])
        self.assertEqual(report['translations_m'], [[.5, 0.], [0., 0.], [1.5, 0.]])
        self.assertEqual(report['endpoints_by_id'], [[7, 105], [42, 7], [7, 42]])
        self.assertEqual(report['reference_lengths_m'], [3., 2., 2.])
        self.assertEqual(report['derived_lengths_m'], [4., 2.5, 2.5])
        self.assertEqual(report['reference_conductance_S'], [6., 10., 4.])
        np.testing.assert_allclose(report['conductance_S'], [4.5, 8., 3.2], rtol=1e-15)
        self.assertEqual(report['frozen_sigma_A_S_m'], [18., 20., 8.])
        self.assertEqual(pickle.dumps(graph, protocol=5), original)
        self.assertSpatialMembership(graph)
        self.assertSpatialMembership(derived)

    def test_report_is_finite_json_with_explicit_prepare_only_scope(self):
        graph = _manual_graph()
        derived, _, report = self.prepare(graph)
        expected = {
            'schema': 'fibernet.deformed_transport_preparation/1',
            'scope': 'preparation_time_correspondence_only',
            'input_kind': 'caller_prescribed_translations', 'length_unit': 'm',
            'translation_unit': 'm', 'conductance_unit': 'S',
            'coordinate_frame': 'lab_xyz', 'translation_frame': 'source_graph_xy_axes',
            'geometry_policy': 'endpoint_chord', 'area_policy': 'fixed_effective_area',
            'conductivity_policy': 'frozen_reference_sigma_A',
            'source_node_insertion_order': _NODE_ORDER,
            'source_edge_insertion_order': _EDGE_ORDER,
            'reference_box_size_m': [6., 2., 0.],
        }
        for name, value in expected.items():
            self.assertEqual(report[name], value, name)
        for name in ('source_full_graph_sha256', 'derived_full_graph_sha256',
                     'source_graph_order_envelope_sha256',
                     'derived_graph_order_envelope_sha256', 'input_sha256'):
            self.assertRegex(report[name], r'^[0-9a-f]{64}$', name)
        self.assertEqual(report['source_full_graph_sha256'], _full_hash(graph))
        self.assertEqual(report['derived_full_graph_sha256'], _full_hash(derived))
        self.assertEqual(report['derived_node_AABB_m'], [[0., 0., 0.], [5., 0., 0.]])
        for prefix, owner in (('source', graph), ('derived', derived)):
            envelope = {'graph': owner.to_dict(),
                        'node_insertion_order': list(owner.nodes),
                        'edge_insertion_order': list(owner.edges)}
            self.assertEqual(report[prefix + '_graph_order_envelope_sha256'],
                             hashlib.sha256(_canonical_bytes(envelope)).hexdigest())
        self.assertIn('size_admission', report)
        self.assertFiniteJson(report)

    def test_zero_and_rigid_translation_preserve_chords_and_conductance(self):
        for translation in ([0., 0.], [1.25, -2.5]):
            with self.subTest(translation=translation):
                graph = _manual_graph(plane_z_m=.125)
                shifts = {nid: list(translation) for nid in graph.nodes}
                derived, conductance, report = self.prepare(graph, translations_m_by_id=shifts)
                self.assertEqual(conductance, _REFERENCE_G)
                self.assertEqual(report['reference_lengths_m'], report['derived_lengths_m'])
                for nid in graph.nodes:
                    np.testing.assert_array_equal(
                        derived.nodes[nid].position,
                        graph.nodes[nid].position + [*translation, 0.])
                    self.assertEqual(derived.nodes[nid].position[2], .125)
                self.assertSpatialMembership(graph)
                self.assertSpatialMembership(derived)

    def test_units_m_mm_um_represent_same_si_preparation(self):
        reports = []
        for unit, scale in _SCALE.items():
            graph = _manual_graph(unit, plane_z_m=.125)
            derived, conductance, report = self.prepare(
                graph, length_unit=unit,
                translations_m_by_id={nid: list(v) for nid, v in _TRANSLATIONS.items()})
            self.assertEqual(report['length_unit'], unit)
            np.testing.assert_allclose([conductance[eid] for eid in [9, 61, 307]],
                                       [4.5, 8., 3.2], rtol=1e-14)
            for nid in graph.nodes:
                expected = graph.nodes[nid].position * scale
                expected[:2] += _TRANSLATIONS[nid]
                np.testing.assert_allclose(derived.nodes[nid].position * scale,
                                           expected, rtol=1e-14)
                self.assertEqual(derived.nodes[nid].position[2], graph.nodes[nid].position[2])
            reports.append(report)
        for key in ('reference_lengths_m', 'derived_lengths_m', 'conductance_S',
                    'frozen_sigma_A_S_m', 'reference_box_size_m'):
            np.testing.assert_allclose(reports[0][key], reports[1][key], rtol=1e-14)
            np.testing.assert_allclose(reports[0][key], reports[2][key], rtol=1e-14)

    def test_every_noncoordinate_graph_field_is_preserved_and_deeply_owned(self):
        graph = _manual_graph()
        before = graph.to_dict()
        derived, _, _ = self.prepare(graph, translations_m_by_id=_TRANSLATIONS)
        expected = json.loads(_canonical_bytes(before))
        for node in expected['nodes']:
            node['position'][:2] = (
                np.asarray(node['position'][:2]) + _TRANSLATIONS[node['id']]).tolist()
        self.assertEqual(derived.to_dict(), expected)
        self.assertEqual(derived._next_node_id, 1000)
        self.assertEqual(derived._next_edge_id, 5000)
        self.assertIsNot(derived._box_size, graph._box_size)
        self.assertFalse(np.shares_memory(derived._box_size, graph._box_size))
        self.assertIsNot(derived.metadata, graph.metadata)
        self.assertIsNot(derived.metadata['nested']['values'], graph.metadata['nested']['values'])
        for nid in graph.nodes:
            old, new = graph.nodes[nid], derived.nodes[nid]
            self.assertIsNot(old, new)
            self.assertFalse(np.shares_memory(old.position, new.position))
            self.assertEqual(old.boundary, new.boundary)
            self.assertIsNot(old.metadata['nested']['values'], new.metadata['nested']['values'])
        for eid in graph.edges:
            old, new = graph.edges[eid], derived.edges[eid]
            self.assertIsNot(old, new)
            self.assertIsNot(old.material, new.material)
            self.assertFalse(np.shares_memory(old.material.stiffness_tensor,
                                             new.material.stiffness_tensor))
            self.assertIsNot(old.material.extra['calibration']['values'],
                             new.material.extra['calibration']['values'])
            self.assertIsNot(old.metadata['nested']['values'], new.metadata['nested']['values'])
        self.assertEqual(derived.edges[61].internal_points.shape, (0, 3))
        self.assertIsNot(derived.edges[61].internal_points, graph.edges[61].internal_points)
        self.assertIsNone(derived.edges[9].internal_points)
        source_bytes = _canonical_bytes(graph.to_dict())
        derived.metadata['nested']['values'].append(99)
        derived.nodes[7].metadata['nested']['values'].append(99)
        derived.edges[61].metadata['nested']['values'].append(99)
        derived.edges[61].material.extra['calibration']['values'].append(99)
        derived.edges[61].material.stiffness_tensor[0, 0] = 88
        derived._box_size[0] = 77
        derived.edges[61].radius = 66
        self.assertEqual(_canonical_bytes(graph.to_dict()), source_bytes)

    def test_returned_maps_and_report_are_detached_from_all_inputs(self):
        graph = _manual_graph()
        translations = {nid: list(v) for nid, v in _TRANSLATIONS.items()}
        reference = dict(_REFERENCE_G)
        derived, conductance, report = self.prepare(
            graph, translations_m_by_id=translations, reference_conductance_S=reference)
        report_bytes = self.assertFiniteJson(report)
        derived_bytes = _canonical_bytes(derived.to_dict())
        translations[7][0] = 999.
        reference[61] = 999.
        graph.metadata['nested']['values'].append(999)
        graph.nodes[7].metadata['nested']['values'].append(999)
        graph.edges[61].material.extra['calibration']['values'].append(999)
        conductance[61] = 888.
        self.assertEqual(_canonical_bytes(report), report_bytes)
        self.assertEqual(_canonical_bytes(derived.to_dict()), derived_bytes)
        report['translations_m'][0][0] = 777.
        report['conductance_S'][1] = 777.
        self.assertEqual(_canonical_bytes(derived.to_dict()), derived_bytes)
        self.assertEqual(conductance[61], 888.)

    def test_spatial_lookup_uses_derived_coordinates_without_changing_source(self):
        graph = _manual_graph()
        source_lookup = pickle.dumps(graph._spatial_hash, protocol=5)
        derived, _, _ = self.prepare(graph, translations_m_by_id=_TRANSLATIONS)
        self.assertEqual(pickle.dumps(graph._spatial_hash, protocol=5), source_lookup)
        self.assertSpatialMembership(derived)
        for nid in (7, 105):
            old_key = graph._pos_key(graph.nodes[nid].position)
            self.assertNotIn(nid, derived._spatial_hash.get(old_key, []))
            self.assertIn(nid, derived._spatial_hash[derived._pos_key(derived.nodes[nid].position)])
        # Public lookup should find existing nodes, not merge or renumber them.
        for owner in (graph, derived):
            count = owner.num_nodes
            for nid in _NODE_ORDER:
                self.assertEqual(owner.add_node(owner.nodes[nid].position, merge=True), nid)
            self.assertEqual(owner.num_nodes, count)

    def test_tiny_change_hidden_by_fingerprint_changes_full_binding(self):
        graph = _manual_graph()
        altered = graph.copy()
        altered.set_node_position(7, [2. + 1e-12, 0., 0.])
        self.assertEqual(graph.fingerprint(), altered.fingerprint())
        _, _, original = self.prepare(graph)
        _, _, changed = self.prepare(altered)
        self.assertNotEqual(original['source_full_graph_sha256'], changed['source_full_graph_sha256'])
        self.assertNotEqual(original['input_sha256'], changed['input_sha256'])

    def test_metadata_material_counter_changes_are_bound_beyond_fingerprint(self):
        graph = _manual_graph()
        _, _, original = self.prepare(graph)
        for mutation in (
            lambda g: g.metadata.update({'new': 'retained'}),
            lambda g: setattr(g.edges[61].material, 'electrical_conductivity', 3.),
            lambda g: setattr(g, '_next_node_id', 1001),
            lambda g: setattr(g.edges[61], 'segments', 8),
        ):
            altered = graph.copy()
            mutation(altered)
            with self.subTest(mutation=mutation):
                self.assertEqual(graph.fingerprint(), altered.fingerprint())
                _, _, changed = self.prepare(altered)
                self.assertNotEqual(original['source_full_graph_sha256'], changed['source_full_graph_sha256'])

    def test_insertion_order_has_separate_hash_and_is_preserved(self):
        graph = _manual_graph()
        reordered = graph.copy()
        for mapping in (reordered.nodes, reordered.edges):
            pairs = list(mapping.items())[::-1]
            mapping.clear()
            mapping.update(pairs)
        self.assertEqual(_full_hash(graph), _full_hash(reordered))
        _, _, original = self.prepare(graph)
        derived, _, changed = self.prepare(reordered)
        self.assertEqual(original['source_full_graph_sha256'], changed['source_full_graph_sha256'])
        self.assertNotEqual(original['source_graph_order_envelope_sha256'],
                            changed['source_graph_order_envelope_sha256'])
        self.assertEqual(list(derived.nodes), list(reordered.nodes))
        self.assertEqual(list(derived.edges), list(reordered.edges))
        self.assertEqual(changed['source_node_insertion_order'], list(reordered.nodes))
        self.assertEqual(changed['source_edge_insertion_order'], list(reordered.edges))

    def test_input_mapping_insertion_order_does_not_change_identity(self):
        graph = _manual_graph()
        _, ga, a = self.prepare(graph, translations_m_by_id=_TRANSLATIONS,
                                reference_conductance_S=_REFERENCE_G)
        _, gb, b = self.prepare(graph,
                                translations_m_by_id=dict(reversed(list(_TRANSLATIONS.items()))),
                                reference_conductance_S=dict(reversed(list(_REFERENCE_G.items()))))
        self.assertEqual(ga, gb)
        self.assertEqual(a, b)

    def test_list_tuple_and_numeric_array_translation_rows_are_accepted(self):
        for row in ([.5, 0.], (.5, 0.), np.array([.5, 0.]),
                    np.array([1, 0], dtype=np.int64)):
            with self.subTest(row=repr(row)):
                graph = _manual_graph()
                translations = _zero_translations(graph)
                translations[7] = row
                derived, _, _ = self.prepare(graph, translations_m_by_id=translations)
                np.testing.assert_array_equal(derived.nodes[7].position,
                                              [2. + float(row[0]), float(row[1]), 0.])

    def test_nonrepresentable_integer_translation_components_are_rejected(self):
        # NumPy scalar equality may coerce both sides to the same rounded float.
        # Admission must compare integer values rather than use that equality.
        invalid = (2 ** 53 + 1, -(2 ** 53) - 1,
                   np.int64(2 ** 53 + 1), np.int64(-(2 ** 53) - 1),
                   np.uint64(2 ** 53 + 1), np.int64(2 ** 63 - 1),
                   np.int64(-(2 ** 63) + 1), np.uint64(2 ** 63 + 1),
                   np.uint64(2 ** 64 - 1))
        for value in invalid:
            rows = [[value, 0]]
            if isinstance(value, np.integer):
                rows.append(np.array([value, 0], dtype=type(value)))
            for row in rows:
                graph = _manual_graph()
                translations = _zero_translations(graph)
                translations[105] = row
                before = pickle.dumps(graph, protocol=5)
                with self.subTest(value=int(value), kind=type(value).__name__, row=type(row).__name__):
                    with self.assertRaisesRegex(ValueError, 'exactly representable'):
                        self.prepare(graph, translations_m_by_id=translations)
                    self.assertEqual(pickle.dumps(graph, protocol=5), before)

    def test_nonrepresentable_integer_geometry_and_material_tensors_are_rejected(self):
        invalid = (np.int64(2 ** 53 + 1), np.uint64(2 ** 53 + 1),
                   np.int64(2 ** 63 - 1), np.int64(-(2 ** 63) + 1),
                   np.uint64(2 ** 63 + 1), np.uint64(2 ** 64 - 1))
        for value in invalid:
            for location in ('position', 'material_tensor'):
                graph = _manual_graph()
                if location == 'position':
                    graph.nodes[105].position = np.array([value, 0, 0], dtype=type(value))
                    # Keep the lookup consistent, so a stale-index rejection
                    # cannot hide a failure of the numeric exactness check.
                    graph._spatial_hash.clear()
                    for nid, node in graph.nodes.items():
                        graph._spatial_hash.setdefault(graph._pos_key(node.position), []).append(nid)
                else:
                    tensor = np.zeros((6, 6), dtype=type(value))
                    tensor[0, 0] = value
                    graph.edges[61].material.stiffness_tensor = tensor
                before = pickle.dumps(graph, protocol=5)
                with self.subTest(value=int(value), kind=type(value).__name__, location=location):
                    with self.assertRaisesRegex(ValueError, 'exactly representable'):
                        self.prepare(graph)
                    self.assertEqual(pickle.dumps(graph, protocol=5), before)

    def test_exact_large_integer_translation_geometry_and_tensor_are_accepted(self):
        for scalar_type in (np.int64, np.uint64):
            value = scalar_type(2 ** 53)
            for location in ('translation', 'position', 'material_tensor'):
                graph = _manual_graph()
                translations = _zero_translations(graph)
                if location == 'translation':
                    translations[105] = np.array([value, 0], dtype=scalar_type)
                elif location == 'position':
                    graph.nodes[105].position = np.array([value, 0, 0], dtype=scalar_type)
                    graph._spatial_hash.clear()
                    for nid, node in graph.nodes.items():
                        graph._spatial_hash.setdefault(graph._pos_key(node.position), []).append(nid)
                else:
                    tensor = np.zeros((6, 6), dtype=scalar_type)
                    tensor[0, 0] = value
                    graph.edges[61].material.stiffness_tensor = tensor
                with self.subTest(kind=scalar_type.__name__, location=location):
                    derived, conductance, report = self.prepare(
                        graph, translations_m_by_id=translations)
                    self.assertTrue(all(np.isfinite(v) and v > 0 for v in conductance.values()))
                    self.assertFiniteJson(report)
                    if location == 'translation':
                        self.assertEqual(report['translations_m'][report['node_ids'].index(105)][0], float(value))
                    elif location == 'position':
                        self.assertEqual(derived.nodes[105].position[0], float(value))
                    else:
                        self.assertEqual(int(derived.edges[61].material.stiffness_tensor[0, 0]), int(value))

    def test_empty_numeric_curve_arrays_are_accepted_and_detached(self):
        for dtype in (np.float32, np.float64, np.int64):
            graph = _manual_graph()
            graph.edges[61].internal_points = np.empty((0, 3), dtype=dtype)
            derived, _, _ = self.prepare(graph)
            self.assertEqual(derived.edges[61].internal_points.shape, (0, 3))
            self.assertIsNot(derived.edges[61].internal_points, graph.edges[61].internal_points)

    def test_length_unit_is_required_and_never_inferred(self):
        graph = _manual_graph()
        common = {'translations_m_by_id': _zero_translations(graph),
                  'reference_conductance_S': dict(_REFERENCE_G)}
        with self.assertRaises(TypeError):
            self.module.prepare_deformed_transport(graph, **common)
        for bad in (None, '', ' ', 'cm', 'meters', True, 1):
            with self.subTest(length_unit=bad):
                self.assertRejected(_manual_graph(), length_unit=bad)
        # No caller coordinate-frame option is part of this first bounded API.
        with self.assertRaises(TypeError):
            self.module.prepare_deformed_transport(graph, length_unit='m',
                coordinate_frame='lab_xyz', **common)

    def test_absent_frame_is_reported_null_without_relabeling(self):
        graph = _manual_graph()
        del graph.metadata['coordinate_frame']
        source = _canonical_bytes(graph.to_dict())
        derived, _, report = self.prepare(graph)
        self.assertIsNone(report['coordinate_frame'])
        self.assertEqual(report['translation_frame'], 'source_graph_xy_axes')
        self.assertEqual(_canonical_bytes(derived.to_dict()), source)
        self.assertNotIn('coordinate_frame', derived.metadata)

    def test_blank_and_nonstrings_stored_frames_are_rejected(self):
        for location in ('graph', 'node', 'edge'):
            for bad in (None, '', ' ', True, 1, ['lab_xyz']):
                graph = _manual_graph()
                target = {'graph': graph.metadata, 'node': graph.nodes[7].metadata,
                          'edge': graph.edges[61].metadata}[location]
                target['coordinate_frame'] = bad
                with self.subTest(location=location, value=repr(bad)):
                    self.assertRejected(graph)

    def test_consistent_root_geometry_unit_frame_labels_are_preserved(self):
        graph = _manual_graph()
        graph.metadata.update({'length_unit': 'm', 'coordinate_frame': 'lab_xyz',
                               'geometry': {'length_unit': 'm', 'coordinate_frame': 'lab_xyz'}})
        graph.nodes[7].metadata.update({'length_unit': 'm', 'coordinate_frame': 'lab_xyz'})
        graph.edges[61].metadata['geometry'] = {'length_unit': 'm', 'coordinate_frame': 'lab_xyz'}
        before = _canonical_bytes(graph.metadata)
        derived, _, report = self.prepare(graph)
        self.assertEqual(_canonical_bytes(derived.metadata), before)
        self.assertEqual(report['coordinate_frame'], 'lab_xyz')
        self.assertEqual(report['translation_frame'], 'source_graph_xy_axes')

    def test_unrelated_nested_units_and_frames_are_uninterpreted_data(self):
        graph = _manual_graph()
        graph.metadata.update({
            'length_unit': 'm', 'coordinate_frame': 'lab_xyz',
            'load': {'length_unit': 'um', 'coordinate_frame': 'load_axes'},
            'stress': {'units': 'Pa', 'coordinate_frame': 'principal_axes'},
            'calibration': {'length_unit': 'mm', 'coordinate_frame': 'camera'},
            'geometry': {
                'length_unit': 'm', 'coordinate_frame': 'lab_xyz',
                'calibration': {'length_unit': 'nm', 'coordinate_frame': 'sensor'},
                'history': [{'length_unit': 'mm', 'coordinate_frame': 'old_frame'}],
            },
        })
        graph.edges[61].material.extra.update({
            'units': 'Pa', 'length_unit': 'mm', 'coordinate_frame': 'crystal',
            'geometry': {'length_unit': 'angstrom', 'coordinate_frame': 'material_axes'},
        })
        source = _canonical_bytes(graph.to_dict())
        derived, _, report = self.prepare(graph)
        self.assertEqual(_canonical_bytes(derived.to_dict()), source)
        self.assertEqual(report['coordinate_frame'], 'lab_xyz')

    def test_contradictory_root_or_direct_geometry_declarations_are_rejected(self):
        for branch in ('root', 'geometry'):
            for key, value in (('length_unit', 'mm'), ('coordinate_frame', 'camera_xyz')):
                with self.subTest(branch=branch, key=key):
                    graph = _manual_graph()
                    graph.metadata['geometry'] = {'length_unit': 'm', 'coordinate_frame': 'lab_xyz'}
                    target = graph.metadata if branch == 'root' else graph.metadata['geometry']
                    target[key] = value
                    self.assertRejected(graph)

    def test_existing_units_walker_scope_on_graph_node_and_edge_metadata(self):
        length_keys = ('length_unit', 'coordinate_unit', 'coordinate_units',
                       'position_unit', 'radius_unit', 'internal_points_unit',
                       'curve_unit', 'box_size_unit')
        unit_fields = ('length', 'coordinates', 'position', 'radius',
                       'internal_points', 'curve', 'box_size')
        for location in ('graph', 'node', 'edge'):
            for branch in ('root', 'nested_geometry'):
                for key in length_keys:
                    graph = _manual_graph()
                    target = {'graph': graph.metadata, 'node': graph.nodes[7].metadata,
                              'edge': graph.edges[61].metadata}[location]
                    if branch == 'nested_geometry':
                        target = target.setdefault('calibration', {}).setdefault('geometry', {})
                    target[key] = 'mm'
                    with self.subTest(location=location, branch=branch, key=key):
                        self.assertRejected(graph)
                for value in ('mm', {'length': 'mm'}, None):
                    graph = _manual_graph()
                    target = {'graph': graph.metadata, 'node': graph.nodes[7].metadata,
                              'edge': graph.edges[61].metadata}[location]
                    if branch == 'nested_geometry':
                        target = target.setdefault('load', {}).setdefault('geometry', {})
                    target['units'] = value
                    with self.subTest(location=location, branch=branch, units=repr(value)):
                        self.assertRejected(graph)
            graph = _manual_graph()
            target = {'graph': graph.metadata, 'node': graph.nodes[7].metadata,
                      'edge': graph.edges[61].metadata}[location]
            target.update({key: 'm' for key in length_keys})
            target['units'] = {field: 'm' for field in unit_fields}
            self.prepare(graph)

    def test_explicit_geometry_frame_nested_under_other_metadata_is_recognized(self):
        for location in ('graph', 'node', 'edge'):
            graph = _manual_graph()
            target = {'graph': graph.metadata, 'node': graph.nodes[7].metadata,
                      'edge': graph.edges[61].metadata}[location]
            target['calibration'] = {'geometry': {'coordinate_frame': 'camera_xyz'}}
            with self.subTest(location=location):
                self.assertRejected(graph)
        graph = _manual_graph()
        del graph.metadata['coordinate_frame']
        graph.nodes[7].metadata['load'] = {'geometry': {'coordinate_frame': 'retained_axes'}}
        derived, _, report = self.prepare(graph)
        self.assertEqual(report['coordinate_frame'], 'retained_axes')
        self.assertEqual(derived.metadata, graph.metadata)
        self.assertEqual(derived.nodes[7].metadata, graph.nodes[7].metadata)

    def test_bad_translation_rows_are_rejected_without_source_mutation(self):
        bad_rows = (None, 1., '0,0', [], [0.], [0., 0., 0.],
                    [[0., 0.]], [True, 0.], [np.bool_(False), 0.],
                    ['0', 0.], [complex(0, 0), 0.], [float('nan'), 0.],
                    [float('inf'), 0.], [0., -float('inf')],
                    np.array([[0., 0.]]), np.array([False, False]),
                    np.array(['0', '0']), np.array([0., 0.], dtype=object))
        for row in bad_rows:
            with self.subTest(row=repr(row)):
                graph = _manual_graph()
                translations = _zero_translations(graph)
                translations[7] = row
                self.assertRejected(graph, translations_m_by_id=translations)

    def test_translation_maps_require_exact_complete_python_integer_ids(self):
        graph = _manual_graph()
        zero = _zero_translations(graph)
        maps = [{nid: value for nid, value in zero.items() if nid != 7},
                dict(zero, **{'7': [0., 0.]}),
                {**zero, 99: [0., 0.]}, {**zero, True: [0., 0.]},
                [(nid, value) for nid, value in zero.items()], None]
        for invalid in (np.int64(7), 7.0, '7', -1, 2 ** 63):
            value = {nid: row for nid, row in zero.items() if nid != 7}
            value[invalid] = [0., 0.]
            maps.append(value)
        for translations in maps:
            with self.subTest(translations=repr(translations)):
                self.assertRejected(graph, translations_m_by_id=translations)

    def test_conductance_requires_complete_exact_edge_ids(self):
        maps = [{eid: value for eid, value in _REFERENCE_G.items() if eid != 61},
                {**_REFERENCE_G, 99: 1.}, {**_REFERENCE_G, True: 1.},
                list(_REFERENCE_G.items()), None]
        for invalid in (np.int64(61), 61.0, '61', -1, 2 ** 63):
            values = {eid: value for eid, value in _REFERENCE_G.items() if eid != 61}
            values[invalid] = 10.
            maps.append(values)
        for conductance in maps:
            with self.subTest(conductance=repr(conductance)):
                self.assertRejected(_manual_graph(), reference_conductance_S=conductance)

    def test_nonpositive_nonfinite_and_nonnumeric_conductance_are_rejected(self):
        for bad in (0., -1., float('nan'), float('inf'), -float('inf'),
                    True, np.bool_(False), '10', 1 + 0j, [10.], np.array([10.])):
            values = dict(_REFERENCE_G)
            values[61] = bad
            with self.subTest(value=repr(bad)):
                self.assertRejected(_manual_graph(), reference_conductance_S=values)

    def test_nonplanar_and_invalid_stored_positions_are_rejected(self):
        graph = _manual_graph()
        graph._dimension = 3
        self.assertRejected(graph)
        graph = _manual_graph()
        graph.set_node_position(7, [2., 0., .01])
        self.assertRejected(graph)
        for bad in (np.array([2., 0.]), np.array([2., 0., 0., 0.]),
                    np.array([[2., 0., 0.]]), np.array([2., np.nan, 0.]),
                    np.array([2., np.inf, 0.]), np.array(['2', '0', '0']),
                    np.array([2., 0., 0.], dtype=object), [2., 0., 0.]):
            graph = _manual_graph()
            graph.nodes[7].position = bad
            with self.subTest(position=repr(bad)):
                self.assertRejected(graph)

    def test_nonempty_or_malformed_curve_data_are_rejected(self):
        for points in (np.array([[1., 0., 0.]]), np.empty((0,)), np.empty((0, 2)),
                       np.empty((0, 3, 1)), np.empty((0, 3), dtype=bool),
                       np.empty((0, 3), dtype=object), np.empty((0, 3), dtype=str),
                       np.array([[np.nan, 0., 0.]]), []):
            graph = _manual_graph()
            graph.edges[61].internal_points = points
            with self.subTest(shape=getattr(points, 'shape', None), dtype=getattr(points, 'dtype', None)):
                self.assertRejected(graph)

    def test_source_and_derived_collapsed_chords_are_rejected(self):
        graph = _manual_graph()
        graph.set_node_position(105, graph.nodes[7].position)
        self.assertRejected(graph)
        graph = _manual_graph()
        translations = _zero_translations(graph)
        translations[105] = [-3., 0.]
        self.assertRejected(graph, translations_m_by_id=translations)

    def test_nonfinite_derived_position_and_conductance_are_rejected(self):
        graph = _manual_graph('um')
        translations = _zero_translations(graph)
        translations[105] = [np.finfo(float).max, 0.]
        self.assertRejected(graph, translations_m_by_id=translations, length_unit='um')
        graph = _manual_graph()
        values = dict(_REFERENCE_G)
        values[61] = np.finfo(float).max
        self.assertRejected(graph, reference_conductance_S=values)
        graph = _manual_graph()
        translations = _zero_translations(graph)
        translations[105] = [1e100, 0.]
        values = dict(_REFERENCE_G)
        values[9] = np.nextafter(0., 1.)
        self.assertRejected(graph, translations_m_by_id=translations,
                            reference_conductance_S=values)

    def test_source_entities_and_map_ids_must_be_ordinary_and_consistent(self):
        for field, invalid in (('node', np.int64(7)), ('node', 7.0),
                               ('node', -1), ('node', 2 ** 63),
                               ('edge', np.int64(61)), ('edge', 61.0),
                               ('edge', -1), ('edge', 2 ** 63)):
            graph = _manual_graph()
            mapping, old_id = (graph.nodes, 7) if field == 'node' else (graph.edges, 61)
            mapping[invalid] = mapping.pop(old_id)
            with self.subTest(field=field, invalid=repr(invalid)):
                self.assertRejected(graph)
        for mutate in (
            lambda g: setattr(g.nodes[7], 'node_id', 42),
            lambda g: setattr(g.edges[61], 'edge_id', 9),
            lambda g: setattr(g.edges[61], 'node_i', 999),
            lambda g: setattr(g.edges[61], 'node_i', np.int64(42)),
            lambda g: setattr(g.edges[61], 'node_j', 42),
            lambda g: setattr(g, '_next_node_id', 105),
            lambda g: setattr(g, '_next_edge_id', True),
        ):
            graph = _manual_graph()
            mutate(graph)
            self.assertRejected(graph)
        graph = _manual_graph()
        graph.nodes[7] = _NodeSubclass(**vars(graph.nodes[7]))
        self.assertRejected(graph)
        graph = _manual_graph()
        graph.edges[61] = _EdgeSubclass(**vars(graph.edges[61]))
        self.assertRejected(graph)
        graph = _GraphSubclass()
        graph.__dict__.update(_manual_graph().__dict__)
        self.assertRejected(graph)
        with self.assertRaises((TypeError, ValueError)):
            self.module.prepare_deformed_transport({}, translations_m_by_id={},
                reference_conductance_S={}, length_unit='m')

    def test_optional_material_none_and_empty_extra_preserved_without_redefault(self):
        graph = _manual_graph()
        material = graph.edges[61].material
        material.name = 'optional-none-control'
        material.density = 987.0
        material.youngs_modulus = 7654.0
        material.poissons_ratio = .125
        required = {'name', 'density', 'youngs_modulus', 'poissons_ratio', 'extra'}
        for name in Material.__dataclass_fields__:
            if name not in required:
                setattr(material, name, None)
        # In particular, set shear_modulus after initialization: a reconstruction
        # through Material(**fields) would incorrectly recompute this None.
        material.shear_modulus = None
        material.stiffness_tensor = None
        material.extra = {}
        before = pickle.dumps(graph, protocol=5)
        derived, _, _ = self.prepare(graph)
        copied = derived.edges[61].material
        self.assertIs(type(copied), Material)
        self.assertIsNot(copied, material)
        self.assertEqual(vars(copied), vars(material))
        self.assertIsNone(copied.shear_modulus)
        self.assertIsNone(copied.stiffness_tensor)
        for name in Material.__dataclass_fields__:
            if name not in required:
                self.assertIsNone(getattr(copied, name), name)
        self.assertEqual(copied.extra, {})
        self.assertIsNot(copied.extra, material.extra)
        self.assertEqual(pickle.dumps(graph, protocol=5), before)
        copied.extra['derived_only'] = [1, 2]
        self.assertEqual(material.extra, {})
        missing = _manual_graph()
        missing.edges[61].material = None
        with patch.object(StructureGraph, 'copy', side_effect=AssertionError('copy before material admission')):
            self.assertRejected(missing)

    def test_invalid_material_payloads_are_rejected(self):
        for value in (None, {}, object(), _MaterialSubclass()):
            graph = _manual_graph()
            graph.edges[61].material = value
            with self.subTest(material=type(value).__name__):
                self.assertRejected(graph)
        for field, value in (('density', float('nan')), ('youngs_modulus', True),
                             ('electrical_conductivity', float('inf')),
                             ('name', 5), ('extra', []),
                             ('stiffness_tensor', np.ones((3, 3))),
                             ('stiffness_tensor', np.full((6, 6), np.nan))):
            graph = _manual_graph()
            setattr(graph.edges[61].material, field, value)
            with self.subTest(field=field, value=repr(value)):
                self.assertRejected(graph)

    def test_invalid_graph_scalars_box_flags_and_segments_are_rejected(self):
        mutations = [
            lambda g: setattr(g, '_tolerance', 0.),
            lambda g: setattr(g, '_tolerance', float('nan')),
            lambda g: setattr(g, '_box_size', np.array([6., np.inf, 0.])),
            lambda g: setattr(g.edges[61], 'radius', float('nan')),
            lambda g: setattr(g.edges[61], 'radius', -1.),
            lambda g: setattr(g.edges[61], 'segments', True),
            lambda g: setattr(g.nodes[7], 'boundary', (False,) * 5),
            lambda g: setattr(g.nodes[7], 'boundary', (1, False, False, False, False, False)),
        ]
        for mutate in mutations:
            graph = _manual_graph()
            mutate(graph)
            self.assertRejected(graph)

    def test_metadata_is_finite_json_not_objects_or_nonstring_keys(self):
        for bad in ({'value': float('nan')}, {'value': float('inf')},
                    {'value': object()}, {'value': (1, 2)}, {7: 'not-a-string-key'},
                    {'value': np.array([1, 2])}, []):
            for location in ('graph', 'node', 'edge', 'material'):
                graph = _manual_graph()
                if location == 'graph':
                    graph._metadata = bad
                elif location == 'node':
                    graph.nodes[7].metadata = bad
                elif location == 'edge':
                    graph.edges[61].metadata = bad
                else:
                    graph.edges[61].material.extra = bad
                with self.subTest(location=location, bad=repr(bad)):
                    self.assertRejected(graph)

    def test_cyclic_and_excessively_deep_metadata_rejected_before_copy(self):
        for location in ('graph', 'node', 'edge', 'material'):
            for kind in ('cycle', 'deep'):
                graph = _manual_graph()
                root = {}
                if kind == 'cycle':
                    root['self'] = root
                else:
                    cursor = root
                    for _ in range(130):
                        cursor['child'] = {}
                        cursor = cursor['child']
                if location == 'graph':
                    graph._metadata = root
                elif location == 'node':
                    graph.nodes[7].metadata = root
                elif location == 'edge':
                    graph.edges[61].metadata = root
                else:
                    graph.edges[61].material.extra = root
                with self.subTest(location=location, kind=kind):
                    with patch.object(StructureGraph, 'copy', side_effect=AssertionError('copy before admission')):
                        self.assertRejected(graph)

    def test_size_limits_are_narrowing_only_exact_positive_python_ints(self):
        defaults = {'max_nodes': 4096, 'max_edges': 16384,
                    'max_metadata_bytes': 4194304, 'max_report_bytes': 16777216}
        for name, ceiling in defaults.items():
            for invalid in (True, False, 0, -1, 1.5, 3.0, np.int64(3),
                            '3', None, ceiling + 1):
                with self.subTest(name=name, invalid=repr(invalid)):
                    self.assertRejected(_manual_graph(), **{name: invalid})
        derived, _, _ = self.prepare(_manual_graph(), max_nodes=3, max_edges=3)
        self.assertEqual((derived.num_nodes, derived.num_edges), (3, 3))

    def test_narrowed_node_edge_and_metadata_limits_reject_before_copy(self):
        for limits in ({'max_nodes': 2}, {'max_edges': 2}, {'max_metadata_bytes': 32}):
            with self.subTest(limits=limits):
                with patch.object(StructureGraph, 'copy', side_effect=AssertionError('copy before admission')):
                    self.assertRejected(_manual_graph(), **limits)
        graph = _manual_graph()
        graph.metadata['oversized'] = 'x' * 4096
        with patch.object(StructureGraph, 'copy', side_effect=AssertionError('copy before admission')):
            self.assertRejected(graph, max_metadata_bytes=1024)
        self.assertRejected(_manual_graph(), max_report_bytes=64)

    def test_bad_spatial_membership_and_unbounded_empty_bins_are_rejected(self):
        for mode in ('missing', 'duplicate', 'foreign', 'wrong_cell', 'too_many_empty_bins'):
            graph = _manual_graph()
            key = graph._pos_key(graph.nodes[7].position)
            if mode == 'missing':
                graph._spatial_hash[key].remove(7)
            elif mode == 'duplicate':
                graph._spatial_hash[key].append(7)
            elif mode == 'foreign':
                graph._spatial_hash[key].append(999)
            elif mode == 'wrong_cell':
                graph._spatial_hash[key].remove(7)
                graph._spatial_hash[(999, 999, 999)] = [7]
            else:
                for i in range(13):
                    graph._spatial_hash[(1000 + i, 999, 999)] = []
            with self.subTest(mode=mode):
                self.assertRejected(graph, max_nodes=3)

    def test_parallel_edge_set_requires_valid_representatives(self):
        graph = _manual_graph()
        pair = frozenset((7, 42))
        # Either actual parallel edge can represent this pair; do not impose an
        # unadvertised first-edge-only restriction on an otherwise valid graph.
        graph._edge_set[pair] = 61
        derived, _, _ = self.prepare(graph)
        self.assertEqual(derived._edge_set, graph._edge_set)
        for mode in ('missing', 'foreign_edge', 'wrong_pair', 'extra_pair'):
            graph = _manual_graph()
            if mode == 'missing':
                del graph._edge_set[pair]
            elif mode == 'foreign_edge':
                graph._edge_set[pair] = 999
            elif mode == 'wrong_pair':
                graph._edge_set[pair] = 9
            else:
                graph._edge_set[frozenset((42, 105))] = 61
            with self.subTest(mode=mode):
                self.assertRejected(graph)

    def test_edge_set_rejects_bool_and_numeric_surrogates_for_endpoint_ids(self):
        # Use IDs 0 and 1 so bool equality would otherwise bypass set equality.
        record = _manual_graph().to_dict()
        remap = {42: 0, 7: 1, 105: 2}
        for node in record['nodes']:
            node['id'] = remap[node['id']]
        for edge in record['edges']:
            edge['node_i'] = remap[edge['node_i']]
            edge['node_j'] = remap[edge['node_j']]
        graph = StructureGraph.from_dict(record)
        self.prepare(graph)  # Positive control: remapping itself is valid.
        for surrogate in (False, True, np.int64(0), np.int64(1), 0.0, 1.0):
            graph = StructureGraph.from_dict(record)
            pair = frozenset((0, 1))
            representative = graph._edge_set.pop(pair)
            endpoints = (surrogate, 1) if int(surrogate) == 0 else (0, surrogate)
            graph._edge_set[frozenset(endpoints)] = representative
            self.assertEqual(set(graph._edge_set), {frozenset((0, 1)), frozenset((1, 2))})
            with self.subTest(value=repr(surrogate), kind=type(surrogate).__name__):
                self.assertRejected(graph)

    def test_report_hashes_change_with_prescribed_input_not_only_geometry(self):
        graph = _manual_graph()
        _, _, base = self.prepare(graph)
        values = dict(_REFERENCE_G)
        values[61] = 11.
        _, _, other_g = self.prepare(graph, reference_conductance_S=values)
        reframed = graph.copy()
        reframed.metadata['coordinate_frame'] = 'another_stored_frame'
        _, _, other_frame = self.prepare(reframed)
        self.assertEqual(base['source_full_graph_sha256'], other_g['source_full_graph_sha256'])
        self.assertNotEqual(base['input_sha256'], other_g['input_sha256'])
        self.assertNotEqual(base['input_sha256'], other_frame['input_sha256'])


if __name__ == '__main__':
    unittest.main()
