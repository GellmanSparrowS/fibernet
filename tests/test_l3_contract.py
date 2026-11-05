"""Synthetic preparation tests. No provider, subprocess or numerical solver.

Written for review; NOT executed in the initial preparation-only change.
"""
import unittest
from unittest import mock
from contextlib import ExitStack
from pathlib import Path
import tempfile
import hashlib
import json
import numpy as np
from fibernet.core.structure_graph import StructureGraph
from fibernet.sim import l3

def graph():
    material = dict(name='explicit', density=1000.0, youngs_modulus=2300000000.0, poissons_ratio=0.35, shear_modulus=2300000000.0 / 2.7)
    return StructureGraph.from_dict({'version': '3.1', 'dimension': 2, 'nodes': [{'id': 19, 'position': [0.0, 2.0, 0.0]}, {'id': 7, 'position': [0.0, 0.0, 0.0]}], 'edges': [{'id': i, 'node_i': 7, 'node_j': 19, 'radius': 0.5, 'material': material.copy(), 'segments': 1} for i in (91, 25)], 'metadata': {'coordinate_units': 'mm', 'radius_units': 'mm'}})

def data():
    return {i: {'failure_strain': f, 'physical_span_id': span, 'source_edge_id': i, 'initially_active': True, 'reference_length_m': 0.002} for i, f, span in ((91, 0.03, '1'), (25, 0.01, 1))}

def bind(g, **changes):
    kwargs = dict(coordinate_unit='mm', radius_unit='mm', edge_data=data(), fixed_node_ids=[7], loaded_node_ids=[19], model=dict(l3._MODEL))
    kwargs.update(changes)
    return l3.bind_l3(g, **kwargs)

def settings():
    return dict(young_Pa=2300000000.0, poisson=0.35, radius_m=0.0005, shear_correction=6 / 7, load_increment_m=1e-05, minimum_increment_m=1e-08, event_displacement_tolerance_m=1e-09, max_displacement_m=0.001, residual_tolerance=2e-07, timeout_s=10, post_rupture_opening_m=0.0, event_bisection_cap=20, frame_cap=50, notched=True, material_source={'note': 'synthetic'}, failure_parameter_status='uncalibrated synthetic threshold', memory_budget_MiB=64, reserve_MiB=128)

def prepare(g, binding, **changes):
    kwargs = dict(case='synthetic', settings=settings(), acceptance={'initial_inactive_segment_ids': []}, resources=l3.L3Resources(2, 2, 10, 128 * 1024 ** 2, 1024 ** 2, 1), provider=l3.L3ProviderSpec('/external/provider', '/external/python'))
    kwargs.update(changes)
    return l3.prepare_l3(g, binding, **kwargs)

def _synthetic_seal_fixture(tmp_path, stack):
    root = tmp_path / 'provider'
    root.mkdir()
    interpreter = tmp_path / 'python'
    interpreter.write_bytes(b'never executed')
    seal = json.loads(json.dumps(l3._SEAL))
    for index, name in enumerate(seal['files_sha256']):
        source = root / name
        source.parent.mkdir(exist_ok=True)
        raw = ('inert identity fixture %d\n' % index).encode()
        source.write_bytes(raw)
        seal['files_sha256'][name] = hashlib.sha256(raw).hexdigest()
    (root / 'SOURCE_IDENTITY.json').write_text(json.dumps(seal))
    stack.enter_context(mock.patch.object(l3, '_SEAL', seal))
    return (root, l3.L3ProviderSpec(str(root), str(interpreter)))

class L3ContractTests(unittest.TestCase):

    def test_multiedges_stable_sorted_ids_units_and_distinct_span_types(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            stack.enter_context(mock.patch.object(l3, '_provider', lambda spec: {'source_verified': False, 'runtime_verified': False, 'execution_qualified': False, 'test_double': True}))
            g = graph()
            b = bind(g)
            p = json.loads(prepare(g, b).canonical)
            assert p['node_ids'] == [7, 19]
            assert p['edge_ids'] == [25, 91]
            assert len(p['request']['network']['segments']) == 2
            assert p['request']['network']['nodes'][1]['position'] == [0.0, 0.002]
            assert p['request']['network']['segments'][0]['radius'] == 0.0005
            assert p['normalized_settings']['memory_budget_mb'] == 64
            assert 'memory_budget_MiB' not in p['normalized_settings']
            assert p['original_settings']['memory_budget_MiB'] == 64
            assert p['provider']['runtime_verified'] is False
            assert p['status'] == 'prepared_not_executable'

    def test_reordering_and_detachment(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            stack.enter_context(mock.patch.object(l3, '_provider', lambda spec: {'source_verified': False, 'runtime_verified': False, 'execution_qualified': False, 'test_double': True}))
            g = graph()
            source = data()
            b = bind(g, edge_data=source)
            g._nodes = dict(reversed(list(g._nodes.items())))
            g._edges = dict(reversed(list(g._edges.items())))
            assert bind(g, edge_data=source) == b
            source[25]['failure_strain'] = 99.0
            assert json.loads(b.canonical)['edge_data'][0]['data']['failure_strain'] == 0.01
            s = settings()
            a = {'initial_inactive_segment_ids': [], 'notes': {'source': 'before'}}
            p = prepare(g, b, settings=s, acceptance=a)
            s['material_source']['note'] = 'after'
            a['notes']['source'] = 'after'
            assert b'after' not in p.canonical

    def test_graph_mutation_invalidates_binding__case_01(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            mutation = lambda g: g.nodes[7].metadata.update(note='changed')
            stack.enter_context(mock.patch.object(l3, '_provider', lambda spec: {'source_verified': False, 'runtime_verified': False, 'execution_qualified': False, 'test_double': True}))
            g = graph()
            b = bind(g)
            mutation(g)
            with self.assertRaises(l3.L3InputError):
                prepare(g, b)

    def test_graph_mutation_invalidates_binding__case_02(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            mutation = lambda g: setattr(g.edges[25].material, 'name', 'changed')
            stack.enter_context(mock.patch.object(l3, '_provider', lambda spec: {'source_verified': False, 'runtime_verified': False, 'execution_qualified': False, 'test_double': True}))
            g = graph()
            b = bind(g)
            mutation(g)
            with self.assertRaises(l3.L3InputError):
                prepare(g, b)

    def test_graph_mutation_invalidates_binding__case_03(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            mutation = lambda g: setattr(g.nodes[7], 'node_id', 8)
            stack.enter_context(mock.patch.object(l3, '_provider', lambda spec: {'source_verified': False, 'runtime_verified': False, 'execution_qualified': False, 'test_double': True}))
            g = graph()
            b = bind(g)
            mutation(g)
            with self.assertRaises(l3.L3InputError):
                prepare(g, b)

    def test_graph_mutation_invalidates_binding__case_04(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            mutation = lambda g: setattr(g.edges[25], 'edge_id', 26)
            stack.enter_context(mock.patch.object(l3, '_provider', lambda spec: {'source_verified': False, 'runtime_verified': False, 'execution_qualified': False, 'test_double': True}))
            g = graph()
            b = bind(g)
            mutation(g)
            with self.assertRaises(l3.L3InputError):
                prepare(g, b)

    def test_graph_mutation_invalidates_binding__case_05(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            mutation = lambda g: setattr(g.edges[25], 'segments', 2)
            stack.enter_context(mock.patch.object(l3, '_provider', lambda spec: {'source_verified': False, 'runtime_verified': False, 'execution_qualified': False, 'test_double': True}))
            g = graph()
            b = bind(g)
            mutation(g)
            with self.assertRaises(l3.L3InputError):
                prepare(g, b)

    def test_graph_mutation_invalidates_binding__case_06(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            mutation = lambda g: setattr(g.edges[25], 'internal_points', np.array([[0.0, 1.0, 0.0]]))
            stack.enter_context(mock.patch.object(l3, '_provider', lambda spec: {'source_verified': False, 'runtime_verified': False, 'execution_qualified': False, 'test_double': True}))
            g = graph()
            b = bind(g)
            mutation(g)
            with self.assertRaises(l3.L3InputError):
                prepare(g, b)

    def test_graph_mutation_invalidates_binding__case_07(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            mutation = lambda g: setattr(g.edges[25].material, 'stiffness_tensor', np.eye(6))
            stack.enter_context(mock.patch.object(l3, '_provider', lambda spec: {'source_verified': False, 'runtime_verified': False, 'execution_qualified': False, 'test_double': True}))
            g = graph()
            b = bind(g)
            mutation(g)
            with self.assertRaises(l3.L3InputError):
                prepare(g, b)

    def test_graph_mutation_invalidates_binding__case_08(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            mutation = lambda g: setattr(g.edges[25].material, 'shear_modulus', 1.0)
            stack.enter_context(mock.patch.object(l3, '_provider', lambda spec: {'source_verified': False, 'runtime_verified': False, 'execution_qualified': False, 'test_double': True}))
            g = graph()
            b = bind(g)
            mutation(g)
            with self.assertRaises(l3.L3InputError):
                prepare(g, b)

    def test_graph_mutation_invalidates_binding__case_09(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            mutation = lambda g: g.nodes[7].position.__setitem__(2, 1e-20)
            stack.enter_context(mock.patch.object(l3, '_provider', lambda spec: {'source_verified': False, 'runtime_verified': False, 'execution_qualified': False, 'test_double': True}))
            g = graph()
            b = bind(g)
            mutation(g)
            with self.assertRaises(l3.L3InputError):
                prepare(g, b)

    def test_bad_thresholds__case_01(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            value = True
            d = data()
            d[25]['failure_strain'] = value
            with self.assertRaises(l3.L3InputError):
                bind(graph(), edge_data=d)

    def test_bad_thresholds__case_02(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            value = np.float64(0.01)
            d = data()
            d[25]['failure_strain'] = value
            with self.assertRaises(l3.L3InputError):
                bind(graph(), edge_data=d)

    def test_bad_thresholds__case_03(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            value = float('nan')
            d = data()
            d[25]['failure_strain'] = value
            with self.assertRaises(l3.L3InputError):
                bind(graph(), edge_data=d)

    def test_bad_thresholds__case_04(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            value = float('inf')
            d = data()
            d[25]['failure_strain'] = value
            with self.assertRaises(l3.L3InputError):
                bind(graph(), edge_data=d)

    def test_bad_thresholds__case_05(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            value = 0.0
            d = data()
            d[25]['failure_strain'] = value
            with self.assertRaises(l3.L3InputError):
                bind(graph(), edge_data=d)

    def test_bad_thresholds__case_06(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            value = -1.0
            d = data()
            d[25]['failure_strain'] = value
            with self.assertRaises(l3.L3InputError):
                bind(graph(), edge_data=d)

    def test_bad_coordinate_containers__case_01(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            value = np.array([0, 0, 0], dtype=object)
            g = graph()
            g.nodes[7].position = value
            with self.assertRaises(l3.L3InputError):
                bind(g)

    def test_bad_coordinate_containers__case_02(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            value = np.array([False, False, False])
            g = graph()
            g.nodes[7].position = value
            with self.assertRaises(l3.L3InputError):
                bind(g)

    def test_bad_coordinate_containers__case_03(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            value = np.ma.array([0.0, 0.0, 0.0], mask=[False, True, False])
            g = graph()
            g.nodes[7].position = value
            with self.assertRaises(l3.L3InputError):
                bind(g)

    def test_bad_coordinate_containers__case_04(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            value = np.array([float('nan'), 0, 0])
            g = graph()
            g.nodes[7].position = value
            with self.assertRaises(l3.L3InputError):
                bind(g)

    def test_unit_model_span_and_grip_rejections(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            g = graph()
            for kwargs in (dict(coordinate_unit='m'), dict(radius_unit='m'), dict(fixed_node_ids=[True]), dict(loaded_node_ids=[7]), dict(model=dict(l3._MODEL, joint='hinged'))):
                with self.assertRaises(l3.L3InputError):
                    bind(g, **kwargs)
            d = data()
            d[91]['physical_span_id'] = 1
            with self.assertRaises(l3.L3InputError):
                bind(g, edge_data=d)

    def test_lineage_preserved_and_not_reinterpreted(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            d = data()
            d[25]['native_lineage'] = {'strength_span_id': 1, 'fiber_id': 123, 'parent_segment_id': 456, 'source_planar_line_indices': [8, 9], 'initial_kerf_inactive': False}
            b = bind(graph(), edge_data=d)
            assert json.loads(b.canonical)['edge_data'][0]['data']['native_lineage'] == d[25]['native_lineage']
            d[25]['native_lineage']['strength_span_id'] = '1'
            with self.assertRaises(l3.L3InputError):
                bind(graph(), edge_data=d)

    def test_invalid_settings__case_01(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            change = {'memory_budget_mb': 65}
            stack.enter_context(mock.patch.object(l3, '_provider', lambda spec: {'source_verified': False, 'runtime_verified': False, 'execution_qualified': False, 'test_double': True}))
            g = graph()
            with self.assertRaises(l3.L3InputError):
                prepare(g, bind(g), settings=dict(settings(), **change))

    def test_invalid_settings__case_02(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            change = {'shear_correction': 0}
            stack.enter_context(mock.patch.object(l3, '_provider', lambda spec: {'source_verified': False, 'runtime_verified': False, 'execution_qualified': False, 'test_double': True}))
            g = graph()
            with self.assertRaises(l3.L3InputError):
                prepare(g, bind(g), settings=dict(settings(), **change))

    def test_invalid_settings__case_03(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            change = {'timeout_s': 11}
            stack.enter_context(mock.patch.object(l3, '_provider', lambda spec: {'source_verified': False, 'runtime_verified': False, 'execution_qualified': False, 'test_double': True}))
            g = graph()
            with self.assertRaises(l3.L3InputError):
                prepare(g, bind(g), settings=dict(settings(), **change))

    def test_invalid_settings__case_04(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            change = {'frame_cap': True}
            stack.enter_context(mock.patch.object(l3, '_provider', lambda spec: {'source_verified': False, 'runtime_verified': False, 'execution_qualified': False, 'test_double': True}))
            g = graph()
            with self.assertRaises(l3.L3InputError):
                prepare(g, bind(g), settings=dict(settings(), **change))

    def test_invalid_settings__case_05(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            change = {'output': 'elsewhere'}
            stack.enter_context(mock.patch.object(l3, '_provider', lambda spec: {'source_verified': False, 'runtime_verified': False, 'execution_qualified': False, 'test_double': True}))
            g = graph()
            with self.assertRaises(l3.L3InputError):
                prepare(g, bind(g), settings=dict(settings(), **change))

    def test_invalid_settings__case_06(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            change = {'unknown': 1}
            stack.enter_context(mock.patch.object(l3, '_provider', lambda spec: {'source_verified': False, 'runtime_verified': False, 'execution_qualified': False, 'test_double': True}))
            g = graph()
            with self.assertRaises(l3.L3InputError):
                prepare(g, bind(g), settings=dict(settings(), **change))

    def test_invalid_settings__case_07(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            change = {'young_Pa': 3500000000.0}
            stack.enter_context(mock.patch.object(l3, '_provider', lambda spec: {'source_verified': False, 'runtime_verified': False, 'execution_qualified': False, 'test_double': True}))
            g = graph()
            with self.assertRaises(l3.L3InputError):
                prepare(g, bind(g), settings=dict(settings(), **change))

    def test_invalid_settings__case_08(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            change = {'strength_field_mean': 0.01}
            stack.enter_context(mock.patch.object(l3, '_provider', lambda spec: {'source_verified': False, 'runtime_verified': False, 'execution_qualified': False, 'test_double': True}))
            g = graph()
            with self.assertRaises(l3.L3InputError):
                prepare(g, bind(g), settings=dict(settings(), **change))

    def test_missing_budget_never_falls_back(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            stack.enter_context(mock.patch.object(l3, '_provider', lambda spec: {'source_verified': False, 'runtime_verified': False, 'execution_qualified': False, 'test_double': True}))
            g = graph()
            s = settings()
            del s['memory_budget_MiB']
            with self.assertRaises(l3.L3InputError):
                prepare(g, bind(g), settings=s)

    def test_notch_and_post_notch_strength_statistics(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            stack.enter_context(mock.patch.object(l3, '_provider', lambda spec: {'source_verified': False, 'runtime_verified': False, 'execution_qualified': False, 'test_double': True}))
            g = graph()
            d = data()
            d[91]['initially_active'] = False
            b = bind(g, edge_data=d)
            a = {'initial_inactive_segment_ids': [91]}
            p = json.loads(prepare(g, b, acceptance=a, settings=dict(settings(), strength_field_mean=0.01, strength_field_cv=0.0)).canonical)
            assert p['strength_active_length_weighted_mean'] == 0.01
            assert p['strength_active_length_weighted_cv'] == 0.0
            assert len(p['request']['network']['segments']) == 2
            with self.assertRaises(l3.L3InputError):
                prepare(g, b, acceptance=a, settings=dict(settings(), notched=False))

    def test_native_and_stronger_digest_are_distinct(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            stack.enter_context(mock.patch.object(l3, '_provider', lambda spec: {'source_verified': False, 'runtime_verified': False, 'execution_qualified': False, 'test_double': True}))
            g = graph()
            b = bind(g)
            plan = prepare(g, b)
            payload = json.loads(plan.canonical)
            request = payload['request']
            native = {k: request[k] for k in ('network', 'boundary', 'settings', 'acceptance')}
            raw = json.dumps(native, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
            assert plan.native_request_sha256 == hashlib.sha256(raw).hexdigest()
            other = prepare(g, b, case='other-case')
            assert other.sha256 != plan.sha256
            assert other.native_request_sha256 == plan.native_request_sha256

    def test_provider_absent_and_symlink_fail_closed(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            missing = l3.L3ProviderSpec(str(tmp_path / 'absent'), str(tmp_path / 'python'))
            with self.assertRaises(l3.L3ProviderIdentityError):
                l3._provider(missing)
            target = tmp_path / 'target'
            target.mkdir()
            alias = tmp_path / 'alias'
            alias.symlink_to(target, target_is_directory=True)
            with self.assertRaises(l3.L3ProviderIdentityError):
                l3._provider(l3.L3ProviderSpec(str(alias), str(tmp_path / 'python')))

    def test_no_execution_api(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            assert not hasattr(l3, 'simulate_l3')
            assert not hasattr(l3, 'run_case')

    def test_weighting_is_reference_length_not_edge_count(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            stack.enter_context(mock.patch.object(l3, '_provider', lambda spec: {'source_verified': False, 'runtime_verified': False, 'execution_qualified': False, 'test_double': True}))
            g = graph()
            g._nodes[31] = type(g.nodes[7])(position=np.array([0.0, 6.0, 0.0]), node_id=31)
            g.edges[91].node_j = 31
            d = data()
            d[91]['reference_length_m'] = 0.006
            b = bind(g, edge_data=d)
            p = json.loads(prepare(g, b, resources=l3.L3Resources(3, 2, 10, 128 * 1024 ** 2, 1024 ** 2, 1), settings=dict(settings(), strength_field_mean=0.025, strength_field_cv=3 ** 0.5 / 5)).canonical)
            self.assertAlmostEqual(p['strength_active_length_weighted_mean'], 0.025, places=12)
            self.assertAlmostEqual(p['strength_active_length_weighted_cv'], 3 ** 0.5 / 5, places=12)

    def test_alias_does_not_hide_invalid_canonical_type__case_01(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            canonical = True
            stack.enter_context(mock.patch.object(l3, '_provider', lambda spec: {'source_verified': False, 'runtime_verified': False, 'execution_qualified': False, 'test_double': True}))
            g = graph()
            with self.assertRaises(l3.L3InputError):
                prepare(g, bind(g), settings=dict(settings(), memory_budget_mb=canonical))

    def test_alias_does_not_hide_invalid_canonical_type__case_02(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            canonical = 64.0
            stack.enter_context(mock.patch.object(l3, '_provider', lambda spec: {'source_verified': False, 'runtime_verified': False, 'execution_qualified': False, 'test_double': True}))
            g = graph()
            with self.assertRaises(l3.L3InputError):
                prepare(g, bind(g), settings=dict(settings(), memory_budget_mb=canonical))

    def test_alias_does_not_hide_invalid_canonical_type__case_03(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            canonical = np.int64(64)
            stack.enter_context(mock.patch.object(l3, '_provider', lambda spec: {'source_verified': False, 'runtime_verified': False, 'execution_qualified': False, 'test_double': True}))
            g = graph()
            with self.assertRaises(l3.L3InputError):
                prepare(g, bind(g), settings=dict(settings(), memory_budget_mb=canonical))

    def test_radius_si_underflow_and_length_overflow_rejected(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            g = graph()
            g.edges[25].radius = 5e-324
            g.edges[91].radius = 5e-324
            with self.assertRaises(l3.L3InputError):
                bind(g)
            g = graph()
            g.metadata.clear()
            g.nodes[7].position = np.array([0.0, -1.7e+308, 0.0])
            g.nodes[19].position = np.array([0.0, 1.7e+308, 0.0])
            with self.assertRaises(l3.L3InputError):
                bind(g, coordinate_unit='m')

    def test_stale_or_forged_binding_rejected(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            stack.enter_context(mock.patch.object(l3, '_provider', lambda spec: {'source_verified': False, 'runtime_verified': False, 'execution_qualified': False, 'test_double': True}))
            g = graph()
            b = bind(g)
            with self.assertRaises(l3.L3InputError):
                prepare(g, l3.L3Binding(b.canonical + b' ', b.sha256))
            record = json.loads(b.canonical)
            record['edge_data'].append(record['edge_data'][0])
            raw = json.dumps(record, sort_keys=True, separators=(',', ':')).encode()
            with self.assertRaises(l3.L3InputError):
                prepare(g, l3.L3Binding(raw, hashlib.sha256(raw).hexdigest()))

    def test_provider_seal_invalid_rejected_without_import__case_01(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            seal_text = '{}'
            root = tmp_path / 'provider'
            root.mkdir()
            interpreter = tmp_path / 'python'
            interpreter.write_bytes(b'not executed')
            (root / 'SOURCE_IDENTITY.json').write_text(seal_text)
            with self.assertRaises(l3.L3ProviderIdentityError):
                l3._provider(l3.L3ProviderSpec(str(root), str(interpreter)))

    def test_provider_seal_invalid_rejected_without_import__case_02(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            seal_text = '{"python":"3.14.3","python":"3.14.3"}'
            root = tmp_path / 'provider'
            root.mkdir()
            interpreter = tmp_path / 'python'
            interpreter.write_bytes(b'not executed')
            (root / 'SOURCE_IDENTITY.json').write_text(seal_text)
            with self.assertRaises(l3.L3ProviderIdentityError):
                l3._provider(l3.L3ProviderSpec(str(root), str(interpreter)))

    def test_provider_seal_invalid_rejected_without_import__case_03(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            seal_text = '{"python":NaN}'
            root = tmp_path / 'provider'
            root.mkdir()
            interpreter = tmp_path / 'python'
            interpreter.write_bytes(b'not executed')
            (root / 'SOURCE_IDENTITY.json').write_text(seal_text)
            with self.assertRaises(l3.L3ProviderIdentityError):
                l3._provider(l3.L3ProviderSpec(str(root), str(interpreter)))

    def test_provider_source_hash_mismatch_without_import(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            root = tmp_path / 'provider'
            root.mkdir()
            interpreter = tmp_path / 'python'
            interpreter.write_bytes(b'not executed')
            (root / 'SOURCE_IDENTITY.json').write_text(json.dumps(l3._SEAL))
            (root / 'run_case.py').write_text("raise RuntimeError('must never execute')\n")
            with self.assertRaisesRegex(l3.L3ProviderIdentityError, 'Changed sealed source'):
                l3._provider(l3.L3ProviderSpec(str(root), str(interpreter)))

    def test_geometric_face_flags_are_preserved_not_interpreted_as_dofs(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            g = graph()
            g.nodes[7].boundary = (True, False, True, False, False, False)
            g.nodes[19].boundary = (True, False, False, True, False, False)
            b = json.loads(bind(g).canonical)
            assert b['record']['graph']['nodes'][0]['boundary'] == [True, False, True, False, False, False]
            assert b['fixed_node_ids'] == [7]
            assert b['loaded_node_ids'] == [19]
            g.metadata['loading'] = 'free_rotation'
            with self.assertRaises(l3.L3InputError):
                bind(g)

    def test_graph_native_real_scalars_are_losslessly_normalized__case_01(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            scalar = np.float32
            g = graph()
            for edge in g.edges.values():
                edge.material.youngs_modulus = scalar(2)
                edge.material.poissons_ratio = scalar(0)
                edge.material.shear_modulus = scalar(1)
                edge.radius = scalar(1)
            b = json.loads(bind(g).canonical)
            assert b['record']['young_Pa'] == 2
            assert b['record']['poisson'] == 0
            assert b['record']['radius_m'] == 0.001
            assert type(g.edges[25].radius) is scalar

    def test_graph_native_real_scalars_are_losslessly_normalized__case_02(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            scalar = np.float64
            g = graph()
            for edge in g.edges.values():
                edge.material.youngs_modulus = scalar(2)
                edge.material.poissons_ratio = scalar(0)
                edge.material.shear_modulus = scalar(1)
                edge.radius = scalar(1)
            b = json.loads(bind(g).canonical)
            assert b['record']['young_Pa'] == 2
            assert b['record']['poisson'] == 0
            assert b['record']['radius_m'] == 0.001
            assert type(g.edges[25].radius) is scalar

    def test_graph_native_real_scalars_are_losslessly_normalized__case_03(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            scalar = np.longdouble
            g = graph()
            for edge in g.edges.values():
                edge.material.youngs_modulus = scalar(2)
                edge.material.poissons_ratio = scalar(0)
                edge.material.shear_modulus = scalar(1)
                edge.radius = scalar(1)
            b = json.loads(bind(g).canonical)
            assert b['record']['young_Pa'] == 2
            assert b['record']['poisson'] == 0
            assert b['record']['radius_m'] == 0.001
            assert type(g.edges[25].radius) is scalar

    def test_graph_native_real_scalars_are_losslessly_normalized__case_04(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            scalar = np.int32
            g = graph()
            for edge in g.edges.values():
                edge.material.youngs_modulus = scalar(2)
                edge.material.poissons_ratio = scalar(0)
                edge.material.shear_modulus = scalar(1)
                edge.radius = scalar(1)
            b = json.loads(bind(g).canonical)
            assert b['record']['young_Pa'] == 2
            assert b['record']['poisson'] == 0
            assert b['record']['radius_m'] == 0.001
            assert type(g.edges[25].radius) is scalar

    def test_graph_native_real_scalars_are_losslessly_normalized__case_05(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            scalar = np.int64
            g = graph()
            for edge in g.edges.values():
                edge.material.youngs_modulus = scalar(2)
                edge.material.poissons_ratio = scalar(0)
                edge.material.shear_modulus = scalar(1)
                edge.radius = scalar(1)
            b = json.loads(bind(g).canonical)
            assert b['record']['young_Pa'] == 2
            assert b['record']['poisson'] == 0
            assert b['record']['radius_m'] == 0.001
            assert type(g.edges[25].radius) is scalar

    def test_bad_graph_native_scalars_rejected__case_01(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            bad = np.bool_(True)
            g = graph()
            g.edges[25].radius = bad
            with self.assertRaises(l3.L3InputError):
                bind(g)

    def test_bad_graph_native_scalars_rejected__case_02(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            bad = np.complex128(1)
            g = graph()
            g.edges[25].radius = bad
            with self.assertRaises(l3.L3InputError):
                bind(g)

    def test_bad_graph_native_scalars_rejected__case_03(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            bad = np.float64(float('nan'))
            g = graph()
            g.edges[25].radius = bad
            with self.assertRaises(l3.L3InputError):
                bind(g)

    def test_bad_graph_native_scalars_rejected__case_04(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            bad = np.float32(float('inf'))
            g = graph()
            g.edges[25].radius = bad
            with self.assertRaises(l3.L3InputError):
                bind(g)

    def test_bad_graph_native_scalars_rejected__case_05(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            bad = np.timedelta64(1, 's')
            g = graph()
            g.edges[25].radius = bad
            with self.assertRaises(l3.L3InputError):
                bind(g)

    def test_bad_graph_native_scalars_rejected__case_06(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            bad = np.datetime64('2026-01-01')
            g = graph()
            g.edges[25].radius = bad
            with self.assertRaises(l3.L3InputError):
                bind(g)

    def test_bad_graph_native_scalars_rejected__case_07(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            bad = np.ma.masked
            g = graph()
            g.edges[25].radius = bad
            with self.assertRaises(l3.L3InputError):
                bind(g)

    def test_longdouble_precision_loss_rejected(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
                self.skipTest('This platform has no wider longdouble significand')
            g = graph()
            g.edges[25].radius = np.nextafter(np.longdouble(1), np.longdouble(2))
            with self.assertRaisesRegex(l3.L3InputError, 'losslessly'):
                bind(g)

    def test_malformed_graph_stores_reject_deliberately__case_01(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            target = 'node'
            g = graph()
            if target == 'node':
                g._nodes[7] = None
            elif target == 'edge':
                g._edges[25] = None
            elif target == 'nodes':
                g._nodes = []
            else:
                g._edges = []
            with self.assertRaises(l3.L3InputError):
                bind(g)

    def test_malformed_graph_stores_reject_deliberately__case_02(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            target = 'edge'
            g = graph()
            if target == 'node':
                g._nodes[7] = None
            elif target == 'edge':
                g._edges[25] = None
            elif target == 'nodes':
                g._nodes = []
            else:
                g._edges = []
            with self.assertRaises(l3.L3InputError):
                bind(g)

    def test_malformed_graph_stores_reject_deliberately__case_03(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            target = 'nodes'
            g = graph()
            if target == 'node':
                g._nodes[7] = None
            elif target == 'edge':
                g._edges[25] = None
            elif target == 'nodes':
                g._nodes = []
            else:
                g._edges = []
            with self.assertRaises(l3.L3InputError):
                bind(g)

    def test_malformed_graph_stores_reject_deliberately__case_04(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            target = 'edges'
            g = graph()
            if target == 'node':
                g._nodes[7] = None
            elif target == 'edge':
                g._edges[25] = None
            elif target == 'nodes':
                g._nodes = []
            else:
                g._edges = []
            with self.assertRaises(l3.L3InputError):
                bind(g)

    def test_nonuniform_graph_properties_reject__case_01(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            field, value = ('youngs_modulus', 4600000000.0)
            g = graph()
            if field == 'radius':
                setattr(g.edges[25], field, value)
            else:
                mat = g.edges[25].material
                setattr(mat, field, value)
                mat.shear_modulus = mat.youngs_modulus / (2 * (1 + mat.poissons_ratio))
            with self.assertRaises(l3.L3InputError):
                bind(g)

    def test_nonuniform_graph_properties_reject__case_02(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            field, value = ('poissons_ratio', 0.25)
            g = graph()
            if field == 'radius':
                setattr(g.edges[25], field, value)
            else:
                mat = g.edges[25].material
                setattr(mat, field, value)
                mat.shear_modulus = mat.youngs_modulus / (2 * (1 + mat.poissons_ratio))
            with self.assertRaises(l3.L3InputError):
                bind(g)

    def test_nonuniform_graph_properties_reject__case_03(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            field, value = ('radius', 1.0)
            g = graph()
            if field == 'radius':
                setattr(g.edges[25], field, value)
            else:
                mat = g.edges[25].material
                setattr(mat, field, value)
                mat.shear_modulus = mat.youngs_modulus / (2 * (1 + mat.poissons_ratio))
            with self.assertRaises(l3.L3InputError):
                bind(g)

    def test_unknown_keywords_and_resource_counts(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            stack.enter_context(mock.patch.object(l3, '_provider', lambda spec: {'source_verified': False, 'runtime_verified': False, 'execution_qualified': False, 'test_double': True}))
            g = graph()
            with self.assertRaises(TypeError):
                bind(g, unknown=True)
            with self.assertRaises(TypeError):
                prepare(g, bind(g), unknown=True)
            for nodes, edges in ((1, 2), (2, 1)):
                with self.assertRaises(l3.L3InputError):
                    prepare(g, bind(g), resources=l3.L3Resources(nodes, edges, 10, 1024, 1024, 1))

    def test_malformed_binding_bytes_have_input_error__case_01(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            raw = b'\xff'
            stack.enter_context(mock.patch.object(l3, '_provider', lambda spec: {'source_verified': False, 'runtime_verified': False, 'execution_qualified': False, 'test_double': True}))
            g = graph()
            with self.assertRaises(l3.L3InputError):
                prepare(g, l3.L3Binding(raw, hashlib.sha256(raw).hexdigest()))

    def test_malformed_binding_bytes_have_input_error__case_02(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            raw = b'[' * 2000 + b']' * 2000
            stack.enter_context(mock.patch.object(l3, '_provider', lambda spec: {'source_verified': False, 'runtime_verified': False, 'execution_qualified': False, 'test_double': True}))
            g = graph()
            with self.assertRaises(l3.L3InputError):
                prepare(g, l3.L3Binding(raw, hashlib.sha256(raw).hexdigest()))

    def test_all_six_source_hashes_are_checked(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            synthetic_seal = _synthetic_seal_fixture(tmp_path, stack)
            root, spec = synthetic_seal
            record = l3._provider(spec)
            assert record['source_verified'] is True
            assert record['runtime_verified'] is False
            for name in l3._SEAL['files_sha256']:
                source = root / name
                raw = source.read_bytes()
                source.write_bytes(raw + b'changed')
                with self.assertRaisesRegex(l3.L3ProviderIdentityError, 'Changed sealed source'):
                    l3._provider(spec)
                source.write_bytes(raw)

    def test_later_source_filesystem_failures__case_01(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            failure = 'missing'
            synthetic_seal = _synthetic_seal_fixture(tmp_path, stack)
            root, spec = synthetic_seal
            name = list(l3._SEAL['files_sha256'])[-1]
            source = root / name
            if failure == 'missing':
                source.unlink()
            elif failure == 'symlink':
                target = tmp_path / 'external'
                target.write_bytes(source.read_bytes())
                source.unlink()
                source.symlink_to(target)
            else:
                original = type(source).read_bytes

                def read_bytes(path):
                    if path == source:
                        if failure == 'unreadable':
                            raise PermissionError('synthetic denied read')
                        raise FileNotFoundError('synthetic race')
                    return original(path)
                stack.enter_context(mock.patch.object(type(source), 'read_bytes', read_bytes))
            with self.assertRaises(l3.L3ProviderIdentityError):
                l3._provider(spec)

    def test_later_source_filesystem_failures__case_02(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            failure = 'symlink'
            synthetic_seal = _synthetic_seal_fixture(tmp_path, stack)
            root, spec = synthetic_seal
            name = list(l3._SEAL['files_sha256'])[-1]
            source = root / name
            if failure == 'missing':
                source.unlink()
            elif failure == 'symlink':
                target = tmp_path / 'external'
                target.write_bytes(source.read_bytes())
                source.unlink()
                source.symlink_to(target)
            else:
                original = type(source).read_bytes

                def read_bytes(path):
                    if path == source:
                        if failure == 'unreadable':
                            raise PermissionError('synthetic denied read')
                        raise FileNotFoundError('synthetic race')
                    return original(path)
                stack.enter_context(mock.patch.object(type(source), 'read_bytes', read_bytes))
            with self.assertRaises(l3.L3ProviderIdentityError):
                l3._provider(spec)

    def test_later_source_filesystem_failures__case_03(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            failure = 'unreadable'
            synthetic_seal = _synthetic_seal_fixture(tmp_path, stack)
            root, spec = synthetic_seal
            name = list(l3._SEAL['files_sha256'])[-1]
            source = root / name
            if failure == 'missing':
                source.unlink()
            elif failure == 'symlink':
                target = tmp_path / 'external'
                target.write_bytes(source.read_bytes())
                source.unlink()
                source.symlink_to(target)
            else:
                original = type(source).read_bytes

                def read_bytes(path):
                    if path == source:
                        if failure == 'unreadable':
                            raise PermissionError('synthetic denied read')
                        raise FileNotFoundError('synthetic race')
                    return original(path)
                stack.enter_context(mock.patch.object(type(source), 'read_bytes', read_bytes))
            with self.assertRaises(l3.L3ProviderIdentityError):
                l3._provider(spec)

    def test_later_source_filesystem_failures__case_04(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            failure = 'disappeared'
            synthetic_seal = _synthetic_seal_fixture(tmp_path, stack)
            root, spec = synthetic_seal
            name = list(l3._SEAL['files_sha256'])[-1]
            source = root / name
            if failure == 'missing':
                source.unlink()
            elif failure == 'symlink':
                target = tmp_path / 'external'
                target.write_bytes(source.read_bytes())
                source.unlink()
                source.symlink_to(target)
            else:
                original = type(source).read_bytes

                def read_bytes(path):
                    if path == source:
                        if failure == 'unreadable':
                            raise PermissionError('synthetic denied read')
                        raise FileNotFoundError('synthetic race')
                    return original(path)
                stack.enter_context(mock.patch.object(type(source), 'read_bytes', read_bytes))
            with self.assertRaises(l3.L3ProviderIdentityError):
                l3._provider(spec)

    def test_invalid_provider_path_types_have_identity_error__case_01(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            path = None
            with self.assertRaises(l3.L3ProviderIdentityError):
                l3._provider(l3.L3ProviderSpec(path, '/never/executed'))

    def test_invalid_provider_path_types_have_identity_error__case_02(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            path = 2
            with self.assertRaises(l3.L3ProviderIdentityError):
                l3._provider(l3.L3ProviderSpec(path, '/never/executed'))

    def test_invalid_provider_path_types_have_identity_error__case_03(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            path = ''
            with self.assertRaises(l3.L3ProviderIdentityError):
                l3._provider(l3.L3ProviderSpec(path, '/never/executed'))

    def test_invalid_provider_path_types_have_identity_error__case_04(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            path = 'relative'
            with self.assertRaises(l3.L3ProviderIdentityError):
                l3._provider(l3.L3ProviderSpec(path, '/never/executed'))

    def test_invalid_provider_path_types_have_identity_error__case_05(self):
        with ExitStack() as stack:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            path = '\x00'
            with self.assertRaises(l3.L3ProviderIdentityError):
                l3._provider(l3.L3ProviderSpec(path, '/never/executed'))
if __name__ == '__main__':
    unittest.main(verbosity=2)
