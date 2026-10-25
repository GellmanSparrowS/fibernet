"""Saved-byte/API regression only: every numerical owner is forbidden/stubbed.

The bundled axial-chain result is reopened without recomputing its trajectory.
SAVED_ELECTROTHERMAL_TEST_FILE may override its location.
No fixtures perform physics. Synthetic partial/diagnostic records test format
handling only and are not promoted to additional physical evidence.
"""
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from fibernet.core.structure_graph import StructureGraph
from fibernet.sim import saved_electrothermal as bridge


def rehash(saved):
    saved['content_sha256'] = bridge._hash({k: v for k, v in saved.items() if k != 'content_sha256'})
    return saved


@pytest.fixture(autouse=True)
def no_physics(monkeypatch):
    import fibernet.sim.electrothermal as electrothermal
    import fibernet.sim.linear_transport as electrical
    import fibernet.sim.transient_thermal as thermal
    import fibernet.sim.joule_heating as joule
    import fibernet.sim._thermoelastic as mechanics
    def forbidden(*args, **kwargs):
        pytest.fail('API tests must never execute a numerical solver')
    for module, name in ((electrothermal, 'solve_electrothermal'),
                         (electrothermal, 'solve_linear_transport'),
                         (electrothermal, '_solve_thermal_interval'),
                         (electrical, 'solve_linear_transport'),
                         (thermal, 'solve_transient_thermal'),
                         (thermal, '_solve_transient_thermal'),
                         (thermal, '_solve_thermal_interval'),
                         (joule, 'run_joule_heating'),
                         (mechanics, 'solve_thermoelastic_snapshot'),
                         (mechanics, 'solve_thermoelastic_network_snapshot')):
        monkeypatch.setattr(module, name, forbidden)


@pytest.fixture
def saved_fixture(tmp_path):
    location = os.environ.get('SAVED_ELECTROTHERMAL_TEST_FILE')
    if location:
        path = Path(location)
    else:
        from benchmarks.saved_electrothermal_fixtures import fixture_bytes
        path = tmp_path / 'axial_chain.json'
        path.write_bytes(fixture_bytes('axial_chain.json'))
    original = path.read_bytes()
    data = bridge.read_saved_electrothermal_json(path)
    graph = StructureGraph.from_dict(data['graph'])
    yield graph, data
    assert path.read_bytes() == original, 'test changed the sole saved evidence file'


def prepare(graph, saved, **kwargs):
    return bridge.prepare_saved_electrothermal_temperature(graph, saved, frame_index=kwargs.pop('frame_index', 10),
                                                          length_unit=kwargs.pop('length_unit', 'm'), **kwargs)


def mechanical_kwargs():
    return dict(profile='network', E_Pa=1e6, expansion_alpha_per_K={11: 1e-6, 29: 1e-6},
                stress_free_temperature_K=300., constraints={101: (0., 0., 0.)})


def native_from_saved(saved):
    from fibernet.sim.electrothermal import ElectrothermalResult, ElectrothermalFrame
    source = saved['source']
    frames = []
    for data in source['frames']:
        row = {name: tuple(data[name]) for name in bridge._NODE_ARRAYS + bridge._EDGE_ARRAYS + bridge._DIAGNOSTICS}
        row.update(time_s=data['time_s'], accepted_iterations=data['accepted_iterations'],
                   interval_diagnostics_available=data['interval_diagnostics_available'],
                   electrical_json=json.dumps(data['electrical'], allow_nan=False),
                   heat_load_json=json.dumps(data['heat_load'], allow_nan=False))
        frames.append(ElectrothermalFrame(**row))
    fields = {name: source[name] for name in bridge._SOURCE_FIELDS}
    for name in bridge._SOURCE_FIELDS:
        if isinstance(fields[name], list):
            fields[name] = tuple(tuple(row) if isinstance(row, list) else row for row in fields[name])
    return ElectrothermalResult(**fields, frames=tuple(frames), provenance_json=json.dumps(source['provenance']))


def test_saved_binding_is_detached_exact_and_order_independent(saved_fixture):
    graph, data = saved_fixture
    original = copy.deepcopy(data)
    snapshot = prepare(graph, data)
    expected = dict(zip(data['source']['node_ids'], data['source']['frames'][10]['temperature_K']))
    assert snapshot.mechanical_kwargs(graph) == {'length_unit': 'm', 'temperature_K': expected}
    data['source']['frames'][10]['temperature_K'][0] = 999.
    snapshot.to_dict()['temperature_K'][0] = 888.
    graph.nodes.update(reversed(list(graph.nodes.items())))
    for mapping in (graph.nodes, graph.edges):
        entries = list(mapping.items())[::-1]
        mapping.clear()
        mapping.update(entries)
    assert snapshot.mechanical_kwargs(graph)['temperature_K'] == expected
    reordered = prepare(graph, original)
    assert reordered.to_dict() == snapshot.to_dict()


def test_export_native_only_detached_and_exact(saved_fixture):
    graph, saved = saved_fixture
    native = native_from_saved(saved)
    exported = bridge.export_saved_electrothermal(graph, native, length_unit='m')
    assert exported == saved
    exported['source']['frames'][0]['temperature_K'][0] = 999.
    assert native.frames[0].temperature_K[0] == 300.
    with pytest.raises(TypeError):
        bridge.export_saved_electrothermal(graph, SimpleNamespace(to_dict=lambda: saved), length_unit='m')


def test_public_facade_and_owner_error_propagation(saved_fixture):
    graph, data = saved_fixture
    snap = prepare(graph, data)
    from fibernet.sim import thermoelastic
    limits = thermoelastic.ThermoelasticSizeLimits(3, 2)
    with patch.object(thermoelastic, 'solve_thermoelastic_network_snapshot',
                      return_value=SimpleNamespace(to_dict=lambda: {'test_double': True}), create=True) as owner:
        out = snap.solve(graph, **mechanical_kwargs(), size_limits=limits)
        assert owner.call_count == 1
        assert owner.call_args.kwargs['size_limits'] is limits
        assert owner.call_args.kwargs['temperature_K'] == snap.mechanical_kwargs(graph)['temperature_K']
    assert out['execution'] == 'new_mechanical_solve_from_saved_electrothermal_temperature'
    assert out['upstream_electrical_thermal_feedback'] is True
    assert out['feedback_to_electrical_or_thermal'] is False
    failure = RuntimeError('unchanged owner error')
    with patch.object(thermoelastic, 'solve_thermoelastic_network_snapshot', side_effect=failure, create=True):
        with pytest.raises(RuntimeError) as caught:
            snap.solve(graph, **mechanical_kwargs())
    assert caught.value is failure
    with pytest.raises(ValueError):
        snap.solve(graph, **dict(mechanical_kwargs(), profile='elementary'), size_limits=limits)


@pytest.mark.parametrize('kind', ['position', 'radius', 'material', 'endpoint', 'internal', 'dimension', 'metadata'])
def test_changed_graph_rejects_before_delegation(saved_fixture, kind):
    graph, data = saved_fixture
    snap = prepare(graph, data)
    if kind == 'position': graph.nodes[303].position[0] += 1e-9
    elif kind == 'radius': graph.edges[11].radius *= 2
    elif kind == 'material': graph.edges[11].material.youngs_modulus *= 2
    elif kind == 'endpoint': graph.edges[11].node_i, graph.edges[11].node_j = 101, 303
    elif kind == 'internal':
        import numpy as np
        graph.edges[11].internal_points = np.array([[.25, .001, 0.]])
    elif kind == 'dimension': graph._dimension = 3
    else: graph.nodes[101].metadata['changed'] = True
    with pytest.raises(ValueError, match='graph changed'):
        snap.solve(graph, **mechanical_kwargs())
    with pytest.raises(ValueError, match='captured graph'):
        prepare(graph, data)


def mutate(saved, kind):
    source = saved['source']; frame = source['frames'][1]
    if kind == 'frame_G': frame['conductance_S'][0] += .01
    elif kind == 'nested_G': frame['electrical']['conductance_S'][0] += .01
    elif kind == 'G0': source['provenance']['law']['G0_S']['11'] = 2.
    elif kind == 'nested_heat_power': frame['heat_load']['edge_power_W'][0] += .01
    elif kind == 'heat_link': frame['heat_load']['provenance']['electrical_input_sha256'] = 'a'*64
    elif kind == 'instantaneous_label': frame['heat_load']['provenance']['model'] = 'two_way'
    elif kind == 'node_duplicate': source['node_ids'][1] = source['node_ids'][0]
    elif kind == 'node_bool': source['node_ids'][0] = True
    elif kind == 'edge_duplicate': source['edge_ids'][1] = source['edge_ids'][0]
    elif kind == 'missing_edge': source['edge_ids'].pop()
    elif kind == 'orientation': source['endpoints_by_id'][0].reverse()
    elif kind == 'nested_orientation': frame['electrical']['endpoints_by_id'][0].reverse()
    elif kind == 'heat_orientation': frame['heat_load']['endpoints_by_id'][0].reverse()
    elif kind == 'nested_missing_ID': frame['heat_load']['node_ids'].pop()
    elif kind == 'wrong_model': source['provenance']['model'] = 'constant_lumped_thermal'
    elif kind == 'wrong_method': source['provenance']['method'] = 'explicit_euler'
    elif kind == 'wrong_units': source['provenance']['units']['temperature'] = 'C'
    elif kind == 'wrong_length_unit': source['provenance']['length_unit'] = 'mm'
    elif kind == 'time_nonuniform': frame['time_s'] += .01
    elif kind == 'time_nonmonotonic': frame['time_s'] = 0.
    elif kind == 'time_bool': frame['time_s'] = True
    elif kind == 'completed_not_bool': source['completed'] = 1
    elif kind == 'temperature_bool': frame['temperature_K'][0] = True
    elif kind == 'temperature_zero': frame['temperature_K'][0] = 0.
    elif kind == 'temperature_null': frame['temperature_K'][0] = None
    elif kind == 'temperature_shape': frame['temperature_K'].pop()
    elif kind == 'null_free_diagnostic': frame['free_node_residual_J'][0] = None
    elif kind == 'diagnostic_bool': frame['free_node_residual_J'][0] = True
    elif kind == 'negative_bound': frame['node_acceptance_bound_J'][0] = -1.
    elif kind == 'initial_diagnostic_zero': source['frames'][0]['free_node_residual_J'] = [0., 0., 0.]
    elif kind == 'initial_available': source['frames'][0]['interval_diagnostics_available'] = True
    elif kind == 'available_not_bool': frame['interval_diagnostics_available'] = 1
    elif kind == 'residual_beyond_bound': frame['free_node_residual_J'][0] = frame['node_acceptance_bound_J'][0] + 1.
    elif kind == 'component_beyond_bound': frame['component_energy_residual_J'][0] = frame['component_acceptance_bound_J'][0] + 1.
    elif kind == 'geometry_inference': source['provenance']['geometry_inference'] = True
    elif kind == 'time_accuracy': source['provenance']['time_accuracy_verified'] = True
    elif kind == 'capacity_profile': source['provenance']['thermal_capacity_reconstruction_profile'] = 'unknown'
    elif kind == 'component_shape': frame['component_energy_residual_J'].append(0.)
    elif kind == 'component_label': source['thermal_component_labels'][0] = 1
    elif kind == 'iteration_count': frame['accepted_iterations'] = 25
    elif kind == 'iteration_bool': source['iteration_history'][0][1] = True
    elif kind == 'law_key': source['provenance']['law']['G0_S']['011'] = source['provenance']['law']['G0_S'].pop('11')
    elif kind == 'voltage_key': frame['electrical']['prescribed_voltage_V']['0101'] = frame['electrical']['prescribed_voltage_V'].pop('101')
    elif kind == 'source_hash': source['input_sha256'] = 'not a digest'
    elif kind == 'shape': source['frames'] = {}
    elif kind == 'content_hash': saved['content_sha256'] = 'a'*64; return saved
    else: raise AssertionError(kind)
    return rehash(saved)


@pytest.mark.parametrize('kind', ['frame_G', 'nested_G', 'G0', 'nested_heat_power', 'heat_link',
    'instantaneous_label', 'node_duplicate', 'node_bool', 'edge_duplicate', 'missing_edge',
    'orientation', 'nested_orientation', 'heat_orientation', 'nested_missing_ID', 'wrong_model',
    'wrong_method', 'wrong_units', 'wrong_length_unit', 'time_nonuniform', 'time_nonmonotonic',
    'time_bool', 'completed_not_bool', 'temperature_bool', 'temperature_zero', 'temperature_null',
    'temperature_shape', 'null_free_diagnostic', 'diagnostic_bool', 'negative_bound',
    'initial_diagnostic_zero', 'initial_available', 'available_not_bool', 'component_shape',
    'component_label', 'iteration_count', 'iteration_bool', 'law_key', 'voltage_key', 'source_hash',
    'shape', 'content_hash', 'residual_beyond_bound', 'component_beyond_bound',
    'geometry_inference', 'time_accuracy', 'capacity_profile'])
def test_corruption_rejects_before_mechanics(saved_fixture, kind):
    graph, saved = saved_fixture
    mutate(saved, kind)
    with pytest.raises(ValueError):
        prepare(graph, saved)


@pytest.mark.parametrize('index', [True, 1.0, -1, 11, None])
def test_exact_existing_frame_required(saved_fixture, index):
    graph, saved = saved_fixture
    with pytest.raises(ValueError): prepare(graph, saved, frame_index=index)


@pytest.mark.parametrize('unit', ['cm', 'nm', 'mm', None])
def test_explicit_matching_units_required(saved_fixture, unit):
    graph, saved = saved_fixture
    with pytest.raises(ValueError): prepare(graph, saved, length_unit=unit)


def test_one_frame_partial_and_explicit_failure_preserved(saved_fixture):
    graph, saved = saved_fixture
    saved['source']['completed'] = False
    saved['source']['frames'] = saved['source']['frames'][:1]
    saved['source']['iteration_history'] = []
    saved['failure'] = {'reason': 'synthetic serialization fixture, not a physical run',
                        'attempted_step': 1, 'attempted_time_s': .1}
    rehash(saved)
    with pytest.raises(ValueError, match='allow_partial'): prepare(graph, saved, frame_index=0)
    with pytest.raises(ValueError, match='exact bool'): prepare(graph, saved, frame_index=0, allow_partial=1)
    snapshot = prepare(graph, saved, frame_index=0, allow_partial=True)
    assert snapshot.to_dict()['source_completed'] is False
    assert snapshot.to_dict()['source_frame_count'] == 1
    assert snapshot.to_dict()['source_failure'] == saved['failure']
    native = native_from_saved(saved)
    exported = bridge.export_saved_electrothermal(graph, native, length_unit='m', failure=saved['failure'])
    assert exported == saved


def test_native_fixed_diagnostic_nan_only_becomes_null(saved_fixture):
    graph, saved = saved_fixture
    native = native_from_saved(saved)
    initial = native.frames[0]
    later = replace(initial, time_s=.1, accepted_iterations=1, interval_diagnostics_available=True,
                    free_node_residual_J=(float('nan'), 0., 0.),
                    node_acceptance_bound_J=(float('nan'), 1., 1.),
                    component_energy_residual_J=(0.,), component_acceptance_bound_J=(1.,))
    fixture = replace(native, fixed_node_ids=(101,), fixed_temperature_K=(300.,),
                      frames=(initial, later), iteration_history=((1, 1, 0., True, True, None),))
    exported = bridge.export_saved_electrothermal(graph, fixture, length_unit='m')
    assert exported['source']['frames'][0]['free_node_residual_J'] == []
    assert exported['source']['frames'][1]['free_node_residual_J'] == [None, 0., 0.]
    assert exported['source']['frames'][1]['node_acceptance_bound_J'] == [None, 1., 1.]
    bad = replace(later, free_node_residual_J=(float('nan'), float('nan'), 0.))
    with pytest.raises(ValueError):
        bridge.export_saved_electrothermal(graph, replace(fixture, frames=(initial, bad)), length_unit='m')
    with pytest.raises(ValueError):
        bridge.export_saved_electrothermal(graph, replace(fixture, frames=(initial, replace(later,
            temperature_K=(float('nan'), 300., 300.)))), length_unit='m')


def test_three_dimensional_graph_rejected_without_projection(saved_fixture):
    graph, saved = saved_fixture
    graph._dimension = 3
    saved['graph'] = bridge._graph_record(graph)
    saved['graph_sha256'] = bridge._hash(saved['graph'])
    rehash(saved)
    with pytest.raises(ValueError, match='dimension=2'): prepare(graph, saved)


def test_reported_native_hash_is_not_claimed_authenticated(saved_fixture):
    graph, saved = saved_fixture
    saved['source']['input_sha256'] = 'a'*64
    rehash(saved)
    snap = prepare(graph, saved)
    assert snap.to_dict()['source_reported_input_sha256'] == 'a'*64
    assert 'not_reconstructed_or_authenticated' in snap.to_dict()['provenance_scope']['source_input_sha256']


def test_native_law_hash_uses_integer_keys_not_lexical_strings():
    from fibernet.sim.electrothermal import TemperatureConductanceLaw
    graph = StructureGraph.from_dict({'dimension': 2, 'nodes': [
        {'id': 2, 'position': [0., 0., 0.]}, {'id': 11, 'position': [1., 0., 0.]}],
        'edges': [{'id': 11, 'node_i': 2, 'node_j': 11, 'radius': .003},
                  {'id': 2, 'node_i': 11, 'node_j': 2, 'radius': .003}]})
    law = TemperatureConductanceLaw({11: 1., 2: 1.}, {11: .01, 2: .01},
        {11: 300., 2: 300.}, {11: {2: .5, 11: .5}, 2: {11: .5, 2: .5}}, (290., 350.))
    digest = hashlib.sha256(json.dumps(law.to_dict(), sort_keys=True, allow_nan=False).encode()).hexdigest()
    raw = json.loads(json.dumps(law.to_dict()))
    wrong = hashlib.sha256(json.dumps(raw, sort_keys=True, allow_nan=False).encode()).hexdigest()
    assert wrong != digest
    restored = bridge._law(raw, graph, digest)
    assert restored.to_dict() == law.to_dict()
    with pytest.raises(ValueError, match='integer-key'): bridge._law(raw, graph, wrong)


@pytest.mark.parametrize('raw', ['{"x":NaN}', '{"x":Infinity}', '{"x":1e999}',
                                '{"x":1,"x":2}', '{"a":{"x":1,"x":2}}', '[]'])
def test_strict_reader(tmp_path, raw):
    file = tmp_path/'bad.json'; file.write_text(raw)
    with pytest.raises(ValueError): bridge.read_saved_electrothermal_json(file)
    assert file.read_text() == raw


def test_reader_size_ceiling(tmp_path, monkeypatch):
    monkeypatch.setattr(bridge, '_MAX_JSON_BYTES', 10)
    file = tmp_path/'large.json'; file.write_text('{"long":12345}')
    with pytest.raises(ValueError, match='ceiling'): bridge.read_saved_electrothermal_json(file)


def test_module_import_is_stdlib_only():
    import builtins
    path = Path(bridge.__file__)
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert not name.startswith(('numpy', 'scipy', 'torch', 'taichi', 'fibernet'))
        return original(name, *args, **kwargs)
    spec = importlib.util.spec_from_file_location('_electrothermal_import_only', path)
    module = importlib.util.module_from_spec(spec)
    with patch.object(builtins, '__import__', side_effect=guarded): spec.loader.exec_module(module)


def test_partial_failure_known_dt_must_match_grid(saved_fixture):
    graph, saved = saved_fixture
    saved['source']['completed'] = False
    saved['failure'] = {'reason': 'synthetic failure context, not physics',
                        'attempted_step': 11, 'attempted_time_s': 2.}
    rehash(saved)
    with pytest.raises(ValueError, match='uniform grid'):
        prepare(graph, saved, allow_partial=True)


def test_recaptured_coordinate_tamper_still_fails_electrical_binding(saved_fixture):
    graph, saved = saved_fixture
    graph.nodes[303].position[0] += 1e-9
    saved['graph'] = bridge._graph_record(graph)
    saved['graph_sha256'] = bridge._hash(saved['graph'])
    rehash(saved)
    with pytest.raises(ValueError, match='electrical input fingerprint'):
        prepare(graph, saved)


def test_partial_without_failure_context_does_not_invent_it(saved_fixture):
    graph, saved = saved_fixture
    saved['source']['completed'] = False
    rehash(saved)
    with pytest.raises(ValueError, match='allow_partial'): prepare(graph, saved)
    snap = prepare(graph, saved, allow_partial=True)
    assert snap.to_dict()['source_failure'] is None
    assert snap.to_dict()['source_completed'] is False


def test_saved_hash_tampering_and_resource_shapes_reject_early(saved_fixture, monkeypatch):
    graph, saved = saved_fixture
    for key, value in [('frames', [None] * 2050), ('node_ids', [1] * 4097),
                       ('edge_ids', [1] * 16385), ('iteration_history', [None] * 51201)]:
        altered = copy.deepcopy(saved)
        altered['source'][key] = value
        with pytest.raises(ValueError, match='resource bounds'):
            prepare(graph, altered)
    altered = copy.deepcopy(saved)
    altered['graph_sha256'] = 'a' * 64
    rehash(altered)
    with pytest.raises(ValueError, match='captured graph SHA256'):
        prepare(graph, altered)


def test_partial_retention_failure_can_keep_terminal_trial_history(saved_fixture):
    graph, saved = saved_fixture
    saved['source']['completed'] = False
    saved['source']['frames'] = saved['source']['frames'][:1]
    saved['source']['iteration_history'] = [[1, 1, 0., True, True, None]]
    saved['failure'] = {'reason': 'synthetic append_frame retention failure, not physics',
                        'attempted_step': 1, 'attempted_time_s': .1}
    rehash(saved)
    snap = prepare(graph, saved, frame_index=0, allow_partial=True)
    assert snap.to_dict()['source_frame_count'] == 1
    assert snap.to_dict()['source_iteration_history'][-1][3:5] == [True, True]
    with pytest.raises(ValueError, match='existing accepted frame'):
        prepare(graph, saved, frame_index=1, allow_partial=True)
    saved['source']['iteration_history'].append([1, 2, 0., True, True, None])
    rehash(saved)
    with pytest.raises(ValueError, match='continued after'):
        prepare(graph, saved, frame_index=0, allow_partial=True)
