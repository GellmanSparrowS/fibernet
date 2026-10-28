"""Checked legacy projection contracts; no solver or new manufacturing model."""
import copy
from dataclasses import FrozenInstanceError, replace
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from fibernet import (StructureGraph, Material, FiberNetworkProfile,
                      FiberNetworkProjectionLimits, FiberNetworkProjectionError)
from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork
from test_fiber_network_bridge_ownership import scene, graph_state, network_state


def profile(**kwargs):
    return FiberNetworkProfile('mm', 'lab_xyz', **kwargs)


def lossy(**kwargs):
    return profile(geometry_policy='endpoint_chord', short_edge_policy='omit_legacy', **kwargs)


def simple(points=None, dimension=2):
    graph = StructureGraph(dimension=dimension)
    a = graph.add_node([0., 0., 2.], merge=False)
    b = graph.add_node([1., 0., 3.], merge=False)
    graph.add_edge(a, b, internal_points=points)
    return graph


def forbid_construction(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Fiber construction reached before complete admission')
    monkeypatch.setattr(Fiber, 'straight', forbidden)
    monkeypatch.setattr(FiberNetwork, '__init__', forbidden)
    import fibernet.core.structure_graph as owner
    monkeypatch.setattr(owner, 'copy', SimpleNamespace(deepcopy=forbidden))


def test_existing_scene_exact_mapping_values_and_report():
    graph = scene()
    before = graph_state(graph)
    network, report = graph.to_fiber_network(profile=lossy())
    assert type(network) is FiberNetwork
    assert network_state(network) == network_state(graph.to_fiber_network())
    assert [(r['edge_id'], r['fiber_row'], r['node_i'], r['node_j']) for r in report['edges']] == [
        (90, 0, 10, 30), (7, 1, 70, 30), (50, 2, 30, 10),
        (1000, 3, 100, 101), (1001, 4, 101, 100), (1002, None, 102, 103)]
    assert report['projected_internal_point_edge_ids'] == [90, 1000, 1001]
    assert report['omitted_edge_ids'] == [1002]
    assert report['unrepresented_node_ids'] == [80, 102, 103]
    assert report['missing_material_edge_ids'] == report['defaulted_material_edge_ids'] == []
    assert report['counts'] == {'source_nodes': 8, 'source_edges': 6, 'source_points': 12,
                                'fibers': 5, 'centerline_points': 50, 'crosslinks': 6}
    assert report['unit_conversion'] == report['frame_transform'] == 'none'
    assert report['metadata']['graph_disposition'] == 'copied_to_network_metadata'
    assert report['edges'][0]['metadata_fields'] == list(graph.edges[90].metadata)
    json.dumps(report, allow_nan=False)
    assert graph_state(graph) == before
    assert all(np.isinf(c.strength) and np.isinf(c.stiffness) for c in network.crosslinks)


def test_strict_scene_reports_all_policy_groups_before_construction(monkeypatch):
    graph = scene()
    graph.edges[1002].material = None
    forbid_construction(monkeypatch)
    with pytest.raises(FiberNetworkProjectionError) as caught:
        graph.to_fiber_network(profile=profile())
    assert caught.value.violations == {'internal_points': (90, 1000, 1001),
                                        'short_edges': (1002,), 'missing_materials': (1002,)}


@pytest.mark.parametrize('points', [None, np.empty((0, 3))])
def test_empty_absent_full_xyz_and_declared_dimension(points):
    graph = simple(points)
    network, report = graph.to_fiber_network(profile=profile())
    assert network.dimension == 2
    np.testing.assert_array_equal(network.fibers[0].start_point, [0., 0., 2.])
    np.testing.assert_array_equal(network.fibers[0].end_point, [1., 0., 3.])
    assert report['edges'][0]['internal_points_state'] == ('absent' if points is None else 'empty')


@pytest.mark.parametrize('kind', ['empty', 'isolate', 'disconnected', 'odd', 'coincident'])
def test_no_route_restrictions_or_implicit_welding(kind):
    graph = StructureGraph()
    if kind != 'empty':
        graph.add_node([0., 0.], merge=False)
    if kind in ('disconnected', 'odd', 'coincident'):
        b = graph.add_node([1., 0.], merge=False)
        graph.add_edge(0, b)
    if kind == 'disconnected':
        graph.add_node([4., 0.], merge=False)
    if kind == 'coincident':
        a = graph.add_node([0., 0.], merge=False)
        b = graph.add_node([1., 0.], merge=False)
        graph.add_edge(a, b)
    network, report = graph.to_fiber_network(profile=profile())
    assert network.num_crosslinks == 0
    assert report['counts']['fibers'] == graph.num_edges


@pytest.mark.parametrize('distance,omitted', [(0., True), (5e-13, True), (1e-12, False), (5e-11, False)])
def test_raw_legacy_threshold_not_obj_threshold(distance, omitted):
    graph = StructureGraph()
    a = graph.add_node([0., 0.], merge=False)
    b = graph.add_node([distance, 0.], merge=False)
    graph.add_edge(a, b)
    if omitted:
        with pytest.raises(FiberNetworkProjectionError):
            graph.to_fiber_network(profile=profile())
    network, report = graph.to_fiber_network(profile=lossy())
    assert network.num_fibers == (0 if omitted else 1)
    assert report['omitted_edge_ids'] == ([0] if omitted else [])


def test_omitted_curve_is_not_listed_as_projected():
    graph = simple(np.array([[2., 3., 4.]]))
    graph.nodes[1].position = graph.nodes[0].position.copy()
    _, report = graph.to_fiber_network(profile=lossy())
    assert report['nonempty_internal_point_edge_ids'] == [0]
    assert report['projected_internal_point_edge_ids'] == []
    assert report['omitted_edge_ids'] == [0]


def test_missing_material_only_defaults_retained_edges():
    graph = scene()
    graph.edges[90].material = graph.edges[1002].material = None
    network, report = graph.to_fiber_network(profile=lossy(missing_material_policy='legacy_default'))
    assert report['missing_material_edge_ids'] == [90, 1002]
    assert report['defaulted_material_edge_ids'] == [90]
    assert network.fibers[0].material.to_dict() == Material().to_dict()
    assert graph.edges[90].material is graph.edges[1002].material is None


def test_optional_none_shear_is_not_recomputed():
    graph = simple()
    graph.edges[0].material.shear_modulus = None
    network, report = graph.to_fiber_network(profile=profile())
    assert network.fibers[0].material.shear_modulus is None
    assert graph.edges[0].material.shear_modulus is None
    assert report['edges'][0]['material_origin'] == 'stored_origin_unknown'
    assert 'shear_modulus' in report['edges'][0]['material_missing_optional_fields']


@pytest.mark.parametrize('direction', ['source', 'output', 'report'])
def test_three_way_detachment_and_repeated_conversion(direction):
    graph = scene()
    first, report = graph.to_fiber_network(profile=lossy())
    second, second_report = graph.to_fiber_network(profile=lossy())
    old_graph, old_first, old_second = graph_state(graph), network_state(first), network_state(second)
    old_report = copy.deepcopy(second_report)
    if direction == 'source':
        graph.edges[90].material.extra['nested']['tags'].append('changed')
        graph.metadata['nested']['values'].append(2)
        assert network_state(first) == old_first
    elif direction == 'output':
        first.fibers[0].material.extra['nested']['tags'].append('changed')
        first.box_size[0] += 1
        first.metadata['nested']['values'].append(2)
        assert graph_state(graph) == old_graph
    else:
        report['edges'][0]['metadata_fields'].append('new')
        report['source_node_order'].append(999)
        assert graph_state(graph) == old_graph and network_state(first) == old_first
    assert network_state(second) == old_second and second_report == old_report


@pytest.mark.parametrize('change', ['radius_zero', 'radius_nan', 'segments_bool', 'segments_zero',
    'edge_id', 'node_id', 'endpoint', 'self_loop', 'curve_shape', 'curve_nan', 'masked',
    'box_shape', 'box_negative', 'material_dynamic', 'material_extra', 'material_nan',
    'material_tensor', 'metadata_cycle', 'metadata_object', 'metadata_keys', 'unit', 'curve_unit', 'box_unit'])
def test_invalid_input_even_on_omitted_edges_rejects_before_construction(monkeypatch, change):
    graph = scene()
    edge = graph.edges[1002]
    if change == 'radius_zero': edge.radius = 0.
    elif change == 'radius_nan': edge.radius = float('nan')
    elif change == 'segments_bool': edge.segments = True
    elif change == 'segments_zero': edge.segments = 0
    elif change == 'edge_id': edge.edge_id = 9
    elif change == 'node_id': graph.nodes[102].node_id = 9
    elif change == 'endpoint': edge.node_j = 999
    elif change == 'self_loop': edge.node_j = edge.node_i
    elif change == 'curve_shape': edge.internal_points = np.zeros((2, 2))
    elif change == 'curve_nan': edge.internal_points = np.array([[0., np.nan, 0.]])
    elif change == 'masked': graph.nodes[102].position = np.ma.array([0., 0., 0.], mask=False)
    elif change == 'box_shape': graph._box_size = np.zeros(1000)
    elif change == 'box_negative': graph._box_size[0] = -1
    elif change == 'material_dynamic': edge.material.unbounded = object()
    elif change == 'material_extra': edge.material.extra = {'bad': object()}
    elif change == 'material_nan': edge.material.density = float('nan')
    elif change == 'material_tensor': edge.material.stiffness_tensor = np.zeros((600, 600))
    elif change == 'metadata_cycle': graph.metadata['self'] = graph.metadata
    elif change == 'metadata_object': edge.metadata['bad'] = object()
    elif change == 'metadata_keys': edge.metadata[1] = 2
    elif change == 'unit': graph.metadata['units'] = 'm'
    elif change == 'curve_unit': edge.metadata['geometry'] = {'internal_points_unit': 'm'}
    else: graph.metadata['box_size_unit'] = 'm'
    forbid_construction(monkeypatch)
    with pytest.raises(ValueError): graph.to_fiber_network(profile=lossy())


@pytest.mark.parametrize('limit', ['max_nodes', 'max_edges', 'max_source_points',
    'max_output_points', 'max_crosslinks', 'max_metadata_bytes', 'max_report_bytes'])
def test_every_budget_rejects_before_construction(monkeypatch, limit):
    graph = scene()
    forbid_construction(monkeypatch)
    with pytest.raises(MemoryError):
        graph.to_fiber_network(profile=lossy(limits=replace(FiberNetworkProjectionLimits(), **{limit: 1})))


def test_shared_material_copies_are_charged_per_edge(monkeypatch):
    graph = simple()
    graph.edges[0].material.extra = {'payload': 'x' * 900}
    limits = replace(FiberNetworkProjectionLimits(), max_metadata_bytes=2000)
    graph.to_fiber_network(profile=profile(limits=limits))
    graph.add_edge(1, 0, material=graph.edges[0].material, allow_parallel=True)
    forbid_construction(monkeypatch)
    with pytest.raises(MemoryError): graph.to_fiber_network(profile=profile(limits=limits))


def test_retained_default_materials_are_charged(monkeypatch):
    graph = simple()
    graph.edges[0].material = None
    limits = replace(FiberNetworkProjectionLimits(), max_metadata_bytes=1000)
    graph.to_fiber_network(profile=profile(limits=limits, missing_material_policy='legacy_default'))
    graph.add_edge(1, 0, allow_parallel=True)
    graph.edges[1].material = None
    graph.add_edge(0, 1, allow_parallel=True)
    graph.edges[2].material = None
    forbid_construction(monkeypatch)
    with pytest.raises(MemoryError):
        graph.to_fiber_network(profile=profile(limits=limits, missing_material_policy='legacy_default'))


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.longdouble])
@pytest.mark.parametrize('distance', [1e-12, .5, 1., 2.])
def test_accepted_storage_uses_exact_legacy_decisions(dtype, distance):
    graph = simple()
    graph.nodes[0].position = np.array([0., 0., 0.], dtype=dtype)
    graph.nodes[1].position = np.array([distance, 0., 0.], dtype=dtype)
    # Decimal longdouble values are constructed from binary64 values above,
    # so every coordinate itself is exactly representable as binary64.
    network, report = graph.to_fiber_network(profile=lossy())
    legacy = graph.to_fiber_network()
    assert network_state(network) == network_state(legacy)
    assert report['counts']['centerline_points'] == sum(len(f.centerline) for f in legacy.fibers)
    assert report['counts']['fibers'] == len(legacy.fibers)


@pytest.mark.parametrize('location', ['limit', 'edge_id', 'endpoint', 'dimension'])
def test_custom_integer_hooks_are_not_executed(monkeypatch, location):
    calls = []
    class HookedInt(int):
        def __int__(self):
            calls.append('int')
            raise AssertionError('custom integer conversion ran')
        def __lt__(self, other):
            calls.append('lt')
            raise AssertionError('custom integer comparison ran')
    if location == 'limit':
        with pytest.raises(ValueError): FiberNetworkProjectionLimits(max_edges=HookedInt(1))
    else:
        graph = simple()
        if location == 'edge_id': graph.edges[0].edge_id = HookedInt(0)
        elif location == 'endpoint': graph.edges[0].node_i = HookedInt(0)
        else: graph._dimension = HookedInt(2)
        forbid_construction(monkeypatch)
        with pytest.raises(ValueError): graph.to_fiber_network(profile=profile())
    assert calls == []


def test_active_json_integer_digit_guard(monkeypatch):
    current = getattr(sys, 'get_int_max_str_digits', lambda: 0)()
    if not current:
        # Exercise the guard without changing the process-wide encoder setting.
        monkeypatch.setattr(sys, 'get_int_max_str_digits', lambda: 4300, raising=False)
        current = 4300
    graph = simple()
    edge = graph.edges.pop(0)
    edge.edge_id = 10 ** (current + 1)
    graph.edges[edge.edge_id] = edge
    forbid_construction(monkeypatch)
    with pytest.raises(ValueError, match='JSON digit limit'):
        graph.to_fiber_network(profile=profile())


@pytest.mark.parametrize('case', ['norm_overflow', 'norm_underflow', 'subnormal', 'integer_subtraction'])
def test_numerical_failure_is_not_authorized_omission(monkeypatch, case):
    graph = simple()
    if case == 'norm_overflow':
        graph.nodes[0].position = np.array([0., 0., 0.])
        graph.nodes[1].position = np.array([1e200, 0., 0.])
    elif case == 'norm_underflow':
        graph.nodes[0].position = np.array([0., 0., 0.])
        graph.nodes[1].position = np.array([1e-200, 0., 0.])
    elif case == 'subnormal':
        graph.nodes[0].position = np.array([0., 0., 0.])
        graph.nodes[1].position = np.array([float.fromhex('0x0.0000000000001p-1022'), 0., 0.])
    else:
        graph.nodes[0].position = np.array([-2**62, 0, 0], dtype=np.int64)
        graph.nodes[1].position = np.array([2**62, 0, 0], dtype=np.int64)
    forbid_construction(monkeypatch)
    with pytest.raises(OverflowError): graph.to_fiber_network(profile=lossy())


def test_large_stable_ids_are_not_int64_rows():
    huge = 2**80
    graph = StructureGraph.from_dict({'nodes': [{'id': huge, 'position': [0., 0.]},
                                               {'id': huge + 9, 'position': [1., 0.]}],
        'edges': [{'id': huge + 40, 'node_i': huge + 9, 'node_j': huge}]})
    _, report = graph.to_fiber_network(profile=profile())
    assert report['edges'][0]['edge_id'] == huge + 40
    assert report['edges'][0]['node_i'] == huge + 9
    assert report['edges'][0]['fiber_row'] == 0


def test_giant_report_id_is_bounded_before_decimal_encoding(monkeypatch):
    graph = simple()
    edge = graph.edges.pop(0)
    edge.edge_id = 1 << 100000
    graph.edges[edge.edge_id] = edge
    forbid_construction(monkeypatch)
    with pytest.raises(MemoryError):
        graph.to_fiber_network(profile=profile(limits=replace(FiberNetworkProjectionLimits(), max_report_bytes=4096)))


@pytest.mark.parametrize('kwargs', [{'geometry_policy': True}, {'short_edge_policy': 'omit'},
    {'missing_material_policy': 'ignore'}, {'length_unit': ''}, {'coordinate_frame': ''}, {'limits': {}}])
def test_invalid_profile(kwargs):
    options = dict(length_unit='mm', coordinate_frame='lab_xyz')
    options.update(kwargs)
    with pytest.raises((ValueError, TypeError)): FiberNetworkProfile(**options)


@pytest.mark.parametrize('value', [True, False, 0, -1, 1., np.bool_(True)])
def test_invalid_limits(value):
    with pytest.raises(ValueError): FiberNetworkProjectionLimits(max_edges=value)


def test_profile_frozen_and_public_aliases():
    from fibernet.core import FiberNetworkProfile as public
    assert public is FiberNetworkProfile
    with pytest.raises(FrozenInstanceError): profile().length_unit = 'm'
    assert type(simple().to_fiber_network()) is FiberNetwork
    with pytest.raises(TypeError): simple().to_fiber_network(profile={})


@pytest.mark.parametrize('value', ['\x7f', {'\x7f': ['\x7f' * 5]}, '\"\\\b\f\n\r\t\u0080\U0001f600'])
def test_json_byte_budget_matches_ascii_escape_boundaries(value):
    from fibernet.core.fiber_network_profile import _JsonBudget
    actual = len(json.dumps(value, ensure_ascii=True, separators=(',', ':')).encode('ascii'))
    exact = _JsonBudget(actual, 'test')
    exact.charge(value)
    assert exact.used == actual
    with pytest.raises(MemoryError): _JsonBudget(actual - 1, 'test').charge(value)


@pytest.mark.parametrize('location', ['graph', 'node', 'edge'])
def test_unrelated_nested_units_and_frames_are_not_geometry(location):
    graph = simple()
    target = graph.metadata if location == 'graph' else graph.nodes[0].metadata if location == 'node' else graph.edges[0].metadata
    target.update(load={'units': 'N', 'coordinate_frame': 'force_local', 'length_unit': 'm'},
                  stress={'units': 'Pa'}, calibration={'coordinate_unit': 'custom'})
    before = copy.deepcopy(target)
    network, _ = graph.to_fiber_network(profile=profile())
    assert target == before
    if location == 'graph':
        assert network.metadata == before


@pytest.mark.parametrize('location', ['root', 'geometry', 'nested_geometry'])
@pytest.mark.parametrize('declaration', [{'units': 'm'}, {'units': {'radius': 'm'}},
    {'coordinate_frame': 'other'}, {'internal_points_unit': 'm'}, {'box_size_unit': 'm'}])
def test_exact_geometry_declaration_scopes_reject_conflicts(monkeypatch, location, declaration):
    graph = simple()
    if location == 'root': graph.metadata.update(declaration)
    elif location == 'geometry': graph.metadata['geometry'] = declaration
    else: graph.metadata['provenance'] = {'geometry': declaration}
    forbid_construction(monkeypatch)
    with pytest.raises(ValueError, match='declaration conflicts'):
        graph.to_fiber_network(profile=profile())


def test_binary64_identity_preserves_subnormals_and_equates_signed_zero():
    from fibernet.core.fiber_network_profile import _float64_identity
    assert _float64_identity(0.) == _float64_identity(-0.) == 0
    assert _float64_identity(float.fromhex('0x0.0000000000001p-1022')) == 1
    assert _float64_identity(float.fromhex('0x0.0000000000002p-1022')) == 2
    graph = simple()
    graph.nodes[0].position = np.array([0., 0., 0.])
    graph.nodes[1].position = np.array([-0., 0., -0.])
    network, report = graph.to_fiber_network(profile=lossy())
    assert network.num_fibers == 0 and report['omitted_edge_ids'] == [0]
