"""Hand-built legacy summaries and explicit native graph analysis contracts."""
import copy
import numpy as np
import pytest
from unittest.mock import patch
from fibernet.api import analyze
from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork, Crosslink
from fibernet.core.structure_graph import StructureGraph

KEYS = ('num_fibers', 'num_crosslinks', 'dimension', 'nematic_order',
        'mean_length', 'total_length', 'mean_tortuosity', 'num_nodes',
        'num_edges', 'mean_degree', 'is_connected', 'num_components')


def fiber(points):
    return Fiber(np.array(points, dtype=np.float64))


def test_empty_and_one():
    empty = analyze(FiberNetwork())
    assert tuple(empty) == KEYS
    assert empty['num_components'] == empty['mean_tortuosity'] == 0
    assert empty['is_connected'] is False
    result = analyze(FiberNetwork(fibers=[fiber([[0, 0, 0], [2, 0, 0]])]))
    assert tuple(result) == KEYS
    assert (result['num_nodes'], result['num_edges'], result['num_components']) == (2, 1, 1)
    assert result['total_length'] == result['mean_length'] == 2
    assert result['mean_tortuosity'] == result['mean_degree'] == result['nematic_order'] == 1
    assert result['is_connected'] is True


def test_endpoint_chain_curvature_and_no_mutation():
    network = FiberNetwork(fibers=[fiber([[0, 0, 0], [1, 1, 0], [2, 0, 0]]),
                                  fiber([[2, 0, 0], [3, 0, 0]])], dimension=2)
    before = copy.deepcopy(network.to_dict())
    with patch('fibernet.analysis._archived.topology.TopologyAnalyzer.full_report',
               side_effect=AssertionError('full report forbidden')):
        result = analyze(network)
    assert network.to_dict() == before
    assert (result['num_nodes'], result['num_edges']) == (3, 2)
    assert result['total_length'] == pytest.approx(2 * np.sqrt(2) + 1)
    assert result['mean_tortuosity'] == pytest.approx((np.sqrt(2) + 1) / 2)
    assert result['mean_degree'] == pytest.approx(4 / 3)


def test_disconnected():
    result = analyze(FiberNetwork(fibers=[fiber([[0, 0, 0], [1, 0, 0]]),
                                             fiber([[3, 0, 0], [4, 0, 0]])]))
    assert result['num_components'] == 2 and result['is_connected'] is False


@pytest.mark.parametrize('fibers', [
    [fiber([[0, 0, 0], [1, 0, 0]]), fiber([[1, 0, 0], [0, 0, 0]])],
    [fiber([[0, 0, 0], [1, 0, 0]]), fiber([[1.0000001, 0, 0], [2, 0, 0]])],
    [fiber([[0, 0, 0], [1, 0, 0], [0, 0, 0]])],
    [fiber([[0, 0, 0], [1e-13, 0, 0]])],
    [fiber([[0, 0, 0], [float('nan'), 0, 0]])],
    [fiber([[0, 0, 0], [1e200, 0, 0]])],
])
def test_reject_ambiguous_or_numerically_undefined(fibers):
    with pytest.raises((ValueError, FloatingPointError, OverflowError)):
        analyze(FiberNetwork(fibers=fibers))


def test_no_conversion_periodic_or_crosslink_guessing():
    with pytest.raises(ValueError, match="kind='topology'"):
        analyze(StructureGraph())
    with pytest.raises(ValueError, match='periodic'):
        analyze(FiberNetwork(periodic=True))
    with pytest.raises(ValueError, match='crosslink'):
        analyze(FiberNetwork(crosslinks=[Crosslink(0, 1, 1., 0., np.zeros(3))]))


def test_numpy_state_underflow_and_malformed_geometry():
    state = np.geterr().copy()
    bad = fiber([[0, 0, 0], [1e-200, 0, 0], [1, 0, 0]])
    with pytest.raises(ValueError, match='underflow'):
        analyze(FiberNetwork(fibers=[bad]))
    assert np.geterr() == state
    for points in (np.zeros((1, 3)), np.zeros((2, 2)),
                   np.zeros((2, 3), dtype=np.float32)):
        bad = fiber([[0, 0, 0], [1, 0, 0]])
        bad.centerline = points
        with pytest.raises(ValueError, match='centerline'):
            analyze(FiberNetwork(fibers=[bad]))
    with pytest.raises(ValueError, match='2D'):
        analyze(FiberNetwork(fibers=[fiber([[0, 0, 0], [1, 0, 1]])], dimension=2))
    assert np.geterr() == state


def test_orientation_conventions_and_dependency_error():
    orthogonal = [fiber([[0, 0, 0], [1, 0, 0]]),
                  fiber([[2, 0, 0], [2, 1, 0]])]
    assert analyze(FiberNetwork(fibers=orthogonal, dimension=2))['nematic_order'] == pytest.approx(0, abs=1e-14)
    axes = orthogonal + [fiber([[3, 0, 0], [3, 0, 1]])]
    assert analyze(FiberNetwork(fibers=axes, dimension=3))['nematic_order'] == pytest.approx(0, abs=1e-14)
    import sys
    with patch.dict(sys.modules, {'networkx': None}):
        with pytest.raises(ImportError):
            analyze(FiberNetwork())


def test_native_topology_explicit_delegation_and_wrong_kinds():
    from fibernet.analysis import analyze_topology, TopologyMetricsV1
    graph = StructureGraph.from_dict({
        'version': '3.1', 'dimension': 3,
        'nodes': [{'id': 9, 'position': [0, 0, 0]},
                  {'id': 20, 'position': [2, 0, 0]},
                  {'id': 31, 'position': [5, 0, 0]}],
        'edges': [{'id': 4, 'node_i': 9, 'node_j': 20, 'radius': .1,
                   'internal_points': [[1, 1, 0]]},
                  {'id': 8, 'node_i': 9, 'node_j': 20, 'radius': .2}],
    })
    before = copy.deepcopy(graph.to_dict())
    expected = analyze_topology(graph)
    with patch('fibernet.analysis.analyze_topology', wraps=analyze_topology) as delegated:
        result = analyze(graph, kind='topology')
    delegated.assert_called_once_with(graph)
    assert isinstance(result, TopologyMetricsV1) and result == expected
    assert result.schema == 'fibernet.topology/1'
    assert result.n_physical_edges == 2 and result.n_unique_connections == 1
    assert result.node_ids == (9, 20, 31) and result.edge_ids == (4, 8)
    assert result.n_isolates == 1 and graph.to_dict() == before
    # Real source-package journey: native JSON representation roundtrip, then
    # topology and declared-unit curve geometry; never a legacy conversion.
    import json
    from fibernet.analysis import analyze_geometry_paths
    restored = StructureGraph.from_dict(json.loads(json.dumps(before)))
    assert restored.to_dict() == before
    assert analyze(restored, kind='topology') == result
    geometry = analyze_geometry_paths(restored, length_unit='mm',
                                      coordinate_frame='handbuilt_reference')
    assert geometry.edge_ids == (4, 8)
    assert geometry.polyline_length_by_edge == pytest.approx((2 * np.sqrt(2), 2))
    assert geometry.length_unit == 'mm'
    for kind in (None, 'geometry', 'unknown'):
        with pytest.raises(ValueError, match='topology'):
            analyze(graph, kind=kind)
    with pytest.raises(ValueError, match='FiberNetwork'):
        analyze(FiberNetwork(), kind='topology')
    with pytest.raises(TypeError, match='no conversion'):
        analyze(object(), kind='topology')
