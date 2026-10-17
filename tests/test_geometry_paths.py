"""Independent analytic geometry/path oracles, separate from legacy features."""
import math

import numpy as np
import pytest

from fibernet.analysis.geometry_paths import analyze_geometry_paths
from fibernet.core.structure_graph import StructureGraph


def make_graph(points, edges):
    g = StructureGraph.from_dict({'dimension': 3, 'tolerance': 1.0,
        'nodes': [{'id': n, 'position': p} for n, p in points],
        'edges': [{'id': e, 'node_i': a, 'node_j': b, 'internal_points': ip}
                  for e, a, b, ip in edges]})
    return g


def analyze(g, pairs=(), **kwargs):
    return analyze_geometry_paths(g, length_unit='mm', coordinate_frame='lab',
                                  node_pairs=pairs, **kwargs)


def test_xyz_curved_parallel_edges_and_pair_order():
    g = make_graph([(10, [0, 0, 0]), (30, [0, 0, 4]), (70, [0, 0, 8])],
                   [(90, 10, 30, [[3, 0, 2]]), (7, 30, 10, None),
                    (50, 30, 70, None)])
    m = analyze(g, [(10, 70), (30, 10), (10, 70)])
    assert m.schema == 'fibernet.geometry_paths/1'
    assert m.edge_ids == (90, 7, 50)
    assert m.chord_length_by_edge == (4, 4, 4)
    assert m.polyline_length_by_edge == pytest.approx((2 * math.sqrt(13), 4, 4))
    assert m.curve_straightness_by_edge == pytest.approx((2 / math.sqrt(13), 1, 1))
    assert m.pair_node_ids == ((10, 70), (30, 10), (10, 70))
    assert m.pair_distance == (8, 4, 8)
    assert m.pair_tortuosity == (1, 1, 1)
    assert m.pair_status == ('ok',) * 3
    assert (m.length_unit, m.coordinate_frame) == ('mm', 'lab')
    g.remove_edge(7)
    assert analyze(g, [(10, 30)]).pair_distance[0] == pytest.approx(2 * math.sqrt(13))
    assert analyze(g, [(10, 30)], distance_mode='chord').pair_distance == (4,)


@pytest.mark.parametrize('scale', [1., 1e-200, 1e200])
def test_stable_diagonal_polyline_and_scale_invariance(scale):
    g = make_graph([(0, [0, 0, 0]), (1, [scale, scale, scale]),
                    (2, [2 * scale, 2 * scale, 2 * scale])],
                   [(4, 0, 1, [[scale / 2] * 3]), (8, 1, 2, None)])
    m = analyze(g, [(0, 2), (2, 0)])
    # Divide first: absolute pytest tolerance must not hide tiny values as zero.
    assert m.chord_length_by_edge[0] / scale == pytest.approx(math.sqrt(3))
    assert m.polyline_length_by_edge[0] / scale == pytest.approx(math.sqrt(3))
    assert m.pair_distance[0] / scale == pytest.approx(2 * math.sqrt(3))
    assert m.curve_straightness_by_edge == pytest.approx((1, 1))
    assert m.pair_tortuosity == pytest.approx((1, 1))


def test_empty_unreachable_and_exact_zero_chord():
    empty = analyze(make_graph([], []))
    assert empty.edge_ids == empty.pair_status == ()
    g = make_graph([(0, [0, 0, 0]), (1, [0, 0, 0]), (2, [2, 0, 0])],
                   [(4, 0, 1, None)])
    m = analyze(g, [(0, 2), (0, 1), (0, 0)])
    assert m.degenerate_edge_ids == (4,)
    assert m.curve_straightness_by_edge == (None,)
    assert m.pair_status == ('unreachable', 'zero_chord', 'zero_chord')
    assert m.pair_distance == (None, 0, 0)
    assert m.pair_tortuosity == (None, None, None)
    g.edges[4].internal_points = np.array([[1, 0, 0]])
    m = analyze(g, [(0, 1)])
    assert m.curve_straightness_by_edge == (0,)
    assert m.pair_distance == (2,)
    assert m.pair_tortuosity == (None,)


def test_reversed_records_and_geometry_do_not_mutate_graph():
    g = make_graph([(10, [1, 2, 3]), (30, [4, 6, 9])],
                   [(90, 10, 30, [[2, 4, 5], [3, 5, 8]])])
    before = g.to_dict()
    a = analyze(g, [(10, 30), (30, 10)])
    assert g.to_dict() == before
    edge = g.edges[90]
    edge.node_i, edge.node_j = edge.node_j, edge.node_i
    edge.internal_points = edge.internal_points[::-1].copy()
    b = analyze(g, [(10, 30), (30, 10)])
    assert a == b


@pytest.mark.parametrize('bad', [True, 1.0, -1, np.bool_(True)])
def test_invalid_pair_ids(bad):
    g = make_graph([(0, [0, 0, 0]), (1, [1, 0, 0])], [(1, 0, 1, None)])
    with pytest.raises(ValueError, match='pair'):
        analyze(g, [(0, bad)])


@pytest.mark.parametrize('bad', [True, 1.0, 0, -1, np.bool_(True)])
def test_strict_pair_budget(bad):
    with pytest.raises(ValueError, match='max_pairs'):
        analyze(make_graph([], []), max_pairs=bad)


def test_pair_budget_shapes_and_declared_units():
    g = make_graph([(0, [0, 0, 0])], [])
    with pytest.raises(MemoryError, match='max_pairs'):
        analyze(g, ((0, 0) for _ in range(3)), max_pairs=2)
    for pairs in [[(0,)], [(0, 0, 0)], np.ma.array([[0, 0]])]:
        with pytest.raises(ValueError):
            analyze(g, pairs)
    assert analyze(g, [(np.int64(0), np.uint64(0))], max_pairs=np.int64(1)).pair_node_ids == ((0, 0),)
    with pytest.raises(ValueError, match='distance_mode'):
        analyze(g, distance_mode='hops')
    for unit, frame in [('', 'lab'), ('mm', ''), (True, 'lab')]:
        with pytest.raises(ValueError):
            analyze_geometry_paths(g, length_unit=unit, coordinate_frame=frame)


@pytest.mark.parametrize('field,bad', [('position', [0, float('nan'), 0]),
    ('position', [0, 1]), ('position', np.ma.array([0, 1, 2])),
    ('internal_points', [[0, float('inf'), 0]]),
    ('internal_points', [[0, 1]]), ('internal_points', np.ma.array([[0, 1, 2]]))])
def test_invalid_geometry_and_masks(field, bad):
    g = make_graph([(0, [0, 0, 0]), (1, [1, 0, 0])], [(1, 0, 1, None)])
    target = g.nodes[0] if field == 'position' else g.edges[1]
    setattr(target, field, bad)
    with pytest.raises(ValueError):
        analyze(g)


def test_subtraction_and_polyline_sum_overflow():
    g = make_graph([(0, [-1e308, 0, 0]), (1, [1e308, 0, 0])], [(1, 0, 1, None)])
    with pytest.raises(OverflowError):
        analyze(g)
    g = make_graph([(0, [0, 0, 0]), (1, [1e308, 0, 0])],
                   [(1, 0, 1, [[1e308, 0, 0], [0, 0, 0]])])
    with pytest.raises(OverflowError):
        analyze(g)


@pytest.mark.parametrize('budget', ['max_nodes', 'max_edges', 'max_points', 'max_path_work'])
@pytest.mark.parametrize('bad', [True, 1.0, 0, -1])
def test_all_budgets_are_strict_positive_integers(budget, bad):
    with pytest.raises(ValueError, match=budget):
        analyze(make_graph([], []), **{budget: bad})


@pytest.mark.parametrize('kwargs', [{'max_nodes': 1}, {'max_edges': 1},
    {'max_points': 2}, {'max_path_work': 3}])
def test_budgets_fail_before_traversal(kwargs, monkeypatch):
    import networkx as nx
    def unexpected(*args, **kw):
        raise AssertionError('path algorithm must not run above budget')
    monkeypatch.setattr(nx, 'shortest_path', unexpected)
    g = make_graph([(0, [0, 0, 0]), (1, [1, 1, 1])],
                   [(3, 0, 1, [[.5, .5, .5]]), (8, 1, 0, None)])
    with pytest.raises(MemoryError):
        analyze(g, [(0, 1)], **kwargs)


@pytest.mark.parametrize('bad', [True, 1.0, -1])
@pytest.mark.parametrize('kind', ['node_key', 'node_record', 'edge_key', 'edge_record', 'endpoint'])
def test_graph_identity_is_strict(kind, bad):
    g = make_graph([(0, [0, 0, 0]), (1, [1, 1, 1])], [(1, 0, 1, None)])
    if kind == 'node_key':
        g.nodes[bad] = g.nodes.pop(1)
    elif kind == 'node_record':
        g.nodes[1].node_id = bad
    elif kind == 'edge_key':
        g.edges[bad] = g.edges.pop(1)
    elif kind == 'edge_record':
        g.edges[1].edge_id = bad
    else:
        g.edges[1].node_j = bad
    with pytest.raises(ValueError):
        analyze(g)


def test_disconnected_coincident_precedence_and_path_overflow():
    g = make_graph([(0, [0, 0, 0]), (1, [0, 0, 0])], [])
    m = analyze(g, [(0, 1)])
    assert m.pair_status == ('unreachable',) and m.pair_chord == (0,)
    assert m.pair_distance == m.pair_tortuosity == (None,)
    g = make_graph([(0, [0, 0, 0]), (1, [1e308, 0, 0]), (2, [0, 0, 0])],
                   [(1, 0, 1, None), (2, 1, 2, None)])
    with pytest.raises(OverflowError):
        analyze(g, [(0, 2)])


def test_graph_numpy_ids_output_and_self_loop():
    g = make_graph([(0, [0, 0, 0]), (1, [1, 1, 1])], [(1, 0, 1, None)])
    g.nodes[0].node_id = np.int64(0)
    g.edges[1].edge_id = np.uint64(1)
    g.edges[1].node_j = np.int64(1)
    m = analyze(g, [(np.int64(0), np.uint64(1))])
    assert all(type(x) is int for x in m.node_ids + m.edge_ids + m.pair_node_ids[0])
    g.edges[1].node_j = 0
    with pytest.raises(ValueError, match='self-loops'):
        analyze(g)


def test_ratio_range_failures_are_not_valid_zero_or_infinity():
    g = make_graph([(0, [0, 0, 0]), (1, [1e-200, 0, 0])],
                   [(1, 0, 1, [[1e200, 0, 0]])])
    with pytest.raises(OverflowError, match='straightness'):
        analyze(g)


def test_alternative_route_uses_polyline_cost_and_result_is_frozen():
    from dataclasses import FrozenInstanceError
    g = make_graph([(10, [0, 0, 0]), (30, [1, 0, 0]), (70, [2, 0, 0])],
        [(1, 70, 10, [[1, 3, 0]]), (2, 10, 30, None), (3, 70, 30, None)])
    m = analyze(g, [(10, 70), (70, 10)])
    assert m.pair_distance == (2, 2)
    assert m.pair_tortuosity == (1, 1)
    with pytest.raises(FrozenInstanceError):
        m.distance_mode = 'chord'
