"""Focused generation contracts; TPMS tests exercise routing, never meshing."""
import copy

import numpy as np
import pytest

import fibernet as fn
from fibernet.gen import pattern


TPMS_UNITS = ('gyroid', 'schwarz_p', 'schwarz_d', 'iwp', 'neovius', 'lidinoid')


@pytest.mark.parametrize('count', (
    -1, np.int64(-1), True, False, np.bool_(True), np.bool_(False),
    0.0, 1.0, -0.5, float('nan'), float('inf'), '1', None,
))
def test_public_3d_invalid_node_count_precedes_factory_and_allocation(monkeypatch, count):
    def forbidden(*args, **kwargs):
        raise AssertionError('invalid count must fail before delegation or allocation')
    monkeypatch.setitem(pattern._UNIT_FACTORIES_3D, 'cubic', forbidden)
    monkeypatch.setattr(pattern, 'StructureGraph', forbidden)
    monkeypatch.setattr(pattern, 'normalize_grid', forbidden)
    with pytest.raises(ValueError, match='n_pts_per_side must be a nonnegative integer'):
        fn.pattern_3d(unit='cubic', grid=(1, 1, 1), n_pts_per_side=count)


@pytest.mark.parametrize('count', (0, 1, 2, np.int32(0), np.int32(1), np.int64(0), np.int64(1), np.uint64(1)))
def test_public_3d_integer_node_counts_preserve_geometry(monkeypatch, count):
    # Exact explicit displacements avoid RNG and isolate count admission.
    def forbidden_rng(*args, **kwargs):
        raise AssertionError('explicit displacements must not create an RNG')
    monkeypatch.setattr(pattern.np.random, 'default_rng', forbidden_rng)
    kwargs = dict(unit='cubic', box=(10., 10., 10.), grid=(1, 1, 1),
                  point_displacements=[(0., 0., 0.)] * (12 * int(count)))
    expected = fn.pattern_3d(n_pts_per_side=int(count), **kwargs)
    actual = fn.pattern_3d(n_pts_per_side=count, **kwargs)
    assert _snapshot(actual) == _snapshot(expected)
    assert (actual.num_nodes, actual.num_edges) == (8 + 12 * int(count), 12 * (int(count) + 1))
    assert actual.metadata['pattern']['n_pts_per_side'] == int(count)
    assert type(actual.metadata['pattern']['n_pts_per_side']) is int


def _snapshot(graph):
    return {
        'nodes': [(nid, node.position.tolist(), tuple(node.boundary))
                  for nid, node in graph.nodes.items()],
        'edges': [(eid, edge.node_i, edge.node_j, edge.radius, edge.segments,
                   None if edge.internal_points is None else edge.internal_points.tolist())
                  for eid, edge in graph.edges.items()],
    }


def _cubic(**kwargs):
    return fn.pattern_3d(unit='cubic', box=(10., 10., 10.), grid=(1, 1, 1),
                         radius=.1, seed=23, **kwargs)


@pytest.mark.parametrize('unit', TPMS_UNITS)
def test_tpms_rejects_unsupported_controls_before_builder(monkeypatch, unit):
    def forbidden(*args, **kwargs):
        raise AssertionError('TPMS geometry builder must not run')
    monkeypatch.setattr(pattern, '_build_tpms_unit', forbidden)
    for controls in ({'n_internal': 1}, {'n_pts_per_side': 1},
                     {'point_displacements': [(0., 0., 0.)]}):
        with pytest.raises(ValueError, match='TPMS units require'):
            fn.pattern_3d(unit=unit, grid=(1, 1, 1), **controls)


@pytest.mark.parametrize('unit', TPMS_UNITS)
def test_tpms_default_and_empty_controls_preserve_routing(monkeypatch, unit):
    calls = []
    def sentinel(kind, **kwargs):
        calls.append((kind, kwargs))
        graph = fn.StructureGraph(dimension=3, box_size=kwargs['box'])
        a = graph.add_node((0., 0., 0.))
        b = graph.add_node((10., 0., 0.))
        graph.add_edge(a, b, radius=kwargs['radius'], material=kwargs['material'])
        return graph
    monkeypatch.setattr(pattern, '_build_tpms_unit', sentinel)
    baseline = None
    for empty in (None, [], (), np.empty((0, 3))):
        graph = fn.pattern_3d(unit=unit, grid=(1, 1, 1), seed=23,
                              n_internal=0, n_pts_per_side=0,
                              point_displacements=empty,
                              unit_kwargs={'num_periods': (1, 1, 1), 'resolution': 8})
        assert (graph.num_nodes, graph.num_edges) == (2, 1)
        assert all(edge.internal_points is None for edge in graph.edges.values())
        assert graph.metadata['pattern']['n_pts_per_side'] == 0
        current = _snapshot(graph)
        if baseline is None:
            baseline = current
        else:
            assert current == baseline
    assert len(calls) == 4
    assert all(call == calls[0] for call in calls)
    assert calls[0][0] == unit
    assert calls[0][1]['resolution'] == 8
    assert not {'n_internal', 'n_pts_per_side', 'point_displacements'} & calls[0][1].keys()


@pytest.mark.parametrize('count,length,expected', ((0, 1, 0), (1, 1, 12), (1, 13, 12)))
def test_cubic_nonempty_wrong_cardinality_rejects(count, length, expected):
    with pytest.raises(ValueError, match=f'Expected {expected} point_displacements.*got {length}'):
        _cubic(n_pts_per_side=count, point_displacements=[(0., 0., 0.)] * length)


@pytest.mark.parametrize('count,length', ((0, 1), (2, 1), (1, 2)))
def test_helper_mismatch_precedes_graph_mutation_and_rng(monkeypatch, count, length):
    graph = fn.StructureGraph(dimension=3)
    a = graph.add_node((0., 0., 0.))
    b = graph.add_node((10., 0., 0.))
    graph.add_edge(a, b, radius=.1)
    graph.metadata['sentinel'] = {'keep': [1, 2]}
    before = copy.deepcopy(graph.to_dict())
    counters = (graph._next_node_id, graph._next_edge_id)
    def forbidden_rng(*args, **kwargs):
        raise AssertionError('invalid explicit list must not create an RNG')
    monkeypatch.setattr(pattern.np.random, 'default_rng', forbidden_rng)
    with pytest.raises(ValueError, match='point_displacements'):
        pattern._add_3d_edges(graph, [a, b], [(0, 1)], count,
                              np.zeros((length, 3)), .1, fn.Material(), 0, 23)
    assert graph.to_dict() == before
    assert (graph._next_node_id, graph._next_edge_id) == counters


@pytest.mark.parametrize('as_array', (False, True))
def test_exact_size_explicit_zeros_preserve_midpoints_without_rng(monkeypatch, as_array):
    def forbidden_rng(*args, **kwargs):
        raise AssertionError('explicit displacements must not create an RNG')
    monkeypatch.setattr(pattern.np.random, 'default_rng', forbidden_rng)
    displacements = np.zeros((12, 3)) if as_array else [(0., 0., 0.)] * 12
    graph = _cubic(n_pts_per_side=1, point_displacements=displacements)
    assert (graph.num_nodes, graph.num_edges) == (20, 24)
    assert all(edge.internal_points is None for edge in graph.edges.values())
    mids = [nid for nid, node in graph.nodes.items() if 5. in node.position]
    assert len(mids) == 12
    for nid in mids:
        point = graph.nodes[nid].position
        neighbors = graph.neighbors(nid)
        assert len(neighbors) == 2
        assert sum(value == 5. for value in point) == 1
        assert all(value in (0., 5., 10.) for value in point)
        np.testing.assert_array_equal(
            point, (graph.nodes[neighbors[0]].position + graph.nodes[neighbors[1]].position) / 2)


def test_none_and_empty_preserve_existing_seeded_auto_and_ids():
    # Mirror the pre-existing draw sequence, not a new scale or displacement model.
    rng = np.random.default_rng(23)
    explicit = [tuple(float(rng.uniform(-.3, .3)) for _ in range(3)) for _ in range(12)]
    expected = _snapshot(_cubic(n_pts_per_side=1, point_displacements=explicit))
    for empty in (None, [], (), np.empty((0, 3))):
        graph = _cubic(n_pts_per_side=1, point_displacements=empty)
        assert _snapshot(graph) == expected


def test_zero_count_none_and_empty_remain_noop():
    expected = None
    for empty in (None, [], (), np.empty((0, 3))):
        graph = _cubic(n_pts_per_side=0, point_displacements=empty)
        assert (graph.num_nodes, graph.num_edges) == (8, 12)
        current = _snapshot(graph)
        if expected is None:
            expected = current
        else:
            assert current == expected


def test_existing_custom_fit_guard_is_preserved():
    with pytest.raises(ValueError, match='fit_to_box=True require n_pts_per_side=0'):
        fn.pattern_2d(points=[(0., 0.), (1., 0.), (1., 1.)], grid=(1, 1),
                      fit_to_box=True, n_pts_per_side=1)


def test_edge_free_helper_explicit_empty_does_not_create_rng(monkeypatch):
    def forbidden_rng(*args, **kwargs):
        raise AssertionError('explicit empty vector for zero edges must not create an RNG')
    monkeypatch.setattr(pattern.np.random, 'default_rng', forbidden_rng)
    graph = fn.StructureGraph(dimension=3)
    before = copy.deepcopy(graph.to_dict())
    for empty in ([], (), np.empty((0, 3))):
        pattern._add_3d_edges(graph, [], [], 1, empty, .1, fn.Material(), 0, 23)
        assert graph.to_dict() == before


@pytest.mark.parametrize('count,length', ((0, 1), (1, 12), (1, 19), (1, 21), (2, 39)))
def test_octet_total_mismatch_precedes_graph_construction(monkeypatch, count, length):
    def forbidden_graph(*args, **kwargs):
        raise AssertionError('octet must validate total vector before graph construction')
    monkeypatch.setattr(pattern, 'StructureGraph', forbidden_graph)
    with pytest.raises(ValueError, match=f'Expected {20 * count} point_displacements.*got {length}'):
        fn.pattern_3d(unit='octet', grid=(1, 1, 1), n_pts_per_side=count,
                      point_displacements=[(0., 0., 0.)] * length)


@pytest.mark.parametrize('as_array', (False, True))
def test_octet_exact_total_places_center_tail_and_preserves_ids_radii(monkeypatch, as_array):
    def forbidden_rng(*args, **kwargs):
        raise AssertionError('explicit octet vector must not create an RNG')
    monkeypatch.setattr(pattern.np.random, 'default_rng', forbidden_rng)
    displacements = [(0., 0., 0.)] * 12 + [(0., 0., .25)] * 8
    if as_array:
        displacements = np.asarray(displacements)
    graph = fn.pattern_3d(unit='octet', box=(10., 10., 10.), grid=(1, 1, 1),
                          radius=.1, n_pts_per_side=1, point_displacements=displacements)
    assert (graph.num_nodes, graph.num_edges) == (29, 40)
    assert list(graph.nodes) == list(range(29))
    assert list(graph.edges) == list(range(40))
    np.testing.assert_array_equal(graph.nodes[20].position, [5., 5., 5.])
    assert all(graph.edges[eid].radius == .1 for eid in range(24))
    assert all(graph.edges[eid].radius == .1 * .7 for eid in range(24, 40))
    for corner in range(8):
        expected = (graph.nodes[corner].position + graph.nodes[20].position) / 2
        expected = expected + [0., 0., .25]
        np.testing.assert_array_equal(graph.nodes[21 + corner].position, expected)
        assert graph.edges[24 + 2 * corner].node_i == corner
        assert graph.edges[24 + 2 * corner].node_j == 21 + corner
        assert graph.edges[25 + 2 * corner].node_i == 21 + corner
        assert graph.edges[25 + 2 * corner].node_j == 20


def test_octet_none_and_empty_keep_two_seeded_sequences():
    def legacy_draws(count):
        rng = np.random.default_rng(23)
        return [tuple(float(rng.uniform(-.3, .3)) for _ in range(3)) for _ in range(count)]
    # Existing outer/center helpers each restart the supplied seed; do not unify them.
    explicit = legacy_draws(12) + legacy_draws(8)
    def generate(displacements):
        return fn.pattern_3d(unit='octet', box=(10., 10., 10.), grid=(1, 1, 1),
                             radius=.1, n_pts_per_side=1, seed=23,
                             point_displacements=displacements)
    expected = _snapshot(generate(explicit))
    for empty in (None, [], (), np.empty((0, 3))):
        graph = generate(empty)
        assert _snapshot(graph) == expected
        np.testing.assert_array_equal(graph.nodes[20].position, [5., 5., 5.])
