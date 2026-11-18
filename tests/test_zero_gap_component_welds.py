"""Exact component coincidences become direct welds without artificial fibers."""

from copy import deepcopy
import math

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from fibernet import Crosslink, Fiber, FiberNetwork, Material, StructureGraph
from fibernet.analysis.topology_metrics import analyze_topology
from fibernet.core.sampled_weld_import import convert_sampled_welded_fibers


@pytest.mark.parametrize(
    'first, second, parameters, position',
    [
        ([[0., 0., 0.], [1., 0., 0.]], [[1., 0., 0.], [1., 1., 0.]],
         (1., 0.), [1., 0., 0.]),
        ([[-4., 0., 0.], [-2., 0., 0.], [0., 0., 0.], [2., 0., 0.]],
         [[0., -2., 0.], [0., 0., 0.], [0., 2., 0.]], (2/3, .5), [0., 0., 0.]),
        ([[0., 1., 0.], [1., -0., 0.]], [[2., 1., 0.], [1., 0., 0.]],
         (1., 1.), [1., 0., 0.]),
    ],
    ids=['endpoint', 'unequal-interior-knots', 'signed-zero-endpoints'],
)
def test_direct_weld_preserves_geometry_and_correct_attachment(first, second, parameters, position):
    fibers = [
        Fiber(first, radius=.2, material=Material(name='first'), fiber_id=17),
        Fiber(second, radius=.4, material=Material(name='second'), fiber_id=-5),
    ]
    network = FiberNetwork(fibers=list(fibers), dimension=2)
    originals = deepcopy(network.to_dict()['fibers'])
    length, volume = network.total_length, network.total_volume

    assert network.connect_components(max_gap=-1.) == 0
    assert network.num_crosslinks == 0
    assert network.connect_components(max_gap=0.) == 1
    assert network.to_dict()['fibers'] == originals
    assert all(a is b for a, b in zip(fibers, network.fibers))
    assert (network.total_length, network.total_volume) == (length, volume)
    assert (network.num_fibers, network.num_crosslinks) == (2, 1)
    link = network.crosslinks[0]
    assert (link.fiber_i, link.fiber_j) == (0, 1)
    assert (link.param_i, link.param_j) == parameters
    assert link.crosslink_type == 'welded'
    assert link.strength == link.stiffness == math.inf
    np.testing.assert_array_equal(link.position, position)
    assert not np.shares_memory(link.position, fibers[0].centerline)
    assert not np.shares_memory(link.position, fibers[1].centerline)
    assert network.connect_components(max_gap=0.) == 0

    graph, report = convert_sampled_welded_fibers(network)
    assert analyze_topology(graph).n_components == 1
    np.testing.assert_array_equal(graph.nodes[report['crosslink_junction_node_ids'][0]].position,
                                  position)
    for f, p in zip(range(2), parameters):
        sample = round(p * (len(fibers[f].centerline) - 1))
        assert report['fibers'][f]['sample_node_ids'][sample] == report['crosslink_junction_node_ids'][0]


@pytest.mark.parametrize(
    'left_x, right_x, max_gap, underflowed',
    [(1., math.nextafter(1., 2.), 1., False), (0., 1e-200, 1e-199, True)],
    ids=['one-ulp-gap', 'underflowed-distance'],
)
def test_distinct_coordinates_keep_the_positive_gap_path(left_x, right_x, max_gap, underflowed):
    network = FiberNetwork([
        Fiber([[-1., 0., 0.], [left_x, 0., 0.]]),
        Fiber([[right_x, 0., 0.], [2., 0., 0.]], fiber_id=1),
    ], dimension=2)
    first, second = network.fibers[0].end_point, network.fibers[1].start_point
    assert not np.array_equal(first, second)
    if underflowed:
        assert cdist([first], [second])[0, 0] == 0.
    assert network.connect_components(max_gap=max_gap) == 1
    assert (network.num_fibers, network.num_crosslinks) == (3, 2)
    np.testing.assert_array_equal(network.fibers[2].centerline, np.linspace(first, second, 5))
    for index, link in enumerate(network.crosslinks):
        assert (link.fiber_i, link.fiber_j) == (index, 2)
        assert (link.param_i, link.param_j) == (1. - index, float(index))
    if underflowed:
        graph, _ = convert_sampled_welded_fibers(network)
        assert (graph.num_nodes, graph.num_edges) == (7, 6)


def test_mixed_component_joins_keep_indices_through_json_and_graph_import(tmp_path):
    network = FiberNetwork(dimension=2)
    for points in ([[0., 0., 0.], [2., 0., 0.]],
                   [[2., 0., 0.], [2., 1., 0.]],
                   [[2., 2., 0.], [2., 3., 0.]],
                   [[2., 3., 0.], [3., 3., 0.]]):
        network.add_fiber(Fiber(points))
    assert network.connect_components(max_gap=1.) == 3
    assert (network.num_fibers, network.num_crosslinks) == (5, 4)
    assert [(link.fiber_i, link.fiber_j, link.param_i, link.param_j)
            for link in network.crosslinks] == [
        (0, 1, 1., 0.), (1, 4, 1., 0.), (2, 4, 0., 1.), (2, 3, 1., 0.),
    ]
    source_path = tmp_path / 'mixed.json'
    network.save_json(source_path)
    restored = FiberNetwork.load_json(source_path)
    assert restored.to_dict() == network.to_dict()
    graph, report = convert_sampled_welded_fibers(restored)
    topo = analyze_topology(graph)
    assert (topo.n_nodes, topo.n_physical_edges, topo.n_components) == (9, 8, 1)
    for link, node in zip(restored.crosslinks, report['crosslink_junction_node_ids']):
        np.testing.assert_array_equal(graph.nodes[node].position, link.position)
    graph_path = tmp_path / 'mixed-native.json'
    graph.save_json(graph_path)
    assert StructureGraph.load_json(graph_path).to_dict() == graph.to_dict()


def test_large_sampled_component_direct_weld_retains_json_integer_indices(tmp_path):
    network = FiberNetwork(dimension=2)
    for _ in range(2):
        for x in range(60):
            start = np.array([x, 0., 0.])
            index = network.add_fiber(Fiber([start, [x + 1, 0., 0.]]))
            if x:
                network.add_crosslink(Crosslink(index - 1, index, 1., 0., start))
    state = np.random.get_state()
    try:
        np.random.seed(12)
        assert network.connect_components(max_gap=0.) == 1
    finally:
        np.random.set_state(state)
    assert (network.num_fibers, network.num_crosslinks) == (120, 119)
    link = network.crosslinks[-1]
    assert type(link.fiber_i) is type(link.fiber_j) is int
    assert link.fiber_i < 60 <= link.fiber_j
    source_path = tmp_path / 'large.json'
    network.save_json(source_path)
    restored = FiberNetwork.load_json(source_path)
    graph, report = convert_sampled_welded_fibers(restored)
    topo = analyze_topology(graph)
    assert (topo.n_nodes, topo.n_physical_edges, topo.n_components) == (121, 120, 1)
    np.testing.assert_array_equal(graph.nodes[report['crosslink_junction_node_ids'][-1]].position,
                                  link.position)


def test_direct_weld_leaves_degenerate_source_geometry_for_import_validation():
    network = FiberNetwork([
        Fiber([[0., 0., 0.], [0., 0., 0.]]), Fiber([[0., 0., 0.], [1., 0., 0.]]),
    ], dimension=2)
    before = deepcopy(network.to_dict()['fibers'])
    assert network.connect_components(max_gap=0.) == 1
    assert (network.num_fibers, network.num_crosslinks) == (2, 1)
    assert network.to_dict()['fibers'] == before
    with pytest.raises(ValueError, match='consecutive centerline samples must be distinct'):
        convert_sampled_welded_fibers(network)
