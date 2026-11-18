"""Component bridges locate their welded attachments on the source samples."""

from copy import deepcopy

import numpy as np

from fibernet import Crosslink, Fiber, FiberNetwork, Material
from fibernet.analysis.topology_metrics import analyze_topology
from fibernet.core.sampled_weld_import import convert_sampled_welded_fibers
from fibernet.gen import ZigZagGenerator


def assert_weld_positions(network, links):
    for link in links:
        assert link.crosslink_type == 'welded'
        for index, parameter in ((link.fiber_i, link.param_i), (link.fiber_j, link.param_j)):
            points = network.fibers[index].centerline
            sample_parameters = np.linspace(0., 1., len(points))
            interpolated = [np.interp(parameter, sample_parameters, axis)
                            for axis in points.T]
            np.testing.assert_allclose(interpolated, link.position, rtol=0., atol=1e-13)
            matches = np.flatnonzero(np.all(points == link.position, axis=1))
            assert len(matches) == 1
            assert parameter == int(matches[0]) / max(1, len(points) - 1)


def test_component_bridge_welds_reach_endpoint_and_non_midpoint_sample(tmp_path):
    network = FiberNetwork(
        fibers=[Fiber([(i, 0., 0.) for i in range(23)], radius=.25),
                Fiber([(15., 1., 0.), (15., 3., 0.)], radius=.5, fiber_id=1)],
        dimension=2,
    )
    original = deepcopy(network.to_dict())
    assert network.connect_components(max_gap=.99) == 0
    assert network.to_dict() == original
    assert network.connect_components(max_gap=1.) == 1
    assert network.to_dict()['fibers'][:2] == original['fibers']
    assert_weld_positions(network, network.crosslinks)
    first, second = network.crosslinks
    assert (first.fiber_i, first.param_i, first.fiber_j, first.param_j) == (0, 15/22, 2, 0.)
    assert (second.fiber_i, second.param_i, second.fiber_j, second.param_j) == (1, 0., 2, 1.)

    path = tmp_path / 'repaired.json'
    network.save_json(path)
    restored = FiberNetwork.load_json(path)
    assert restored.to_dict() == network.to_dict()
    graph, mapping = convert_sampled_welded_fibers(restored)
    topo = analyze_topology(graph)
    assert (topo.n_nodes, topo.n_physical_edges, topo.n_components) == (28, 27, 1)
    for link, node in zip(restored.crosslinks, mapping['crosslink_junction_node_ids']):
        np.testing.assert_array_equal(graph.nodes[node].position, link.position)


def test_component_repair_preserves_real_welds_in_multiple_curved_islands():
    network = FiberNetwork(dimension=2)
    for island in range(6):
        material = Material(name=f'zigzag-{island}')
        local = ZigZagGenerator.simple_zigzag(amplitude=2., wavelength=4., n_periods=3)
        local = local.to_fiber_network(radius=.15 + island * .01, material=material)
        shift = np.array([island * 15., (island % 3) * .4, 0.])
        first_index = network.num_fibers
        for fiber in local.fibers:
            fiber.centerline += shift
            network.add_fiber(fiber)
        for link in local.crosslinks:
            link.fiber_i += first_index
            link.fiber_j += first_index
            link.position += shift
            network.add_crosslink(link)
    initial_graph, _ = convert_sampled_welded_fibers(network)
    assert analyze_topology(initial_graph).n_components == 6
    initial = deepcopy(network.to_dict())
    assert network.connect_components(max_gap=4.) == 5
    assert (network.num_fibers, network.num_crosslinks) == (41, 40)
    records = network.to_dict()
    assert records['fibers'][:36] == initial['fibers']
    assert records['crosslinks'][:30] == initial['crosslinks']
    assert_weld_positions(network, network.crosslinks)
    repaired_graph, mapping = convert_sampled_welded_fibers(network)
    topo = analyze_topology(repaired_graph)
    assert (topo.n_nodes, topo.n_physical_edges, topo.n_components) == (741, 740, 1)
    for bridge in range(36, 41):
        offset = 30 + 2 * (bridge - 36)
        first, second = network.crosslinks[offset:offset + 2]
        assert first.fiber_j == second.fiber_j == bridge
        assert (first.param_j, second.param_j) == (0., 1.)
        fiber = network.fibers[bridge]
        assert fiber.material is network.fibers[0].material
        assert fiber.radius == network.fibers[first.fiber_i].radius
        assert len(fiber.centerline) == 5
        assert len(mapping['fibers'][bridge]['segment_edge_ids']) == 4


def test_bridge_can_still_attach_to_single_sample_fiber():
    network = FiberNetwork(
        fibers=[Fiber([(0., 0., 0.)]), Fiber([(0., 1., 0.), (0., 2., 0.)])],
        dimension=2,
    )
    assert network.connect_components(max_gap=1.) == 1
    assert_weld_positions(network, network.crosslinks)


def test_sampled_large_components_keep_serializable_fiber_indices(tmp_path):
    network = FiberNetwork(dimension=2)
    for layer in (0., 1.):
        for x in range(60):
            start = np.array([x, layer, 0.])
            end = np.array([x + 1, layer, 0.])
            index = network.add_fiber(Fiber([start, end]))
            if x:
                network.add_crosslink(Crosslink(index - 1, index, 1., 0., start))
    assert (network.num_fibers, network.num_crosslinks) == (120, 118)

    state = np.random.get_state()
    try:
        np.random.seed(12)
        assert network.connect_components(max_gap=2.) == 1
    finally:
        np.random.set_state(state)
    assert (network.num_fibers, network.num_crosslinks) == (121, 120)
    for link in network.crosslinks[-2:]:
        assert type(link.fiber_i) is int
        assert link.fiber_j == 120
    assert_weld_positions(network, network.crosslinks)

    path = tmp_path / 'large-repair.json'
    network.save_json(path)
    restored = FiberNetwork.load_json(path)
    graph, _ = convert_sampled_welded_fibers(restored)
    assert restored.to_dict() == network.to_dict()
    topo = analyze_topology(graph)
    assert (topo.n_nodes, topo.n_physical_edges, topo.n_components) == (125, 124, 1)
