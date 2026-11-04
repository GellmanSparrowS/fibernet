"""Legacy edge-incidence projection contracts, not lossless graph interchange."""
import copy
from collections import Counter
from itertools import combinations

import numpy as np
import pytest

nx = pytest.importorskip("networkx")

from fibernet import Material, StructureGraph
from fibernet.graph.io import from_networkx, load_graph_json, save_graph_json, to_networkx


def _position(value):
    value = tuple(value)
    return value + (0.0,) if len(value) == 2 else value


def _snapshot(graph):
    return copy.deepcopy((graph.graph, list(graph.nodes(data=True)),
                          list(graph.edges(data=True))))


def _graph(edges, nodes=None, cls=nx.Graph):
    graph = cls()
    if nodes is not None:
        graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    for i, node in enumerate(graph):
        graph.nodes[node]["pos"] = (float(i), float(i % 2), 0.0)
    return graph


def _assert_incidence(graph, network, pos_attr="pos"):
    """Verify all geometric endpoint associations and every source incidence."""
    edges = list(graph.edges(data=True))
    assert network.num_fibers == len(edges)
    expected = []
    for node in graph:
        incidences = []
        for index, (u, v, _) in enumerate(edges):
            if node == u:
                incidences.append((index, 0.0))
            if node == v:
                incidences.append((index, 1.0))
        expected.extend((a[0], b[0], a[1], b[1], _position(graph.nodes[node][pos_attr]))
                        for a, b in combinations(incidences, 2))
    actual = []
    for link in network.crosslinks:
        position = tuple(link.position)
        actual.append((link.fiber_i, link.fiber_j, link.param_i, link.param_j, position))
        for index, parameter in ((link.fiber_i, link.param_i), (link.fiber_j, link.param_j)):
            assert parameter in (0.0, 1.0)
            np.testing.assert_array_equal(network.fibers[index].centerline[int(parameter)], position)
        assert link.crosslink_type == "welded"
        assert np.isinf(link.strength) and np.isinf(link.stiffness)
    assert Counter(actual) == Counter(expected)
    for fiber, (u, v, _) in zip(network.fibers, edges):
        np.testing.assert_array_equal(fiber.centerline,
                                      [_position(graph.nodes[u][pos_attr]),
                                       _position(graph.nodes[v][pos_attr])])
        assert fiber.centerline.shape == (2, 3)
    nonisolated_components = sum(graph.subgraph(nodes).number_of_edges() > 0
                                 for nodes in nx.connected_components(graph.to_undirected()))
    assert nx.number_connected_components(nx.from_numpy_array(network.connectivity_matrix())) == nonisolated_components


def test_three_node_path_keeps_degree_two_junction():
    graph = _graph([(0, 1), (1, 2)])
    network = from_networkx(graph)
    _assert_incidence(graph, network)
    assert (network.num_fibers, network.num_crosslinks) == (2, 1)
    assert network.validate()["valid"]


@pytest.mark.parametrize("node_order,edges", [
    ([0, 1, 2, 3], [(0, 1), (0, 2), (0, 3)]),
    ([3, 2, 1, 0], [(3, 0), (2, 0), (1, 0)]),
    ([1, 0, 3, 2], [(0, 3), (2, 0), (1, 0)]),
    ([2, 3, 0, 1], [(1, 0), (0, 2), (3, 0)]),
])
def test_star_keeps_all_pairs_independent_of_insertion(node_order, edges):
    graph = _graph(edges, nodes=node_order)
    network = from_networkx(graph)
    _assert_incidence(graph, network)
    assert (network.num_fibers, network.num_crosslinks) == (3, 3)
    assert len({(link.fiber_i, link.fiber_j) for link in network.crosslinks}) == 3
    assert network.validate()["valid"]


@pytest.mark.parametrize("edges,counts", [
    ([(0, 1), (1, 2), (2, 0)], (3, 3)),
    ([(0, 1), (1, 2), (3, 4), (4, 5)], (4, 2)),
    ([(0, 1), (1, 2), (1, 3), (3, 4)], (4, 4)),
])
def test_cycles_components_and_mixed_degrees(edges, counts):
    graph = _graph(edges)
    network = from_networkx(graph)
    _assert_incidence(graph, network)
    assert (network.num_fibers, network.num_crosslinks) == counts


def test_unorderable_node_labels_do_not_require_sorting():
    graph = _graph([("left", 7), (7, ("right", 2))])
    _assert_incidence(graph, from_networkx(graph))


def test_coincident_distinct_nodes_are_not_welded():
    graph = _graph([(0, 1), (1, 2), (3, 4), (4, 5)])
    for first, second in ((0, 3), (1, 4), (2, 5)):
        graph.nodes[second]["pos"] = graph.nodes[first]["pos"]
    network = from_networkx(graph)
    _assert_incidence(graph, network)
    assert network.num_crosslinks == 2
    assert nx.number_connected_components(nx.from_numpy_array(network.connectivity_matrix())) == 2


def test_parallel_edges_keep_separate_fibers_and_edge_attributes():
    graph = _graph([], nodes=[0, 1, 2], cls=nx.MultiGraph)
    graph.add_edge(0, 1, key="first", radius=0.2, youngs_modulus=12.0)
    graph.add_edge(1, 0, key="second", radius=0.3, youngs_modulus=23.0)
    graph.add_edge(1, 2, key="third", radius=0.4, youngs_modulus=34.0)
    network = from_networkx(graph)
    _assert_incidence(graph, network)
    assert (network.num_fibers, network.num_crosslinks) == (3, 4)
    assert [(fiber.radius, fiber.material.youngs_modulus) for fiber in network.fibers] == [
        (0.2, 12.0), (0.3, 23.0), (0.4, 34.0)]


def test_two_dimensional_custom_position_key_and_defaults():
    graph = nx.Graph()
    graph.add_node("left", xy=(0.0, 0.0))
    graph.add_node("mid", xy=(1.0, 1.0))
    graph.add_node("right", xy=(2.0, 0.0))
    graph.add_edges_from([("left", "mid"), ("mid", "right")])
    material = Material(name="default", youngs_modulus=123., poissons_ratio=0.2, density=456.)
    network = from_networkx(graph, material, "xy")
    _assert_incidence(graph, network, "xy")
    assert network.dimension == 3  # Unchanged legacy container default.
    for fiber in network.fibers:
        assert fiber.radius == 0.1 and fiber.segments == 20 and fiber.fiber_id == 0
        assert fiber.material.name == "default"
        assert fiber.material.youngs_modulus == 123.
        assert fiber.material.poissons_ratio == 0.2 and fiber.material.density == 456.
        assert fiber.material is not material


def test_input_and_crosslink_positions_are_independently_owned():
    graph = _graph([(0, 1), (0, 2), (0, 3)])
    before = _snapshot(graph)
    network = from_networkx(graph)
    expected = network.crosslinks[1].position.copy()
    network.crosslinks[0].position[:] = 100
    np.testing.assert_array_equal(network.crosslinks[1].position, expected)
    assert _snapshot(graph) == before
    assert not np.shares_memory(network.crosslinks[1].position, network.fibers[0].centerline)


@pytest.mark.parametrize("kind", ["empty", "isolated", "edge_and_isolate"])
def test_isolated_nodes_remain_an_explicit_legacy_projection_limit(kind):
    graph = _graph([(0, 1)] if kind == "edge_and_isolate" else [])
    if kind != "empty":
        graph.add_node("isolated", pos=(9., 9., 9.))
    network = from_networkx(graph)
    _assert_incidence(graph, network)
    assert network.num_crosslinks == 0
    assert network.num_fibers == (1 if kind == "edge_and_isolate" else 0)
    assert "Isolated nodes" in from_networkx.__doc__


def test_curve_and_rich_metadata_are_not_newly_promised_as_preserved():
    graph = _graph([(0, 1)])
    graph.graph["units"] = "mm"
    graph.nodes[0]["boundary"] = (True,) * 6
    graph.edges[0, 1].update(internal_points=[[0.5, 9., 0.]],
                             material={"youngs_modulus": 999.}, metadata={"tag": [1]})
    material = Material(name="limited", youngs_modulus=111., shear_modulus=77., extra={"tag": [2]})
    before = _snapshot(graph)
    network = from_networkx(graph, material=material)
    _assert_incidence(graph, network)
    assert network.metadata == {}
    assert network.fibers[0].material.youngs_modulus == 111.
    assert network.fibers[0].material.shear_modulus != 77.
    assert network.fibers[0].material.extra == {}
    assert _snapshot(graph) == before
    assert "lossy projection" in from_networkx.__doc__
    assert "to_networkx_lossless" in from_networkx.__doc__


def test_direction_remains_projected_to_undirected_incidence():
    graph = _graph([(0, 1), (2, 1)], cls=nx.DiGraph)
    network = from_networkx(graph)
    _assert_incidence(graph, network)
    assert network.num_crosslinks == 1
    np.testing.assert_array_equal(network.connectivity_matrix(), [[0, 1], [1, 0]])


def test_missing_position_keeps_explicit_error_including_isolates():
    graph = _graph([(0, 1)])
    graph.add_node("missing")
    with pytest.raises(ValueError, match="missing 'pos' attribute"):
        from_networkx(graph)


@pytest.mark.parametrize("self_loop", [False, True])
def test_degenerate_edge_remains_a_separate_legacy_validation_limit(self_loop):
    graph = nx.Graph()
    graph.add_node(0, pos=(0., 0., 0.))
    graph.add_node(1, pos=(0., 0., 0.))
    graph.add_edge(0, 0 if self_loop else 1)
    network = from_networkx(graph)
    assert network.num_fibers == 1 and network.fibers[0].length == 0.0
    assert not network.validate()["valid"]
    assert any("zero length" in error for error in network.validate()["errors"])


def test_json_loader_inherits_shared_junction_fix(tmp_path):
    graph = _graph([(0, 1), (1, 2), (1, 3)])
    path = tmp_path / "junction.json"
    save_graph_json(graph, path)
    network = load_graph_json(path)
    _assert_incidence(graph, network)
    assert network.num_crosslinks == 3


def test_exporter_is_still_crosslink_only_not_a_source_graph_roundtrip():
    graph = _graph([(0, 1), (1, 2)])
    exported = to_networkx(from_networkx(graph))
    assert (exported.number_of_nodes(), exported.number_of_edges()) == (1, 0)
    assert exported.number_of_nodes() != graph.number_of_nodes()


def test_legacy_incidence_matches_canonical_structure_graph_bridge():
    graph = _graph([(0, 1), (1, 2), (1, 3), (3, 4)])
    native = StructureGraph.from_networkx(graph, dimension=3)
    canonical = native.to_fiber_network()
    legacy = from_networkx(graph)
    signature = lambda network: [(link.fiber_i, link.fiber_j, link.param_i, link.param_j,
                                  tuple(link.position)) for link in network.crosslinks]
    assert signature(legacy) == signature(canonical)
    assert all(fiber.num_points == 2 for fiber in legacy.fibers)
    assert all(fiber.num_points >= 5 for fiber in canonical.fibers)


def test_native_lossless_route_preserves_isolates_curves_parallel_members_and_materials():
    graph = StructureGraph(dimension=3)
    deleted = graph.add_node([8., 8., 8.])
    graph.remove_node(deleted)
    a = graph.add_node([0., 0., 0.])
    b = graph.add_node([1., 0., 0.])
    graph.add_node([5., 5., 5.])
    graph.add_node([0., 0., 0.], merge=False)
    graph.metadata["units"] = {"length": "mm"}
    material = Material(name="full", youngs_modulus=123., shear_modulus=77., extra={"tag": [1]})
    graph.add_edge(a, b, material=material, internal_points=np.array([[0.5, 1., 0.]]))
    graph.add_edge(b, a, radius=0.3, material=Material(name="other"), allow_parallel=True)
    before = copy.deepcopy(graph.to_dict())
    restored = StructureGraph.from_networkx_lossless(graph.to_networkx_lossless())
    assert restored.to_dict() == before == graph.to_dict()
    assert len(restored.nodes) == 4 and len(restored.edges) == 2
