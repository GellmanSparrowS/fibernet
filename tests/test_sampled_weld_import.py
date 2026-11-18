"""Analytical and generated-network checks for the sampled weld import."""

from copy import deepcopy
import math

import networkx as nx
import numpy as np
import pytest

from fibernet import Crosslink, Fiber, FiberNetwork, Material, ReducedBeamSolver, StructureGraph
from fibernet.analysis.geometry_paths import analyze_geometry_paths
from fibernet.analysis.topology_metrics import analyze_topology
from fibernet.core.fiber import CrossSection
from fibernet.core.sampled_weld_import import convert_sampled_welded_fibers
from fibernet.gen import ZigZagGenerator


def check_sample_mapping(network, graph, report):
    for i, fiber in enumerate(network.fibers):
        entry = report['fibers'][i]
        nodes, edges = entry['sample_node_ids'], entry['segment_edge_ids']
        assert entry['source_fiber_id'] == fiber.fiber_id
        assert len(nodes) == len(fiber.centerline)
        assert len(edges) == len(fiber.centerline) - 1
        np.testing.assert_array_equal([graph.nodes[node].position for node in nodes],
                                      fiber.centerline)
        for edge_id, start, end in zip(edges, nodes, nodes[1:]):
            edge = graph.edges[edge_id]
            assert (edge.node_i, edge.node_j) == (start, end)
            assert edge.internal_points is None
            assert edge.radius == fiber.radius
            assert edge.material.to_dict() == fiber.material.to_dict()
            assert edge.material is not fiber.material


def test_curve_shape_length_and_recorded_topology(tmp_path):
    material = Material(name='curve', extra={'tags': ['test']})
    network = FiberNetwork(
        fibers=[
            Fiber([[0, 0, 0], [2, 0, 0], [2, 1, 0], [4, 1, 0]], radius=.12,
                  material=material, fiber_id=42),
            Fiber([[1, -1, 0], [2, 1, 0], [2, 3, 0]], radius=.2,
                  material=material, fiber_id=42),
        ],
        # The two curves also cross at (1.5, 0, 0) without a weld record.
        crosslinks=[Crosslink(0, 1, 2/3, .5, np.array([2., 1., 0.]))],
        dimension=2, box_size=np.array([4., 5.]),
        metadata={'tags': ['source'], 'weld_triplets': [[42, 84, 126]],
                  'custom': {'selection': {42}}},
    )
    original = deepcopy(network.to_dict())
    graph, report = convert_sampled_welded_fibers(network)
    check_sample_mapping(network, graph, report)
    topo = analyze_topology(graph)
    assert (topo.n_nodes, topo.n_physical_edges, topo.n_components) == (6, 5, 1)
    assert topo.cycle_rank_physical == 0
    assert topo.degree_physical_by_node[report['crosslink_junction_node_ids'][0]] == 4
    start = report['fibers'][0]['sample_node_ids'][0]
    far_end = report['fibers'][1]['sample_node_ids'][-1]
    geometry = analyze_geometry_paths(graph, length_unit='m', coordinate_frame='declared',
                                      node_pairs=[(start, far_end)])
    np.testing.assert_allclose(geometry.polyline_length_by_edge, [2, 1, 2, math.sqrt(5), 2])
    assert geometry.pair_distance == (5.,)
    assert graph.metadata == {'sampled_weld_import': 'fibernet.sampled_weld_import/1'}
    assert report['source_metadata'] == original['metadata']
    graph_file = tmp_path / 'graph.json'
    graph.save_json(graph_file)
    restored = StructureGraph.load_json(graph_file)
    assert restored.to_dict() == graph.to_dict()
    # The engine looks up weld_triplets as node IDs when that key is present.
    assert ReducedBeamSolver(restored).node_ids == sorted(restored.nodes)

    report['source_metadata']['tags'].append('report')
    report['source_metadata']['weld_triplets'][0][0] = 21
    report['source_metadata']['custom']['selection'].add(21)
    graph.edges[0].material.extra['tags'].append('edit')
    assert graph.edges[1].material.extra['tags'] == ['test']
    report['fibers'][0]['sample_node_ids'].clear()
    assert network.to_dict() == original
    assert (graph.num_nodes, graph.num_edges) == (6, 5)


def test_canonical_sample_knot_and_nearby_parameter_rejection():
    parameter = 15/22
    assert parameter * 22 != 15
    first = Fiber([[i, i % 2, i % 3] for i in range(23)])
    second = Fiber([first.centerline[15], [20, 10, 2]])
    link = Crosslink(0, 1, parameter, 0., first.centerline[15].copy())
    network = FiberNetwork([first, second], [link])
    graph, report = convert_sampled_welded_fibers(network)
    assert (graph.num_nodes, graph.num_edges) == (24, 23)
    assert report['crosslink_junction_node_ids'] == [report['fibers'][0]['sample_node_ids'][15]]
    link.param_i = math.nextafter(parameter, 1.)
    with pytest.raises(ValueError, match='canonical sample-knot'):
        convert_sampled_welded_fibers(network)
    link.param_i = parameter
    link.position[0] = math.nextafter(15., 16.)
    with pytest.raises(ValueError, match='stored position exactly'):
        convert_sampled_welded_fibers(network)


def test_parallel_edges_and_unwelded_coincident_occurrences():
    a, b, c = [0., 0., 0.], [1., 0., 0.], [0., 1., 0.]
    network = FiberNetwork(
        fibers=[Fiber([a, b]), Fiber([b, a]), Fiber([a, c])],
        crosslinks=[Crosslink(0, 1, 0., 1., np.array(a)),
                    Crosslink(0, 1, 1., 0., np.array(b)),
                    Crosslink(1, 0, 1., 0., np.array(a))],
    )
    graph, report = convert_sampled_welded_fibers(network)
    check_sample_mapping(network, graph, report)
    assert report['fibers'][0]['sample_node_ids'][0] != report['fibers'][2]['sample_node_ids'][0]
    first, second, duplicate = report['crosslink_junction_node_ids']
    assert first == duplicate and first != second
    topo = analyze_topology(graph)
    assert (topo.n_nodes, topo.n_physical_edges, topo.n_components) == (4, 3, 2)
    assert topo.n_unique_connections == 2


@pytest.mark.parametrize('record, field, value, message', [
    ('network', 'periodic', True, 'periodic'),
    ('fiber', 'cross_section', CrossSection.RECTANGULAR, 'circular'),
    ('fiber', 'centerline', np.array([[0., 0., 0.], [0., 0., 0.]]), 'distinct'),
    ('fiber', 'centerline', np.array([[0., 0., 0.], [1., np.nan, 0.]]), 'finite'),
    ('link', 'strength', 10., 'positive infinity'),
    ('link', 'fiber_i', 1, 'fiber list indices'),
])
def test_incompatible_records_are_rejected(record, field, value, message):
    first = Fiber([[0, 0, 0], [1, 0, 0]])
    second = Fiber([[1, 0, 0], [1, 1, 0]])
    link = Crosslink(0, 1, 1., 0., np.array([1., 0., 0.]))
    network = FiberNetwork([first, second], [link], dimension=2)
    setattr({'network': network, 'fiber': first, 'link': link}[record], field, value)
    with pytest.raises(ValueError, match=message):
        convert_sampled_welded_fibers(network)


def test_generated_zigzag_json_geometry_and_topology(tmp_path):
    generator = ZigZagGenerator(n_cols=3, n_rows=3)
    reference = generator.generate()
    network = generator.to_fiber_network()
    assert (network.num_fibers, network.num_crosslinks) == (45, 90)
    legacy_file = tmp_path / 'zigzag.json'
    network.save_json(legacy_file)
    network = FiberNetwork.load_json(legacy_file)
    graph, report = convert_sampled_welded_fibers(network)
    check_sample_mapping(network, graph, report)
    topo = analyze_topology(graph)
    assert (topo.n_nodes, topo.n_physical_edges) == (19 * 45 + len(reference), 20 * 45)
    assert topo.n_components == nx.number_connected_components(reference)
    assert topo.cycle_rank_physical == reference.number_of_edges() - len(reference) + topo.n_components
    for node, crosslink in zip(report['crosslink_junction_node_ids'], network.crosslinks):
        np.testing.assert_array_equal(graph.nodes[node].position, crosslink.position)
    geometry = analyze_geometry_paths(graph, length_unit='m', coordinate_frame='declared')
    expected = math.fsum(math.dist(reference.nodes[a]['pos'], reference.nodes[b]['pos'])
                         for a, b in reference.edges())
    assert math.fsum(geometry.polyline_length_by_edge) == pytest.approx(expected, rel=2e-15)
