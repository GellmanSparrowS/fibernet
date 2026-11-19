"""Merge and tile add proximity welds only between their distinct inputs."""

from copy import deepcopy

import numpy as np

from fibernet import Crosslink, Fiber, FiberNetwork
from fibernet.api import export, load, tile
from fibernet.analysis.topology_metrics import analyze_topology
from fibernet.core.sampled_weld_import import convert_sampled_welded_fibers
from fibernet.core.transform import merge
from fibernet.gen import ZigZagGenerator


def test_far_apart_zigzag_tiles_keep_one_copy_of_each_original_weld_after_json(tmp_path):
    source = ZigZagGenerator(n_cols=3, n_rows=3).to_fiber_network()
    assert (source.num_fibers, source.num_crosslinks) == (45, 90)
    original = deepcopy(source.to_dict())
    saved = tmp_path / 'separated-tiles.json'
    tiled = tile(source, repeats=(2, 1, 1), spacing=[1000., 10., 1.])
    export(tiled, saved)
    restored = load(saved)

    assert (restored.num_fibers, restored.num_crosslinks) == (90, 180)
    assert restored.periodic
    assert restored.validate()['stats']['num_components'] == 2
    expected_links = []
    for cell_index in range(2):
        offset = cell_index * source.num_fibers
        shift = np.array([1000. * cell_index, 0., 0.])
        for link in original['crosslinks']:
            expected_links.append({
                **link, 'fiber_i': link['fiber_i'] + offset, 'fiber_j': link['fiber_j'] + offset,
                'position': (np.array(link['position']) + shift).tolist(),
            })
    assert restored.to_dict()['crosslinks'] == expected_links
    assert int(restored.connectivity_matrix(sparse=True).sum()) == 360
    assert source.to_dict() == original


def test_nonperiodic_merge_of_separated_zigzags_preserves_sampled_welds():
    source = ZigZagGenerator.simple_zigzag(amplitude=2., wavelength=4., n_periods=2)
    cell = source.to_fiber_network()
    offsets = [[0., 0., 0.], [20., 0., 0.]]
    expected = merge([cell, cell], offsets=offsets, merge_crosslinks=False)
    merged = merge([cell, cell], offsets=offsets, proximity_threshold=.5)

    graph, report = convert_sampled_welded_fibers(merged)
    assert merged.to_dict() == expected.to_dict()
    topo = analyze_topology(graph)
    assert (topo.n_nodes, topo.n_physical_edges, topo.n_components) == (162, 160, 2)
    for link, node in zip(merged.crosslinks, report['crosslink_junction_node_ids']):
        np.testing.assert_array_equal(graph.nodes[node].position, link.position)


def test_repeated_input_at_shared_endpoint_gets_a_cross_input_weld():
    source = ZigZagGenerator.simple_zigzag(amplitude=2., wavelength=4., n_periods=2)
    cell = source.to_fiber_network()
    merged = merge([cell, cell], offsets=[[0., 0., 0.], [8., 0., 0.]],
                   proximity_threshold=.01)

    assert (merged.num_fibers, merged.num_crosslinks) == (8, 7)
    contact = merged.crosslinks[-1]
    assert (contact.fiber_i, contact.fiber_j, contact.param_i, contact.param_j) == (3, 4, 1., 0.)
    np.testing.assert_array_equal(contact.position, [8., 0., 0.])
    graph, _ = convert_sampled_welded_fibers(merged)
    topology = analyze_topology(graph)
    assert (topology.n_nodes, topology.n_physical_edges, topology.n_components) == (161, 160, 1)


def test_merge_keeps_original_parallel_records_and_detects_cross_input_contacts():
    left = FiberNetwork([
        Fiber([[0., 0., 0.], [1., 0., 0.]], fiber_id=97),
        Fiber([[0., .05, 0.], [1., .05, 0.]], fiber_id=97),
    ], [
        Crosslink(0, 1, 0., 0., np.array([0., .025, 0.]), 'pinned', 2., 3.),
        Crosslink(1, 0, 1., 1., np.array([1., .025, 0.]), 'bonded', 5., 7.),
    ], dimension=2)
    right = FiberNetwork([Fiber([[0., 0., 0.], [1., 0., 0.]], fiber_id=-9)], dimension=2)
    networks = [left, FiberNetwork(), right]
    offsets = [[0., 0., 0.], [0., 0., 0.], [1., 0., 0.]]
    original = deepcopy([source.to_dict() for source in networks])

    unchanged = merge(networks, offsets=offsets, merge_crosslinks=False)
    detected = unchanged.detect_contacts(threshold=.06)
    expected = [link for link in detected if 2 in (link.fiber_i, link.fiber_j)]
    assert {(link.fiber_i, link.fiber_j) for link in expected} == {(0, 2), (1, 2)}

    merged = merge(networks, offsets=offsets, proximity_threshold=.06)
    assert merged.to_dict()['crosslinks'][:2] == original[0]['crosslinks']
    added = merged.crosslinks[2:]
    assert len(added) == len(expected)
    for actual, candidate in zip(added, expected):
        assert (actual.fiber_i, actual.fiber_j, actual.param_i, actual.param_j) == (
            candidate.fiber_i, candidate.fiber_j, candidate.param_i, candidate.param_j,
        )
        assert actual.crosslink_type == 'welded'
        assert actual.strength == candidate.strength
        assert actual.stiffness == candidate.stiffness
        np.testing.assert_array_equal(actual.position, candidate.position)
    assert merged.to_dict()['fibers'] == unchanged.to_dict()['fibers']
    assert [source.to_dict() for source in networks] == original
