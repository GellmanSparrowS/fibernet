"""Native junctions remain usable after the legacy sampled trim bridge."""

import copy
import json

import numpy as np
import pytest

from fibernet import FiberNetwork, StructureGraph
from fibernet.core.transform import trim_to_box


@pytest.mark.parametrize("remove_outside", [True, False])
def test_native_chain_trim_preserves_topology_and_json_roundtrip(remove_outside):
    graph = StructureGraph()
    graph.add_polyline([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])
    graph_before = copy.deepcopy(graph.to_dict())
    network = graph.to_fiber_network()
    network_before = copy.deepcopy(network.to_dict())
    trimmed = trim_to_box(network, [1.2, -1, -1], [3.8, 1, 1], remove_outside)
    assert trimmed.num_fibers == 3
    assert [
        (c.fiber_i, c.fiber_j, c.param_i, c.param_j) for c in trimmed.crosslinks
    ] == [(0, 1, 1.0, 0.0), (1, 2, 1.0, 0.0)]
    for c in trimmed.crosslinks:
        np.testing.assert_array_equal(trimmed.fibers[c.fiber_i].end_point, c.position)
        np.testing.assert_array_equal(trimmed.fibers[c.fiber_j].start_point, c.position)
    expected = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    np.testing.assert_array_equal(trimmed.connectivity_matrix(), expected)
    np.testing.assert_array_equal(trimmed.degree_distribution(), [1, 2, 1])
    assert trimmed.validate()["valid"]
    restored = FiberNetwork.from_dict(json.loads(json.dumps(trimmed.to_dict())))
    assert restored.to_dict() == trimmed.to_dict()
    np.testing.assert_array_equal(restored.connectivity_matrix(), expected)
    assert network.to_dict() == network_before
    assert graph.to_dict() == graph_before
