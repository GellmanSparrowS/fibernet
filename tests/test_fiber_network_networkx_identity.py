"""Fiber adjacency projections use list positions and retain source labels."""

from copy import deepcopy

import networkx as nx
import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components

from fibernet import Crosslink, Fiber, FiberNetwork, Material, gen


def assert_fiber_attributes(graph, network):
    assert list(graph.nodes) == list(range(network.num_fibers))
    for index, fiber in enumerate(network.fibers):
        assert graph.nodes[index] == {
            'fiber_id': fiber.fiber_id,
            'length': fiber.length,
            'radius': fiber.radius,
            'material': fiber.material.name,
            'youngs_modulus': fiber.material.youngs_modulus,
        }


def test_projection_uses_list_indices_even_with_sparse_duplicate_labels():
    labels = [91, 17, np.int64(91), -8]
    network = FiberNetwork(
        fibers=[Fiber([[0., i, 0.], [i + 1., i, 0.]], fiber_id=label,
                      radius=.2 + i, material=Material(name=f'material-{i}',
                                                        youngs_modulus=1e6 * (i + 1)))
                for i, label in enumerate(labels)],
        crosslinks=[
            Crosslink(0, 1, .2, .3, np.zeros(3), 'pinned', 2., 3.),
            Crosslink(1, 0, .7, .8, np.ones(3), 'bonded', 7., 11.),
            Crosslink(2, 2, .2, .9, np.ones(3), 'sliding', 5., 6.),
        ],
    )
    before = deepcopy(network.to_dict())
    graph = network.to_networkx()
    assert type(graph) is nx.Graph
    assert_fiber_attributes(graph, network)
    assert graph.nodes[2]['fiber_id'] is labels[2]
    assert list(graph.edges) == [(0, 1), (2, 2)]
    assert graph.edges[0, 1] == {
        'stiffness': 11., 'strength': 7., 'crosslink_type': 'bonded',
    }
    assert graph.edges[2, 2] == {
        'stiffness': 6., 'strength': 5., 'crosslink_type': 'sliding',
    }
    assert nx.number_connected_components(graph) == 3
    assert network.connectivity_matrix(sparse=True)[0, 1] == 2
    assert network.to_dict() == before


@pytest.mark.parametrize('generate, fibers, crosslinks, components', [
    (lambda: gen.curved_random_2d(num_fibers=12, fiber_length=10., box_size=30.,
                                  num_points=9, seed=23), 12, 4, 8),
    (lambda: gen.hierarchical_lattice(levels=2), 39, 75, 1),
    (lambda: gen.ZigZagGenerator(n_cols=3, n_rows=3).to_fiber_network(), 45, 90, 1),
], ids=['curved-random', 'hierarchical', 'dense-label-zigzag'])
def test_generated_network_projection_includes_every_fiber_and_matches_csr(
    generate, fibers, crosslinks, components,
):
    network = generate()
    assert (network.num_fibers, network.num_crosslinks) == (fibers, crosslinks)
    before = deepcopy(network.to_dict())
    graph = network.to_networkx()
    assert_fiber_attributes(graph, network)
    csr = network.connectivity_matrix(sparse=True)
    assert nx.number_connected_components(graph) == components
    assert connected_components(csr, directed=False, return_labels=False) == components
    np.testing.assert_array_equal(
        nx.to_numpy_array(graph, nodelist=range(fibers), weight=None, dtype=bool),
        csr.toarray() > 0,
    )
    assert network.to_dict() == before
