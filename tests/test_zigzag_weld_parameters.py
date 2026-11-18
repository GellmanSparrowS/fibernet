"""Crosslinks on a generated zigzag reference the actual sampled endpoints."""

from itertools import combinations

import numpy as np
import pytest

from fibernet import FiberNetwork, Material
from fibernet.core.transform import trim_to_box
from fibernet.gen import ZigZagGenerator


def assert_welds_match_samples(network, source):
    source_edges = list(source.edges())
    assert len(network.fibers) == len(source_edges)
    for fiber_id, (fiber, (start, end)) in enumerate(zip(network.fibers, source_edges)):
        assert fiber.fiber_id == fiber_id
        assert fiber.segments == 20
        np.testing.assert_array_equal(fiber.centerline, np.linspace(
            source.nodes[start]['pos'], source.nodes[end]['pos'], 21))

    # Derive expected pairs from the source graph, in the established order.
    seen_in_edge_order = dict.fromkeys(node for edge in source_edges for node in edge)
    expected = []
    for node in seen_in_edge_order:
        incident_indices = [i for i, edge in enumerate(source_edges) if node in edge]
        for first, second in combinations(incident_indices, 2):
            expected.append((first, second, source.nodes[node]['pos']))

    assert len(network.crosslinks) == len(expected)
    roles = set()
    for link, (first, second, position) in zip(network.crosslinks, expected):
        assert (link.fiber_i, link.fiber_j) == (first, second)
        assert link.crosslink_type == 'welded'
        assert link.strength == link.stiffness == float('inf')
        np.testing.assert_array_equal(link.position, position)
        for index, parameter in ((first, link.param_i), (second, link.param_j)):
            assert parameter in (0.0, 1.0)
            fiber = network.fibers[index]
            sample_parameters = np.linspace(0., 1., len(fiber.centerline))
            point = [np.interp(parameter, sample_parameters, axis)
                     for axis in fiber.centerline.T]
            np.testing.assert_array_equal(point, position)
        roles.add((link.param_i, link.param_j))
    return roles


@pytest.mark.parametrize(
    'generator, counts, expected_roles',
    [
        (ZigZagGenerator.simple_zigzag(amplitude=2., wavelength=4., n_periods=2),
         (4, 3), {(1., 0.)}),
        (ZigZagGenerator(n_cols=3, n_rows=3),
         (45, 90), {(0., 0.), (1., 0.), (1., 1.)}),
        (ZigZagGenerator(base_points=[(0, 0), (2, 0), (1, 1), (0, 0), (2, 0), (3, 1)],
                         n_cols=1, n_rows=1),
         (4, 5), {(0., 0.), (1., 0.), (1., 1.)}),
        (ZigZagGenerator(base_points=[(0, 0), (0, 0), (2, 1), (2, 1)],
                         n_cols=1, n_rows=1),
         (1, 0), set()),
        (ZigZagGenerator(base_points=[(0, 0), (0, 0)], n_cols=1, n_rows=1),
         (0, 0), set()),
    ],
    ids=['triangular-teeth', 'mirrored-tiles', 'revisited-fork',
         'single-edge-with-repeat', 'collapsed-edge'],
)
def test_generated_zigzag_welds_preserve_fibers_and_locate_shared_nodes(
    generator, counts, expected_roles,
):
    source = generator.generate()
    material = Material(name='custom-zigzag')
    network = generator.to_fiber_network(radius=.23, material=material)
    assert (network.num_fibers, network.num_crosslinks) == counts
    assert network.dimension == 2
    assert network.metadata == {'generator': 'zigzag'}
    assert all(f.material is material and f.radius == .23 for f in network.fibers)
    assert assert_welds_match_samples(network, source) == expected_roles


def test_mirrored_zigzag_native_json_keeps_weld_endpoints(tmp_path):
    generator = ZigZagGenerator(n_cols=3, n_rows=3)
    network = generator.to_fiber_network()
    path = tmp_path / 'zigzag.json'
    network.save_json(path)
    restored = FiberNetwork.load_json(path)
    assert restored.to_dict() == network.to_dict()
    assert all(f.material.name == 'zigzag' for f in restored.fibers)
    assert assert_welds_match_samples(restored, generator.generate()) == {
        (0., 0.), (1., 0.), (1., 1.),
    }


def test_trim_near_zigzag_peak_keeps_shared_endpoint_weld():
    network = ZigZagGenerator.simple_zigzag(
        amplitude=2., wavelength=4., n_periods=2,
    ).to_fiber_network()
    trimmed = trim_to_box(network, [1.75, 1.75, -1.], [2.25, 2.25, 1.],
                          remove_outside=False)

    assert (trimmed.num_fibers, trimmed.num_crosslinks) == (2, 1)
    link = trimmed.crosslinks[0]
    assert (link.fiber_i, link.param_i, link.fiber_j, link.param_j) == (0, 1., 1, 0.)
    np.testing.assert_array_equal(link.position, [2., 2., 0.])
    np.testing.assert_array_equal(trimmed.fibers[0].end_point, link.position)
    np.testing.assert_array_equal(trimmed.fibers[1].start_point, link.position)
