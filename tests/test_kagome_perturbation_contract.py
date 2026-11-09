"""Seeded lattice displacement after connectivity construction, before rotation."""

import numpy as np
import pytest

from fibernet import api
from fibernet.core.material import Material
from fibernet.gen import unified


@pytest.fixture(params=[False, True], ids=["direct", "factory"])
def create_lattice(request):
    if request.param:
        return lambda **kwargs: api.create("lattice_2d", **kwargs)
    return unified.lattice_2d


def centerlines(network):
    return np.array([fiber.centerline for fiber in network.fibers])


def incidences(network):
    return [
        (link.fiber_i, link.fiber_j, link.param_i, link.param_j)
        for link in network.crosslinks
    ]


@pytest.mark.parametrize("grid_size", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_kagome_seed_controls_geometry_without_changing_incidence(
    create_lattice, grid_size
):
    options = dict(topology="KaGoMe", grid_size=grid_size, cell_size=2.5)
    regular = create_lattice(**options, perturbation=0, seed=11)
    first = create_lattice(**options, perturbation=0.1, seed=11)
    repeat = create_lattice(**options, perturbation=0.1, seed=11)
    other = create_lattice(**options, perturbation=0.1, seed=29)

    assert not np.array_equal(centerlines(regular), centerlines(first))
    np.testing.assert_array_equal(centerlines(first), centerlines(repeat))
    assert not np.array_equal(centerlines(first), centerlines(other))
    assert incidences(first) == incidences(regular) == incidences(other)
    assert len(first.fibers) == len(regular.fibers) == len(other.fibers)
    np.testing.assert_array_equal(centerlines(first)[:, :, 2], 0)
    assert np.max(np.abs(centerlines(first) - centerlines(regular))) <= 0.25
    for link in first.crosslinks:
        # Every pre-existing shared endpoint remains at a single position.
        for index, parameter in (
            (link.fiber_i, link.param_i),
            (link.fiber_j, link.param_j),
        ):
            np.testing.assert_array_equal(
                link.position, first.fibers[index].centerline[int(parameter)]
            )


@pytest.mark.parametrize("topology", ["square", "triangular", "honeycomb", "kagome"])
def test_displacement_uses_common_cell_size_and_existing_node_order(
    create_lattice, topology, monkeypatch
):
    material = Material(name="perturbation-contract", youngs_modulus=1234)

    def capture(nodes, edges, **kwargs):
        assert kwargs == {"radius": 0.3, "material": material, "dimension": 2}
        return nodes, edges

    monkeypatch.setattr(unified, "_edges_to_network", capture)
    options = dict(
        topology=topology,
        grid_size=(2, 2),
        cell_size=2.5,
        material=material,
        radius=0.3,
    )
    original, original_edges = create_lattice(**options, perturbation=0)
    displaced, edges = create_lattice(**options, perturbation=0.2, seed=17)
    assert list(displaced) == list(original)
    assert edges == original_edges
    rng = np.random.RandomState(17)
    for node, position in original.items():
        expected = np.array(position) + rng.uniform(-0.2, 0.2, 2) * 2.5
        np.testing.assert_array_equal(displaced[node], expected)


def test_kagome_rotates_after_displacement_and_preserves_material(create_lattice):
    material = Material(name="perturbation-contract", youngs_modulus=1234)
    options = dict(
        topology="kagome",
        grid_size=(2, 2),
        cell_size=2.5,
        perturbation=0.1,
        radius=0.3,
        material=material,
        seed=19,
    )
    original = create_lattice(**options)
    angle = 0.37
    rotated = create_lattice(**options, rotation=angle)
    matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    np.testing.assert_allclose(
        centerlines(rotated)[:, :, :2],
        centerlines(original)[:, :, :2] @ matrix.T,
        rtol=1e-14,
        atol=1e-14,
    )
    assert incidences(rotated) == incidences(original)
    assert rotated.dimension == 2
    assert all(f.material is material and f.radius == 0.3 for f in rotated.fibers)


def test_kagome_keeps_distinct_coincident_node_identities(create_lattice, monkeypatch):
    monkeypatch.setattr(
        unified, "_edges_to_network", lambda nodes, edges, **kwargs: (nodes, edges)
    )
    options = dict(topology="kagome", grid_size=(2, 2), cell_size=1)
    regular, regular_edges = create_lattice(**options)
    displaced, displaced_edges = create_lattice(**options, perturbation=0.1, seed=17)
    first, second = "t0_0_3", "t0_1_1"
    referenced = {node for edge in regular_edges for node in edge}
    assert first in referenced and second in referenced
    assert regular[first] == regular[second]
    assert displaced[first] != displaced[second]
    assert displaced_edges == regular_edges


@pytest.mark.parametrize("perturbation", [0.0, -0.1])
def test_legacy_no_displacement_is_seed_independent(create_lattice, perturbation):
    options = dict(topology="kagome", grid_size=(2, 2))
    regular = create_lattice(**options)
    # No RNG seed admission occurs on the historical nonpositive branch.
    actual = create_lattice(**options, perturbation=perturbation, seed=object())
    np.testing.assert_array_equal(centerlines(actual), centerlines(regular))
    assert incidences(actual) == incidences(regular)


@pytest.mark.parametrize("seed", [None, 23])
def test_kagome_does_not_consume_global_random_state(create_lattice, seed):
    before = np.random.get_state()
    create_lattice(topology="kagome", grid_size=(2, 2), perturbation=0.1, seed=seed)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
