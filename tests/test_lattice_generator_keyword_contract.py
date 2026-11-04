"""2D lattice keyword admission without geometry or backend execution."""

import pytest

from fibernet import api
from fibernet.gen import unified


@pytest.fixture(autouse=True)
def forbid_geometry_rng_and_conversion(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("keyword admission reached geometry, RNG, or conversion")

    for name in (
        "_build_square_direct",
        "_build_triangular_direct",
        "_build_honeycomb_direct",
        "_unit_cell_kagome",
        "_tile_unit_cell",
        "_edges_to_network",
    ):
        monkeypatch.setattr(unified, name, forbidden)
    monkeypatch.setattr(unified.np.random, "RandomState", forbidden)


@pytest.fixture(params=[False, True], ids=["direct", "api"])
def create_lattice(request):
    if request.param:
        return lambda **kwargs: api.create("lattice_2d", **kwargs)
    return unified.lattice_2d


@pytest.mark.parametrize("topology", ["square", "triangular", "honeycomb", "kagome"])
@pytest.mark.parametrize(
    "options",
    [
        {"cell_szie": 2.0},
        {"z_option": 1, "a_option": 2},
        {"a_option": 2, "z_option": 1},
    ],
)
def test_unknown_options_reject_before_geometry_rng_or_conversion(
    create_lattice, topology, options
):
    with pytest.raises(TypeError) as caught:
        create_lattice(
            topology=topology,
            grid_size=(1, 1),
            perturbation=0.2,
            rotation=0.5,
            seed=17,
            **options,
        )
    assert str(caught.value) == (
        "lattice_2d does not support keyword options: "
        + ", ".join(sorted(options))
    )


@pytest.mark.parametrize("argument", ["topology", "grid_size"])
def test_unknown_options_reject_before_declared_argument_access(
    create_lattice, argument
):
    class Unreadable:
        def lower(self):
            pytest.fail("unknown options reached topology normalization")

        def __iter__(self):
            pytest.fail("unknown options reached grid unpacking")

    options = {"topology": "square", "grid_size": (1, 1)}
    options[argument] = Unreadable()
    with pytest.raises(TypeError) as caught:
        create_lattice(cell_szie=2.0, **options)
    assert str(caught.value) == (
        "lattice_2d does not support keyword options: cell_szie"
    )


@pytest.mark.parametrize(
    "options,expected_size",
    [({}, 10.0), ({"cell_size": 2.0}, 2.0)],
    ids=["default-size", "declared-size"],
)
def test_declared_options_reach_capture_builder_and_conversion(
    create_lattice, options, expected_size, monkeypatch
):
    material = object()
    result = object()
    nodes, edges = {}, []
    builder_calls = []
    conversion_calls = []

    def capture_builder(nx, ny, cell_size):
        builder_calls.append((nx, ny, cell_size))
        return nodes, edges, 0.0, 0.0

    def capture_conversion(actual_nodes, actual_edges, **kwargs):
        assert actual_nodes is nodes
        assert actual_edges is edges
        conversion_calls.append(kwargs)
        return result

    monkeypatch.setattr(unified, "_build_square_direct", capture_builder)
    monkeypatch.setattr(unified, "_edges_to_network", capture_conversion)
    actual = create_lattice(
        topology="SQUARE",
        grid_size=(1, 2),
        perturbation=0.0,
        rotation=0.0,
        radius=0.3,
        material=material,
        seed=17,
        **options,
    )
    assert actual is result
    assert builder_calls == [(1, 2, expected_size)]
    assert conversion_calls == [
        {"radius": 0.3, "material": material, "dimension": 2}
    ]
