"""Historical metamaterial factory names keep their own parameter contracts."""

import inspect
from copy import deepcopy

import numpy as np
import pytest

import fibernet
from fibernet import api, gen
from fibernet.core import transform
from fibernet.core.material import Material

CASES = [
    (
        "reentrant_honeycomb_2d",
        {"cell_height": 7, "cell_width": 9, "reentrant_angle": 135},
        "vertical_thickness",
        1.0,
        2,
    ),
    (
        "chiral_honeycomb_2d",
        {"node_radius": 2, "ligament_length": 5},
        "num_node_points",
        6,
        8,
    ),
    (
        "star_honeycomb_2d",
        {"cell_size": 7, "star_angle": 75, "num_arms": 5},
        "star_inner_angle",
        30.0,
        45,
    ),
    (
        "arrowhead_auxetic_2d",
        {"cell_size": 7, "arrow_angle": 75},
        "arm_angle",
        None,
        40,
    ),
    (
        "hierarchical_lattice_2d",
        {"cell_size": 7, "levels": 1},
        "base_type",
        "triangular",
        "square",
    ),
    ("missing_rib_auxetic_2d", {"cell_size": 7}, "rib_angle", 45.0, 60),
]
NAMES = [case[0] for case in CASES]


def assert_same_network(actual, expected):
    assert actual.dimension == expected.dimension
    np.testing.assert_array_equal(actual.box_size, expected.box_size)
    assert len(actual.fibers) == len(expected.fibers) > 0
    for first, second in zip(actual.fibers, expected.fibers):
        np.testing.assert_array_equal(first.centerline, second.centerline)
        assert first.radius == second.radius
        assert vars(first.material) == vars(second.material)
    assert len(actual.crosslinks) == len(expected.crosslinks)
    for first, second in zip(actual.crosslinks, expected.crosslinks):
        assert (first.fiber_i, first.fiber_j, first.param_i, first.param_j) == (
            second.fiber_i,
            second.fiber_j,
            second.param_i,
            second.param_j,
        )
        assert first.crosslink_type == second.crosslink_type == "welded"
        np.testing.assert_array_equal(first.position, second.position)


@pytest.mark.parametrize("name,options,inactive,default,unsupported", CASES, ids=NAMES)
def test_registered_geometry_and_signature_match_historical_factory(
    name, options, inactive, default, unsupported
):
    factory = getattr(gen, name)
    assert name in api.list_generators()
    assert inspect.signature(api._registry.generators[name]) == inspect.signature(
        factory
    )
    material = Material(name="factory-contract", youngs_modulus=1234)
    params = dict(options, grid_size=(1, 1), radius=0.3, material=material)
    direct = factory(**params)
    actual = api.create(name, **params)
    assert all(fiber.material is material for fiber in actual.fibers)
    assert_same_network(actual, direct)
    assert_same_network(api.create(name, **params, **{inactive: default}), direct)


@pytest.mark.parametrize("name,options,inactive,default,unsupported", CASES, ids=NAMES)
@pytest.mark.parametrize("array_size", [(1, 1), (2, 1)])
def test_real_factory_matches_historical_tiling_and_welding(
    name, options, inactive, default, unsupported, array_size
):
    material = Material(name="factory-contract", youngs_modulus=1234)
    params = dict(options, grid_size=(1, 1), radius=0.3, material=material)
    params[inactive] = default
    cell = getattr(gen, name)(**params)
    lower, upper = cell.bounding_box()
    expected = transform.tile(
        cell, repeats=(*array_size, 1), spacing=upper - lower + 0.1
    )
    expected.crosslinks = []
    expected.auto_crosslink(threshold=0.15, crosslink_type="welded")
    actual = api.create_metamaterial(
        name, array_size=array_size, weld_threshold=0.15, **params
    )
    assert_same_network(actual, expected)
    assert actual.metadata == dict(
        unit_cell=name, array_size=array_size, cell_params=params, weld_threshold=0.15
    )


@pytest.mark.parametrize("name,options,inactive,default,unsupported", CASES, ids=NAMES)
@pytest.mark.parametrize("entry", [api.create, api.create_metamaterial])
def test_nondefault_unused_option_is_rejected(
    name, options, inactive, default, unsupported, entry, monkeypatch
):
    def must_not_tile(*args, **kwargs):
        pytest.fail("Invalid option must fail before tiling")

    monkeypatch.setattr(transform, "tile", must_not_tile)
    with pytest.raises(ValueError, match=rf"{name} does not implement {inactive}"):
        entry(name, **{inactive: unsupported})


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("entry", [api.create, api.create_metamaterial])
def test_unknown_keyword_is_not_swallowed(name, entry):
    with pytest.raises(TypeError, match="unexpected keyword"):
        entry(name, misspelled_parameter=3)


def test_default_factory_selects_historical_reentrant_and_retains_defaults(monkeypatch):
    calls = []
    cell = gen.reentrant_honeycomb_2d(grid_size=(1, 1))
    result = deepcopy(cell)

    def generate(**kwargs):
        calls.append(kwargs)
        return cell

    def tile(source, repeats, spacing):
        assert source is cell
        assert repeats == (3, 3, 1)
        lower, upper = cell.bounding_box()
        np.testing.assert_array_equal(spacing, upper - lower + 0.1)
        return result

    def crosslink(self, **kwargs):
        assert self is result and self.crosslinks == []
        assert kwargs == dict(threshold=0.5, crosslink_type="welded")

    monkeypatch.setitem(api._registry.generators, "reentrant_honeycomb_2d", generate)
    monkeypatch.setattr(transform, "tile", tile)
    monkeypatch.setattr(type(result), "auto_crosslink", crosslink)
    assert api.create_metamaterial() is result
    assert calls == [{}]
    assert result.metadata == dict(
        unit_cell="reentrant_honeycomb_2d",
        array_size=(3, 3),
        cell_params={},
        weld_threshold=0.5,
    )


def test_custom_and_unified_routes_are_not_restricted(monkeypatch):
    assert api._registry.generators["metamaterial_2d"] is gen.metamaterial_2d
    assert not hasattr(fibernet, "create_metamaterial")
    with monkeypatch.context() as context:
        context.setattr(api._registry, "generators", api._registry.generators.copy())

        @api.register_generator("custom_factory_cell")
        def custom(scale):
            return gen.missing_rib_auxetic_2d(cell_size=scale, grid_size=(1, 1))

        result = api.create_metamaterial(
            "custom_factory_cell", array_size=(1, 1), scale=3
        )
        assert result.num_fibers > 0
        assert result.metadata["cell_params"] == {"scale": 3}
        registered = api._registry.generators.copy()
        api._register_builtin_generators()
        assert api._registry.generators == registered
    with pytest.raises(ValueError, match="Unknown unit cell"):
        api.create_metamaterial("missing_factory_cell")


@pytest.mark.parametrize("name,options,inactive,default,unsupported", CASES, ids=NAMES)
def test_real_options_change_geometry(name, options, inactive, default, unsupported):
    common = dict(grid_size=(1, 1))
    original = api.create(name, **common)
    changed = api.create(name, **common, **options)
    before = [fiber.centerline.tolist() for fiber in original.fibers]
    after = [fiber.centerline.tolist() for fiber in changed.fibers]
    assert before != after


def test_real_unchanged_default_factory_is_usable():
    result = api.create_metamaterial()
    assert result.num_fibers > 0 and result.num_crosslinks > 0
    assert result.dimension == 2
    assert result.metadata == dict(
        unit_cell="reentrant_honeycomb_2d",
        array_size=(3, 3),
        cell_params={},
        weld_threshold=0.5,
    )
