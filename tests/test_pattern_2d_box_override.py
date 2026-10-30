"""Effective 2-D factory box drives native geometry; no physics or providers."""
import copy

import numpy as np
import pytest

from fibernet.core.tiling import tile_2d
from fibernet.gen import pattern


def _overridden(**kwargs):
    return pattern.pattern_2d(
        unit="square", box=(10.0, 10.0), grid=(2, 1),
        unit_kwargs={"box": (5.0, 10.0)}, **kwargs)


def _direct(**kwargs):
    return pattern.pattern_2d(unit="square", box=(5.0, 10.0), grid=(2, 1), **kwargs)


def _assert_same_graph(actual, expected):
    # Native serialization includes node/edge IDs and counters, every boundary
    # flag, coordinates, radii/materials, internal points, box and metadata.
    # No raw unit_kwargs provenance is stored by these existing factories.
    assert actual.to_dict() == expected.to_dict()


@pytest.mark.parametrize("controls", (
    {},
    {"mirror_x": True},
    {"mirror_y": True},
    {"rotation": 37.0},
    {"mirror_x": True, "mirror_y": True, "rotation": 15.0},
    {"boundary_mode": "extend"},
    {"boundary_mode": "error"},
))
def test_effective_factory_box_matches_direct_geometry_and_metadata(controls):
    actual, expected = _overridden(**controls), _direct(**controls)
    _assert_same_graph(actual, expected)
    assert actual.metadata["pattern"]["box"] == [5.0, 10.0]
    assert actual.metadata["tiling"]["unit_box"] == [5.0, 10.0]
    np.testing.assert_array_equal(actual.box_size, [10.0, 10.0, 0.0])
    if not controls:
        assert (actual.num_nodes, actual.num_edges) == (6, 7)
        assert len(actual.connected_components()) == 1
        assert sorted({node.position[0] for node in actual.nodes.values()}) == [0.0, 5.0, 10.0]
        assert sum(node.boundary[1] for node in actual.nodes.values()) == 2


@pytest.mark.parametrize("superseded_box", (None, (0.0, -1.0), "unused"))
def test_valid_override_wins_before_invalid_top_level_box_is_inspected(superseded_box):
    actual = pattern.pattern_2d(
        unit="square", box=superseded_box, grid=(2, 1),
        unit_kwargs={"box": (5.0, 10.0)})
    _assert_same_graph(actual, _direct())


@pytest.mark.parametrize("box", ([5, 10], np.array([5.0, 10.0])))
def test_ordinary_box_containers_remain_valid_and_detached_from_output(box):
    options = {"box": box}
    before = copy.deepcopy(options)
    actual = pattern.pattern_2d(
        unit="square", box=(10.0, 10.0), grid=(2, 1), unit_kwargs=options)
    _assert_same_graph(actual, _direct())
    assert options["box"] is box
    np.testing.assert_array_equal(options["box"], before["box"])
    # The resulting graph must not alias the input dimensions.
    actual.box_size = (99.0, 99.0)
    np.testing.assert_array_equal(options["box"], before["box"])


@pytest.mark.parametrize("box", (
    None, (5.0,), (5.0, 10.0, 15.0), (0.0, 10.0), (-1.0, 10.0),
    (float("nan"), 10.0), (float("inf"), 10.0), (True, 10.0),
    ("5", 10.0), np.array([[5.0], [10.0]]), np.array(5.0),
))
def test_invalid_winning_box_rejects_before_factory(box, monkeypatch):
    def forbidden_factory(**kwargs):
        raise AssertionError("invalid effective box reached the factory")
    monkeypatch.setitem(pattern._UNIT_FACTORIES, "square", forbidden_factory)
    with pytest.raises(ValueError, match="box must contain two finite positive real dimensions"):
        pattern.pattern_2d(unit="square", grid=(2, 1), unit_kwargs={"box": box})


def test_registered_factory_receives_the_same_effective_box_as_postprocessing(monkeypatch):
    square = pattern._UNIT_FACTORIES["square"]
    calls = []
    def custom(**kwargs):
        calls.append(kwargs["box"])
        return square(**kwargs)
    monkeypatch.setitem(pattern._UNIT_FACTORIES, "__box_contract__", custom)
    actual = pattern.pattern_2d(
        unit="__box_contract__", box=(10.0, 10.0), grid=(2, 1),
        unit_kwargs={"box": (5.0, 10.0)}, mirror_x=True)
    expected = pattern.pattern_2d(
        unit="__box_contract__", box=(5.0, 10.0), grid=(2, 1), mirror_x=True)
    _assert_same_graph(actual, expected)
    assert calls == [(5.0, 10.0), (5.0, 10.0)]


def test_custom_points_keep_top_level_box_and_ignore_unit_kwargs():
    options = dict(points=[(0.0, 0.0), (5.0, 0.0), (5.0, 10.0)],
                   box=(5.0, 10.0), grid=(2, 1))
    actual = pattern.pattern_2d(unit_kwargs={"box": None}, **options)
    expected = pattern.pattern_2d(**options)
    _assert_same_graph(actual, expected)


def test_three_dimensional_top_level_precedence_is_unchanged():
    options = dict(unit="cubic", box=(5.0, 10.0, 15.0), grid=(1, 1, 1))
    actual = pattern.pattern_3d(unit_kwargs={"box": None}, **options)
    expected = pattern.pattern_3d(**options)
    _assert_same_graph(actual, expected)


def test_direct_tiling_retains_an_explicit_spacing_override():
    unit = pattern.pattern_2d(unit="square", box=(5.0, 10.0), grid=(1, 1))
    graph = tile_2d(unit, grid=(2, 1), box_size=(10.0, 10.0))
    assert (graph.num_nodes, graph.num_edges) == (8, 8)
    assert len(graph.connected_components()) == 2
    assert sorted({node.position[0] for node in graph.nodes.values()}) == [0.0, 5.0, 10.0, 15.0]
    np.testing.assert_array_equal(graph.box_size, [20.0, 10.0, 0.0])
    assert sum(node.boundary[1] for node in graph.nodes.values()) == 0
