"""Public 3D rotations depend on axis direction, not finite axis magnitude."""

import copy

import numpy as np
import pytest

from fibernet.core.structure_graph import StructureGraph
from fibernet.core.transforms import compose, make_rotate, rotate


def graph():
    return StructureGraph.from_dict(
        {
            "version": "3.1",
            "dimension": 3,
            "next_node_id": 50,
            "next_edge_id": 70,
            "nodes": [
                {"id": 4, "position": [2.0, 3.0, 4.0]},
                {"id": 9, "position": [5.0, 7.0, 8.0]},
            ],
            "edges": [
                {
                    "id": 17,
                    "node_i": 4,
                    "node_j": 9,
                    "radius": 0.2,
                    "internal_points": [[3.0, 6.0, 5.0]],
                }
            ],
        }
    )


@pytest.mark.parametrize("magnitude", [1.0, 1e300, 1e-300])
@pytest.mark.parametrize("direction", [[0.0, 0.0, 1.0], [1.0, -2.0, 3.0]])
@pytest.mark.parametrize("via_factory", [False, True])
def test_axis_magnitude_does_not_change_rotation(magnitude, direction, via_factory):
    source = graph()
    before = copy.deepcopy(source.to_dict())
    axis = np.asarray(direction) * magnitude
    axis_before = axis.copy()
    center = [1.0, 2.0, 3.0]
    expected = rotate(source, 73.0, center=center, axis=direction)
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        if via_factory:
            actual = compose(source, make_rotate(73.0, center=center, axis=axis))
        else:
            actual = rotate(source, 73.0, center=center, axis=axis)
    assert list(actual.nodes) == [4, 9]
    assert list(actual.edges) == [17]
    assert (actual._next_node_id, actual._next_edge_id) == (50, 70)
    for node_id in source.nodes:
        np.testing.assert_allclose(
            actual.nodes[node_id].position, expected.nodes[node_id].position
        )
    np.testing.assert_allclose(
        actual.edges[17].internal_points, expected.edges[17].internal_points
    )
    original_distance = np.linalg.norm(
        source.nodes[4].position - source.nodes[9].position
    )
    rotated_distance = np.linalg.norm(
        actual.nodes[4].position - actual.nodes[9].position
    )
    assert rotated_distance == pytest.approx(original_distance)
    assert actual.edges[17].radius == source.edges[17].radius
    assert source.to_dict() == before
    np.testing.assert_array_equal(axis, axis_before)


@pytest.mark.parametrize("magnitude", [np.finfo(float).max, np.nextafter(0.0, 1.0)])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_extreme_finite_coordinate_axes_use_right_hand_rule(magnitude, sign):
    source = StructureGraph(dimension=3)
    source.add_node([1.0, 0.0, 0.0])
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        actual = rotate(source, 90.0, axis=[0.0, 0.0, sign * magnitude])
    np.testing.assert_allclose(actual.nodes[0].position, [0.0, sign, 0.0], atol=1e-15)


@pytest.mark.parametrize(
    "axis",
    [
        [0.0, 0.0, 0.0],
        [np.inf, 0.0, 0.0],
        [np.nan, 0.0, 1.0],
        [],
        [1.0, 2.0],
        [1.0, 2.0, 3.0, 4.0],
        [[1.0, 2.0, 3.0]],
        1.0,
        ["invalid", 0.0, 1.0],
        [1.0 + 1.0j, 0.0, 1.0],
        np.array([1.0 + 1.0j, 0.0, 1.0]),
        np.array([np.complex128(1.0 + 2.0j), 0.0, 0.0], dtype=object),
        np.array([np.complex128(1.0), 0.0, 0.0], dtype=object),
    ],
)
@pytest.mark.parametrize("empty", [False, True])
def test_invalid_3d_axis_is_rejected_before_copy(axis, empty, monkeypatch):
    source = StructureGraph(dimension=3) if empty else graph()
    before = copy.deepcopy(source.to_dict())

    def unexpected_copy():
        pytest.fail("invalid axis must be rejected before copying the graph")

    monkeypatch.setattr(source, "copy", unexpected_copy)
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        with pytest.raises(ValueError, match="axis"):
            rotate(source, 90.0, axis=axis)
    assert source.to_dict() == before


def test_default_and_2d_axis_behavior_is_unchanged():
    for dimension in (2, 3):
        source = StructureGraph(dimension=dimension)
        source.add_node([1.0, 0.0, 0.0])
        actual = rotate(source, 90.0)
        np.testing.assert_allclose(
            actual.nodes[0].position, [0.0, 1.0, 0.0], atol=1e-15
        )
        if dimension == 2:
            ignored = rotate(source, 90.0, axis=[0.0, 0.0, 0.0])
            np.testing.assert_array_equal(
                ignored.nodes[0].position, actual.nodes[0].position
            )
