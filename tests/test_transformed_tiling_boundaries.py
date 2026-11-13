"""Transformed tiling labels the nominal outer XY faces after welding."""

import copy

import numpy as np
import pytest

import fibernet as fn
from fibernet.core.tiling import tile_with_transforms
from fibernet.core.transforms import mirror_x, rotate, translate


def _square(dimension=2):
    cell = fn.StructureGraph(dimension=dimension, box_size=(1, 1, 2))
    cell.add_polyline([(0, 0), (1, 0), (1, 1), (0, 1)], closed=True)
    return cell


def _assert_faces(graph):
    width, height = graph.box_size[:2]
    for node in graph.nodes.values():
        x, y = node.position[:2]
        assert node.boundary == (
            abs(x) < graph.tolerance,
            abs(x - width) < graph.tolerance,
            abs(y) < graph.tolerance,
            abs(y - height) < graph.tolerance,
            False,
            False,
        )


@pytest.mark.parametrize("grid", [(1, 1), (2, 2), (3, 2), np.int64(2)])
@pytest.mark.parametrize("mode", ["none", "empty", "identity"])
def test_identity_tiling_matches_tile_2d_labels_and_geometry(grid, mode):
    cell = _square()
    transforms = {"none": None, "empty": {}, "identity": {(0, 0): lambda g: g}}
    result = tile_with_transforms(cell, grid=grid, cell_transforms=transforms[mode])
    expected = fn.tile_2d(cell, grid=grid)
    actual_data, expected_data = result.to_dict(), expected.to_dict()
    # tile_2d alone adds a tiling metadata record; that existing distinction stays.
    actual_data.pop("metadata")
    expected_data.pop("metadata")
    assert actual_data == expected_data
    _assert_faces(result)
    if (grid == (2, 2)) if isinstance(grid, tuple) else (grid == 2):
        assert (result.num_nodes, result.num_edges) == (9, 12)
        assert len(result.get_boundary_nodes()) == 8
        assert len(result.get_internal_nodes()) == 1


@pytest.mark.parametrize(
    "transform",
    [
        lambda g: rotate(g, 90, center=(0.5, 0.5)),
        lambda g: mirror_x(g, origin=0.5),
    ],
)
def test_transformed_cells_recompute_faces_in_the_final_frame(transform):
    cell = _square()
    # Deliberately stale and conflicting labels must not leak into the result.
    for node in cell.nodes.values():
        node.boundary = (True,) * 6
    center = cell.add_node((0.5, 0.5), boundary=(True,) * 6)
    cell.nodes[center].metadata["source"] = {"values": [1]}
    cell.metadata["units"] = {"length": "mm"}
    before = copy.deepcopy(cell.to_dict())
    calls = []

    def callback(graph):
        calls.append(graph)
        return transform(graph)

    result = tile_with_transforms(
        cell, grid=(2, 2), cell_transforms={(1, 0): callback, (0, 1): callback}
    )
    assert len(calls) == 2 and all(graph is cell for graph in calls)
    assert (result.num_nodes, result.num_edges) == (13, 12)
    assert len(result.get_boundary_nodes()) == 8
    assert len(result.get_internal_nodes()) == 5
    _assert_faces(result)
    assert cell.to_dict() == before
    result.metadata["units"]["length"] = "m"
    next(n for n in result.nodes.values() if "source" in n.metadata).metadata["source"][
        "values"
    ].append(2)
    assert cell.to_dict() == before
    restored = fn.StructureGraph.from_dict(result.to_dict())
    assert restored.get_boundary_nodes() == result.get_boundary_nodes()
    assert restored.get_internal_nodes() == result.get_internal_nodes()


def test_faces_are_nominal_cell_planes_not_transformed_geometry_extrema():
    cell = _square()
    transforms = {
        (ix, iy): lambda g: translate(g, (0.25, 0.375))
        for ix in range(2)
        for iy in range(2)
    }
    result = tile_with_transforms(cell, grid=2, cell_transforms=transforms)
    assert (result.num_nodes, result.num_edges) == (9, 12)
    np.testing.assert_array_equal(result.box_size, [2, 2, 0])
    assert result.get_boundary_nodes() == []
    assert len(result.get_internal_nodes()) == 9
    _assert_faces(result)


@pytest.mark.parametrize(
    "distance,flagged",
    [
        (-0.375, False),
        (-0.25, False),
        (-0.125, True),
        (0, True),
        (0.125, True),
        (0.25, False),
        (0.375, False),
    ],
)
@pytest.mark.parametrize("override", [False, True])
def test_face_tolerance_is_strict_and_uses_effective_tolerance(
    distance, flagged, override
):
    cell = fn.StructureGraph(
        dimension=2, box_size=(2, 2), tolerance=1e-6 if override else 0.25
    )
    cell.add_node((distance, 1), boundary=(True,) * 6)
    kwargs = {"tolerance": 0.25} if override else {}
    result = tile_with_transforms(cell, grid=1, **kwargs)
    assert result.nodes[0].boundary == (flagged, False, False, False, False, False)
    assert result.tolerance == 0.25


@pytest.mark.parametrize("reverse", [False, True])
def test_merged_labels_belong_to_retained_representative_not_union(reverse):
    cell = fn.StructureGraph(dimension=2, box_size=(2, 2))
    positions = [(0.375, 1), (0.1875, 1)]
    if reverse:
        positions.reverse()
    for position in positions:
        cell.add_node(position, merge=False)
    result = tile_with_transforms(cell, grid=1, tolerance=0.25)
    assert result.num_nodes == 1
    np.testing.assert_array_equal(result.nodes[0].position, [*positions[0], 0])
    assert result.nodes[0].boundary == (reverse, False, False, False, False, False)
    assert (
        result.to_dict()["nodes"]
        == fn.tile_2d(cell, grid=1, tolerance=0.25).to_dict()["nodes"]
    )


@pytest.mark.parametrize("dimension", [2, 3])
def test_xy_tiling_keeps_z_faces_false_and_respects_box_override(dimension):
    cell = fn.StructureGraph(dimension=dimension, box_size=(8, 9, 10))
    cell.add_node((0, 0, -3), boundary=(True,) * 6)
    cell.add_node((1, 1, 7), boundary=(True,) * 6)
    cell.add_edge(0, 1)
    result = tile_with_transforms(cell, grid=(2, 2), box_size=(1, 1, 99))
    assert result.dimension == dimension
    np.testing.assert_array_equal(result.box_size, [2, 2, 0])
    _assert_faces(result)
    assert len(result.get_boundary_nodes()) == 6
    assert all(not any(node.boundary[4:]) for node in result.nodes.values())


@pytest.mark.parametrize("grid", [0, (0, 2), (2, 0)])
def test_empty_grid_stays_empty_and_does_not_invoke_transforms(grid):
    def forbidden(graph):
        raise AssertionError("empty grid must not invoke transforms")

    result = tile_with_transforms(
        _square(), grid=grid, cell_transforms={(0, 0): forbidden}
    )
    assert result.num_nodes == result.num_edges == 0
    assert result.get_boundary_nodes() == result.get_internal_nodes() == []


def test_empty_cell_stays_empty():
    cell = fn.StructureGraph(dimension=2, box_size=(1, 1))
    result = tile_with_transforms(cell, grid=2)
    assert result.num_nodes == result.num_edges == 0
    assert result.get_boundary_nodes() == []
