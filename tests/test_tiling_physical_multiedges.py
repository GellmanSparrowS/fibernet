"""Tiling keeps independent fibers and only welds equivalent shared records."""

import copy

import numpy as np
import pytest

import fibernet as fn
from fibernet.core.material import Material
from fibernet.core.structure_graph import StructureGraph
from fibernet.core.tiling import tile_2d, tile_3d, tile_with_transforms
from fibernet.gen import pattern


TILERS = [tile_2d, tile_3d, tile_with_transforms]


def _dimension(tiler):
    return 3 if tiler is tile_3d else 2


def _grid(tiler, nx=1):
    return (nx, 1, 1) if tiler is tile_3d else (nx, 1)


def _square(tiler):
    dim = _dimension(tiler)
    cell = StructureGraph(dimension=dim, box_size=(1, 1, 1) if dim == 3 else (1, 1))
    cell.add_polyline([(0, 0), (1, 0), (1, 1), (0, 1)], closed=True, n_internal=2)
    return cell


@pytest.mark.parametrize("tiler", TILERS)
def test_one_cell_preserves_every_independent_parallel_fiber(tiler):
    dim = _dimension(tiler)
    graph = StructureGraph(dimension=dim, box_size=[1] * dim)
    a = graph.add_node([0, 0, 0])
    b = graph.add_node([1, 1, 1] if dim == 3 else [1, 1, 0])
    curve = np.array([[0.25, 0.7, 0.0], [0.8, 0.8, 0.0]])
    graph.add_edge(a, b)
    graph.add_edge(a, b, allow_parallel=True)
    graph.add_edge(a, b, radius=0.2, allow_parallel=True)
    graph.add_edge(a, b, material=Material(youngs_modulus=7e8), allow_parallel=True)
    graph.add_edge(a, b, internal_points=curve, segments=6, allow_parallel=True)
    graph.add_edge(b, a, internal_points=curve[::-1].copy(), segments=6, allow_parallel=True)
    graph.edges[0].metadata = {"allow_parallel": False, "radius": "source label"}
    graph.nodes[a].metadata = {"position": "source label", "merge": "not a constructor option"}
    before = copy.deepcopy(graph.to_dict())
    result = tiler(graph, grid=_grid(tiler))
    assert result.num_edges == 6
    assert result.to_dict()["edges"] == before["edges"]
    for nid, node in graph.nodes.items():
        np.testing.assert_array_equal(result.nodes[nid].position, node.position)
        assert result.nodes[nid].metadata == node.metadata
    assert graph.to_dict() == before


@pytest.mark.parametrize("tiler", TILERS)
def test_matching_shared_edges_are_welded_in_both_directions(tiler):
    cell = _square(tiler)
    # Right edge is bottom-to-top, left edge top-to-bottom; paths must reverse.
    result = tiler(cell, grid=_grid(tiler, 2))
    assert result.num_nodes == 6
    assert result.num_edges == 7


@pytest.mark.parametrize("tiler", TILERS)
def test_shared_bundles_keep_maximum_cell_multiplicity(tiler):
    cell = _square(tiler)
    # Left side has two physical fibers; right side has three.
    for edge_id, extra in [(3, 1), (1, 2)]:
        edge = cell.edges[edge_id]
        for _ in range(extra):
            cell.add_edge(edge.node_i, edge.node_j, radius=edge.radius,
                          material=copy.deepcopy(edge.material),
                          internal_points=edge.internal_points.copy(),
                          allow_parallel=True)
    result = tiler(cell, grid=_grid(tiler, 2))
    counts = {0: 0, 1: 0, 2: 0}
    for edge in result.edges.values():
        x0 = result.nodes[edge.node_i].position[0]
        x1 = result.nodes[edge.node_j].position[0]
        if x0 == x1:
            counts[int(x0)] += 1
    assert counts == {0: 2, 1: 3, 2: 3}
    assert result.num_edges == 12  # Four horizontal members plus eight vertical.


@pytest.mark.parametrize("tiler", TILERS)
@pytest.mark.parametrize("difference", ["radius", "material", "tensor", "curve", "segments", "metadata"])
def test_shared_endpoints_do_not_hide_different_physical_members(tiler, difference):
    cell = _square(tiler)
    right = cell.edges[1]
    if difference == "radius":
        right.radius *= 2
    elif difference == "material":
        right.material = Material(youngs_modulus=3e8, electrical_conductivity=9)
    elif difference == "tensor":
        right.material = Material(stiffness_tensor=np.eye(6))
    elif difference == "curve":
        right.internal_points[:, 0] -= 0.2
    elif difference == "segments":
        right.segments = 9
    else:
        right.metadata = {"phase": "different physical fiber"}
    result = tiler(cell, grid=_grid(tiler, 2))
    assert result.num_edges == 8


@pytest.mark.parametrize("tiler", TILERS)
def test_nested_records_are_detached_from_source_and_other_copies(tiler):
    cell = _square(tiler)
    cell.metadata["units"] = {"length": "mm"}
    for node in cell.nodes.values():
        node.metadata["tag"] = {"values": [1]}
    for edge in cell.edges.values():
        edge.metadata["tag"] = {"values": [2]}
        edge.material = Material(extra={"nested": [3]}, stiffness_tensor=np.eye(6))
    before = copy.deepcopy(cell.to_dict())
    result = tiler(cell, grid=_grid(tiler, 2))
    result.nodes[0].metadata["tag"]["values"].append(99)
    result.edges[0].metadata["tag"]["values"].append(99)
    result.edges[0].material.extra["nested"].append(99)
    result.edges[0].material.stiffness_tensor[0, 0] = 99
    result.metadata["units"]["length"] = "m"
    assert cell.to_dict() == before
    assert result.nodes[1].metadata["tag"]["values"] == [1]
    assert result.edges[1].metadata["tag"]["values"] == [2]
    assert result.edges[1].material.extra["nested"] == [3]
    assert result.edges[1].material.stiffness_tensor[0, 0] == 1


class EqualityTrap:
    def __eq__(self, other):
        raise AssertionError("custom equality must not infer physical identity")

    def __hash__(self):
        raise AssertionError("custom hashing must not infer physical identity")


@pytest.mark.parametrize("field", ["metadata", "material"])
def test_unknown_records_are_retained_without_custom_equality_or_hashing(field):
    cell = _square(tile_2d)
    for edge in cell.edges.values():
        if field == "metadata":
            edge.metadata["custom"] = EqualityTrap()
        else:
            edge.material = Material(extra={"custom": EqualityTrap()})
    result = tile_2d(cell, grid=(2, 1))
    assert result.num_edges == 8


def test_cyclic_metadata_is_copied_without_being_used_as_identity_evidence():
    cell = _square(tile_2d)
    for edge in cell.edges.values():
        edge.metadata["self"] = edge.metadata
    result = tile_2d(cell, grid=(2, 1))
    assert result.num_edges == 8
    assert result.edges[0].metadata["self"] is result.edges[0].metadata
    assert result.edges[0].metadata is not cell.edges[0].metadata


@pytest.mark.parametrize("unit,nodes,edges", [
    ("square", 9, 12), ("honeycomb", 18, 22), ("kagome", 25, 40),
])
def test_regular_2d_lattice_counts_are_unchanged(unit, nodes, edges, monkeypatch):
    # APP callers can register different cells under these public names.
    factories = {
        "square": pattern._unit_square,
        "honeycomb": pattern._unit_honeycomb,
        "kagome": pattern._unit_kagome,
    }
    monkeypatch.setitem(pattern._UNIT_FACTORIES, unit, factories[unit])
    graph = fn.pattern_2d(unit=unit, box=(10, 10), grid=(2, 2), n_internal=4)
    assert (graph.num_nodes, graph.num_edges) == (nodes, edges)


@pytest.mark.parametrize("unit,nodes,edges", [
    ("cubic", 27, 54), ("octet", 35, 118), ("diamond_3d", 63, 144),
])
def test_regular_3d_lattice_counts_are_unchanged(unit, nodes, edges):
    graph = fn.pattern_3d(unit=unit, box=(10, 10, 10), grid=(2, 2, 2), n_internal=4)
    assert (graph.num_nodes, graph.num_edges) == (nodes, edges)


@pytest.mark.parametrize("dimension,target", [(2, (1, 1)), (3, (2, 4, 6))])
@pytest.mark.parametrize("padding", [0.0, 0.1])
def test_fit_unit_to_box_bounds_the_complete_curve(dimension, target, padding):
    graph = StructureGraph(dimension=dimension)
    a = graph.add_node([0, 0, 0])
    b = graph.add_node([1, 0, 0])
    curve = np.array([[0.5, 2, 3 if dimension == 3 else 0]], dtype=float)
    graph.add_edge(a, b, internal_points=curve)
    before = copy.deepcopy(graph.to_dict())
    result = fn.fit_unit_to_box(graph, target, padding=padding)
    points = np.vstack((result.node_positions(), result.edges[0].internal_points))
    assert np.all(points[:, :dimension] >= padding - 1e-12)
    assert np.all(points[:, :dimension] <= np.asarray(target) - padding + 1e-12)
    np.testing.assert_allclose((points.min(0) + points.max(0))[:dimension] / 2,
                               np.asarray(target) / 2)
    assert graph.to_dict() == before


@pytest.mark.parametrize("tiler", TILERS)
@pytest.mark.parametrize("field,left,right", [
    ("radius", 1, 1.0),
    ("radius", 1, np.float64(1)),
    ("radius", np.int64(1), np.float32(1)),
    ("segments", 4, np.int64(4)),
    ("density", 1000, 1000.0),
    ("tensor", np.eye(6, dtype=np.int64), np.eye(6, dtype=np.float64)),
    ("metadata", {"value": 1}, {"value": np.float64(1)}),
])
def test_equivalent_builtin_and_numpy_numbers_match_shared_fibers(tiler, field, left, right):
    cell = _square(tiler)
    for edge_id, value in ((3, left), (1, right)):
        edge = cell.edges[edge_id]
        if field == "density":
            edge.material.density = value
        elif field == "tensor":
            edge.material.stiffness_tensor = value
        else:
            setattr(edge, field, value)
    assert tiler(cell, grid=_grid(tiler, 2)).num_edges == 7


class IntegerEqualityTrap(int):
    def __eq__(self, other):
        raise AssertionError("custom integer equality must not infer physical identity")

    def __hash__(self):
        raise AssertionError("custom integer hashing must not infer physical identity")


@pytest.mark.parametrize("left,right", [
    (True, 1),
    (np.bool_(True), np.int64(1)),
    (IntegerEqualityTrap(1), 1),
    (2 ** 53 + 1, float(2 ** 53)),
    (np.uint64(2 ** 64 - 1), float(2 ** 64)),
    (np.array([2 ** 53 + 1], dtype=np.int64), np.array([float(2 ** 53)])),
])
def test_numeric_comparison_does_not_erase_boolean_custom_or_integer_identity(left, right):
    cell = _square(tile_2d)
    cell.edges[3].metadata["value"] = left
    cell.edges[1].metadata["value"] = right
    assert tile_2d(cell, grid=(2, 1)).num_edges == 8


@pytest.mark.parametrize("tiler", TILERS)
@pytest.mark.parametrize("reverse_left", [False, True])
@pytest.mark.parametrize("reverse_right", [False, True])
def test_tolerant_shared_curves_use_maximum_matching_independent_of_edge_order(
        tiler, reverse_left, reverse_right):
    dim = _dimension(tiler)
    cell = StructureGraph(dimension=dim, tolerance=0.001, box_size=[1] * dim)
    a, b, c, d = [cell.add_node(position) for position in
                  [(0, 0), (0, 1), (1, 0), (1, 1)]]
    left = [0.0001, -0.0009]
    right = [1.0, 1.0009]
    for x in left[::-1] if reverse_left else left:
        cell.add_edge(a, b, internal_points=np.array([[x, 0.5, 0]]), allow_parallel=True)
    for x in right[::-1] if reverse_right else right:
        cell.add_edge(c, d, internal_points=np.array([[x, 0.5, 0]]), allow_parallel=True)
    result = tiler(cell, grid=_grid(tiler, 2))
    assert result.num_edges == 6
    interface = [edge for edge in result.edges.values()
                 if result.nodes[edge.node_i].position[0] == 1
                 and result.nodes[edge.node_j].position[0] == 1]
    assert len(interface) == 2
