"""Preserve declared graph geometry and identity at the VTU export boundary."""

import builtins
import copy

import numpy as np
import pytest

from fibernet import StructureGraph
from fibernet.io.meshio_adapter import to_meshio, write_vtu_bundle


def graph():
    result = StructureGraph(dimension=3, box_size=[10, 10, 10])
    a = result.add_node([0, 0, 0])
    removed = result.add_node([9, 9, 9])
    b = result.add_node([1, 2, 3])
    result.remove_node(removed)
    result.add_node([4, 0, 2])  # Isolated node remains subject to validation.
    result.add_edge(a, b, radius=0.1)
    result.add_edge(b, a, radius=0.2, allow_parallel=True)
    first = result.edges.pop(0)
    result.edges[0] = first
    return result


def no_meshio(monkeypatch):
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "meshio":
            pytest.fail("invalid source reached the optional meshio dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)


@pytest.mark.parametrize("mask", [False, True])
@pytest.mark.parametrize(
    "field",
    [
        "position",
        "isolated_position",
        "nested_position",
        "object_position",
        "boundary",
        "nested_boundary",
        "radius",
        "empty_curve",
        "box_size",
    ],
)
def test_masked_source_rejected_before_dependency_and_io(
    tmp_path, monkeypatch, field, mask
):
    source = graph()
    if field in ("position", "isolated_position"):
        node = source.nodes[3 if field == "isolated_position" else 2]
        node.position = np.ma.array(node.position, mask=mask)
    elif field in ("nested_position", "object_position"):
        values = [np.ma.array(9.0, mask=mask), 2.0, 3.0]
        source.nodes[2].position = (
            values if field == "nested_position" else np.array(values, dtype=object)
        )
    elif field == "boundary":
        source.nodes[2].boundary = np.ma.array([True] * 6, mask=mask)
    elif field == "nested_boundary":
        source.nodes[2].boundary = [np.ma.array(True, mask=mask)] + [False] * 5
    elif field == "radius":
        source.edges[0].radius = np.ma.array(0.2, mask=mask)
    elif field == "empty_curve":
        source.edges[0].internal_points = np.ma.array(np.empty((0, 3)), mask=mask)
    else:
        source._box_size = np.ma.array(source.box_size, mask=mask)
    no_meshio(monkeypatch)
    for export in (
        lambda: to_meshio(source, length_unit="mm"),
        lambda: write_vtu_bundle(tmp_path / "bundle", source, length_unit="mm"),
    ):
        with pytest.raises(ValueError, match="masked arrays/scalars"):
            export()
        assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("kind", ["node", "isolated_node", "edge"])
@pytest.mark.parametrize(
    "invalid_id", [99, True, 2.0, -1, 2**63, np.ma.array(2, mask=False)]
)
def test_record_ids_rejected_before_dependency_and_io(
    tmp_path, monkeypatch, kind, invalid_id
):
    source = graph()
    if kind == "edge":
        source.edges[0].edge_id = invalid_id
    else:
        source.nodes[3 if kind == "isolated_node" else 2].node_id = invalid_id
    no_meshio(monkeypatch)
    with pytest.raises(ValueError, match="record ID"):
        write_vtu_bundle(tmp_path / "bundle", source, length_unit="m")
    assert list(tmp_path.iterdir()) == []


def test_valid_numpy_record_ids_and_empty_curves_preserve_bundle(tmp_path):
    meshio = pytest.importorskip("meshio")
    source = graph()
    source.nodes[2].node_id = np.int64(2)
    source.edges[0].edge_id = np.uint64(0)
    source.edges[0].internal_points = np.empty((0, 3))
    before = copy.deepcopy(source.to_dict())
    target = write_vtu_bundle(tmp_path / "bundle", source, length_unit="mm")
    mesh = meshio.read(target / "network.vtu")
    np.testing.assert_array_equal(mesh.point_data["node_id"], [0, 2, 3])
    np.testing.assert_array_equal(mesh.cell_data_dict["edge_id"]["line"], [1, 0])
    np.testing.assert_array_equal(mesh.cells_dict["line"], [[1, 0], [0, 1]])
    np.testing.assert_array_equal(mesh.points, [[0, 0, 0], [1, 2, 3], [4, 0, 2]])
    assert source.to_dict() == before
