"""Archived visual-geometry owner only: no numerical solver or native backend.

The complete source module is imported with its optional Taichi import disabled.
This is not qualification of a supported FEM facade or a deformed-transport model.
Golden strings preserve exact legacy save_json bytes for prescribed inputs.
"""
import builtins
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

from fibernet.core.structure_graph import StructureGraph


FIXTURE = json.loads((Path(__file__).parent / "fixtures" /
                      "legacy_deformed_graph_golden.json").read_text())


def load_archive(path, module_name):
    """Execute the unmodified complete module, not an extracted method."""
    original_import = builtins.__import__

    def no_taichi(name, *args, **kwargs):
        if name.split(".")[0] == "taichi":
            raise ImportError("Taichi disabled for archived geometry tests")
        return original_import(name, *args, **kwargs)

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        with patch("builtins.__import__", side_effect=no_taichi):
            spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    assert module.HAS_TAICHI is False

    def forbidden(*args, **kwargs):
        pytest.fail("A solver method was called by a geometry-only test")

    module.spsolve = forbidden
    for name in dir(module.BeamFEM):
        if (not name.startswith("__") and name != "_build_deformed_graph"
                and callable(getattr(module.BeamFEM, name))):
            setattr(module.BeamFEM, name, forbidden)
    return module


@pytest.fixture(scope="module")
def archive():
    path = Path(__file__).parents[1] / "fibernet/sim/_fem_archive.py"
    return load_archive(path, "_legacy_geometry_runtime_under_test")


def input_graph():
    return StructureGraph.from_dict(copy.deepcopy(FIXTURE["graph"]))


def snapshot(graph):
    return copy.deepcopy((graph.to_dict(), graph._spatial_hash, graph._edge_set,
                          list(graph.nodes), list(graph.edges)))


def assert_membership_exact(graph):
    memberships = [(key, nid) for key, bucket in graph._spatial_hash.items()
                   for nid in bucket]
    assert len(memberships) == graph.num_nodes
    assert sorted(nid for key, nid in memberships) == sorted(graph.nodes)
    for key, nid in memberships:
        assert key == graph._pos_key(graph.nodes[nid].position)


@pytest.mark.parametrize("case", FIXTURE["cases"],
                         ids=lambda case: f'{case["name"]}:{case["scale"]}')
def test_legacy_numeric_json_bytes_and_exact_membership(archive, case, tmp_path):
    graph = input_graph()
    before = snapshot(graph)
    fem = archive.BeamFEM(graph)
    assert fem._node_ids == FIXTURE["node_order"] == [4, 9]
    assert list(graph.nodes) == [9, 4]  # Distinct from the DOF ordering.
    result = fem._build_deformed_graph(np.array(case["u"]), case["scale"])
    path = tmp_path / "deformed.json"
    result.save_json(path)
    expected = case["json"].encode()
    assert hashlib.sha256(expected).hexdigest() == case["sha256"]
    assert json.dumps(result.to_dict(), indent=2).encode("utf-8") == expected
    # save_json keeps platform-native text newlines; do not normalize numbers.
    assert path.read_bytes() == expected.replace(b"\n", os.linesep.encode("ascii"))
    assert snapshot(graph) == before
    assert_membership_exact(result)
    assert list(result.nodes) == list(graph.nodes)
    assert list(result.edges) == list(graph.edges)
    assert result._edge_set == graph._edge_set
    assert (result._next_node_id, result._next_edge_id) == (22, 30)

    # Inspect returned object directly. copy() would conceal a stale lookup.
    count = result.num_nodes
    for node in list(result.nodes.values()):
        found = result.add_node(node.position, merge=True)
        assert found in result.nodes
        np.testing.assert_array_equal(result.nodes[found].position, node.position)
        # A coincident point can match either existing ID, never a new node.
        assert result.num_nodes == count
    assert_membership_exact(result)


def test_public_merge_preserves_original_id_and_vacated_position(archive):
    graph = input_graph()
    result = archive.BeamFEM(graph)._build_deformed_graph(
        np.array(FIXTURE["cases"][0]["u"]), 1.0)
    before = snapshot(result)
    assert result.add_node([.25, -.125, 0.], merge=True) == 4
    assert result.add_node([1.875, .25, 0.], merge=True) == 9
    assert snapshot(result) == before
    # This old position is genuinely vacated, unlike a swapped/coincident case.
    assert result.add_node([0., 0., 0.], merge=True) == 22
    assert_membership_exact(result)
    assert graph.num_nodes == 2 and graph._next_node_id == 22


def test_zero_scale_keeps_legacy_curve_flattening(archive):
    graph = input_graph()
    result = archive.BeamFEM(graph)._build_deformed_graph(np.zeros(6), 0.0)
    assert result.edges[13].internal_points.shape == (0, 3)
    assert result.edges[17].internal_points is None
    assert not np.array_equal(result.edges[12].internal_points,
                              graph.edges[12].internal_points)
    np.testing.assert_array_equal(result.edges[12].internal_points[:, 1], [0., 0.])
    assert result.metadata["deformation_scale"] == 0.0
    assert result.num_edges == 3  # All physical parallel records survive.


def test_output_mutable_ownership_is_independent(archive):
    graph = input_graph()
    before = snapshot(graph)
    u = np.array(FIXTURE["cases"][0]["u"])
    result = archive.BeamFEM(graph)._build_deformed_graph(u, 1.0)
    expected = result.nodes[4].position.copy()
    u[:] = 900
    np.testing.assert_array_equal(result.nodes[4].position, expected)
    for nid in graph.nodes:
        assert result.nodes[nid] is not graph.nodes[nid]
        assert not np.shares_memory(result.nodes[nid].position,
                                    graph.nodes[nid].position)
    for eid in graph.edges:
        assert result.edges[eid] is not graph.edges[eid]
        assert result.edges[eid].material is not graph.edges[eid].material
        if graph.edges[eid].internal_points is not None:
            assert not np.shares_memory(result.edges[eid].internal_points,
                                        graph.edges[eid].internal_points)
    assert not np.shares_memory(result._box_size, graph._box_size)
    result.metadata["units"]["length"] = "changed"
    result.nodes[4].metadata["label"]["value"] = "changed"
    result.edges[12].metadata["nested"]["tag"] = "changed"
    result.edges[12].material.extra["phase"]["name"] = "changed"
    result.edges[12].material.youngs_modulus = 7.
    result._box_size[0] = 99.
    result.edges[12].internal_points[0, 0] = 99.
    result.set_node_positions({4: [99., 99., 0.]})
    assert snapshot(graph) == before
    assert graph.add_node([0., 0., 0.]) == 4


@pytest.mark.parametrize("theta", [np.nan, np.inf, -np.inf])
def test_nonfinite_rotations_are_still_ignored(archive, theta):
    graph = input_graph()
    u = np.array(FIXTURE["cases"][0]["u"])
    expected = archive.BeamFEM(graph)._build_deformed_graph(u, 1.)
    u[2::3] = theta
    result = archive.BeamFEM(graph)._build_deformed_graph(u, 1.)
    assert json.dumps(result.to_dict(), indent=2) == json.dumps(
        expected.to_dict(), indent=2)
    assert_membership_exact(result)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("location", ["early_x", "late_y", "scale"])
def test_nonfinite_translated_position_is_rejected(archive, bad, location):
    graph = input_graph()
    before = snapshot(graph)
    u = np.array(FIXTURE["cases"][0]["u"])
    scale = 1.0
    if location == "scale":
        scale = bad
    else:
        u[0 if location == "early_x" else 4] = bad
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="position must be finite"):
            archive.BeamFEM(graph)._build_deformed_graph(u, scale)
    assert snapshot(graph) == before


def test_finite_but_hash_unrepresentable_position_is_rejected(archive):
    graph = input_graph()
    before = snapshot(graph)
    u = np.zeros(6)
    u[3] = np.finfo(float).max
    with pytest.raises(ValueError, match="spatial hash coordinate range"):
        archive.BeamFEM(graph)._build_deformed_graph(u)
    assert snapshot(graph) == before


def test_finite_inputs_overflowing_to_nonfinite_are_rejected(archive):
    graph = input_graph()
    before = snapshot(graph)
    u = np.zeros(6)
    u[0] = np.finfo(float).max
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="position must be finite"):
            archive.BeamFEM(graph)._build_deformed_graph(u, 2.0)
    assert snapshot(graph) == before
