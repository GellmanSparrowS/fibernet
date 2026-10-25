"""Archived visual-geometry owner only: no numerical solver or native backend.

The complete source module is imported with its optional Taichi import disabled.
This is not qualification of a supported FEM facade or a deformed-transport model.
A separately pinned historical method excerpt supplies a same-host oracle.
Static golden strings remain an additional guard, with narrowly checked
historical subnormal-to-zero differences only; candidate parity is exact.
"""
import builtins
import copy
from decimal import Decimal
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import sys
from unittest.mock import patch
from typing import NamedTuple

import numpy as np
import pytest

from fibernet.core.structure_graph import StructureGraph


FIXTURE = json.loads((Path(__file__).parent / "fixtures" /
                      "legacy_deformed_graph_golden.json").read_text())


# This literal is copied from the independently verified historical owner.
# Pin normalized-LF source text rather than version-dependent AST dump hashes.
HISTORICAL_METHOD_SHA256 = "cb8f174424f2fb90c15e7b66a21cb691b53f5093a3e5cabdf9971e7a6bd08513"
HISTORICAL_MODULE_SHA256 = "210e679eb617f49514f46b260bf53478f9141cd84f64949a8cf8066eb55b099c"
_MIN_NORMAL = Decimal.from_float(sys.float_info.min)
_JSON_TOKENS = re.compile(
    r'"(?:\\.|[^"\\])*"|(?P<number>-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)'
)
_SUBNORMAL_PATHS = frozenset({
    ("nodes", 0, "position", 0),
    ("nodes", 0, "position", 1),
    ("edges", 0, "internal_points", 0, 1),
    ("edges", 0, "internal_points", 1, 1),
})


class _NumberToken(NamedTuple):
    kind: str
    text: str


class _ObjectPairs(tuple):
    """Retain JSON object order and duplicate keys for strict comparison."""


def _json_tokens(raw):
    def invalid_constant(value):
        raise AssertionError(f"Nonfinite JSON constant: {value}")

    return json.loads(
        raw, object_pairs_hook=_ObjectPairs,
        parse_int=lambda value: _NumberToken("int", value),
        parse_float=lambda value: _NumberToken("float", value),
        parse_constant=invalid_constant,
    )


def _assert_historical_golden(reference, expected, case_name):
    """Allow only observed historical-host subnormal-to-zero differences.

    The candidate never uses this exception. It must equal the historical
    reference byte for byte, including zero signs and all numeric tokens.
    Decimal/token checks do not depend on the host's subnormal float mode.
    """
    if reference == expected:
        return []
    differences = []
    number_tokens = []
    replacements = {}

    def compare(actual, old, path=()):
        assert type(actual) is type(old), (path, "JSON type changed")
        if isinstance(old, _ObjectPairs):
            assert tuple(key for key, _ in actual) == tuple(key for key, _ in old), (
                path, "JSON object keys/order changed")
            for (key, value), (_, old_value) in zip(actual, old):
                compare(value, old_value, path + (key,))
        elif isinstance(old, list):
            assert len(actual) == len(old), (path, "JSON array size changed")
            for index, (value, old_value) in enumerate(zip(actual, old)):
                compare(value, old_value, path + (index,))
        elif isinstance(old, _NumberToken):
            token_index = len(number_tokens)
            number_tokens.append(old.text)
            if actual == old:
                return
            value = Decimal(old.text)
            allowed = (
                case_name == "tiny_signed" and path in _SUBNORMAL_PATHS
                and old.kind == actual.kind == "float"
                and value.is_finite() and not value.is_zero()
                and value.copy_abs() < _MIN_NORMAL
                and actual.text in ("0.0", "-0.0")
            )
            assert allowed, (path, "Unexpected numeric token change", old, actual)
            replacements[token_index] = actual.text
            differences.append({"path": path, "golden": old.text,
                                "historical_host": actual.text})
        else:
            assert actual == old, (path, "JSON value changed")

    compare(_json_tokens(reference), _json_tokens(expected))
    # Prove every other byte remains exact, including whitespace and escaping.
    # The lexer consumes quoted strings before considering numeric tokens.
    expected_text = expected.decode("utf-8")
    matches = [match for match in _JSON_TOKENS.finditer(expected_text)
               if match.group("number") is not None]
    assert [match.group("number") for match in matches] == number_tokens
    parts = []
    cursor = 0
    for index, match in enumerate(matches):
        if index in replacements:
            parts.extend((expected_text[cursor:match.start()], replacements[index]))
            cursor = match.end()
    parts.append(expected_text[cursor:])
    assert "".join(parts).encode("utf-8") == reference
    return differences


def _assert_same_host_json(candidate, historical):
    """No numeric or platform exceptions apply to candidate/reference parity."""
    assert candidate == historical


@pytest.fixture(scope="module")
def historical_reference():
    path = Path(__file__).parent / "fixtures/legacy_deformed_graph_reference.py.txt"
    # read_text performs universal-newline decoding on every supported platform.
    source = path.read_text(encoding="utf-8")
    assert hashlib.sha256(source.encode("utf-8")).hexdigest() == HISTORICAL_METHOD_SHA256
    assert FIXTURE["source_module_sha256"] == HISTORICAL_MODULE_SHA256
    assert FIXTURE["source_commit"] == "39b3c493293ca58d77c311f1eead1d5c7f3b6895"
    namespace = {"np": np, "StructureGraph": StructureGraph}
    exec(compile(source, str(path), "exec"), namespace)
    return namespace["_build_deformed_graph"]



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
def test_legacy_numeric_json_bytes_and_exact_membership(
        archive, historical_reference, case, tmp_path, record_property):
    # Execute the literal historical method first, on independent identical input.
    reference_graph = input_graph()
    reference_before = snapshot(reference_graph)
    reference_fem = archive.BeamFEM(reference_graph)
    assert reference_fem._node_ids == FIXTURE["node_order"] == [4, 9]
    legacy = historical_reference(reference_fem, np.array(case["u"]), case["scale"])
    historical = json.dumps(legacy.to_dict(), indent=2).encode("utf-8")
    legacy_path = tmp_path / "historical.json"
    legacy.save_json(legacy_path)
    assert snapshot(reference_graph) == reference_before
    expected = case["json"].encode("utf-8")
    assert hashlib.sha256(expected).hexdigest() == case["sha256"]
    differences = _assert_historical_golden(historical, expected, case["name"])
    record_property("historical_subnormal_zero_differences", json.dumps(differences))
    assert legacy_path.read_bytes() == historical.replace(
        b"\n", os.linesep.encode("ascii"))

    graph = input_graph()
    before = snapshot(graph)
    assert before == reference_before
    fem = archive.BeamFEM(graph)
    assert fem._node_ids == reference_fem._node_ids
    assert list(graph.nodes) == [9, 4]  # Distinct from the DOF ordering.
    result = fem._build_deformed_graph(np.array(case["u"]), case["scale"])
    path = tmp_path / "deformed.json"
    result.save_json(path)
    # No subnormal, signed-zero, numeric or platform exception applies here.
    _assert_same_host_json(
        json.dumps(result.to_dict(), indent=2).encode("utf-8"), historical)
    assert path.read_bytes() == legacy_path.read_bytes()
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


@pytest.mark.parametrize("expected,reference,paths", [
    (b'{"nodes":[{"position":[5e-324,-5e-324,0.0]}]}',
     b'{"nodes":[{"position":[0.0,-0.0,0.0]}]}',
     [("nodes", 0, "position", 0), ("nodes", 0, "position", 1)]),
    (b'{"edges":[{"internal_points":[[0.0,5e-324,0.0],[0.0,-5e-324,0.0]]}]}',
     b'{"edges":[{"internal_points":[[0.0,-0.0,0.0],[0.0,0.0,0.0]]}]}',
     [("edges", 0, "internal_points", 0, 1),
      ("edges", 0, "internal_points", 1, 1)]),
])
def test_historical_golden_allows_only_observed_coordinate_flush(expected, reference, paths):
    changes = _assert_historical_golden(reference, expected, "tiny_signed")
    assert [change["path"] for change in changes] == paths


@pytest.mark.parametrize("label,expected,reference,case_name", [
    ("normal_to_zero", b'{"nodes":[{"position":[0.25]}]}',
     b'{"nodes":[{"position":[0.0]}]}', "tiny_signed"),
    ("different_nonzero_subnormal", b'{"nodes":[{"position":[5e-324]}]}',
     b'{"nodes":[{"position":[1e-323]}]}', "tiny_signed"),
    ("metadata", b'{"metadata":{"value":5e-324}}',
     b'{"metadata":{"value":0.0}}', "tiny_signed"),
    ("wrong_coordinate", b'{"nodes":[{"position":[0.0,0.0,5e-324]}]}',
     b'{"nodes":[{"position":[0.0,0.0,0.0]}]}', "tiny_signed"),
    ("wrong_node", b'{"nodes":[{}, {"position":[5e-324]}]}',
     b'{"nodes":[{}, {"position":[0.0]}]}', "tiny_signed"),
    ("integer_zero", b'{"nodes":[{"position":[5e-324]}]}',
     b'{"nodes":[{"position":[0]}]}', "tiny_signed"),
    ("bool_zero", b'{"nodes":[{"position":[5e-324]}]}',
     b'{"nodes":[{"position":[false]}]}', "tiny_signed"),
    ("zero_spelling", b'{"nodes":[{"position":[5e-324]}]}',
     b'{"nodes":[{"position":[0e0]}]}', "tiny_signed"),
    ("zero_sign", b'{"nodes":[{"position":[0.0]}]}',
     b'{"nodes":[{"position":[-0.0]}]}', "tiny_signed"),
    ("zero_to_subnormal", b'{"nodes":[{"position":[0.0]}]}',
     b'{"nodes":[{"position":[5e-324]}]}', "tiny_signed"),
    ("wrong_case", b'{"nodes":[{"position":[5e-324]}]}',
     b'{"nodes":[{"position":[0.0]}]}', "baseline"),
    ("array_order", b'{"labels":["a","b"]}',
     b'{"labels":["b","a"]}', "tiny_signed"),
    ("object_order", b'{"a":1,"b":2}', b'{"b":2,"a":1}', "tiny_signed"),
    ("extra_key", b'{"a":1}', b'{"a":1,"b":2}', "tiny_signed"),
    ("duplicate_key", b'{"a":1}', b'{"a":1,"a":1}', "tiny_signed"),
    ("normal_float_spelling", b'{"a":1.0}', b'{"a":1.00}', "tiny_signed"),
    ("integer_float_type", b'{"a":1}', b'{"a":1.0}', "tiny_signed"),
    ("nonfinite", b'{"a":1.0}', b'{"a":NaN}', "tiny_signed"),
    ("whitespace", b'{"a":1.0}', b'{ "a":1.0}', "tiny_signed"),
    ("string_escape", b'{"a":"x"}', b'{"a":"\\u0078"}', "tiny_signed"),
])
def test_historical_golden_rejects_unrelated_drift(label, expected, reference, case_name):
    with pytest.raises(AssertionError):
        _assert_historical_golden(reference, expected, case_name)


@pytest.mark.parametrize("historical,candidate", [
    (b'{"nodes":[{"position":[5e-324]}]}', b'{"nodes":[{"position":[0.0]}]}'),
    (b'{"nodes":[{"position":[0.0]}]}', b'{"nodes":[{"position":[-0.0]}]}'),
])
def test_candidate_cannot_use_the_historical_golden_exception(historical, candidate):
    with pytest.raises(AssertionError):
        _assert_same_host_json(candidate, historical)
