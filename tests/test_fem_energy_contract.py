"""Additive, detached linear-beam energy without changing the legacy proxy."""

import copy
import json
from unittest.mock import patch

import numpy as np
import pytest
from scipy import sparse

import fibernet as fn
from fibernet.ml.beam_frame_fem import BeamFrameFEM


def graph_fixture(dim=2):
    graph = fn.StructureGraph(dimension=dim)
    graph.add_node([0.0, 0.0, 0.0])
    graph.add_node([1.2, 0.0, 0.0])
    graph.add_edge(0, 1, radius=0.02, material=fn.Material(youngs_modulus=2e6))
    return graph


def raw_fixture(dim=2):
    graph = graph_fixture(dim)
    fem = BeamFrameFEM(E=2e6)
    prepared = fem.graph_to_fem_input(graph, dim=dim)
    solve = fem.solve_2d if dim == 2 else fem.solve_3d
    raw = solve(
        prepared["edge_index"],
        prepared["node_pos"],
        prepared["radii"],
        fixed_nodes=[0],
        prescribed_disp={1: (0.001, 0.0) if dim == 2 else (0.001, 0.0, 0.0)},
        damping=0.0,
    )
    return fem, graph, raw


@pytest.mark.parametrize("dim", [2, 3])
def test_additive_snapshot_without_graph_and_with_other_converter(dim):
    fem, graph, raw = raw_fixture(dim)
    expected = copy.deepcopy(raw["beam_energy"])
    converted = BeamFrameFEM(E=9e9).to_sim_result(raw, graph)
    bare = fem.to_sim_result(raw)
    assert bare.energy == 0.0  # Preserve graph-free historical proxy behavior.
    assert converted.metadata["beam_energy"] == bare.metadata["beam_energy"] == expected
    assert expected["strain_energy"] > 0
    assert expected["unit_system"] == "not_declared"
    assert converted.metadata["energy_kind"] == "axial_stress_proxy"
    # The physical snapshot does not consume later graph/converter mutations.
    graph.edges[0].radius *= 2
    graph.edges[0].material.youngs_modulus *= 3
    raw["u"][:] = 0
    raw["K"].data[:] = 0
    assert fem.to_sim_result(raw, graph).metadata["beam_energy"] == expected
    raw["beam_energy"]["included_terms"].append("fake")
    raw["beam_energy"]["strain_energy"] = 999.0
    assert converted.metadata["beam_energy"] == expected
    converted.metadata["beam_energy"]["included_terms"].clear()
    assert bare.metadata["beam_energy"] == expected


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("detailed", [False, True])
def test_graph_si_route_and_json_persistence(tmp_path, dim, detailed):
    graph = graph_fixture(dim)
    result = fn.simulate(
        graph,
        backend="fem",
        dim=dim,
        stretch_ratio=1.001,
        nonlinear=False,
        material_mode="graph",
        length_unit="m",
    )
    record = result.metadata["beam_energy"]
    assert record["unit_system"] == "SI"
    assert record["dimension"] == dim
    assert record["status"] == "available"
    assert record["included_terms"] == ["axial", "bending"] + (
        ["torsion"] if dim == 3 else []
    )
    np.testing.assert_allclose(record["strain_energy"], result.energy, rtol=1e-13)
    path = tmp_path / "result.json"
    result.save(path, detailed=detailed)
    payload = json.loads(path.read_text())
    assert payload["beam_energy"] == record  # Historical JSON metadata is flat.
    restored = fn.SimResult.load(path)
    assert restored.metadata["beam_energy"] == record
    assert restored.energy == result.energy


def test_legacy_raw_dictionary_does_not_infer_energy_from_stiffness():
    fem, graph, raw = raw_fixture()
    with_record = fem.to_sim_result(raw, graph)
    del raw["beam_energy"]
    legacy = fem.to_sim_result(raw, graph)
    assert "beam_energy" not in legacy.metadata
    assert legacy.energy == with_record.energy
    assert legacy.metadata == {
        k: v for k, v in with_record.metadata.items() if k != "beam_energy"
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("dimension", True),
        ("dimension", 3),
        ("model", "spring"),
        ("status", "estimated"),
        ("strain_energy", None),
        ("strain_energy", True),
        ("strain_energy", -1.0),
        ("strain_energy", float("inf")),
        ("strain_energy", float("nan")),
        ("strain_energy", "1"),
        ("definition", "0.5 * u.T @ K_damped @ u"),
        ("included_terms", ["axial"]),
        ("excluded_terms", []),
        ("unit_system", "J"),
        ("extra", 0),
    ],
)
def test_malformed_snapshot_rejects_conversion_without_mutation(field, value):
    fem, graph, raw = raw_fixture()
    raw["beam_energy"][field] = value
    before = copy.deepcopy(raw["beam_energy"])
    with pytest.raises(ValueError, match="beam_energy"):
        fem.to_sim_result(raw, graph)
    assert repr(raw["beam_energy"]) == repr(before)


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "model",
        "dimension",
        "status",
        "strain_energy",
        "definition",
        "included_terms",
        "excluded_terms",
        "unit_system",
    ],
)
def test_missing_snapshot_fields_reject(field):
    fem, graph, raw = raw_fixture()
    del raw["beam_energy"][field]
    with pytest.raises(ValueError, match="beam_energy"):
        fem.to_sim_result(raw, graph)


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", np.ma.array("linear_beam_frame", mask=True)),
        ("definition", np.ma.array("0.5 * u.T @ K_physical @ u", mask=True)),
        ("model", np.array("linear_beam_frame")),
        ("status", np.array(["available"])),
        ("unit_system", np.array("SI")),
        ("included_terms", np.array(["axial", "bending"])),
        ("included_terms", [np.ma.array("axial", mask=True), "bending"]),
        ("excluded_terms", [np.array("numerical_regularization")]),
        ("strain_energy", 10**1000),
        ("strain_energy", np.ma.array(1.0, mask=True)),
    ],
)
def test_non_json_and_unrepresentable_values_reject(field, value):
    fem, graph, raw = raw_fixture()
    raw["beam_energy"][field] = value
    with pytest.raises(ValueError, match="beam_energy"):
        fem.to_sim_result(raw, graph)


@pytest.mark.parametrize(
    "value,reason",
    [
        (np.inf, "nonfinite_quadratic_form"),
        (np.nan, "nonfinite_quadratic_form"),
        (-1.0, "negative_quadratic_form"),
    ],
)
def test_invalid_quadratic_is_unavailable_not_zero(value, reason):
    fem, graph, raw = raw_fixture()
    record = fem._linear_beam_energy(sparse.eye(6) * value, np.ones(6), 2)
    assert record["status"] == "unavailable"
    assert record["strain_energy"] is None
    assert record["reason"] == reason
    json.dumps(record, allow_nan=False)
    raw["beam_energy"] = record
    converted = fem.to_sim_result(raw, graph)
    assert converted.metadata["beam_energy"] == record
    raw["beam_energy"]["strain_energy"] = 0.0
    with pytest.raises(ValueError, match="beam_energy"):
        fem.to_sim_result(raw, graph)


def test_nonlinear_inner_energy_is_not_retained_as_final_energy(tmp_path):
    fem, graph, _ = raw_fixture()
    prepared = fem.graph_to_fem_input(graph)
    with patch.object(
        fem, "_linear_beam_energy", wraps=fem._linear_beam_energy
    ) as inner:
        raw = fem.solve_2d_nonlinear(
            prepared["edge_index"],
            prepared["node_pos"],
            prepared["radii"],
            prescribed_disp={1: (0.001, 0.0)},
            fixed_nodes=[0],
            n_steps=2,
            max_iter=1,
        )
    assert inner.call_count == 2
    expected = {
        "schema_version": 1,
        "model": "incremental_updated_geometry_linear_solves",
        "dimension": 2,
        "status": "unavailable",
        "strain_energy": None,
        "reason": "final_nonlinear_energy_not_recovered",
        "unit_system": "not_declared",
    }
    assert raw["beam_energy"] == expected
    result = fem.to_sim_result(raw, graph)
    assert result.metadata["beam_energy"] == expected
    assert result.metadata["energy_kind"] == "axial_stress_proxy"
    result.save(tmp_path / "nonlinear.json", detailed=True)
    assert (
        fn.SimResult.load(tmp_path / "nonlinear.json").metadata["beam_energy"]
        == expected
    )
    raw["beam_energy"]["strain_energy"] = 0.0
    with pytest.raises(ValueError, match="beam_energy"):
        fem.to_sim_result(raw, graph)
