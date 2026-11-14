"""Admission, ownership and precedence for opt-in isotropic edge materials."""

import copy
from unittest.mock import patch

import numpy as np
import pytest

import fibernet as fn
from fibernet.ml.beam_frame_fem import BeamFrameFEM


def graph_fixture(dim=2):
    graph = fn.StructureGraph(dimension=dim)
    for x in (0.0, 1.0, 2.0):
        graph.add_node([x, 0.0, 0.0])
    for i, modulus in enumerate((1e6, 4e6)):
        graph.add_edge(
            i, i + 1, radius=0.02, material=fn.Material(youngs_modulus=modulus)
        )
    return graph


def graph_solve(graph, **kwargs):
    return fn.simulate(
        graph,
        backend="fem",
        stretch_ratio=1.001,
        nonlinear=False,
        material_mode="graph",
        length_unit="m",
        **kwargs,
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_graph_mode_retains_uniform_defaults_and_reuses_solver(dim):
    graph = graph_fixture(dim)
    original = graph.to_dict()
    fem = BeamFrameFEM(E=1e6)
    before = fem.stretch_test(graph, target_stretch=1.001, dim=dim, nonlinear=False)
    heterogeneous = fem.stretch_test(
        graph,
        target_stretch=1.001,
        dim=dim,
        nonlinear=False,
        material_mode="graph",
        length_unit="m",
    )
    after = fem.stretch_test(graph, target_stretch=1.001, dim=dim, nonlinear=False)
    assert "material_contract" not in before and "material_contract" not in after
    for field in ("u", "sigma_axial", "sigma_bending", "edge_forces", "reactions"):
        np.testing.assert_array_equal(before[field], after[field])
    assert not np.allclose(before["edge_forces"], heterogeneous["edge_forces"])
    assert graph.to_dict() == original
    assert fem.E == 1e6 and fem.nu == 0.3
    np.testing.assert_array_equal(
        graph_solve(graph).edge_forces,
        fem.to_sim_result(heterogeneous, graph).edge_forces,
    )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"material_mode": "other"}, "material_mode"),
        ({"material_mode": True}, "material_mode"),
        ({"material_mode": "graph"}, "nonlinear=False"),
        ({"material_mode": "graph", "nonlinear": True}, "nonlinear=False"),
        ({"material_mode": "graph", "nonlinear": False}, "length_unit='m'"),
        (
            {"material_mode": "graph", "nonlinear": False, "length_unit": "mm"},
            "length_unit='m'",
        ),
        ({"length_unit": "m"}, "only supported"),
    ],
)
def test_graph_controls_reject_before_preparation(kwargs, message):
    with patch.object(BeamFrameFEM, "graph_to_fem_input") as prepare:
        with pytest.raises(ValueError, match=message):
            BeamFrameFEM().stretch_test(graph_fixture(), target_stretch=1.001, **kwargs)
        prepare.assert_not_called()


@pytest.mark.parametrize(
    "kwargs", [{"E": 1e6}, {"nu": 0.3}, {"E": None}, {"E": 1e9, "nu": 0.3}]
)
def test_facade_conflicting_override_rejects_before_construction(kwargs):
    with patch.object(
        BeamFrameFEM, "__init__", side_effect=AssertionError("unexpected construction")
    ):
        with pytest.raises(ValueError, match="explicit E or nu"):
            graph_solve(graph_fixture(), **kwargs)


def test_history_does_not_admit_static_graph_mode():
    with pytest.raises(TypeError, match="material_mode"):
        fn.simulate(
            graph_fixture(),
            backend="fem",
            mode="history",
            material_mode="graph",
            material=object(),
            times=[0, 1],
            constraints={},
            length_unit="m",
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("youngs_modulus", 0),
        ("youngs_modulus", -1),
        ("youngs_modulus", True),
        ("youngs_modulus", np.nan),
        ("youngs_modulus", np.inf),
        ("youngs_modulus", "1000"),
        ("youngs_modulus", np.array(1000.0)),
        ("youngs_modulus", 1 + 0j),
        ("youngs_modulus", 10**1000),
        ("poissons_ratio", -0.9999999999999999999999999),
        ("poissons_ratio", 0.5),
        ("poissons_ratio", np.nan),
        ("poissons_ratio", True),
        ("shear_modulus", True),
        ("shear_modulus", 0),
        ("shear_modulus", np.nan),
        ("shear_modulus", 1 + 0j),
        ("shear_modulus", 123),
        ("stiffness_tensor", np.eye(6)),
    ],
)
@pytest.mark.parametrize("dim", [2, 3])
def test_all_material_rows_reject_before_solve_including_discarded_parallel(
    field, value, dim
):
    material = fn.Material(youngs_modulus=1e6)
    setattr(material, field, value)
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])[:, :dim]
    edges = np.array([[0, 1], [1, 0]])
    with patch("fibernet.ml.beam_frame_fem.spsolve") as solve:
        with pytest.raises(ValueError, match="input edge row 1"):
            getattr(BeamFrameFEM(), f"solve_{dim}d")(
                edges,
                positions,
                np.array([0.02, 0.02]),
                fixed_nodes=[0],
                deduplicate=True,
                edge_materials=[fn.Material(), material],
            )
        solve.assert_not_called()


@pytest.mark.parametrize("material", [None, {"youngs_modulus": 1e6}, object()])
def test_graph_missing_or_custom_material_has_edge_identity_error(material):
    graph = graph_fixture()
    graph.edges[1].material = material
    with patch.object(BeamFrameFEM, "graph_to_fem_input") as prepare:
        with pytest.raises(ValueError, match="FEM graph edge 1"):
            graph_solve(graph)
        prepare.assert_not_called()


def test_material_subclass_not_silently_downgraded():
    class CustomMaterial(fn.Material):
        pass

    graph = graph_fixture()
    graph.edges[0].material = CustomMaterial()
    with pytest.raises(ValueError, match="custom constitutive"):
        graph_solve(graph)


@pytest.mark.parametrize("materials", [[], [fn.Material()], [fn.Material()] * 3, 1.0])
def test_raw_material_count_is_input_count(materials):
    inp = BeamFrameFEM.graph_to_fem_input(graph_fixture())
    with patch("fibernet.ml.beam_frame_fem.spsolve") as solve:
        with pytest.raises(ValueError, match="one Material per input edge row"):
            BeamFrameFEM().solve_2d(
                inp["edge_index"],
                inp["node_pos"],
                inp["radii"],
                fixed_nodes=[0],
                edge_materials=materials,
            )
        solve.assert_not_called()


def test_optional_shear_is_derived_and_float32_consistency_is_accepted():
    graph = graph_fixture()
    graph.edges[0].material.shear_modulus = None
    graph.edges[1].material = fn.Material(
        youngs_modulus=np.float32(4e6), poissons_ratio=np.float32(0.3)
    )
    result = graph_solve(graph)
    record = result.metadata["material_contract"]
    np.testing.assert_allclose(
        record["shear_modulus"],
        np.array(record["youngs_modulus"])
        / (2 * (1 + np.array(record["poissons_ratio"]))),
        rtol=0,
        atol=0,
    )
    assert graph.edges[0].material.shear_modulus is None


def test_uniform_mode_does_not_start_consuming_material_records():
    graph = graph_fixture()
    before = fn.simulate(
        graph, backend="fem", stretch_ratio=1.001, E=1e6, nonlinear=False
    )
    for edge in graph.edges.values():
        edge.material = None
    after = fn.simulate(
        graph, backend="fem", stretch_ratio=1.001, E=1e6, nonlinear=False
    )
    np.testing.assert_array_equal(before.displacements, after.displacements)
    np.testing.assert_array_equal(before.edge_forces, after.edge_forces)


@pytest.mark.parametrize("kind", ["nonplanar", "coincident", "short", "nonfinite"])
def test_graph_mode_does_not_claim_projected_or_skipped_members(kind):
    graph = graph_fixture()
    if kind == "nonplanar":
        graph.nodes[1].position[2] = 0.1
    elif kind == "coincident":
        graph.nodes[1].position[:] = graph.nodes[0].position
    elif kind == "short":
        graph.nodes[1].position[0] = 1e-13
    else:
        graph.nodes[1].position[0] = np.nan
    with patch("fibernet.ml.beam_frame_fem.spsolve") as solve:
        with pytest.raises(ValueError, match="graph material mode"):
            graph_solve(graph)
        solve.assert_not_called()


def test_solver_snapshot_survives_input_mutation_during_solve_and_later_conversion():
    import fibernet.ml.beam_frame_fem as module

    graph = graph_fixture()
    inp = BeamFrameFEM.graph_to_fem_input(graph)
    materials = [edge.material for edge in graph.edges.values()]
    kwargs = dict(
        fixed_nodes=[0], prescribed_disp={2: (0.002, 0.0)}, edge_materials=materials
    )
    fem = BeamFrameFEM(E=123.0)
    reference = fem.solve_2d(inp["edge_index"], inp["node_pos"], inp["radii"], **kwargs)
    original_solve = module.spsolve

    def mutate(*args, **options):
        for material in materials:
            material.youngs_modulus = 99e6
            material.poissons_ratio = -0.2
        return original_solve(*args, **options)

    with patch.object(module, "spsolve", side_effect=mutate):
        raw = fem.solve_2d(inp["edge_index"], inp["node_pos"], inp["radii"], **kwargs)
    for name in ("u", "edge_forces", "sigma_axial", "sigma_bending"):
        np.testing.assert_array_equal(raw[name], reference[name])
    result = BeamFrameFEM(E=999.0).to_sim_result(raw, graph)
    expected = np.sum(
        np.asarray(raw["sigma_axial"]) ** 2
        * np.pi
        * 0.02**2
        / (2 * np.array([1e6, 4e6]))
    )
    assert result.energy == pytest.approx(expected)
    assert result.metadata["material_contract"]["edge_ids"] == [0, 1]
    raw["material_contract"]["youngs_modulus"][0] = 17.0
    assert result.metadata["material_contract"]["youngs_modulus"] == [1e6, 4e6]


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("schema_version", 1.0),
        ("model", "anisotropic"),
        ("source", "unknown"),
        ("unit_system", "mm"),
        ("input_edge_rows", [1, 0]),
        ("input_edge_rows", [False, 1]),
        ("youngs_modulus", [1e6]),
        ("youngs_modulus", [None, 4e6]),
        ("poissons_ratio", [-1.0, 0.3]),
        ("shear_modulus", [None, 1.0]),
    ],
)
def test_material_provenance_corruption_rejects_conversion(key, value):
    graph = graph_fixture()
    fem = BeamFrameFEM()
    raw = fem.stretch_test(
        graph,
        target_stretch=1.001,
        nonlinear=False,
        material_mode="graph",
        length_unit="m",
    )
    raw["material_contract"][key] = value
    with pytest.raises(ValueError):
        fem.to_sim_result(raw, graph)


def test_material_snapshot_survives_summary_and_detailed_json(tmp_path):
    graph = graph_fixture()
    graph.save_json(tmp_path / "graph.json")
    restored_graph = fn.StructureGraph.load_json(tmp_path / "graph.json")
    result = graph_solve(restored_graph)
    before = copy.deepcopy(result.metadata["material_contract"])
    for detailed in (False, True):
        path = tmp_path / f"result_{detailed}.json"
        result.save(path, detailed=detailed)
        loaded = fn.SimResult.load(path)
        assert loaded.metadata["material_contract"] == before
        assert loaded.metadata["energy_kind"] == "axial_stress_proxy"
        assert loaded.energy == result.energy
    assert restored_graph.to_dict() == graph.to_dict()


@pytest.mark.parametrize("rows", [[-1, 0], [0, 0], [0.0, 1.0], [False, 1]])
def test_material_rows_reject_without_graph(rows):
    fem = BeamFrameFEM()
    raw = fem.stretch_test(
        graph_fixture(),
        target_stretch=1.001,
        nonlinear=False,
        material_mode="graph",
        length_unit="m",
    )
    raw["edge_list"] = rows
    raw["material_contract"]["input_edge_rows"] = rows
    with pytest.raises(ValueError, match="input_edge_rows"):
        fem.to_sim_result(raw)


@pytest.mark.parametrize("dim", [2, 3])
def test_raw_caller_converted_metre_millimetre_series(dim):
    positions = np.array([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [2.3, 0.0, 0.0]])[:, :dim]
    edges = np.array([[0, 1], [1, 2]])
    radii = np.array([0.017, 0.029])
    outputs = []
    for scale in (1.0, 1000.0):
        laws = [fn.Material(youngs_modulus=E / scale**2) for E in (2e6, 7e6)]
        displacement = (0.0023 * scale,) + (0.0,) * (dim - 1)
        raw = getattr(BeamFrameFEM(), f"solve_{dim}d")(
            edges,
            positions * scale,
            radii * scale,
            fixed_nodes=[0],
            prescribed_disp={2: displacement},
            damping=0.0,
            edge_materials=laws,
        )
        outputs.append(raw)
    np.testing.assert_allclose(
        outputs[0]["u"][:, :dim],
        outputs[1]["u"][:, :dim] / 1000.0,
        rtol=1e-12,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        outputs[0]["edge_forces"][:, 0], outputs[1]["edge_forces"][:, 0], rtol=1e-12
    )
    assert outputs[1]["material_contract"]["unit_system"] == "not_declared"
