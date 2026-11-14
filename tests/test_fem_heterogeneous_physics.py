"""Bounded heterogeneous linear-beam checks against independent energy laws.

Synthetic SI cases only; these tests establish neither convergence nor calibration.
The quadrature helper integrates Hermite curvature, never production stiffness.
"""

import json
import copy

import numpy as np
import pytest
import fibernet as fn
from fibernet.ml.beam_frame_fem import BeamFrameFEM
from test_static_network_workflow import (
    quadrature_matrix,
    dense_oracle,
    synthetic_graph,
)


def materials():
    return [
        fn.Material(youngs_modulus=2e6, poissons_ratio=0.12),
        fn.Material(youngs_modulus=7e6, poissons_ratio=0.43),
    ]


@pytest.mark.parametrize("reverse", [False, True])
def test_unequal_series_force_displacement_and_recovery(reverse):
    pos = np.array([[0.0, 0.0], [0.7, 0.0], [2.3, 0.0]])
    edges = np.array([[0, 1], [1, 2]])
    radii = np.array([0.017, 0.029])
    laws = materials()
    moduli = np.array([m.youngs_modulus for m in laws])
    lengths = np.diff(pos[:, 0])
    area = np.pi * radii**2
    delta = 0.0023
    force = delta / np.sum(lengths / (moduli * area))
    order = [1, 0] if reverse else [0, 1]
    raw = BeamFrameFEM(E=9e8).solve_2d(
        edges[order].T,
        pos,
        radii[order],
        fixed_nodes=[0],
        prescribed_disp={2: (delta, 0.0)},
        damping=0.0,
        edge_materials=[laws[i] for i in order],
    )
    np.testing.assert_allclose(raw["edge_forces"][:, 0], force, rtol=1e-13)
    np.testing.assert_allclose(raw["sigma_axial"], force / area[order], rtol=1e-13)
    np.testing.assert_allclose(
        raw["u"][1, 0], force * lengths[0] / (moduli[0] * area[0])
    )
    np.testing.assert_allclose(raw["reactions"][[0, 2], 0], [-force, force])
    u = raw["u"].ravel()
    np.testing.assert_allclose(0.5 * u @ raw["K"] @ u, 0.5 * force * delta)
    assert raw["material_contract"]["youngs_modulus"] == moduli[order].tolist()


@pytest.mark.parametrize("deduplicate", [False, True])
def test_parallel_distinct_materials_first_row_deduplication(deduplicate):
    pos = np.array([[0.0, 0.0], [1.7, 0.0]])
    edges = np.array([[0, 1], [1, 0]]).T
    radii = np.array([0.023, 0.031])
    laws = materials()
    delta = 0.0017
    raw = BeamFrameFEM().solve_2d(
        edges,
        pos,
        radii,
        fixed_nodes=[0],
        prescribed_disp={1: (delta, 0.0)},
        damping=0.0,
        deduplicate=deduplicate,
        edge_materials=laws,
    )
    selected = [0] if deduplicate else [0, 1]
    expected = np.array(
        [laws[i].youngs_modulus * np.pi * radii[i] ** 2 * delta / 1.7 for i in selected]
    )
    np.testing.assert_allclose(raw["edge_forces"][:, 0], expected)
    np.testing.assert_allclose(raw["reactions"][1, 0], expected.sum())
    assert raw["material_contract"]["input_edge_rows"] == selected


@pytest.mark.parametrize("grid,counts", [(2, (18, 22)), (4, (60, 84))])
def test_bent_honeycomb_dense_quadrature_identity_and_roundtrip(tmp_path, grid, counts):
    graph = synthetic_graph(grid)
    assert (graph.num_nodes, graph.num_edges) == counts
    original = graph.to_dict()
    fem = BeamFrameFEM(E=9e8)
    prepared = fem.graph_to_fem_input(graph)
    pos, edges, radii = (
        prepared["node_pos"],
        prepared["edge_index"].T,
        prepared["radii"],
    )
    laws = [edge.material for edge in graph.edges.values()]
    moduli = np.array([m.youngs_modulus for m in laws])
    K = np.zeros((3 * len(pos), 3 * len(pos)))
    axial = []
    for edge, radius, modulus in zip(edges, radii, moduli):
        part, records = quadrature_matrix(pos, [edge], [radius], modulus)
        K += part
        axial.extend(records)
    left, right = prepared["boundaries"]["left"], prepared["boundaries"]["right"]
    delta = prepared["x_range"] * (1.001 - 1)
    u, free, fixed = dense_oracle(K, left, right, delta, 1e-6)
    raw = fem.stretch_test(
        graph,
        target_stretch=1.001,
        nonlinear=False,
        material_mode="graph",
        length_unit="m",
    )
    result = fn.simulate(
        graph,
        backend="fem",
        stretch_ratio=1.001,
        nonlinear=False,
        material_mode="graph",
        length_unit="m",
    )
    np.testing.assert_allclose(raw["K"].toarray(), K, rtol=3e-14, atol=2e-10)
    np.testing.assert_allclose(raw["u"].ravel(), u, rtol=2e-8, atol=2e-11)
    np.testing.assert_allclose(
        result.displacements[:, :2], u.reshape(-1, 3)[:, :2], atol=2e-11
    )
    residual = K @ u
    np.testing.assert_allclose(
        raw["reactions"].ravel(), residual, rtol=2e-7, atol=3e-11
    )
    np.testing.assert_allclose(residual[free], -1e-6 * u[free], atol=3e-11)
    strains = np.array([B @ u[dofs] for dofs, B, EA, length in axial])
    forces = np.array([EA * strain for strain, (_, _, EA, _) in zip(strains, axial)])
    proxy = sum(
        0.5 * EA * length * strain**2
        for strain, (_, _, EA, length) in zip(strains, axial)
    )
    np.testing.assert_allclose(result.edge_forces, forces, rtol=2e-7, atol=3e-11)
    np.testing.assert_allclose(
        raw["sigma_axial"], moduli * strains, rtol=2e-7, atol=3e-8
    )
    np.testing.assert_allclose(result.energy, proxy, rtol=2e-7)
    # Conversion consumes the producer's snapshot, not this converter's E.
    converted = BeamFrameFEM(E=3e11).to_sim_result(raw, graph)
    np.testing.assert_allclose(converted.energy, proxy, rtol=2e-7)
    full = 0.5 * u @ K @ u
    work = 0.5 * u[fixed] @ residual[fixed]
    np.testing.assert_allclose(full, work - 0.5e-6 * u[free] @ u[free], atol=3e-13)
    assert result.metadata["energy_kind"] == "axial_stress_proxy"
    assert full > 1.1 * proxy  # Bending energy must not be mislabeled as the proxy.
    assert graph.to_dict() == original
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(original))
    restored_graph = fn.StructureGraph.from_dict(json.loads(graph_path.read_text()))
    assert restored_graph.to_dict() == original
    result.save(tmp_path / "result.json", detailed=True)
    restored = fn.SimResult.load(tmp_path / "result.json")
    assert (
        restored.metadata["material_contract"] == result.metadata["material_contract"]
    )
    assert restored.metadata["node_ids"] == list(graph.nodes)
    assert restored.metadata["edge_ids"] == list(graph.edges)
    for name in (
        "displacements",
        "deformed_positions",
        "edge_forces",
        "edge_stretches",
    ):
        np.testing.assert_array_equal(getattr(restored, name), getattr(result, name))
    assert restored.energy == result.energy
    # Graph mode changes actual mechanics; the old default remains uniform.
    uniform = fn.simulate(
        graph, backend="fem", stretch_ratio=1.001, E=1e6, nonlinear=False
    )
    assert not np.allclose(result.edge_forces, uniform.edge_forces, rtol=0.01)
    changed = copy.deepcopy(graph)
    for edge in changed.edges.values():
        edge.material = fn.Material(
            youngs_modulus=3 * edge.material.youngs_modulus,
            poissons_ratio=edge.material.poissons_ratio,
        )
    scaled = fn.simulate(
        changed,
        backend="fem",
        stretch_ratio=1.001,
        nonlinear=False,
        material_mode="graph",
        length_unit="m",
    )
    # Fixed numerical damping does not scale with E: solve the scaled oracle.
    scaled_u, _, _ = dense_oracle(3 * K, left, right, delta, 1e-6)
    scaled_forces = np.array(
        [3 * EA * (B @ scaled_u[dofs]) for dofs, B, EA, length in axial]
    )
    np.testing.assert_allclose(scaled.edge_forces, scaled_forces, rtol=2e-7, atol=3e-11)
    print(
        f'grid={grid}: max_u_error={np.max(abs(raw["u"].ravel()-u)):.3g}, '
        f"max_force_error={np.max(abs(result.edge_forces-forces)):.3g}, "
        f"full_energy={full:.12g}, axial_proxy={proxy:.12g}"
    )


@pytest.mark.parametrize("axis", [1, 2])
def test_3d_two_material_cantilever_bending_closed_form(axis):
    lengths = np.array([0.7, 1.6])
    pos = np.array([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [2.3, 0.0, 0.0]])
    radii = np.array([0.017, 0.029])
    laws = materials()
    EI = np.array([m.youngs_modulus for m in laws]) * np.pi * radii**4 / 4
    force = 0.002
    loads = np.zeros((3, 3))
    loads[2, axis] = force
    raw = BeamFrameFEM().solve_3d(
        np.array([[0, 1], [1, 2]]),
        pos,
        radii,
        forces=loads,
        fixed_nodes=[0],
        damping=0.0,
        edge_materials=laws,
    )
    x = pos[:, 0]
    L = x[-1]
    tip = force * sum(
        ((L - x[i]) ** 3 - (L - x[i + 1]) ** 3) / (3 * EI[i]) for i in range(2)
    )
    slope = force * sum(
        ((L - x[i]) ** 2 - (L - x[i + 1]) ** 2) / (2 * EI[i]) for i in range(2)
    )
    np.testing.assert_allclose(raw["u"][2, axis], tip, rtol=2e-13)
    np.testing.assert_allclose(
        raw["u"][2, 5 if axis == 1 else 4], slope * (1 if axis == 1 else -1), rtol=2e-13
    )
    np.testing.assert_allclose(
        raw["sigma_bending"],
        force * (L - x[:-1]) * radii / (np.pi * radii**4 / 4),
        rtol=2e-13,
    )
    u = raw["u"].ravel()
    np.testing.assert_allclose(0.5 * u @ raw["K"] @ u, 0.5 * force * tip, rtol=2e-13)


def test_3d_mixed_poisson_torsion_series_prescribed_rotation():
    pos = np.array([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [2.3, 0.0, 0.0]])
    radii = np.array([0.017, 0.029])
    laws = materials()
    lengths = np.diff(pos[:, 0])
    J = np.pi * radii**4 / 2
    G = np.array([m.youngs_modulus / (2 * (1 + m.poissons_ratio)) for m in laws])
    angle = 0.003
    torque = angle / np.sum(lengths / (G * J))
    raw = BeamFrameFEM(nu=0.25).solve_3d(
        np.array([[0, 1], [1, 2]]),
        pos,
        radii,
        fixed_nodes=[0],
        damping=0.0,
        prescribed_disp={2: (0.0, 0.0, 0.0, angle, 0.0, 0.0)},
        edge_materials=laws,
    )
    np.testing.assert_allclose(raw["u"][1, 3], torque * lengths[0] / (G[0] * J[0]))
    np.testing.assert_allclose(raw["reactions"][[0, 2], 3], [-torque, torque])
    np.testing.assert_allclose(raw["material_contract"]["shear_modulus"], G)
    np.testing.assert_allclose(raw["sigma_axial"], 0.0, atol=1e-14)
    u = raw["u"].ravel()
    np.testing.assert_allclose(0.5 * u @ raw["K"] @ u, 0.5 * torque * angle)
    common_nu_G = np.array([m.youngs_modulus for m in laws]) / 2.5
    wrong_torque = angle / np.sum(lengths / (common_nu_G * J))
    assert not np.isclose(torque, wrong_torque, rtol=0.01, atol=0.0)


def test_raw_permuted_rows_and_ids_preserve_material_and_physics():
    graph = synthetic_graph(2)
    fem = BeamFrameFEM()
    original = fem.graph_to_fem_input(graph)
    node_ids = list(graph.nodes)[::2] + list(graph.nodes)[1::2]
    edge_ids = list(graph.edges)[::-1]
    rows = {nid: row for row, nid in enumerate(node_ids)}
    edges = np.array(
        [
            [rows[graph.edges[eid].node_i], rows[graph.edges[eid].node_j]]
            for eid in edge_ids
        ]
    ).T
    pos = np.array([graph.nodes[nid].position[:2] for nid in node_ids])
    radii = np.array([graph.edges[eid].radius for eid in edge_ids])
    laws = [graph.edges[eid].material for eid in edge_ids]
    original_ids = list(graph.nodes)
    left = [rows[original_ids[int(i)]] for i in original["boundaries"]["left"]]
    right = [rows[original_ids[int(i)]] for i in original["boundaries"]["right"]]
    delta = original["x_range"] * (1.001 - 1)
    raw = fem.solve_2d(
        edges,
        pos,
        radii,
        fixed_nodes=left,
        prescribed_disp={i: (delta, 0.0) for i in right},
        edge_materials=laws,
    )
    result = BeamFrameFEM(E=8e12).to_sim_result(
        raw, graph, node_ids=node_ids, edge_ids=edge_ids
    )
    reference = fn.simulate(
        graph,
        backend="fem",
        stretch_ratio=1.001,
        material_mode="graph",
        length_unit="m",
        nonlinear=False,
    )
    node_order = [original_ids.index(nid) for nid in node_ids]
    edge_order = [list(graph.edges).index(eid) for eid in edge_ids]
    np.testing.assert_allclose(
        result.displacements, reference.displacements[node_order], atol=3e-12
    )
    np.testing.assert_allclose(
        result.edge_forces, reference.edge_forces[edge_order], atol=3e-11
    )
    np.testing.assert_allclose(result.energy, reference.energy, rtol=2e-10)
    assert result.metadata["node_ids"] == node_ids
    assert result.metadata["edge_ids"] == edge_ids
    assert result.metadata["material_contract"]["edge_ids"] == edge_ids
    assert result.metadata["material_contract"]["youngs_modulus"] == [
        m.youngs_modulus for m in laws
    ]
