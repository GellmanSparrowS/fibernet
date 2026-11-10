"""Actual bounded static workflow; stored Material is not a heterogeneous law.

All geometry is synthetic. Lengths are interpreted as metres, E as Pa and
forces as newtons by this audit only; the static solver does not infer units.
"""

import json
import time

import numpy as np
import networkx as nx
import pytest
import fibernet as fn
from fibernet.analysis import analyze_tensile_recruitment
from fibernet.ml.beam_frame_fem import BeamFrameFEM
from fibernet.io.vtk import to_vtk

E = 1e6
DELTA_RATIO = 0.001
REGULARIZATION = 1e-6
MIN_ACTIVE = 1e-7


def synthetic_graph(grid):
    graph = fn.pattern_2d(
        unit="honeycomb", grid=(grid, grid), box=(1.0, 1.0), radius=0.02, seed=23
    )
    data = graph.to_dict()
    # Deliberately noncontiguous IDs and non-sorted insertion order.
    node_map = {record["id"]: 101 + 7 * i for i, record in enumerate(data["nodes"])}
    for record in data["nodes"]:
        record["id"] = node_map[record["id"]]
    data["nodes"].reverse()
    for i, record in enumerate(data["edges"]):
        record.update(
            id=503 + 11 * i,
            node_i=node_map[record["node_i"]],
            node_j=node_map[record["node_j"]],
            radius=0.015 if i % 2 else 0.02,
            material=fn.Material(
                name="soft" if i % 2 else "stiff",
                youngs_modulus=E if i % 2 else 4 * E,
                poissons_ratio=0.3,
                density=1000.0 if i % 2 else 1200.0,
            ).to_dict(),
        )
    data["edges"].reverse()
    data.pop("next_node_id", None)
    data.pop("next_edge_id", None)
    return fn.StructureGraph.from_dict(data)


def quadrature_matrix(pos, edges, radii, modulus):
    """Euler-Bernoulli energy Hessian via exact two-point Gauss integration.

    Independent of production stiffness, section and stress functions. Axial
    displacement is linear, transverse displacement cubic Hermite. Each
    element integrates EA * epsilon^2 + EI * curvature^2 over its length.
    """
    K = np.zeros((3 * len(pos), 3 * len(pos)))
    axial = []
    for (i, j), radius in zip(edges, radii):
        vector = pos[j] - pos[i]
        length = np.linalg.norm(vector)
        tangent = vector / length
        normal = np.array([-tangent[1], tangent[0]])
        area, inertia = np.pi * radius**2, np.pi * radius**4 / 4
        dofs = np.r_[np.arange(3 * i, 3 * i + 3), np.arange(3 * j, 3 * j + 3)]
        Ba = np.r_[-tangent / length, 0.0, tangent / length, 0.0]
        ke = modulus * area * length * np.outer(Ba, Ba)
        for t in ((1 - 1 / np.sqrt(3)) / 2, (1 + 1 / np.sqrt(3)) / 2):
            coefficients = np.array(
                [
                    (-6 + 12 * t) / length**2,
                    (-4 + 6 * t) / length,
                    (6 - 12 * t) / length**2,
                    (-2 + 6 * t) / length,
                ]
            )
            Bb = np.r_[
                coefficients[0] * normal,
                coefficients[1],
                coefficients[2] * normal,
                coefficients[3],
            ]
            ke += modulus * inertia * length / 2 * np.outer(Bb, Bb)
        K[np.ix_(dofs, dofs)] += ke
        axial.append((dofs, Ba, modulus * area, length))
    return K, axial


def dense_oracle(K, left, right, displacement, damping):
    prescribed = {3 * int(i) + k: 0.0 for i in left for k in range(3)}
    for i in right:
        prescribed.update({3 * int(i): displacement, 3 * int(i) + 1: 0.0})
    fixed = np.array(sorted(prescribed))
    free = np.setdiff1d(np.arange(len(K)), fixed)
    u = np.zeros(len(K))
    u[fixed] = [prescribed[i] for i in fixed]
    effective = K + damping * np.eye(len(K))
    u[free] = np.linalg.solve(
        effective[np.ix_(free, free)], -effective[np.ix_(free, fixed)] @ u[fixed]
    )
    return u, free, fixed


def recruitment_oracle(strain, edges, n, left, right, alpha=0.05):
    threshold = max(alpha * float(strain.max()), MIN_ACTIVE)
    active = strain > threshold
    network = nx.Graph()
    network.add_nodes_from(range(n))
    network.add_edges_from(edges[active])
    connected = np.zeros(len(edges), dtype=bool)
    spanning = np.zeros(len(edges), dtype=bool)
    span_nodes = set()
    for component in nx.connected_components(network):
        touch_l = bool(component.intersection(left))
        touch_r = bool(component.intersection(right))
        for k, (i, j) in enumerate(edges):
            if active[k] and i in component and j in component:
                connected[k] = touch_l or touch_r
                spanning[k] = touch_l and touch_r
        if touch_l and touch_r:
            span_nodes.update(component)
    return threshold, connected, spanning, len(span_nodes) / n


def export_and_read(graph, result, recruitment, path, *, native_vtk=False):
    rows = {nid: row for row, nid in enumerate(result.metadata["node_ids"])}
    edge_rows = {eid: row for row, eid in enumerate(result.metadata["edge_ids"])}
    network = graph.to_fiber_network()
    assert len(network.fibers) == graph.num_edges
    displacement, force, stored_E, used_E, active = [], [], [], [], []
    for fiber, (eid, edge) in zip(network.fibers, graph.edges.items()):
        count = len(fiber.centerline)
        np.testing.assert_array_equal(
            fiber.centerline[0], graph.nodes[edge.node_i].position
        )
        np.testing.assert_array_equal(
            fiber.centerline[-1], graph.nodes[edge.node_j].position
        )
        assert fiber.radius == edge.radius
        assert fiber.material.to_dict() == edge.material.to_dict()
        displacement.extend(
            np.linspace(
                result.displacements[rows[edge.node_i]],
                result.displacements[rows[edge.node_j]],
                count,
            )
        )
        force.extend([result.edge_forces[edge_rows[eid]]] * (count - 1))
        stored_E.extend([edge.material.youngs_modulus] * (count - 1))
        used_E.extend([E] * (count - 1))
        active.extend([recruitment.active_edges[-1, edge_rows[eid]]] * (count - 1))
    point_data = {"endpoint_interpolated_displacement_m": np.asarray(displacement)}
    cell_data = {
        "signed_axial_force_N": np.asarray(force),
        "stored_material_E_Pa": np.asarray(stored_E),
        "used_uniform_E_Pa": np.asarray(used_E),
        "threshold_active": np.asarray(active),
    }
    to_vtk(network, path, point_data=point_data, cell_data=cell_data)
    expected_points = np.vstack([fiber.centerline for fiber in network.fibers])
    expected_lines = []
    offset = 0
    for fiber in network.fibers:
        expected_lines.extend(
            [[2, offset + i, offset + i + 1] for i in range(len(fiber.centerline) - 1)]
        )
        offset += len(fiber.centerline)
    # Reader-independent syntax/value checks run with core dependencies only.
    tokens = path.read_text().split()
    start = tokens.index("POINTS")
    assert (
        int(tokens[start + 1]) == len(expected_points) and tokens[start + 2] == "double"
    )
    points = np.asarray(
        tokens[start + 3 : start + 3 + expected_points.size], dtype=float
    ).reshape(-1, 3)
    np.testing.assert_array_equal(points, expected_points)
    start = tokens.index("LINES")
    assert int(tokens[start + 1]) == len(expected_lines)
    assert int(tokens[start + 2]) == 3 * len(expected_lines)
    lines = np.asarray(
        tokens[start + 3 : start + 3 + 3 * len(expected_lines)], dtype=int
    ).reshape(-1, 3)
    np.testing.assert_array_equal(lines, expected_lines)
    assert int(tokens[tokens.index("POINT_DATA") + 1]) == len(expected_points)
    assert int(tokens[tokens.index("CELL_DATA") + 1]) == len(expected_lines)
    for key, expected in {**point_data, **cell_data}.items():
        start = tokens.index(key)
        assert tokens[start + 1] == "float"
        vector = expected.ndim == 2
        assert tokens[start - 1] == ("VECTORS" if vector else "SCALARS")
        if not vector:
            assert tokens[start + 2 : start + 5] == ["1", "LOOKUP_TABLE", "default"]
        start += 2 if vector else 5
        actual = np.array(
            tokens[start : start + expected.size], dtype=np.float32
        ).reshape(expected.shape)
        encoded = np.array(
            [f"{x:.6e}" for x in expected.flat], dtype=np.float32
        ).reshape(expected.shape)
        np.testing.assert_array_equal(actual, encoded)
        assert np.isfinite(actual).all()
    if native_vtk:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy

        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(str(path))
        reader.ReadAllScalarsOn()
        reader.ReadAllVectorsOn()
        reader.Update()
        assert reader.GetErrorCode() == 0
        poly = reader.GetOutput()
        np.testing.assert_array_equal(
            vtk_to_numpy(poly.GetPoints().GetData()), expected_points
        )
        np.testing.assert_array_equal(
            vtk_to_numpy(poly.GetLines().GetData()).reshape(-1, 3), expected_lines
        )
        assert poly.GetNumberOfPoints() == len(expected_points)
        assert poly.GetNumberOfLines() == len(expected_lines)
        for dataset, fields in (
            (poly.GetPointData(), point_data),
            (poly.GetCellData(), cell_data),
        ):
            for key, expected in fields.items():
                actual = vtk_to_numpy(dataset.GetArray(key))
                assert actual.dtype == np.float32 and np.isfinite(actual).all()
                encoded = np.array(
                    [f"{x:.6e}" for x in expected.flat], dtype=np.float32
                ).reshape(expected.shape)
                np.testing.assert_array_equal(actual, encoded)
    return {
        "points": len(expected_points),
        "line_cells": len(expected_lines),
        "native_reader": native_vtk,
        "field_roundtrip": "exact formatted float32",
        "coordinate_roundtrip": "exact float64",
    }


def network_case(grid, case_dir, *, native_vtk=False):
    start = time.perf_counter()
    graph = synthetic_graph(grid)
    case_dir.mkdir(exist_ok=True)
    original = graph.to_dict()
    graph.save_json(case_dir / "graph.json")
    graph = fn.StructureGraph.load_json(case_dir / "graph.json")
    assert graph.to_dict() == original
    fem = BeamFrameFEM(E=E, nu=0.3)
    prepared = fem.graph_to_fem_input(graph, dim=2, pct=0.1)
    pos, edges, radii = (
        prepared["node_pos"],
        prepared["edge_index"].T,
        prepared["radii"],
    )
    node_rows = {node_id: row for row, node_id in enumerate(graph.nodes)}
    np.testing.assert_array_equal(
        pos, [node.position[:2] for node in graph.nodes.values()]
    )
    np.testing.assert_array_equal(
        edges,
        [[node_rows[e.node_i], node_rows[e.node_j]] for e in graph.edges.values()],
    )
    np.testing.assert_array_equal(radii, [edge.radius for edge in graph.edges.values()])
    left, right = prepared["boundaries"]["left"], prepared["boundaries"]["right"]
    count = max(int(graph.num_nodes * 0.1), 1)
    sorted_x = np.argsort(pos[:, 0])
    np.testing.assert_array_equal(left, sorted_x[:count])
    np.testing.assert_array_equal(right, sorted_x[-count:])
    assert set(left).isdisjoint(right)
    # Use the exact represented lambda rather than assume subtraction is exact.
    assert prepared["x_range"] == np.ptp(pos[:, 0])
    delta = np.ptp(pos[:, 0]) * ((1 + DELTA_RATIO) - 1)
    raw = fem.stretch_test(graph, target_stretch=1 + DELTA_RATIO, nonlinear=False)
    result = fn.simulate(
        graph,
        backend="fem",
        stretch_ratio=1 + DELTA_RATIO,
        E=E,
        nu=0.3,
        pct=0.1,
        nonlinear=False,
    )
    np.testing.assert_array_equal(result.displacements[:, :2], raw["u"][:, :2])
    K, axial = quadrature_matrix(pos, edges, radii, E)
    np.testing.assert_allclose(raw["K"].toarray(), K, rtol=2e-14, atol=1e-10)
    u, free, fixed = dense_oracle(K, left, right, delta, REGULARIZATION)
    np.testing.assert_allclose(raw["u"].ravel(), u, rtol=2e-8, atol=2e-11)
    physical_residual = K @ u
    np.testing.assert_allclose(
        raw["reactions"].ravel(), physical_residual, rtol=2e-7, atol=2e-11
    )
    np.testing.assert_allclose(
        physical_residual[free], -REGULARIZATION * u[free], atol=2e-11
    )
    strains = np.array([B @ u[dofs] for dofs, B, EA, length in axial])
    forces = np.array([EA * strain for strain, (_, _, EA, _) in zip(strains, axial)])
    axial_energy = sum(
        0.5 * EA * length * strain**2
        for strain, (_, _, EA, length) in zip(strains, axial)
    )
    np.testing.assert_allclose(result.edge_forces, forces, rtol=2e-7, atol=2e-11)
    np.testing.assert_allclose(result.energy, axial_energy, rtol=2e-7, atol=2e-11)
    assert result.metadata["energy_kind"] == "axial_stress_proxy"
    full_energy = 0.5 * u @ K @ u
    work = 0.5 * np.dot(u[fixed], physical_residual[fixed])
    np.testing.assert_allclose(
        full_energy, work - 0.5 * REGULARIZATION * np.dot(u[free], u[free]), atol=2e-11
    )
    eigenvalues = np.linalg.eigvalsh(K[np.ix_(free, free)])
    assert eigenvalues[0] > 0
    u_zero, _, _ = dense_oracle(K, left, right, delta, 0.0)
    raw_zero = fem.solve_2d(
        prepared["edge_index"],
        pos,
        radii,
        fixed_nodes=left,
        prescribed_disp={int(i): (delta, 0.0) for i in right},
        damping=0.0,
    )
    np.testing.assert_allclose(raw_zero["u"].ravel(), u_zero, rtol=2e-8, atol=2e-11)
    np.testing.assert_allclose((K @ u_zero)[free], 0.0, atol=2e-11)
    # Metadata-only material changes must not silently modify the uniform law.
    changed = fn.StructureGraph.from_dict(graph.to_dict())
    for edge in changed.edges.values():
        edge.material = fn.Material(name="changed", youngs_modulus=80 * E)
    unchanged = fn.simulate(
        changed,
        backend="fem",
        stretch_ratio=1 + DELTA_RATIO,
        E=E,
        nonlinear=False,
        pct=0.1,
    )
    np.testing.assert_array_equal(unchanged.edge_forces, result.edge_forces)
    np.testing.assert_array_equal(unchanged.displacements, result.displacements)
    assert graph.to_dict() == original
    result.save(case_dir / "result.json", detailed=True)
    loaded = fn.SimResult.load(case_dir / "result.json")

    def json_record(value):
        return json.loads(
            json.dumps(
                value.to_dict(detailed=True),
                default=lambda x: x.tolist(),
                allow_nan=False,
            )
        )

    assert json_record(result) == json_record(loaded)
    for name in (
        "displacements",
        "deformed_positions",
        "edge_forces",
        "edge_stretches",
    ):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(result, name))
    assert loaded.metadata["node_ids"] == list(graph.nodes)
    assert loaded.metadata["edge_ids"] == list(graph.edges)
    # Analyze the reloaded solver record; the quadrature result is only the oracle.
    # Linear axial strain is deliberately selected, not chord-length strain.
    loaded_strains = np.asarray(loaded.metadata["sigma_axial"]) / E
    np.testing.assert_allclose(loaded_strains, strains, rtol=2e-7, atol=2e-14)
    history = np.stack([np.zeros_like(loaded_strains), loaded_strains])
    recruitment = analyze_tensile_recruitment(
        history, edges, left, right, graph.num_nodes, alpha=0.05, min_active=MIN_ACTIVE
    )
    threshold, active, spanning, span_fraction = recruitment_oracle(
        loaded_strains, edges, graph.num_nodes, set(left), set(right)
    )
    np.testing.assert_array_equal(recruitment.active_edges[-1], active)
    np.testing.assert_array_equal(recruitment.edge_in_spanning[-1], spanning)
    np.testing.assert_allclose(recruitment.spanning_frac[-1], span_fraction, rtol=1e-7)
    np.testing.assert_allclose(recruitment.threshold_curve[-1], threshold, rtol=1e-7)
    assert active.any() and not active.all()
    export = export_and_read(
        graph, loaded, recruitment, case_dir / "fields.vtk", native_vtk=native_vtk
    )
    return {
        "grid": grid,
        "nodes": graph.num_nodes,
        "edges": graph.num_edges,
        "stored_materials": sorted({e.material.name for e in graph.edges.values()}),
        "used_E_Pa": E,
        "stretch_ratio": 1 + DELTA_RATIO,
        "dofs": len(K),
        "minimum_free_stiffness_eigenvalue": float(eigenvalues[0]),
        "free_stiffness_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "max_displacement_oracle_error": float(np.max(np.abs(raw["u"].ravel() - u))),
        "max_axial_force_oracle_error": float(
            np.max(np.abs(result.edge_forces - forces))
        ),
        "max_free_physical_residual": float(np.max(np.abs(physical_residual[free]))),
        "regularization": REGULARIZATION,
        "regularization_max_displacement_change": float(np.max(np.abs(u - u_zero))),
        "axial_energy_proxy_J": result.energy,
        "full_beam_energy_J": float(full_energy),
        "bending_energy_fraction": float(1 - axial_energy / full_energy),
        "recruitment_min_active": MIN_ACTIVE,
        "recruitment_threshold": threshold,
        "recruited_edges": int(active.sum()),
        "spanning_edges": int(spanning.sum()),
        "graph_and_result_roundtrip": "exact",
        "material_precedence": "uniform E unchanged",
        "vtk": export,
        "elapsed_seconds": time.perf_counter() - start,
    }


def series_material_gap():
    graph = fn.StructureGraph(dimension=2)
    for x in (0.0, 1.0, 2.0):
        graph.add_node([x, 0.0])
    for i, modulus in enumerate((E, 4 * E)):
        graph.add_edge(
            i,
            i + 1,
            radius=0.02,
            material=fn.Material(name=f"E{i+1}", youngs_modulus=modulus),
        )
    result = fn.simulate(
        graph, backend="fem", stretch_ratio=1.001, E=E, nonlinear=False, pct=0.1
    )
    delta = 2 * (1.001 - 1)
    area = np.pi * 0.02**2
    uniform_force = E * area * delta / 2
    heterogeneous_force = delta / (1 / (E * area) + 1 / (4 * E * area))
    np.testing.assert_allclose(result.edge_forces, uniform_force, rtol=1e-8)
    assert not np.isclose(result.edge_forces[0], heterogeneous_force, rtol=0.1)
    return {
        "stored_moduli_Pa": [E, 4 * E],
        "used_uniform_E_Pa": E,
        "actual_forces_N": result.edge_forces.tolist(),
        "uniform_oracle_force_N": uniform_force,
        "heterogeneous_series_oracle_force_N": heterogeneous_force,
        "actual_midpoint_displacement_m": float(result.displacements[1, 0]),
        "heterogeneous_midpoint_displacement_m": heterogeneous_force / (E * area),
        "conclusion": "Heterogeneous static Material mechanics are not implemented by this API; not a regression.",
    }


@pytest.mark.parametrize("grid", [2, 4])
def test_material_metadata_static_network_workflow(tmp_path, grid):
    network_case(grid, tmp_path)


def test_optional_native_vtk_static_network_readback(tmp_path):
    pytest.importorskip("vtk")
    network_case(2, tmp_path, native_vtk=True)


def test_stored_materials_do_not_enable_heterogeneous_static_mechanics():
    series_material_gap()


def test_quadrature_oracle_axial_bending_and_rigid_motion():
    # Independent closed-form checks protect the test oracle itself.
    length, radius, modulus = 2.0, 0.1, 1e6
    K, _ = quadrature_matrix(
        np.array([[0.0, 0.0], [length, 0.0]]), np.array([[0, 1]]), [radius], modulus
    )
    area, inertia = np.pi * radius**2, np.pi * radius**4 / 4
    extension, curvature = 0.003, 0.002
    axial = np.array([0.0, 0.0, 0.0, extension, 0.0, 0.0])
    bending = np.array(
        [0.0, 0.0, 0.0, 0.0, curvature * length**2 / 2, curvature * length]
    )
    np.testing.assert_allclose(
        0.5 * axial @ K @ axial, 0.5 * modulus * area * extension**2 / length
    )
    np.testing.assert_allclose(
        0.5 * bending @ K @ bending, 0.5 * modulus * inertia * curvature**2 * length
    )
    for rigid in (
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, length, 1.0],
    ):
        np.testing.assert_allclose(K @ rigid, 0.0, atol=1e-9)
