"""Generated octet radii through linear FEM, saved analysis and VTK fields.

Run: python -m examples.heterogeneous_radius_workflow --output-dir NEW_DIRECTORY
This bounded software example uses straight welded beams and declared SI units.
It is not a physical calibration, convergence or printed-solid validation.
"""

import argparse
import csv
import json
from pathlib import Path


def run(output_dir):
    """Generate once and retain the graph, result, analysis and visualization."""
    import numpy as np

    import fibernet as fn
    from fibernet.analysis import analyze_tensile_recruitment
    from fibernet.analysis.graph_features_3d import GraphFeatureExtractor3D
    from fibernet.io.vtk import to_vtk
    from fibernet.ml.beam_frame_fem import BeamFrameFEM

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    # box is the per-cell size, not the complete grid's extent.
    graph = fn.pattern_3d(
        unit="octet",
        grid=(2, 2, 2),
        box=(0.01, 0.015, 0.02),
        radius=0.0005,
        n_pts_per_side=0,
        seed=23,
    )
    graph.save_json(out / "graph.json")
    graph = fn.StructureGraph.load_json(out / "graph.json")
    # Native octet edges and body diagonals have different stored radii.
    assert len({edge.radius for edge in graph.edges.values()}) == 2
    prepared = BeamFrameFEM.graph_to_fem_input(graph, dim=3, pct=0.1)
    result = fn.simulate(
        graph,
        backend="fem",
        stretch_ratio=1.002,
        nonlinear=False,
        E=1e6,
        nu=0.3,
        pct=0.1,
    )
    result.save(out / "result.json", detailed=True)
    restored = fn.SimResult.load(out / "result.json")
    for field in (
        "displacements",
        "deformed_positions",
        "edge_forces",
        "edge_stretches",
    ):
        np.testing.assert_array_equal(getattr(restored, field), getattr(result, field))
        assert np.isfinite(getattr(restored, field)).all()

    node_ids, edge_ids = restored.metadata["node_ids"], restored.metadata["edge_ids"]
    rows = {nid: i for i, nid in enumerate(node_ids)}
    edge_rows = {eid: i for i, eid in enumerate(edge_ids)}
    edge_pairs = np.array(
        [
            [rows[graph.edges[eid].node_i], rows[graph.edges[eid].node_j]]
            for eid in edge_ids
        ]
    )
    # The preparation and result producers use the same native node row order.
    assert node_ids == list(graph.nodes)
    strain = np.stack([np.zeros(len(edge_ids)), restored.edge_stretches - 1])
    recruitment = analyze_tensile_recruitment(
        strain,
        edge_pairs,
        prepared["boundaries"]["left"],
        prepared["boundaries"]["right"],
        n_nodes=len(node_ids),
    )

    features = GraphFeatureExtractor3D().extract(graph)
    names = (
        "volume_fraction",
        "surface_area_approx",
        "specific_surface",
        "sphericity",
        "compactness",
    )
    with (out / "cylinder_features.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerow({name: features[name] for name in names})

    # Independently total each stored member, with stable edge identity.
    members = []
    for eid, edge in graph.edges.items():
        length = float(
            np.linalg.norm(
                graph.nodes[edge.node_j].position - graph.nodes[edge.node_i].position
            )
        )
        members.append(
            {
                "edge_id": eid,
                "radius_m": edge.radius,
                "chord_length_m": length,
                "cylinder_volume_m3": float(np.pi * edge.radius**2 * length),
                "lateral_area_m2": float(2 * np.pi * edge.radius * length),
            }
        )
    volume = sum(member["cylinder_volume_m3"] for member in members)
    area = sum(member["lateral_area_m2"] for member in members)
    positions = np.array([node.position for node in graph.nodes.values()])
    bbox_volume = float(np.prod(np.ptp(positions, axis=0)))
    np.testing.assert_allclose(
        features["volume_fraction"], min(volume / bbox_volume, 1.0)
    )
    np.testing.assert_allclose(features["surface_area_approx"], area)

    # Legacy VTK represents separate fiber centerlines, not welded graph nodes.
    # For these nondegenerate straight members, conversion retains edge order.
    network = graph.to_fiber_network()
    assert len(network.fibers) == len(graph.edges)
    displacement, ids, radii, forces, recruited = [], [], [], [], []
    for fiber, (eid, edge) in zip(network.fibers, graph.edges.items()):
        count = len(fiber.centerline)
        displacement.extend(
            np.linspace(
                restored.displacements[rows[edge.node_i]],
                restored.displacements[rows[edge.node_j]],
                count,
            )
        )
        ids.extend([eid] * (count - 1))
        radii.extend([edge.radius] * (count - 1))
        forces.extend([restored.edge_forces[edge_rows[eid]]] * (count - 1))
        recruited.extend([recruitment.active_edges[-1, edge_rows[eid]]] * (count - 1))
    to_vtk(
        network,
        out / "member_fields.vtk",
        point_data={"endpoint_interpolated_displacement_m": displacement},
        cell_data={
            "source_edge_id": ids,
            "radius_m": radii,
            "signed_axial_force_N": forces,
            "tensile_recruited": recruited,
        },
    )
    report = {
        "nodes": len(node_ids),
        "edges": len(edge_ids),
        "node_ids": node_ids,
        "edge_ids": edge_ids,
        "units": {"length": "m", "force": "N", "modulus": "Pa"},
        "solver": {
            "backend": "linear 3D BeamFrameFEM",
            "E_Pa": 1e6,
            "nu": 0.3,
            "material_mode": "uniform",
            "stretch_ratio": 1.002,
            "grip_fraction": 0.1,
        },
        "grips": {
            name: [node_ids[i] for i in prepared["boundaries"][name]]
            for name in ("left", "right")
        },
        "max_displacement_m": restored.max_displacement,
        "max_axial_force_N": restored.max_force,
        "tensile_recruited_count": int(recruitment.active_edges[-1].sum()),
        "cylinder_features": {name: features[name] for name in names},
        "summed_cylinder_volume_m3": volume,
        "summed_lateral_area_m2": area,
        "node_bbox_volume_m3": bbox_volume,
        "members": members,
        "scope": "Additive reference-chord cylinders; overlapping junctions double-count. "
        "Lateral area excludes end caps. VTK displacement is endpoint interpolation, "
        "not the beam bending field. Recruitment measures positive chord strain, "
        "not total force flow. Numerical FEM regularization remains unchanged.",
    }
    (out / "analysis.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = run(args.output_dir)
    print(
        json.dumps(
            {key: report[key] for key in ("nodes", "edges", "cylinder_features")},
            indent=2,
        )
    )
