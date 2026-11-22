"""Generate, solve and save a small 3D beam graph with two material moduli."""
import argparse
import json
from pathlib import Path

import fibernet as fn
import numpy as np


def run(output):
    """Solve one linear stretch; save the original graph and detailed result."""
    try:
        output.mkdir()  # Atomic admission: never replace prior files.
    except FileExistsError:
        raise SystemExit(f"error: output directory already exists: {output}") from None
    except FileNotFoundError:
        raise SystemExit(f"error: output parent must exist: {output.parent}") from None
    length, radius, stretch = 0.002, 0.00008, 1.001  # m, m, ratio
    soft, stiff = 1e9, 4e9  # Pa
    graph = fn.pattern_3d(unit="cubic", box=(length,) * 3, grid=(2, 1, 1), radius=radius)

    modulus_by_id = {}
    axial_ids = set()
    for edge_id, edge in graph.edges.items():
        x0 = graph.nodes[edge.node_i].position[0]
        x1 = graph.nodes[edge.node_j].position[0]
        if x0 != x1:
            axial_ids.add(edge_id)
        modulus = soft if edge_id in axial_ids and min(x0, x1) < length else stiff
        # Replace the record; do not edit a Material that other edges might share.
        edge.material = fn.Material(youngs_modulus=modulus, poissons_ratio=0.30)
        modulus_by_id[edge_id] = modulus

    # This particular 12-node graph has four nodes on each end face. Check
    # that before using pct; fractional grips are not general cap selectors.
    xs = [node.position[0] for node in graph.nodes.values()]
    if (len(xs), len(graph.edges), len(axial_ids)) != (12, 20, 8) or [
        xs.count(x) for x in (0.0, length, 2 * length)
    ] != [4, 4, 4]:
        raise RuntimeError("Unexpected geometry for the four-chain reference")
    result = fn.simulate(
        graph, backend="fem", stretch_ratio=stretch, nonlinear=False,
        material_mode="graph", length_unit="m", pct=1 / 3,
    )

    material = result.metadata["material_contract"]
    edge_ids = result.metadata["edge_ids"]
    if (material["edge_ids"] != edge_ids
            or dict(zip(edge_ids, material["youngs_modulus"])) != modulus_by_id):
        raise RuntimeError("Result material rows do not match the graph edge IDs")
    # Each of the four x-directed paths has two bars in series. All four
    # paths move identically; transverse ties have zero axial force here.
    area = np.pi * radius**2
    stroke = 2 * length * (stretch - 1)
    path_force = stroke / (length / (soft * area) + length / (stiff * area))
    expected_energy = 0.5 * 4 * path_force * stroke
    np.testing.assert_allclose(
        result.edge_forces, [path_force if eid in axial_ids else 0.0 for eid in edge_ids],
        rtol=1e-7, atol=1e-13,
    )
    energy = result.metadata["beam_energy"]["strain_energy"]
    np.testing.assert_allclose(energy, expected_energy, rtol=1e-9, atol=0.0)
    # A cut at the left ends crosses exactly one rod per axial load path.
    first_span = {eid for eid in axial_ids if modulus_by_id[eid] == soft}
    net_force = sum(float(force) for eid, force in zip(edge_ids, result.edge_forces)
                    if eid in first_span)

    graph.save_json(output / "graph.json")
    result.save(output / "result.json", detailed=True)
    return {
        "fibernet_version": fn.__version__,
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "prescribed_stroke_m": stroke,
        "axial_cut_force_N": net_force,
        "expected_axial_cut_force_N": 4 * path_force,
        "beam_strain_energy_J": energy,
        "expected_strain_energy_J": expected_energy,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="new directory for graph.json and detailed result.json")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.output), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
