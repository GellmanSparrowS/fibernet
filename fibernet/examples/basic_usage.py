"""Generate, simulate, analyze, and save a small fiber network.

Run after installing FiberNet, from any directory::

    python -m fibernet.examples.basic_usage --output-dir fibernet_demo

The example uses the base dependencies only. Coordinates and radii are in m,
Young's modulus is in Pa, and forces are in N. The illustrative linear beam
model is not calibrated to an experimental material.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import fibernet as fn
from fibernet.analysis import analyze_topology


def run_example(output_dir="fibernet_demo"):
    """Write a graph, its linear FEM result, a figure, and a JSON summary."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    graph = fn.pattern_2d(
        unit="honeycomb", box=(10e-3, 10e-3), grid=(2, 2),
        radius=0.05e-3, seed=23,
    )
    graph.save_json(str(output / "network.json"))
    figure = fn.show(graph, theme="publication",
                     save_path=str(output / "network.png"))
    import matplotlib.pyplot as plt
    plt.close(figure)

    result = fn.simulate(
        graph, backend="fem", mode="stretch", stretch_ratio=1.01,
        nonlinear=False, E=1e9, nu=0.3,
    )
    result.save(str(output / "fem_result.json"), detailed=True)

    restored_graph = fn.StructureGraph.load_json(str(output / "network.json"))
    restored_result = fn.SimResult.load(str(output / "fem_result.json"))
    np.testing.assert_allclose(restored_result.displacements, result.displacements)
    topology = analyze_topology(restored_graph)
    summary = {
        "nodes": topology.n_nodes,
        "physical_edges": topology.n_physical_edges,
        "connected_components": topology.n_components,
        "cycle_rank": topology.cycle_rank_physical,
        "stretch_ratio": 1.01,
        "max_displacement_m": restored_result.max_displacement,
        "max_member_axial_force_N": restored_result.max_force,
        "model": "small-deformation linear beam FEM",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="fibernet_demo")
    args = parser.parse_args(argv)
    summary = run_example(args.output_dir)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
