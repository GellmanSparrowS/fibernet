"""One four-node linear FEM example, with explicit loads, IDs and diagnostics.

Run from the checkout: python -m examples.linear_fem_quickstart --output-dir NEWDIR
Redraw saved evidence: add --plot-only with its existing output directory.
Only base dependencies are needed. Importing this module runs no experiment.
"""
import argparse
import json
from pathlib import Path


def run(output_dir):
    """Solve once, save/reload evidence, and draw the declared loading setup."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)  # Refuse even an empty existing folder.

    import numpy as np

    import fibernet as fn
    from fibernet.ml.beam_frame_fem import BeamFrameFEM

    # Caller declares SI units; FiberNet does not convert or infer these units.
    side_m, radius_m = 0.01, 0.0001
    E_Pa, nu, force_x_N = 1e6, 0.3, 1e-5
    coefficient = 1e-6  # Numerical diagonal regularization, not physical support.
    graph = fn.pattern_2d(
        unit="square", box=(side_m, side_m), grid=(1, 1),
        n_pts_per_side=0, radius=radius_m, seed=23,
        material=fn.Material(name="quickstart_linear", youngs_modulus=E_Pa,
                             poissons_ratio=nu))
    assert graph.num_nodes == 4 and graph.num_edges == 4

    # to_numpy sorts node IDs but preserves edge insertion order.
    positions, edges, radii = graph.to_numpy()  # (N, 3), (M, 2), (M,)
    node_ids, edge_ids = sorted(graph.nodes), list(graph.edges)
    node_row = {node_id: row for row, node_id in enumerate(node_ids)}
    left_ids = [nid for nid in node_ids
                if np.isclose(graph.nodes[nid].position[0], 0.0)]
    right_ids = [nid for nid in node_ids
                 if np.isclose(graph.nodes[nid].position[0], side_m)]
    assert len(left_ids) == len(right_ids) == 2
    loads_by_node_id = {nid: [force_x_N, 0.0] for nid in right_ids}
    clamps_by_node_id = {nid: {"ux_m": 0.0, "uy_m": 0.0, "rz_rad": 0.0}
                        for nid in left_ids}

    # Solver inputs use ROW indices, not stable graph IDs.
    forces = np.zeros((len(node_ids), 2))  # Exact (N, 2): [Fx, Fy], in N.
    for nid, force in loads_by_node_id.items():
        forces[node_row[nid]] = force
    fixed_rows = [node_row[nid] for nid in clamps_by_node_id]
    assert forces.shape == (4, 2)  # No rotational/nodal-moment load column.
    experiment = {
        "description": "One linear welded-joint beam-frame square; API example only",
        "units": {"coordinates": "m", "radius": "m", "E": "Pa", "nu": "1",
                  "translation": "m", "rotation": "rad", "force": "N", "moment": "N m"},
        "geometry": {"unit": "square", "grid": [1, 1], "side_m": side_m,
                     "radius_m": radius_m, "n_pts_per_side": 0, "seed": 23},
        "material": {"E_Pa": E_Pa, "nu": nu,
                     "source": "explicit global BeamFrameFEM constructor; graph Material is descriptive"},
        "solver": {"method": "BeamFrameFEM.solve_2d", "nonlinear": False,
                   "deduplicate": False, "damping": coefficient,
                   "damping_meaning": "numerical mixed-equation coefficient, not a physical spring"},
        "input_node_ids": node_ids, "input_edge_ids": edge_ids,
        "loads_N_by_node_id": loads_by_node_id,
        "clamps_by_node_id": clamps_by_node_id,
        "boundary_meaning": "left nodes: ux=uy=rz=0; right nodes: free ux,uy,rz",
        "force_array_shape": list(forces.shape), "force_array_N": forces.tolist(),
        "fixed_input_node_rows": fixed_rows,
        "id_key_note": "JSON object keys are strings; convert to int when reading node IDs",
    }
    graph.save_json(out / "graph.json")
    (out / "experiment.json").write_text(
        json.dumps(experiment, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    # fn.simulate(..., backend='fem') is a stretch-only convenience route.
    # This public module route intentionally exposes actual force/BC inputs.
    fem = BeamFrameFEM(E=E_Pa, nu=nu)
    raw = fem.solve_2d(edges.T, positions[:, :2], radii,
                       forces=forces, fixed_nodes=fixed_rows,
                       damping=coefficient, deduplicate=False)
    result = fem.to_sim_result(raw, graph, node_ids=node_ids, edge_ids=edge_ids)
    result.save(out / "result.json", detailed=True)

    # Reload without a second solve. Check identity, values and diagnostics.
    restored_graph = fn.StructureGraph.load_json(out / "graph.json")
    restored = fn.SimResult.load(out / "result.json")
    assert sorted(restored_graph.nodes) == node_ids
    assert set(restored_graph.edges) == set(edge_ids)
    for nid in node_ids:
        np.testing.assert_array_equal(restored_graph.nodes[nid].position,
                                      graph.nodes[nid].position)
    for eid in edge_ids:
        before, after = graph.edges[eid], restored_graph.edges[eid]
        assert (after.node_i, after.node_j, after.radius) == (
            before.node_i, before.node_j, before.radius)
    assert restored.metadata["node_ids"] == node_ids
    assert restored.metadata["edge_ids"] == [edge_ids[i] for i in raw["edge_list"]]
    np.testing.assert_array_equal(restored.displacements, result.displacements)
    np.testing.assert_array_equal(restored.edge_forces, result.edge_forces)
    diagnostic = restored.metadata["numerical_diagnostics"]
    assert diagnostic == result.metadata["numerical_diagnostics"]
    residual = diagnostic["physical_free_residual"]
    raw_residual = raw["numerical_diagnostics"]["physical_free_residual"]
    assert residual["status"] == "available"
    assert residual["index_kind"] == "stable_node_id"
    assert residual["node_ids"] == [node_ids[row] for row in raw_residual["node_rows"]]
    assert residual["values"] == raw_residual["values"]
    assert set(residual["node_ids"]) == set(right_ids)
    assert len(residual["values"]) == 6  # ux, uy, rz at each of two free nodes.
    assert diagnostic["regularization"]["coefficient"] == coefficient
    for name in ("physical_equilibrium", "load_compatibility", "solution_uniqueness"):
        assert diagnostic[name] == "not_evaluated"

    # Never combine translational forces and rotational moments into one norm.
    force_residuals = [v for c, v in zip(residual["dof_components"], residual["values"])
                       if c in ("ux", "uy")]
    moment_residuals = [v for c, v in zip(residual["dof_components"], residual["values"])
                        if c == "rz"]
    summary = {
        "nodes": graph.num_nodes, "edges": graph.num_edges,
        "input_node_ids": node_ids, "input_edge_ids": edge_ids,
        "clamped_node_ids": left_ids, "loaded_node_ids": right_ids,
        "force_per_loaded_node_N": [force_x_N, 0.0],
        "max_translation_m": restored.max_displacement,
        "max_abs_free_force_residual_N": float(np.max(np.abs(force_residuals))),
        "max_abs_free_moment_residual_N_m": float(np.max(np.abs(moment_residuals))),
        "physical_free_residual_status": residual["status"],
        "regularization": diagnostic["regularization"],
        "physical_equilibrium": diagnostic["physical_equilibrium"],
        "load_compatibility": diagnostic["load_compatibility"],
        "solution_uniqueness": diagnostic["solution_uniqueness"],
        "reload_checks": "graph/result IDs, displacements, axial forces and diagnostics equal",
        "scope": "software workflow check; no stability, uniqueness or physical-validity verdict",
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    draw_from_saved(out)
    return summary


def draw_from_saved(output_dir):
    """Draw A4 portrait PNG/SVG from four saved JSON files; never solve/generate.

    Only the two image files are replaced. Saved experiment/results stay intact.
    This is a renderer for this example's setup, not a general experiment loader.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import fibernet as fn

    out = Path(output_dir)
    restored_graph = fn.StructureGraph.load_json(out / "graph.json")
    restored = fn.SimResult.load(out / "result.json")
    experiment = json.loads((out / "experiment.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    node_ids = experiment["input_node_ids"]
    loads_by_node_id = {int(nid): value for nid, value in experiment["loads_N_by_node_id"].items()}
    clamps_by_node_id = {int(nid): value for nid, value in experiment["clamps_by_node_id"].items()}
    left_ids = list(clamps_by_node_id)
    side_m, radius_m = experiment["geometry"]["side_m"], experiment["geometry"]["radius_m"]
    E_Pa, nu = experiment["material"]["E_Pa"], experiment["material"]["nu"]
    coefficient = experiment["solver"]["damping"]
    diagnostic = restored.metadata["numerical_diagnostics"]
    residual = diagnostic["physical_free_residual"]
    assert restored.metadata["node_ids"] == node_ids
    assert restored.max_displacement == summary["max_translation_m"]
    assert coefficient == diagnostic["regularization"]["coefficient"]
    for name in ("physical_equilibrium", "load_compatibility", "solution_uniqueness"):
        assert diagnostic[name] == summary[name] == "not_evaluated"

    fig = plt.figure(figsize=(210 / 25.4, 297 / 25.4))  # A4 portrait, no crop.
    ax = fig.add_axes([0.10, 0.39, 0.84, 0.50])
    fn.render_graph(restored_graph, ax=ax, theme="publication", line_width=3,
                    show_nodes=True, node_size=40, show_boundary=False, tight=False)
    fig.suptitle("Linear FEM: forces and clamps", x=0.10, y=0.96,
                 ha="left", fontsize=20, fontweight="bold", color="#17283b")
    fig.text(0.10, 0.925,
             f"Synthetic API example  |  {side_m * 1000:g} mm square, "
             f"{restored_graph.num_nodes} nodes, {restored_graph.num_edges} beams",
             fontsize=10.5, color="#4b5967")
    for nid in node_ids:
        x, y = restored_graph.nodes[nid].position[:2]
        ax.annotate(f"ID {nid}", (x, y),
                    xytext=(10 if nid in left_ids else -10, 12 if y < side_m / 2 else -19),
                    textcoords="offset points", ha="left" if nid in left_ids else "right",
                    fontsize=11, color="#17283b")
    # These overlays use the actual declared solver BCs, not graph.boundary flags.
    for nid in clamps_by_node_id:
        x, y = restored_graph.nodes[nid].position[:2]
        ax.plot([x - 0.0003] * 2, [y - 0.0006, y + 0.0006], color="#246a92", lw=2)
        for dy in (-0.0005, 0.0, 0.0005):
            ax.plot([x - 0.00065, x - 0.0003], [y + dy - 0.0002, y + dy],
                    color="#246a92", lw=1)
        ax.scatter([x], [y], marker="s", s=70, facecolor="white",
                   edgecolor="#246a92", linewidth=2, zorder=7)
    for nid, (fx, fy) in loads_by_node_id.items():
        assert fx > 0 and fy == 0  # This example's explicit +x loading setup.
        x, y = restored_graph.nodes[nid].position[:2]
        ax.annotate("", xy=(x + 0.0026, y), xytext=(x, y),
                    arrowprops={"arrowstyle": "-|>", "color": "#b63f28", "lw": 2,
                                "mutation_scale": 16})
        ax.text(x + 0.0013, y + (0.00065 if y > side_m / 2 else -0.00085),
                f"Fx = {fx:.0e} N", color="#b63f28", ha="center", fontsize=10)
    ax.text(0.0002, side_m / 2, "Left nodes clamped\nux = uy = rz = 0",
            color="#246a92", fontsize=11, va="center")
    ax.set_xlim(-0.0012, 0.014)
    ax.set_ylim(-0.0015, 0.0115)
    ax.set_xticks([0, 0.005, 0.01])
    ax.set_yticks([0, 0.005, 0.01])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.text(0.10, 0.33,
             "Undeformed geometry; arrows show the declared forces.\n"
             "Arrow lengths are schematic. Clamps mark the actual solver boundary conditions.",
             fontsize=9, color="#4b5967", linespacing=1.5)
    fig.text(0.10, 0.275, "Parameters", fontsize=12, fontweight="bold", color="#17283b")
    fig.text(0.53, 0.275, "Numerical diagnostics", fontsize=12, fontweight="bold", color="#17283b")
    fig.text(0.10, 0.25,
             f"Caller-declared SI units\nGlobal E = {E_Pa:.0e} Pa\nPoisson ratio = {nu:g}\n"
             f"Radius = {radius_m:.0e} m\nNumerical coefficient = {coefficient:.0e}\n"
             "Coefficient is not physical support.",
             va="top", fontsize=10, linespacing=1.5, color="#17283b")
    fig.text(0.53, 0.25,
             f"Max translation: {summary['max_translation_m']:.6e} m\n"
             f"Max |free force residual|\n{summary['max_abs_free_force_residual_N']:.3e} N\n"
             f"Max |free moment residual|\n{summary['max_abs_free_moment_residual_N_m']:.3e} N m\n"
             f"Residual data: {residual['status']}",
             va="top", fontsize=10, linespacing=1.5, color="#17283b")
    fig.text(0.10, 0.09,
             "Equilibrium / load compatibility / solution uniqueness: not_evaluated\n"
             "Reload assertions check persistence. No physical-validity verdict is made.",
             fontsize=9, color="#755529", linespacing=1.5)
    for extension in ("png", "svg"):
        fig.savefig(out / f"linear_fem_quickstart.{extension}", dpi=160,
                    facecolor=fig.get_facecolor())
    plt.close(fig)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True,
                        help="new directory for a solve; existing saved directory for --plot-only")
    parser.add_argument("--plot-only", action="store_true",
                        help="redraw only PNG/SVG from saved JSON; do not generate or solve")
    args = parser.parse_args()
    action = draw_from_saved if args.plot_only else run
    print(json.dumps(action(args.output_dir), indent=2, allow_nan=False))
