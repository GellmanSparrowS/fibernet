"""Run the shared reduced stretch solver and save a recruitment trajectory.

Run: python -m examples.reduced_recruitment_workflow --output-dir demo_stretch
The solver is a reduced spring/bending/contact model, not calibrated FEM.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np

from fibernet.gen import PlanarManufacturingConfig, manufacturable_graph
from fibernet.sim import ReducedBeamConfig, ReducedBeamSolver
from fibernet.analysis import compute_percolation


class ReducedRecruitmentWorkflow:
    def __init__(self, output_dir, unit="kagome", grid=3, seed=23,
                 stretch=1.4, alpha=0.05):
        self.output_dir = Path(output_dir)
        self.unit = unit
        self.grid = int(grid)
        self.seed = int(seed)
        self.stretch = float(stretch)
        self.alpha = float(alpha)

    def run(self):
        if not 1 <= self.grid <= 8:
            raise ValueError("grid must be 1..8 for this example")
        graph_config = PlanarManufacturingConfig(
            unit=self.unit, grid_x=self.grid, grid_y=self.grid,
            n_pts_per_side=2, perturbation=0.25, seed=self.seed)
        graph = manufacturable_graph(graph_config)
        if graph.num_nodes > 2500 or graph.num_edges > 3500:
            raise MemoryError("example network exceeds node/edge budget")
        cfg = ReducedBeamConfig(target_stretch=self.stretch, n_increments=20,
                                num_steps=2000, use_contact=False,
                                use_bending=True)
        result = ReducedBeamSolver(graph, cfg).run()
        recruitment_config = dict(alpha=self.alpha, hysteresis=0.6,
                                  grip_connected=True, quantile=None,
                                  min_active=1e-4)
        recruited = compute_percolation(result, **recruitment_config)
        graph_data = graph.to_dict()
        # Native serialization sorts edge IDs; the solver uses insertion order.
        # Preserve that order for a native from_dict -> solver replay as well.
        edge_records = {edge["id"]: edge for edge in graph_data["edges"]}
        graph_data["edges"] = [edge_records[eid]
                               for eid in result.metadata["edge_ids"]]
        json_records = {
            "graph_json": graph_data,
            "graph_config_json": asdict(graph_config),
            "solver_config_json": asdict(cfg),
            "recruitment_config_json": recruitment_config,
            "result_metadata_json": result.metadata,
        }
        # Unicode scalars are readable with allow_pickle=False. Reject non-JSON
        # numbers before touching either output, rather than storing objects.
        replay = {key: json.dumps(value, allow_nan=False)
                  for key, value in json_records.items()}
        replay.update({"energy_" + key: values
                       for key, values in result.energies.items()})
        self.output_dir.mkdir(parents=True, exist_ok=True)
        target = self.output_dir / "trajectory.npz"
        temporary = target.with_suffix(".tmp.npz")
        try:
            np.savez_compressed(temporary, frames_xy=result.frames_xy,
                                edges=result.edges,
                                edge_strain=result.edge_strain,
                                force_curve=result.force_curve,
                                stretch_levels=result.strain_levels,
                                active_edges=recruited.active_edges,
                                spanning_edges=recruited.edge_in_spanning,
                                trajectory_schema_version=np.int64(1),
                                node_ids=np.asarray(result.metadata["node_ids"],
                                                    dtype=np.int64),
                                edge_ids=np.asarray(result.metadata["edge_ids"],
                                                    dtype=np.int64),
                                left_nodes=result.left_nodes,
                                right_nodes=result.right_nodes,
                                edge_rest=result.edge_rest,
                                spanning_frac=recruited.spanning_frac,
                                backbone_frac=recruited.backbone_frac,
                                edge_depth_norm=recruited.edge_depth_norm,
                                threshold_curve=recruited.threshold_curve,
                                first_spanning_frame=np.int64(recruited.perc_frame),
                                **replay)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        summary = {
            "unit": self.unit, "grid": self.grid, "seed": self.seed,
            "nodes": graph.num_nodes, "edges": graph.num_edges,
            "frames": result.n_frames, "alpha": self.alpha,
            "first_spanning_frame": int(recruited.perc_frame),
            "final_raw_reaction": float(result.force_curve[-1]),
            "scope": "positive axial strain recruitment in a reduced solver; "
                     "not complete force flow or experimental calibration",
        }
        target = self.output_dir / "summary.json"
        temporary = target.with_suffix(".tmp.json")
        try:
            temporary.write_text(json.dumps(summary, indent=2) + "\n",
                                 encoding="utf-8")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        print("[reduced_recruitment_workflow] trajectory saved")
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="demo_stretch")
    parser.add_argument("--unit", default="kagome")
    parser.add_argument("--grid", type=int, default=3)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--stretch", type=float, default=1.4)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    ReducedRecruitmentWorkflow(args.output_dir, args.unit, args.grid,
                               args.seed, args.stretch, args.alpha).run()
