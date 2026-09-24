"""Search a target stretch-curve shape with the shared reduced solver.

Run: python -m examples.curve_inverse_workflow --output-dir demo_inverse
This numerical objective is not an experimentally calibrated material target.
"""
import argparse
import json
import os
from pathlib import Path

from fibernet.gen import PlanarManufacturingConfig, manufacturable_graph
from fibernet.ml import CurveInverseDesigner
from fibernet.ml.curve_inverse import metrics_of


class CurveInverseWorkflow:
    def __init__(self, output_dir, unit="hexagon", budget=8, seed=23,
                 stretch=1.3, grid=2):
        self.output_dir = Path(output_dir)
        self.unit = unit
        self.budget = int(budget)
        self.seed = int(seed)
        self.stretch = float(stretch)
        self.grid = int(grid)

    def _build(self, unit, perturbation, line_displacements):
        graph = manufacturable_graph(PlanarManufacturingConfig(
            unit=unit, grid_x=self.grid, grid_y=self.grid,
            n_pts_per_side=2, seed=self.seed,
            perturbation=perturbation,
            line_displacements=line_displacements))
        if graph.num_nodes > 1000 or graph.num_edges > 1500:
            raise MemoryError("inverse example exceeds graph budget")
        return graph

    def run(self):
        if not 2 <= self.budget <= 30 or not 1 <= self.grid <= 4:
            raise ValueError("budget must be 2..30 and grid must be 1..4")
        search = CurveInverseDesigner(
            self._build, budget=self.budget, seed=self.seed,
            stretch=self.stretch)
        result = search.run("J", fixed_unit=self.unit)
        summary = {
            "unit": self.unit, "target": "J", "seed": self.seed,
            "budget": self.budget, "evaluations": result["evaluations"],
            "best_objective": float(result["best_dist"]),
            "best_spec": result["best_spec"],
            "best_metrics": metrics_of(result["best_run"]),
            "scope": "reduced-solver curve objective; not experimental calibration",
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        target = self.output_dir / "summary.json"
        temporary = target.with_suffix(".tmp.json")
        try:
            temporary.write_text(json.dumps(summary, indent=2) + "\n",
                                 encoding="utf-8")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        print("[curve_inverse_workflow] search summary saved")
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="demo_inverse")
    parser.add_argument("--unit", default="hexagon")
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--stretch", type=float, default=1.3)
    parser.add_argument("--grid", type=int, default=2)
    args = parser.parse_args()
    CurveInverseWorkflow(args.output_dir, args.unit, args.budget,
                         args.seed, args.stretch, args.grid).run()
