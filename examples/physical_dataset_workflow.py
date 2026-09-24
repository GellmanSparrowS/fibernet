"""Build a resumable physical-label dataset using the library alone.

Run: python -m examples.physical_dataset_workflow --output-dir demo_labels
All targets are outputs of a reduced numerical solver, not experiment.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from fibernet.ml import PlanarPhysicalDataset


class PhysicalDatasetWorkflow:
    def __init__(self, output_dir, unit="hexagon", samples=8, seed=23,
                 grid=2):
        self.output_dir = Path(output_dir)
        self.unit = unit
        self.samples = int(samples)
        self.seed = int(seed)
        self.grid = int(grid)

    def run(self):
        if not 1 <= self.samples <= 200 or not 1 <= self.grid <= 4:
            raise ValueError("samples must be 1..200 and grid must be 1..4")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        dataset = PlanarPhysicalDataset(
            unit=self.unit, seed0=self.seed, grid=self.grid,
            path=self.output_dir / "physical_labels.npz",
            max_nodes=2500, max_edges=3500)
        X, Y = dataset.generate(self.samples)
        if X.shape != (self.samples, 14) or Y.shape != (self.samples, 3):
            raise RuntimeError("dataset did not reach requested sample count")
        summary = {
            "unit": self.unit, "grid": self.grid, "seed0": self.seed,
            "samples": len(X), "next_seed": dataset.next_seed,
            "feature_count": X.shape[1],
            "targets": ["peak", "early_stiffness", "integrated_force"],
            "finite_labels": bool(np.isfinite(Y).all()),
            "engine_source": dataset.config["engine_source"],
            "scope": "reduced-solver labels; not experimental measurements",
        }
        target = self.output_dir / "summary.json"
        temporary = target.with_suffix(".tmp.json")
        try:
            temporary.write_text(json.dumps(summary, indent=2) + "\n",
                                 encoding="utf-8")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        print("[physical_dataset_workflow] checkpoint and summary saved")
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="demo_labels")
    parser.add_argument("--unit", default="hexagon")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--grid", type=int, default=2)
    args = parser.parse_args()
    PhysicalDatasetWorkflow(args.output_dir, args.unit, args.samples,
                            args.seed, args.grid).run()
