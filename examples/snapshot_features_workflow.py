"""Analyze a generated network snapshot without importing FiberScope.

Run: python -m examples.snapshot_features_workflow --output-dir demo_features
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from fibernet.analysis import ContactConfig, SnapshotFeatureExtractor
from fibernet.gen import PlanarManufacturingConfig, manufacturable_graph


class SnapshotFeatureWorkflow:
    def __init__(self, unit="ring", grid=2, seed=11, width=0.16,
                 include_contact=False):
        self.unit = unit
        self.grid = int(grid)
        self.seed = int(seed)
        self.width = float(width)
        self.include_contact = bool(include_contact)
        if not 1 <= self.grid <= 4:
            raise ValueError("grid must be 1..4 for this bounded example")

    def run(self, output_dir):
        graph = manufacturable_graph(PlanarManufacturingConfig(
            unit=self.unit, grid_x=self.grid, grid_y=self.grid,
            n_pts_per_side=1, seed=self.seed))
        extractor = SnapshotFeatureExtractor(
            include_contact=self.include_contact,
            contact_config=ContactConfig(width=self.width, resolution=128),
            max_edges=800)
        values = extractor.extract_graph(graph)
        scalars = {key: (value.item() if isinstance(value, np.generic) else value)
                   for key, value in values.items() if np.isscalar(value)}
        arrays = {key: np.asarray(value) for key, value in values.items()
                  if not np.isscalar(value)}
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        summary = {"unit": self.unit, "grid": self.grid, "seed": self.seed,
                   "coordinate_unit": "arbitrary length unit",
                   "include_contact": self.include_contact,
                   "contact_width": self.width if self.include_contact else None,
                   "scalars": scalars,
                   "histogram_file": "snapshot_distributions.npz"}
        npz = folder / "snapshot_distributions.npz"
        temporary = npz.with_suffix(".tmp.npz")
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            os.replace(temporary, npz)
        finally:
            temporary.unlink(missing_ok=True)
        path = folder / "snapshot_summary.json"
        temporary = path.with_suffix(".tmp.json")
        try:
            temporary.write_text(json.dumps(summary, indent=2) + "\n",
                                 encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"[snapshot_features] {graph.num_nodes} nodes, {graph.num_edges} edges")
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="demo_features")
    parser.add_argument("--unit", default="ring")
    parser.add_argument("--grid", type=int, default=2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--width", type=float, default=0.16)
    parser.add_argument("--contact", action="store_true")
    arguments = parser.parse_args()
    SnapshotFeatureWorkflow(arguments.unit, arguments.grid, arguments.seed,
                            arguments.width, arguments.contact).run(
                                arguments.output_dir)
