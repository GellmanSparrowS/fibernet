"""Map repeated fiber cells to an OBJ surface with explicit seam stitches.

Run: python -m examples.surface_mapping_workflow input.obj --output-dir mapped_cells
This produces a stitched segment network, not a closed printer route.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from fibernet.gen import MappingConfig, load_obj, map_cells


class SurfaceMappingWorkflow:
    def __init__(self, obj_path, output_dir, unit="hexagon",
                 target_faces=500):
        self.obj_path = Path(obj_path)
        self.output_dir = Path(output_dir)
        self.unit = unit
        self.target_faces = int(target_faces)

    def run(self):
        vertices, faces, info = load_obj(
            self.obj_path, return_info=True, target_faces=self.target_faces)
        mapping = MappingConfig(max_points=150000, max_segments=200000)
        points, segments = map_cells(
            vertices, faces, [[0.02, -0.01], [0.01, 0.03]],
            unit=self.unit, config=mapping)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        target = self.output_dir / "mapped_cells.npz"
        temporary = target.with_suffix(".tmp.npz")
        try:
            np.savez_compressed(temporary, points=points, segments=segments)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        summary = {"source": str(self.obj_path), "input": info,
                   "unit": self.unit, "mapped_faces": mapping.mapped_faces,
                   "points": len(points), "segments": len(segments),
                   "scope": "stitched geometry; not a closed printer route"}
        target = self.output_dir / "summary.json"
        temporary = target.with_suffix(".tmp.json")
        try:
            temporary.write_text(json.dumps(summary, indent=2) + "\n",
                                 encoding="utf-8")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        print("[surface_mapping_workflow] mapped network saved")
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("obj_path")
    parser.add_argument("--output-dir", default="mapped_cells")
    parser.add_argument("--unit", default="hexagon")
    parser.add_argument("--target-faces", type=int, default=500)
    args = parser.parse_args()
    SurfaceMappingWorkflow(args.obj_path, args.output_dir,
                           args.unit, args.target_faces).run()
