"""Import an OBJ and compile a continuous curved fiber route.

Run: python -m examples.obj_surface_workflow input.obj --output-dir curved_route
The OBJ must be a surface mesh; output coordinates retain its input units.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from fibernet.gen import PlanarManufacturingConfig, compile_surface, load_obj


class ObjSurfaceWorkflow:
    def __init__(self, obj_path, output_dir, target_faces=500):
        self.obj_path = Path(obj_path)
        self.output_dir = Path(output_dir)
        self.target_faces = int(target_faces)

    def run(self):
        vertices, faces, info = load_obj(
            self.obj_path, return_info=True, target_faces=self.target_faces)
        config = PlanarManufacturingConfig(
            unit="kagome", grid_x=1, grid_y=1, n_pts_per_side=2,
            line_displacements=[[0.02, -0.01], [0.01, 0.03]], seed=23)
        network = compile_surface(vertices, faces, config)
        if (network.health["components"] != 1 or network.health["odd"] != 0 or
                len(network.route_edges) != len(network.edges)):
            raise AssertionError("curved route is not continuous and closed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        target = self.output_dir / "curved_route.npz"
        temporary = target.with_suffix(".tmp.npz")
        try:
            np.savez_compressed(temporary, positions=network.positions,
                                edges=network.edges,
                                route_nodes=network.route_nodes,
                                route_edges=network.route_edges)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        summary = {
            "source": str(self.obj_path), "input": info,
            "nodes": len(network.positions), "edges": len(network.edges),
            "topology_id": network.topology_id,
            "route_closed": bool(network.route_nodes[0] == network.route_nodes[-1]),
            "scope": "geometry and Euler route; printability is not certified",
        }
        target = self.output_dir / "summary.json"
        temporary = target.with_suffix(".tmp.json")
        try:
            temporary.write_text(json.dumps(summary, indent=2) + "\n",
                                 encoding="utf-8")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        print("[obj_surface_workflow] curved route saved")
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("obj_path")
    parser.add_argument("--output-dir", default="curved_route")
    parser.add_argument("--target-faces", type=int, default=500)
    args = parser.parse_args()
    ObjSurfaceWorkflow(args.obj_path, args.output_dir,
                       args.target_faces).run()
