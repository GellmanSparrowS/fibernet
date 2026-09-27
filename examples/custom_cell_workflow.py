"""Persist an APP-compatible custom unit and compile its closed planar route.

Run: python -m examples.custom_cell_workflow --output-dir demo_custom_cell
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from fibernet.gen import (CustomCell, CustomCellRegistry,
                          PlanarManufacturingConfig, compile_planar)


class CustomCellWorkflow:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)

    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        spec = {"nodes": [[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]],
                "edges": [[0, 1], [1, 2], [2, 3], [3, 0],
                          [0, 4], [1, 4], [2, 4], [3, 4]],
                "zh": "中心连接", "en": "Center-connected"}
        registry = CustomCellRegistry(self.output_dir / "custom_units.json")
        key = registry.save("center-connected", CustomCell.from_mapping(spec))
        cell = CustomCellRegistry(registry.path).load()[key]
        network = compile_planar(PlanarManufacturingConfig(
            unit=key, grid_x=2, grid_y=2, n_pts_per_side=1,
            base_graph=cell.to_graph()))
        if network.health["components"] != 1 or network.health["odd"] != 0:
            raise AssertionError("custom route is not connected and Eulerian")
        if network.route_nodes[0] != network.route_nodes[-1]:
            raise AssertionError("custom route is not closed")
        destination = self.output_dir / "custom_route.npz"
        temporary = destination.with_suffix(".tmp.npz")
        try:
            np.savez_compressed(temporary, reference=network.reference,
                                positions=network.positions, edges=network.edges,
                                route_nodes=network.route_nodes,
                                route_edges=network.route_edges)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        summary = {"unit": key, "nodes": len(network.positions),
                   "edges": len(network.edges),
                   "route_closed": True,
                   "scope": "topological route only; physical printability is not verified"}
        destination = self.output_dir / "summary.json"
        temporary = destination.with_suffix(".tmp.json")
        try:
            temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        print("[custom_cell_workflow] APP-compatible cell and route saved")
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="demo_custom_cell")
    args = parser.parse_args()
    CustomCellWorkflow(args.output_dir).run()
