"""Independent beam-frame FEM screen for directional tensile recruitment.

Run: python benchmarks/recruitment_fem_validation.py
Results are model comparisons, not calibrated experimental validation.
"""
import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "FiberScope"))

from fslab.structure import StructureFactory
from fibernet.easy import simulate
from fibernet.analysis.tensile_recruitment import analyze_tensile_recruitment


@dataclass
class FemValidationConfig:
    unit: str = "hexagon"
    seeds: tuple = (11, 23, 41)
    directions: tuple = ("x", "y")
    stretches: tuple = (1.08, 1.16, 1.24)
    min_actives: tuple = (0.03, 0.05, 0.10)
    grid: int = 3
    points_per_edge: int = 1
    perturbation: float = 0.04
    max_nodes: int = 500
    max_edges: int = 800


class FemRecruitmentValidation:
    def __init__(self, config=None):
        self.config = config or FemValidationConfig()
        cfg = self.config
        if (cfg.grid < 1 or cfg.max_nodes < 1 or cfg.max_edges < 1 or
                any(direction not in ("x", "y")
                    for direction in cfg.directions) or
                any(stretch <= 1 or stretch > 1.5
                    for stretch in cfg.stretches) or
                any(threshold <= 0 or threshold >= 1
                    for threshold in cfg.min_actives)):
            raise ValueError("invalid FEM validation configuration")

    @staticmethod
    def _source_hash():
        files = (Path(__file__), ROOT / "fibernet" / "ml" /
                 "beam_frame_fem.py", ROOT / "fibernet" / "easy.py",
                 ROOT / "fibernet" / "analysis" /
                 "tensile_recruitment.py")
        digest = hashlib.sha256()
        for file in files:
            digest.update(file.read_bytes())
        return digest.hexdigest()[:16]

    def one(self, seed, direction, stretch):
        cfg = self.config
        graph = StructureFactory(
            unit=cfg.unit, grid_x=cfg.grid, grid_y=cfg.grid,
            n_pts_per_side=cfg.points_per_edge,
            perturbation=cfg.perturbation, seed=seed).build()
        if graph.num_nodes > cfg.max_nodes or graph.num_edges > cfg.max_edges:
            raise MemoryError("FEM validation graph exceeds node/edge budget")
        if direction == "y":
            positions = np.asarray(graph.node_positions(), dtype=float)
            rotated = np.column_stack((positions[:, 1], -positions[:, 0]))
            graph.set_node_positions({i: value
                                      for i, value in enumerate(rotated)})
        geometry = hashlib.sha256()
        geometry.update(np.asarray(graph.node_positions(), dtype=float).tobytes())
        geometry.update(np.asarray(graph.edge_array(), dtype=int).tobytes())
        result = simulate(graph, backend="fem", mode="stretch",
                          strain=stretch, nonlinear=True, pct=0.05)
        positions = np.asarray(graph.node_positions(), dtype=float)[:, :2]
        final = np.asarray(result.deformed_positions, dtype=float)[:, :2]
        edges = np.asarray(graph.edge_array(), dtype=int)[:, :2]
        rest = np.linalg.norm(positions[edges[:, 1]] -
                              positions[edges[:, 0]], axis=1)
        if not np.all(rest > 0):
            raise ValueError("zero-length edge in FEM graph")
        edge_strain = (np.linalg.norm(final[edges[:, 1]] -
                                      final[edges[:, 0]], axis=1) / rest - 1)
        x = positions[:, 0]
        width = .05 * np.ptp(x)
        left = np.flatnonzero(x <= x.min() + width)
        right = np.flatnonzero(x >= x.max() - width)
        summary = {"nodes": int(graph.num_nodes),
                   "edges": int(graph.num_edges),
                   "geometry_hash": geometry.hexdigest()[:16],
                   "max_force": float(result.max_force),
                   "max_edge_strain": float(edge_strain.max()),
                   "thresholds": {}}
        for minimum in cfg.min_actives:
            recruited = analyze_tensile_recruitment(
                edge_strain[None, :], edges, left, right, len(positions),
                alpha=0.05, min_active=minimum)
            summary["thresholds"][format(minimum, ".4f")] = {
                "spanning": bool(recruited.perc_frame >= 0),
                "spanning_edge_fraction": float(recruited.backbone_frac[0]),
                "recruited_edge_fraction": float(recruited.active_edges[0].mean()),
            }
        return summary

    def run(self, output):
        output = Path(output)
        config = json.loads(json.dumps(asdict(self.config)))
        signature = {"config": config, "source_hash": self._source_hash()}
        if output.exists():
            data = json.loads(output.read_text(encoding="utf-8"))
            if any(data.get(key) != value for key, value in signature.items()):
                raise ValueError("FEM checkpoint config or source changed")
        else:
            data = dict(signature, cases={})
        for seed in self.config.seeds:
            for direction in self.config.directions:
                for stretch in self.config.stretches:
                    key = f"seed{seed}:{direction}:stretch{stretch:.3f}"
                    if key in data["cases"]:
                        continue
                    data["cases"][key] = self.one(seed, direction, stretch)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    temporary = output.with_suffix(output.suffix + ".tmp")
                    try:
                        temporary.write_text(json.dumps(data, indent=2) + "\n",
                                             encoding="utf-8")
                        os.replace(temporary, output)
                    finally:
                        temporary.unlink(missing_ok=True)
                    print(f"[fem_validation] {key} completed")
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "benchmarks" / "results" /
                                               "recruitment_fem_validation_parallel.json"))
    args = parser.parse_args()
    FemRecruitmentValidation().run(args.output)
