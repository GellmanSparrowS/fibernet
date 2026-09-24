"""Resumable multi-topology screen of strain-threshold recruitment.

Run: python benchmarks/recruitment_sensitivity.py --output benchmarks/results/recruitment_sensitivity.json
This is an observational solver screen. Threshold recruitment is not a
measurement of all force pathways or proof that an edge can be removed.
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

from fslab.engine2 import Engine2, Engine2Config
from fslab.simcache import ENGINE_SRC_HASH, ENGINE_VERSION
from fslab.structure import StructureFactory
from fibernet.analysis.tensile_recruitment import compute_percolation


@dataclass
class SensitivityConfig:
    units: tuple = ("square", "hexagon", "ring", "kagome")
    seeds: tuple = (11, 23, 41)
    directions: tuple = ("x", "y")
    contact: tuple = (False, True)
    alphas: tuple = (0.02, 0.05, 0.10)
    min_actives: tuple = (0.0001, 0.03, 0.05, 0.10)
    grid: int = 3
    points_per_edge: int = 1
    line_displacements: tuple = ()
    perturbation: float = 0.04
    target_stretch: float = 1.4
    increments: int = 20
    steps: int = 2000
    max_nodes: int = 3000
    max_edges: int = 5000


class RecruitmentSensitivity:
    def __init__(self, config=None):
        self.config = config or SensitivityConfig()
        cfg = self.config
        if (cfg.grid < 1 or cfg.steps < 1 or cfg.increments < 1 or
                cfg.max_nodes < 1 or cfg.max_edges < 1 or
                not 0 <= cfg.perturbation <= 0.2 or
                not all(0 <= alpha <= 1 for alpha in cfg.alphas) or
                not all(value > 0 and np.isfinite(value)
                        for value in cfg.min_actives) or
                (cfg.line_displacements and
                 len(cfg.line_displacements) != cfg.points_per_edge) or
                any(d not in ("x", "y") for d in cfg.directions)):
            raise ValueError("invalid sensitivity configuration")

    @staticmethod
    def _signature():
        files = (Path(__file__), ROOT / "fibernet" / "analysis" /
                 "tensile_recruitment.py")
        digest = hashlib.sha256()
        for file in files:
            digest.update(file.read_bytes())
        return digest.hexdigest()[:16]

    def one(self, unit, seed, direction, contact):
        cfg = self.config
        graph = StructureFactory(
            unit=unit, grid_x=cfg.grid, grid_y=cfg.grid,
            n_pts_per_side=cfg.points_per_edge, perturbation=cfg.perturbation,
            line_displacements=(list(cfg.line_displacements)
                                if cfg.line_displacements else None),
            seed=seed).build()
        if graph.num_nodes > cfg.max_nodes or graph.num_edges > cfg.max_edges:
            raise MemoryError("sensitivity graph exceeds node/edge budget")
        if direction == "y":
            pos = np.asarray(graph.node_positions(), dtype=float)
            rotated = np.column_stack((pos[:, 1], -pos[:, 0]))
            graph.set_node_positions({i: p for i, p in enumerate(rotated)})
        geometry = hashlib.sha256()
        geometry.update(np.asarray(graph.node_positions(), dtype=float).tobytes())
        geometry.update(np.asarray(graph.edge_array(), dtype=int).tobytes())
        run = Engine2(graph, Engine2Config(
            target_stretch=cfg.target_stretch, num_steps=cfg.steps,
            n_increments=cfg.increments, use_contact=contact,
            use_bending=True)).run()
        result = {
            "nodes": int(graph.num_nodes), "edges": int(graph.num_edges),
            "geometry_hash": geometry.hexdigest()[:16],
            "frames": int(run.n_frames),
            "final_raw_reaction": float(run.force_curve[-1]),
            "work_proxy": float(np.trapz(run.force_curve, run.strain_levels)),
            "final_axial_energy": float(run.energies["axial"][-1]),
            "final_bend_energy": float(run.energies["bend"][-1]),
            "final_contact_energy": float(run.energies["contact"][-1]),
            "max_contact_pairs": int(run.contact_counts.max()),
            "thresholds": {}, "absolute_thresholds": {},
        }
        for alpha in cfg.alphas:
            recruited = compute_percolation(run, alpha=alpha)
            never = ~recruited.active_edges.any(axis=0)
            key = format(alpha, ".3f")
            result["thresholds"][key] = {
                "never_recruited_count": int(never.sum()),
                "never_recruited_fraction": float(never.mean()),
                "never_recruited_length_fraction": float(
                    run.edge_rest[never].sum() / run.edge_rest.sum()),
                "first_spanning_frame": int(recruited.perc_frame),
                "final_spanning_edge_fraction": float(
                    recruited.backbone_frac[-1]),
            }
        for minimum in cfg.min_actives:
            recruited = compute_percolation(run, alpha=0.05,
                                            min_active=minimum)
            never = ~recruited.active_edges.any(axis=0)
            key = format(minimum, ".4f")
            frame = int(recruited.perc_frame)
            result["absolute_thresholds"][key] = {
                "never_recruited_fraction": float(never.mean()),
                "first_spanning_frame": frame,
                "first_spanning_stretch": (
                    float(run.strain_levels[frame]) if frame >= 0 else None),
                "final_spanning_edge_fraction": float(
                    recruited.backbone_frac[-1]),
            }
        return result

    def run(self, output):
        output = Path(output)
        cfg = json.loads(json.dumps(asdict(self.config)))
        signature = {"config": cfg, "engine_version": ENGINE_VERSION,
                     "engine_hash": ENGINE_SRC_HASH,
                     "analysis_hash": self._signature()}
        if output.exists():
            data = json.loads(output.read_text(encoding="utf-8"))
            if any(data.get(key) != value for key, value in signature.items()):
                raise ValueError("checkpoint configuration or source changed")
        else:
            data = dict(signature, cases={})
        for unit in self.config.units:
            for seed in self.config.seeds:
                for direction in self.config.directions:
                    for contact in self.config.contact:
                        key = f"{unit}:seed{seed}:{direction}:contact{int(contact)}"
                        if key in data["cases"]:
                            continue
                        case = self.one(unit, seed, direction, contact)
                        data["cases"][key] = case
                        output.parent.mkdir(parents=True, exist_ok=True)
                        temporary = output.with_suffix(output.suffix + ".tmp")
                        try:
                            temporary.write_text(json.dumps(data, indent=2) + "\n",
                                                 encoding="utf-8")
                            os.replace(temporary, output)
                        finally:
                            temporary.unlink(missing_ok=True)
                        print(f"[sensitivity] {key} completed")
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "benchmarks" / "results" /
                                               "recruitment_sensitivity.json"))
    args = parser.parse_args()
    RecruitmentSensitivity().run(args.output)
