"""Topology-held-out early recruitment screen on the shared reduced solver.

Run: python benchmarks/early_recruitment_prediction.py
Outputs are numerical screening data, not experimental force-flow labels.
Each case is checkpointed atomically and invalidated by configuration/source.
"""
import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fibernet.gen import PlanarManufacturingConfig, manufacturable_graph
from fibernet.sim import ReducedBeamConfig, ReducedBeamSolver


ENGINE_FILE = ROOT / "fibernet" / "sim" / "reduced_beam.py"


@dataclass
class PredictionConfig:
    units: tuple = ("square", "hexagon", "ring", "kagome")
    seeds: tuple = (11, 23, 41)
    directions: tuple = ("x", "y")
    grid: int = 2
    points_per_edge: int = 1
    perturbation: float = 0.04
    use_contact: bool = False
    target_stretch: float = 1.4
    increments: int = 20
    steps: int = 2000
    early_last_frame: int = 2
    future_min_strain: float = 0.05
    max_nodes: int = 3000
    max_edges: int = 5000


class EarlyRecruitmentPrediction:
    def __init__(self, config=None):
        self.config = config or PredictionConfig()
        cfg = self.config
        if (cfg.grid < 1 or cfg.grid > 5 or cfg.steps < 1 or
                cfg.points_per_edge < 1 or cfg.points_per_edge > 5 or
                not 0 <= cfg.perturbation <= 0.3 or
                cfg.increments < 2 or not 0 < cfg.early_last_frame < cfg.increments or
                cfg.future_min_strain <= 0 or not np.isfinite(cfg.future_min_strain) or
                cfg.max_nodes < 1 or cfg.max_edges < 1 or
                any(d not in ("x", "y") for d in cfg.directions)):
            raise ValueError("invalid prediction configuration")
        self.signature = hashlib.sha256(
            json.dumps(asdict(cfg), sort_keys=True).encode() +
            ENGINE_FILE.read_bytes() + Path(__file__).read_bytes()
        ).hexdigest()[:16]

    @staticmethod
    def _features(graph, run, early_last_frame):
        xy = np.asarray(graph.node_positions(), float)[:, :2]
        edges = np.asarray(graph.edge_array(), int)
        displacement = xy[edges[:, 1]] - xy[edges[:, 0]]
        length = np.linalg.norm(displacement, axis=1)
        if np.any(length <= 0):
            raise ValueError("zero-length edge")
        unit = displacement / length[:, None]
        degree = np.bincount(edges.ravel(), minlength=len(xy))
        midpoint = (xy[edges[:, 0]] + xy[edges[:, 1]]) * 0.5
        bounds = np.maximum(np.ptp(xy, axis=0), 1e-12)
        midpoint = (midpoint - xy.min(axis=0)) / bounds
        length = length / max(float(np.median(length)), 1e-12)
        static = np.column_stack((midpoint, np.abs(unit), length,
                                  degree[edges[:, 0]], degree[edges[:, 1]]))
        early = np.maximum(np.asarray(run.edge_strain[1:early_last_frame + 1], float), 0)
        early_max = early.max(axis=0)
        early_delta = early[-1] - early[0]
        X = np.column_stack((static, early_max, early_delta))
        if not np.isfinite(X).all():
            raise ValueError("non-finite edge features")
        return X, early_max

    def one(self, unit, seed, direction):
        cfg = self.config
        graph = manufacturable_graph(PlanarManufacturingConfig(
            unit=unit, grid_x=cfg.grid, grid_y=cfg.grid,
            n_pts_per_side=cfg.points_per_edge,
            perturbation=cfg.perturbation, seed=int(seed)))
        if graph.num_nodes > cfg.max_nodes or graph.num_edges > cfg.max_edges:
            raise MemoryError("prediction case exceeds graph budget")
        if direction == "y":
            xy = np.asarray(graph.node_positions(), float)
            rotated = np.column_stack((xy[:, 1], -xy[:, 0]))
            graph.set_node_positions({i: point for i, point in enumerate(rotated)})
        run = ReducedBeamSolver(graph, ReducedBeamConfig(
            target_stretch=cfg.target_stretch, n_increments=cfg.increments,
            num_steps=cfg.steps, use_contact=cfg.use_contact,
            use_bending=True)).run()
        X, early = self._features(graph, run, cfg.early_last_frame)
        future_max = np.maximum(np.asarray(run.edge_strain[cfg.early_last_frame + 1:], float), 0).max(axis=0)
        y = (future_max >= cfg.future_min_strain).astype(np.uint8)
        geometry = hashlib.sha256(np.asarray(graph.node_positions(), float).tobytes() +
                                  np.asarray(graph.edge_array(), int).tobytes()).hexdigest()[:16]
        return X, y, early, future_max, geometry

    def _case_path(self, folder, unit, seed, direction):
        return folder / f"{unit}_seed{seed}_{direction}.npz"

    def _load_or_run(self, folder, unit, seed, direction):
        path = self._case_path(folder, unit, seed, direction)
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                if str(z["signature"]) != self.signature:
                    raise ValueError(f"incompatible checkpoint: {path}")
                return tuple(np.array(z[key]) for key in
                             ("X", "y", "early", "future_max")), str(z["geometry"])
        X, y, early, future_max, geometry = self.one(unit, seed, direction)
        tmp = path.with_suffix(".tmp.npz")
        try:
            np.savez_compressed(tmp, X=X, y=y, early=early,
                                future_max=future_max, geometry=geometry,
                                signature=self.signature)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        print(f"[early_prediction] {unit} seed{seed} {direction}: {len(y)} edges")
        return (X, y, early, future_max), geometry

    @staticmethod
    def _scores(y, score):
        result = {"n_edges": int(len(y)), "positive_fraction": float(np.mean(y)),
                  "average_precision": float(average_precision_score(y, score))}
        result["roc_auc"] = (float(roc_auc_score(y, score))
                             if np.unique(y).size == 2 else None)
        return result

    def run(self, output_dir):
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        cases = {}
        for unit in self.config.units:
            for seed in self.config.seeds:
                for direction in self.config.directions:
                    key = f"{unit}:seed{seed}:{direction}"
                    data, geometry = self._load_or_run(folder, unit, seed, direction)
                    cases[key] = {"unit": unit, "geometry": geometry, "data": data}
        if len(self.config.units) < 2:
            return {"signature": self.signature, "cases": len(cases), "folds": {}}
        folds = {}
        for held_out in self.config.units:
            train = [c["data"] for c in cases.values() if c["unit"] != held_out]
            test = [c["data"] for c in cases.values() if c["unit"] == held_out]
            X_train = np.concatenate([a[0] for a in train])
            y_train = np.concatenate([a[1] for a in train])
            X_test = np.concatenate([a[0] for a in test])
            y_test = np.concatenate([a[1] for a in test])
            early_test = np.concatenate([a[2] for a in test])
            if np.unique(y_train).size < 2:
                raise ValueError(f"training labels have one class for {held_out}")
            models = {
                "static_logistic": (make_pipeline(StandardScaler(),
                                                   LogisticRegression(max_iter=1000)), 7),
                "early_logistic": (make_pipeline(StandardScaler(),
                                                  LogisticRegression(max_iter=1000)), 9),
                "early_forest": (RandomForestClassifier(n_estimators=120,
                                                         min_samples_leaf=5,
                                                         random_state=23,
                                                         n_jobs=1), 9),
            }
            scores = {"early_strain_rank": self._scores(y_test, early_test)}
            for name, (model, n_features) in models.items():
                model.fit(X_train[:, :n_features], y_train)
                prediction = model.predict_proba(X_test[:, :n_features])[:, 1]
                scores[name] = self._scores(y_test, prediction)
            folds[held_out] = {"training_cases": len(train), "test_cases": len(test),
                               "scores": scores}
        result = {"signature": self.signature, "config": asdict(self.config),
                  "case_geometry": {key: c["geometry"] for key, c in cases.items()},
                  "folds": folds,
                  "scope": "future positive axial-strain threshold in reduced solver; not force flow"}
        target = folder / "summary.json"
        temporary = target.with_suffix(".tmp.json")
        try:
            temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "benchmarks" / "results" /
                                                    "early_recruitment_prediction"))
    parser.add_argument("--units", nargs="+", default=list(PredictionConfig.units))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(PredictionConfig.seeds))
    parser.add_argument("--future-min-strain", type=float, default=0.05)
    parser.add_argument("--perturbation", type=float, default=0.04)
    parser.add_argument("--points-per-edge", type=int, default=1)
    parser.add_argument("--contact", action="store_true")
    args = parser.parse_args()
    study = EarlyRecruitmentPrediction(PredictionConfig(units=tuple(args.units),
        seeds=tuple(args.seeds), future_min_strain=args.future_min_strain,
        perturbation=args.perturbation, points_per_edge=args.points_per_edge,
        use_contact=args.contact))
    print(json.dumps(study.run(args.output_dir).get("folds", {}), indent=2))
