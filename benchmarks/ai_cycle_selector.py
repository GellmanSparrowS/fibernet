"""Topology-held-out, route-preserving cycle selection with actual reruns.

Run: python benchmarks/ai_cycle_selector.py
Use --max-new 1 for an atomic-resume smoke. Labels are reduced-model outcomes.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "FiberScope"))

from benchmarks.constrained_cycle_intervention import (
    ConstrainedCycleConfig, ConstrainedCycleIntervention, ROOT)
from fslab.simcache import ENGINE_SRC_HASH, ENGINE_VERSION


RESULTS = ROOT / "benchmarks" / "results"


class AICycleSelector:
    def __init__(self, max_options_per_case=100, max_total_options=1000):
        self.study = ConstrainedCycleIntervention(ConstrainedCycleConfig())
        self.max_options_per_case = int(max_options_per_case)
        self.max_total_options = int(max_total_options)
        if self.max_options_per_case < 5 or self.max_total_options < 24:
            raise ValueError("AI cycle budgets are too small")
        source = [Path(__file__), ROOT / "benchmarks" /
                  "constrained_cycle_intervention.py",
                  ROOT / "benchmarks" / "percolation_cycle_intervention.py",
                  ROOT / "fibernet" / "sim" / "reduced_beam.py"]
        self.analysis_hash = hashlib.sha256(b"".join(
            path.read_bytes() for path in source)).hexdigest()[:16]
        self.reference_path = RESULTS / "constrained_cycle_intervention.json"

    def _signature(self):
        return {"config": {"max_options_per_case": self.max_options_per_case,
                           "max_total_options": self.max_total_options,
                           "model": "RandomForestRegressor-200-leaf3-seed11",
                           "training_split": "leave-one-topology-out"},
                "analysis_hash": self.analysis_hash,
                "engine_version": ENGINE_VERSION,
                "engine_hash": ENGINE_SRC_HASH,
                "reference_sha256": hashlib.sha256(
                    self.reference_path.read_bytes()).hexdigest()}

    @staticmethod
    def _save(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            temporary.write_text(json.dumps(data, indent=2) + "\n",
                                 encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _prepare(self, unit, seed, direction, reference):
        graph = self.study._graph(unit, seed, direction)
        baseline = self.study._simulate(graph)
        positions = np.asarray(graph.node_positions(), float)
        xy = positions[:, :2]
        edges = np.asarray(graph.edge_array(), int)
        digest = self.study.geometry_digest(positions, edges)
        if digest != reference["geometry_hash"]:
            raise AssertionError("AI candidate geometry differs from reference")
        if not np.isclose(float(baseline.force_curve[-1]),
                          reference["baseline"]["final_raw_reaction"],
                          rtol=1e-8, atol=1e-6):
            raise AssertionError("AI baseline differs from reference")
        lengths = np.asarray(baseline.edge_rest, float)
        vectors = xy[edges[:, 1]] - xy[edges[:, 0]]
        alignment = np.abs(vectors[:, 0]) / np.maximum(
            np.linalg.norm(vectors, axis=1), 1e-12)
        strain = np.maximum(np.asarray(baseline.edge_strain, float), 0)
        early = strain[1:self.study.config.early_last_frame + 1].max(axis=0)
        late = strain.max(axis=0)
        grips = np.union1d(baseline.left_nodes, baseline.right_nodes)
        _, _, length_class, options = self.study._candidates(
            graph, grips, lengths, early, late, alignment)
        if (length_class != reference["selected_length_class"] or
                len(options) != reference["same_length_candidate_count"] or
                len(options) > self.max_options_per_case):
            raise ValueError("AI option class or candidate budget changed")
        edge_mid = xy[edges].mean(axis=1)
        low = xy.min(axis=0)
        span = np.maximum(np.ptp(xy, axis=0), 1e-12)
        for option in options:
            ids = np.asarray(option["removed_ids"], int)
            midpoint = edge_mid[ids].mean(axis=0)
            option["early_features"] = [
                option["early_score"], float(early[ids].max()),
                float(early[ids].std()), option["alignment_score"],
                option["removed_length_fraction"],
                float((midpoint[0] - low[0]) / span[0]),
                float((midpoint[1] - low[1]) / span[1]),
                float(len(ids) / len(edges))]
        return graph, baseline, lengths, options

    def run(self, output, max_new=None):
        output = Path(output)
        reference = json.loads(self.reference_path.read_text(encoding="utf-8"))
        if len(reference["cases"]) != 24:
            raise ValueError("complete 24-case reference study first")
        signature = self._signature()
        if output.exists():
            data = json.loads(output.read_text(encoding="utf-8"))
            if any(data.get(key) != value for key, value in signature.items()):
                raise ValueError("incompatible AI cycle checkpoint")
        else:
            data = dict(signature, cases={})
        new = 0
        total = 0
        for unit in self.study.config.units:
            for seed in self.study.config.seeds:
                for direction in self.study.config.directions:
                    key = f"{unit}:seed{seed}:{direction}"
                    original = reference["cases"][key]
                    graph, baseline, lengths, options = self._prepare(
                        unit, seed, direction, original)
                    case = data["cases"].setdefault(key, {"geometry_hash":
                        original["geometry_hash"], "options": {}})
                    if case["geometry_hash"] != original["geometry_hash"]:
                        raise ValueError("AI case geometry changed")
                    total += len(options)
                    if total > self.max_total_options:
                        raise MemoryError("total AI cycle budget exceeded")
                    for index, option in enumerate(options):
                        item = case["options"].get(str(index))
                        if item is not None:
                            if item["removed_ids"] != option["removed_ids"]:
                                raise ValueError("AI candidate order changed")
                            continue
                        if max_new is not None and new >= max_new:
                            return data
                        result = self.study._rerun(graph, baseline, option,
                                                   lengths)
                        case["options"][str(index)] = {
                            "removed_ids": option["removed_ids"],
                            "early_features": option["early_features"],
                            "final_reaction_ratio": result["final_raw_reaction"] /
                            float(baseline.force_curve[-1]),
                            "work_ratio": result["work_proxy"] /
                            original["baseline"]["work_proxy"],
                            "euler_route_edges": result["euler_route_edges"]}
                        self._save(output, data)
                        new += 1
                        if new == 1 or new % 25 == 0:
                            print(f"[ai_cycle] {new} new candidates; latest {key} #{index}")
        data["evaluation"] = self.evaluate(data, reference)
        self._save(output, data)
        return data

    def evaluate(self, data, reference):
        cases = data["cases"]
        expected = [f"{unit}:seed{seed}:{direction}"
                    for unit in self.study.config.units
                    for seed in self.study.config.seeds
                    for direction in self.study.config.directions]
        if any(key not in cases or len(cases[key]["options"]) !=
               reference["cases"][key]["same_length_candidate_count"]
               for key in expected):
            raise ValueError("AI evaluation requires every candidate rerun")
        evaluation = {}
        for heldout in self.study.config.units:
            training = [item for key, case in cases.items()
                        if not key.startswith(heldout + ":")
                        for item in case["options"].values()]
            X = np.asarray([item["early_features"] for item in training], float)
            y = np.asarray([item["final_reaction_ratio"] for item in training], float)
            model = RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                          random_state=11, n_jobs=1)
            model.fit(X, y)
            outcomes = []
            for seed in self.study.config.seeds:
                for direction in self.study.config.directions:
                    key = f"{heldout}:seed{seed}:{direction}"
                    items = list(cases[key]["options"].values())
                    predictions = model.predict(np.asarray(
                        [item["early_features"] for item in items], float))
                    selected = int(np.argmax(predictions))
                    actual = items[selected]["final_reaction_ratio"]
                    baseline = reference["cases"][key]["interventions"]
                    force = reference["cases"][key]["baseline"]["final_raw_reaction"]
                    outcomes.append({"case": key, "selected_index": selected,
                        "selected_removed_ids": items[selected]["removed_ids"],
                        "predicted_ratio": float(predictions[selected]),
                        "actual_ratio": actual,
                        "best_candidate_ratio": max(item["final_reaction_ratio"]
                                                    for item in items),
                        "low_early_ratio": baseline["low_early"]["final_raw_reaction"] / force,
                        "low_alignment_ratio": baseline["low_alignment"]["final_raw_reaction"] / force,
                        "random_ratio": baseline["random"]["final_raw_reaction"] / force})
            evaluation[heldout] = outcomes
        return evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(RESULTS / "ai_cycle_selector.json"))
    parser.add_argument("--max-new", type=int)
    args = parser.parse_args()
    result = AICycleSelector().run(args.output, args.max_new)
    print("[ai_cycle] saved", sum(len(case["options"]) for case in
                               result["cases"].values()), "candidates")
