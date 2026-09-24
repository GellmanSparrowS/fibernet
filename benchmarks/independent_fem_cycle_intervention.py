"""Matched-grip, matched-stretch linear beam-FEM check of cycle deletions.

Run: python benchmarks/independent_fem_cycle_intervention.py
The two models use different constitutive assumptions and uncalibrated units.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "FiberScope"))

from benchmarks.constrained_cycle_intervention import ConstrainedCycleIntervention
from benchmarks.percolation_cycle_intervention import CycleInterventionStudy
from fibernet.ml.beam_frame_fem import BeamFrameFEM
from fibernet.sim import ReducedBeamConfig, ReducedBeamSolver


class IndependentFemCycleCheck:
    def __init__(self, stretch=1.08, max_nodes=1200, max_edges=1600):
        self.stretch = float(stretch)
        self.max_nodes = int(max_nodes)
        self.max_edges = int(max_edges)
        if not 1.0 < self.stretch <= 1.15 or self.max_nodes < 1 or self.max_edges < 1:
            raise ValueError("invalid independent FEM check configuration")
        self.study = ConstrainedCycleIntervention()
        self.reference_path = (ROOT / "benchmarks" / "results" /
                               "constrained_cycle_intervention.json")
        sources = [Path(__file__), ROOT / "fibernet" / "ml" /
                   "beam_frame_fem.py", ROOT / "fibernet" / "sim" /
                   "reduced_beam.py", ROOT / "benchmarks" /
                   "percolation_cycle_intervention.py"]
        self.source_hash = hashlib.sha256(b"".join(
            path.read_bytes() for path in sources)).hexdigest()[:16]
        self.beam = BeamFrameFEM(E=1e9, nu=.3)

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

    def _signature(self):
        return {"config": {"stretch": self.stretch,
                           "beam_E": 1e9, "beam_nu": .3,
                           "max_nodes": self.max_nodes,
                           "max_edges": self.max_edges,
                           "reduced_steps": 2000,
                           "reduced_increments": 20,
                           "contact": False,
                           "bending": True,
                           "grips": "identical reduced-solver node IDs"},
                "source_hash": self.source_hash,
                "reference_sha256": hashlib.sha256(
                    self.reference_path.read_bytes()).hexdigest()}

    def _measure(self, graph):
        if graph.num_nodes > self.max_nodes or graph.num_edges > self.max_edges:
            raise MemoryError("independent FEM graph exceeds configured budget")
        reduced = ReducedBeamSolver(graph, ReducedBeamConfig(
            target_stretch=self.stretch, n_increments=20, num_steps=2000,
            use_contact=False, use_bending=True)).run()
        inp = self.beam.graph_to_fem_input(graph, dim=2, pct=.05)
        left = np.asarray(reduced.left_nodes, int)
        right = np.asarray(reduced.right_nodes, int)
        displacement = inp["x_range"] * (self.stretch - 1.0)
        fem = self.beam.solve_2d(
            inp["edge_index"], inp["node_pos"], inp["radii"],
            fixed_nodes=left,
            prescribed_disp={int(node): (displacement, 0.0) for node in right},
            deduplicate=False)
        if fem["n_edges"] != graph.num_edges:
            raise AssertionError("independent FEM dropped parallel fiber edges")
        right_force = float(np.sum(fem["reactions"][right, 0]))
        left_force = float(np.sum(fem["reactions"][left, 0]))
        balance = abs(right_force + left_force) / max(abs(right_force), 1.0)
        if (right_force <= 0 or reduced.force_curve[-1] <= 0 or
                balance > 1e-5):
            raise AssertionError("invalid clamp reaction or equilibrium")
        return {"nodes": graph.num_nodes, "edges": graph.num_edges,
                "left_grips": left.tolist(), "right_grips": right.tolist(),
                "fem_right_reaction": right_force,
                "fem_left_reaction": left_force,
                "fem_balance_relative": balance,
                "reduced_raw_reaction": float(reduced.force_curve[-1])}

    def run(self, output, max_new_cases=None):
        output = Path(output)
        reference = json.loads(self.reference_path.read_text(encoding="utf-8"))
        if len(reference["cases"]) != 24:
            raise ValueError("complete 24-case cycle reference first")
        signature = self._signature()
        if output.exists():
            data = json.loads(output.read_text(encoding="utf-8"))
            if any(data.get(key) != value for key, value in signature.items()):
                raise ValueError("incompatible independent FEM checkpoint")
        else:
            data = dict(signature, cases={})
        new_cases = 0
        for unit in self.study.config.units:
            for seed in self.study.config.seeds:
                for direction in self.study.config.directions:
                    key = f"{unit}:seed{seed}:{direction}"
                    if data["cases"].get(key, {}).get("status") == "complete":
                        continue
                    if max_new_cases is not None and new_cases >= max_new_cases:
                        return data
                    graph = self.study._graph(unit, seed, direction)
                    positions = np.asarray(graph.node_positions(), float)
                    edges = np.asarray(graph.edge_array(), int)
                    digest = hashlib.sha256(
                        positions.tobytes() + edges.tobytes()).hexdigest()[:16]
                    original = reference["cases"][key]
                    if (digest != original["geometry_hash"] or
                            original["max_material_class_spread"] > 1e-9):
                        raise ValueError("independent FEM reference graph changed")
                    case = data["cases"].setdefault(key, {
                        "geometry_hash": digest, "interventions": {}})
                    if case["geometry_hash"] != digest:
                        raise ValueError("independent FEM case changed")
                    baseline = self._measure(graph)
                    if "baseline" in case and case["baseline"] != baseline:
                        raise ValueError("independent FEM baseline changed")
                    case["baseline"] = baseline
                    self._save(output, data)
                    cached = {}
                    for strategy in ("low_early", "low_late_oracle",
                                     "low_alignment", "random", "high_late"):
                        removed = original["interventions"][strategy]["removed_ids"]
                        old = case["interventions"].get(strategy)
                        if old is not None:
                            if old["removed_ids"] != removed:
                                raise ValueError("independent FEM candidate changed")
                            continue
                        ids = tuple(removed)
                        if ids not in cached:
                            candidate, mapping = CycleInterventionStudy.compact_graph(
                                graph, removed)
                            route_graph = nx.MultiGraph()
                            route_graph.add_nodes_from(range(candidate.num_nodes))
                            route_graph.add_edges_from((int(a), int(b)) for a, b in
                                                       candidate.edge_array())
                            if not nx.is_eulerian(route_graph):
                                raise AssertionError("deletion lost its Euler route")
                            metrics = self._measure(candidate)
                            for side in ("left", "right"):
                                expected = sorted(mapping[int(node)] for node in
                                                  baseline[f"{side}_grips"])
                                if sorted(metrics[f"{side}_grips"]) != expected:
                                    raise AssertionError("deletion changed clamp nodes")
                            cached[ids] = metrics
                        case["interventions"][strategy] = {
                            "removed_ids": removed, **cached[ids]}
                        self._save(output, data)
                    case["status"] = "complete"
                    self._save(output, data)
                    new_cases += 1
                    print(f"[fem_cycle] {key} complete")
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "benchmarks" / "results" /
                                                "independent_fem_cycle_intervention.json"))
    parser.add_argument("--max-new-cases", type=int)
    args = parser.parse_args()
    result = IndependentFemCycleCheck().run(args.output, args.max_new_cases)
    print("[fem_cycle] saved", len(result["cases"]), "cases")
