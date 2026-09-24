"""Resumable, length-class-matched Euler-cycle removal and rerun study.

Run: python benchmarks/constrained_cycle_intervention.py
These are reduced-solver counterfactuals, not experimentally validated prints.
"""
import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.integrate import trapezoid


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "FiberScope"))

from benchmarks.percolation_cycle_intervention import CycleInterventionStudy
from fslab.structure import StructureFactory
from fslab.simcache import ENGINE_SRC_HASH, ENGINE_VERSION
from fibernet.sim import ReducedBeamConfig, ReducedBeamSolver
from fibernet.analysis import compute_percolation


@dataclass
class ConstrainedCycleConfig:
    units: tuple = ("square", "hexagon", "ring", "kagome")
    seeds: tuple = (11, 23, 41)
    directions: tuple = ("x", "y")
    grid: int = 3
    points_per_edge: int = 2
    perturbation: float = 0.2
    target_stretch: float = 1.4
    increments: int = 20
    steps: int = 2000
    alpha: float = 0.05
    min_active: float = 0.10
    early_last_frame: int = 2
    length_decimals: int = 6
    minimum_length_fraction: float = 0.01
    minimum_options: int = 5
    max_nodes: int = 3000
    max_edges: int = 5000
    max_basis_cycles: int = 500


class ConstrainedCycleIntervention:
    @staticmethod
    def geometry_digest(positions, edges):
        """Hash planar geometry with fixed little-endian float64/int64 dtypes."""
        xy = np.ascontiguousarray(np.asarray(positions, dtype="<f8")[:, :2])
        links = np.ascontiguousarray(np.asarray(edges, dtype="<i8"))
        return hashlib.sha256(xy.tobytes() + links.tobytes()).hexdigest()[:16]

    def __init__(self, config=None):
        self.config = config or ConstrainedCycleConfig()
        cfg = self.config
        if (cfg.grid < 1 or cfg.grid > 5 or cfg.points_per_edge < 1 or
                cfg.points_per_edge > 5 or not 0 <= cfg.perturbation <= 0.3 or
                cfg.steps < 1 or cfg.increments < 2 or
                not 0 < cfg.early_last_frame < cfg.increments or
                not 0 < cfg.min_active < 1 or not 0 <= cfg.alpha <= 1 or
                cfg.length_decimals < 3 or cfg.length_decimals > 9 or
                not 0 < cfg.minimum_length_fraction < 1 or
                cfg.minimum_options < 2 or cfg.max_nodes < 1 or
                cfg.max_edges < 1 or cfg.max_basis_cycles < 1 or
                any(d not in ("x", "y") for d in cfg.directions)):
            raise ValueError("invalid constrained cycle configuration")
        sources = [Path(__file__), ROOT / "benchmarks" /
                   "percolation_cycle_intervention.py",
                   ROOT / "fibernet" / "gen" / "manufacturing.py",
                   ROOT / "fibernet" / "analysis" / "tensile_recruitment.py"]
        digest = hashlib.sha256()
        for source in sources:
            digest.update(source.read_bytes())
        self.analysis_hash = digest.hexdigest()[:16]

    def _simulate(self, graph):
        cfg = self.config
        return ReducedBeamSolver(graph, ReducedBeamConfig(
            target_stretch=cfg.target_stretch,
            n_increments=cfg.increments, num_steps=cfg.steps,
            use_contact=False, use_bending=True)).run()

    def _graph(self, unit, seed, direction):
        cfg = self.config
        graph = StructureFactory(
            unit=unit, grid_x=cfg.grid, grid_y=cfg.grid,
            n_pts_per_side=cfg.points_per_edge,
            perturbation=cfg.perturbation, seed=int(seed)).build()
        if graph.num_nodes > cfg.max_nodes or graph.num_edges > cfg.max_edges:
            raise MemoryError("intervention graph exceeds node/edge budget")
        if direction == "y":
            xy = np.asarray(graph.node_positions(), float)
            rotated = np.column_stack((xy[:, 1], -xy[:, 0]))
            graph.set_node_positions({i: point for i, point in enumerate(rotated)})
        return graph

    def _measure(self, run, edges, original_total_length, removed_length):
        recruited = compute_percolation(run, alpha=self.config.alpha,
                                        min_active=self.config.min_active)
        return {
            "nodes": int(run.frames_xy.shape[1]),
            "edges": int(len(edges)),
            "removed_length_fraction": float(removed_length / original_total_length),
            "final_raw_reaction": float(run.force_curve[-1]),
            "work_proxy": float(trapezoid(run.force_curve, run.strain_levels)),
            "first_spanning_frame": int(recruited.perc_frame),
            "final_axial_energy": float(run.energies["axial"][-1]),
            "final_bend_energy": float(run.energies["bend"][-1]),
            "final_contact_energy": float(run.energies["contact"][-1]),
        }

    @staticmethod
    def _weighted(score, lengths, ids):
        selected = np.asarray(ids, int)
        return float(np.average(score[selected], weights=lengths[selected]))

    def _candidates(self, graph, grips, lengths, early, late, alignment):
        cfg = self.config
        edges = np.asarray(graph.edge_array(), int)
        basis = CycleInterventionStudy.cycle_basis_edge_ids(
            edges, graph.num_nodes)
        if len(basis) > cfg.max_basis_cycles:
            raise MemoryError("cycle basis exceeds configured candidate budget")
        valid = []
        total = float(lengths.sum())
        for ids in basis:
            good, isolated = CycleInterventionStudy.topologically_valid(
                edges, graph.num_nodes, grips, ids)
            if not good:
                continue
            mass = float(lengths[list(ids)].sum())
            fraction = mass / total
            if fraction < cfg.minimum_length_fraction:
                continue
            valid.append({
                "removed_ids": list(ids), "removed_length": mass,
                "removed_length_fraction": fraction,
                "length_class": round(fraction, cfg.length_decimals),
                "isolated_nodes": isolated,
                "early_score": self._weighted(early, lengths, ids),
                "late_score": self._weighted(late, lengths, ids),
                "alignment_score": self._weighted(alignment, lengths, ids),
            })
        classes = {}
        for option in valid:
            classes.setdefault(option["length_class"], []).append(option)
        eligible = [(key, options) for key, options in classes.items()
                    if len(options) >= cfg.minimum_options]
        if not eligible:
            return len(basis), len(valid), None, []
        # Pick the largest adequately sized same-material class. Ties prefer
        # more removed material, then the deterministic numeric class key.
        key, options = max(eligible, key=lambda item:
                           (len(item[1]), item[0]))
        options.sort(key=lambda item: item["removed_ids"])
        return len(basis), len(valid), key, options

    def _rerun(self, graph, baseline, option, lengths):
        removed = option["removed_ids"]
        candidate, mapping = CycleInterventionStudy.compact_graph(
            graph, removed)
        multigraph = nx.MultiGraph()
        multigraph.add_nodes_from(range(candidate.num_nodes))
        edges = np.asarray(candidate.edge_array(), int)
        multigraph.add_edges_from((int(a), int(b)) for a, b in edges)
        if not nx.is_eulerian(multigraph):
            raise AssertionError("compacted candidate has no Euler circuit")
        route = list(nx.eulerian_circuit(multigraph, keys=True))
        if len(route) != len(edges) or (route and route[0][0] != route[-1][1]):
            raise AssertionError("Euler route does not cover all fiber edges")
        expected_grips = np.array(sorted(mapping[int(old)] for old in
            np.union1d(baseline.left_nodes, baseline.right_nodes)), int)
        run = self._simulate(candidate)
        actual_grips = np.union1d(run.left_nodes, run.right_nodes)
        if not np.array_equal(expected_grips, actual_grips):
            raise AssertionError("removal changed clamp identity")
        result = self._measure(run, edges, float(lengths.sum()),
                               option["removed_length"])
        result["removed_ids"] = removed
        result["removed_edge_count"] = len(removed)
        result["isolated_nodes_compacted"] = option["isolated_nodes"]
        result["euler_route_edges"] = len(route)
        result["early_score"] = option["early_score"]
        result["late_score"] = option["late_score"]
        result["alignment_score"] = option["alignment_score"]
        return result

    def one(self, unit, seed, direction):
        cfg = self.config
        graph = self._graph(unit, seed, direction)
        baseline = self._simulate(graph)
        xy = np.asarray(graph.node_positions(), float)
        edges = np.asarray(graph.edge_array(), int)
        lengths = np.asarray(baseline.edge_rest, float)
        vectors = xy[edges[:, 1]] - xy[edges[:, 0]]
        alignment = np.abs(vectors[:, 0]) / np.maximum(
            np.linalg.norm(vectors, axis=1), 1e-12)
        strain = np.maximum(np.asarray(baseline.edge_strain, float), 0)
        early = strain[1:cfg.early_last_frame + 1].max(axis=0)
        late = strain.max(axis=0)
        grips = np.union1d(baseline.left_nodes, baseline.right_nodes)
        basis_count, valid_count, length_class, options = self._candidates(
            graph, grips, lengths, early, late, alignment)
        geometry = self.geometry_digest(xy, edges)
        result = {
            "geometry_hash": geometry,
            "baseline": self._measure(baseline, edges, float(lengths.sum()), 0),
            "basis_cycle_count": basis_count,
            "valid_single_cycle_count": valid_count,
            "selected_length_class": length_class,
            "same_length_candidate_count": len(options),
            "interventions": {},
        }
        if not options:
            result["status"] = "no sufficiently populated material class"
            return result
        rng = np.random.default_rng(int(seed) + sum(map(ord, unit)) +
                                    (1009 if direction == "y" else 0))
        selected = {
            "low_early": min(options, key=lambda a: (a["early_score"],
                                                       a["removed_ids"])),
            "low_late_oracle": min(options, key=lambda a: (a["late_score"],
                                                             a["removed_ids"])),
            "low_alignment": min(options, key=lambda a: (a["alignment_score"],
                                                           a["removed_ids"])),
            "random": options[int(rng.integers(len(options)))],
            "high_late": max(options, key=lambda a: (a["late_score"],
                                                       a["removed_ids"])),
        }
        reruns = {}
        for name, option in selected.items():
            key = tuple(option["removed_ids"])
            if key not in reruns:
                reruns[key] = self._rerun(graph, baseline, option, lengths)
            result["interventions"][name] = reruns[key]
        result["unique_reruns"] = len(reruns)
        result["max_material_class_spread"] = float(
            max(a["removed_length_fraction"] for a in selected.values()) -
            min(a["removed_length_fraction"] for a in selected.values()))
        result["status"] = "complete"
        return result

    def run(self, output, max_cases=None):
        output = Path(output)
        config = json.loads(json.dumps(asdict(self.config)))
        signature = {"config": config, "engine_version": ENGINE_VERSION,
                     "engine_hash": ENGINE_SRC_HASH,
                     "analysis_hash": self.analysis_hash}
        if output.exists():
            data = json.loads(output.read_text(encoding="utf-8"))
            if any(data.get(key) != value for key, value in signature.items()):
                raise ValueError("incompatible intervention checkpoint")
        else:
            data = dict(signature, cases={})
        completed = 0
        for unit in self.config.units:
            for seed in self.config.seeds:
                for direction in self.config.directions:
                    key = f"{unit}:seed{seed}:{direction}"
                    if key in data["cases"]:
                        continue
                    if max_cases is not None and completed >= max_cases:
                        return data
                    data["cases"][key] = self.one(unit, seed, direction)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    temporary = output.with_suffix(output.suffix + ".tmp")
                    try:
                        temporary.write_text(json.dumps(data, indent=2) + "\n",
                                             encoding="utf-8")
                        os.replace(temporary, output)
                    finally:
                        temporary.unlink(missing_ok=True)
                    completed += 1
                    print(f"[cycle_intervention] {key} complete")
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "benchmarks" / "results" /
                                                "constrained_cycle_intervention.json"))
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--units", nargs="+", default=list(ConstrainedCycleConfig.units))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(ConstrainedCycleConfig.seeds))
    parser.add_argument("--directions", nargs="+", default=list(ConstrainedCycleConfig.directions))
    args = parser.parse_args()
    study = ConstrainedCycleIntervention(ConstrainedCycleConfig(
        units=tuple(args.units), seeds=tuple(args.seeds),
        directions=tuple(args.directions)))
    print("completed cases", len(study.run(args.output, args.max_cases)["cases"]))
