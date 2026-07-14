#!/usr/bin/env python3
"""
FiberNet v4.1.0 — Comprehensive 3D Validation Script (v2)

Re-runnable validation with checkpoint/resume for long simulations.
Covers: generation, connectivity, simulation (relaxation), features, visualization.

Usage:
    cd fibernet && source .venv/bin/activate
    python fibernet/scripts/validate_3d_v2.py

Results saved to: output_data/3d_validation_v2/
Checkpoint: output_data/3d_validation_v2/_checkpoint.json
"""

import os
import sys
import json
import time
import traceback
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import fibernet as fn
from fibernet.analysis.graph_features_3d import GraphFeatureExtractor3D

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'output_data', '3d_validation_v2')
VIZ_DIR = os.path.join(OUTPUT_DIR, 'viz')
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, '_checkpoint.json')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

SPECIAL_KWARGS = {
    'chiral_3d': {'chirality': 0.3},
    'reentrant_3d': {'angle': 15.0},
}


def load_checkpoint():
    """Load checkpoint to resume from last completed unit."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"completed": [], "results": []}


def save_checkpoint(data):
    """Save checkpoint atomically."""
    tmp = CHECKPOINT_FILE + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, CHECKPOINT_FILE)


def validate_single_unit(unit_name, engine, ext3d):
    """Full validation pipeline for one unit type."""
    ukw = SPECIAL_KWARGS.get(unit_name, {})
    result = {"unit": unit_name, "errors": []}

    # 1. Generation
    try:
        t0 = time.time()
        g = fn.pattern_3d(unit=unit_name, box=(10, 10, 10), grid=(2, 2, 2), unit_kwargs=ukw)
        result["generation"] = {
            "nodes": g.num_nodes, "edges": g.num_edges,
            "time_s": round(time.time() - t0, 3)
        }
    except Exception as e:
        result["errors"].append(f"generation: {e}")
        return result

    # 2. Connectivity check
    try:
        import networkx as nx
        G = nx.Graph()
        for nid in g.nodes:
            G.add_node(nid)
        for eid in g.edges:
            e = g.edges[eid]
            G.add_edge(e.node_i, e.node_j)
        n_comps = nx.number_connected_components(G)
        degrees = [d for _, d in G.degree()]
        result["connectivity"] = {
            "components": n_comps,
            "min_degree": min(degrees),
            "max_degree": max(degrees),
            "mean_degree": round(np.mean(degrees), 1),
        }
        if n_comps > 1:
            result["errors"].append(f"disconnected: {n_comps} components")
    except Exception as e:
        result["errors"].append(f"connectivity: {e}")

    # 3. Simulation (auto-steps for good relaxation)
    try:
        t0 = time.time()
        sim_result = engine.stretch_test(g, target_stretch=1.2, max_nodes=10000)
        result["simulation"] = {
            "energy": round(sim_result.energy, 1),
            "max_stretch": round(sim_result.max_stretch, 4),
            "mean_stretch": round(sim_result.mean_stretch, 4),
            "std_stretch": round(sim_result.std_stretch, 4),
            "max_force": round(sim_result.max_force, 1),
            "time_s": round(time.time() - t0, 1),
        }
    except Exception as e:
        result["errors"].append(f"simulation: {e}")
        sim_result = None

    # 4. 3D Features
    try:
        t0 = time.time()
        features = ext3d.extract(g)
        vec = ext3d.extract_vector(g)
        n_nan = int(np.sum(np.isnan(vec)))
        result["features"] = {
            "n_features": len(features),
            "nan_count": n_nan,
            "time_s": round(time.time() - t0, 3),
        }
        if n_nan > 0:
            result["errors"].append(f"features: {n_nan} NaN values")
    except Exception as e:
        result["errors"].append(f"features: {e}")

    # 5. Visualization (dark theme, combined figure)
    if sim_result is not None:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            # Stress visualization (dark theme)
            path = os.path.join(VIZ_DIR, f"stress_{unit_name}.png")
            fig = fn.render_stress_3d(g, sim_result, color_by="force", theme="dark",
                                       save_path=path)
            plt.close(fig)

            # Comparison (3-panel)
            path2 = os.path.join(VIZ_DIR, f"comparison_{unit_name}.png")
            fig2 = fn.render_comparison_3d(g, sim_result, theme="dark",
                                            save_path=path2)
            plt.close(fig2)

            result["visualization"] = {
                "stress_saved": os.path.exists(path),
                "comparison_saved": os.path.exists(path2),
            }
        except Exception as e:
            result["errors"].append(f"visualization: {e}")

    return result


def main():
    """Run full validation with checkpoint/resume."""
    checkpoint = load_checkpoint()
    completed = set(checkpoint["completed"])
    all_results = checkpoint["results"]

    engine = fn.TaichiEngine()
    ext3d = GraphFeatureExtractor3D()

    units = fn.list_units_3d()
    pending = [u for u in units if u not in completed]

    print(f"FiberNet v{fn.__version__} — 3D Validation v2")
    print(f"Units: {len(units)} total, {len(completed)} done, {len(pending)} pending\n")

    for unit_name in pending:
        print(f"Validating {unit_name}...", end=" ", flush=True)
        t0 = time.time()
        result = validate_single_unit(unit_name, engine, ext3d)
        elapsed = time.time() - t0

        status = "✅" if not result["errors"] else f"⚠️ {len(result['errors'])} errors"
        sim = result.get("simulation", {})
        conn = result.get("connectivity", {})
        print(f"{status} | comps={conn.get('components', '?')} "
              f"max_s={sim.get('max_stretch', '?')} "
              f"E={sim.get('energy', '?')} ({elapsed:.1f}s)")

        if result["errors"]:
            for err in result["errors"]:
                print(f"  ERROR: {err[:100]}")

        all_results.append(result)
        completed.add(unit_name)
        save_checkpoint({"completed": list(completed), "results": all_results})

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    total_errors = sum(len(r["errors"]) for r in all_results)
    passed = sum(1 for r in all_results if not r["errors"])
    print(f"Units: {len(all_results)}, Passed: {passed}/{len(all_results)}, Errors: {total_errors}")

    print(f"\n{'Unit':<15} {'Comps':>5} {'Nodes':>6} {'MaxS':>7} {'MeanS':>7} {'Energy':>12} {'Time':>6}")
    print("-" * 70)
    for r in all_results:
        conn = r.get("connectivity", {})
        gen = r.get("generation", {})
        sim = r.get("simulation", {})
        print(f"{r['unit']:<15} {conn.get('components', '?'):>5} {gen.get('nodes', '?'):>6} "
              f"{sim.get('max_stretch', '?'):>7} {sim.get('mean_stretch', '?'):>7} "
              f"{sim.get('energy', '?'):>12} {sim.get('time_s', '?'):>5}s")

    # Save final results
    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults: {results_path}")

    return total_errors == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
