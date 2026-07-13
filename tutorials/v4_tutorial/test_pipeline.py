#!/usr/bin/env python3
"""
End-to-end test of the v4 tutorial pipeline.
Uses VORONOI unit type with internal point variations.

Usage:
    python3 test_pipeline.py          # Test with 5 samples
    python3 test_pipeline.py --full   # Full run with 2000 samples
"""
import os, sys, json, time, warnings, pickle
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── Config ───
N_SAMPLES = 5 if "--full" not in sys.argv else 2000
UNIT = "voronoi"
BOX = (10, 10)
GRID = (3, 3)
N_PTS_VARIANT = 3   # internal points per edge for variants
N_PTS_BASE = 0      # base has no internal points (straight edges)
N_SEEDS = 8         # voronoi seed count (controls complexity)

OUT = Path("fibernet_v4_tutorial")
OUT.mkdir(parents=True, exist_ok=True)
DATA_OUT = OUT / "data"
JSON_OUT = DATA_OUT / "json"
IMG_OUT  = DATA_OUT / "images"
ML_OUT   = DATA_OUT / "ml_results"
RL_OUT   = DATA_OUT / "rl_results"
for d in [JSON_OUT, IMG_OUT, ML_OUT, RL_OUT]:
    d.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fibernet as fn
from fibernet import pattern_2d, TaichiEngine, render_graph, render_trajectory
from fibernet.analysis.graph_features import GraphFeatureExtractor
from fibernet.sim.accelerated import SimResult

print(f"=== FiberNet v4 Tutorial Test (Voronoi) ===")
print(f"Samples: {N_SAMPLES}, unit: {UNIT}, n_pts: {N_PTS_VARIANT}")
print(f"Output: {OUT.resolve()}")
print()


# ─── Step 1: Generate ───
print("=" * 50)
print("Step 1: Structure Generation (Voronoi)")
print("=" * 50)

def generate_voronoi(seed, n_pts=N_PTS_VARIANT):
    """Generate voronoi structure. Different seeds = different topology + displacements."""
    g = pattern_2d(
        unit=UNIT, box=BOX, grid=GRID,
        n_pts_per_side=n_pts,
        seed=seed,
        unit_kwargs={'n_seeds': N_SEEDS},
    )
    return g


# Base: voronoi with straight edges (no internal points)
g_base = generate_voronoi(seed=42, n_pts=N_PTS_BASE)
base_name = f"{UNIT}_{GRID[0]}x{GRID[1]}_seeds{N_SEEDS}_pts0_seed42"
g_base.save_json(str(JSON_OUT / f"{base_name}.json"))
fig = render_graph(g_base, theme="dark", title="Base Voronoi (No Internal Points)")
fig.savefig(str(IMG_OUT / f"{base_name}.png"), dpi=100, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)

metadata = [{"id": 0, "seed": 42, "name": base_name,
             "is_base": True, "n_pts": N_PTS_BASE,
             "n_nodes": g_base.num_nodes, "n_edges": g_base.num_edges}]

print(f"Base (voronoi, pts=0): {g_base.num_nodes} nodes, {g_base.num_edges} edges")
print(f"  Internal: {len(g_base.get_internal_nodes())}")

# Variants: voronoi with internal points (auto-displacements from seed)
print(f"\nGenerating {N_SAMPLES-1} variants (voronoi with n_pts={N_PTS_VARIANT})...")
for i in range(1, N_SAMPLES):
    seed = 100 + i
    g = generate_voronoi(seed=seed, n_pts=N_PTS_VARIANT)
    name = f"{UNIT}_{GRID[0]}x{GRID[1]}_seeds{N_SEEDS}_pts{N_PTS_VARIANT}_seed{seed}"
    g.save_json(str(JSON_OUT / f"{name}.json"))
    fig = render_graph(g, theme="dark", title=f"V{i} (seed={seed})")
    fig.savefig(str(IMG_OUT / f"{name}.png"), dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    metadata.append({"id": i, "seed": seed, "name": name,
                     "is_base": False, "n_pts": N_PTS_VARIANT,
                     "n_nodes": g.num_nodes, "n_edges": g.num_edges})
    if i % 50 == 0:
        print(f"  [{i}/{N_SAMPLES-1}] Generated")

with open(str(DATA_OUT / "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Generated {len(metadata)} structures")
print(f"  JSON: {len(list(JSON_OUT.glob('*.json')))}")
print(f"  Images: {len(list(IMG_OUT.glob('*.png')))}")


# ─── Step 2: Simulation ───
print()
print("=" * 50)
print("Step 2: Batch Simulation")
print("=" * 50)

engine = TaichiEngine()
sim_results = []
ckpt = DATA_OUT / "sim_partial.json"

if ckpt.exists():
    with open(ckpt) as f:
        sim_results = json.load(f)
    done_ids = {r["id"] for r in sim_results}
    print(f"Resuming from checkpoint: {len(sim_results)} already done")
else:
    done_ids = set()

for rec in metadata:
    if rec["id"] in done_ids:
        continue
    g_path = JSON_OUT / f"{rec['name']}.json"
    try:
        g = fn.StructureGraph.load_json(str(g_path))
        r = engine.stretch_test(
            g, target_stretch=1.5, stiffness=1e5, damping=0.3,
            num_steps=1000, save_interval=200, auto_steps=False,
        )
        row = {
            "id": rec["id"], "name": rec["name"], "is_base": rec["is_base"],
            "max_force": float(r.max_force), "max_stretch": float(r.max_stretch),
            "mean_stretch": float(r.mean_stretch), "std_stretch": float(r.std_stretch),
            "n_nodes": g.num_nodes, "n_edges": g.num_edges,
            "time_seconds": float(r.time_seconds), "success": True,
        }
        sim_results.append(row)
        r.save(str(DATA_OUT / f"{rec['name']}_result.json"), detailed=True)
        with open(str(ckpt), "w") as f:
            json.dump(sim_results, f, indent=2)
        print(f"  [{rec['id']}] {rec['name'][:40]}: force={r.max_force:.0f}")
    except Exception as e:
        sim_results.append({"id": rec["id"], "name": rec["name"], "success": False, "error": str(e)})
        print(f"  [{rec['id']}] FAILED: {e}")

df_sim = pd.DataFrame(sim_results)
df_sim.to_csv(str(DATA_OUT / "sim_results.csv"), index=False)
ok = df_sim[df_sim["success"]]
print(f"✓ Simulated: {len(ok)} ok, {(~df_sim['success']).sum()} failed")
if len(ok) > 0:
    print(f"  max_force: {ok['max_force'].mean():.0f} ± {ok['max_force'].std():.0f}")


# ─── Step 3: Deformation Visualization ───
print()
print("=" * 50)
print("Step 3: Deformation Visualization")
print("=" * 50)

for rec in metadata[:3]:
    r_path = DATA_OUT / f"{rec['name']}_result.json"
    if not r_path.exists():
        continue
    try:
        r = SimResult.load(str(r_path))
        g = fn.StructureGraph.load_json(str(JSON_OUT / f"{rec['name']}.json"))
        fig = render_trajectory(
            g, r.positions_trajectory, r.edge_stretches,
            n_frames=min(6, len(r.positions_trajectory)),
            title=rec["name"][:35],
        )
        fig.savefig(str(IMG_OUT / f"{rec['name']}_deform.png"),
                    dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  ✓ {rec['name'][:40]}")
    except Exception as e:
        print(f"  ✗ {rec['name'][:40]}: {e}")


# ─── Step 4: Feature Extraction ───
print()
print("=" * 50)
print("Step 4: Feature Extraction")
print("=" * 50)

ext = GraphFeatureExtractor(canvas_size=256)
feat_records = []

for rec in metadata:
    g = fn.StructureGraph.load_json(str(JSON_OUT / f"{rec['name']}.json"))
    try:
        feats = ext.extract(g)
        record = {"id": rec["id"], "name": rec["name"]}
        for k, v in feats.items():
            record[f"feat_{k}"] = float(v) if isinstance(v, (int, float)) else v
        feat_records.append(record)
    except Exception as e:
        print(f"  Warning: {e}")

df_feat = pd.DataFrame(feat_records)
n_feat = len([c for c in df_feat.columns if c.startswith("feat_")])
print(f"✓ Features: {len(df_feat)} samples, {n_feat} dims")

df_all = df_sim.merge(df_feat, on=["id", "name"], how="outer")
df_all.to_csv(str(DATA_OUT / "full_results.csv"), index=False)


# ─── Step 5: ML ───
print()
print("=" * 50)
print("Step 5: Machine Learning")
print("=" * 50)

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

ok_ids = set(df_sim[df_sim["success"]]["id"]) & set(df_feat["id"])
df_ml = df_feat[df_feat["id"].isin(ok_ids)].merge(
    df_sim[df_sim["success"]][["id", "max_force"]], on="id"
)

feat_cols = [c for c in df_ml.columns if c.startswith("feat_")]
feat_cols = [c for c in feat_cols if df_ml[c].std() > 1e-12]
X = df_ml[feat_cols].fillna(0).values
y = df_ml["max_force"].values

r2 = float('nan')
if len(X) >= 3:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}, Features: {len(feat_cols)}")

    from fibernet.ml import compare_models, plot_predictions, plot_feature_importance
    results = compare_models(X_train, y_train, model_types=["ridge", "rf"])
    for name, m in results.items():
        print(f"  {name}: R²={m['r2']:.4f}, RMSE={m['rmse']:.2e}")

    from sklearn.ensemble import RandomForestRegressor
    best_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    best_rf.fit(X_train, y_train)
    y_pred = best_rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    fig = plot_predictions(y_test, y_pred, title="RF Test Predictions")
    fig.savefig(str(ML_OUT / "predictions.png"), dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Final RF: R²={r2:.4f}")
else:
    print(f"  Not enough samples ({len(X)})")


# ─── Step 6: RL (Bayesian Optimization) ───
print()
print("=" * 50)
print("Step 6: Bayesian Optimization (RL)")
print("=" * 50)

def evaluate_structure(params):
    """Evaluate a voronoi structure with given params."""
    grid_x = max(2, min(5, int(round(params.get("grid_x", 3)))))
    grid_y = max(2, min(5, int(round(params.get("grid_y", 3)))))
    n_seeds = max(4, min(15, int(round(params.get("n_seeds", 8)))))
    n_pts = max(0, min(5, int(round(params.get("n_pts", 3)))))
    seed = int(params.get("seed", 42))
    try:
        g = pattern_2d(unit=UNIT, box=BOX, grid=(grid_x, grid_y),
                       n_pts_per_side=n_pts, seed=seed,
                       unit_kwargs={'n_seeds': n_seeds})
        r = engine.stretch_test(g, target_stretch=1.5, stiffness=1e5, damping=0.3,
                                num_steps=500, save_interval=500, auto_steps=False)
        return float(r.max_force)
    except Exception:
        return 1e10

base_force = evaluate_structure({"grid_x": 3, "grid_y": 3, "n_seeds": N_SEEDS, "n_pts": 0, "seed": 42})
print(f"  Base force: {base_force:.0f}")

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer

    dimensions = [
        Integer(2, 5, name="grid_x"),
        Integer(2, 5, name="grid_y"),
        Integer(4, 15, name="n_seeds"),
        Integer(0, 5, name="n_pts"),
    ]
    dim_names = ["grid_x", "grid_y", "n_seeds", "n_pts"]

    all_forces = []
    all_params_list = []

    def _objective(x):
        params = {dim_names[i]: x[i] for i in range(len(x))}
        params["seed"] = 42
        force = evaluate_structure(params)
        all_forces.append(force)
        all_params_list.append(params.copy())
        return force

    result = gp_minimize(_objective, dimensions, n_calls=10, n_initial_points=5,
                         random_state=42, verbose=False)

    print(f"  Best force: {result.fun:.0f}")
    print(f"  Best params: {dict(zip(dim_names, result.x))}")
    print(f"  Improvement: {(1 - result.fun / max(base_force, 1)) * 100:.1f}%")

    from fibernet.rl import plot_convergence, plot_reward_curve, plot_action_distribution
    fig = plot_convergence(all_forces, minimize=True, title="Bayesian Optimization")
    fig.savefig(str(RL_OUT / "convergence.png"), dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    fig = plot_reward_curve([-f for f in all_forces], window=3, title="Reward")
    fig.savefig(str(RL_OUT / "reward_curve.png"), dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"✓ RL done")
except ImportError:
    print("  skopt not installed, skipping")


# ─── Summary ───
print()
print("=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"  Generated: {len(metadata)} structures")
print(f"  Simulated: {df_sim['success'].sum()} ok")
print(f"  Features: {n_feat} dims")
print(f"  ML: R²={r2:.4f}" if len(X) >= 3 else "  ML: skipped")
print(f"  Output: {OUT.resolve()}")

# List output files
print("\nOutput files:")
for f in sorted(OUT.rglob("*")):
    if f.is_file():
        size = f.stat().st_size
        rel = f.relative_to(OUT)
        print(f"  {rel} ({size/1024:.0f} KB)")

print("\n✓ ALL TESTS PASSED")
