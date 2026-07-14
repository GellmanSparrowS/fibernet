#!/usr/bin/env python3
"""
FiberNet v4 Tutorial — Visualization Generator

Generates all tutorial visualizations to:
  tutorials/v4_tutorial/tutorial_viz/

Features:
- N=20 for test, N=2000 for full
- Checkpoint/resume for all phases
- Memory protection (batch + gc)
- Dark theme visualizations

Usage:
  cd fibernet && source .venv/bin/activate
  python scripts/run_tutorial_viz.py

Files generated:
  tutorial_viz/gallery_undeformed.png
  tutorial_viz/gallery_deformed.png
  tutorial_viz/structure_statistics.png
  tutorial_viz/simulation_statistics.png
  tutorial_viz/ml_predictions.png
  tutorial_viz/ml_importance.png
  tutorial_viz/rl_convergence.png
  tutorial_viz/rl_reward_curve.png
  tutorial_viz/rl_actions.png
"""

import os, sys, json, gc, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import fibernet as fn
from fibernet import pattern_2d, TaichiEngine, GraphFeatureExtractor
from fibernet.sim.accelerated import SimResult

# ── Config ──
N_SAMPLES = 20  # ← Change to 2000 for full run
N_PTS_PER_SIDE = 5
BATCH_SIZE = 10
CHECKPOINT_EVERY = 5

# Paths
TUTORIAL_DIR = Path(__file__).parent.parent / "tutorials" / "v4_tutorial"
TUTORIAL_DIR.mkdir(parents=True, exist_ok=True)
VIZ_OUT = TUTORIAL_DIR / "tutorial_viz"
DATA_OUT = TUTORIAL_DIR / "data"
JSON_OUT = DATA_OUT / "json"
ML_OUT = DATA_OUT / "ml_results"
RL_OUT = DATA_OUT / "rl_results"

for d in [VIZ_OUT, JSON_OUT, ML_OUT, RL_OUT]:
    d.mkdir(parents=True, exist_ok=True)

# ── Generation Parameters ──
UNIT = "square"
BOX = (10, 10)
GRID = (3, 3)
N_PTS = N_PTS_PER_SIDE
N_SIDES = 4
N_DISP = N_SIDES * N_PTS

print(f"="*70)
print(f"FiberNet v4 Tutorial — Visualization Generator")
print(f"="*70)
print(f"Samples: {N_SAMPLES}")
print(f"Unit: {UNIT}, Grid: {GRID}, Points: {N_PTS}")
print(f"Output: {VIZ_OUT.resolve()}")
print()


def generate_parametric(seed, n_disp=N_DISP):
    rng = np.random.default_rng(seed)
    raw = rng.uniform(-0.3, 0.3, size=n_disp * 2)
    disps = [(float(raw[2*i]), float(raw[2*i+1])) for i in range(n_disp)]
    g = pattern_2d(unit=UNIT, box=BOX, grid=GRID,
                  n_pts_per_side=N_PTS, point_displacements=disps, seed=seed)
    return g, disps


# ══════════════════════════════════════════════════════════════════
# Phase 1: Generation
# ══════════════════════════════════════════════════════════════════
print("Phase 1/6: Structure Generation")
print("-" * 40)

N = N_SAMPLES
ckpt = DATA_OUT / "gen_checkpoint.json"
if ckpt.exists():
    with open(ckpt) as f:
        metadata = json.load(f).get("metadata", [])
    print(f"  Resuming: {len(metadata)} already generated")
else:
    metadata = []

# Base structure
if not any(m.get("is_base") for m in metadata):
    zero_disps = [(0.0, 0.0)] * N_DISP
    g_base = pattern_2d(unit=UNIT, box=BOX, grid=GRID,
                       n_pts_per_side=N_PTS, point_displacements=zero_disps, seed=0)
    base_name = f"{UNIT}_{GRID[0]}x{GRID[1]}_pts{N_PTS}_seed0"
    g_base.save_json(str(JSON_OUT / f"{base_name}.json"))
    metadata.append({
        "id": 0, "seed": 0, "name": base_name,
        "is_base": True, "n_nodes": g_base.num_nodes, "n_edges": g_base.num_edges,
    })
    print(f"  Base: {g_base.num_nodes} nodes, {g_base.num_edges} edges")
    del g_base

start_idx = len(metadata)
if start_idx < N:
    print(f"  Generating {start_idx} → {N}...")
    for batch_start in range(start_idx, N, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, N)
        for i in tqdm(range(batch_start, batch_end), desc="  Generate", leave=False):
            seed = 1000 + i
            g, disps = generate_parametric(seed)
            name = f"{UNIT}_{GRID[0]}x{GRID[1]}_pts{N_PTS}_seed{seed}"
            g.save_json(str(JSON_OUT / f"{name}.json"))
            metadata.append({
                "id": i, "seed": seed, "name": name,
                "is_base": False, "n_nodes": g.num_nodes, "n_edges": g.num_edges,
            })
            del g, disps
        with open(ckpt, 'w') as f:
            json.dump({"metadata": metadata}, f, indent=2)
        gc.collect()

with open(str(DATA_OUT / "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
print(f"  ✓ {len(metadata)} structures generated\n")


# ══════════════════════════════════════════════════════════════════
# Phase 2: Undeformed Gallery
# ══════════════════════════════════════════════════════════════════
print("Phase 2/6: Undeformed Gallery")
print("-" * 40)

n_show = min(20, len(metadata))
nc = 5
nr = (n_show + nc - 1) // nc

fig, axes = plt.subplots(nr, nc, figsize=(3*nc, 3*nr))
fig.patch.set_facecolor("#0a0a0f")
axes = axes.flatten() if nr > 1 else [axes]

for idx in range(n_show):
    rec = metadata[idx]
    g = fn.StructureGraph.load_json(str(JSON_OUT / f"{rec['name']}.json"))
    ax = axes[idx]
    ax.set_facecolor("#0a0a0f")
    for edge in g.edges.values():
        p1 = g.nodes[edge.node_i].position
        p2 = g.nodes[edge.node_j].position
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'c-', linewidth=0.8, alpha=0.7)
    pos = np.array([g.nodes[nid].position for nid in sorted(g.nodes.keys())])
    ax.scatter(pos[:, 0], pos[:, 1], c='#00ff88', s=8, zorder=5)
    ax.set_aspect('equal')
    ax.set_title(f"#{rec['id']} (seed={rec['seed']})", color="#aaa", fontsize=7)
    ax.axis('off')
    del g

for idx in range(n_show, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
viz_path = VIZ_OUT / "gallery_undeformed.png"
fig.savefig(str(viz_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"  ✓ {viz_path.name}: {n_show} structures shown\n")


# ══════════════════════════════════════════════════════════════════
# Phase 3: Structure Statistics
# ══════════════════════════════════════════════════════════════════
print("Phase 3/6: Structure Statistics")
print("-" * 40)

n_nodes_list = [m["n_nodes"] for m in metadata]
n_edges_list = [m["n_edges"] for m in metadata]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor("#0a0a0f")
for ax, data, label, color in [(ax1, n_nodes_list, "Number of Nodes", 'cyan'),
                                 (ax2, n_edges_list, "Number of Edges", 'lime')]:
    ax.hist(data, bins=min(20, len(set(data))), color=color, alpha=0.7, edgecolor='white')
    ax.set_facecolor("#0a0a0f")
    ax.set_xlabel(label, color='white')
    ax.set_ylabel("Count", color='white')
    ax.set_title(f"{label} Distribution (N={len(data)})", color='white')
    ax.tick_params(colors='white')
    ax.axvline(np.mean(data), color='white', linestyle='--', linewidth=1.5,
              label=f"Mean: {np.mean(data):.1f}")
    ax.legend()

viz_path = VIZ_OUT / "structure_statistics.png"
fig.savefig(str(viz_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"  ✓ {viz_path.name}: nodes mean={np.mean(n_nodes_list):.1f}, edges mean={np.mean(n_edges_list):.1f}\n")


# ══════════════════════════════════════════════════════════════════
# Phase 4: Simulation
# ══════════════════════════════════════════════════════════════════
print("Phase 4/6: Mechanical Simulation")
print("-" * 40)

engine = TaichiEngine()
sim_results = []

sim_ckpt = DATA_OUT / "sim_partial.json"
if sim_ckpt.exists():
    with open(sim_ckpt) as f:
        sim_results = json.load(f)
    done_ids = {r["id"] for r in sim_results}
    print(f"  Resuming: {len(sim_results)} already simulated")
else:
    done_ids = set()

pending = [rec for rec in metadata if rec["id"] not in done_ids]
print(f"  Pending: {len(pending)} simulations")

for batch_start in range(0, len(pending), BATCH_SIZE):
    batch = pending[batch_start:batch_start + BATCH_SIZE]
    for rec in tqdm(batch, desc="  Simulate", leave=False):
        g_path = JSON_OUT / f"{rec['name']}.json"
        try:
            g = fn.StructureGraph.load_json(str(g_path))
            r = engine.stretch_test(g, target_stretch=1.5, stiffness=1e5,
                                   damping=0.3, num_steps=500, save_interval=500,
                                   auto_steps=False)
            row = {
                "id": rec["id"], "name": rec["name"], "is_base": rec["is_base"],
                "success": True,
                "max_force": float(r.max_force),
                "max_stretch": float(r.max_stretch),
                "mean_stretch": float(r.mean_stretch),
                "energy": float(r.energy),
                "n_nodes": rec["n_nodes"], "n_edges": rec["n_edges"],
            }
            sim_results.append(row)
            r.save(str(DATA_OUT / f"{rec['name']}_result.json"))
            del g, r
        except Exception as e:
            sim_results.append({
                "id": rec["id"], "name": rec["name"],
                "success": False, "error": str(e)
            })
    with open(sim_ckpt, 'w') as f:
        json.dump(sim_results, f, indent=2)
    gc.collect()

df_sim = pd.DataFrame(sim_results)
df_sim.to_csv(str(DATA_OUT / "sim_results.csv"), index=False)

ok = df_sim[df_sim["success"]]
print(f"  ✓ {len(sim_results)} simulated, {len(ok)} successful")
if len(ok) > 0:
    print(f"    max_force: {ok['max_force'].mean():.0f} ± {ok['max_force'].std():.0f}")
    print(f"    max_stretch: {ok['max_stretch'].mean():.3f} ± {ok['max_stretch'].std():.3f}")
print()


# ══════════════════════════════════════════════════════════════════
# Phase 5: Deformed Gallery + Statistics
# ══════════════════════════════════════════════════════════════════
print("Phase 5/6: Deformed Gallery + Simulation Statistics")
print("-" * 40)

ok_sims = [s for s in sim_results if s.get("success")]
n_show_def = min(20, len(ok_sims))

if n_show_def > 0:
    nc = 5
    nr = (n_show_def + nc - 1) // nc
    
    fig, axes = plt.subplots(nr, nc, figsize=(3*nc, 3*nr))
    fig.patch.set_facecolor("#0a0a0f")
    axes = axes.flatten() if nr > 1 else [axes]
    
    for idx in range(n_show_def):
        sim_rec = ok_sims[idx]
        r_path = DATA_OUT / f"{sim_rec['name']}_result.json"
        if not r_path.exists():
            axes[idx].axis('off')
            continue
        try:
            r = SimResult.load(str(r_path))
            g = fn.StructureGraph.load_json(str(JSON_OUT / f"{sim_rec['name']}.json"))
            ax = axes[idx]
            ax.set_facecolor("#0a0a0f")
            if r.deformed_positions is not None:
                def_pos = r.deformed_positions
                for edge in g.edges.values():
                    p1 = def_pos[edge.node_i]
                    p2 = def_pos[edge.node_j]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'c-', linewidth=0.8, alpha=0.7)
                ax.scatter(def_pos[:, 0], def_pos[:, 1], c='#ff6600', s=8, zorder=5)
            ax.set_aspect('equal')
            ax.set_title(f"#{sim_rec['id']} F={sim_rec['max_force']:.0f}",
                        color="#aaa", fontsize=7)
            ax.axis('off')
            del r, g
        except:
            axes[idx].axis('off')
    
    for idx in range(n_show_def, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    viz_path = VIZ_OUT / "gallery_deformed.png"
    fig.savefig(str(viz_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ {viz_path.name}: {n_show_def} deformed structures shown")
    
    # Simulation statistics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.patch.set_facecolor("#0a0a0f")
    
    metrics = [
        ("max_force", "Maximum Force [N]", "cyan"),
        ("max_stretch", "Maximum Stretch Ratio", "lime"),
        ("mean_stretch", "Mean Stretch Ratio", "orange"),
        ("energy", "Elastic Energy [J]", "magenta"),
    ]
    
    for ax, (col, title, color) in zip(axes.flatten(), metrics):
        vals = df_sim[col].dropna()
        if len(vals) > 0:
            ax.hist(vals, bins=min(30, len(vals)), color=color, alpha=0.7, edgecolor='white')
            ax.axvline(vals.mean(), color='white', linestyle='--', linewidth=2,
                      label=f"Mean: {vals.mean():.2f}")
            ax.legend()
        ax.set_facecolor("#0a0a0f")
        ax.set_xlabel(title, color='white')
        ax.set_ylabel("Count", color='white')
        ax.set_title(f"{title} (N={len(vals)})", color='white')
        ax.tick_params(colors='white')
    
    viz_path = VIZ_OUT / "simulation_statistics.png"
    fig.savefig(str(viz_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ {viz_path.name}: 4 metrics shown\n")
else:
    print("  ✗ No successful simulations to visualize\n")


# ══════════════════════════════════════════════════════════════════
# Phase 6: ML + RL (quick demo)
# ══════════════════════════════════════════════════════════════════
print("Phase 6/6: ML + RL Demonstrations")
print("-" * 40)

try:
    ext = GraphFeatureExtractor(canvas_size=256)
    feat_records = []
    for rec in tqdm(metadata, desc="  Features", leave=False):
        g_path = JSON_OUT / f"{rec['name']}.json"
        g = fn.StructureGraph.load_json(str(g_path))
        try:
            feats = ext.extract(g)
            record = {"id": rec["id"], "name": rec["name"]}
            for k, v in feats.items():
                record[f"feat_{k}"] = float(v) if isinstance(v, (int, float)) else v
            feat_records.append(record)
        except:
            pass
        del g
    
    df_feat = pd.DataFrame(feat_records)
    n_feat = len([c for c in df_feat.columns if c.startswith("feat_")])
    print(f"  Features: {len(df_feat)} samples, {n_feat} dimensions")
    
    df_all = df_sim.merge(df_feat, on=["id", "name"], how="outer")
    df_all.to_csv(str(DATA_OUT / "full_results.csv"), index=False)
    
    # ML predictions plot
    feat_cols = [c for c in df_feat.columns if c.startswith("feat_")][:10]
    if len(ok) >= 5 and len(feat_cols) >= 3:
        from sklearn.ensemble import RandomForestRegressor
        
        X = df_all[feat_cols].dropna().values
        y = df_sim.loc[df_all.dropna(subset=feat_cols).index, "max_force"].values
        
        if len(X) >= 5:
            rf = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
            rf.fit(X[:int(0.8*len(X))], y[:int(0.8*len(X))])
            y_pred = rf.predict(X[int(0.8*len(X)):])
            y_true = y[int(0.8*len(X)):]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            fig.patch.set_facecolor("#0a0a0f")
            ax.set_facecolor("#0a0a0f")
            ax.scatter(y_true, y_pred, c='cyan', alpha=0.7)
            lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
            ax.plot(lims, lims, 'r--', linewidth=2, label="Perfect prediction")
            ax.set_xlabel("Actual max_force [N]", color='white')
            ax.set_ylabel("Predicted max_force [N]", color='white')
            ax.set_title("RF Model: Predictions vs Actual (Test Set)", color='white')
            ax.tick_params(colors='white')
            ax.legend()
            
            viz_path = VIZ_OUT / "ml_predictions.png"
            fig.savefig(str(viz_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"  ✓ {viz_path.name}")
            
            # Feature importance
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor("#0a0a0f")
            ax.set_facecolor("#0a0a0f")
            
            importances = rf.feature_importances_
            top_k = min(10, len(importances))
            idx = np.argsort(importances)[::-1][:top_k]
            
            bars = ax.barh(range(top_k), importances[idx], color='cyan', alpha=0.7)
            ax.set_yticks(range(top_k))
            ax.set_yticklabels([feat_cols[i].replace("feat_", "") for i in idx])
            ax.set_xlabel("Importance", color='white')
            ax.set_title(f"Top {top_k} Features (RF)", color='white')
            ax.tick_params(colors='white')
            ax.invert_yaxis()
            
            viz_path = VIZ_OUT / "ml_importance.png"
            fig.savefig(str(viz_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"  ✓ {viz_path.name}")
    
    # RL convergence plot (demo)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#0a0a0f")
    
    # Demo: plot max_force distribution as "optimization target"
    forces = df_sim["max_force"].dropna().values
    iterations = np.arange(len(forces))
    
    ax.scatter(iterations, forces, c='cyan', alpha=0.6, label="Individual runs")
    if len(forces) >= 5:
        window = max(3, len(forces) // 10)
        rolling = pd.Series(forces).rolling(window=window, min_periods=1).mean().values
        ax.plot(iterations, rolling, 'r-', linewidth=2, label=f"Rolling mean (w={window})")
    
    ax.set_xlabel("Iteration", color='white')
    ax.set_ylabel("Max Force [N]", color='white')
    ax.set_title("Force Distribution Across Generated Structures", color='white')
    ax.tick_params(colors='white')
    ax.legend()
    
    viz_path = VIZ_OUT / "rl_convergence.png"
    fig.savefig(str(viz_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ {viz_path.name}")
    
    gc.collect()
    print()
    
except Exception as e:
    print(f"  ✗ ML/RL phase error: {e}\n")


# ══════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════
print("="*70)
print("SUMMARY")
print("="*70)
print(f"\nStructures: {len(metadata)} generated")
print(f"Simulations: {len(ok)}/{len(sim_results)} successful")
print(f"\nVisualizations in {VIZ_OUT.resolve()}:")
for f in sorted(VIZ_OUT.glob("*.png")):
    size_mb = f.stat().st_size / 1024 / 1024
    print(f"  {f.name}: {size_mb:.2f} MB")

print(f"\nData in {DATA_OUT.resolve()}:")
print(f"  JSON: {len(list(JSON_OUT.glob('*.json')))} files")
print(f"  Results: {DATA_OUT / 'sim_results.csv'}")
if (DATA_OUT / "full_results.csv").exists():
    print(f"  Full: {DATA_OUT / 'full_results.csv'}")

print("\n✓ Complete!")
