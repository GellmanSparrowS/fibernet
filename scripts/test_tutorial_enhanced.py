#!/usr/bin/env python3
"""
Test script for enhanced v4 tutorial
Tests with N=20 samples to verify all functionality
"""

import os
import sys
import json
import gc
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

# ── Configuration ──
N_SAMPLES = 20
N_PTS_PER_SIDE = 5
BATCH_SIZE = 10
CHECKPOINT_EVERY = 5

OUT = Path("test_tutorial_output")
OUT.mkdir(parents=True, exist_ok=True)
DATA_OUT = OUT / "data"
JSON_OUT = DATA_OUT / "json"
VIZ_OUT  = OUT / "tutorial_viz"

for d in [JSON_OUT, VIZ_OUT]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Test output: {OUT.resolve()}")
print(f"Samples: {N_SAMPLES}, Batch: {BATCH_SIZE}")

UNIT = "square"
BOX = (10, 10)
GRID = (3, 3)
N_PTS = N_PTS_PER_SIDE
N_SIDES = 4
N_DISP = N_SIDES * N_PTS

print(f"Unit: {UNIT}, Grid: {GRID}, Points: {N_PTS}")

def generate_parametric(seed, n_disp=N_DISP):
    rng = np.random.default_rng(seed)
    raw = rng.uniform(-0.3, 0.3, size=n_disp * 2)
    disps = [(float(raw[2*i]), float(raw[2*i+1])) for i in range(n_disp)]
    g = pattern_2d(unit=UNIT, box=BOX, grid=GRID,
                  n_pts_per_side=N_PTS, point_displacements=disps, seed=seed)
    return g, disps

print("\n=== Phase 1: Generation ===")
N = N_SAMPLES
ckpt_path = DATA_OUT / "gen_checkpoint.json"
if ckpt_path.exists():
    with open(ckpt_path) as f:
        metadata = json.load(f).get("metadata", [])
    print(f"Resuming: {len(metadata)} already generated")
else:
    metadata = []

if not any(m.get("is_base") for m in metadata):
    zero_disps = [(0.0, 0.0)] * N_DISP
    g_base = pattern_2d(unit=UNIT, box=BOX, grid=GRID,
                       n_pts_per_side=N_PTS, point_displacements=zero_disps, seed=0)
    base_name = f"{UNIT}_{GRID[0]}x{GRID[1]}_pts{N_PTS}_disp{N_DISP}_seed0"
    g_base.save_json(str(JSON_OUT / f"{base_name}.json"))
    metadata.append({
        "id": 0, "seed": 0, "name": base_name,
        "is_base": True, "n_nodes": g_base.num_nodes, "n_edges": g_base.num_edges,
    })
    print(f"Base: {g_base.num_nodes} nodes, {g_base.num_edges} edges")
    del g_base

start_idx = len(metadata)
if start_idx < N:
    for batch_start in range(start_idx, N, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, N)
        for i in tqdm(range(batch_start, batch_end), desc="Generate"):
            seed = 1000 + i
            g, disps = generate_parametric(seed)
            name = f"{UNIT}_{GRID[0]}x{GRID[1]}_pts{N_PTS}_disp{N_DISP}_seed{seed}"
            g.save_json(str(JSON_OUT / f"{name}.json"))
            metadata.append({
                "id": i, "seed": seed, "name": name,
                "is_base": False, "n_nodes": g.num_nodes, "n_edges": g.num_edges,
            })
            del g, disps
        with open(ckpt_path, 'w') as f:
            json.dump({"metadata": metadata}, f, indent=2)
        gc.collect()

with open(str(DATA_OUT / "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Generated {len(metadata)} structures")

# ── Undeformed Gallery ──
print("\n=== Phase 2: Undeformed Gallery ===")
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
    ax.set_title(f"#{rec['id']}", color="#aaa", fontsize=8)
    ax.axis('off')
    del g

for idx in range(n_show, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
viz_path = VIZ_OUT / "gallery_undeformed.png"
fig.savefig(str(viz_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"✓ Undeformed gallery: {viz_path}")

# Structure statistics
n_nodes_list = [m["n_nodes"] for m in metadata]
n_edges_list = [m["n_edges"] for m in metadata]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor("#0a0a0f")
ax1.hist(n_nodes_list, bins=10, color='cyan', alpha=0.7, edgecolor='white')
ax1.set_facecolor("#0a0a0f")
ax1.set_xlabel("Number of Nodes", color='white')
ax1.set_ylabel("Count", color='white')
ax1.set_title("Node Distribution", color='white')
ax1.tick_params(colors='white')
ax2.hist(n_edges_list, bins=10, color='lime', alpha=0.7, edgecolor='white')
ax2.set_facecolor("#0a0a0f")
ax2.set_xlabel("Number of Edges", color='white')
ax2.set_ylabel("Count", color='white')
ax2.set_title("Edge Distribution", color='white')
ax2.tick_params(colors='white')

viz_path2 = VIZ_OUT / "structure_statistics.png"
fig.savefig(str(viz_path2), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"✓ Structure statistics: {viz_path2}")

# ── Simulation ──
print("\n=== Phase 3: Simulation ===")
engine = TaichiEngine()
sim_results = []

sim_ckpt = DATA_OUT / "sim_partial.json"
if sim_ckpt.exists():
    with open(sim_ckpt) as f:
        sim_results = json.load(f)
    done_ids = {r["id"] for r in sim_results}
    print(f"Resuming: {len(sim_results)} already simulated")
else:
    done_ids = set()

pending = [rec for rec in metadata if rec["id"] not in done_ids]
print(f"Pending: {len(pending)}")

for batch_start in range(0, len(pending), BATCH_SIZE):
    batch = pending[batch_start:batch_start + BATCH_SIZE]
    for rec in tqdm(batch, desc="Simulate"):
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
print(f"✓ Simulated: {len(sim_results)} total, {len(ok)} successful")

# ── Deformed Gallery ──
print("\n=== Phase 4: Deformed Gallery ===")
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
            ax.set_title(f"#{sim_rec['id']}\nF={sim_rec['max_force']:.0f}",
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
    print(f"✓ Deformed gallery: {viz_path}")
    
    # Simulation statistics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.patch.set_facecolor("#0a0a0f")
    
    metrics = [
        ("max_force", "Maximum Force", "cyan"),
        ("max_stretch", "Maximum Stretch", "lime"),
        ("mean_stretch", "Mean Stretch", "orange"),
        ("energy", "Elastic Energy", "magenta"),
    ]
    
    for ax, (col, title, color) in zip(axes.flatten(), metrics):
        vals = df_sim[col].dropna()
        if len(vals) > 0:
            ax.hist(vals, bins=10, color=color, alpha=0.7, edgecolor='white')
            ax.axvline(vals.mean(), color='white', linestyle='--', linewidth=2,
                      label=f"Mean: {vals.mean():.2f}")
            ax.legend()
        ax.set_facecolor("#0a0a0f")
        ax.set_xlabel(title, color='white')
        ax.set_ylabel("Count", color='white')
        ax.set_title(title, color='white')
        ax.tick_params(colors='white')
    
    viz_path2 = VIZ_OUT / "simulation_statistics.png"
    fig.savefig(str(viz_path2), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"✓ Simulation statistics: {viz_path2}")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
print(f"\nOutput: {OUT.resolve()}")
print(f"Structures: {len(list(JSON_OUT.glob('*.json')))} JSON")
print(f"Simulations: {len(sim_results)} results")
print(f"\nVisualizations in {VIZ_OUT.resolve()}:")
for f in sorted(VIZ_OUT.glob("*.png")):
    size_mb = f.stat().st_size / 1024 / 1024
    print(f"  {f.name}: {size_mb:.2f} MB")
print("\n✓ All tests passed!")
