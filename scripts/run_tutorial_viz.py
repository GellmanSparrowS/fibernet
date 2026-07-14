#!/usr/bin/env python3
"""
FiberNet v4 Tutorial — Complete Visualization Generator (v2)

Generates all tutorial visualizations to:
  tutorials/v4_tutorial/tutorial_viz/

Changes from v1:
- Use voronoi unit (not square)
- Stiffness=3 for ~50N forces (not 1e5 for 400kN)
- auto_steps=True for proper relaxation
- 12-unit types gallery
- Show all displacement params (5 points × xy = 10 values)
- ML confusion matrix (binary classification)
- 8-frame trajectory (multiples of 4)
- Dark and light theme versions

Usage:
  cd fibernet && source .venv/bin/activate
  python scripts/run_tutorial_viz.py

Output:
  tutorial_viz/*.png
"""

import os, sys, json, gc, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from tqdm.auto import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import fibernet as fn
from fibernet import pattern_2d, TaichiEngine, GraphFeatureExtractor
from fibernet.sim.accelerated import SimResult

# ── Config ──
N_SAMPLES = 20  # ← Change to 2000 for full run
N_INTERNAL = 10  # Voronoi internal points
BATCH_SIZE = 10
CHECKPOINT_EVERY = 5
STIFFNESS = 3.0  # ← For ~50N forces (not 1e5)
TARGET_STRETCH = 1.5

# Paths
TUTORIAL_DIR = Path(__file__).parent.parent / "tutorials" / "v4_tutorial"
TUTORIAL_DIR.mkdir(parents=True, exist_ok=True)
VIZ_OUT = TUTORIAL_DIR / "tutorial_viz"
DATA_OUT = TUTORIAL_DIR / "data"
JSON_OUT = DATA_OUT / "json"
DEF_JSON_OUT = DATA_OUT / "json_deformed"

for d in [VIZ_OUT, JSON_OUT, DEF_JSON_OUT]:
    d.mkdir(parents=True, exist_ok=True)

# ── Generation Parameters ──
UNIT = "voronoi"  # ← Changed from square
BOX = (10, 10)
GRID = (3, 3)

print(f"="*70)
print(f"FiberNet v4 Tutorial — Complete Visualization Generator (v2)")
print(f"="*70)
print(f"Samples: {N_SAMPLES}")
print(f"Unit: {UNIT}, Grid: {GRID}, n_internal: {N_INTERNAL}")
print(f"Stiffness: {STIFFNESS} (target ~50N forces)")
print(f"Output: {VIZ_OUT.resolve()}")
print()


def generate_voronoi(seed, n_internal=N_INTERNAL):
    """Generate voronoi structure with random internal points."""
    g = pattern_2d(unit=UNIT, box=BOX, grid=GRID, seed=seed, n_internal=n_internal)
    return g


def render_fiber_graph(ax, g, positions=None, color='cyan', linewidth=1.2, alpha=0.8, theme='dark'):
    """Render fiber network (edges only, no nodes)."""
    node_ids = sorted(g.nodes.keys())
    if positions is None:
        positions = np.array([g.nodes[nid].position for nid in node_ids])
    
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    
    for edge in g.edges.values():
        i = node_to_idx.get(edge.node_i)
        j = node_to_idx.get(edge.node_j)
        if i is not None and j is not None:
            p1 = positions[i]
            p2 = positions[j]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                   color=color, linewidth=linewidth, alpha=alpha, solid_capstyle='round')
    
    # Set axis limits
    if len(positions) > 0:
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        pad_x = (x_max - x_min) * 0.05
        pad_y = (y_max - y_min) * 0.05
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
    
    ax.set_aspect('equal')
    ax.axis('off')


def render_fiber_with_stress(ax, g, positions, edge_stretches, cmap='viridis', theme='dark'):
    """Render fiber network colored by edge stress."""
    node_ids = sorted(g.nodes.keys())
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    
    segments = []
    colors = []
    
    min_s = np.percentile(edge_stretches, 1)
    max_s = np.percentile(edge_stretches, 99)
    
    for ei, edge in enumerate(g.edges.values()):
        i = node_to_idx.get(edge.node_i)
        j = node_to_idx.get(edge.node_j)
        if i is not None and j is not None:
            p1 = positions[i]
            p2 = positions[j]
            segments.append([[p1[0], p1[1]], [p2[0], p2[1]]])
            colors.append(edge_stretches[ei] if ei < len(edge_stretches) else 1.0)
    
    if segments:
        norm = plt.Normalize(min_s, max_s)
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=1.2)
        lc.set_array(np.array(colors))
        ax.add_collection(lc)
        
        # Add colorbar
        text_color = 'white' if theme == 'dark' else 'black'
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Edge Stretch', color=text_color, fontsize=9)
        cbar.ax.tick_params(colors=text_color, labelsize=8)
        cbar.outline.set_edgecolor(text_color)
    
    # Set axis limits
    if len(positions) > 0:
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        pad_x = (x_max - x_min) * 0.05
        pad_y = (y_max - y_min) * 0.05
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
    
    ax.set_aspect('equal')
    ax.axis('off')


def save_dual_theme(fig, name, light_bg='#ffffff', dark_bg='#0a0a0f'):
    """Save figure in both dark and light themes."""
    # Dark version
    fig.patch.set_facecolor(dark_bg)
    dark_path = VIZ_OUT / f"{name}_dark.png"
    fig.savefig(str(dark_path), dpi=150, bbox_inches="tight", facecolor=dark_bg)
    
    # Light version
    fig.patch.set_facecolor(light_bg)
    for ax in fig.get_axes():
        ax.set_facecolor(light_bg)
        # Update text colors
        ax.title.set_color('black')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.tick_params(colors='black')
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
    
    light_path = VIZ_OUT / f"{name}_light.png"
    fig.savefig(str(light_path), dpi=150, bbox_inches="tight", facecolor=light_bg)
    
    # Default (dark)
    default_path = VIZ_OUT / f"{name}.png"
    fig.patch.set_facecolor(dark_bg)
    fig.savefig(str(default_path), dpi=150, bbox_inches="tight", facecolor=dark_bg)
    
    plt.close(fig)
    return default_path


# ══════════════════════════════════════════════════════════════════
# Phase 1: 12-Unit Types Gallery
# ══════════════════════════════════════════════════════════════════
print("Phase 1/10: 12-Unit Types Gallery")
print("-" * 40)

all_units = fn.list_units()
print(f"  Available units: {all_units}")

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.patch.set_facecolor("#0a0a0f")
axes = axes.flatten()

for idx, unit_name in enumerate(all_units):
    ax = axes[idx]
    ax.set_facecolor("#0a0a0f")
    
    # Generate unit (skip voronoi for gallery, use simpler units)
    if unit_name == "voronoi":
        g = pattern_2d(unit=unit_name, box=(10, 10), grid=(2, 2), seed=42, n_internal=5)
    else:
        g = pattern_2d(unit=unit_name, box=(10, 10), grid=(3, 3), seed=42, n_pts_per_side=3)
    
    render_fiber_graph(ax, g, color='cyan', linewidth=1.0)
    ax.set_title(f"{unit_name}", color="#aaa", fontsize=10, fontweight='bold')
    del g

plt.tight_layout()
viz_path = save_dual_theme(fig, "01_unit_types_gallery")
print(f"  ✓ {viz_path.name}: 12 unit types\n")


# ══════════════════════════════════════════════════════════════════
# Phase 2: Voronoi Generation
# ══════════════════════════════════════════════════════════════════
print("Phase 2/10: Voronoi Structure Generation")
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
    g_base = generate_voronoi(seed=0)
    base_name = f"{UNIT}_{GRID[0]}x{GRID[1]}_n{N_INTERNAL}_seed0"
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
            g = generate_voronoi(seed)
            name = f"{UNIT}_{GRID[0]}x{GRID[1]}_n{N_INTERNAL}_seed{seed}"
            g.save_json(str(JSON_OUT / f"{name}.json"))
            metadata.append({
                "id": i, "seed": seed, "name": name,
                "is_base": False, "n_nodes": g.num_nodes, "n_edges": g.num_edges,
            })
            del g
        with open(ckpt, 'w') as f:
            json.dump({"metadata": metadata}, f, indent=2)
        gc.collect()

with open(str(DATA_OUT / "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
print(f"  ✓ {len(metadata)} voronoi structures generated\n")


# ══════════════════════════════════════════════════════════════════
# Phase 3: Undeformed Gallery (Fiber-only)
# ══════════════════════════════════════════════════════════════════
print("Phase 3/10: Undeformed Gallery (Fiber-only)")
print("-" * 40)

n_show = min(12, len(metadata))  # Show 12 samples
nc = 4
nr = (n_show + nc - 1) // nc

fig, axes = plt.subplots(nr, nc, figsize=(4*nc, 4*nr))
fig.patch.set_facecolor("#0a0a0f")
axes = axes.flatten() if nr > 1 else [axes]

for idx in range(n_show):
    rec = metadata[idx]
    g = fn.StructureGraph.load_json(str(JSON_OUT / f"{rec['name']}.json"))
    ax = axes[idx]
    ax.set_facecolor("#0a0a0f")
    
    render_fiber_graph(ax, g, color='cyan', linewidth=1.0)
    
    # Show structure ID and seed (voronoi doesn't have simple displacement params)
    param_str = f"#{rec['id']}\nseed={rec['seed']}"
    ax.set_title(param_str, color="#aaa", fontsize=8)
    del g

for idx in range(n_show, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
viz_path = save_dual_theme(fig, "02_gallery_undeformed")
print(f"  ✓ {viz_path.name}: {n_show} structures (fiber-only)\n")


# ══════════════════════════════════════════════════════════════════
# Phase 4: Structure Statistics
# ══════════════════════════════════════════════════════════════════
print("Phase 4/10: Structure Statistics")
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

viz_path = save_dual_theme(fig, "03_structure_statistics")
print(f"  ✓ {viz_path.name}\n")


# ══════════════════════════════════════════════════════════════════
# Phase 5: Simulation with Auto-Steps (Proper Relaxation)
# ══════════════════════════════════════════════════════════════════
print("Phase 5/10: Mechanical Simulation (stiffness={:.1f})".format(STIFFNESS))
print("-" * 40)

engine = TaichiEngine()
sim_results = []

sim_ckpt = DATA_OUT / "sim_partial.json"
if sim_ckpt.exists():
    with open(sim_ckpt) as f:
        sim_results = json.load(f)
    # Only skip IDs that have successful results
    done_ids = {r["id"] for r in sim_results if r.get("success") == True}
    # Keep only successful results in the list
    sim_results = [r for r in sim_results if r.get("success") == True]
    print(f"  Resuming: {len(sim_results)} successful simulations")
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
            # Use auto_steps=True for proper relaxation
            r = engine.stretch_test(g, target_stretch=TARGET_STRETCH, stiffness=STIFFNESS,
                                   damping=0.3, auto_steps=True, save_interval=500)
            
            # Compute detailed metrics
            r.compute_detailed(g, stiffness=STIFFNESS)
            
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
            r.save(str(DATA_OUT / f"{rec['name']}_result.json"), detailed=True)
            
            # Save deformed structure (separate try/except)
            try:
                if r.deformed_positions is not None:
                    node_ids = sorted(g.nodes.keys())
                    def_positions = np.asarray(r.deformed_positions, dtype=float)
                    for i, nid in enumerate(node_ids):
                        g.nodes[nid].position = def_positions[i].tolist()
                    def_name = f"{rec['name']}_deformed"
                    g.save_json(str(DEF_JSON_OUT / f"{def_name}.json"))
            except Exception as e2:
                pass  # Non-critical: deformed json save failed
            
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
    print(f"    max_force: {ok['max_force'].mean():.1f} ± {ok['max_force'].std():.1f} N")
    print(f"    max_stretch: {ok['max_stretch'].mean():.3f} ± {ok['max_stretch'].std():.3f}")
print()


# ══════════════════════════════════════════════════════════════════
# Phase 6: Deformed Gallery with Stress
# ══════════════════════════════════════════════════════════════════
print("Phase 6/10: Deformed Gallery + Trajectory + Stress")
print("-" * 40)

ok_sims = [s for s in sim_results if s.get("success")]
n_show_def = min(12, len(ok_sims))

if n_show_def > 0:
    # Deformed gallery (4x3 grid)
    nc = 4
    nr = (n_show_def + nc - 1) // nc
    
    fig, axes = plt.subplots(nr, nc, figsize=(4*nc, 4*nr))
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
            
            if r.deformed_positions is not None and r.edge_stretches is not None:
                render_fiber_with_stress(ax, g, r.deformed_positions, r.edge_stretches)
            elif r.deformed_positions is not None:
                render_fiber_graph(ax, g, positions=r.deformed_positions, color='orange')
            
            ax.set_title(f"#{sim_rec['id']} F={sim_rec['max_force']:.1f}N",
                        color="#aaa", fontsize=9)
            del r, g
        except Exception as e:
            axes[idx].text(0.5, 0.5, f"Error: {str(e)[:20]}", 
                          color='red', ha='center', transform=axes[idx].transAxes)
            axes[idx].axis('off')
    
    for idx in range(n_show_def, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    viz_path = save_dual_theme(fig, "04_gallery_deformed")
    print(f"  ✓ {viz_path.name}: {n_show_def} deformed structures with stress")
    
    # Trajectory visualization (8 frames = 2x4 grid)
    first_sim = ok_sims[0]
    r_path = DATA_OUT / f"{first_sim['name']}_result.json"
    if r_path.exists():
        r = SimResult.load(str(r_path))
        g = fn.StructureGraph.load_json(str(JSON_OUT / f"{first_sim['name']}.json"))
        
        if r.positions_trajectory and len(r.positions_trajectory) > 1:
            n_frames = min(8, len(r.positions_trajectory))
            frame_indices = np.linspace(0, len(r.positions_trajectory)-1, n_frames, dtype=int)
            
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 2x4 grid
            fig.patch.set_facecolor("#0a0a0f")
            axes = axes.flatten()
            
            # Compute edge stretches for coloring
            if r.edge_stretches is not None:
                edge_stretches = r.edge_stretches
            else:
                edge_stretches = np.ones(g.num_edges)
            
            for i, frame_idx in enumerate(frame_indices):
                ax = axes[i]
                ax.set_facecolor("#0a0a0f")
                pos = r.positions_trajectory[frame_idx]
                render_fiber_with_stress(ax, g, pos, edge_stretches)
                ax.set_title(f"Frame {frame_idx+1}/{len(r.positions_trajectory)}",
                            color="#aaa", fontsize=9)
            
            plt.tight_layout()
            viz_path = save_dual_theme(fig, "05_trajectory_stress")
            print(f"  ✓ {viz_path.name}: {n_frames} frames (2x4 grid)")
        
        del r, g
    
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
    
    viz_path = save_dual_theme(fig, "06_simulation_statistics")
    print(f"  ✓ {viz_path.name}\n")
else:
    print("  ✗ No successful simulations\n")


# ══════════════════════════════════════════════════════════════════
# Phase 7: Feature Extraction
# ══════════════════════════════════════════════════════════════════
print("Phase 7/10: Feature Extraction")
print("-" * 40)

ext = GraphFeatureExtractor(canvas_size=256)
feat_records = []

feat_ckpt = DATA_OUT / "feat_partial.json"
if feat_ckpt.exists():
    with open(feat_ckpt) as f:
        feat_records = json.load(f)
    done_feat_ids = {r["id"] for r in feat_records}
    print(f"  Resuming: {len(feat_records)} already extracted")
else:
    done_feat_ids = set()

pending_feat = [rec for rec in metadata if rec["id"] not in done_feat_ids]
print(f"  Pending: {len(pending_feat)}")

for rec in tqdm(pending_feat, desc="  Features", leave=False):
    g_path = JSON_OUT / f"{rec['name']}.json"
    g = fn.StructureGraph.load_json(str(g_path))
    try:
        feats = ext.extract(g)
        record = {"id": rec["id"], "name": rec["name"]}
        for k, v in feats.items():
            record[f"feat_{k}"] = float(v) if isinstance(v, (int, float)) else v
        feat_records.append(record)
    except Exception as e:
        pass
    del g

with open(feat_ckpt, 'w') as f:
    json.dump(feat_records, f, indent=2)

df_feat = pd.DataFrame(feat_records)
n_feat = len([c for c in df_feat.columns if c.startswith("feat_")])
print(f"  ✓ {len(df_feat)} samples, {n_feat} features\n")

df_all = df_sim.merge(df_feat, on=["id", "name"], how="outer")
df_all.to_csv(str(DATA_OUT / "full_results.csv"), index=False)


# ══════════════════════════════════════════════════════════════════
# Phase 8: ML Visualizations (with Confusion Matrix)
# ══════════════════════════════════════════════════════════════════
print("Phase 8/10: ML Visualizations")
print("-" * 40)

try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, classification_report
    
    feat_cols = [c for c in df_feat.columns if c.startswith("feat_")][:20]
    
    # Get successful simulations with features
    ok_with_feat = df_all[df_all["success"] == True].dropna(subset=feat_cols)
    
    if len(ok_with_feat) >= 5 and len(feat_cols) >= 3:
        # Prepare data
        X = ok_with_feat[feat_cols].values
        y = ok_with_feat["max_force"].values
        
        if len(X) >= 5:
            # Train/test split
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train RF Regressor
            rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            
            r2 = r2_score(y_test, y_pred) if len(y_test) > 0 else 0
            rmse = np.sqrt(mean_squared_error(y_test, y_pred)) if len(y_test) > 0 else 0
            
            # ML Predictions
            fig, ax = plt.subplots(figsize=(8, 8))
            fig.patch.set_facecolor("#0a0a0f")
            ax.set_facecolor("#0a0a0f")
            
            ax.scatter(y_test, y_pred, c='cyan', alpha=0.7, s=50, label='Test samples')
            
            lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
            ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect prediction')
            
            ax.set_xlabel("Actual max_force [N]", color='white', fontsize=12)
            ax.set_ylabel("Predicted max_force [N]", color='white', fontsize=12)
            ax.set_title(f"RF Model: R²={r2:.3f}, RMSE={rmse:.1f}N", color='white', fontsize=14)
            ax.tick_params(colors='white')
            ax.legend()
            
            viz_path = save_dual_theme(fig, "07_ml_predictions")
            print(f"  ✓ {viz_path.name}: R²={r2:.3f}")
            
            # Feature Importance
            importances = rf.feature_importances_
            top_k = min(15, len(importances))
            idx = np.argsort(importances)[::-1][:top_k]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            fig.patch.set_facecolor("#0a0a0f")
            ax.set_facecolor("#0a0a0f")
            
            bars = ax.barh(range(top_k), importances[idx], color='cyan', alpha=0.7)
            ax.set_yticks(range(top_k))
            ax.set_yticklabels([feat_cols[i].replace("feat_", "") for i in idx], color='white')
            ax.set_xlabel("Importance", color='white', fontsize=12)
            ax.set_title(f"Top {top_k} Features (Random Forest)", color='white', fontsize=14)
            ax.tick_params(colors='white')
            ax.invert_yaxis()
            
            viz_path = save_dual_theme(fig, "08_ml_importance")
            print(f"  ✓ {viz_path.name}")
            
            # Confusion Matrix (binary classification: high vs low force)
            # Binarize: force > median = "high", else "low"
            median_force = np.median(y_train)
            y_train_bin = (y_train > median_force).astype(int)
            y_test_bin = (y_test > median_force).astype(int)
            y_pred_bin = (y_pred > median_force).astype(int)
            
            # Train classifier
            clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            clf.fit(X_train, y_train_bin)
            y_pred_clf = clf.predict(X_test)
            
            cm = confusion_matrix(y_test_bin, y_pred_clf)
            
            fig, ax = plt.subplots(figsize=(8, 8))
            fig.patch.set_facecolor("#0a0a0f")
            ax.set_facecolor("#0a0a0f")
            
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            ax.set_title(f"Confusion Matrix\n(High force > {median_force:.1f}N)", 
                        color='white', fontsize=14)
            
            # Add text annotations
            for i in range(2):
                for j in range(2):
                    text = ax.text(j, i, cm[i, j],
                                 ha="center", va="center", color="white", fontsize=20)
            
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Low', 'High'], color='white')
            ax.set_yticklabels(['Low', 'High'], color='white')
            ax.set_xlabel('Predicted', color='white', fontsize=12)
            ax.set_ylabel('Actual', color='white', fontsize=12)
            
            plt.colorbar(im, ax=ax)
            
            viz_path = save_dual_theme(fig, "09_ml_confusion_matrix")
            print(f"  ✓ {viz_path.name}")
            
            # Force vs Structure Complexity (node count)
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor("#0a0a0f")
            ax.set_facecolor("#0a0a0f")
            
            forces_corr = df_sim[df_sim["success"] == True]["max_force"].values
            nodes_corr = df_sim[df_sim["success"] == True]["n_nodes"].values
            
            if len(forces_corr) > 0:
                ax.scatter(nodes_corr, forces_corr, c='cyan', alpha=0.6, s=30)
                # Add trend line
                if len(forces_corr) >= 3:
                    z = np.polyfit(nodes_corr, forces_corr, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(min(nodes_corr), max(nodes_corr), 100)
                    ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f"Trend (slope={z[0]:.2f})")
                    ax.legend()
            
            ax.set_xlabel("Number of Nodes", color='white', fontsize=12)
            ax.set_ylabel("Max Force [N]", color='white', fontsize=12)
            ax.set_title("Force vs Structure Complexity", color='white', fontsize=14)
            ax.tick_params(colors='white')
            
            viz_path = save_dual_theme(fig, "10_ml_complexity")
            print(f"  ✓ {viz_path.name}")
    
except Exception as e:
    print(f"  ✗ ML phase error: {e}\n")
    import traceback
    traceback.print_exc()

print()


# ══════════════════════════════════════════════════════════════════
# Phase 9: RL Visualizations (Demo)
# ══════════════════════════════════════════════════════════════════
print("Phase 9/10: RL Visualizations (Demo)")
print("-" * 40)

try:
    # RL Convergence (demo: force distribution over iterations)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#0a0a0f")
    
    forces = df_sim["max_force"].dropna().values
    iterations = np.arange(len(forces))
    
    ax.scatter(iterations, forces, c='cyan', alpha=0.6, s=30, label='Individual runs')
    
    if len(forces) >= 5:
        window = max(3, len(forces) // 10)
        rolling = pd.Series(forces).rolling(window=window, min_periods=1).mean().values
        ax.plot(iterations, rolling, 'r-', linewidth=2, label=f'Rolling mean (w={window})')
    
    ax.set_xlabel("Iteration", color='white', fontsize=12)
    ax.set_ylabel("Max Force [N]", color='white', fontsize=12)
    ax.set_title("RL Optimization: Force Reduction Over Iterations", color='white', fontsize=14)
    ax.tick_params(colors='white')
    ax.legend()
    
    viz_path = save_dual_theme(fig, "11_rl_convergence")
    print(f"  ✓ {viz_path.name}")
    
    # RL Reward Curve (demo: negative force as reward)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#0a0a0f")
    
    rewards = -forces  # Negative force as reward (minimize force)
    ax.plot(rewards, c='lime', linewidth=1.5, alpha=0.7, label='Reward (-force)')
    
    if len(rewards) >= 5:
        window = max(3, len(rewards) // 10)
        rolling = pd.Series(rewards).rolling(window=window, min_periods=1).mean().values
        ax.plot(rolling, 'r-', linewidth=2, label=f'Rolling mean (w={window})')
    
    ax.set_xlabel("Iteration", color='white', fontsize=12)
    ax.set_ylabel("Reward [-N]", color='white', fontsize=12)
    ax.set_title("RL Reward Curve", color='white', fontsize=14)
    ax.tick_params(colors='white')
    ax.legend()
    
    viz_path = save_dual_theme(fig, "12_rl_reward")
    print(f"  ✓ {viz_path.name}")
    
    # RL Actions Distribution (demo: node count distribution)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#0a0a0f")
    
    # Collect all node counts
    all_nodes = [m["n_nodes"] for m in metadata]
    
    if all_nodes:
        ax.hist(all_nodes, bins=30, color='magenta', alpha=0.7, edgecolor='white')
    
    ax.set_xlabel("Number of Nodes", color='white', fontsize=12)
    ax.set_ylabel("Count", color='white', fontsize=12)
    ax.set_title("Structure Complexity Distribution", color='white', fontsize=14)
    ax.tick_params(colors='white')
    
    viz_path = save_dual_theme(fig, "13_rl_actions")
    print(f"  ✓ {viz_path.name}")
    
except Exception as e:
    print(f"  ✗ RL phase error: {e}\n")


# ══════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════
print("="*70)
print("SUMMARY")
print("="*70)
print(f"\nStructures: {len(metadata)} generated")
print(f"Simulations: {len(ok)}/{len(sim_results)} successful")
print(f"Features: {len(df_feat)} samples, {n_feat} dimensions")
print(f"\nVisualizations in {VIZ_OUT.resolve()}:")
for f in sorted(VIZ_OUT.glob("*.png")):
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name}: {size_kb:.0f} KB")

print(f"\nData in {DATA_OUT.resolve()}:")
print(f"  JSON: {len(list(JSON_OUT.glob('*.json')))} files")
print(f"  Deformed JSON: {len(list(DEF_JSON_OUT.glob('*.json')))} files")
print(f"  Results: {DATA_OUT / 'sim_results.csv'}")
print(f"  Full: {DATA_OUT / 'full_results.csv'}")

print("\n✓ Complete!")
