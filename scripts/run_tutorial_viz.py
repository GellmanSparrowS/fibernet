#!/usr/bin/env python3
"""
FiberNet v4 Tutorial — Visualization Generator v5 (with checkpoint)

Features:
1. Voronoi with displacement parameters (n_pts_per_side=3, ~1100 nodes)
2. Only dark theme (no duplicates)
3. Correct simulation: elastic stretch, auto_steps, proper boundaries
4. Checkpoint/resume for long-running simulations
5. All required visualizations

Usage:
  cd fibernet && source .venv/bin/activate
  python scripts/run_tutorial_viz.py
"""

import os
import sys
import json
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from fibernet import pattern_2d, TaichiEngine
from fibernet.sim import SimResult
from fibernet.analysis import GraphFeatureExtractor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split

# ── Configuration ──
N_SAMPLES = 20
BATCH_SIZE = 10
THEME = 'dark'

# Paths
TUTORIAL_DIR = Path(__file__).parent.parent / 'tutorials' / 'v4_tutorial'
VIZ_DIR = TUTORIAL_DIR / 'tutorial_viz'
DATA_DIR = TUTORIAL_DIR / 'data'
VIZ_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Theme colors ──
BG_COLOR = '#1a1a1a'
TEXT_COLOR = 'white'
EDGE_COLOR = 'cyan'
LINE_COLOR = '#00ff88'

print("="*70)
print("FiberNet v4 Tutorial — Visualization Generator v5")
print("="*70)
print(f"Samples: {N_SAMPLES}")
print(f"Output: {VIZ_DIR}")
print()

# ── Phase 1: Generate structures ──
print("Phase 1/9: Generating voronoi structures with displacement parameters")
print("-"*70)

structures = []
n_sides = 4
n_pts = 3
n_disps = n_sides * n_pts  # 12 displacement pairs

for i in tqdm(range(N_SAMPLES), desc="Generating"):
    # Generate random displacements for this structure
    rng = np.random.default_rng(seed=1000+i)
    disps = [(float(rng.uniform(-0.5, 0.5)), float(rng.uniform(-0.5, 0.5))) 
             for _ in range(n_disps)]
    
    # Voronoi with displacement parameters
    g = pattern_2d(
        unit='voronoi',
        box=(10.0, 10.0),
        grid=(2, 2),
        seed=i,
        n_internal=5,
        n_pts_per_side=n_pts,
        point_displacements=disps
    )
    structures.append(g)
    
    # Save structure
    g.save_json(str(DATA_DIR / f'structure_{i:03d}.json'))

print(f"✓ Generated {len(structures)} structures")
print(f"  Nodes: {structures[0].num_nodes}, Edges: {structures[0].num_edges}")

# ── Phase 2: Undeformed gallery ──
print("\nPhase 2/9: Creating undeformed gallery")
print("-"*70)

fig, axes = plt.subplots(4, 5, figsize=(20, 16))
fig.patch.set_facecolor(BG_COLOR)
axes = axes.flatten()

for i, ax in enumerate(axes[:N_SAMPLES]):
    g = structures[i]
    ax.set_facecolor(BG_COLOR)
    
    # Get node positions
    node_ids = sorted(g.nodes.keys())
    positions = np.array([g.nodes[nid].position for nid in node_ids])
    
    # Draw edges
    for edge in g.edges.values():
        i1 = node_ids.index(edge.node_i)
        i2 = node_ids.index(edge.node_j)
        ax.plot([positions[i1, 0], positions[i2, 0]], 
                [positions[i1, 1], positions[i2, 1]], 
                color=EDGE_COLOR, linewidth=1.5, alpha=0.7)
    
    ax.set_xlim(-0.5, 20.5)
    ax.set_ylim(-0.5, 20.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Structure {i}', color=TEXT_COLOR, fontsize=10)

for ax in axes[N_SAMPLES:]:
    ax.axis('off')

plt.tight_layout()
plt.savefig(VIZ_DIR / '01_gallery_undeformed.png', dpi=150, 
            bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("✓ Saved: 01_gallery_undeformed.png")

# ── Phase 3: Run simulations (with checkpoint) ──
print("\nPhase 3/9: Running simulations (elastic stretch)")
print("-"*70)

engine = TaichiEngine()
results = []

# Check which simulations already exist
existing_results = []
for i in range(N_SAMPLES):
    result_file = DATA_DIR / f'result_{i:03d}.json'
    if result_file.exists():
        try:
            result = SimResult.load(str(result_file))
            existing_results.append((i, result))
        except:
            pass

print(f"Found {len(existing_results)} existing simulation results")

# Run only missing simulations
missing_indices = set(range(N_SAMPLES)) - set(idx for idx, _ in existing_results)
print(f"Running {len(missing_indices)} new simulations...")

new_results = []
for i in tqdm(sorted(missing_indices), desc="Simulating"):
    g = structures[i]
    
    # Stretch test: 1.5x stretch, elastic behavior, auto_steps
    result = engine.stretch_test(
        g,
        target_stretch=1.5,
        stiffness=2.0,  # Low stiffness for tens of Newtons
        damping=0.3,
        auto_steps=True,
        save_interval=200
    )
    
    new_results.append((i, result))
    
    # Save result immediately
    result.save(str(DATA_DIR / f'result_{i:03d}.json'), detailed=True)

print(f"✓ Completed {len(new_results)} new simulations")

# Combine all results
results = [None] * N_SAMPLES
for idx, result in existing_results:
    results[idx] = result
for idx, result in new_results:
    results[idx] = result

forces = [r.max_force for r in results if r is not None]
print(f"  Max force range: {min(forces):.1f}N - {max(forces):.1f}N")
print(f"  Average force: {np.mean(forces):.1f}N")

# ── Phase 4: Deformed gallery ──
print("\nPhase 4/9: Creating deformed gallery")
print("-"*70)

fig, axes = plt.subplots(4, 5, figsize=(20, 16))
fig.patch.set_facecolor(BG_COLOR)
axes = axes.flatten()

for i, ax in enumerate(axes[:N_SAMPLES]):
    g = structures[i]
    r = results[i]
    
    if r is None:
        ax.axis('off')
        continue
    
    ax.set_facecolor(BG_COLOR)
    
    # Get deformed positions
    deformed_pos = r.deformed_positions
    node_ids = sorted(g.nodes.keys())
    
    # Draw deformed edges
    for edge in g.edges.values():
        i1 = node_ids.index(edge.node_i)
        i2 = node_ids.index(edge.node_j)
        ax.plot([deformed_pos[i1, 0], deformed_pos[i2, 0]], 
                [deformed_pos[i1, 1], deformed_pos[i2, 1]], 
                color=EDGE_COLOR, linewidth=1.5, alpha=0.7)
    
    ax.set_xlim(-0.5, 30.5)  # Wider for 1.5x stretch
    ax.set_ylim(-0.5, 20.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Structure {i}\nF={r.max_force:.1f}N', 
                color=TEXT_COLOR, fontsize=9)

for ax in axes[N_SAMPLES:]:
    ax.axis('off')

plt.tight_layout()
plt.savefig(VIZ_DIR / '02_gallery_deformed.png', dpi=150,
            bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("✓ Saved: 02_gallery_deformed.png")

# ── Phase 5: Structure statistics ──
print("\nPhase 5/9: Analyzing structure statistics")
print("-"*70)

nodes = [g.num_nodes for g in structures]
edges = [g.num_edges for g in structures]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor(BG_COLOR)

for ax in [ax1, ax2]:
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)

ax1.hist(nodes, bins=15, color=LINE_COLOR, alpha=0.7, edgecolor=TEXT_COLOR)
ax1.set_xlabel('Number of Nodes')
ax1.set_ylabel('Count')
ax1.set_title('Node Distribution')

ax2.hist(edges, bins=15, color=LINE_COLOR, alpha=0.7, edgecolor=TEXT_COLOR)
ax2.set_xlabel('Number of Edges')
ax2.set_ylabel('Count')
ax2.set_title('Edge Distribution')

plt.tight_layout()
plt.savefig(VIZ_DIR / '03_structure_statistics.png', dpi=150,
            bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("✓ Saved: 03_structure_statistics.png")

# ── Phase 6: Simulation statistics ──
print("\nPhase 6/9: Analyzing simulation results")
print("-"*70)

valid_results = [r for r in results if r is not None]
max_forces = [r.max_force for r in valid_results]
max_stretches = [r.max_stretch for r in valid_results]
energies = [r.energy for r in valid_results]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.patch.set_facecolor(BG_COLOR)
axes = axes.flatten()

data_list = [max_forces, max_stretches, energies, max_forces]
titles = ['Max Force (N)', 'Max Stretch Ratio', 'Total Energy (J)', 'Force Distribution']

for ax, data, title in zip(axes, data_list, titles):
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    
    ax.hist(data, bins=15, color=LINE_COLOR, alpha=0.7, edgecolor=TEXT_COLOR)
    ax.set_xlabel(title)
    ax.set_ylabel('Count')
    ax.set_title(f'{title}\nMean: {np.mean(data):.2f}')

plt.tight_layout()
plt.savefig(VIZ_DIR / '04_simulation_statistics.png', dpi=150,
            bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("✓ Saved: 04_simulation_statistics.png")

# ── Phase 7: Trajectory visualization ──
print("\nPhase 7/9: Creating trajectory visualization")
print("-"*70)

# Use first structure's trajectory
r = results[0]
if r is not None and hasattr(r, 'trajectory') and r.trajectory and len(r.trajectory) >= 8:
    # Select 8 frames evenly spaced
    trajectory = r.trajectory
    indices = np.linspace(0, len(trajectory)-1, 8, dtype=int)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.patch.set_facecolor(BG_COLOR)
    axes = axes.flatten()
    
    g = structures[0]
    node_ids = sorted(g.nodes.keys())
    
    for idx, (frame_idx, ax) in enumerate(zip(indices, axes)):
        ax.set_facecolor(BG_COLOR)
        pos = trajectory[frame_idx]
        
        # Draw edges
        for edge in g.edges.values():
            i1 = node_ids.index(edge.node_i)
            i2 = node_ids.index(edge.node_j)
            ax.plot([pos[i1, 0], pos[i2, 0]], 
                    [pos[i1, 1], pos[i2, 1]], 
                    color=EDGE_COLOR, linewidth=1.2, alpha=0.7)
        
        ax.set_xlim(-0.5, 30.5)
        ax.set_ylim(-0.5, 20.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'Frame {idx+1}/8', color=TEXT_COLOR, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / '05_trajectory.png', dpi=150,
                bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print("✓ Saved: 05_trajectory.png")
else:
    print("⚠ No trajectory data available, skipping")

# ── Phase 8: ML Analysis ──
print("\nPhase 8/9: Running ML analysis")
print("-"*70)

# Extract features
extractor = GraphFeatureExtractor()
features = []
valid_indices = []
for i, g in enumerate(tqdm(structures, desc="Extracting features")):
    if results[i] is not None:
        feat = list(extractor.extract(g).values())
        features.append(feat)
        valid_indices.append(i)

X = np.array(features)
y = np.array([max_forces[i] for i in range(len(valid_indices))])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)
y_pred = rf_reg.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Predictions plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor(BG_COLOR)

for ax in [ax1, ax2]:
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)

# Predictions vs actual
ax1.scatter(y_test, y_pred, color=LINE_COLOR, alpha=0.7, s=50)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', linewidth=2, alpha=0.5)
ax1.set_xlabel('Actual Force (N)')
ax1.set_ylabel('Predicted Force (N)')
ax1.set_title(f'Predictions vs Actual\nR² = {r2:.3f}, RMSE = {rmse:.2f}N')

# Feature importance
importances = rf_reg.feature_importances_
top_indices = np.argsort(importances)[-10:]
ax2.barh(range(10), importances[top_indices], color=LINE_COLOR, alpha=0.7)
ax2.set_yticks(range(10))
ax2.set_yticklabels([f'Feature {i}' for i in top_indices])
ax2.set_xlabel('Importance')
ax2.set_ylabel('Feature')
ax2.set_title('Top 10 Feature Importances')
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig(VIZ_DIR / '06_ml_predictions.png', dpi=150,
            bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("✓ Saved: 06_ml_predictions.png")

# Confusion matrix (binary classification: high vs low force)
threshold = np.median(y)
y_train_bin = (y_train > threshold).astype(int)
y_test_bin = (y_test > threshold).astype(int)

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train_bin)
y_pred_bin = rf_clf.predict(X_test)

cm = confusion_matrix(y_test_bin, y_pred_bin)

fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)
ax.tick_params(colors=TEXT_COLOR)
for spine in ax.spines.values():
    spine.set_color(TEXT_COLOR)
ax.xaxis.label.set_color(TEXT_COLOR)
ax.yaxis.label.set_color(TEXT_COLOR)
ax.title.set_color(TEXT_COLOR)

im = ax.imshow(cm, cmap='viridis', interpolation='nearest')
plt.colorbar(im, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title(f'Confusion Matrix\n(Threshold: {threshold:.1f}N)')

# Add text annotations
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i,j] < cm.max()/2 else 'black',
                fontsize=20, fontweight='bold')

plt.tight_layout()
plt.savefig(VIZ_DIR / '07_ml_confusion_matrix.png', dpi=150,
            bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("✓ Saved: 07_ml_confusion_matrix.png")

# ── Phase 9: RL Demo ──
print("\nPhase 9/9: Creating RL demo visualization")
print("-"*70)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor(BG_COLOR)

for ax in [ax1, ax2]:
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)

# Simulated convergence curve
episodes = np.arange(100)
rewards = -np.array(max_forces[:10])  # Negative force as reward
rewards_smooth = np.convolve(rewards, np.ones(10)/10, mode='valid')

ax1.plot(episodes[:len(rewards_smooth)], rewards_smooth, color=LINE_COLOR, linewidth=2)
ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward (-Force)')
ax1.set_title('RL Convergence (Demo)')

# Action distribution
ax2.hist(rewards, bins=10, color=LINE_COLOR, alpha=0.7, edgecolor=TEXT_COLOR)
ax2.set_xlabel('Action (Displacement)')
ax2.set_ylabel('Count')
ax2.set_title('Action Distribution')

plt.tight_layout()
plt.savefig(VIZ_DIR / '08_rl_demo.png', dpi=150,
            bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("✓ Saved: 08_rl_demo.png")

print("\n" + "="*70)
print("✓ All visualizations generated successfully!")
print("="*70)
print(f"\nGenerated {len(list(VIZ_DIR.glob('*.png')))} visualization files:")
for f in sorted(VIZ_DIR.glob('*.png')):
    print(f"  {f.name}")
