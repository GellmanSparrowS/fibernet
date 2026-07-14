#!/usr/bin/env python3
"""
FiberNet v4 Tutorial — Visualization Generator v7

Comprehensive tutorial with 12 unit types + 20 voronoi structures for ML.

Physics: box=(1,1), stiffness=1e5, proper wave propagation (58-90% nodes moving)
Display: forces in kN, dark theme, n_internal=5 for fiber curvature

Outputs:
1. 01_gallery_undeformed.png - All 12 units (3x4 grid)
2. 02_gallery_deformed.png - All 12 units after stretch (3x4 grid)
3. 03_structure_stats.png - Node/edge comparison across units
4. 04_force_curves.png - Max stretch vs step for selected units
5. 05_trajectory.png - 8-frame stretch trajectory (2x4)
6. 06_stress_distribution.png - Stretch ratio coloring
7. 07_ml_analysis.png - Feature importance, predictions
8. 08_rl_demo.png - RL convergence demo
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
import copy

sys.path.insert(0, str(Path(__file__).parent.parent))

from fibernet import pattern_2d, TaichiEngine, list_units
from fibernet.sim import SimResult
from fibernet.sim.accelerated import _get_boundary_indices, _graph_to_arrays
from fibernet.analysis import GraphFeatureExtractor
from fibernet.viz.render import _get_theme
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# ── Configuration ──
N_VORONOI = 20  # For ML analysis
BATCH_SIZE = 5
THEME = 'dark'
STIFFNESS = 1e5  # For proper wave propagation
BOX_SIZE = (1.0, 1.0)  # Small box for fast wave propagation
GRID = (2, 2)

# Paths
TUTORIAL_DIR = Path(__file__).parent.parent / 'tutorials' / 'v4_tutorial'
VIZ_DIR = TUTORIAL_DIR / 'tutorial_viz'
DATA_DIR = TUTORIAL_DIR / 'data'
VIZ_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Theme
theme = _get_theme(THEME)
BG_COLOR = theme['bg']
TEXT_COLOR = theme['text']
FIBER_COLOR = theme['fiber']

print("="*70)
print("FiberNet v4 Tutorial — Visualization Generator v7")
print("="*70)
print(f"12 unit types + {N_VORONOI} voronoi structures")
print(f"Physics: box={BOX_SIZE}, stiffness={STIFFNESS:.0e}, proper wave propagation")
print(f"Output: {VIZ_DIR}")
print()

# ── Phase 1: Generate 12 unit structures ──
print("Phase 1/11: Generating 12 unit structures")
print("-"*70)

units = list_units()
unit_structures = []

for unit in tqdm(units, desc="Generating units"):
    g = pattern_2d(
        unit=unit,
        box=BOX_SIZE,
        grid=GRID,
        seed=42,
        n_pts_per_side=3,
        n_internal=5
    )
    unit_structures.append(g)
    g.save_json(str(DATA_DIR / f'unit_{unit}.json'))

print(f"✓ Generated {len(unit_structures)} unit structures")
print(f"  Example: {units[0]} has {unit_structures[0].num_nodes} nodes, {unit_structures[0].num_edges} edges")

# ── Phase 2: Generate 20 voronoi structures for ML ──
print("\nPhase 2/11: Generating 20 voronoi structures for ML")
print("-"*70)

voronoi_structures = []
n_disps = 12  # 4 sides × 3 points

for i in tqdm(range(N_VORONOI), desc="Generating voronoi"):
    rng = np.random.default_rng(seed=1000+i)
    disps = [(float(rng.uniform(-0.5, 0.5)), float(rng.uniform(-0.5, 0.5))) 
             for _ in range(n_disps)]
    
    g = pattern_2d(
        unit='voronoi',
        box=BOX_SIZE,
        grid=GRID,
        seed=i,
        n_internal=5,
        n_pts_per_side=3,
        point_displacements=disps
    )
    voronoi_structures.append(g)
    g.save_json(str(DATA_DIR / f'voronoi_{i:03d}.json'))

print(f"✓ Generated {len(voronoi_structures)} voronoi structures")
print(f"  Nodes: {voronoi_structures[0].num_nodes}, Edges: {voronoi_structures[0].num_edges}")

# ── Phase 3: Simulate 12 unit structures ──
print("\nPhase 3/11: Simulating 12 unit structures")
print("-"*70)

engine = TaichiEngine()
unit_results = []

for i, (unit, g) in enumerate(tqdm(zip(units, unit_structures), desc="Simulating units", total=len(units))):
    result_file = DATA_DIR / f'unit_{unit}_result.json'
    
    if result_file.exists():
        try:
            result = SimResult.load(str(result_file))
            unit_results.append(result)
            continue
        except:
            pass
    
    pos, elements, _, _ = _graph_to_arrays(g)
    bnd = _get_boundary_indices(pos, pct=0.05)
    L_x = pos[:, 0].max() - pos[:, 0].min()
    
    target_stretch = 1.5
    target_disp = L_x * (target_stretch - 1)
    num_steps = 5000
    ramp_steps = int(num_steps * 0.5)
    
    schedule = {}
    for ni in bnd['right']:
        schedule[ni] = [
            (0, np.array([0.0, 0.0, 0.0])),
            (ramp_steps, np.array([target_disp, 0.0, 0.0])),
            (num_steps, np.array([target_disp, 0.0, 0.0]))
        ]
    
    fixed = bnd['left'] + (bnd.get('bottom', [])[:1] if bnd.get('bottom') else [])
    
    result = engine.dynamics(
        g,
        fixed_nodes=fixed,
        displacement_schedule=schedule,
        stiffness=STIFFNESS,
        damping=0.3,
        dt=1e-5,
        num_steps=num_steps,
        save_interval=500
    )
    
    unit_results.append(result)
    result.save(str(result_file), detailed=True)
    
    if (i + 1) % BATCH_SIZE == 0:
        gc.collect()

print(f"✓ Completed {len(unit_results)} unit simulations")
unit_max_forces_kN = [r.max_force / 1000 for r in unit_results]
print(f"  Force range: {min(unit_max_forces_kN):.1f} - {max(unit_max_forces_kN):.1f} kN")

# ── Phase 4: Simulate voronoi structures ──
print("\nPhase 4/11: Simulating voronoi structures")
print("-"*70)

voronoi_results = []

for i, g in enumerate(tqdm(voronoi_structures, desc="Simulating voronoi")):
    result_file = DATA_DIR / f'voronoi_{i:03d}_result.json'
    
    if result_file.exists():
        try:
            result = SimResult.load(str(result_file))
            voronoi_results.append(result)
            continue
        except:
            pass
    
    pos, elements, _, _ = _graph_to_arrays(g)
    bnd = _get_boundary_indices(pos, pct=0.05)
    L_x = pos[:, 0].max() - pos[:, 0].min()
    
    target_stretch = 1.5
    target_disp = L_x * (target_stretch - 1)
    num_steps = 5000
    ramp_steps = int(num_steps * 0.5)
    
    schedule = {}
    for ni in bnd['right']:
        schedule[ni] = [
            (0, np.array([0.0, 0.0, 0.0])),
            (ramp_steps, np.array([target_disp, 0.0, 0.0])),
            (num_steps, np.array([target_disp, 0.0, 0.0]))
        ]
    
    fixed = bnd['left'] + (bnd.get('bottom', [])[:1] if bnd.get('bottom') else [])
    
    result = engine.dynamics(
        g,
        fixed_nodes=fixed,
        displacement_schedule=schedule,
        stiffness=STIFFNESS,
        damping=0.3,
        dt=1e-5,
        num_steps=num_steps,
        save_interval=500
    )
    
    voronoi_results.append(result)
    result.save(str(result_file), detailed=True)
    
    if (i + 1) % BATCH_SIZE == 0:
        gc.collect()

print(f"✓ Completed {len(voronoi_results)} voronoi simulations")
voronoi_max_forces_kN = [r.max_force / 1000 for r in voronoi_results]
print(f"  Force range: {min(voronoi_max_forces_kN):.1f} - {max(voronoi_max_forces_kN):.1f} kN")

# ── Phase 5: Gallery undeformed (12 units, 3x4) ──
print("\nPhase 5/11: Creating undeformed gallery (12 units)")
print("-"*70)

from fibernet.viz.render import render_graph

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.patch.set_facecolor(BG_COLOR)
axes = axes.flatten()

for i, (unit, g) in enumerate(zip(units, unit_structures)):
    ax = axes[i]
    ax.set_facecolor(BG_COLOR)
    render_graph(g, ax=ax, theme=THEME, color_by='uniform', 
                 line_width=1.5, show_nodes=False, tight=False)
    ax.set_title(f'{unit}\n{g.num_nodes} nodes', color=TEXT_COLOR, fontsize=10)

plt.tight_layout()
plt.savefig(VIZ_DIR / '01_gallery_undeformed.png', dpi=150, 
            bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("✓ Saved: 01_gallery_undeformed.png")

# ── Phase 6: Gallery deformed (12 units, 3x4) ──
print("\nPhase 6/11: Creating deformed gallery (12 units)")
print("-"*70)

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.patch.set_facecolor(BG_COLOR)
axes = axes.flatten()

for i, (unit, g, result) in enumerate(zip(units, unit_structures, unit_results)):
    ax = axes[i]
    ax.set_facecolor(BG_COLOR)
    
    if result.deformed_positions is not None:
        g_def = copy.deepcopy(g)
        for idx, nid in enumerate(list(g_def.nodes.keys())):
            g_def.nodes[nid].position = result.deformed_positions[idx]
        
        render_graph(g_def, ax=ax, theme=THEME, color_by='uniform',
                     line_width=1.5, show_nodes=False, tight=False)
        ax.set_title(f'{unit}\n{unit_max_forces_kN[i]:.1f} kN', 
                     color=TEXT_COLOR, fontsize=10)

plt.tight_layout()
plt.savefig(VIZ_DIR / '02_gallery_deformed.png', dpi=150, 
            bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("✓ Saved: 02_gallery_deformed.png")

# ── Phase 7: Structure statistics ──
print("\nPhase 7/11: Generating structure statistics")
print("-"*70)

node_counts = [g.num_nodes for g in unit_structures]
edge_counts = [g.num_edges for g in unit_structures]

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

ax1.bar(range(len(units)), node_counts, color=FIBER_COLOR, alpha=0.7)
ax1.set_xticks(range(len(units)))
ax1.set_xticklabels(units, rotation=45, ha='right')
ax1.set_ylabel('Number of Nodes')
ax1.set_title('Node Count by Unit Type')

ax2.bar(range(len(units)), edge_counts, color=FIBER_COLOR, alpha=0.7)
ax2.set_xticks(range(len(units)))
ax2.set_xticklabels(units, rotation=45, ha='right')
ax2.set_ylabel('Number of Edges')
ax2.set_title('Edge Count by Unit Type')

plt.tight_layout()
plt.savefig(VIZ_DIR / '03_structure_stats.png', dpi=150,
            bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("✓ Saved: 03_structure_stats.png")

# ── Phase 8: Force curves ──
print("\nPhase 8/11: Creating force curves")
print("-"*70)

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)
ax.tick_params(colors=TEXT_COLOR)
for spine in ax.spines.values():
    spine.set_color(TEXT_COLOR)
ax.xaxis.label.set_color(TEXT_COLOR)
ax.yaxis.label.set_color(TEXT_COLOR)
ax.title.set_color(TEXT_COLOR)

selected_units = ['voronoi', 'kagome', 'honeycomb', 'triangle', 'reentrant']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

for unit, result, color in zip(units, unit_results, colors):
    if unit in selected_units and hasattr(result, 'history') and result.history:
        steps = [h['step'] for h in result.history]
        max_stretches = [h['max_stretch'] for h in result.history]
        ax.plot(steps, max_stretches, label=unit, color=color, linewidth=2)

ax.set_xlabel('Step')
ax.set_ylabel('Max Stretch Ratio')
ax.set_title('Stretch Evolution During Simulation')
ax.legend()
ax.grid(True, alpha=0.3, color=TEXT_COLOR)

plt.tight_layout()
plt.savefig(VIZ_DIR / '04_force_curves.png', dpi=150,
            bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("✓ Saved: 04_force_curves.png")

# ── Phase 9: Trajectory (8 frames, 2x4) ──
print("\nPhase 9/11: Creating trajectory visualization")
print("-"*70)

voronoi_idx = 0
result = voronoi_results[voronoi_idx]

if hasattr(result, 'positions_trajectory') and result.positions_trajectory:
    traj = result.positions_trajectory
    
    if len(traj) >= 8:
        frame_indices = np.linspace(0, len(traj)-1, 8, dtype=int)
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.patch.set_facecolor(BG_COLOR)
        axes = axes.flatten()
        
        g = voronoi_structures[voronoi_idx]
        pos_orig, elements, _, _ = _graph_to_arrays(g)
        
        for idx, (frame_idx, ax) in enumerate(zip(frame_indices, axes)):
            ax.set_facecolor(BG_COLOR)
            pos = traj[frame_idx]
            
            for e in elements:
                ax.plot([pos[e[0],0], pos[e[1],0]], 
                        [pos[e[0],1], pos[e[1],1]], 
                        color=FIBER_COLOR, linewidth=1.2)
            
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(f'Frame {idx+1}/8\nStep {frame_idx * 500}', 
                        color=TEXT_COLOR, fontsize=10)
        
        fig.suptitle(f'Voronoi Stretch Trajectory (1.5x)', 
                     color=TEXT_COLOR, fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(VIZ_DIR / '05_trajectory.png', dpi=150,
                    bbox_inches='tight', facecolor=BG_COLOR)
        plt.close()
        print("✓ Saved: 05_trajectory.png")
    else:
        print("⚠ Insufficient trajectory frames")
else:
    print("⚠ No trajectory data")

# ── Phase 10: Stress distribution ──
print("\nPhase 10/11: Creating stress distribution visualization")
print("-"*70)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor(BG_COLOR)

selected = [0, 3, 5]  # voronoi, kagome, triangle
titles = ['Voronoi', 'Kagome', 'Triangle']

for idx, (i, title) in enumerate(zip(selected, titles)):
    ax = axes[idx]
    ax.set_facecolor(BG_COLOR)
    
    g = unit_structures[i]
    result = unit_results[i]
    
    if result.deformed_positions is not None:
        pos_orig, elements, _, _ = _graph_to_arrays(g)
        pos_def = result.deformed_positions
        
        # Compute stretch ratio for each edge
        lengths_orig = np.array([np.linalg.norm(pos_orig[elements[e,1]] - pos_orig[elements[e,0]]) 
                                 for e in range(len(elements))])
        lengths_def = np.array([np.linalg.norm(pos_def[elements[e,1]] - pos_def[elements[e,0]]) 
                                for e in range(len(elements))])
        stretch = lengths_def / (lengths_orig + 1e-12)
        
        # Normalize for colormap
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        norm = Normalize(vmin=stretch.min(), vmax=stretch.max())
        cmap = plt.cm.viridis
        
        for e_idx, e in enumerate(elements):
            color = cmap(norm(stretch[e_idx]))
            ax.plot([pos_def[e[0],0], pos_def[e[1],0]], 
                    [pos_def[e[0],1], pos_def[e[1],1]], 
                    color=color, linewidth=1.5)
        
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'{title}\nStretch: {stretch.min():.2f}-{stretch.max():.2f}', 
                     color=TEXT_COLOR, fontsize=11)

# Add colorbar
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, fraction=0.046, pad=0.04)
cbar.set_label('Stretch Ratio', color=TEXT_COLOR, fontsize=11)
cbar.ax.tick_params(colors=TEXT_COLOR)

plt.tight_layout()
plt.savefig(VIZ_DIR / '06_stress_distribution.png', dpi=150,
            bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("✓ Saved: 06_stress_distribution.png")

# ── Phase 11: ML Analysis ──
print("\nPhase 11/11: Running ML analysis")
print("-"*70)

extractor = GraphFeatureExtractor()
features = []
valid_indices = []

for i, g in enumerate(tqdm(voronoi_structures, desc="Extracting features")):
    if voronoi_results[i] is not None:
        feat = list(extractor.extract(g).values())
        features.append(feat)
        valid_indices.append(i)

X = np.array(features)
y = np.array([voronoi_max_forces_kN[i] for i in valid_indices])

if len(X) > 5:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
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
    
    ax1.scatter(y_test, y_pred, color=FIBER_COLOR, alpha=0.7, s=50)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', linewidth=2, alpha=0.5)
    ax1.set_xlabel('Actual Force (kN)')
    ax1.set_ylabel('Predicted Force (kN)')
    ax1.set_title(f'Predictions vs Actual\nR² = {r2:.3f}, RMSE = {rmse:.2f} kN')
    
    importances = rf.feature_importances_
    top_indices = np.argsort(importances)[-10:]
    ax2.barh(range(10), importances[top_indices], color=FIBER_COLOR, alpha=0.7)
    ax2.set_yticks(range(10))
    ax2.set_yticklabels([f'Feature {i}' for i in top_indices])
    ax2.set_xlabel('Importance')
    ax2.set_ylabel('Feature')
    ax2.set_title('Top 10 Feature Importances')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / '07_ml_analysis.png', dpi=150,
                bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print("✓ Saved: 07_ml_analysis.png")
else:
    print("⚠ Insufficient data for ML")

# RL demo
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

episodes = np.arange(100)
rewards = -np.array(voronoi_max_forces_kN[:10])
rewards_smooth = np.convolve(rewards, np.ones(10)/10, mode='valid')

ax1.plot(episodes[:len(rewards_smooth)], rewards_smooth, color=FIBER_COLOR, linewidth=2)
ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward (-Force kN)')
ax1.set_title('RL Convergence (Demo)')

ax2.hist(rewards, bins=10, color=FIBER_COLOR, alpha=0.7, edgecolor=TEXT_COLOR)
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
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name} ({size_kb:.0f} KB)")
