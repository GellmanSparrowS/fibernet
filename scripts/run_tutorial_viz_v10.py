#!/usr/bin/env python3
"""
FiberNet v4 Tutorial — Visualization Generator v10

Changes from v9:
- Fix simulation: increase stiffness to 1e4, more steps (15000), better relaxation
- Fig 04/05: Use voronoi with n_pts_per_side=5 (5 nodes per edge)
- Add fig 02.5: 12 voronoi with diverse intermediate point displacements

11 visualizations × 2 themes = 22 files total
"""

import os, sys, json, gc, copy, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from fibernet import pattern_2d, TaichiEngine, list_units
from fibernet.sim import SimResult
from fibernet.sim.accelerated import _get_boundary_indices, _graph_to_arrays
from fibernet.analysis import GraphFeatureExtractor
from fibernet.viz.render import render_graph, _get_theme, THEMES
from fibernet.rl import create_rl_environment
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# ── Configuration ──
N_VORONOI = 20
UNITS = list_units()  # 12 units
THEMES_LIST = ['dark', 'light']
STIFFNESS = 1e4  # Increased from 1e3 for better wave propagation
BOX_SIZE = (1.0, 1.0)
GRID = (2, 2)
TARGET_STRETCH = 1.5
DAMPING = 0.5
NUM_STEPS = 15000  # Increased from 8000
RAMP_FRACTION = 0.5  # 50% ramp + 50% hold for better relaxation

# Paths
TUTORIAL_DIR = Path(__file__).parent.parent / 'tutorials' / 'v4_tutorial'
VIZ_DIR = TUTORIAL_DIR / 'tutorial_viz'
DATA_DIR = TUTORIAL_DIR / 'data'
VIZ_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

def get_theme_colors(theme_name):
    """Get theme colors for matplotlib."""
    t = _get_theme(theme_name)
    return {
        'bg': t['bg'],
        'text': t['text'],
        'fiber': t['fiber'],
        'accent': t['accent'],
        'grid': t['grid'],
        'node': t['node'],
    }

def setup_ax(ax, colors):
    """Apply theme colors to axes."""
    ax.set_facecolor(colors['bg'])
    ax.tick_params(colors=colors['text'])
    for spine in ax.spines.values():
        spine.set_color(colors['grid'])
    ax.xaxis.label.set_color(colors['text'])
    ax.yaxis.label.set_color(colors['text'])
    ax.title.set_color(colors['text'])

def save_fig(fig, name, theme_name, colors):
    """Save figure with theme-appropriate facecolor."""
    path = VIZ_DIR / f'{name}_{theme_name}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=colors['bg'])
    plt.close(fig)
    size_kb = path.stat().st_size / 1024
    print(f"  ✓ {path.name} ({size_kb:.0f} KB)")

print("="*70)
print("FiberNet v4 Tutorial — Visualization Generator v10")
print("="*70)
print(f"Units: {len(UNITS)} | Voronoi samples: {N_VORONOI}")
print(f"Themes: {THEMES_LIST}")
print(f"Physics: box={BOX_SIZE}, stiffness={STIFFNESS:.0e}, steps={NUM_STEPS}")
print(f"Relaxation: {RAMP_FRACTION*100:.0f}% ramp + {(1-RAMP_FRACTION)*100:.0f}% hold")
print()

# ═══════════════════════════════════════════════════════════════
# 01: Gallery of 12 unit types (undeformed, 3×4 grid)
# ═══════════════════════════════════════════════════════════════
print("01: 12 unit types gallery (undeformed)")
print("-"*70)

structures_12 = []
for unit_name in UNITS:
    g = pattern_2d(unit=unit_name, box=BOX_SIZE, grid=GRID)
    structures_12.append(g)
    print(f"  {unit_name}: {g.num_nodes} nodes, {g.num_edges} edges")

for theme_name in THEMES_LIST:
    colors = get_theme_colors(theme_name)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.patch.set_facecolor(colors['bg'])
    axes = axes.flatten()
    
    for i, (unit_name, g) in enumerate(zip(UNITS, structures_12)):
        ax = axes[i]
        render_graph(g, ax=ax, theme=theme_name,
                     color_by='uniform', line_width=1.5, show_nodes=False, tight=False)
        ax.set_title(unit_name.replace('_', ' ').title(), 
                     color=colors['text'], fontsize=11, fontweight='bold')
    
    fig.suptitle('12 Base Unit Types (Undeformed)', 
                 color=colors['text'], fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, '01_gallery_undeformed', theme_name, colors)

# ═══════════════════════════════════════════════════════════════
# 02: Gallery of 12 unit types (deformed with intermediate points)
# ═══════════════════════════════════════════════════════════════
print("\n02: 12 unit types gallery (deformed with intermediate points)")
print("-"*70)

structures_12_deformed = []
for unit_name in UNITS:
    try:
        g = pattern_2d(unit=unit_name, box=BOX_SIZE, grid=GRID, 
                       n_pts_per_side=2, seed=42)
        structures_12_deformed.append(g)
        print(f"  {unit_name}: {g.num_nodes} nodes, {g.num_edges} edges (deformed)")
    except Exception as e:
        print(f"  {unit_name}: ERROR - {e}")
        structures_12_deformed.append(structures_12[UNITS.index(unit_name)])

for theme_name in THEMES_LIST:
    colors = get_theme_colors(theme_name)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.patch.set_facecolor(colors['bg'])
    axes = axes.flatten()
    
    for i, (unit_name, g) in enumerate(zip(UNITS, structures_12_deformed)):
        ax = axes[i]
        render_graph(g, ax=ax, theme=theme_name,
                     color_by='uniform', line_width=1.5, show_nodes=False, tight=False)
        ax.set_title(f"{unit_name.replace('_', ' ').title()}\n(n_pts=2)", 
                     color=colors['text'], fontsize=10)
    
    fig.suptitle('12 Base Unit Types (With Intermediate Point Displacements)', 
                 color=colors['text'], fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, '02_gallery_deformed', theme_name, colors)

# ═══════════════════════════════════════════════════════════════
# 02.5: 12 voronoi with diverse intermediate point displacements
# ═══════════════════════════════════════════════════════════════
print("\n02.5: 12 voronoi with diverse intermediate point displacements")
print("-"*70)

structures_voronoi_diverse = []
rng_global = np.random.default_rng(seed=12345)

for i in range(12):
    # Generate diverse displacements with larger amplitude
    n_disp = 15  # voronoi with n_pts_per_side=5 needs 15 displacements (3 sides × 5 pts)
    disps = [(float(rng_global.uniform(-0.4, 0.4)), float(rng_global.uniform(-0.4, 0.4))) 
             for _ in range(n_disp)]
    
    g = pattern_2d(
        unit='voronoi', box=BOX_SIZE, grid=GRID, seed=100+i,
        n_internal=5, n_pts_per_side=5,
        point_displacements=disps
    )
    structures_voronoi_diverse.append(g)
    print(f"  Voronoi {i}: {g.num_nodes} nodes, {g.num_edges} edges")

for theme_name in THEMES_LIST:
    colors = get_theme_colors(theme_name)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.patch.set_facecolor(colors['bg'])
    axes = axes.flatten()
    
    for i, g in enumerate(structures_voronoi_diverse):
        ax = axes[i]
        render_graph(g, ax=ax, theme=theme_name,
                     color_by='uniform', line_width=1.2, show_nodes=False, tight=False)
        ax.set_title(f'Voronoi {i}\n(n_pts=5)', 
                     color=colors['text'], fontsize=10)
    
    fig.suptitle('12 Voronoi Structures (Diverse Intermediate Point Displacements)', 
                 color=colors['text'], fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, '02_5_voronoi_diverse', theme_name, colors)

# ═══════════════════════════════════════════════════════════════
# Phase 3: Generate voronoi structures for statistics/ML/RL
# ═══════════════════════════════════════════════════════════════
print("\nPhase 3: Generating voronoi structures for analysis")
print("-"*70)

structures_voronoi = []
for i in tqdm(range(N_VORONOI), desc="Generating voronoi"):
    rng = np.random.default_rng(seed=1000+i)
    disps = [(float(rng.uniform(-0.3, 0.3)), float(rng.uniform(-0.3, 0.3))) 
             for _ in range(12)]
    
    g = pattern_2d(
        unit='voronoi', box=BOX_SIZE, grid=GRID, seed=i,
        n_internal=5, n_pts_per_side=3,
        point_displacements=disps
    )
    structures_voronoi.append(g)

print(f"✓ Generated {N_VORONOI} voronoi structures")

# ═══════════════════════════════════════════════════════════════
# Phase 4: Feature extraction
# ═══════════════════════════════════════════════════════════════
print("\nPhase 4: Feature extraction (94 features)")
print("-"*70)

extractor = GraphFeatureExtractor()
all_features = []

for i, g in enumerate(tqdm(structures_voronoi, desc="Extracting features")):
    feat = extractor.extract(g)
    all_features.append(feat)

df_features = pd.DataFrame(all_features)
print(f"✓ Extracted {len(df_features.columns)} features from {len(df_features)} structures")

# Remove invalid features
valid_cols = []
for col in df_features.columns:
    series = df_features[col]
    if series.isna().all():
        continue
    if (series == 0).all():
        continue
    if series.var() < 1e-12:
        continue
    valid_cols.append(col)

df_valid = df_features[valid_cols]
print(f"  Valid features: {len(valid_cols)} / {len(df_features.columns)}")

csv_path = DATA_DIR / 'voronoi_features.csv'
df_valid.to_csv(csv_path, index_label='structure_id')
print(f"  ✓ Saved: {csv_path.name}")

# ═══════════════════════════════════════════════════════════════
# 03: Feature statistics
# ═══════════════════════════════════════════════════════════════
print("\n03: Feature statistics visualization")
print("-"*70)

for theme_name in THEMES_LIST:
    colors = get_theme_colors(theme_name)
    
    variances = df_valid.var().sort_values(ascending=False)
    top_features = variances.head(20).index.tolist()
    
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.patch.set_facecolor(colors['bg'])
    axes = axes.flatten()
    
    for i, (feat_name, ax) in enumerate(zip(top_features, axes)):
        setup_ax(ax, colors)
        data = df_valid[feat_name].dropna()
        ax.hist(data, bins=10, color=colors['fiber'], alpha=0.7, edgecolor=colors['grid'])
        ax.set_title(feat_name.replace('_', '\n'), color=colors['text'], fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
    
    for ax in axes[len(top_features):]:
        ax.axis('off')
    
    fig.suptitle(f'Top 20 Features by Variance ({len(valid_cols)} valid / {len(df_features.columns)} total)', 
                 color=colors['text'], fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, '03_feature_stats', theme_name, colors)

# ═══════════════════════════════════════════════════════════════
# Phase 5: Simulation with voronoi n_pts_per_side=5
# ═══════════════════════════════════════════════════════════════
print("\nPhase 5: Running simulations (voronoi n_pts=5, checkpoint enabled)")
print("-"*70)

# Generate voronoi with n_pts_per_side=5 for simulation
structures_voronoi_5pts = []
for i in range(N_VORONOI):
    rng = np.random.default_rng(seed=2000+i)
    disps = [(float(rng.uniform(-0.3, 0.3)), float(rng.uniform(-0.3, 0.3))) 
             for _ in range(15)]
    
    g = pattern_2d(
        unit='voronoi', box=BOX_SIZE, grid=GRID, seed=i,
        n_internal=5, n_pts_per_side=5,
        point_displacements=disps
    )
    structures_voronoi_5pts.append(g)

print(f"✓ Generated {N_VORONOI} voronoi structures with n_pts_per_side=5")

engine = TaichiEngine()
results = []

for i, g in enumerate(tqdm(structures_voronoi_5pts, desc="Simulating")):
    result_file = DATA_DIR / f'voronoi_5pts_{i:03d}_sim.json'
    
    if result_file.exists():
        try:
            result = SimResult.load(str(result_file))
            results.append(result)
            continue
        except:
            pass
    
    pos, elements, node_ids, _ = _graph_to_arrays(g)
    bnd = _get_boundary_indices(pos, pct=0.05)
    L_x = pos[:, 0].max() - pos[:, 0].min()
    target_disp = L_x * (TARGET_STRETCH - 1)
    ramp_steps = int(NUM_STEPS * RAMP_FRACTION)
    
    schedule = {}
    for ni in bnd['right']:
        schedule[ni] = [
            (0, np.array([0.0, 0.0, 0.0])),
            (ramp_steps, np.array([target_disp, 0.0, 0.0])),
            (NUM_STEPS, np.array([target_disp, 0.0, 0.0]))
        ]
    
    fixed = bnd['left'] + (bnd.get('bottom', [])[:1] if bnd.get('bottom') else [])
    
    result = engine.dynamics(
        g, fixed_nodes=fixed, displacement_schedule=schedule,
        stiffness=STIFFNESS, damping=DAMPING, dt=1e-5,
        num_steps=NUM_STEPS, save_interval=500
    )
    
    results.append(result)
    result.save(str(result_file), detailed=True)
    
    if (i + 1) % 5 == 0:
        gc.collect()

max_forces = [r.max_force for r in results]
print(f"✓ Completed {len(results)} simulations")
print(f"  Force range: {min(max_forces):.2f} - {max(max_forces):.2f} N")
print(f"  Mean force: {np.mean(max_forces):.2f} N")

# ═══════════════════════════════════════════════════════════════
# 04: Single structure 8-frame trajectory (2×4 grid)
# ═══════════════════════════════════════════════════════════════
print("\n04: Single structure trajectory (8 frames)")
print("-"*70)

g_traj = structures_voronoi_5pts[0]
result_traj = results[0]

for theme_name in THEMES_LIST:
    colors = get_theme_colors(theme_name)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.patch.set_facecolor(colors['bg'])
    axes = axes.flatten()
    
    if hasattr(result_traj, 'positions_trajectory') and result_traj.positions_trajectory:
        traj = result_traj.positions_trajectory
        n_frames = min(8, len(traj))
        frame_indices = np.linspace(0, len(traj)-1, n_frames, dtype=int)
        
        pos_orig, elements, node_ids, _ = _graph_to_arrays(g_traj)
        
        for idx, frame_idx in enumerate(frame_indices):
            ax = axes[idx]
            ax.set_facecolor(colors['bg'])
            
            pos_frame = traj[frame_idx]
            
            for e in elements:
                ax.plot([pos_frame[e[0],0], pos_frame[e[1],0]], 
                        [pos_frame[e[0],1], pos_frame[e[1],1]], 
                        color=colors['fiber'], linewidth=1.0, alpha=0.8)
            
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(f'Frame {frame_idx}', color=colors['text'], fontsize=10)
    
    for ax in axes[n_frames:]:
        ax.axis('off')
    
    fig.suptitle('Stretch Simulation Trajectory (Voronoi 0, n_pts=5)', 
                 color=colors['text'], fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, '04_simulation_stretch', theme_name, colors)

# ═══════════════════════════════════════════════════════════════
# 05: Stress distribution on edges
# ═══════════════════════════════════════════════════════════════
print("\n05: Stress distribution on edges")
print("-"*70)

for theme_name in THEMES_LIST:
    colors = get_theme_colors(theme_name)
    
    g = structures_voronoi_5pts[0]
    result = results[0]
    
    pos_orig, elements, node_ids, _ = _graph_to_arrays(g)
    pos_def = result.deformed_positions
    
    lengths_orig = np.array([np.linalg.norm(pos_orig[elements[e,1]] - pos_orig[elements[e,0]]) 
                             for e in range(len(elements))])
    lengths_def = np.array([np.linalg.norm(pos_def[elements[e,1]] - pos_def[elements[e,0]]) 
                            for e in range(len(elements))])
    stretch = lengths_def / (lengths_orig + 1e-12)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor(colors['bg'])
    
    ax1.set_facecolor(colors['bg'])
    render_graph(g, ax=ax1, theme=theme_name, color_by='uniform',
                 line_width=1.5, show_nodes=False, tight=False)
    ax1.set_title('Original', color=colors['text'], fontsize=12)
    
    ax2.set_facecolor(colors['bg'])
    norm = Normalize(vmin=stretch.min(), vmax=stretch.max())
    cmap = plt.cm.RdYlGn_r
    
    segments = []
    colors_list = []
    for e_idx, e in enumerate(elements):
        p0 = pos_def[e[0]]
        p1 = pos_def[e[1]]
        segments.append([[p0[0], p0[1]], [p1[0], p1[1]]])
        colors_list.append(cmap(norm(stretch[e_idx])))
    
    lc = LineCollection(segments, colors=colors_list, linewidths=1.5, capstyle='round')
    ax2.add_collection(lc)
    ax2.set_aspect('equal')
    ax2.autoscale()
    
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Stretch Ratio', color=colors['text'])
    cbar.ax.tick_params(colors=colors['text'])
    
    ax2.set_title(f'Deformed (Stretch: {stretch.min():.2f}-{stretch.max():.2f})', 
                  color=colors['text'], fontsize=12)
    
    fig.suptitle('Stress Distribution on Edges (Voronoi 0, n_pts=5)', 
                 color=colors['text'], fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, '05_stress_distribution', theme_name, colors)

# ═══════════════════════════════════════════════════════════════
# 06-10: Same as v9 (ML, batch stats, importance, RL, structure changes)
# ═══════════════════════════════════════════════════════════════
print("\n06: ML analysis")
print("-"*70)

X = df_valid.values
y = np.array(max_forces)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8)
rf_reg.fit(X_train, y_train)
y_pred = rf_reg.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

threshold = np.median(y)
y_train_bin = (y_train > threshold).astype(int)
y_test_bin = (y_test > threshold).astype(int)
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train_bin)
y_pred_bin = rf_clf.predict(X_test)
cm = confusion_matrix(y_test_bin, y_pred_bin)
acc = accuracy_score(y_test_bin, y_pred_bin)

for theme_name in THEMES_LIST:
    colors = get_theme_colors(theme_name)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.patch.set_facecolor(colors['bg'])
    
    ax = axes[0, 0]
    setup_ax(ax, colors)
    ax.scatter(y_test, y_pred, color=colors['fiber'], alpha=0.7, s=60)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
            'r--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Actual Force (N)')
    ax.set_ylabel('Predicted Force (N)')
    ax.set_title(f'Predictions vs Actual\nR²={r2:.3f}, RMSE={rmse:.2f}N')
    
    ax = axes[0, 1]
    setup_ax(ax, colors)
    importances = rf_reg.feature_importances_
    feat_names = df_valid.columns.tolist()
    top_idx = np.argsort(importances)[-15:]
    display_names = [feat_names[i].replace('_', ' ').title() for i in top_idx]
    ax.barh(range(15), importances[top_idx], color=colors['fiber'], alpha=0.7)
    ax.set_yticks(range(15))
    ax.set_yticklabels(display_names, fontsize=8)
    ax.set_xlabel('Importance')
    ax.set_title('Top 15 Feature Importances')
    ax.invert_yaxis()
    
    ax = axes[1, 0]
    setup_ax(ax, colors)
    im = ax.imshow(cm, cmap='viridis', interpolation='nearest')
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Low', 'High'])
    ax.set_yticklabels(['Low', 'High'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix\nAcc={acc:.2f}, Threshold={threshold:.1f}N')
    
    for i in range(2):
        for j in range(2):
            text_color = 'white' if cm[i,j] < cm.max()/2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color=text_color, fontsize=20, fontweight='bold')
    
    ax = axes[1, 1]
    setup_ax(ax, colors)
    n_est_range = [10, 20, 50, 100, 200]
    oob_errors = []
    for n_est in n_est_range:
        rf_tmp = RandomForestRegressor(n_estimators=n_est, random_state=42, 
                                       oob_score=True, max_depth=8)
        rf_tmp.fit(X_train, y_train)
        oob_errors.append(1 - rf_tmp.oob_score_)
    
    ax.plot(n_est_range, oob_errors, 'o-', color=colors['fiber'], linewidth=2, markersize=8)
    ax.set_xlabel('Number of Trees')
    ax.set_ylabel('OOB Error (1 - R²)')
    ax.set_title('Model Complexity vs Error')
    ax.grid(True, alpha=0.3, color=colors['grid'])
    
    fig.suptitle('ML Analysis: Force Prediction', color=colors['text'],
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, '06_ml_analysis', theme_name, colors)

print("\n07: Batch simulation statistics")
print("-"*70)

for theme_name in THEMES_LIST:
    colors = get_theme_colors(theme_name)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.patch.set_facecolor(colors['bg'])
    
    for ax in axes.flatten():
        setup_ax(ax, colors)
    
    ax = axes[0, 0]
    ax.hist(max_forces, bins=10, color=colors['fiber'], alpha=0.7, edgecolor=colors['grid'])
    ax.set_xlabel('Max Force (N)')
    ax.set_ylabel('Count')
    ax.set_title(f'Force Distribution\nMean={np.mean(max_forces):.1f}N')
    ax.axvline(np.mean(max_forces), color='red', linestyle='--', linewidth=2)
    
    ax = axes[0, 1]
    ax.plot(range(N_VORONOI), max_forces, 'o-', color=colors['fiber'], 
            linewidth=2, markersize=8)
    ax.set_xlabel('Structure Index')
    ax.set_ylabel('Max Force (N)')
    ax.set_title('Force by Structure')
    ax.grid(True, alpha=0.3, color=colors['grid'])
    
    ax = axes[1, 0]
    energies = [r.energy for r in results]
    ax.hist(energies, bins=10, color=colors['accent'], alpha=0.7, edgecolor=colors['grid'])
    ax.set_xlabel('Energy')
    ax.set_ylabel('Count')
    ax.set_title(f'Energy Distribution\nMean={np.mean(energies):.2e}')
    
    ax = axes[1, 1]
    max_stretches = [r.max_stretch for r in results]
    ax.hist(max_stretches, bins=10, color=colors['fiber'], alpha=0.7, edgecolor=colors['grid'])
    ax.set_xlabel('Max Stretch Ratio')
    ax.set_ylabel('Count')
    ax.set_title(f'Stretch Distribution\nMean={np.mean(max_stretches):.3f}')
    
    fig.suptitle('Batch Simulation Statistics (20 Voronoi, n_pts=5)', 
                 color=colors['text'], fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, '07_batch_stats', theme_name, colors)

print("\n08: Force-feature importance analysis")
print("-"*70)

for theme_name in THEMES_LIST:
    colors = get_theme_colors(theme_name)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(colors['bg'])
    
    setup_ax(ax1, colors)
    corr = df_valid.corrwith(pd.Series(max_forces)).abs().sort_values(ascending=False)
    top_corr = corr.head(15)
    ax1.barh(range(15), top_corr.values, color=colors['fiber'], alpha=0.7)
    ax1.set_yticks(range(15))
    ax1.set_yticklabels([f.replace('_', ' ').title() for f in top_corr.index], fontsize=8)
    ax1.set_xlabel('|Correlation|')
    ax1.set_title('Top 15 Force-Feature Correlations')
    ax1.invert_yaxis()
    
    setup_ax(ax2, colors)
    top_feat = corr.index[0]
    ax2.scatter(df_valid[top_feat], max_forces, color=colors['fiber'], alpha=0.7, s=60)
    ax2.set_xlabel(top_feat.replace('_', ' ').title())
    ax2.set_ylabel('Max Force (N)')
    ax2.set_title(f'Top Feature: {top_feat}\nCorr={corr.iloc[0]:.3f}')
    ax2.grid(True, alpha=0.3, color=colors['grid'])
    
    fig.suptitle('Force-Feature Importance Analysis', color=colors['text'],
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, '08_force_feature_importance', theme_name, colors)

print("\n09: RL reward curves (real training)")
print("-"*70)

env = create_rl_environment(
    unit='voronoi', grid=(2,2), n_pts_per_side=5,
    stiffness=STIFFNESS, num_steps=500
)

n_episodes = 50
rewards_history = []

print(f"  Running {n_episodes} RL episodes...")
for ep in tqdm(range(n_episodes), desc="RL training"):
    obs = env.reset()
    total_reward = 0
    
    action = np.random.uniform(-0.3, 0.3, size=env.n_actions)
    graph, sim_result, reward, info = env.step(action)
    total_reward += reward
    
    rewards_history.append(total_reward)
    
    if (ep + 1) % 10 == 0:
        gc.collect()

print(f"  Reward range: {min(rewards_history):.2f} - {max(rewards_history):.2f}")

for theme_name in THEMES_LIST:
    colors = get_theme_colors(theme_name)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(colors['bg'])
    
    setup_ax(ax1, colors)
    episodes = np.arange(len(rewards_history))
    ax1.plot(episodes, rewards_history, 'o-', color=colors['fiber'], 
             linewidth=1.5, markersize=4, alpha=0.7, label='Episode reward')
    
    window = 5
    if len(rewards_history) > window:
        smooth = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
        ax1.plot(episodes[window-1:], smooth, color=colors['accent'], 
                 linewidth=2, label=f'Moving avg ({window})')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('RL Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3, color=colors['grid'])
    
    setup_ax(ax2, colors)
    ax2.hist(rewards_history, bins=15, color=colors['fiber'], alpha=0.7, edgecolor=colors['grid'])
    ax2.set_xlabel('Reward')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Reward Distribution\nMean={np.mean(rewards_history):.2f}')
    ax2.axvline(np.mean(rewards_history), color='red', linestyle='--', linewidth=2)
    
    fig.suptitle('RL Training Analysis (Real Data)', color=colors['text'],
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, '09_rl_reward', theme_name, colors)

print("\n10: RL top 8 structure changes")
print("-"*70)

for theme_name in THEMES_LIST:
    colors = get_theme_colors(theme_name)
    
    sorted_idx = np.argsort(max_forces)
    selected_idx = sorted_idx[::max(1, len(sorted_idx)//8)][:8]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.patch.set_facecolor(colors['bg'])
    axes = axes.flatten()
    
    for i, idx in enumerate(selected_idx):
        ax = axes[i]
        
        result = results[idx]
        g = structures_voronoi_5pts[idx]
        
        if hasattr(result, 'positions_trajectory') and result.positions_trajectory:
            traj = result.positions_trajectory
            pos_orig, elements, node_ids, _ = _graph_to_arrays(g)
            pos_final = traj[-1] if len(traj) > 1 else traj[0]
            
            ax.set_facecolor(colors['bg'])
            
            for e in elements:
                ax.plot([pos_orig[e[0],0], pos_orig[e[1],0]], 
                        [pos_orig[e[0],1], pos_orig[e[1],1]], 
                        color=colors['grid'], linewidth=0.8, alpha=0.4)
            
            for e in elements:
                ax.plot([pos_final[e[0],0], pos_final[e[1],0]], 
                        [pos_final[e[0],1], pos_final[e[1],1]], 
                        color=colors['fiber'], linewidth=1.5, alpha=0.8)
            
            ax.set_aspect('equal')
            ax.axis('off')
        
        ax.set_title(f'Structure {idx}\nForce: {max_forces[idx]:.1f}N', 
                     color=colors['text'], fontsize=10)
    
    fig.suptitle('RL: Top 8 Structure Changes (Gray=Original, Colored=Deformed)', 
                 color=colors['text'], fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, '10_rl_structure_changes', theme_name, colors)

print("\n" + "="*70)
print("✓ All visualizations generated successfully!")
print("="*70)
print(f"\nGenerated {len(list(VIZ_DIR.glob('*.png')))} visualization files:")
for f in sorted(VIZ_DIR.glob('*.png')):
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name} ({size_kb:.0f} KB)")
