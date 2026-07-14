#!/usr/bin/env python3
"""Generate the v4 tutorial notebook."""
import json
from pathlib import Path

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": src if isinstance(src, list) else [src]})

def code(src):
    cells.append({"cell_type": "code", "metadata": {},
                  "source": src if isinstance(src, list) else [src],
                  "execution_count": None, "outputs": []})

# ═══ Title ═══
md([
    "# FiberNet v4.0 Tutorial — 从生成到优化的完整流水线\n",
    "\n",
    "## Complete Pipeline: Structure Generation → Simulation → Analysis → ML → RL\n",
    "\n",
    "1. 生成 12 种基础结构类型 + 蜂巢基(honeycomb)变体\n",
    "2. 机械模拟（拉伸测试）\n",
    "3. 结构特征提取（94维）\n",
    "4. 机器学习预测力学性能\n",
    "5. 强化学习优化结构参数\n",
    "\n",
    "**关键参数**: `perturbation=0.40` (40% of mean edge length)"
])

# ═══ 1. Setup ═══
md("## 1. Setup / 环境设置")
code("""import os, sys, json, time, warnings, gc, copy
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

DATA_OUT = Path('tutorial_data')
VIZ_OUT = Path('tutorial_viz')
DATA_OUT.mkdir(exist_ok=True)
VIZ_OUT.mkdir(exist_ok=True)

print(f'✓ Data: {DATA_OUT.resolve()}')
print(f'✓ Viz:  {VIZ_OUT.resolve()}')""")

# ═══ 2. Import ═══
md("## 2. Import FiberNet / 导入验证")
code("""import fibernet as fn
from fibernet import pattern_2d, TaichiEngine, list_units
from fibernet.viz.render import render_graph, _get_theme
from fibernet.sim.accelerated import _graph_to_arrays
from fibernet.analysis import GraphFeatureExtractor

HAS_RL = False
try:
    from fibernet.rl import ParametricStructureEnv
    HAS_RL = True
except ImportError:
    try:
        from fibernet.rl.parametric import ParametricStructureEnv
        HAS_RL = True
    except ImportError:
        try:
            from fibernet.rl import create_rl_environment
            HAS_RL = True
        except ImportError:
            pass

print(f'✓ FiberNet loaded')
print(f'  Units ({len(list_units())}): {list_units()}')
print(f'  RL available: {HAS_RL}')""")

# ═══ 3.1 Base Units ═══
md(["## 3. Structure Generation / 结构生成\n",
    "\n", "### 3.1 12 种基础单元类型"])
code("""BOX = (1.0, 1.0)
GRID = (4, 4)
units = list_units()

base_structures = {}
print('Generating 12 base unit types (grid=4×4):')
for unit in units:
    g = pattern_2d(unit=unit, box=BOX, grid=GRID)
    base_structures[unit] = g
    print(f'  {unit:12s}: {g.num_nodes:4d} nodes, {g.num_edges:4d} edges')

gallery_data = []
for unit, g in base_structures.items():
    pos, elems, nids, _ = _graph_to_arrays(g)
    gallery_data.append({'unit': unit, 'num_nodes': g.num_nodes, 'num_edges': g.num_edges,
                         'positions': pos[:, :2].tolist(), 'elements': elems.tolist()})

with open(DATA_OUT / 'base_structures_gallery.json', 'w') as f:
    json.dump(gallery_data, f, indent=2)
print(f'\\n✓ Saved: base_structures_gallery.json')""")

md("### 3.1.1 Gallery — 12 Base Unit Types")
code("""for theme in ['dark', 'light']:
    colors = _get_theme(theme)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.patch.set_facecolor(colors['bg'])
    for ax, unit in zip(axes.flat, units):
        g = base_structures[unit]
        render_graph(g, ax=ax, theme=theme, color_by='uniform', line_width=1.5, show_nodes=False)
        ax.set_title(unit.replace('_', ' ').title(), color=colors['text'], fontsize=12, fontweight='bold')
    fig.suptitle('12 Base Unit Types (Undeformed, 4x4 grid)', color=colors['text'], fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = VIZ_OUT / f'01_gallery_undeformed_{theme}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=colors['bg'])
    plt.close(fig)
    print(f'  ✓ {path.name} ({path.stat().st_size/1024:.0f} KB)')
print('\\n✓ Gallery saved (dark + light)')""")

# ═══ 3.2 Perturbation ═══
md(["### 3.2 Intermediate Point Deformations (中间点变形)\n",
    "\n", "`perturbation`: 位移幅度 = **平均边长 × 百分比**"])
code("""perturbations = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80]
N_PTS = 3

for theme in ['dark', 'light']:
    colors = _get_theme(theme)
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.patch.set_facecolor(colors['bg'])
    for ax, pert in zip(axes.flat, perturbations):
        if pert == 0.0:
            g = pattern_2d(unit='honeycomb', box=BOX, grid=GRID, seed=42, n_pts_per_side=0)
        else:
            g = pattern_2d(unit='honeycomb', box=BOX, grid=GRID, seed=42, n_pts_per_side=N_PTS, perturbation=pert)
        render_graph(g, ax=ax, theme=theme, color_by='uniform', line_width=1.2, show_nodes=False)
        if pert == 0.0:
            label = 'perturbation=0.00\\n(no intermediate points)'
        else:
            label = f'perturbation={pert:.2f}\\n({pert*100:.0f}% of edge length)'
        ax.set_title(label, color=colors['text'], fontsize=10)
    fig.suptitle('Intermediate Point Deformations (Honeycomb, n_pts_per_side=3)', color=colors['text'], fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = VIZ_OUT / f'02_perturbation_comparison_{theme}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=colors['bg'])
    plt.close(fig)
    print(f'  ✓ {path.name} ({path.stat().st_size/1024:.0f} KB)')
print('\\n✓ Perturbation comparison saved')""")

# ═══ 3.3 Deformed ═══
md("### 3.3 12 种基础单元 + 中间点变形")
code("""deformed_structures = {}
print('Generating deformed structures (n_pts_per_side=3, perturbation=0.40):')
for unit in units:
    g = pattern_2d(unit=unit, box=BOX, grid=GRID, n_pts_per_side=3, perturbation=0.40, seed=42)
    deformed_structures[unit] = g
    print(f'  {unit:12s}: {g.num_nodes:4d} nodes, {g.num_edges:4d} edges')

deformed_data = []
for unit, g in deformed_structures.items():
    pos, elems, nids, _ = _graph_to_arrays(g)
    deformed_data.append({'unit': unit, 'num_nodes': g.num_nodes, 'num_edges': g.num_edges,
                          'n_pts_per_side': 3, 'perturbation': 0.40,
                          'positions': pos[:, :2].tolist(), 'elements': elems.tolist()})
with open(DATA_OUT / 'deformed_structures_gallery.json', 'w') as f:
    json.dump(deformed_data, f, indent=2)
print(f'\\n✓ Saved: deformed_structures_gallery.json')""")

code("""for theme in ['dark', 'light']:
    colors = _get_theme(theme)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.patch.set_facecolor(colors['bg'])
    for ax, unit in zip(axes.flat, units):
        g = deformed_structures[unit]
        render_graph(g, ax=ax, theme=theme, color_by='uniform', line_width=1.2, show_nodes=False)
        ax.set_title(f"{unit.replace('_', ' ').title()}\\n(n_pts=3, pert=40%)", color=colors['text'], fontsize=10)
    fig.suptitle('12 Base Unit Types (With Intermediate Point Deformations)', color=colors['text'], fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = VIZ_OUT / f'03_gallery_deformed_{theme}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=colors['bg'])
    plt.close(fig)
    print(f'  ✓ {path.name} ({path.stat().st_size/1024:.0f} KB)')
print('\\n✓ Deformed gallery saved')""")

# ═══ 3.4 Batch ═══
md(["### 3.4 Batch Generation: Honeycomb Variants\n",
    "\n", "**教程用 20 个，生产用 2000 个。**"])
code("""UNIT = 'honeycomb'
N_PTS_PER_SIDE = 3
PERTURBATION = 0.40

N_STRUCTURES = 20
## N_STRUCTURES = 2000  # Production (uncomment)

print('Batch generation parameters:')
print(f'  UNIT:            {UNIT}')
print(f'  N_PTS_PER_SIDE:  {N_PTS_PER_SIDE}')
print(f'  PERTURBATION:    {PERTURBATION} ({PERTURBATION*100:.0f}%)')
print(f'  N_STRUCTURES:    {N_STRUCTURES}')
print(f'  GRID:            {GRID}')
print()

all_structures = []
all_metadata = []
for i in tqdm(range(N_STRUCTURES), desc='Generating'):
    g = pattern_2d(unit=UNIT, box=BOX, grid=GRID, seed=100+i,
                   n_pts_per_side=N_PTS_PER_SIDE, perturbation=PERTURBATION)
    all_structures.append(g)
    all_metadata.append({'name': f'honeycomb_{i:03d}', 'seed': 100+i,
                         'nodes': g.num_nodes, 'edges': g.num_edges, 'perturbation': PERTURBATION})

print(f'\\n✓ Generated {N_STRUCTURES} honeycomb structures')""")

# ═══ 3.5 Gallery ═══
md("### 3.5 Gallery — Random Selection")
code("""import random
if len(all_structures) <= 20:
    show_structures = all_structures
    show_metadata = all_metadata
    n_show = len(show_structures)
else:
    indices = sorted(random.sample(range(len(all_structures)), 20))
    show_structures = [all_structures[i] for i in indices]
    show_metadata = [all_metadata[i] for i in indices]
    n_show = 20

n_cols = 5
n_rows = (n_show + n_cols - 1) // n_cols

for theme in ['dark', 'light']:
    colors = _get_theme(theme)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    fig.patch.set_facecolor(colors['bg'])
    for idx, (ax, g, meta) in enumerate(zip(axes.flat, show_structures, show_metadata)):
        render_graph(g, ax=ax, theme=theme, color_by='uniform', line_width=1.0, show_nodes=False)
        ax.set_title(f"{meta['name']} (seed={meta['seed']})", color=colors['text'], fontsize=9)
    for ax in axes.flat[n_show:]:
        ax.axis('off')
    title = f'{n_show} Honeycomb Variants (perturbation={PERTURBATION})'
    if N_STRUCTURES > 20:
        title += f'\\n(Random sample from {N_STRUCTURES} total)'
    fig.suptitle(title, color=colors['text'], fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = VIZ_OUT / f'04_gallery_batch_{theme}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=colors['bg'])
    plt.close(fig)
    print(f'  ✓ {path.name} ({path.stat().st_size/1024:.0f} KB)')
print('\\n✓ Batch gallery saved')""")

# ═══ 4. Simulation ═══
md(["## 4. Simulation / 模拟\n",
    "\n", "**API**: `TaichiEngine.stretch_test()`\n",
    "\n", "**参数**: stiffness=1e5, damping=0.3, 15000 steps, 50% ramp, stretch=1.5x"])
code("""STIFFNESS = 1e5
DAMPING = 0.3
NUM_STEPS = 15000
RAMP_FRACTION = 0.5
TARGET_STRETCH = 1.5

print('Simulation parameters:')
print(f'  STIFFNESS:       {STIFFNESS:.1e} N/m')
print(f'  DAMPING:         {DAMPING}')
print(f'  NUM_STEPS:       {NUM_STEPS}')
print(f'  RAMP_FRACTION:   {RAMP_FRACTION}')
print(f'  TARGET_STRETCH:  {TARGET_STRETCH}x')
print()

engine = TaichiEngine()
sim_results = []

ckpt_path = DATA_OUT / 'sim_checkpoint.json'
start_idx = 0
if ckpt_path.exists():
    with open(ckpt_path) as f:
        ckpt = json.load(f)
    start_idx = len(ckpt)
    sim_results = ckpt
    print(f'Resuming from checkpoint: {start_idx}/{N_STRUCTURES}')

for i in tqdm(range(start_idx, N_STRUCTURES), desc='Simulating'):
    g = all_structures[i]
    result = engine.stretch_test(
        g, target_stretch=TARGET_STRETCH, stiffness=STIFFNESS,
        damping=DAMPING, num_steps=NUM_STEPS, ramp_fraction=RAMP_FRACTION,
        save_interval=500
    )
    sim_dict = result.to_dict()
    sim_dict['index'] = i
    sim_dict['name'] = all_metadata[i]['name']
    sim_results.append(sim_dict)
    result.save(str(DATA_OUT / f"{all_metadata[i]['name']}_sim.json"), detailed=True)
    if (i + 1) % 5 == 0:
        with open(ckpt_path, 'w') as f:
            json.dump(sim_results, f)
        gc.collect()

with open(ckpt_path, 'w') as f:
    json.dump(sim_results, f)

for meta, sr in zip(all_metadata, sim_results):
    meta['max_force'] = sr['max_force'] if isinstance(sr, dict) else sr.max_force
    meta['max_stretch'] = sr['max_stretch'] if isinstance(sr, dict) else sr.max_stretch
    meta['energy'] = sr['energy'] if isinstance(sr, dict) else sr.energy

forces = [m['max_force'] for m in all_metadata]
print(f'\\n✓ Completed {len(sim_results)} simulations')
print(f'  Force range: {min(forces):.0f} - {max(forces):.0f} N')
print(f'  Mean force:  {np.mean(forces):.0f} N')""")

# ═══ 4.2 Helper ═══
md(["### 4.2 Deformation Trajectory (变形轨迹)"])
code("""def _setup_ax(ax, colors):
    ax.set_facecolor(colors['bg'])
    ax.tick_params(colors=colors['text'])
    for spine in ax.spines.values():
        spine.set_color(colors['grid'])
    ax.xaxis.label.set_color(colors['text'])
    ax.yaxis.label.set_color(colors['text'])
    ax.title.set_color(colors['text'])

def draw_deformed_structure(g, sim_result_or_dict, ax, colors, color_by_stretch=False):
    '''Draw a structure in its deformed state.'''
    if isinstance(sim_result_or_dict, dict):
        pos_def = np.array(sim_result_or_dict['deformed_positions'])
    else:
        pos_def = sim_result_or_dict.deformed_positions
    pos_orig, elements, node_ids, _ = _graph_to_arrays(g)
    if color_by_stretch:
        lo = np.array([np.linalg.norm(pos_orig[elements[e,1]] - pos_orig[elements[e,0]]) for e in range(len(elements))])
        ld = np.array([np.linalg.norm(pos_def[elements[e,1]] - pos_def[elements[e,0]]) for e in range(len(elements))])
        stretch = ld / (lo + 1e-12)
        norm = Normalize(vmin=stretch.min(), vmax=stretch.max())
        cmap = plt.cm.RdYlGn_r
        segments, colors_list = [], []
        for ei, e in enumerate(elements):
            p0, p1 = pos_def[e[0]], pos_def[e[1]]
            segments.append([[p0[0], p0[1]], [p1[0], p1[1]]])
            colors_list.append(cmap(norm(stretch[ei])))
        lc = LineCollection(segments, colors=colors_list, linewidths=1.5, capstyle='round')
        ax.add_collection(lc); ax.set_aspect('equal'); ax.autoscale()
        return stretch
    else:
        for e in elements:
            ax.plot([pos_def[e[0],0], pos_def[e[1],0]], [pos_def[e[0],1], pos_def[e[1],1]],
                    color=colors['fiber'], linewidth=1.2, alpha=0.8)
        ax.set_aspect('equal'); ax.autoscale()
        return None

print('✓ Helpers defined: _setup_ax(), draw_deformed_structure()')""")

code("""g0 = all_structures[0]
result0 = engine.stretch_test(g0, target_stretch=TARGET_STRETCH, stiffness=STIFFNESS,
    damping=DAMPING, num_steps=NUM_STEPS, ramp_fraction=RAMP_FRACTION, save_interval=500)
print(f'Structure 0: max_force={result0.max_force:.0f} N, max_stretch={result0.max_stretch:.3f}')

traj = result0.positions_trajectory
if traj is None:
    pos_orig, _, _, _ = _graph_to_arrays(g0)
    traj = [pos_orig, result0.deformed_positions]
print(f'Trajectory frames: {len(traj)}')

for theme in ['dark', 'light']:
    colors = _get_theme(theme)
    n_frames = min(8, len(traj))
    frame_indices = np.linspace(0, len(traj)-1, n_frames, dtype=int)
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.patch.set_facecolor(colors['bg'])
    axes = axes.flatten()
    pos_orig, elements, node_ids, _ = _graph_to_arrays(g0)
    for idx, fi in enumerate(frame_indices):
        ax = axes[idx]; ax.set_facecolor(colors['bg'])
        pos_frame = np.array(traj[fi])
        for e in elements:
            ax.plot([pos_frame[e[0],0], pos_frame[e[1],0]], [pos_frame[e[0],1], pos_frame[e[1],1]],
                    color=colors['fiber'], linewidth=1.0, alpha=0.8)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(f'Frame {fi}/{len(traj)-1}', color=colors['text'], fontsize=10)
    fig.suptitle(f'Deformation Trajectory: {all_metadata[0]["name"]} (8 frames)',
                 color=colors['text'], fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = VIZ_OUT / f'05_trajectory_{theme}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=colors['bg'])
    plt.close(fig)
    print(f'  ✓ {path.name} ({path.stat().st_size/1024:.0f} KB)')
print('\\n✓ Trajectory saved')""")

# ═══ 4.3 Stress ═══
md("### 4.3 Stress Distribution (应力分布)")
code("""for theme in ['dark', 'light']:
    colors = _get_theme(theme)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    fig.patch.set_facecolor(colors['bg'])
    render_graph(g0, ax=ax1, theme=theme, color_by='uniform', line_width=1.5, show_nodes=False)
    ax1.set_title('Original', color=colors['text'], fontsize=14)
    ax2.set_facecolor(colors['bg'])
    stretch = draw_deformed_structure(g0, result0, ax2, colors, color_by_stretch=True)
    ax2.tick_params(colors=colors['text'])
    for spine in ax2.spines.values(): spine.set_color(colors['grid'])
    norm = Normalize(vmin=stretch.min(), vmax=stretch.max())
    sm = ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Stretch Ratio', color=colors['text']); cbar.ax.tick_params(colors=colors['text'])
    ax2.set_title(f'Deformed (Stretch: {stretch.min():.2f}-{stretch.max():.2f})', color=colors['text'], fontsize=14)
    fig.suptitle(f'Stress Distribution: {all_metadata[0]["name"]}', color=colors['text'], fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = VIZ_OUT / f'06_stress_distribution_{theme}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=colors['bg'])
    plt.close(fig)
    print(f'  ✓ {path.name} ({path.stat().st_size/1024:.0f} KB)')
print('\\n✓ Stress distribution saved')""")

# ═══ 4.4 Stats ═══
md("### 4.4 Batch Statistics")
code("""forces = [m['max_force'] for m in all_metadata]
stretches = [m['max_stretch'] for m in all_metadata]
energies = [m['energy'] for m in all_metadata]

for theme in ['dark', 'light']:
    colors = _get_theme(theme)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12)); fig.patch.set_facecolor(colors['bg'])
    for ax in axes.flat: _setup_ax(ax, colors)
    ax = axes[0,0]; ax.hist(forces, bins=max(5, N_STRUCTURES//2), color=colors['fiber'], alpha=0.7, edgecolor=colors['grid'])
    ax.axvline(np.mean(forces), color='red', ls='--', lw=2); ax.set_xlabel('Max Force (N)'); ax.set_ylabel('Count')
    ax.set_title(f'Force Distribution\\nMean={np.mean(forces):.0f} N')
    ax = axes[0,1]; ax.plot(range(len(forces)), forces, 'o-', color=colors['fiber'], lw=2, ms=6)
    ax.set_xlabel('Structure Index'); ax.set_ylabel('Max Force (N)'); ax.set_title('Force by Structure')
    ax.grid(True, alpha=0.3, color=colors['grid'])
    ax = axes[1,0]; ax.hist(energies, bins=max(5, N_STRUCTURES//2), color=colors.get('accent', colors['fiber']), alpha=0.7, edgecolor=colors['grid'])
    ax.set_xlabel('Energy'); ax.set_ylabel('Count'); ax.set_title(f'Energy Distribution\\nMean={np.mean(energies):.0f}')
    ax = axes[1,1]; ax.hist(stretches, bins=max(5, N_STRUCTURES//2), color=colors['fiber'], alpha=0.7, edgecolor=colors['grid'])
    ax.set_xlabel('Max Stretch Ratio'); ax.set_ylabel('Count'); ax.set_title(f'Stretch Distribution\\nMean={np.mean(stretches):.3f}')
    fig.suptitle(f'Batch Statistics ({N_STRUCTURES} structures)', color=colors['text'], fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = VIZ_OUT / f'07_batch_stats_{theme}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=colors['bg']); plt.close(fig)
    print(f'  ✓ {path.name} ({path.stat().st_size/1024:.0f} KB)')
print('\\n✓ Batch stats saved')""")

# ═══ 5. Features ═══
md(["## 5. Feature Extraction / 特征提取\n", "\n", "提取 94 维结构特征。"])
code("""extractor = GraphFeatureExtractor()
for i, (g, meta) in enumerate(tqdm(list(zip(all_structures, all_metadata)), desc='Extracting')):
    meta.update(extractor.extract(g))

feature_keys = [k for k in all_metadata[0].keys() if k not in
    ['name','seed','nodes','edges','max_force','max_stretch','energy','perturbation']]
valid_features = [k for k in feature_keys
    if np.std([m[k] for m in all_metadata]) > 1e-12 and not all(m[k]==0 for m in all_metadata)]

print(f'✓ {len(valid_features)} valid features / {len(feature_keys)} total')
for k in valid_features[:8]:
    vals = [m[k] for m in all_metadata]
    print(f'  {k:25s}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}')

import pandas as pd
df = pd.DataFrame(all_metadata)
df.to_csv(DATA_OUT / 'structures_features.csv', index=False)
print(f'\\n✓ Saved: structures_features.csv')""")

md("### 5.1 Feature Statistics")
code("""variances = {k: np.var([m[k] for m in all_metadata]) for k in valid_features}
top_features = sorted(variances, key=variances.get, reverse=True)[:20]

for theme in ['dark', 'light']:
    colors = _get_theme(theme)
    fig, axes = plt.subplots(4, 5, figsize=(20, 16)); fig.patch.set_facecolor(colors['bg'])
    for i, (feat, ax) in enumerate(zip(top_features, axes.flat)):
        _setup_ax(ax, colors)
        vals = [m[feat] for m in all_metadata]
        ax.hist(vals, bins=max(5,N_STRUCTURES//3), color=colors['fiber'], alpha=0.7, edgecolor=colors['grid'])
        ax.set_title(feat.replace('_','\\n'), color=colors['text'], fontsize=8); ax.set_ylabel('Count', fontsize=8)
    for ax in axes.flat[len(top_features):]: ax.axis('off')
    fig.suptitle(f'Top 20 Features ({len(valid_features)} valid / {len(feature_keys)} total)', color=colors['text'], fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = VIZ_OUT / f'08_feature_stats_{theme}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=colors['bg']); plt.close(fig)
    print(f'  ✓ {path.name} ({path.stat().st_size/1024:.0f} KB)')
print('\\n✓ Feature stats saved')""")

# ═══ 6. ML ═══
md(["## 6. Machine Learning / 机器学习"])
code("""from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, accuracy_score

X = np.array([[m[k] for k in valid_features] for m in all_metadata])
y = np.array([m['max_force'] for m in all_metadata])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8, oob_score=True)
rf_reg.fit(X_train, y_train)
y_pred = rf_reg.predict(X_test)
r2 = r2_score(y_test, y_pred); rmse = np.sqrt(mean_squared_error(y_test, y_pred))

threshold = np.median(y)
y_train_bin = (y_train > threshold).astype(int); y_test_bin = (y_test > threshold).astype(int)
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train_bin); y_pred_bin = rf_clf.predict(X_test)
cm = confusion_matrix(y_test_bin, y_pred_bin); acc = accuracy_score(y_test_bin, y_pred_bin)

print(f'✓ R²={r2:.3f}, RMSE={rmse:.0f} N, OOB={rf_reg.oob_score_:.3f}, Acc={acc:.3f}')""")

code("""importances = rf_reg.feature_importances_
top_idx = np.argsort(importances)[-15:][::-1]

for theme in ['dark', 'light']:
    colors = _get_theme(theme)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14)); fig.patch.set_facecolor(colors['bg'])
    for ax in axes.flat: _setup_ax(ax, colors)
    ax = axes[0,0]
    ax.scatter(y_test, y_pred, color=colors['fiber'], alpha=0.7, s=60)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, alpha=0.5)
    ax.set_xlabel('Actual (N)'); ax.set_ylabel('Predicted (N)')
    ax.set_title(f'Predictions vs Actual\\nR²={r2:.3f}, RMSE={rmse:.0f}N')
    ax = axes[0,1]
    dn = [valid_features[i].replace('_',' ').title() for i in top_idx]
    ax.barh(range(15), importances[top_idx], color=colors['fiber'], alpha=0.7)
    ax.set_yticks(range(15)); ax.set_yticklabels(dn, fontsize=8)
    ax.set_xlabel('Importance'); ax.set_title('Top 15 Features'); ax.invert_yaxis()
    ax = axes[1,0]
    im = ax.imshow(cm, cmap='viridis'); plt.colorbar(im, ax=ax)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Low','High']); ax.set_yticklabels(['Low','High'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title(f'Confusion Matrix\\nAcc={acc:.2f}')
    for ci in range(2):
        for cj in range(2):
            ax.text(cj, ci, str(cm[ci,cj]), ha='center', va='center', color='white' if cm[ci,cj]<cm.max()/2 else 'black', fontsize=20, fontweight='bold')
    ax = axes[1,1]
    n_est_range = [10, 20, 50, 100, 200]; oob_errors = []
    for ne in n_est_range:
        rft = RandomForestRegressor(n_estimators=ne, random_state=42, oob_score=True, max_depth=8)
        rft.fit(X_train, y_train); oob_errors.append(1 - rft.oob_score_)
    ax.plot(n_est_range, oob_errors, 'o-', color=colors['fiber'], lw=2, ms=8)
    ax.set_xlabel('Trees'); ax.set_ylabel('OOB Error'); ax.set_title('Complexity vs Error')
    ax.grid(True, alpha=0.3, color=colors['grid'])
    fig.suptitle('ML Analysis', color=colors['text'], fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = VIZ_OUT / f'09_ml_analysis_{theme}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=colors['bg']); plt.close(fig)
    print(f'  ✓ {path.name} ({path.stat().st_size/1024:.0f} KB)')
print('\\n✓ ML analysis saved')""")

# ═══ 6.3 Correlation ═══
md("### 6.3 Force-Feature Correlation")
code("""corr = {}
y_series = pd.Series([m['max_force'] for m in all_metadata])
for k in valid_features:
    corr[k] = abs(pd.Series([m[k] for m in all_metadata]).corr(y_series))
top_corr = sorted(corr.items(), key=lambda x: x[1], reverse=True)[:15]

for theme in ['dark', 'light']:
    colors = _get_theme(theme)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8)); fig.patch.set_facecolor(colors['bg'])
    for ax in [ax1, ax2]: _setup_ax(ax, colors)
    ax1.barh(range(15), [c for _,c in top_corr], color=colors['fiber'], alpha=0.7)
    ax1.set_yticks(range(15)); ax1.set_yticklabels([k.replace('_',' ').title() for k,_ in top_corr], fontsize=8)
    ax1.set_xlabel('|Correlation|'); ax1.set_title('Top 15 Correlations'); ax1.invert_yaxis()
    tf = top_corr[0][0]
    ax2.scatter([m[tf] for m in all_metadata], [m['max_force'] for m in all_metadata], color=colors['fiber'], alpha=0.7, s=60)
    ax2.set_xlabel(tf.replace('_',' ').title()); ax2.set_ylabel('Max Force (N)')
    ax2.set_title(f'Top: {tf}\\nCorr={top_corr[0][1]:.3f}'); ax2.grid(True, alpha=0.3, color=colors['grid'])
    fig.suptitle('Force-Feature Importance', color=colors['text'], fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = VIZ_OUT / f'10_force_feature_importance_{theme}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=colors['bg']); plt.close(fig)
    print(f'  ✓ {path.name} ({path.stat().st_size/1024:.0f} KB)')
print('\\n✓ Correlation saved')""")

# ═══ 7. RL ═══
md(["## 7. Reinforcement Learning / 强化学习\n",
    "\n", "**注意**: 需要 fibernet >= 4.0。如导入失败请从源码安装。"])
code("""if not HAS_RL:
    print('⚠ RL module not available. Install from source:')
    print('  pip install git+https://github.com/GellmanSparrowS/fibernet')
    print('Skipping RL section.')
else:
    try:
        env = ParametricStructureEnv(unit=UNIT, box=BOX, grid=GRID, n_pts_per_side=N_PTS_PER_SIDE,
            stiffness=STIFFNESS, damping=DAMPING, num_steps=5000, target_stretch=TARGET_STRETCH, reward_mode='minimize_force')
    except Exception:
        from fibernet.rl import create_rl_environment
        env = create_rl_environment(unit=UNIT, grid=GRID, n_pts_per_side=N_PTS_PER_SIDE,
            stiffness=STIFFNESS, num_steps=5000, target_stretch=TARGET_STRETCH, reward_mode='minimize_force', box=BOX)
    print(f'✓ RL Environment: n_actions={env.n_actions}, reward_mode={env.reward_mode}')""")

code("""if HAS_RL:
    N_EPISODES = 50
    rewards_history = []
    print(f'Running {N_EPISODES} episodes...')
    for ep in tqdm(range(N_EPISODES), desc='RL'):
        obs = env.reset()
        action = np.random.uniform(-0.3, 0.3, size=env.n_actions)
        graph, sim_result, reward, info = env.step(action)
        rewards_history.append(reward)
        if (ep + 1) % 10 == 0: gc.collect()
    print(f'\\n✓ Reward range: {min(rewards_history):.0f} - {max(rewards_history):.0f}')
else:
    rewards_history = []
    print('⚠ Skipping RL training')""")

code("""if HAS_RL and rewards_history:
    for theme in ['dark', 'light']:
        colors = _get_theme(theme)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6)); fig.patch.set_facecolor(colors['bg'])
        for ax in [ax1, ax2]: _setup_ax(ax, colors)
        ep = np.arange(len(rewards_history))
        ax1.plot(ep, rewards_history, 'o-', color=colors['fiber'], lw=1.5, ms=4, alpha=0.7, label='Reward')
        w = min(5, len(rewards_history))
        if len(rewards_history) > w:
            sm = np.convolve(rewards_history, np.ones(w)/w, mode='valid')
            ax1.plot(ep[w-1:], sm, color=colors.get('accent','red'), lw=2, label=f'Moving avg ({w})')
        ax1.set_xlabel('Episode'); ax1.set_ylabel('Reward'); ax1.set_title('RL Progress'); ax1.legend()
        ax1.grid(True, alpha=0.3, color=colors['grid'])
        ax2.hist(rewards_history, bins=15, color=colors['fiber'], alpha=0.7, edgecolor=colors['grid'])
        ax2.axvline(np.mean(rewards_history), color='red', ls='--', lw=2)
        ax2.set_xlabel('Reward'); ax2.set_ylabel('Count')
        ax2.set_title(f'Distribution\\nMean={np.mean(rewards_history):.0f}')
        fig.suptitle('RL Training Analysis', color=colors['text'], fontsize=14, fontweight='bold', y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        path = VIZ_OUT / f'11_rl_reward_{theme}.png'
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=colors['bg']); plt.close(fig)
        print(f'  ✓ {path.name} ({path.stat().st_size/1024:.0f} KB)')
    print('\\n✓ RL visualization saved')
else:
    print('⚠ Skipping RL visualization')""")

# ═══ 8. Summary ═══
md(["## 8. Summary / 总结\n",
    "\n", "| # | 可视化 | dark+light |\n",
    "|---|--------|------------|\n",
    "| 01 | 12种基础单元 | ✓ |\n",
    "| 02 | perturbation对比 0%~80% | ✓ |\n",
    "| 03 | 12种+变形 | ✓ |\n",
    "| 04 | 批量gallery | ✓ |\n",
    "| 05 | 变形轨迹 8帧 | ✓ |\n",
    "| 06 | 应力分布 | ✓ |\n",
    "| 07 | 批量统计 | ✓ |\n",
    "| 08 | 特征统计 | ✓ |\n",
    "| 09 | ML分析 | ✓ |\n",
    "| 10 | 力-特征相关性 | ✓ |\n",
    "| 11 | RL奖励曲线 | ✓ (需RL) |\n",
    "\n",
    "**生产参数**: `N_STRUCTURES=2000, PERTURBATION=0.40`"
])

# ═══ Save ═══
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.3"}
    },
    "cells": cells
}

for p in ['/media/sf_share/fibernet_v4_tutorial_updated.ipynb',
          '/home/codex/projects/codex_test/fibernet/tutorials/v4_tutorial/fibernet_v4_tutorial_updated.ipynb']:
    with open(p, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

md_count = sum(1 for c in cells if c["cell_type"] == "markdown")
code_count = sum(1 for c in cells if c["cell_type"] == "code")
print(f'✓ Notebook saved: {len(cells)} cells ({md_count} MD, {code_count} code)')
