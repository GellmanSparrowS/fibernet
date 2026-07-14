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
    "# FiberNet v4.0 Tutorial — Complete Pipeline\n",
    "\n",
    "## Structure Generation → Simulation → Feature Extraction → ML → RL\n",
    "\n",
    "从生成到优化的完整流水线：\n",
    "\n",
    "1. Generate 12 base structure types + honeycomb variants\n",
    "2. Mechanical simulation (stretch test)\n",
    "3. Structural feature extraction (94-dim)\n",
    "4. Machine learning prediction of mechanical properties\n",
    "5. Reinforcement learning optimization of structure parameters\n",
    "\n",
    "**Key parameter**: `perturbation=0.40` (40% of mean edge length)"
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
from fibernet.sim.accelerated import _graph_to_arrays, _get_boundary_indices
from fibernet.sim import SimResult
from fibernet.analysis import GraphFeatureExtractor

# ── Version check ──
MIN_VERSION = '4.0.0'
v = fn.__version__
print(f'✓ FiberNet v{v} loaded')
# Simple version check (no external dependency)
if tuple(int(x) for x in v.split('.')) < tuple(int(x) for x in MIN_VERSION.split('.')):
    print(f'⚠ Version {v} < {MIN_VERSION}. Some features may not work.')
    print(f'  Upgrade: pip install fibernet --upgrade')
else:
    print(f'  Version OK (>= {MIN_VERSION})')

print(f'  Available units ({len(list_units())}): {list_units()}')

# RL import — try multiple paths for version compatibility
HAS_RL = False
ParametricStructureEnv = None
for _path in [
    'from fibernet.rl import ParametricStructureEnv',
    'from fibernet.rl.parametric import ParametricStructureEnv',
]:
    try:
        exec(_path)
        HAS_RL = True
        break
    except (ImportError, ModuleNotFoundError, Exception):
        continue

if not HAS_RL:
    try:
        from fibernet.rl import create_rl_environment
        HAS_RL = 'factory'
    except (ImportError, ModuleNotFoundError, Exception):
        pass

print(f'  RL available: {bool(HAS_RL)}')""")

# ═══ 3.1 Base Units ═══
md(["## 3. Structure Generation / 结构生成\n",
    "\n", "### 3.1 Twelve Base Unit Types (12种基础单元类型)"])
code("""BOX = (1.0, 1.0)
GRID = (4, 4)
units = list_units()

base_structures = {}
print('Generating 12 base unit types (grid=4×4):')
for unit in units:
    g = pattern_2d(unit=unit, box=BOX, grid=GRID)
    base_structures[unit] = g
    print(f'  {unit:12s}: {g.num_nodes:4d} nodes, {g.num_edges:4d} edges')

# Save gallery JSON (structure data for each unit)
STRUCT_DIR = DATA_OUT / 'structures'
STRUCT_DIR.mkdir(parents=True, exist_ok=True)

gallery_data = []
for unit, g in base_structures.items():
    pos, elems, nids, _ = _graph_to_arrays(g)
    gdata = {'unit': unit, 'num_nodes': g.num_nodes, 'num_edges': g.num_edges,
             'positions': pos[:, :2].tolist(), 'elements': elems.tolist()}
    gallery_data.append(gdata)
    with open(STRUCT_DIR / f'base_{unit}.json', 'w') as f:
        json.dump(gdata, f, indent=1)

with open(STRUCT_DIR / 'base_gallery_all.json', 'w') as f:
    json.dump(gallery_data, f, indent=1)
print(f'\\n✓ Saved {len(base_structures)} base structure JSONs to {STRUCT_DIR}')""")

md("### 3.1.1 Gallery — 12 Base Unit Types (Undeformed / 未变形)")
code("""for theme in ['dark', 'light']:
    colors = _get_theme(theme)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.patch.set_facecolor(colors['bg'])
    for ax, unit in zip(axes.flat, units):
        g = base_structures[unit]
        render_graph(g, ax=ax, theme=theme, color_by='uniform', line_width=1.5, show_nodes=False)
        ax.set_title(unit.replace('_', ' ').title(), color=colors['text'], fontsize=12, fontweight='bold')
    fig.suptitle('12 Base Unit Types (Undeformed, 4×4 grid)', color=colors['text'], fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = VIZ_OUT / f'01_gallery_undeformed_{theme}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=colors['bg'])
    plt.close(fig)
    print(f'  ✓ {path.name} ({path.stat().st_size/1024:.0f} KB)')
print('\\n✓ Gallery saved (dark + light)')""")

# ═══ 3.2 Perturbation ═══
md(["### 3.2 Intermediate Point Deformations (中间点变形)\n",
    "\n", "`perturbation`: displacement amplitude = **mean edge length × percentage** (位移幅度 = 平均边长 × 百分比)"])
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
md("### 3.3 12 Base Units + Intermediate Point Deformations (12种基础单元+中间点变形)")
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
md(["### 3.4 Batch Generation: Honeycomb Variants (批量生成: 蜂巢基变体)\n",
    "\n", "**Tutorial uses 20 structures; production uses 2000.** (教程用20个，生产用2000个)"])
code("""UNIT = 'honeycomb'
N_PTS_PER_SIDE = 3
PERTURBATION = 0.40

N_STRUCTURES = 2000
## N_STRUCTURES = 200000  # Production (uncomment for full batch)

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
md("### 3.5 Gallery — Random Selection (随机选取展示)")
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
    "\n", "**API**: `TaichiEngine.dynamics()` — displacement-controlled uniaxial stretch with rigid plate boundaries (单轴拉伸+刚性板边界)\n",
    "\n", "**Key**: Left 10% nodes = fixed (rigid plate), right 10% nodes = uniform displacement (rigid plate). 80% time for relaxation.\n",
    "\n", "**Parameters**: stiffness=1e5, damping=0.3, 30000 steps, 20% ramp + 80% relaxation, stretch=1.5×"])
code("""STIFFNESS = 1e5
DAMPING = 0.3
NUM_STEPS = 30000
RAMP_FRACTION = 0.2
TARGET_STRETCH = 1.5

print('Simulation parameters:')
print(f'  STIFFNESS:       {STIFFNESS:.1e} N/m')
print(f'  DAMPING:         {DAMPING}')
print(f'  NUM_STEPS:       {NUM_STEPS}')
print(f'  RAMP_FRACTION:   {RAMP_FRACTION} ({RAMP_FRACTION*100:.0f}% ramp + {(1-RAMP_FRACTION)*100:.0f}% relaxation)')
print(f'  TARGET_STRETCH:  {TARGET_STRETCH}x')
print(f'  Boundary:        10% each side (rigid plates)')
print()

engine = TaichiEngine()

# ── Directories for saving ──
SIM_DIR = DATA_OUT / 'simulations'
SIM_DIR.mkdir(parents=True, exist_ok=True)

# Save all batch structures as JSON
for i, g in enumerate(all_structures):
    pos, elems, nids, _ = _graph_to_arrays(g)
    sdata = {'name': all_metadata[i]['name'], 'seed': all_metadata[i]['seed'],
             'num_nodes': g.num_nodes, 'num_edges': g.num_edges,
             'perturbation': PERTURBATION,
             'positions': pos[:, :2].tolist(), 'elements': elems.tolist()}
    with open(STRUCT_DIR / f'{all_metadata[i]["name"]}.json', 'w') as f:
        json.dump(sdata, f, indent=1)
print(f'✓ Saved {N_STRUCTURES} structure JSONs to {STRUCT_DIR}')


# ── Helper: get boundary indices (compatible with older fibernet versions) ──
def get_boundary_nodes(pos, pct=0.10):
    # Get left/right boundary node indices. Compatible with all fibernet versions.
    try:
        bnd = _get_boundary_indices(pos, pct=pct)
        return bnd['left'], bnd['right']
    except TypeError:
        x_sorted = np.argsort(pos[:, 0])
        n = len(pos)
        n_bnd = max(int(n * pct), 1)
        return list(x_sorted[:n_bnd]), list(x_sorted[-n_bnd:])

# ── Helper: run dynamics simulation with rigid plate boundaries ──
def run_stretch(g, stiffness=STIFFNESS, damping=DAMPING, num_steps=NUM_STEPS,
                ramp_fraction=RAMP_FRACTION, target_stretch=TARGET_STRETCH,
                save_interval=500, boundary_pct=0.10):
    pos, elems, nids, _ = _graph_to_arrays(g)
    left_nodes, right_nodes = get_boundary_nodes(pos, pct=boundary_pct)
    L_x = pos[:, 0].max() - pos[:, 0].min()
    target_disp = L_x * (target_stretch - 1)
    ramp_steps = int(num_steps * ramp_fraction)

    # Right boundary: uniform displacement schedule (rigid plate)
    disp_schedule = {}
    for ni in right_nodes:
        disp_schedule[ni] = [(0, np.zeros(3)), (num_steps, np.array([target_disp, 0, 0]))]

    result = engine.dynamics(
        g,
        fixed_nodes=left_nodes,
        displacement_schedule=disp_schedule,
        stiffness=stiffness,
        damping=damping,
        dt=1e-4,
        num_steps=num_steps,
        save_interval=save_interval,
        dashpot=10.0,
        drag=1.0,
    )
    return result

# ── Checkpoint support ──
ckpt_path = SIM_DIR / 'checkpoint.json'
start_idx = 0
if ckpt_path.exists():
    with open(ckpt_path) as f:
        ckpt = json.load(f)
    start_idx = len(ckpt)
    print(f'Resuming from checkpoint: {start_idx}/{N_STRUCTURES}')
else:
    ckpt = []

for i in tqdm(range(start_idx, N_STRUCTURES), desc='Simulating'):
    g = all_structures[i]
    result = run_stretch(g)

    sim_dict = result.to_dict()
    sim_dict['index'] = i
    sim_dict['name'] = all_metadata[i]['name']
    ckpt.append(sim_dict)

    # Save individual result (lightweight, no trajectory in JSON)
    result.save(str(SIM_DIR / f'{all_metadata[i]["name"]}_sim.json'), detailed=False)

    # Update metadata
    all_metadata[i]['max_force'] = result.max_force
    all_metadata[i]['max_stretch'] = result.max_stretch
    all_metadata[i]['energy'] = result.energy

    # Checkpoint every 50 structures
    if (i + 1) % 50 == 0 or i == N_STRUCTURES - 1:
        with open(ckpt_path, 'w') as f:
            json.dump(ckpt, f, indent=1)
        gc.collect()

# Save final checkpoint
with open(ckpt_path, 'w') as f:
    json.dump(ckpt, f, indent=1)

# Load checkpoint if we resumed
if start_idx == 0:
    sim_results = ckpt
else:
    with open(ckpt_path) as f:
        sim_results = json.load(f)
    for meta, sr in zip(all_metadata, sim_results):
        meta['max_force'] = sr['max_force']
        meta['max_stretch'] = sr['max_stretch']
        meta['energy'] = sr['energy']

forces_kN = np.array([m['max_force'] for m in all_metadata]) / 1000.0
print(f'\\n✓ Completed {len(sim_results)} simulations')
print(f'  Force range: {forces_kN.min():.1f} - {forces_kN.max():.1f} kN')
print(f'  Mean force:  {forces_kN.mean():.1f} kN')
print(f'  Saved to: {SIM_DIR}')""")

# ═══ 4.2 Helper ═══
md(["### 4.2 Deformation Trajectory (变形轨迹)\n",
    "\n", "Visualize the stretching process frame-by-frame with stress coloring. (逐帧展示拉伸过程+应力着色)"])
code("""def _setup_ax(ax, colors):
    ax.set_facecolor(colors['bg'])
    ax.tick_params(colors=colors['text'])
    for spine in ax.spines.values():
        spine.set_color(colors['grid'])
    ax.xaxis.label.set_color(colors['text'])
    ax.yaxis.label.set_color(colors['text'])
    ax.title.set_color(colors['text'])

def draw_deformed_structure(g, sim_result_or_dict, ax, colors, color_by_stretch=False):
    if isinstance(sim_result_or_dict, dict):
        pos_def = np.array(sim_result_or_dict['deformed_positions'])
    else:
        pos_def = sim_result_or_dict.deformed_positions
    pos_orig, elements, node_ids, _ = _graph_to_arrays(g)
    if color_by_stretch:
        lo = np.array([np.linalg.norm(pos_orig[elements[e,1], :2] - pos_orig[elements[e,0], :2]) for e in range(len(elements))])
        ld = np.array([np.linalg.norm(pos_def[elements[e,1], :2] - pos_def[elements[e,0], :2]) for e in range(len(elements))])
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

code("""# ── Random sample for visualization ──
import random
if N_STRUCTURES <= 20:
    viz_indices = list(range(N_STRUCTURES))
else:
    viz_indices = sorted(random.sample(range(N_STRUCTURES), 20))
print(f'Visualization sample: {len(viz_indices)} structures (indices: {viz_indices[:10]}{"..." if len(viz_indices) > 10 else ""})')

# Load first viz structure for trajectory
idx0 = viz_indices[0]
g0 = all_structures[idx0]

# Re-run simulation for trajectory (detailed save)
result0 = run_stretch(g0, save_interval=500)
print(f'Structure {idx0}: max_force={result0.max_force/1000:.1f} kN, max_stretch={result0.max_stretch:.3f}')

traj = result0.positions_trajectory
if traj is None:
    pos_orig, _, _, _ = _graph_to_arrays(g0)
    traj = [pos_orig, result0.deformed_positions]
print(f'Trajectory frames: {len(traj)}')

# Boundary info for visualization
pos0, elems0, _, _ = _graph_to_arrays(g0)
left0, right0 = get_boundary_nodes(pos0, pct=0.10)

for theme in ['dark', 'light']:
    colors = _get_theme(theme)
    n_frames = min(8, len(traj))
    frame_indices = np.linspace(0, len(traj)-1, n_frames, dtype=int)
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.patch.set_facecolor(colors['bg'])
    axes = axes.flatten()
    for idx, fi in enumerate(frame_indices):
        ax = axes[idx]; ax.set_facecolor(colors['bg'])
        pos_frame = np.array(traj[fi])
        # Color edges by stretch ratio
        lo = np.array([np.linalg.norm(pos0[elems0[e,1], :2] - pos0[elems0[e,0], :2]) for e in range(len(elems0))])
        ld = np.array([np.linalg.norm(pos_frame[elems0[e,1], :2] - pos_frame[elems0[e,0], :2]) for e in range(len(elems0))])
        stretch = ld / (lo + 1e-12)
        norm = Normalize(vmin=stretch.min(), vmax=stretch.max())
        cmap = plt.cm.RdYlGn_r
        for ei, e in enumerate(elems0):
            p0, p1 = pos_frame[e[0]], pos_frame[e[1]]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                    color=cmap(norm(stretch[ei])), linewidth=1.0, alpha=0.9)
        # Mark boundary nodes
        ax.scatter(pos_frame[left0, 0], pos_frame[left0, 1],
                   c='cyan', s=6, zorder=5, alpha=0.5, label='Fixed (L)' if idx == 0 else '')
        ax.scatter(pos_frame[right0, 0], pos_frame[right0, 1],
                   c='magenta', s=6, zorder=5, alpha=0.5, label='Moved (R)' if idx == 0 else '')
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(f'Frame {fi}/{len(traj)-1}\\nStretch: {stretch.min():.3f}-{stretch.max():.3f}',
                     color=colors['text'], fontsize=9)
        if idx == 0:
            ax.legend(fontsize=7, loc='lower left', framealpha=0.5)
    fig.suptitle(f'Deformation Trajectory: {all_metadata[idx0]["name"]} (8 frames, rigid plates)',
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
    # Mark boundaries
    pos_def0 = result0.deformed_positions
    ax2.scatter(pos_def0[left0, 0], pos_def0[left0, 1], c='cyan', s=6, zorder=5, alpha=0.5, label='Fixed (L)')
    ax2.scatter(pos_def0[right0, 0], pos_def0[right0, 1], c='magenta', s=6, zorder=5, alpha=0.5, label='Moved (R)')
    ax2.legend(fontsize=8, loc='upper left', framealpha=0.5)
    ax2.set_title(f'Deformed (Stretch: {stretch.min():.2f}-{stretch.max():.2f})', color=colors['text'], fontsize=14)
    fig.suptitle(f'Stress Distribution: {all_metadata[idx0]["name"]} (rigid plates)', color=colors['text'], fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = VIZ_OUT / f'06_stress_distribution_{theme}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=colors['bg'])
    plt.close(fig)
    print(f'  ✓ {path.name} ({path.stat().st_size/1024:.0f} KB)')
print('\\n✓ Stress distribution saved')""")

# ═══ 4.4 Stats ═══
md("### 4.4 Batch Statistics (批量统计)")
code("""forces_kN = np.array([m['max_force'] for m in all_metadata]) / 1000.0
stretches = np.array([m['max_stretch'] for m in all_metadata])
energies = np.array([m['energy'] for m in all_metadata])
sort_idx = np.argsort(forces_kN)

for theme in ['dark', 'light']:
    colors = _get_theme(theme)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12)); fig.patch.set_facecolor(colors['bg'])
    for ax in axes.flat: _setup_ax(ax, colors)

    ax = axes[0,0]
    ax.hist(forces_kN, bins='auto', color=colors['fiber'], alpha=0.7, edgecolor=colors['grid'])
    ax.axvline(forces_kN.mean(), color='red', ls='--', lw=2, label=f'Mean={forces_kN.mean():.1f} kN')
    ax.axvline(np.median(forces_kN), color='orange', ls=':', lw=2, label=f'Median={np.median(forces_kN):.1f} kN')
    ax.set_xlabel('Max Force (kN)'); ax.set_ylabel('Count')
    ax.set_title(f'Force Distribution (n={N_STRUCTURES})'); ax.legend(fontsize=9)

    ax = axes[0,1]
    sorted_forces = forces_kN[sort_idx]
    sorted_labels = [all_metadata[i]['name'].replace('honeycomb_','#') for i in sort_idx]
    y_pos = np.arange(len(sorted_forces))
    ax.barh(y_pos, sorted_forces, color=colors['fiber'], alpha=0.7, edgecolor=colors['grid'], height=0.7)
    ax.set_yticks(y_pos); ax.set_yticklabels(sorted_labels, fontsize=8)
    ax.set_xlabel('Max Force (kN)'); ax.set_title('Force by Structure (sorted)')
    ax.grid(True, axis='x', alpha=0.3, color=colors['grid'])

    ax = axes[1,0]
    e_max = energies.max()
    if e_max > 1e6: e_plot, e_unit = energies/1e6, 'MJ'
    elif e_max > 1e3: e_plot, e_unit = energies/1e3, 'kJ'
    else: e_plot, e_unit = energies, 'J'
    ax.hist(e_plot, bins='auto', color=colors.get('accent', colors['fiber']), alpha=0.7, edgecolor=colors['grid'])
    ax.set_xlabel(f'Energy ({e_unit})'); ax.set_ylabel('Count')
    ax.set_title(f'Energy Distribution (Mean={energies.mean():.2e} {e_unit})')

    ax = axes[1,1]
    ax.hist(stretches, bins='auto', color=colors['fiber'], alpha=0.7, edgecolor=colors['grid'])
    ax.axvline(stretches.mean(), color='red', ls='--', lw=2, label=f'Mean={stretches.mean():.3f}')
    ax.set_xlabel('Max Stretch Ratio'); ax.set_ylabel('Count')
    ax.set_title('Stretch Distribution'); ax.legend(fontsize=9)

    fig.suptitle(f'Batch Statistics ({N_STRUCTURES} structures)', color=colors['text'], fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = VIZ_OUT / f'07_batch_stats_{theme}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=colors['bg']); plt.close(fig)
    print(f'  ✓ {path.name} ({path.stat().st_size/1024:.0f} KB)')
print('\\n✓ Batch stats saved')""")

# ═══ 5. Features ═══
md(["## 5. Feature Extraction / 特征提取\n", "\n", "Extract 94-dimensional structural features. (提取94维结构特征)"])
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

md("### 5.1 Feature Statistics (特征统计分布)")
code("""from scipy.stats import gaussian_kde

variances = {k: np.var([m[k] for m in all_metadata]) for k in valid_features}
top_features = sorted(variances, key=variances.get, reverse=True)[:20]

for theme in ['dark', 'light']:
    colors = _get_theme(theme)
    fig, axes = plt.subplots(4, 5, figsize=(20, 16)); fig.patch.set_facecolor(colors['bg'])
    for i, (feat, ax) in enumerate(zip(top_features, axes.flat)):
        _setup_ax(ax, colors)
        vals = np.array([m[feat] for m in all_metadata])
        n_bins = max(5, N_STRUCTURES // 3)
        ax.hist(vals, bins=n_bins, color=colors['fiber'], alpha=0.5, edgecolor=colors['grid'], density=True)
        if len(np.unique(vals)) > 1 and vals.std() > 0:
            try:
                kde = gaussian_kde(vals)
                x_range = np.linspace(vals.min() - vals.std()*0.5, vals.max() + vals.std()*0.5, 200)
                ax.plot(x_range, kde(x_range), color=colors.get('accent', 'red'), linewidth=2)
            except Exception:
                pass
        ax.set_title(feat.replace('_','\\n'), color=colors['text'], fontsize=8)
        ax.set_ylabel('Density', fontsize=8)
    for ax in axes.flat[len(top_features):]: ax.axis('off')
    fig.suptitle(f'Top 20 Features with KDE ({len(valid_features)} valid / {len(feature_keys)} total)', color=colors['text'], fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = VIZ_OUT / f'08_feature_stats_{theme}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=colors['bg']); plt.close(fig)
    print(f'  ✓ {path.name} ({path.stat().st_size/1024:.0f} KB)')
print('\\n✓ Feature stats saved')""")

# ═══ 6. ML ═══
md(["## 6. Machine Learning / 机器学习\n",
    "\n", "Train a Random Forest to predict mechanical properties from structural features. (使用随机森林预测力学性能)"])
code("""from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, accuracy_score
import joblib

X = np.array([[m[k] for k in valid_features] for m in all_metadata])
y_kN = np.array([m['max_force'] for m in all_metadata]) / 1000.0

# Train with early stopping via OOB score monitoring
print('Training RandomForest with early stopping...')
rf_reg = RandomForestRegressor(n_estimators=1, warm_start=True, random_state=42, max_depth=8, oob_score=True)
oob_scores = []
best_oob = -np.inf
patience = 50
no_improve = 0

for n_trees in range(10, 801, 10):
    rf_reg.n_estimators = n_trees
    rf_reg.fit(X, y_kN)
    oob_scores.append(rf_reg.oob_score_)
    
    if rf_reg.oob_score_ > best_oob + 1e-4:
        best_oob = rf_reg.oob_score_
        no_improve = 0
    else:
        no_improve += 1
    
    if no_improve >= patience:
        print(f'  Early stop at {n_trees} trees (best OOB={best_oob:.4f})')
        break
    
    if n_trees % 100 == 0:
        print(f'  Trees: {n_trees}, OOB: {rf_reg.oob_score_:.4f}')

print(f'Final model: {rf_reg.n_estimators} trees, OOB={rf_reg.oob_score_:.4f}')

# Cross-validation
cv_scores = cross_val_score(rf_reg, X, y_kN, cv=5, scoring='r2')
print(f'5-fold CV: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})')

# Save model
model_path = VIZ_OUT / 'rf_force_model.joblib'
joblib.dump(rf_reg, model_path)
print(f'✓ Model saved: {model_path}')

y_oob = rf_reg.oob_prediction_
r2 = r2_score(y_kN, y_oob)
rmse = np.sqrt(mean_squared_error(y_kN, y_oob))

terciles = np.percentile(y_kN, [33.3, 66.7])
y_cat = np.digitize(y_kN, terciles)
y_oob_cat = np.digitize(y_oob, terciles)
cm = confusion_matrix(y_cat, y_oob_cat, labels=[0,1,2])
acc = accuracy_score(y_cat, y_oob_cat)
cm_labels = ['Low', 'Mid', 'High']
cm_threshold = f'Low<{terciles[0]:.1f}<{terciles[1]:.1f}<High kN'

print(f'✓ OOB R²={r2:.3f}, RMSE={rmse:.1f} kN, OOB={rf_reg.oob_score_:.3f}, Acc={acc:.3f}')""")

code("""importances = rf_reg.feature_importances_
top_idx = np.argsort(importances)[::-1][:15]

n_est_range = np.arange(10, len(oob_scores)*10+1, 10)
oob_rmse = []
# Reconstruct RMSE from saved OOB scores
rf_tmp = RandomForestRegressor(n_estimators=1, warm_start=True, random_state=42, max_depth=8, oob_score=True)
for n_est in n_est_range:
    rf_tmp.n_estimators = n_est
    rf_tmp.fit(X, y_kN)
    oob_rmse.append(np.sqrt(mean_squared_error(y_kN, rf_tmp.oob_prediction_)))

for theme in ['dark', 'light']:
    colors = _get_theme(theme)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14)); fig.patch.set_facecolor(colors['bg'])
    for ax in axes.flat: _setup_ax(ax, colors)

    ax = axes[0,0]
    ax.scatter(y_kN, y_oob, color=colors['fiber'], alpha=0.7, s=60, edgecolors=colors.get('grid','white'), linewidths=0.5)
    lo = min(y_kN.min(), y_oob.min()); hi = max(y_kN.max(), y_oob.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=2, alpha=0.5, label='Perfect')
    ax.set_xlabel('Actual (kN)'); ax.set_ylabel('OOB Predicted (kN)')
    ax.set_title(f'OOB Predictions vs Actual (n={N_STRUCTURES})\\nR²={r2:.3f}, RMSE={rmse:.1f} kN')
    ax.legend(fontsize=9)

    ax = axes[0,1]
    dn = [valid_features[i].replace('_',' ').title() for i in top_idx]
    ax.barh(range(15), importances[top_idx], color=colors['fiber'], alpha=0.7)
    ax.set_yticks(range(15)); ax.set_yticklabels(dn, fontsize=8)
    ax.set_xlabel('Importance'); ax.set_title('Top 15 Features'); ax.invert_yaxis()

    ax = axes[1,0]
    im = ax.imshow(cm, cmap='viridis', aspect='auto'); plt.colorbar(im, ax=ax)
    n_cls = len(cm_labels)
    ax.set_xticks(range(n_cls)); ax.set_yticks(range(n_cls))
    ax.set_xticklabels(cm_labels); ax.set_yticklabels(cm_labels)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix (OOB, n={N_STRUCTURES})\\nAcc={acc:.2f}, {cm_threshold}')
    for ci in range(n_cls):
        for cj in range(n_cls):
            val = cm[ci,cj]
            tc = 'white' if val < (cm.max() or 1)/2 else 'black'
            ax.text(cj, ci, str(int(val)), ha='center', va='center', color=tc, fontsize=16, fontweight='bold')

    ax = axes[1,1]
    ax.plot(n_est_range, oob_rmse, 'o-', color=colors['fiber'], lw=2, ms=4, label='OOB RMSE (kN)')
    ax.set_xlabel('Number of Trees'); ax.set_ylabel('OOB RMSE (kN)')
    ax.set_title(f'Loss Curve (Final RMSE={oob_rmse[-1]:.1f} kN)'); ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, color=colors['grid'])

    fig.suptitle('ML Analysis (kN)', color=colors['text'], fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = VIZ_OUT / f'09_ml_analysis_{theme}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=colors['bg']); plt.close(fig)
    print(f'  ✓ {path.name} ({path.stat().st_size/1024:.0f} KB)')
print('\\n✓ ML analysis saved')""")

# ═══ 6.3 Correlation ═══
md("### 6.3 Force-Feature Correlation (力-特征相关性)")
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
    "\n", "**Note**: Requires `fibernet >= 4.0`. If import fails, install from source. (需要 fibernet >= 4.0，如导入失败请从源码安装)"])
code("""if not HAS_RL:
    print('⚠ RL module not available in your installed fibernet version.')
    print('  Your pip version may not include the RL module.')
    print('  Install from source to get RL:')
    print('    pip install git+https://github.com/GellmanSparrowS/fibernet')
    print('  Or clone and install:')
    print('    git clone https://github.com/GellmanSparrowS/fibernet.git')
    print('    cd fibernet && pip install -e .')
    print('\\nSkipping RL section.')
else:
    try:
        if HAS_RL == 'factory':
            from fibernet.rl import create_rl_environment
            env = create_rl_environment(
                unit=UNIT, grid=GRID, n_pts_per_side=N_PTS_PER_SIDE,
                stiffness=STIFFNESS, num_steps=5000,
                target_stretch=TARGET_STRETCH, reward_mode='minimize_force', box=BOX)
        else:
            env = ParametricStructureEnv(
                unit=UNIT, box=BOX, grid=GRID, n_pts_per_side=N_PTS_PER_SIDE,
                stiffness=STIFFNESS, damping=DAMPING, num_steps=5000,
                target_stretch=TARGET_STRETCH, reward_mode='minimize_force')
        print(f'✓ RL Environment: n_actions={env.n_actions}, reward_mode={env.reward_mode}')
    except Exception as e:
        print(f'⚠ RL Environment creation failed: {e}')
        HAS_RL = False
        print('Skipping RL section.')""")

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
    "\n", "| # | Visualization | dark+light |\n",
    "|---|---------------|------------|\n",
    "| 01 | 12 base unit types (undeformed) | ✓ |\n",
    "| 02 | Perturbation comparison 0%–80% | ✓ |\n",
    "| 03 | 12 units + deformations | ✓ |\n",
    "| 04 | Batch gallery | ✓ |\n",
    "| 05 | Deformation trajectory (8 frames) | ✓ |\n",
    "| 06 | Stress distribution | ✓ |\n",
    "| 07 | Batch statistics | ✓ |\n",
    "| 08 | Feature statistics | ✓ |\n",
    "| 09 | ML analysis | ✓ |\n",
    "| 10 | Force-feature correlation | ✓ |\n",
    "| 11 | RL reward curves | ✓ (requires RL) |\n",
    "\n",
    "**Production**: `N_STRUCTURES=2000, PERTURBATION=0.40`"
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
