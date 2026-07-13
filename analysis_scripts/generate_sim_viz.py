#!/usr/bin/env python3
"""
FiberNet Simulation Visualization — TaichiFEMSolver + TaichiEngine

Generates visualizations for:
1. TaichiFEMSolver: uniaxial tension, biaxial, compression, shear (complex networks)
2. TaichiEngine: 2x stretch with trajectory

Usage:
    python3 analysis_scripts/generate_sim_viz.py
    python3 analysis_scripts/generate_sim_viz.py --resume
"""
import sys, os, json, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fibernet import pattern_2d, render_graph
from fibernet.sim.accelerated import TaichiFEMSolver, TaichiEngine, _graph_to_arrays, _element_data

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output_viz" / "sim"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT = OUTPUT_DIR / "_checkpoint.json"
DPI = 150


def load_checkpoint():
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text())
    return {}

def save_checkpoint(state):
    CHECKPOINT.write_text(json.dumps(state, indent=2))

def save_fig(fig, name, state):
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    state[name] = {"path": str(path), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    save_checkpoint(state)
    print(f"  ✓ {name}.png ({path.stat().st_size/1024:.0f} KB)")


def render_sim(graph, deformed_pos, title="", ax=None, theme="dark"):
    """Render original vs deformed side-by-side."""
    from fibernet.viz.render import THEMES, _get_theme
    t = _get_theme(theme)
    
    if ax is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.patch.set_facecolor(t["bg"])
    else:
        ax1, ax2 = ax
        fig = ax1.figure
    
    render_graph(graph, ax=ax1, theme=theme, color_by="uniform",
                 line_width=1.2, show_nodes=False, title="Original")
    
    # Render deformed using a temporary graph with displaced positions
    from fibernet.core.structure_graph import StructureGraph
    import copy
    deformed = copy.deepcopy(graph)
    node_ids = list(deformed.nodes.keys())
    for i, nid in enumerate(node_ids):
        deformed.nodes[nid].position = deformed_pos[i]
    
    render_graph(deformed, ax=ax2, theme=theme, color_by="uniform",
                 line_width=1.2, show_nodes=False, title="Deformed")
    
    if title:
        fig.suptitle(title, color=t["text"], fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def gen_01_fem_tension(state, skip):
    """01: TaichiFEM — Uniaxial tension (voronoi)"""
    name = "01_fem_tension"
    if skip and name in state: return
    print("01. TaichiFEM uniaxial tension (voronoi)...")
    
    g = pattern_2d(unit="voronoi", box=(10, 10), grid=(4, 4), seed=42, n_internal=20)
    solver = TaichiFEMSolver()
    r = solver.uniaxial_tension(g, strain=0.05, radius=0.05)
    r.save(str(OUTPUT_DIR / "fem_tension.json"))
    
    fig = render_sim(g, r.deformed_positions,
                     title=f"Uniaxial Tension (ε=0.05) — E*={r.effective_youngs_modulus:.2e} Pa")
    save_fig(fig, name, state)


def gen_02_fem_biaxial(state, skip):
    """02: TaichiFEM — Biaxial tension"""
    name = "02_fem_biaxial"
    if skip and name in state: return
    print("02. TaichiFEM biaxial tension...")
    
    g = pattern_2d(unit="honeycomb", box=(10, 10), grid=(5, 5), seed=42)
    solver = TaichiFEMSolver()
    r = solver.biaxial_tension(g, strain_x=0.03, strain_y=0.03, radius=0.05)
    r.save(str(OUTPUT_DIR / "fem_biaxial.json"))
    
    fig = render_sim(g, r.deformed_positions,
                     title=f"Biaxial Tension (εx=εy=0.03) — energy={r.energy:.2f}")
    save_fig(fig, name, state)


def gen_03_fem_compression(state, skip):
    """03: TaichiFEM — Compression"""
    name = "03_fem_compression"
    if skip and name in state: return
    print("03. TaichiFEM compression...")
    
    g = pattern_2d(unit="kagome", box=(10, 10), grid=(4, 4), seed=42)
    solver = TaichiFEMSolver()
    r = solver.compression(g, strain=0.03, radius=0.05)
    r.save(str(OUTPUT_DIR / "fem_compression.json"))
    
    fig = render_sim(g, r.deformed_positions,
                     title=f"Compression (ε=-0.03) — E*={abs(r.effective_youngs_modulus):.2e} Pa")
    save_fig(fig, name, state)


def gen_04_fem_shear(state, skip):
    """04: TaichiFEM — Shear"""
    name = "04_fem_shear"
    if skip and name in state: return
    print("04. TaichiFEM shear...")
    
    g = pattern_2d(unit="honeycomb", box=(10, 10), grid=(5, 5), seed=42)
    solver = TaichiFEMSolver()
    r = solver.shear_test(g, strain=0.05, radius=0.05)
    r.save(str(OUTPUT_DIR / "fem_shear.json"))
    
    fig = render_sim(g, r.deformed_positions,
                     title=f"Shear (γ=0.05) — G*={r.effective_shear_modulus:.2e} Pa")
    save_fig(fig, name, state)


def gen_05_spring_stretch(state, skip):
    """05: TaichiEngine — Mass-spring 2x stretch"""
    name = "05_spring_stretch"
    if skip and name in state: return
    print("05. TaichiEngine 2x stretch (honeycomb)...")
    
    g = pattern_2d(unit="honeycomb", box=(10, 10), grid=(4, 4), seed=42)
    engine = TaichiEngine()
    r = engine.stretch_test(g, target_stretch=2.0, stiffness=1e5, damping=0.3,
                            num_steps=5000, save_interval=1000)
    r.save(str(OUTPUT_DIR / "spring_stretch.json"))
    
    # Plot trajectory
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor('#0a0a0f')
    
    pos_orig, elements, _, _ = _graph_to_arrays(g)
    trajectory = r.positions_trajectory
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(trajectory):
            pos = trajectory[idx]
            for e in elements:
                ax.plot([pos[e[0], 0], pos[e[1], 0]], [pos[e[0], 1], pos[e[1], 1]],
                        color='#b388ff', linewidth=0.8)
            ax.set_facecolor('#0a0a0f')
            ax.set_aspect('equal')
            step_num = idx * 1000
            ax.set_title(f"Step {step_num}", color='#d0d0d0', fontsize=9)
            ax.set_xlim(pos_orig[:, 0].min()-2, pos_orig[:, 0].max()+pos_orig[:, 0].max()*1.1)
            ax.set_ylim(pos_orig[:, 1].min()-2, pos_orig[:, 1].max()+2)
            ax.tick_params(colors='#d0d0d0', labelsize=6)
        else:
            ax.set_visible(False)
    
    fig.suptitle("Mass-Spring Dynamics: 2x Uniaxial Stretch", color='white', fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, name, state)


def gen_06_spring_stretch_voronoi(state, skip):
    """06: TaichiEngine — Voronoi 1.5x stretch"""
    name = "06_spring_stretch_voronoi"
    if skip and name in state: return
    print("06. TaichiEngine 1.5x stretch (voronoi)...")
    
    g = pattern_2d(unit="voronoi", box=(10, 10), grid=(3, 3), seed=42, n_internal=10)
    engine = TaichiEngine()
    r = engine.stretch_test(g, target_stretch=1.5, stiffness=1e5, damping=0.3,
                            num_steps=3000, save_interval=600)
    r.save(str(OUTPUT_DIR / "spring_voronoi.json"))
    
    # Before/after
    pos_orig, elements, _, _ = _graph_to_arrays(g)
    pos_final = r.deformed_positions
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor('#0a0a0f')
    
    for e in elements:
        ax1.plot([pos_orig[e[0], 0], pos_orig[e[1], 0]], [pos_orig[e[0], 1], pos_orig[e[1], 1]],
                 color='#b388ff', linewidth=0.6)
    ax1.set_facecolor('#0a0a0f')
    ax1.set_aspect('equal')
    ax1.set_title("Original", color='white', fontsize=12)
    
    for e in elements:
        ax2.plot([pos_final[e[0], 0], pos_final[e[1], 0]], [pos_final[e[0], 1], pos_final[e[1], 1]],
                 color='#b388ff', linewidth=0.6)
    ax2.set_facecolor('#0a0a0f')
    ax2.set_aspect('equal')
    ax2.set_title("Stretched 1.5x", color='white', fontsize=12)
    
    fig.suptitle(f"Voronoi Mass-Spring Stretch — {g.num_nodes} nodes, {g.num_edges} edges",
                 color='white', fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, name, state)


SCENARIOS = [
    gen_01_fem_tension,
    gen_02_fem_biaxial,
    gen_03_fem_compression,
    gen_04_fem_shear,
    gen_05_spring_stretch,
    gen_06_spring_stretch_voronoi,
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only", type=int, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("FiberNet Sim Viz — TaichiFEMSolver + TaichiEngine")
    print("=" * 60)
    
    state = load_checkpoint() if args.resume else {}
    skip = args.resume
    t0 = time.time()
    
    for fn in SCENARIOS:
        if args.only is not None:
            num = int(fn.__name__.split("_")[1])
            if num != args.only: continue
        try:
            fn(state, skip)
        except Exception as exc:
            print(f"  ✗ {fn.__name__}: {exc}")
            import traceback; traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Done — {len(state)}/{len(SCENARIOS)} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
