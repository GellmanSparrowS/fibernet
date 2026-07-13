#!/usr/bin/env python3
"""
FiberNet Sim Viz v2 — TaichiFEMSolver (static) + TaichiEngine (dynamics)

FEM: triangulated structures (kagome, triangle, square) — clean static solve
Spring: complex networks (voronoi, honeycomb) — dynamics with dashpot+drag
"""
import sys, json, time, argparse
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

def load_cp():
    return json.loads(CHECKPOINT.read_text()) if CHECKPOINT.exists() else {}
def save_cp(s):
    CHECKPOINT.write_text(json.dumps(s, indent=2))
def save_fig(fig, name, state):
    p = OUTPUT_DIR / f"{name}.png"
    fig.savefig(p, dpi=DPI, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    state[name] = str(p)
    save_cp(state)
    print(f"  ✓ {name}.png ({p.stat().st_size/1024:.0f} KB)")

def render_before_after(graph, deformed_pos, title="", suptitle=""):
    from fibernet.viz.render import _get_theme
    import copy
    t = _get_theme("dark")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor(t["bg"])
    render_graph(graph, ax=ax1, theme="dark", color_by="uniform", line_width=1.2, show_nodes=False)
    ax1.set_title("Original", color=t["text"], fontsize=11)
    dg = copy.deepcopy(graph)
    for i, nid in enumerate(list(dg.nodes.keys())):
        dg.nodes[nid].position = deformed_pos[i]
    render_graph(dg, ax=ax2, theme="dark", color_by="uniform", line_width=1.2, show_nodes=False)
    ax2.set_title(title, color=t["text"], fontsize=11)
    if suptitle:
        fig.suptitle(suptitle, color='white', fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def gen_01_fem_kagome_tension(state, skip):
    """FEM: kagome (triangulated) — static tension"""
    name = "01_fem_kagome_tension"
    if skip and name in state: return
    print("01. FEM kagome tension...")
    g = pattern_2d(unit="kagome", box=(10, 10), grid=(5, 5), seed=42)
    solver = TaichiFEMSolver()
    r = solver.uniaxial_tension(g, strain=0.05, radius=0.05)
    r.save(str(OUTPUT_DIR / "fem_kagome.json"))
    fig = render_before_after(g, r.deformed_positions,
        title=f"E*={r.effective_youngs_modulus:.2e} Pa",
        suptitle=f"Static FEM: Kagome Uniaxial Tension (ε=0.05)")
    save_fig(fig, name, state)


def gen_02_fem_triangle_biaxial(state, skip):
    """FEM: triangle — biaxial"""
    name = "02_fem_triangle_biaxial"
    if skip and name in state: return
    print("02. FEM triangle biaxial...")
    g = pattern_2d(unit="triangle", box=(10, 10), grid=(5, 5), seed=42)
    solver = TaichiFEMSolver()
    r = solver.biaxial_tension(g, strain_x=0.03, strain_y=0.03, radius=0.05)
    r.save(str(OUTPUT_DIR / "fem_triangle_biaxial.json"))
    fig = render_before_after(g, r.deformed_positions,
        title=f"energy={r.energy:.2f}",
        suptitle="Static FEM: Triangle Biaxial Tension (εx=εy=0.03)")
    save_fig(fig, name, state)


def gen_03_fem_square_shear(state, skip):
    """FEM: square — shear"""
    name = "03_fem_square_shear"
    if skip and name in state: return
    print("03. FEM square shear...")
    g = pattern_2d(unit="square", box=(10, 10), grid=(5, 5), seed=42)
    solver = TaichiFEMSolver()
    r = solver.shear_test(g, strain=0.05, radius=0.05)
    r.save(str(OUTPUT_DIR / "fem_square_shear.json"))
    fig = render_before_after(g, r.deformed_positions,
        title=f"G*={r.effective_shear_modulus:.2e} Pa",
        suptitle="Static FEM: Square Shear Test (γ=0.05)")
    save_fig(fig, name, state)


def gen_04_spring_honeycomb_2x(state, skip):
    """Spring: honeycomb 2x stretch with trajectory"""
    name = "04_spring_honeycomb_2x"
    if skip and name in state: return
    print("04. Spring honeycomb 2x stretch...")
    g = pattern_2d(unit="honeycomb", box=(10, 10), grid=(4, 4), seed=42)
    engine = TaichiEngine()
    r = engine.stretch_test(g, target_stretch=2.0, stiffness=1e5, damping=0.3,
                            num_steps=5000, save_interval=1000)
    r.save(str(OUTPUT_DIR / "spring_honeycomb_2x.json"))

    # Plot trajectory (up to 6 snapshots)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor('#0a0a0f')
    pos_orig, elements, _, _ = _graph_to_arrays(g)
    traj = r.positions_trajectory
    for idx, ax in enumerate(axes.flat):
        if idx < len(traj):
            pos = traj[idx]
            for e in elements:
                ax.plot([pos[e[0],0], pos[e[1],0]], [pos[e[0],1], pos[e[1],1]],
                        color='#b388ff', linewidth=0.8)
            ax.set_facecolor('#0a0a0f')
            ax.set_aspect('equal')
            step = idx * 1000
            ax.set_title(f"Step {step}", color='#d0d0d0', fontsize=9)
            xlim = (pos_orig[:,0].min()-5, pos_orig[:,0].max()+pos_orig[:,0].max()*1.2)
            ax.set_xlim(*xlim)
            ax.set_ylim(pos_orig[:,1].min()-2, pos_orig[:,1].max()+2)
            ax.tick_params(colors='#333', labelsize=6)
        else:
            ax.set_visible(False)
    fig.suptitle("Mass-Spring: Honeycomb 2x Stretch", color='white', fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, name, state)


def gen_05_spring_voronoi_15x(state, skip):
    """Spring: voronoi 1.5x stretch"""
    name = "05_spring_voronoi_15x"
    if skip and name in state: return
    print("05. Spring voronoi 1.5x stretch...")
    g = pattern_2d(unit="voronoi", box=(10, 10), grid=(3, 3), seed=42, n_internal=10)
    engine = TaichiEngine()
    r = engine.stretch_test(g, target_stretch=1.5, stiffness=1e5, damping=0.3,
                            num_steps=3000, save_interval=600)
    r.save(str(OUTPUT_DIR / "spring_voronoi_15x.json"))
    pos_orig, elements, _, _ = _graph_to_arrays(g)
    pos_final = r.deformed_positions

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor('#0a0a0f')
    for e in elements:
        ax1.plot([pos_orig[e[0],0], pos_orig[e[1],0]], [pos_orig[e[0],1], pos_orig[e[1],1]],
                 color='#b388ff', linewidth=0.5)
    ax1.set_facecolor('#0a0a0f'); ax1.set_aspect('equal')
    ax1.set_title("Original", color='white', fontsize=11)
    for e in elements:
        ax2.plot([pos_final[e[0],0], pos_final[e[1],0]], [pos_final[e[0],1], pos_final[e[1],1]],
                 color='#b388ff', linewidth=0.5)
    ax2.set_facecolor('#0a0a0f'); ax2.set_aspect('equal')
    ax2.set_title("Stretched 1.5x", color='white', fontsize=11)
    fig.suptitle(f"Mass-Spring: Voronoi 1.5x Stretch ({g.num_nodes} nodes, {g.num_edges} edges)",
                 color='white', fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, name, state)


def gen_06_spring_honeycomb_3x(state, skip):
    """Spring: honeycomb 3x stretch (extreme)"""
    name = "06_spring_honeycomb_3x"
    if skip and name in state: return
    print("06. Spring honeycomb 3x stretch...")
    g = pattern_2d(unit="honeycomb", box=(10, 10), grid=(3, 3), seed=42)
    engine = TaichiEngine()
    r = engine.stretch_test(g, target_stretch=3.0, stiffness=1e5, damping=0.3,
                            num_steps=5000, save_interval=1000)
    r.save(str(OUTPUT_DIR / "spring_honeycomb_3x.json"))
    pos_orig, elements, _, _ = _graph_to_arrays(g)
    pos_final = r.deformed_positions
    l0, _ = _element_data(pos_orig, elements)
    lf = np.array([np.linalg.norm(pos_final[elements[e,1]] - pos_final[elements[e,0]]) for e in range(len(elements))])
    # Color edges by stretch ratio
    stretch = lf / l0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor('#0a0a0f')
    cmap = plt.get_cmap('coolwarm')
    vmin, vmax = 0.8, 3.5
    for e_idx, e in enumerate(elements):
        c = cmap(np.clip((stretch[e_idx]-vmin)/(vmax-vmin), 0, 1))
        ax2.plot([pos_final[e[0],0], pos_final[e[1],0]], [pos_final[e[0],1], pos_final[e[1],1]],
                 color=c, linewidth=0.8)
    for e in elements:
        ax1.plot([pos_orig[e[0],0], pos_orig[e[1],0]], [pos_orig[e[0],1], pos_orig[e[1],1]],
                 color='#b388ff', linewidth=0.8)
    ax1.set_facecolor('#0a0a0f'); ax1.set_aspect('equal')
    ax1.set_title("Original", color='white', fontsize=11)
    ax2.set_facecolor('#0a0a0f'); ax2.set_aspect('equal')
    ax2.set_title(f"3x Stretch (edge color = stretch ratio)", color='white', fontsize=11)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax2, fraction=0.03, pad=0.04)
    cbar.set_label('Stretch Ratio', color='#d0d0d0')
    cbar.ax.tick_params(colors='#d0d0d0')
    fig.suptitle(f"Honeycomb 3x Stretch — max edge: {np.max(stretch):.2f}x",
                 color='white', fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, name, state)


SCENARIOS = [
    gen_01_fem_kagome_tension,
    gen_02_fem_triangle_biaxial,
    gen_03_fem_square_shear,
    gen_04_spring_honeycomb_2x,
    gen_05_spring_voronoi_15x,
    gen_06_spring_honeycomb_3x,
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only", type=int, default=None)
    args = parser.parse_args()
    print("=" * 60)
    print("Sim Viz v2 — FEM (static) + Spring (dynamics)")
    print("=" * 60)
    state = load_cp() if args.resume else {}
    t0 = time.time()
    for fn in SCENARIOS:
        if args.only is not None:
            num = int(fn.__name__.split("_")[1])
            if num != args.only: continue
        try:
            fn(state, args.resume)
        except Exception as exc:
            print(f"  ✗ {fn.__name__}: {exc}")
            import traceback; traceback.print_exc()
    print(f"\nDone — {len(state)}/{len(SCENARIOS)} in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
