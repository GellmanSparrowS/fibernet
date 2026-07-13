"""
FiberNet Showcase - Fast version using matplotlib for both 2D and 3D
"""

import sys
import os
import gc
import numpy as np
from pathlib import Path

sys.path.insert(0, '/home/codex/projects/codex_test/fibernet')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import fibernet as fn
from fibernet.viz.showcase import ShowcaseStyle

OUTPUT_DIR = Path('/home/codex/projects/codex_test/fibernet/output_viz/showcase')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_gen(func, **kwargs):
    try:
        return func(**kwargs)
    except Exception as e:
        print(f"  WARN: {e}")
        return None


def render_2d_fast(ax, net, title='', bg='#0a0a0a', color='#00ff88', lw=None):
    """Fast 2D rendering on axes."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor(bg)
    
    if net is None or net.num_fibers == 0:
        ax.text(0.5, 0.5, title, ha='center', va='center', color='white', fontsize=9)
        return
    
    lines = [f.centerline[:, :2] for f in net.fibers if len(f.centerline) >= 2]
    if not lines:
        return
    
    all_pts = np.vstack(lines)
    xmin, ymin = all_pts.min(axis=0)
    xmax, ymax = all_pts.max(axis=0)
    scale = max(xmax - xmin, ymax - ymin) or 1.0
    pad = 0.05
    
    norm_lines = [(pts - np.array([xmin, ymin])) / scale * (1 - 2*pad) + pad for pts in lines]
    
    if lw is None:
        lw = ShowcaseStyle.compute_line_width(net.num_fibers)
    
    lc = LineCollection(norm_lines, linewidths=lw, colors=color, alpha=0.9, antialiased=True)
    ax.add_collection(lc)
    
    if title:
        ax.text(0.5, 1.02, title, ha='center', va='bottom', color='white', fontsize=9,
               transform=ax.transAxes, fontweight='bold')


def render_3d_fast(ax, net, title='', bg='#0a0a0a', color='#00ff88', lw=0.3):
    """Fast 3D rendering on 3D axes."""
    ax.set_facecolor(bg)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.axis('off')
    
    if net is None or net.num_fibers == 0:
        ax.text2D(0.5, 0.5, title, ha='center', va='center', color='white', fontsize=9)
        return
    
    lines = [f.centerline for f in net.fibers if len(f.centerline) >= 2]
    if not lines:
        return
    
    all_pts = np.vstack(lines)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    scale = max(maxs - mins) or 1.0
    pad = 0.05
    
    norm_lines = [(pts - mins) / scale * (1 - 2*pad) + pad for pts in lines]
    
    lc = Line3DCollection(norm_lines, linewidths=lw, colors=color, alpha=0.7)
    ax.add_collection3d(lc)
    
    if title:
        ax.text2D(0.5, 1.02, title, ha='center', va='bottom', color='white', fontsize=9,
                 transform=ax.transAxes, fontweight='bold')


def generate_category(name, networks, titles, is_3d=False, save_name=None):
    """Generate a category visualization."""
    print(f"  Generating {name}...")
    
    valid = [(n, t) for n, t in zip(networks, titles) if n is not None]
    if not valid:
        print(f"  SKIP {name}: no valid networks")
        return
    
    nets, titles = zip(*valid)
    n = len(nets)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    
    bg = ShowcaseStyle.bg_color
    figsize = (4 * ncols, 4 * nrows)
    
    if is_3d:
        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor(bg)
        for i, (net, title) in enumerate(zip(nets, titles)):
            ax = fig.add_subplot(nrows, ncols, i+1, projection='3d')
            render_3d_fast(ax, net, title=title, bg=bg)
    else:
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows == 1:
            axes = [axes]
        axes_flat = [ax for row in axes for ax in row]
        fig.patch.set_facecolor(bg)
        for i, (net, title) in enumerate(zip(nets, titles)):
            render_2d_fast(axes_flat[i], net, title=title, bg=bg)
        for i in range(n, len(axes_flat)):
            axes_flat[i].axis('off')
            axes_flat[i].set_facecolor(bg)
    
    plt.tight_layout(pad=0.5)
    
    save_path = OUTPUT_DIR / (save_name or f"{name.lower().replace(' ', '_')}.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=bg, pad_inches=0.1)
    plt.close(fig)
    print(f"  ✓ {save_path.name}")
    
    for net in nets:
        del net
    gc.collect()


# ============================================================================
# Generate all categories
# ============================================================================

from fibernet.gen import disordered, ordered, advanced, bundles

print("\n=== Category 1: Random 2D ===")
nets = [
    safe_gen(disordered.random_straight_2d, num_fibers=50, seed=42),
    safe_gen(disordered.random_straight_2d, num_fibers=150, seed=42),
    safe_gen(disordered.random_straight_2d, num_fibers=300, seed=42),
    safe_gen(disordered.oriented_random_2d, num_fibers=150, angle_std=0.1, seed=42),
    safe_gen(disordered.oriented_random_2d, num_fibers=150, angle_std=0.5, seed=42),
]
generate_category("Random 2D", nets,
    ["Sparse (N=50)", "Medium (N=150)", "Dense (N=300)", "Aligned (σ=0.1)", "Oriented (σ=0.5)"],
    is_3d=False, save_name="01_random_2d.png")

print("\n=== Category 2: Random 3D ===")
nets = [
    safe_gen(disordered.random_straight_3d, num_fibers=40, seed=42),
    safe_gen(disordered.random_straight_3d, num_fibers=80, seed=42),
    safe_gen(disordered.oriented_random_3d, num_fibers=60, angle_std=0.1, seed=42),
    safe_gen(disordered.oriented_random_3d, num_fibers=60, angle_std=0.5, seed=42),
    safe_gen(disordered.random_walk_fibers, num_fibers=30, num_steps=20, seed=42),
]
generate_category("Random 3D", nets,
    ["Sparse", "Medium", "Aligned", "Oriented", "Walk"],
    is_3d=True, save_name="02_random_3d.png")

print("\n=== Category 3: Lattice 2D ===")
nets = [
    safe_gen(ordered.square_lattice_2d, cell_size=8.0, grid_size=(6, 6)),
    safe_gen(ordered.honeycomb_lattice_2d, cell_size=8.0, grid_size=(6, 6)),
    safe_gen(ordered.triangular_lattice_2d, cell_size=8.0, grid_size=(6, 6)),
    safe_gen(ordered.kagome_lattice_2d, cell_size=8.0, grid_size=(6, 6)),
    safe_gen(ordered.honeycomb_lattice_2d, cell_size=8.0, grid_size=(6, 6), perturbation=0.2),
]
generate_category("Lattice 2D", nets,
    ["Square", "Honeycomb", "Triangular", "Kagome", "Perturbed"],
    is_3d=False, save_name="03_lattice_2d.png")

print("\n=== Category 4: Lattice 3D ===")
nets = [
    safe_gen(ordered.cubic_lattice_3d, cell_size=5.0, grid_size=(3, 3, 3)),
    safe_gen(ordered.octet_truss_3d, cell_size=5.0, grid_size=(2, 2, 2)),
    safe_gen(lambda: fn.create("diamond_lattice_3d")),
    safe_gen(lambda: fn.create("gyroid_lattice_3d")),
    safe_gen(lambda: fn.create("plate_lattice_3d")),
]
generate_category("Lattice 3D", nets,
    ["Cubic", "Octet", "Diamond", "Gyroid", "Plate"],
    is_3d=True, save_name="04_lattice_3d.png")

print("\n=== Category 5: Metamaterial 2D ===")
nets = [
    safe_gen(lambda: fn.create("reentrant_honeycomb_2d")),
    safe_gen(lambda: fn.create("star_honeycomb_2d")),
    safe_gen(lambda: fn.create("arrowhead_auxetic_2d")),
    safe_gen(lambda: fn.create("chiral_honeycomb_2d")),
    safe_gen(lambda: fn.create("missing_rib_auxetic_2d")),
]
generate_category("Metamaterial 2D", nets,
    ["Reentrant", "Star", "Arrowhead", "Chiral", "Missing Rib"],
    is_3d=False, save_name="05_metamaterial_2d.png")

print("\n=== Category 6: Metamaterial 3D ===")
nets = [
    safe_gen(lambda: fn.create("reentrant_honeycomb_3d")),
    safe_gen(lambda: fn.create("proper_octet_truss_3d")),
    safe_gen(lambda: fn.create("diamond_lattice_3d")),
    safe_gen(lambda: fn.create("gyroid_lattice_3d")),
    safe_gen(lambda: fn.create("plate_lattice_3d")),
]
generate_category("Metamaterial 3D", nets,
    ["Reentrant", "Octet", "Diamond", "Gyroid", "Plate"],
    is_3d=True, save_name="06_metamaterial_3d.png")

print("\n=== Category 7: Fractal ===")
nets = [
    safe_gen(lambda: fn.create("sierpinski")),
    safe_gen(lambda: fn.create("koch_curve")),
    safe_gen(lambda: fn.create("fractal_tree")),
    safe_gen(lambda: fn.create("hilbert")),
    safe_gen(lambda: fn.create("fractal_network")),
]
generate_category("Fractal", nets,
    ["Sierpinski", "Koch", "Tree", "Hilbert", "Network"],
    is_3d=False, save_name="07_fractal.png")

print("\n=== Category 8: Biomimetic ===")
nets = [
    safe_gen(advanced.biomimetic_collagen, num_fibers=80, seed=42),
    safe_gen(advanced.biomimetic_collagen, num_fibers=150, seed=42),
    safe_gen(lambda: fn.create("electrospun")),
    safe_gen(lambda: fn.create("meltblown")),
    safe_gen(lambda: fn.create("paper_network")),
]
generate_category("Biomimetic", nets,
    ["Collagen(sparse)", "Collagen(dense)", "Electrospun", "Meltblown", "Paper"],
    is_3d=False, save_name="08_biomimetic.png")

print("\n=== Category 9: Bundles ===")
nets = [
    safe_gen(lambda: fn.create("parallel_bundle_2d")),
    safe_gen(lambda: fn.create("twisted_bundle_2d")),
    safe_gen(lambda: fn.create("random_bundle_3d")),
    safe_gen(lambda: fn.create("tendon_like_bundle_3d")),
    safe_gen(lambda: fn.create("braided_bundle_3d")),
]
generate_category("Bundles", nets,
    ["Parallel", "Twisted", "Random", "Tendon", "Braided"],
    is_3d=False, save_name="09_bundles.png")

print("\n=== Category 10: Voronoi ===")
nets = [
    safe_gen(lambda: fn.create("voronoi_2d")),
    safe_gen(lambda: fn.create("voronoi_2d", num_seeds=30)),
    safe_gen(lambda: fn.create("voronoi_2d", num_seeds=80)),
    safe_gen(lambda: fn.create("foam_like_3d")),
    safe_gen(lambda: fn.create("voronoi_3d")),
]
generate_category("Voronoi", nets,
    ["Voronoi 2D", "Sparse(30)", "Dense(80)", "Foam 3D", "Voronoi 3D"],
    is_3d=False, save_name="10_voronoi.png")

print("\n=== Category 11: TPMS ===")
nets = [
    safe_gen(lambda: fn.create("tpms_sheet", resolution=8)),
    safe_gen(lambda: fn.create("tpms_sheet", resolution=12)),
    safe_gen(lambda: fn.create("tpms_lattice", resolution=8)),
    safe_gen(lambda: fn.create("tpms_lattice", resolution=12)),
    safe_gen(lambda: fn.create("tpms_gradient", resolution=6)),
]
generate_category("TPMS", nets,
    ["Sheet(r=8)", "Sheet(r=12)", "Lattice(r=8)", "Lattice(r=12)", "Gradient"],
    is_3d=True, save_name="11_tpms.png")

print("\n=== Category 12: Woven 3D ===")
nets = [
    safe_gen(lambda: fn.create("woven_3d")),
    safe_gen(lambda: fn.create("plain_weave")),
    safe_gen(lambda: fn.create("twill_weave")),
    safe_gen(lambda: fn.create("satin_weave")),
    safe_gen(lambda: fn.create("textile_weave")),
]
generate_category("Woven", nets,
    ["Woven 3D", "Plain", "Twill", "Satin", "Textile"],
    is_3d=False, save_name="12_woven.png")

print("\n=== Category 13: Parametric Random ===")
nets = [
    safe_gen(disordered.random_straight_2d, num_fibers=30, seed=42),
    safe_gen(disordered.random_straight_2d, num_fibers=80, seed=42),
    safe_gen(disordered.random_straight_2d, num_fibers=200, seed=42),
    safe_gen(disordered.random_straight_2d, num_fibers=500, seed=42),
    safe_gen(disordered.random_straight_2d, num_fibers=1000, seed=42),
]
generate_category("Parametric: Random (N)", nets,
    ["N=30", "N=80", "N=200", "N=500", "N=1000"],
    is_3d=False, save_name="13_parametric_random.png")

print("\n=== Category 14: Parametric Honeycomb ===")
nets = [
    safe_gen(ordered.honeycomb_lattice_2d, cell_size=3.0),
    safe_gen(ordered.honeycomb_lattice_2d, cell_size=5.0),
    safe_gen(ordered.honeycomb_lattice_2d, cell_size=8.0),
    safe_gen(ordered.honeycomb_lattice_2d, cell_size=12.0),
    safe_gen(ordered.honeycomb_lattice_2d, cell_size=20.0),
]
generate_category("Parametric: Honeycomb", nets,
    ["cell=3", "cell=5", "cell=8", "cell=12", "cell=20"],
    is_3d=False, save_name="14_parametric_honeycomb.png")

print("\n=== Category 15: Parametric Oriented ===")
nets = [
    safe_gen(disordered.oriented_random_2d, num_fibers=120, angle_std=0.01, seed=42),
    safe_gen(disordered.oriented_random_2d, num_fibers=120, angle_std=0.1, seed=42),
    safe_gen(disordered.oriented_random_2d, num_fibers=120, angle_std=0.3, seed=42),
    safe_gen(disordered.oriented_random_2d, num_fibers=120, angle_std=0.7, seed=42),
    safe_gen(disordered.oriented_random_2d, num_fibers=120, angle_std=1.57, seed=42),
]
generate_category("Parametric: Oriented (σ)", nets,
    ["σ=0.01", "σ=0.1", "σ=0.3", "σ=0.7", "σ=1.57"],
    is_3d=False, save_name="15_parametric_oriented.png")

print("\n" + "="*60)
print("COMPLETE")
print("="*60)
print(f"Output: {OUTPUT_DIR}")
for f in sorted(OUTPUT_DIR.glob("*.png")):
    print(f"  {f.name}")
