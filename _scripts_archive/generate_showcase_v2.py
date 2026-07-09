"""
FiberNet Showcase v2 - Using unified generators
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

OUTPUT_DIR = Path('/home/codex/projects/codex_test/fibernet/output_viz/showcase_v2')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BG = '#0a0a0a'
FG = '#00ff88'


def safe_gen(name, **kwargs):
    try:
        return fn.create(name, **kwargs)
    except Exception as e:
        print(f"  WARN: {e}")
        return None


def render_2d(ax, net, title='', lw=0.8):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor(BG)
    
    if net is None or net.num_fibers == 0:
        ax.text(0.5, 0.5, title, ha='center', va='center', color='white', fontsize=9)
        return
    
    # Get fiber centerlines
    lines = []
    for f in net.fibers:
        pts = f.centerline[:, :2]
        if len(pts) >= 2:
            lines.append(pts)
    
    if not lines:
        ax.text(0.5, 0.5, title, ha='center', va='center', color='white', fontsize=9)
        return
    
    all_pts = np.vstack(lines)
    xmin, ymin = all_pts.min(axis=0)
    xmax, ymax = all_pts.max(axis=0)
    scale = max(xmax - xmin, ymax - ymin) or 1.0
    pad = 0.05
    
    norm_lines = [(pts - np.array([xmin, ymin])) / scale * (1 - 2*pad) + pad for pts in lines]
    
    lc = LineCollection(norm_lines, linewidths=lw, colors=FG, alpha=0.85, antialiased=True)
    ax.add_collection(lc)
    
    if title:
        ax.text(0.5, 1.02, title, ha='center', va='bottom', color='white', fontsize=9,
               transform=ax.transAxes, fontweight='bold')


def render_3d(ax, net, title='', lw=0.3):
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.axis('off')
    
    if net is None or net.num_fibers == 0:
        ax.text2D(0.5, 0.5, title, ha='center', va='center', color='white', fontsize=9)
        return
    
    lines = [f.centerline for f in net.fibers if len(f.centerline) >= 2]
    if not lines:
        ax.text2D(0.5, 0.5, title, ha='center', va='center', color='white', fontsize=9)
        return
    
    all_pts = np.vstack(lines)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    scale = max(maxs - mins) or 1.0
    pad = 0.05
    
    norm_lines = [(pts - mins) / scale * (1 - 2*pad) + pad for pts in lines]
    
    lc = Line3DCollection(norm_lines, linewidths=lw, colors=FG, alpha=0.6)
    ax.add_collection3d(lc)
    
    if title:
        ax.text2D(0.5, 1.02, title, ha='center', va='bottom', color='white', fontsize=9,
                 transform=ax.transAxes, fontweight='bold')


def gen_category(name, nets, titles, is_3d=False, save_name=None):
    print(f"  {name}...")
    valid = [(n, t) for n, t in zip(nets, titles) if n is not None]
    if not valid:
        print(f"  SKIP: no valid")
        return
    
    nets_v, titles_v = zip(*valid)
    n = len(nets_v)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    
    figsize = (4 * ncols, 4 * nrows)
    
    if is_3d:
        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor(BG)
        for i, (net, title) in enumerate(zip(nets_v, titles_v)):
            ax = fig.add_subplot(nrows, ncols, i+1, projection='3d')
            render_3d(ax, net, title=title)
    else:
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows == 1:
            axes = [axes]
        axes_flat = [ax for row in axes for ax in row]
        fig.patch.set_facecolor(BG)
        for i, (net, title) in enumerate(zip(nets_v, titles_v)):
            render_2d(axes_flat[i], net, title=title)
        for i in range(n, len(axes_flat)):
            axes_flat[i].axis('off')
            axes_flat[i].set_facecolor(BG)
    
    plt.tight_layout(pad=0.5)
    save_path = OUTPUT_DIR / (save_name or f"{name.lower().replace(' ', '_')}.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG, pad_inches=0.1)
    plt.close(fig)
    print(f"    ✓ {save_path.name}")
    
    for net in nets_v:
        del net
    gc.collect()


# ============================================================================
print("\n=== 1. Lattice 2D (unified) ===")
nets = [
    safe_gen('lattice_2d', topology='square', cell_size=8.0, grid_size=(6,6)),
    safe_gen('lattice_2d', topology='honeycomb', cell_size=8.0, grid_size=(6,6)),
    safe_gen('lattice_2d', topology='triangular', cell_size=8.0, grid_size=(6,6)),
    safe_gen('lattice_2d', topology='kagome', cell_size=8.0, grid_size=(6,6)),
    safe_gen('lattice_2d', topology='honeycomb', cell_size=8.0, grid_size=(6,6), perturbation=0.2, seed=42),
]
gen_category("Lattice 2D", nets,
    ["Square", "Honeycomb", "Triangular", "Kagome", "Perturbed"],
    save_name="01_lattice_2d.png")

print("\n=== 2. Metamaterial 2D (unified) ===")
nets = [
    safe_gen('metamaterial_2d', mode='reentrant', angle=135, cell_size=8),
    safe_gen('metamaterial_2d', mode='reentrant', angle=120, cell_size=8),
    safe_gen('metamaterial_2d', mode='star', cell_size=8),
    safe_gen('metamaterial_2d', mode='chiral', cell_size=8),
    safe_gen('metamaterial_2d', mode='arrowhead', cell_size=8),
]
gen_category("Metamaterial 2D", nets,
    ["Reentrant 135°", "Reentrant 120°", "Star", "Chiral", "Arrowhead"],
    save_name="02_metamaterial_2d.png")

print("\n=== 3. Curved Random 2D ===")
nets = [
    safe_gen('curved_random_2d', num_fibers=80, curvature_type='sinusoidal', seed=42),
    safe_gen('curved_random_2d', num_fibers=80, curvature_type='bezier', seed=42),
    safe_gen('curved_random_2d', num_fibers=80, curvature_type='arc', seed=42),
    safe_gen('curved_random_2d', num_fibers=80, curvature_type='random_walk', seed=42),
    safe_gen('curved_random_2d', num_fibers=80, curvature_type='sinusoidal', angle_std=0.3, seed=42),
]
gen_category("Curved Random 2D", nets,
    ["Sinusoidal", "Bezier", "Arc", "Random Walk", "Aligned"],
    save_name="03_curved_random_2d.png")

print("\n=== 4. Lattice 3D (unified) ===")
nets = [
    safe_gen('lattice_3d', topology='cubic', cell_size=5, grid_size=(3,3,3)),
    safe_gen('lattice_3d', topology='octet', cell_size=5, grid_size=(2,2,2)),
    safe_gen('lattice_3d', topology='diamond', cell_size=5, grid_size=(2,2,2)),
    safe_gen('lattice_3d', topology='gyroid', cell_size=5, grid_size=(2,2,2)),
    safe_gen('lattice_3d', topology='plate', cell_size=5, grid_size=(2,2,2)),
]
gen_category("Lattice 3D", nets,
    ["Cubic", "Octet", "Diamond", "Gyroid", "Plate"],
    is_3d=True, save_name="04_lattice_3d.png")

print("\n=== 5. Random Networks ===")
nets = [
    safe_gen('random_2d', num_fibers=50, seed=42),
    safe_gen('random_2d', num_fibers=150, seed=42),
    safe_gen('random_2d', num_fibers=300, seed=42),
    safe_gen('random_3d', num_fibers=60, seed=42),
    safe_gen('random_walk', num_fibers=40, num_steps=25, seed=42),
]
gen_category("Random Networks", nets,
    ["2D N=50", "2D N=150", "2D N=300", "3D", "Walk"],
    is_3d=False, save_name="05_random.png")

print("\n=== 6. Entangled 3D ===")
nets = [
    safe_gen('entangled_3d', num_fibers=40, seed=42),
    safe_gen('entangled_3d', num_fibers=60, seed=42),
    safe_gen('entangled_3d', num_fibers=80, seed=42),
    safe_gen('entangled_3d', num_fibers=60, curvature=0.5, seed=42),
    safe_gen('entangled_3d', num_fibers=60, curvature=0.8, seed=42),
]
gen_category("Entangled 3D", nets,
    ["N=40", "N=60", "N=80", "curv=0.5", "curv=0.8"],
    is_3d=True, save_name="06_entangled_3d.png")

print("\n=== 7. Biomimetic ===")
nets = [
    safe_gen('biomimetic_network', network_type='collagen', num_fibers=60, seed=42),
    safe_gen('biomimetic_network', network_type='fibrin', num_fibers=60, seed=42),
    safe_gen('biomimetic_network', network_type='collagen', num_fibers=100, seed=42),
    safe_gen('electrospun'),
    safe_gen('meltblown'),
]
gen_category("Biomimetic", nets,
    ["Collagen", "Fibrin", "Dense Collagen", "Electrospun", "Meltblown"],
    is_3d=True, save_name="07_biomimetic.png")

print("\n=== 8. Fractal ===")
nets = [
    safe_gen('sierpinski'),
    safe_gen('koch_curve'),
    safe_gen('fractal_tree'),
    safe_gen('hilbert'),
    safe_gen('fractal_network'),
]
gen_category("Fractal", nets,
    ["Sierpinski", "Koch", "Tree", "Hilbert", "Network"],
    save_name="08_fractal.png")

print("\n=== 9. Hierarchical ===")
nets = [
    safe_gen('hierarchical_lattice', levels=1, base_topology='triangular', cell_size=30),
    safe_gen('hierarchical_lattice', levels=2, base_topology='triangular', cell_size=30),
    safe_gen('hierarchical_lattice', levels=2, base_topology='honeycomb', cell_size=30),
    safe_gen('hierarchical_lattice', levels=2, base_topology='square', cell_size=30),
    safe_gen('hierarchical_lattice', levels=3, base_topology='triangular', cell_size=30),
]
gen_category("Hierarchical", nets,
    ["Tri L1", "Tri L2", "Honey L2", "Square L2", "Tri L3"],
    save_name="09_hierarchical.png")

print("\n=== 10. TPMS ===")
nets = [
    safe_gen('tpms_sheet', resolution=10),
    safe_gen('tpms_sheet', resolution=15),
    safe_gen('tpms_lattice', resolution=10),
    safe_gen('tpms_lattice', resolution=15),
    safe_gen('tpms_gradient', resolution=8),
]
gen_category("TPMS", nets,
    ["Sheet r=10", "Sheet r=15", "Lattice r=10", "Lattice r=15", "Gradient"],
    is_3d=True, save_name="10_tpms.png")

print("\n=== 11. Voronoi ===")
nets = [
    safe_gen('voronoi_2d'),
    safe_gen('voronoi_2d', num_seeds=30),
    safe_gen('voronoi_2d', num_seeds=80),
    safe_gen('voronoi_3d'),
    safe_gen('foam_like_3d'),
]
gen_category("Voronoi", nets,
    ["2D Default", "2D Sparse", "2D Dense", "3D", "Foam 3D"],
    is_3d=False, save_name="11_voronoi.png")

print("\n=== 12. Field Guided ===")
try:
    net = safe_gen('field_guided')
    nets = [net]
    titles = ["Field Guided"]
    gen_category("Field Guided", nets, titles, save_name="12_field_guided.png")
except Exception as e:
    print(f"  SKIP: {e}")

print("\n=== 13. Parametric: Lattice 2D (cell_size) ===")
nets = [
    safe_gen('lattice_2d', topology='honeycomb', cell_size=3.0, grid_size=(10,10)),
    safe_gen('lattice_2d', topology='honeycomb', cell_size=5.0, grid_size=(8,8)),
    safe_gen('lattice_2d', topology='honeycomb', cell_size=8.0, grid_size=(6,6)),
    safe_gen('lattice_2d', topology='honeycomb', cell_size=12.0, grid_size=(4,4)),
    safe_gen('lattice_2d', topology='honeycomb', cell_size=20.0, grid_size=(3,3)),
]
gen_category("Parametric: Honeycomb", nets,
    ["cell=3", "cell=5", "cell=8", "cell=12", "cell=20"],
    save_name="13_parametric_honeycomb.png")

print("\n=== 14. Parametric: Metamaterial (angle) ===")
nets = [
    safe_gen('metamaterial_2d', mode='reentrant', angle=120, cell_size=8),
    safe_gen('metamaterial_2d', mode='reentrant', angle=135, cell_size=8),
    safe_gen('metamaterial_2d', mode='reentrant', angle=150, cell_size=8),
    safe_gen('metamaterial_2d', mode='reentrant', angle=160, cell_size=8),
    safe_gen('metamaterial_2d', mode='reentrant', angle=170, cell_size=8),
]
gen_category("Parametric: Reentrant Angle", nets,
    ["120°", "135°", "150°", "160°", "170°"],
    save_name="14_parametric_reentrant.png")

print("\n=== 15. Parametric: Curved (amplitude) ===")
nets = [
    safe_gen('curved_random_2d', num_fibers=80, curvature_type='sinusoidal', curvature_amplitude=0.5, seed=42),
    safe_gen('curved_random_2d', num_fibers=80, curvature_type='sinusoidal', curvature_amplitude=2.0, seed=42),
    safe_gen('curved_random_2d', num_fibers=80, curvature_type='sinusoidal', curvature_amplitude=5.0, seed=42),
    safe_gen('curved_random_2d', num_fibers=80, curvature_type='sinusoidal', curvature_amplitude=10.0, seed=42),
    safe_gen('curved_random_2d', num_fibers=80, curvature_type='sinusoidal', curvature_amplitude=20.0, seed=42),
]
gen_category("Parametric: Curvature", nets,
    ["amp=0.5", "amp=2", "amp=5", "amp=10", "amp=20"],
    save_name="15_parametric_curvature.png")

print("\n" + "="*60)
print("COMPLETE")
print("="*60)
for f in sorted(OUTPUT_DIR.glob("*.png")):
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name} ({size_kb:.0f} KB)")
