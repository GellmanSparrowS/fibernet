"""Core Generator Gallery — publication-quality overview of fibernet.gen modules."""
import sys, os, warnings
warnings.filterwarnings("ignore")
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

sys.path.insert(0, os.path.dirname(__file__))
import fibernet as fn
from fibernet import gen
from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork

OUT = os.path.join(os.path.dirname(__file__), "output_viz")
os.makedirs(OUT, exist_ok=True)

BG = "#0d1117"
FG = "#e6edf3"
ACCENT = "#58a6ff"
CAT_COLORS = {
    "Disordered": "#ff6b6b", "Ordered": "#51cf66", "Chiral": "#ffd43b",
    "Woven": "#cc5de8", "Hierarchical": "#ff922b", "Metamaterials": "#20c997",
    "Fractal": "#74c0fc", "Bundles": "#f783ac", "TPMS": "#a9e34b",
    "Advanced": "#e599f7", "Laminates": "#ffa94d",
}


def to_network(obj):
    """Ensure we have a FiberNetwork (wrap single Fiber if needed)."""
    if isinstance(obj, FiberNetwork):
        return obj
    if isinstance(obj, Fiber):
        net = FiberNetwork()
        net.add_fiber(obj)
        return net
    return None


def get_fiber_segments(net):
    """Extract line segments from a FiberNetwork."""
    segs = []
    for f in net.fibers:
        cl = np.asarray(f.centerline)
        if len(cl) >= 2:
            for i in range(len(cl) - 1):
                segs.append((cl[i], cl[i+1]))
    return segs


def draw_2d(net, ax, color="#58a6ff", lw=0.8, alpha=0.85):
    net = to_network(net)
    if net is None:
        return
    segs2d = []
    for s in get_fiber_segments(net):
        segs2d.append([(s[0][0], s[0][1]), (s[1][0], s[1][1])])
    if not segs2d:
        ax.text(0.5, 0.5, "(empty)", transform=ax.transAxes, ha="center", va="center", color=FG, fontsize=8)
        return
    lc = LineCollection(segs2d, colors=color, linewidths=lw, alpha=alpha)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.axis("off")


def draw_3d(net, ax, color="#00ccff", lw=0.6, alpha=0.7):
    net = to_network(net)
    if net is None:
        return
    segs3d = []
    all_pts = []
    for s in get_fiber_segments(net):
        if len(s[0]) >= 3 and len(s[1]) >= 3:
            segs3d.append((tuple(s[0][:3]), tuple(s[1][:3])))
            all_pts.extend([s[0][:3], s[1][:3]])
    if not segs3d:
        return
    lc = Line3DCollection(segs3d, colors=color, linewidths=lw, alpha=alpha)
    ax.add_collection3d(lc)
    arr = np.array(all_pts)
    for dim, setter in [(0, ax.set_xlim3d), (1, ax.set_ylim3d), (2, ax.set_zlim3d)]:
        mn, mx = arr[:, dim].min(), arr[:, dim].max()
        margin = (mx - mn) * 0.05 + 0.1
        setter(mn - margin, mx + margin)
    ax.axis("off")


def safe_gen(name, **kwargs):
    try:
        return getattr(gen, name)(**kwargs)
    except Exception as e:
        print(f"  [SKIP] {name}: {e}")
        return None


# ── FIGURE 1: 2D Generator Gallery ──────────────────────────────────
def fig1():
    cats = {
        "Disordered": [
            ("random_straight_2d",   dict(num_fibers=60, fiber_length=8, box_size=30.0, seed=42)),
            ("oriented_random_2d",   dict(num_fibers=60, preferred_angle=0.0, angular_spread=0.15, seed=42)),
            ("poisson_line_network_2d", dict(intensity=0.3, box_size=30.0, seed=42)),
            ("random_curved_fibers_3d", dict(num_fibers=40, seed=42)),
        ],
        "Ordered": [
            ("square_lattice_2d",    dict(spacing=5, grid_size=(4,4))),
            ("triangular_lattice_2d", dict(spacing=5, grid_size=(4,4))),
            ("honeycomb_lattice_2d", dict(cell_size=5, grid_size=(3,3))),
            ("kagome_lattice_2d",    dict(spacing=5, grid_size=(3,3))),
        ],
        "Chiral": [
            ("single_helix",         dict(helix_radius=3, pitch=2, num_turns=3)),
            ("double_helix",         dict(helix_radius=3, pitch=2, num_turns=3)),
            ("braided_rope",         dict(num_strands=3, rope_radius=2, num_turns=2)),
            ("twisted_bundle",       dict(num_fibers=7, twist_angle=np.pi/4, total_length=20)),
        ],
        "Woven": [
            ("plain_weave_2d",       dict(spacing=2, grid_size=(5,5), radius=0.05)),
            ("twill_weave_2d",       dict(spacing=2, grid_size=(5,5))),
            ("satin_weave_2d",       dict(spacing=2, grid_size=(5,5))),
        ],
    }
    max_cols = max(len(v) for v in cats.values())
    n_cats = len(cats)
    fig, axes = plt.subplots(n_cats, max_cols, figsize=(4*max_cols, 4*n_cats))
    fig.patch.set_facecolor(BG)

    for r, (cat, items) in enumerate(cats.items()):
        col = CAT_COLORS.get(cat, ACCENT)
        for c in range(max_cols):
            ax = axes[r, c]
            ax.set_facecolor(BG)
            ax.axis("off")
            if c < len(items):
                name, kw = items[c]
                net = safe_gen(name, **kw)
                if net is not None:
                    draw_2d(net, ax, color=col, lw=1.0)
                    ax.set_title(name.replace("_", " "), color=FG, fontsize=9, pad=4)
        axes[r, 0].text(-0.3, 0.5, cat, transform=axes[r,0].transAxes,
                        rotation=90, va="center", ha="center",
                        color=col, fontsize=14, fontweight="bold")

    fig.suptitle("FiberNet — 2D Generator Gallery", color=FG, fontsize=18, y=0.98)
    fig.tight_layout(rect=[0.05, 0.02, 1, 0.95])
    p = os.path.join(OUT, "01_2d_generators.png")
    fig.savefig(p, dpi=180, facecolor=BG)
    plt.close(fig)
    print(f"[OK] {p}")


# ── FIGURE 2: 3D Generator Gallery ──────────────────────────────────
def fig2():
    items = [
        ("random_straight_3d",  dict(num_fibers=40, fiber_length=10, box_size=20.0, seed=42)),
        ("cubic_lattice_3d",    dict(spacing=5, grid_size=(2,2,2))),
        ("octet_truss_3d",      dict(spacing=5, grid_size=(2,2,2))),
        ("woven_3d_orthogonal", dict(spacing=3, grid_size=(2,2,2))),
        ("random_bundle_3d",    dict(num_fibers=20, bundle_length=15, seed=42)),
        ("helical_fiber_3d",    dict(radius_helix=3, pitch=2, num_turns=3)),
        ("bezier_fiber_3d",     dict(control_points=[(0,0,0),(5,5,0),(10,0,5),(15,5,5)])),
        ("foam_like_3d",        dict(num_cells=8, seed=42)),
        ("diamond_lattice_3d",  dict(spacing=5, grid_size=(2,2,2))),
    ]
    cols = 3
    rows = (len(items) + cols - 1) // cols
    fig = plt.figure(figsize=(5*cols, 5*rows))
    fig.patch.set_facecolor(BG)

    for i, (name, kw) in enumerate(items):
        ax = fig.add_subplot(rows, cols, i+1, projection="3d")
        ax.set_facecolor(BG)
        net = safe_gen(name, **kw)
        if net is not None:
            draw_3d(net, ax, color="#00ccff", lw=0.8)
        ax.set_title(name.replace("_", " "), color=FG, fontsize=9, pad=2)

    fig.suptitle("FiberNet — 3D Generator Gallery", color=FG, fontsize=18, y=1.0)
    fig.tight_layout()
    p = os.path.join(OUT, "02_3d_generators.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[OK] {p}")


# ── FIGURE 3: Metamaterials & TPMS ──────────────────────────────────
def fig3():
    items = [
        ("reentrant_honeycomb_2d",  dict(reentrant_angle=150, cell_height=10, cell_width=10, grid_size=(3,3))),
        ("chiral_honeycomb_2d",     dict(node_radius=3, ligament_length=8, grid_size=(3,3))),
        ("star_honeycomb_2d",       dict(cell_size=10, grid_size=(3,3))),
        ("arrowhead_auxetic_2d",    dict(cell_size=10, grid_size=(3,3))),
        ("missing_rib_auxetic_2d",  dict(cell_size=10, grid_size=(3,3))),
        ("hierarchical_lattice_2d", dict(cell_size=20, grid_size=(3,3), levels=2)),
        ("tpms_sheet",              dict(kind="gyroid", resolution=20)),
        ("tpms_lattice",            dict(kind="primitive", resolution=20)),
        ("tpms_gradient",           dict(kind="gyroid", resolution=10)),
    ]
    cols = 3
    rows = (len(items) + cols - 1) // cols
    fig = plt.figure(figsize=(5*cols, 5*rows))
    fig.patch.set_facecolor(BG)

    for i, (name, kw) in enumerate(items):
        is_3d = "tpms" in name
        ax = fig.add_subplot(rows, cols, i+1, projection="3d" if is_3d else None)
        ax.set_facecolor(BG)
        net = safe_gen(name, **kw)
        if net is not None:
            if is_3d:
                draw_3d(net, ax, color="#20c997", lw=0.6)
            else:
                draw_2d(net, ax, color="#20c997", lw=1.0)
        ax.set_title(name.replace("_", " "), color=FG, fontsize=9, pad=2)

    fig.suptitle("Metamaterials & TPMS Generators", color=FG, fontsize=18, y=1.0)
    fig.tight_layout()
    p = os.path.join(OUT, "03_metamaterials_tpms.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[OK] {p}")


# ── FIGURE 4: Fractals, Bundles, Hierarchical, Laminates ────────────
def fig4():
    cats = {
        "Fractal": [
            ("sierpinski_triangle", dict(iterations=4, size=20)),
            ("koch_curve",          dict(iterations=4, start=(0,0), end=(20,0))),
            ("fractal_tree",        dict(iterations=5, trunk_length=10)),
            ("hilbert_curve",       dict(order=4, size=20)),
        ],
        "Bundles": [
            ("parallel_bundle_2d",  dict(num_fibers=10, bundle_length=20, bundle_width=5)),
            ("twisted_bundle_2d",   dict(num_fibers=8, bundle_length=30)),
            ("crimped_network_2d",  dict(num_fibers=15, crimp_amplitude=1.0, seed=42)),
        ],
        "Hierarchical": [
            ("gradient_density_network", dict(num_fibers=60, seed=42)),
            ("core_shell_fiber",    dict(num_shell_fibers=6)),
            ("fractal_network",     dict(iterations=2, branch_factor=2)),
        ],
        "Laminates": [
            ("unidirectional_laminate", dict(num_layers=4, fibers_per_layer=15, fiber_length=30)),
            ("crossply_laminate",   dict(num_layers=4, fibers_per_layer=15, fiber_length=30)),
            ("angle_ply_laminate",  dict(num_layers=4, angle=np.pi/4, fibers_per_layer=15, fiber_length=30)),
            ("quasi_isotropic_laminate", dict(num_fibers_per_layer=15, fiber_length=30)),
        ],
    }
    max_cols = max(len(v) for v in cats.values())
    n_cats = len(cats)
    fig, axes = plt.subplots(n_cats, max_cols, figsize=(4*max_cols, 4*n_cats))
    fig.patch.set_facecolor(BG)

    for r, (cat, items) in enumerate(cats.items()):
        col = CAT_COLORS.get(cat, ACCENT)
        for c in range(max_cols):
            ax = axes[r, c]
            ax.set_facecolor(BG)
            ax.axis("off")
            if c < len(items):
                name, kw = items[c]
                net = safe_gen(name, **kw)
                if net is not None:
                    draw_2d(net, ax, color=col, lw=0.9)
                    ax.set_title(name.replace("_", " "), color=FG, fontsize=8, pad=3)
        axes[r, 0].text(-0.3, 0.5, cat, transform=axes[r,0].transAxes,
                        rotation=90, va="center", ha="center",
                        color=col, fontsize=13, fontweight="bold")

    fig.suptitle("Fractals, Bundles, Hierarchical & Laminates", color=FG, fontsize=18, y=0.98)
    fig.tight_layout(rect=[0.05, 0.02, 1, 0.95])
    p = os.path.join(OUT, "04_fractals_bundles_laminates.png")
    fig.savefig(p, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"[OK] {p}")


# ── FIGURE 5: Unified API generators ───────────────────────────────
def fig5():
    items = [
        ("lattice_2d",            dict(topology="honeycomb", cell_size=5, grid_size=(4,4))),
        ("lattice_3d",            dict(topology="cubic", cell_size=5, grid_size=(2,2,2))),
        ("metamaterial_2d",       dict(mode="reentrant", cell_size=5, grid_size=(3,3))),
        ("curved_random_2d",      dict(num_fibers=40, seed=42)),
        ("entangled_3d",          dict(num_fibers=30, seed=42)),
        ("biomimetic_network",    dict(network_type="collagen", num_fibers=40, seed=42)),
        ("hierarchical_lattice", dict(levels=2, base_topology="triangular", cell_size=30)),
    ]
    cols = 4
    rows = (len(items) + cols - 1) // cols
    fig = plt.figure(figsize=(5*cols, 5*rows))
    fig.patch.set_facecolor(BG)

    for i, (name, kw) in enumerate(items):
        is_3d = "3d" in name or "entangled" in name
        ax = fig.add_subplot(rows, cols, i+1, projection="3d" if is_3d else None)
        ax.set_facecolor(BG)
        net = safe_gen(name, **kw)
        if net is not None:
            if is_3d:
                draw_3d(net, ax, color="#e599f7", lw=0.7)
            else:
                draw_2d(net, ax, color="#e599f7", lw=1.0)
        ax.set_title(name.replace("_", " "), color=FG, fontsize=10, pad=2)

    fig.suptitle("Unified API Generators (fn.create)", color=FG, fontsize=18, y=1.0)
    fig.tight_layout()
    p = os.path.join(OUT, "05_unified_api.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[OK] {p}")


# ── FIGURE 6: Taxonomy bar chart ────────────────────────────────────
def fig6():
    cats = {
        "Disordered": 7, "Ordered": 6, "Chiral": 5, "Woven": 4,
        "Hierarchical": 4, "Advanced": 11, "Variants": 7, "Specialized": 6,
        "Fractal": 4, "Gradient": 3, "Bundles": 5, "Curved": 6,
        "Laminates": 6, "Metamaterials": 11, "TPMS": 3, "Field-Guided": 4,
        "Regular": 2, "Unified": 7,
    }
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    names = list(cats.keys())
    counts = list(cats.values())
    colors = [CAT_COLORS.get(n, ACCENT) for n in names]

    bars = ax.barh(names, counts, color=colors, edgecolor="none", alpha=0.85, height=0.7)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                str(c), va="center", ha="left", color=FG, fontsize=11, fontweight="bold")

    ax.set_xlabel("Number of Generators", color=FG, fontsize=12)
    ax.set_title("FiberNet Generator Taxonomy — 101 Core Generators across 18 Modules",
                 color=FG, fontsize=16, pad=12)
    ax.tick_params(colors=FG)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color("#30363d")
    ax.set_xlim(0, max(counts) + 2)

    fig.tight_layout()
    p = os.path.join(OUT, "06_taxonomy.png")
    fig.savefig(p, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"[OK] {p}")


if __name__ == "__main__":
    print("Generating core generator gallery...")
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    print("Done.")
