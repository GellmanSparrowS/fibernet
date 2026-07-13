"""
Generate one visualization per generator category.
Each figure: 1 row × 5 columns showing parametric variation.
"""
import sys, os, gc
sys.path.insert(0, '/home/codex/projects/codex_test/fibernet')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import fibernet as fn
from fibernet.gen import disordered, ordered, bundles, laminates
from fibernet.gen import metamaterials, chiral, fractal, advanced
from fibernet.gen import specialized, curved, gradient, hierarchical
from fibernet.gen import tpms, woven, variants

OUTPUT_DIR = '/home/codex/projects/codex_test/fibernet/output_viz'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_2d(ax, net, title='', color='steelblue', lw=0.8, show_cl=False):
    if net is None or net.num_fibers == 0:
        ax.text(0.5, 0.5, 'Empty', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=8)
        return
    lines = [f.centerline[:, :2] for f in net.fibers if len(f.centerline) >= 2]
    if lines:
        lc = LineCollection(lines, linewidths=lw, colors=color, alpha=0.85)
        ax.add_collection(lc)
    if show_cl and net.crosslinks:
        cl_pts = np.array([cl.position[:2] for cl in net.crosslinks[:800]])
        if len(cl_pts) > 0:
            ax.scatter(cl_pts[:, 0], cl_pts[:, 1], c='red', s=2, zorder=5, alpha=0.4)
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=7, pad=2)
    ax.tick_params(labelsize=5)


def plot_3d(ax, net, title='', color='steelblue', lw=0.5):
    if net is None or net.num_fibers == 0:
        ax.text2D(0.5, 0.5, 'Empty', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=8)
        return
    lines = [f.centerline for f in net.fibers if len(f.centerline) >= 2]
    if lines:
        lc = Line3DCollection(lines, linewidths=lw, colors=color, alpha=0.7)
        ax.add_collection3d(lc)
    all_pts = np.vstack([f.centerline for f in net.fibers if len(f.centerline) >= 2])
    if len(all_pts) > 0:
        mins, maxs = all_pts.min(axis=0), all_pts.max(axis=0)
        center = (mins + maxs) / 2
        span = max(max(maxs - mins) / 2 * 1.1, 0.5)
        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        if maxs[2] - mins[2] > 0.01:
            ax.set_zlim(center[2] - span, center[2] + span)
        else:
            ax.set_zlim(-0.5, 0.5)
    ax.set_title(title, fontsize=7, pad=2)
    ax.tick_params(labelsize=4)


def safe(fn_func, **kwargs):
    try:
        return fn_func(**kwargs)
    except Exception as e:
        print(f"  WARN: {e}")
        return None


COLORS_2D = ['#2266AA', '#DD7733', '#339955', '#8833AA', '#CC3344']
COLORS_3D = ['#4488CC', '#DD7733', '#44AA66', '#8844AA', '#CC4455']


def make_figure(cat_name, items, is_3d=False, filename=None):
    n = len(items)
    if is_3d:
        fig = plt.figure(figsize=(n * 4.5, 4.8))
    else:
        fig, axes = plt.subplots(1, n, figsize=(n * 3.8, 3.8))
        if n == 1:
            axes = [axes]

    fig.suptitle(cat_name, fontsize=11, fontweight='bold', y=1.01)

    for i, (title, gen_fn, kwargs) in enumerate(items):
        if is_3d:
            ax = fig.add_subplot(1, n, i + 1, projection='3d')
        else:
            ax = axes[i]

        net = safe(gen_fn, **kwargs)
        c = COLORS_3D[i] if is_3d else COLORS_2D[i]

        if net is None:
            if is_3d:
                ax.text2D(0.5, 0.5, 'Error', ha='center', va='center')
            else:
                ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=7)
            continue

        # For 3D nets plotted in 2D, project to xy
        if not is_3d and net.dimension == 3:
            plot_2d(ax, net, title=title, color=c, lw=0.5)
        elif is_3d:
            plot_3d(ax, net, title=title, color=c)
        else:
            plot_2d(ax, net, title=title, color=c)

        nf = net.num_fibers if net else 0
        nc = net.num_crosslinks if net else 0
        subtitle = f"{nf}fib {nc}cl"
        if is_3d:
            ax.set_title(f"{title}\n({subtitle})", fontsize=6, pad=1)
        else:
            ax.set_title(f"{title}\n({subtitle})", fontsize=6, pad=1)

        del net
        gc.collect()

    plt.tight_layout()
    if filename is None:
        filename = cat_name.lower().replace(' ', '_').replace('/', '_') + '.png'
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {filename}")


# ====================================================================
# 1. Disordered 2D
# ====================================================================
print("1. Disordered 2D")
make_figure("1. Disordered 2D Networks", [
    ("random_2d (N=60)", disordered.random_straight_2d, {"num_fibers": 60}),
    ("random_2d (N=150)", disordered.random_straight_2d, {"num_fibers": 150}),
    ("random_2d (L=20)", disordered.random_straight_2d, {"num_fibers": 80, "fiber_length": 20.0}),
    ("oriented (σ=0.1)", disordered.oriented_random_2d, {"num_fibers": 100, "angle_std": 0.1}),
    ("oriented (σ=0.5)", disordered.oriented_random_2d, {"num_fibers": 100, "angle_std": 0.5}),
], is_3d=False, filename="01_disordered_2d.png")

# ====================================================================
# 2. Disordered 3D
# ====================================================================
print("2. Disordered 3D")
make_figure("2. Disordered 3D Networks", [
    ("random_3d (N=60)", disordered.random_straight_3d, {"num_fibers": 60}),
    ("random_3d (N=120)", disordered.random_straight_3d, {"num_fibers": 120}),
    ("oriented_3d (σ=0.1)", disordered.oriented_random_3d, {"num_fibers": 80, "angle_std": 0.1}),
    ("oriented_3d (σ=0.4)", disordered.oriented_random_3d, {"num_fibers": 80, "angle_std": 0.4}),
    ("random_walk (N=30)", disordered.random_walk_fibers, {"num_fibers": 30, "num_steps": 20}),
], is_3d=True, filename="02_disordered_3d.png")

# ====================================================================
# 3. Ordered 2D
# ====================================================================
print("3. Ordered 2D")
make_figure("3. Ordered Lattices 2D", [
    ("square_2d", lambda: fn.create("square_2d"), {}),
    ("honeycomb_2d", lambda: fn.create("honeycomb_2d"), {}),
    ("triangular_2d", lambda: fn.create("triangular_2d"), {}),
    ("kagome_2d", lambda: fn.create("kagome_2d"), {}),
    ("poisson_line_2d", lambda: fn.create("poisson_line_2d"), {}),
], is_3d=False, filename="03_ordered_2d.png")

# ====================================================================
# 4. Ordered 3D
# ====================================================================
print("4. Ordered 3D")
make_figure("4. Ordered Lattices 3D", [
    ("cubic_3d", lambda: fn.create("cubic_3d"), {}),
    ("octet_3d", lambda: fn.create("octet_3d"), {}),
    ("diamond_3d", lambda: fn.create("diamond_lattice_3d"), {}),
    ("plate_lattice_3d", lambda: fn.create("plate_lattice_3d"), {}),
    ("woven_3d", lambda: fn.create("woven_3d"), {}),
], is_3d=True, filename="04_ordered_3d.png")

# ====================================================================
# 5. Chiral / Helical
# ====================================================================
print("5. Chiral / Helical")
make_figure("5. Chiral / Helical", [
    ("helix", lambda: fn.create("helix"), {}),
    ("double_helix", lambda: fn.create("double_helix"), {}),
    ("braided_rope", lambda: fn.create("braided_rope"), {}),
    ("twisted_bundle", lambda: fn.create("twisted_bundle"), {}),
    ("chiral_honeycomb_2d", lambda: fn.create("chiral_honeycomb_2d"), {}),
], is_3d=False, filename="05_chiral_helical.png")

# ====================================================================
# 6. Woven
# ====================================================================
print("6. Woven")
make_figure("6. Woven Structures", [
    ("plain_weave", lambda: fn.create("plain_weave"), {}),
    ("twill_weave", lambda: fn.create("twill_weave"), {}),
    ("satin_weave", lambda: fn.create("satin_weave"), {}),
    ("textile_weave", lambda: fn.create("textile_weave"), {}),
    ("woven_3d", lambda: fn.create("woven_3d"), {}),
], is_3d=False, filename="06_woven.png")

# ====================================================================
# 7. Metamaterials 2D
# ====================================================================
print("7. Metamaterials 2D")
make_figure("7. Metamaterials 2D (Auxetic)", [
    ("reentrant_2d", lambda: fn.create("reentrant_honeycomb_2d"), {}),
    ("star_honeycomb_2d", lambda: fn.create("star_honeycomb_2d"), {}),
    ("arrowhead_auxetic", lambda: fn.create("arrowhead_auxetic_2d"), {}),
    ("missing_rib_auxetic", lambda: fn.create("missing_rib_auxetic_2d"), {}),
    ("hierarchical_lattice", lambda: fn.create("hierarchical_lattice_2d"), {}),
], is_3d=False, filename="07_metamaterials_2d.png")

# ====================================================================
# 8. Metamaterials 3D
# ====================================================================
print("8. Metamaterials 3D")
make_figure("8. Metamaterials 3D", [
    ("reentrant_3d", lambda: fn.create("reentrant_honeycomb_3d"), {}),
    ("proper_octet_truss", lambda: fn.create("proper_octet_truss_3d"), {}),
    ("diamond_3d", lambda: fn.create("diamond_lattice_3d"), {}),
    ("gyroid_lattice_3d", lambda: fn.create("gyroid_lattice_3d"), {}),
    ("plate_lattice_3d", lambda: fn.create("plate_lattice_3d"), {}),
], is_3d=True, filename="08_metamaterials_3d.png")

# ====================================================================
# 9. Fractals
# ====================================================================
print("9. Fractals")
make_figure("9. Fractal Structures", [
    ("sierpinski", lambda: fn.create("sierpinski"), {}),
    ("koch_curve", lambda: fn.create("koch_curve"), {}),
    ("fractal_tree", lambda: fn.create("fractal_tree"), {}),
    ("hilbert", lambda: fn.create("hilbert"), {}),
    ("fractal_network", lambda: fn.create("fractal_network"), {}),
], is_3d=False, filename="09_fractals.png")

# ====================================================================
# 10. Biomimetic
# ====================================================================
print("10. Biomimetic")
make_figure("10. Biomimetic / Bio-inspired", [
    ("biomimetic_collagen", lambda: fn.create("biomimetic_collagen"), {}),
    ("biomimetic_fibrin", lambda: fn.create("biomimetic_fibrin"), {}),
    ("electrospun", lambda: fn.create("electrospun"), {}),
    ("meltblown", lambda: fn.create("meltblown"), {}),
    ("paper_network", lambda: fn.create("paper_network"), {}),
], is_3d=False, filename="10_biomimetic.png")

# ====================================================================
# 11. Bundles
# ====================================================================
print("11. Bundles")
make_figure("11. Fiber Bundles", [
    ("parallel_bundle_2d", lambda: fn.create("parallel_bundle_2d"), {}),
    ("twisted_bundle_2d", lambda: fn.create("twisted_bundle_2d"), {}),
    ("random_bundle_3d", lambda: fn.create("random_bundle_3d"), {}),
    ("braided_bundle_3d", lambda: fn.create("braided_bundle_3d"), {}),
    ("tendon_like_bundle", lambda: fn.create("tendon_like_bundle_3d"), {}),
], is_3d=False, filename="11_bundles.png")

# ====================================================================
# 12. Laminates
# ====================================================================
print("12. Laminates")
make_figure("12. Composite Laminates", [
    ("unidirectional", lambda: fn.create("unidirectional_laminate"), {}),
    ("crossply [0/90]", lambda: fn.create("crossply_laminate"), {}),
    ("angle_ply [±45]", lambda: fn.create("angle_ply_laminate"), {}),
    ("quasi_isotropic", lambda: fn.create("quasi_isotropic_laminate"), {}),
    ("sandwich", lambda: fn.create("sandwich_laminate"), {}),
], is_3d=True, filename="12_laminates.png")

# ====================================================================
# 13. Gradient / Hierarchical
# ====================================================================
print("13. Gradient")
make_figure("13. Gradient / Hierarchical", [
    ("density_gradient", lambda: fn.create("density_gradient_2d"), {}),
    ("property_gradient", lambda: fn.create("property_gradient_2d"), {}),
    ("gradient_density", lambda: fn.create("gradient_density"), {}),
    ("hierarchical_lattice", lambda: fn.create("hierarchical_lattice_2d"), {}),
    ("hierarchical_bundle", lambda: fn.create("hierarchical_bundle"), {}),
], is_3d=False, filename="13_gradient.png")

# ====================================================================
# 14. Voronoi / Advanced
# ====================================================================
print("14. Voronoi / Advanced")
make_figure("14. Voronoi / Advanced", [
    ("voronoi_2d", lambda: fn.create("voronoi_2d"), {}),
    ("voronoi_3d", lambda: fn.create("voronoi_3d"), {}),
    ("electrospun_mat", lambda: fn.create("electrospun_mat"), {}),
    ("auxetic_structure", lambda: fn.create("auxetic_structure"), {}),
    ("kirigami_structure", lambda: fn.create("kirigami_structure"), {}),
], is_3d=False, filename="14_voronoi_advanced.png")

# ====================================================================
# 15. TPMS
# ====================================================================
print("15. TPMS")
make_figure("15. TPMS (Triply Periodic Minimal Surfaces)", [
    ("tpms_sheet (r=8)", lambda: fn.create("tpms_sheet", resolution=8), {}),
    ("tpms_sheet (r=12)", lambda: fn.create("tpms_sheet", resolution=12), {}),
    ("tpms_lattice (r=8)", lambda: fn.create("tpms_lattice", resolution=8), {}),
    ("tpms_lattice (r=12)", lambda: fn.create("tpms_lattice", resolution=12), {}),
    ("tpms_gradient (r=6)", lambda: fn.create("tpms_gradient", resolution=6), {}),
], is_3d=True, filename="15_tpms.png")

# ====================================================================
# 16. Curved / Special
# ====================================================================
print("16. Curved / Special")
make_figure("16. Curved & Special Fibers", [
    ("crimped_2d", lambda: fn.create("crimped_network_2d"), {}),
    ("random_curved_3d", lambda: fn.create("random_curved_3d"), {}),
    ("random_curved_net", lambda: fn.create("random_curved_network_3d"), {}),
    ("cnt_network_2d", lambda: fn.create("cnt_network_2d"), {}),
    ("core_shell_fiber", lambda: fn.create("core_shell_fiber"), {}),
], is_3d=False, filename="16_curved_special.png")

# ====================================================================
# 17. Parametric: random_2d
# ====================================================================
print("17. Parametric Demo: random_2d")
make_figure("17. Parametric: random_2d (varying N)", [
    ("N=30", disordered.random_straight_2d, {"num_fibers": 30}),
    ("N=80", disordered.random_straight_2d, {"num_fibers": 80}),
    ("N=200", disordered.random_straight_2d, {"num_fibers": 200}),
    ("N=500", disordered.random_straight_2d, {"num_fibers": 500}),
    ("N=1000", disordered.random_straight_2d, {"num_fibers": 1000}),
], is_3d=False, filename="17_parametric_random.png")

# ====================================================================
# 18. Parametric: honeycomb cell_size
# ====================================================================
print("18. Parametric Demo: honeycomb cell_size")
make_figure("18. Parametric: honeycomb_2d (varying cell_size)", [
    ("cell=3", lambda: fn.create("honeycomb_2d", cell_size=3.0), {}),
    ("cell=5", lambda: fn.create("honeycomb_2d", cell_size=5.0), {}),
    ("cell=8", lambda: fn.create("honeycomb_2d", cell_size=8.0), {}),
    ("cell=12", lambda: fn.create("honeycomb_2d", cell_size=12.0), {}),
    ("cell=20", lambda: fn.create("honeycomb_2d", cell_size=20.0), {}),
], is_3d=False, filename="18_parametric_honeycomb.png")

# ====================================================================
# 19. Parametric: oriented angle_std
# ====================================================================
print("19. Parametric: oriented angle_std")
make_figure("19. Parametric: oriented_2d (varying angle_std)", [
    ("σ=0.01 (aligned)", disordered.oriented_random_2d, {"num_fibers": 100, "angle_std": 0.01}),
    ("σ=0.1", disordered.oriented_random_2d, {"num_fibers": 100, "angle_std": 0.1}),
    ("σ=0.3", disordered.oriented_random_2d, {"num_fibers": 100, "angle_std": 0.3}),
    ("σ=0.7", disordered.oriented_random_2d, {"num_fibers": 100, "angle_std": 0.7}),
    ("σ=1.5 (random)", disordered.oriented_random_2d, {"num_fibers": 100, "angle_std": 1.5}),
], is_3d=False, filename="19_parametric_oriented.png")

# ====================================================================
# 20. Class Generators: Regular + ZigZag
# ====================================================================
print("20. Class Generators")
from fibernet.gen.regular import RegularNetworkGenerator
from fibernet.gen.zigzag import ZigZagGenerator

make_figure("20. Class Generators (Regular / ZigZag)", [
    ("Regular(tiling=2)", lambda: RegularNetworkGenerator(tiling=2).to_fiber_network(), {}),
    ("Regular(tiling=3)", lambda: RegularNetworkGenerator(tiling=3).to_fiber_network(), {}),
    ("Regular(tiling=5)", lambda: RegularNetworkGenerator(tiling=5).to_fiber_network(), {}),
    ("ZigZag(3×3)", lambda: ZigZagGenerator(n_cols=3, n_rows=3).to_fiber_network(), {}),
    ("ZigZag(5×4)", lambda: ZigZagGenerator(n_cols=5, n_rows=4).to_fiber_network(), {}),
], is_3d=False, filename="20_class_generators.png")

print(f"\n=== All categories generated! ===")
print(f"Output: {OUTPUT_DIR}")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  {f}")
