"""
Generate publication-quality visualization gallery for FiberNet.
Replaces content in fibernet_output/structures/
"""
import matplotlib
matplotlib.use('Agg')
import sys
sys.path.insert(0, '/home/codex/projects/codex_test/fibernet')
import fibernet as fn
from fibernet.viz import plot, plot_3d, plot_comparison, plot_statistics, save_figure
from pathlib import Path
import numpy as np

output_dir = Path("fibernet_output/structures")
output_dir.mkdir(parents=True, exist_ok=True)

# Remove old images
for f in output_dir.glob("*.png"):
    f.unlink()
    print(f"  Removed old: {f.name}")

print("Generating visualization gallery...")

# ====== 1. Metamaterial Gallery (dark theme, orientation coloring) ======
print("\n--- Metamaterial Gallery ---")
metamaterials = [
    ("reentrant_honeycomb_2d", "Re-entrant Honeycomb"),
    ("chiral_honeycomb_2d", "Chiral Honeycomb"),
    ("star_honeycomb_2d", "Star Honeycomb"),
    ("arrowhead_auxetic_2d", "Arrowhead Auxetic"),
    ("hierarchical_lattice_2d", "Hierarchical Lattice"),
    ("missing_rib_auxetic_2d", "Missing-Rib Auxetic"),
]

nets = []
labels = []
for name, label in metamaterials:
    net = fn.create(name)
    nets.append(net)
    labels.append(label)
    print(f"  {label}: {net.num_fibers}F {net.num_crosslinks}CL")

fig, axes = plot_comparison(
    nets, labels=labels,
    color_by="orientation",
    theme="dark",
    ncols=3,
    figsize_per=(7, 7),
)
save_figure(fig, str(output_dir / "metamaterial_gallery.png"), dpi=300)
print("  Saved: metamaterial_gallery.png")

# ====== 2. Ordered Lattices (blueprint theme) ======
print("\n--- Ordered Lattices ---")
lattices = [
    ("square_2d", "Square Lattice"),
    ("triangular_2d", "Triangular Lattice"),
    ("honeycomb_2d", "Honeycomb"),
    ("kagome_2d", "Kagome Lattice"),
]

nets2 = []
labels2 = []
for name, label in lattices:
    net = fn.create(name)
    nets2.append(net)
    labels2.append(label)
    print(f"  {label}: {net.num_fibers}F {net.num_crosslinks}CL")

fig, axes = plot_comparison(
    nets2, labels=labels2,
    color_by="uniform",
    theme="blueprint",
    ncols=2,
    figsize_per=(8, 8),
)
save_figure(fig, str(output_dir / "ordered_lattices.png"), dpi=300)
print("  Saved: ordered_lattices.png")

# ====== 3. 3D Structures ======
print("\n--- 3D Structures ---")
structures_3d = [
    ("cubic_3d", "Cubic Lattice"),
    ("diamond_lattice_3d", "Diamond Lattice"),
    ("proper_octet_truss_3d", "Octet Truss"),
    ("reentrant_honeycomb_3d", "Re-entrant 3D"),
]

for name, label in structures_3d:
    net = fn.create(name)
    fig, ax = plot_3d(net, title=label, theme="dark",
                     elevation=25, azimuth=-60)
    save_figure(fig, str(output_dir / f"3d_{name}.png"), dpi=250)
    print(f"  Saved: 3d_{name}.png ({net.num_fibers}F)")

# ====== 4. Fractal Gallery ======
print("\n--- Fractal Gallery ---")
fractals = [
    ("sierpinski", "Sierpinski Triangle"),
    ("fractal_tree", "Fractal Tree"),
    ("hilbert", "Hilbert Curve"),
]

nets_f = []
labels_f = []
for name, label in fractals:
    net = fn.create(name)
    nets_f.append(net)
    labels_f.append(label)
    print(f"  {label}: {net.num_fibers}F")

fig, axes = plot_comparison(
    nets_f, labels=labels_f,
    color_by="orientation",
    theme="dark",
    ncols=3,
    figsize_per=(7, 7),
)
save_figure(fig, str(output_dir / "fractal_gallery.png"), dpi=300)
print("  Saved: fractal_gallery.png")

# ====== 5. Disordered Networks ======
print("\n--- Disordered Networks ---")
disordered = [
    ("random_2d", "Random 2D (Mikado)"),
    ("random_3d", "Random 3D"),
    ("electrospun", "Electrospun"),
    ("voronoi_2d", "Voronoi 2D"),
]

nets_d = []
labels_d = []
for name, label in disordered:
    net = fn.create(name)
    nets_d.append(net)
    labels_d.append(label)
    print(f"  {label}: {net.num_fibers}F")

fig, axes = plot_comparison(
    nets_d, labels=labels_d,
    color_by="orientation",
    theme="light",
    ncols=2,
    figsize_per=(8, 8),
)
save_figure(fig, str(output_dir / "disordered_networks.png"), dpi=300)
print("  Saved: disordered_networks.png")

# ====== 6. Chiral/Helical Structures ======
print("\n--- Chiral Structures ---")
chiral = [
    ("helix", "Single Helix"),
    ("double_helix", "Double Helix"),
    ("braided_rope", "Braided Rope"),
    ("twisted_bundle", "Twisted Bundle"),
]

nets_c = []
labels_c = []
for name, label in chiral:
    net = fn.create(name)
    nets_c.append(net)
    labels_c.append(label)
    print(f"  {label}: {net.num_fibers}F")

fig, axes = plot_comparison(
    nets_c, labels=labels_c,
    color_by="uniform",
    theme="dark",
    ncols=2,
    figsize_per=(8, 8),
)
save_figure(fig, str(output_dir / "chiral_structures.png"), dpi=300)
print("  Saved: chiral_structures.png")

# ====== 7. Woven Structures ======
print("\n--- Woven Structures ---")
woven = [
    ("plain_weave", "Plain Weave"),
    ("twill_weave", "Twill Weave"),
]

nets_w = []
labels_w = []
for name, label in woven:
    net = fn.create(name)
    nets_w.append(net)
    labels_w.append(label)
    print(f"  {label}: {net.num_fibers}F")

fig, axes = plot_comparison(
    nets_w, labels=labels_w,
    color_by="orientation",
    theme="blueprint",
    ncols=2,
    figsize_per=(8, 8),
)
save_figure(fig, str(output_dir / "woven_structures.png"), dpi=300)
print("  Saved: woven_structures.png")

# ====== 8. Statistics Panel (best example) ======
print("\n--- Statistics ---")
net_stats = fn.create("reentrant_honeycomb_2d")
fig = plot_statistics(net_stats, theme="dark")
save_figure(fig, str(output_dir / "statistics_reentrant.png"), dpi=250)
print("  Saved: statistics_reentrant.png")

net_stats2 = fn.create("kagome_2d")
fig = plot_statistics(net_stats2, theme="blueprint")
save_figure(fig, str(output_dir / "statistics_kagome.png"), dpi=250)
print("  Saved: statistics_kagome.png")

# ====== 9. Theme Showcase ======
print("\n--- Theme Showcase ---")
net_theme = fn.create("reentrant_honeycomb_2d")
themes = ["light", "dark", "publication", "blueprint"]
nets_t = [net_theme] * 4
labels_t = [f"Theme: {t}" for t in themes]

fig, axes = plot_comparison(
    nets_t, labels=labels_t,
    color_by="orientation",
    theme="light",  # Will be overridden per-plot in future
    ncols=4,
    figsize_per=(5, 5),
)
save_figure(fig, str(output_dir / "theme_showcase.png"), dpi=250)
print("  Saved: theme_showcase.png")

# ====== 10. Re-entrant Angle Series ======
print("\n--- Re-entrant Angle Series ---")
angles = [120, 135, 150, 165]
nets_a = []
labels_a = []
for angle in angles:
    net = fn.gen.reentrant_honeycomb_2d(reentrant_angle=angle, grid_size=(4, 4))
    nets_a.append(net)
    labels_a.append(f"θ = {angle}°")
    print(f"  θ={angle}°: {net.num_fibers}F")

fig, axes = plot_comparison(
    nets_a, labels=labels_a,
    color_by="uniform",
    theme="dark",
    ncols=4,
    figsize_per=(5, 5),
)
save_figure(fig, str(output_dir / "reentrant_angle_series.png"), dpi=300)
print("  Saved: reentrant_angle_series.png")

print("\n" + "=" * 60)
print(f"Gallery generation complete!")
print(f"Output directory: {output_dir}")
print(f"Files generated: {len(list(output_dir.glob('*.png')))}")
