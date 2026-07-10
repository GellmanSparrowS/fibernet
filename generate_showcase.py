#!/usr/bin/env python3
"""Generate showcase images with intermediate points and non-zero displacement."""

from fibernet import pattern_2d, pattern_3d, list_units, render_graph, render_gallery
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "output_viz"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Clear old images
for f in os.listdir(OUTPUT_DIR):
    if f.endswith('.png'):
        os.remove(os.path.join(OUTPUT_DIR, f))

print("=" * 60)
print("Generating showcase images with intermediate points")
print("=" * 60)

# Image 1: Gallery of all 2D units with n_pts_per_side=3
print("\n1. 2D unit gallery (n_pts_per_side=3)")
units = list_units()
graphs = []
for unit in units:
    g = pattern_2d(unit=unit, box=(10, 10), grid=(3, 3), 
                   n_pts_per_side=3, seed=42, n_internal=8)
    graphs.append(g)
    print(f"  {unit:15s}: {g.num_nodes:3d} nodes, {g.num_edges:3d} edges")

render_gallery(
    graphs=graphs,
    titles=units,
    ncols=4,
    figsize_per_cell=(5, 5),
    theme="dark",
    line_width=1.2,
    show_nodes=False,
    color_by="orientation",
    suptitle="FiberNet 2D Metamaterials (n_pts_per_side=3)",
    save_path=os.path.join(OUTPUT_DIR, "01_2d_gallery.png"),
    dpi=200,
)
plt.close()
print("  ✓ Saved 01_2d_gallery.png")

# Image 2: Honeycomb detail with n_pts_per_side=5
print("\n2. Honeycomb detail (n_pts_per_side=5)")
g = pattern_2d(unit="honeycomb", box=(10, 10), grid=(5, 5), 
               n_pts_per_side=5, seed=42, n_internal=10)
print(f"  {g.num_nodes} nodes, {g.num_edges} edges")

render_graph(
    graph=g,
    figsize=(12, 12),
    theme="dark",
    color_by="orientation",
    line_width=1.8,
    show_nodes=False,
    title="Honeycomb Detail (n_pts_per_side=5)",
    save_path=os.path.join(OUTPUT_DIR, "02_honeycomb_detail.png"),
    dpi=200,
)
plt.close()
print("  ✓ Saved 02_honeycomb_detail.png")

# Image 3: Kagome blueprint with n_pts_per_side=4
print("\n3. Kagome blueprint (n_pts_per_side=4)")
g = pattern_2d(unit="kagome", box=(10, 10), grid=(4, 4), 
               n_pts_per_side=4, seed=42, n_internal=8)
print(f"  {g.num_nodes} nodes, {g.num_edges} edges")

render_graph(
    graph=g,
    figsize=(12, 12),
    theme="blueprint",
    color_by="orientation",
    line_width=2.0,
    show_nodes=False,
    title="Kagome Lattice (n_pts_per_side=4)",
    save_path=os.path.join(OUTPUT_DIR, "03_kagome_blueprint.png"),
    dpi=200,
)
plt.close()
print("  ✓ Saved 03_kagome_blueprint.png")

# Image 4: Auxetic comparison (honeycomb vs reentrant)
print("\n4. Auxetic comparison (honeycomb vs reentrant)")
g1 = pattern_2d(unit="honeycomb", box=(10, 10), grid=(4, 4), 
                n_pts_per_side=4, seed=42, n_internal=8)
g2 = pattern_2d(unit="reentrant", box=(10, 10), grid=(4, 4), 
                n_pts_per_side=4, seed=42, n_internal=8)
print(f"  Honeycomb: {g1.num_nodes} nodes")
print(f"  Reentrant: {g2.num_nodes} nodes")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
render_graph(
    graph=g1,
    ax=axes[0],
    theme="dark",
    color_by="orientation",
    line_width=1.8,
    show_nodes=False,
    title="Honeycomb (regular)",
)
render_graph(
    graph=g2,
    ax=axes[1],
    theme="dark",
    color_by="orientation",
    line_width=1.8,
    show_nodes=False,
    title="Reentrant (auxetic)",
)
fig.suptitle("Regular vs Auxetic Structures", fontsize=16, color='white', y=0.98)
fig.patch.set_facecolor('#1a1a2e')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "04_auxetic_comparison.png"), 
            dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("  ✓ Saved 04_auxetic_comparison.png")

# Image 5: 3D cubic with intermediates
print("\n5. 3D cubic (n_pts_per_side=3)")
g = pattern_3d(unit="cubic", box=(10, 10, 10), grid=(3, 3, 3), 
               n_pts_per_side=3, seed=42, n_internal=8)
print(f"  {g.num_nodes} nodes, {g.num_edges} edges")

from fibernet import render_graph_3d
render_graph_3d(
    graph=g,
    figsize=(12, 12),
    theme="dark",
    line_width=1.5,
    depth_alpha=True,
    title="Cubic 3D (n_pts_per_side=3)",
    save_path=os.path.join(OUTPUT_DIR, "05_3d_cubic.png"),
    dpi=200,
)
plt.close()
print("  ✓ Saved 05_3d_cubic.png")

# Image 6: 3D octet with intermediates
print("\n6. 3D octet (n_pts_per_side=3)")
g = pattern_3d(unit="octet", box=(10, 10, 10), grid=(2, 2, 2), 
               n_pts_per_side=3, seed=42, n_internal=8)
print(f"  {g.num_nodes} nodes, {g.num_edges} edges")

render_graph_3d(
    graph=g,
    figsize=(12, 12),
    theme="dark",
    line_width=1.5,
    depth_alpha=True,
    title="Octet 3D (n_pts_per_side=3)",
    save_path=os.path.join(OUTPUT_DIR, "06_3d_octet.png"),
    dpi=200,
)
plt.close()
print("  ✓ Saved 06_3d_octet.png")

# Image 7: Chiral with stats
print("\n7. Chiral with statistics")
g = pattern_2d(unit="chiral", box=(10, 10), grid=(4, 4), 
               n_pts_per_side=4, seed=42, n_internal=8)
print(f"  {g.num_nodes} nodes, {g.num_edges} edges")

from fibernet import render_with_stats
render_with_stats(
    graph=g,
    figsize=(12, 12),
    theme="dark",
    color_by="orientation",
    line_width=1.5,
    show_nodes=False,
    save_path=os.path.join(OUTPUT_DIR, "07_chiral_stats.png"),
    dpi=200,
)
plt.close()
print("  ✓ Saved 07_chiral_stats.png")

# Image 8: Star pattern with intermediates
print("\n8. Star pattern (n_pts_per_side=5)")
g = pattern_2d(unit="star", box=(10, 10), grid=(3, 3), 
               n_pts_per_side=5, seed=42, n_internal=8)
print(f"  {g.num_nodes} nodes, {g.num_edges} edges")

render_graph(
    graph=g,
    figsize=(12, 12),
    theme="dark",
    color_by="orientation",
    line_width=1.8,
    show_nodes=False,
    title="Star Pattern (n_pts_per_side=5)",
    save_path=os.path.join(OUTPUT_DIR, "08_star_pattern.png"),
    dpi=200,
)
plt.close()
print("  ✓ Saved 08_star_pattern.png")

# Image 9: Cross pattern with intermediates
print("\n9. Cross pattern (n_pts_per_side=4)")
g = pattern_2d(unit="cross", box=(10, 10), grid=(3, 3), 
               n_pts_per_side=4, seed=42, n_internal=8)
print(f"  {g.num_nodes} nodes, {g.num_edges} edges")

render_graph(
    graph=g,
    figsize=(12, 12),
    theme="dark",
    color_by="orientation",
    line_width=1.8,
    show_nodes=False,
    title="Cross Pattern (n_pts_per_side=4)",
    save_path=os.path.join(OUTPUT_DIR, "09_cross_pattern.png"),
    dpi=200,
)
plt.close()
print("  ✓ Saved 09_cross_pattern.png")

print("\n" + "=" * 60)
print("All 9 showcase images generated successfully!")
print("=" * 60)
