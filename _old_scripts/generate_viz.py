"""
Generate comprehensive visualization outputs for FiberNet v2.0 graph API.
Produces high-quality images for manual review.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
import numpy as np
import networkx as nx

import fibernet as fn
from fibernet.viz.graph_plot import plot_graph, plot_graph_comparison, plot_structure_stats

OUT = "/home/codex/projects/codex_test/fibernet/output_viz"

print("=" * 60)
print("FiberNet v2.0 - Visualization Output Generation")
print("=" * 60)

# ============================================================
# 1. RegularNetworkGenerator: Basic → Tiled → Welded
# ============================================================
print("\n[1/8] Regular network: basic → tiled → welded")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1a: Basic unit cell
gen = fn.RegularNetworkGenerator(side_length=10, num_points_per_side=2, tiling=1, scale_to_unit=False)
G_unit = gen.generate()
plot_graph(G_unit, ax=axes[0], edge_width=2.5)
axes[0].set_title(f"Unit Cell\n{G_unit.number_of_nodes()} nodes, {G_unit.number_of_edges()} edges", fontsize=13)

# 1b: Tiled 3x3
gen_tiled = fn.RegularNetworkGenerator(side_length=10, num_points_per_side=2, tiling=3, scale_to_unit=False)
G_tiled = gen_tiled.generate()
plot_graph(G_tiled, ax=axes[1], edge_width=2.0)
axes[1].set_title(f"3×3 Tiled Array\n{G_tiled.number_of_nodes()} nodes, {G_tiled.number_of_edges()} edges", fontsize=13)

# 1c: Welded (intersection nodes detected)
G_welded = fn.weld_graph(G_tiled)
# Color intersection nodes differently
pos = nx.get_node_attributes(G_welded, 'pos')
ix_nodes = [n for n in G_welded.nodes() if str(n).startswith("IX_")]
reg_nodes = [n for n in G_welded.nodes() if not str(n).startswith("IX_")]

# Draw edges
edge_coords = []
for u, v in G_welded.edges():
    if u in pos and v in pos:
        pu = np.array(pos[u])[:2]
        pv = np.array(pos[v])[:2]
        edge_coords.append([pu, pv])
if edge_coords:
    lc = LineCollection(edge_coords, colors='black', linewidths=2.0, zorder=1)
    axes[2].add_collection(lc)

# Draw intersection nodes in red
if ix_nodes:
    ix_pos = np.array([np.array(pos[n])[:2] for n in ix_nodes])
    axes[2].scatter(ix_pos[:, 0], ix_pos[:, 1], c='red', s=30, zorder=3, label=f'Weld points ({len(ix_nodes)})')
    axes[2].legend(fontsize=10)

axes[2].set_aspect('equal')
axes[2].autoscale_view()
axes[2].axis('off')
axes[2].set_title(f"Welded Graph\n{G_welded.number_of_nodes()} nodes ({len(ix_nodes)} weld points)\n{G_welded.number_of_edges()} edges", fontsize=13)

plt.tight_layout()
fig.savefig(f"{OUT}/01_regular_network_pipeline.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: 01_regular_network_pipeline.png")

# ============================================================
# 2. Perturbation Variations
# ============================================================
print("[2/8] Perturbation variations")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
perturbations_list = [
    [(0.0, 0.0)],
    [(0.1, 0.0)],
    [(0.2, 0.1)],
    [(0.3, 0.2)],
    fn.RegularNetworkGenerator.random_perturbations(1, rng=np.random.default_rng(42)),
    fn.RegularNetworkGenerator.random_perturbations(1, rng=np.random.default_rng(123)),
]
labels = ["No perturbation", "dx=0.1", "dx=0.2, dy=0.1", "dx=0.3, dy=0.2", "Random (seed=42)", "Random (seed=123)"]

for idx, (perts, label) in enumerate(zip(perturbations_list, labels)):
    ax = axes[idx // 3][idx % 3]
    gen = fn.RegularNetworkGenerator(side_length=10, num_points_per_side=1,
                                     tiling=3, perturbations=perts, scale_to_unit=False)
    G = gen.generate()
    plot_graph(G, ax=ax, edge_width=1.5)
    ax.set_title(label, fontsize=11)

plt.suptitle("RegularNetworkGenerator: Perturbation Effects on 3×3 Array", fontsize=15, y=1.01)
plt.tight_layout()
fig.savefig(f"{OUT}/02_perturbation_variations.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: 02_perturbation_variations.png")

# ============================================================
# 3. ZigZag Generator
# ============================================================
print("[3/8] ZigZag generator variants")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Default base points
gen_zz = fn.ZigZagGenerator(n_cols=4, n_rows=4)
G_zz = gen_zz.generate()
plot_graph(G_zz, ax=axes[0], edge_width=2.0)
axes[0].set_title(f"ZigZag (default points)\n{G_zz.number_of_nodes()} nodes, {G_zz.number_of_edges()} edges", fontsize=12)

# Simple zigzag
gen_simple = fn.ZigZagGenerator.simple_zigzag(amplitude=40, wavelength=80, n_periods=6)
G_simple = gen_simple.generate()
plot_graph(G_simple, ax=axes[1], edge_width=2.5)
axes[1].set_title(f"Simple ZigZag\n{G_simple.number_of_nodes()} nodes", fontsize=12)

# Custom base points
custom_pts = [(0, 50), (30, 80), (60, 20), (90, 70), (120, 10), (150, 60)]
gen_custom = fn.ZigZagGenerator(base_points=custom_pts, n_cols=3, n_rows=5, mirror_x=True, mirror_y=True)
G_custom = gen_custom.generate()
plot_graph(G_custom, ax=axes[2], edge_width=1.5)
axes[2].set_title(f"Custom ZigZag (mirrored)\n{G_custom.number_of_nodes()} nodes", fontsize=12)

plt.tight_layout()
fig.savefig(f"{OUT}/03_zigzag_variants.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: 03_zigzag_variants.png")

# ============================================================
# 4. Structure + Statistics
# ============================================================
print("[4/8] Structure statistics panels")

G_stats = fn.RegularNetworkGenerator(side_length=10, num_points_per_side=2, tiling=4, scale_to_unit=False).generate()
fig, axes = plot_structure_stats(G_stats, figsize=(14, 5))
fig.suptitle("RegularNetworkGenerator: Structure Analysis", fontsize=14, y=1.02)
fig.savefig(f"{OUT}/04_structure_stats.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: 04_structure_stats.png")

# ============================================================
# 5. Feature Extraction Visualization
# ============================================================
print("[5/8] Feature extraction overview")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

configs = [
    ("Regular 3×3 (no pert)", fn.RegularNetworkGenerator(side_length=10, tiling=3, num_points_per_side=1).generate()),
    ("Regular 3×3 (perturbed)", fn.RegularNetworkGenerator(side_length=10, tiling=3, num_points_per_side=1, perturbations=[(0.2, 0.1)]).generate()),
    ("Regular 5×5", fn.RegularNetworkGenerator(side_length=10, tiling=5, num_points_per_side=2).generate()),
    ("ZigZag 4×4", fn.ZigZagGenerator(n_cols=4, n_rows=4).generate()),
    ("ZigZag (mirrored)", fn.ZigZagGenerator(n_cols=6, n_rows=6, mirror_x=True, mirror_y=True).generate()),
    ("Random geometric", None),
]

# Create random geometric graph
rng = np.random.default_rng(42)
G_rand = nx.Graph()
pts = rng.random((60, 2)) * 100
for i, p in enumerate(pts):
    G_rand.add_node(i, pos=tuple(p))
for i in range(60):
    for j in range(i + 1, 60):
        if np.linalg.norm(pts[i] - pts[j]) < 25:
            G_rand.add_edge(i, j)
configs[-1] = ("Random geometric (60 nodes)", G_rand)

ext = fn.GraphFeatureExtractor(canvas_size=256)

for idx, (label, G) in enumerate(configs):
    ax = axes[idx // 3][idx % 3]
    plot_graph(G, ax=ax, edge_width=1.5)
    feat = ext.extract(G)
    ax.set_title(f"{label}\nNodes={feat['n_node']:.0f}, Edges={feat['n_edge']:.0f}\n"
                 f"Length={feat['total_length']:.1f}, Aniso={feat['anisotropy']:.3f}", fontsize=10)

plt.suptitle("FiberNet v2.0: Graph API Structure Gallery", fontsize=15, y=1.01)
plt.tight_layout()
fig.savefig(f"{OUT}/05_structure_gallery.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: 05_structure_gallery.png")

# ============================================================
# 6. Feature Heatmap
# ============================================================
print("[6/8] Feature heatmap comparison")

networks = []
network_labels = []
for i, dx in enumerate([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]):
    gen = fn.RegularNetworkGenerator(side_length=10, num_points_per_side=1,
                                     tiling=3, perturbations=[(dx, dx*0.5)])
    G = gen.generate()
    networks.append(ext.extract(G))
    network_labels.append(f"dx={dx:.2f}")

# Select key features for heatmap
key_features = ['n_node', 'n_edge', 'total_length', 'mean_edge_len', 'len_cv',
                'deg2_count', 'deg4_count', 'degree_entropy',
                'orient_entropy', 'anisotropy', 'radius_gyration',
                'rigidity_index', 'redundancy_ratio',
                'triangle_count', 'clustering_coef']

matrix = np.array([[net[f] for f in key_features] for net in networks])

fig, ax = plt.subplots(figsize=(14, 7))
im = ax.imshow(matrix, aspect='auto', cmap='RdYlBu_r')
ax.set_xticks(range(len(key_features)))
ax.set_xticklabels(key_features, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(network_labels)))
ax.set_yticklabels(network_labels, fontsize=10)
ax.set_title("Feature Heatmap: Parametric Sweep (perturbation dx)", fontsize=13)

# Add text annotations
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        val = matrix[i, j]
        text = f"{val:.2f}" if abs(val) < 100 else f"{val:.0f}"
        ax.text(j, i, text, ha="center", va="center", color="white", fontsize=7,
                fontweight="bold")

plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
fig.savefig(f"{OUT}/06_feature_heatmap.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: 06_feature_heatmap.png")

# ============================================================
# 7. Weld Graph Detail
# ============================================================
print("[7/8] Weld graph detail view")

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Create a simple crossing graph
G_cross = nx.Graph()
G_cross.add_node('A', pos=(0, 0))
G_cross.add_node('B', pos=(100, 100))
G_cross.add_node('C', pos=(0, 100))
G_cross.add_node('D', pos=(100, 0))
G_cross.add_node('E', pos=(50, 0))
G_cross.add_node('F', pos=(50, 100))
G_cross.add_edge('A', 'B')
G_cross.add_edge('C', 'D')
G_cross.add_edge('E', 'F')

# Before weld
pos_before = nx.get_node_attributes(G_cross, 'pos')
edge_coords = []
for u, v in G_cross.edges():
    pu = np.array(pos_before[u])[:2]
    pv = np.array(pos_before[v])[:2]
    edge_coords.append([pu, pv])
lc = LineCollection(edge_coords, colors='black', linewidths=3)
axes[0].add_collection(lc)
for n, p in pos_before.items():
    axes[0].plot(p[0], p[1], 'o', color='blue', markersize=10, zorder=5)
    axes[0].annotate(str(n), (p[0], p[1]), fontsize=10, ha='center', va='bottom')
axes[0].set_aspect('equal')
axes[0].autoscale_view()
axes[0].set_title("Before Weld\n3 crossing edges, 6 nodes", fontsize=13)
axes[0].axis('off')

# After weld
G_welded = fn.weld_graph(G_cross)
pos_after = nx.get_node_attributes(G_welded, 'pos')
edge_coords = []
for u, v in G_welded.edges():
    if u in pos_after and v in pos_after:
        pu = np.array(pos_after[u])[:2]
        pv = np.array(pos_after[v])[:2]
        edge_coords.append([pu, pv])
if edge_coords:
    lc = LineCollection(edge_coords, colors='black', linewidths=3)
    axes[1].add_collection(lc)

for n, p in pos_after.items():
    if str(n).startswith("IX_"):
        axes[1].plot(p[0], p[1], 's', color='red', markersize=12, zorder=5)
        axes[1].annotate('W', (p[0], p[1]), fontsize=9, ha='center', va='bottom', color='red')
    else:
        axes[1].plot(p[0], p[1], 'o', color='blue', markersize=10, zorder=5)
        axes[1].annotate(str(n), (p[0], p[1]), fontsize=10, ha='center', va='bottom')

axes[1].set_aspect('equal')
axes[1].autoscale_view()
ix_count = len([n for n in G_welded.nodes() if str(n).startswith("IX_")])
axes[1].set_title(f"After Weld\n{G_welded.number_of_nodes()} nodes ({ix_count} weld points), {G_welded.number_of_edges()} edges", fontsize=13)
axes[1].axis('off')

plt.tight_layout()
fig.savefig(f"{OUT}/07_weld_graph_detail.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: 07_weld_graph_detail.png")

# ============================================================
# 8. With Nodes vs Without (visualization principle)
# ============================================================
print("[8/8] Visualization principle: edges-only vs with nodes")

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

G_demo = fn.RegularNetworkGenerator(side_length=10, num_points_per_side=2, tiling=4, scale_to_unit=False).generate()

plot_graph(G_demo, ax=axes[0], show_nodes=False, edge_width=2.0)
axes[0].set_title("Edges Only (Default)\nFocus on fiber structure", fontsize=13)

plot_graph(G_demo, ax=axes[1], show_nodes=True, edge_width=1.5, node_size=15, node_color='steelblue')
axes[1].set_title("With Nodes\nShows all vertices", fontsize=13)

plt.suptitle("Visualization Principle: Follow 可视化graph_json.ipynb Convention", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(f"{OUT}/08_viz_principle.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: 08_viz_principle.png")

print("\n" + "=" * 60)
print("All 8 visualization outputs generated successfully!")
print(f"Output directory: {OUT}")
print("=" * 60)
