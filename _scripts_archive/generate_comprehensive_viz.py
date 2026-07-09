"""
Comprehensive visualization for FiberNet v2.0
Generates professional-quality images showing:
- Structure gallery (all generator types)
- Structure variants (parameter sweeps)
- Weld graph before/after
- Mechanics simulation (deformation, stress)
- Feature analysis
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import networkx as nx
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

OUTPUT_DIR = Path(__file__).parent / "output_viz"
OUTPUT_DIR.mkdir(exist_ok=True)

def get_graph_edges(G):
    """Extract edge coordinates from graph."""
    pos = nx.get_node_attributes(G, 'pos')
    if not pos:
        return []
    
    segments = []
    for u, v in G.edges():
        if u in pos and v in pos:
            p1 = np.array(pos[u][:2])
            p2 = np.array(pos[v][:2])
            segments.append([p1, p2])
    
    if not segments:
        return []
    
    return segments

def plot_graph_structure(G, ax, title="", color='steelblue', linewidth=1.5, 
                         show_nodes=False, node_size=20, edge_colors=None):
    """Plot graph structure on axes."""
    segments = get_graph_edges(G)
    if len(segments) == 0:
        ax.text(0.5, 0.5, "No edges", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    
    if edge_colors is None:
        edge_colors = [color] * len(segments)
    
    lc = LineCollection(segments, colors=edge_colors, linewidths=linewidth, 
                       capstyle='round')
    ax.add_collection(lc)
    
    if show_nodes:
        pos = nx.get_node_attributes(G, 'pos')
        if pos:
            x = [p[0] for p in pos.values()]
            y = [p[1] for p in pos.values()]
            ax.scatter(x, y, s=node_size, c='red', zorder=5, alpha=0.6)
    
    # Set axis limits
    all_pts = np.array(segments).reshape(-1, 2)
    margin = 0.05 * (all_pts.max() - all_pts.min())
    ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
    ax.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axis('off')

def plot_deformation(G, displacements, ax, title="", scale=100):
    """Plot deformed structure."""
    pos = nx.get_node_attributes(G, 'pos')
    if not pos or displacements is None:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    
    # Original positions
    nodes = list(pos.keys())
    orig_coords = np.array([pos[n][:2] for n in nodes])
    
    # Deformed positions
    deformed_coords = orig_coords.copy()
    for i, node in enumerate(nodes):
        if i * 2 < len(displacements):
            deformed_coords[i, 0] += displacements[i * 2] * scale
        if i * 2 + 1 < len(displacements):
            deformed_coords[i, 1] += displacements[i * 2 + 1] * scale
    
    # Plot original (light)
    segments_orig = []
    for u, v in G.edges():
        if u in pos and v in pos:
            i = nodes.index(u)
            j = nodes.index(v)
            segments_orig.append([orig_coords[i], orig_coords[j]])
    
    if len(segments_orig) > 0:
        lc_orig = LineCollection(segments_orig, colors='lightgray', linewidths=1, 
                                capstyle='round', alpha=0.5)
        ax.add_collection(lc_orig)
    
    # Plot deformed (colored by displacement magnitude)
    segments_def = []
    disp_mags = []
    for u, v in G.edges():
        if u in pos and v in pos:
            i = nodes.index(u)
            j = nodes.index(v)
            segments_def.append([deformed_coords[i], deformed_coords[j]])
            if i * 2 < len(displacements) and j * 2 < len(displacements):
                mag = np.sqrt(displacements[i*2]**2 + displacements[i*2+1]**2 + 
                             displacements[j*2]**2 + displacements[j*2+1]**2) / 2
                disp_mags.append(mag)
            else:
                disp_mags.append(0)
    
    if len(segments_def) > 0:
        norm = Normalize(vmin=0, vmax=max(disp_mags) if disp_mags else 1)
        lc_def = LineCollection(segments_def, cmap='viridis', linewidths=2,
                               capstyle='round')
        lc_def.set_array(np.array(disp_mags))
        lc_def.set_norm(norm)
        ax.add_collection(lc_def)
        
        sm = ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Displacement magnitude (scaled)', fontsize=8)
    
    all_pts = np.vstack(segments_def) if segments_def else np.vstack(segments_orig)
    margin = 0.05 * (all_pts.max() - all_pts.min())
    ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
    ax.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10, fontweight='bold')

# ============================================================
# Figure 1: Structure Gallery
# ============================================================
print("Generating Figure 1: Structure Gallery...")

from fibernet.gen.regular import RegularNetworkGenerator
from fibernet.gen.zigzag import ZigZagGenerator
from fibernet.gen.ordered import (
    square_lattice_2d, triangular_lattice_2d, honeycomb_lattice_2d, kagome_lattice_2d
)
from fibernet.gen.disordered import random_straight_2d, oriented_random_2d
from fibernet.gen.chiral import single_helix
from fibernet.gen.woven import plain_weave_2d, twill_weave_2d
from fibernet.gen.metamaterials import (
    reentrant_honeycomb_2d, star_honeycomb_2d, arrowhead_auxetic_2d
)
from fibernet.gen.bundles import parallel_bundle_2d, twisted_bundle_2d
from fibernet.graph.io import to_networkx as fn_to_nx

def _to_nx(net):
    if isinstance(net, nx.Graph):
        return net
    return fn_to_nx(net)

structures = {}

structures['Square Lattice'] = _to_nx(square_lattice_2d(grid_size=(6,6), spacing=1.0))
structures['Triangular Lattice'] = _to_nx(triangular_lattice_2d(grid_size=(6,6), spacing=1.0))
structures['Honeycomb'] = _to_nx(honeycomb_lattice_2d(grid_size=(6,6), cell_size=1.0))
structures['Kagome'] = _to_nx(kagome_lattice_2d(grid_size=(4,4), spacing=1.5))
structures['Random Network'] = _to_nx(random_straight_2d(num_fibers=40, fiber_length=8, box_size=(20,20), seed=42))
structures['Oriented (45°)'] = _to_nx(oriented_random_2d(num_fibers=40, fiber_length=8, box_size=(20,20), preferred_angle=45, angular_spread=15, seed=42))

gen = RegularNetworkGenerator(side_length=10, num_points_per_side=2, 
                             perturbations=[(0.2, -0.2), (0.2, 0.2), (-0.2, 0.2)], tiling=3)
structures['Regular (P1)'] = gen.generate()

gen = ZigZagGenerator(n_cols=3, n_rows=5, mirror_x=True, mirror_y=True)
structures['ZigZag'] = gen.generate()

structures['Helix'] = _to_nx(single_helix(helix_radius=3.0, pitch=2.0, num_turns=4.0))
structures['Plain Weave'] = _to_nx(plain_weave_2d(grid_size=(8,8), spacing=2.0, amplitude=0.3))
structures['Twill Weave'] = _to_nx(twill_weave_2d(grid_size=(8,8), spacing=2.0))
structures['Re-entrant'] = _to_nx(reentrant_honeycomb_2d(grid_size=(4,4), cell_height=5, cell_width=5))
structures['Star Lattice'] = _to_nx(star_honeycomb_2d(grid_size=(3,3), star_arm_length=3.0, num_arms=4))
structures['Arrowhead'] = _to_nx(arrowhead_auxetic_2d(grid_size=(4,4)))
structures['Parallel Bundle'] = _to_nx(parallel_bundle_2d(num_fibers=10, bundle_length=20, bundle_width=5))
structures['Twisted Bundle'] = _to_nx(twisted_bundle_2d(num_fibers=8, bundle_length=20, twist_pitch=10))

fig, axes = plt.subplots(4, 4, figsize=(16, 16))
axes = axes.flatten()

for idx, (name, G) in enumerate(structures.items()):
    if idx < len(axes):
        plot_graph_structure(G, axes[idx], title=f"{name}\n({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")

for idx in range(len(structures), len(axes)):
    axes[idx].axis('off')

plt.suptitle('FiberNet v2.0 Structure Gallery', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.995])
plt.savefig(OUTPUT_DIR / "01_structure_gallery.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved 01_structure_gallery.png")

# ============================================================
# Figure 2: Regular Network Perturbation Sweep
# ============================================================
print("Generating Figure 2: Perturbation Sweep...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

perturbations = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
for idx, p in enumerate(perturbations):
    gen = RegularNetworkGenerator(
        side_length=10, num_points_per_side=2,
        perturbations=[(p, -p), (p, p), (-p, p)] if p > 0 else [],
        tiling=3
    )
    G = gen.generate()
    plot_graph_structure(G, axes[idx], title=f"Perturbation = {p}\n{G.number_of_edges()} edges")

plt.suptitle('Regular Network Perturbation Sweep', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_perturbation_sweep.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved 02_perturbation_sweep.png")

# ============================================================
# Figure 3: ZigZag Mirror Variants
# ============================================================
print("Generating Figure 3: ZigZag Variants...")

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

mirror_configs = [
    (True, True, "Mirror X+Y"),
    (True, False, "Mirror X only"),
    (False, True, "Mirror Y only"),
    (False, False, "No mirror"),
]

for idx, (mx, my, label) in enumerate(mirror_configs):
    gen = ZigZagGenerator(n_cols=3, n_rows=5, mirror_x=mx, mirror_y=my)
    G = gen.generate()
    plot_graph_structure(G, axes[idx], title=f"{label}\n{G.number_of_nodes()} nodes")

plt.suptitle('ZigZag Network Mirror Variants', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_zigzag_variants.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved 03_zigzag_variants.png")

# ============================================================
# Figure 4: Weld Graph Before/After
# ============================================================
print("Generating Figure 4: Weld Graph...")

from fibernet.graph.weld import weld_graph

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

test_structures = {
    'Disordered Random': _to_nx(random_straight_2d(num_fibers=30, fiber_length=10, box_size=(15,15), seed=42)),
    'Re-entrant Meta': _to_nx(reentrant_honeycomb_2d(grid_size=(3,3), cell_height=5, cell_width=5)),
    'Regular (p=0.3)': RegularNetworkGenerator(side_length=10, num_points_per_side=2,
                                               perturbations=[(0.3, -0.3), (0.3, 0.3), (-0.3, 0.3)], tiling=2).generate(),
}

for idx, (name, G_orig) in enumerate(test_structures.items()):
    plot_graph_structure(G_orig, axes[idx*2], title=f"{name}\nBefore Weld\n({G_orig.number_of_nodes()} nodes, {G_orig.number_of_edges()} edges)",
                        show_nodes=True, node_size=15)
    
    G_welded = weld_graph(G_orig)
    plot_graph_structure(G_welded, axes[idx*2+1], 
                        title=f"After Weld\n({G_welded.number_of_nodes()} nodes, {G_welded.number_of_edges()} edges)",
                        show_nodes=True, node_size=15, color='darkgreen')

plt.suptitle('Weld Graph: Crossing Detection & Node Insertion', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_weld_graph.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved 04_weld_graph.png")

# ============================================================
# Figure 5: Mechanics Simulation
# ============================================================
print("Generating Figure 5: Mechanics Simulation...")

from fibernet.core.material import Material
from fibernet.sim.mechanical import FiberFEM
from fibernet.graph.io import from_networkx

fig, axes = plt.subplots(2, 2, figsize=(14, 14))
axes = axes.flatten()

material = Material(name="polymer", youngs_modulus=1e9, poissons_ratio=0.35, density=1200)

sim_cases = [
    ('Woven Plain', _to_nx(plain_weave_2d(grid_size=(6,6), spacing=2.0, amplitude=0.2)), 0.002),
    ('Square Lattice', _to_nx(square_lattice_2d(grid_size=(5,5), spacing=1.0)), 0.005),
    ('Triangular', _to_nx(triangular_lattice_2d(grid_size=(5,5), spacing=1.0)), 0.003),
    ('Random Network', _to_nx(random_straight_2d(num_fibers=30, fiber_length=8, box_size=(15,15), seed=42)), 0.001),
]

for idx, (name, G, strain_val) in enumerate(sim_cases):
    try:
        network = from_networkx(G, material=material)
        if len(network.fibers) < 2:
            axes[idx].text(0.5, 0.5, "Insufficient fibers", ha='center', va='center')
            axes[idx].set_title(name)
            continue
        
        fem = FiberFEM(network, segments_per_fiber=3)
        positions = np.array([cl.position for cl in network.crosslinks])
        
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        L0 = x_max - x_min
        
        if L0 < 1e-6:
            axes[idx].text(0.5, 0.5, "Invalid geometry", ha='center', va='center')
            axes[idx].set_title(name)
            continue
        
        tol = 0.15 * L0
        left_nodes = [i for i, p in enumerate(positions) if p[0] <= x_min + tol]
        right_nodes = [i for i, p in enumerate(positions) if p[0] >= x_max - tol]
        
        if not left_nodes or not right_nodes:
            axes[idx].text(0.5, 0.5, "Boundary issue", ha='center', va='center')
            axes[idx].set_title(name)
            continue
        
        F = np.zeros(fem.num_dof)
        for n in right_nodes:
            F[n * 6] = 1.0
        
        result = fem.solve_static(forces=F, fixed_nodes=left_nodes)
        
        plot_deformation(G, result.displacements, axes[idx], 
                        title=f"{name}\nMax disp: {result.max_displacement():.2e} m\nEnergy: {result.energy:.2e} J",
                        scale=100)
        
    except Exception as e:
        axes[idx].text(0.5, 0.5, f"Error: {str(e)[:30]}", ha='center', va='center', fontsize=8)
        axes[idx].set_title(name)

plt.suptitle('Mechanics Simulation: Uniaxial Tension', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_mechanics_deformation.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved 05_mechanics_deformation.png")

# ============================================================
# Figure 6: Stress Distribution
# ============================================================
print("Generating Figure 6: Stress Distribution...")

fig, axes = plt.subplots(2, 2, figsize=(14, 14))
axes = axes.flatten()

for idx, (name, G, strain_val) in enumerate(sim_cases):
    try:
        network = from_networkx(G, material=material)
        if len(network.fibers) < 2:
            axes[idx].text(0.5, 0.5, "Insufficient fibers", ha='center', va='center')
            axes[idx].set_title(name)
            continue
        
        fem = FiberFEM(network, segments_per_fiber=3)
        positions = np.array([cl.position for cl in network.crosslinks])
        
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        L0 = x_max - x_min
        
        if L0 < 1e-6:
            continue
        
        tol = 0.15 * L0
        left_nodes = [i for i, p in enumerate(positions) if p[0] <= x_min + tol]
        right_nodes = [i for i, p in enumerate(positions) if p[0] >= x_max - tol]
        
        if not left_nodes or not right_nodes:
            continue
        
        F = np.zeros(fem.num_dof)
        for n in right_nodes:
            F[n * 6] = 1.0
        
        result = fem.solve_static(forces=F, fixed_nodes=left_nodes)
        
        pos = nx.get_node_attributes(G, 'pos')
        segments = []
        stress_vals = []
        
        for u, v in G.edges():
            if u in pos and v in pos:
                p1 = np.array(pos[u][:2])
                p2 = np.array(pos[v][:2])
                segments.append([p1, p2])
                i = list(pos.keys()).index(u)
                j = list(pos.keys()).index(v)
                if i * 6 < len(result.displacements) and j * 6 < len(result.displacements):
                    dx = result.displacements[j*6] - result.displacements[i*6]
                    dy = result.displacements[j*6+1] - result.displacements[i*6+1]
                    L = np.linalg.norm(p2 - p1)
                    strain = np.sqrt(dx**2 + dy**2) / L if L > 0 else 0
                    stress_vals.append(strain * 1e9)
                else:
                    stress_vals.append(0)
        
        if len(segments) > 0:
            norm = Normalize(vmin=0, vmax=max(stress_vals) if stress_vals else 1)
            lc = LineCollection(segments, cmap='hot', linewidths=2, capstyle='round')
            lc.set_array(np.array(stress_vals))
            lc.set_norm(norm)
            axes[idx].add_collection(lc)
            
            sm = ScalarMappable(cmap='hot', norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=axes[idx], fraction=0.046, pad=0.04)
            cbar.set_label('Stress (Pa)', fontsize=8)
            
            all_pts = np.array(segments).reshape(-1, 2)
            margin = 0.05 * (all_pts.max() - all_pts.min())
            axes[idx].set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
            axes[idx].set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
            axes[idx].set_aspect('equal')
        
        axes[idx].set_title(f"{name}\nMax stress: {result.max_stress():.2e} Pa", fontsize=10, fontweight='bold')
        
    except Exception as e:
        axes[idx].text(0.5, 0.5, f"Error: {str(e)[:30]}", ha='center', va='center', fontsize=8)
        axes[idx].set_title(name)

plt.suptitle('Stress Distribution Under Load', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_stress_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved 06_stress_distribution.png")

# ============================================================
# Figure 7: Feature Heatmap
# ============================================================
print("Generating Figure 7: Feature Analysis...")

from fibernet.analysis.graph_features import GraphFeatureExtractor

ext = GraphFeatureExtractor(canvas_size=256, thick=5)

feature_data = {}
for name, G in list(structures.items())[:12]:
    try:
        features = ext.extract(G)
        feature_data[name] = features
    except:
        pass

if feature_data:
    feat_names = list(list(feature_data.values())[0].keys())[:50]
    struct_names = list(feature_data.keys())
    
    matrix = np.zeros((len(struct_names), len(feat_names)))
    for i, name in enumerate(struct_names):
        for j, fname in enumerate(feat_names):
            matrix[i, j] = feature_data[name].get(fname, 0)
    
    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(matrix, aspect='auto', cmap='viridis')
    
    ax.set_xticks(range(len(feat_names)))
    ax.set_xticklabels([f.replace('_', '\n') for f in feat_names], rotation=90, fontsize=7)
    ax.set_yticks(range(len(struct_names)))
    ax.set_yticklabels(struct_names, fontsize=9)
    
    plt.colorbar(im, ax=ax, label='Feature value')
    plt.suptitle('94-Dimensional Feature Space (Top 50 features)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_feature_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved 07_feature_heatmap.png")

# ============================================================
# Figure 8: Structure Statistics
# ============================================================
print("Generating Figure 8: Structure Statistics...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

stats = []
for name, G in structures.items():
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    if n_nodes > 0:
        avg_deg = sum(dict(G.degree()).values()) / n_nodes
        density = nx.density(G)
        components = nx.number_connected_components(G)
        try:
            cc = nx.average_clustering(G)
        except:
            cc = 0
        stats.append({
            'name': name,
            'nodes': n_nodes,
            'edges': n_edges,
            'avg_degree': avg_deg,
            'density': density,
            'components': components,
            'clustering': cc,
        })

if stats:
    axes[0, 0].scatter([s['nodes'] for s in stats], [s['edges'] for s in stats], 
                      s=100, alpha=0.7, c='steelblue')
    for s in stats:
        axes[0, 0].annotate(s['name'], (s['nodes'], s['edges']), 
                          fontsize=7, alpha=0.7, xytext=(5,5), textcoords='offset points')
    axes[0, 0].set_xlabel('Nodes')
    axes[0, 0].set_ylabel('Edges')
    axes[0, 0].set_title('Network Size')
    axes[0, 0].grid(True, alpha=0.3)
    
    names = [s['name'] for s in stats]
    avg_degs = [s['avg_degree'] for s in stats]
    axes[0, 1].barh(range(len(names)), avg_degs, color='coral', alpha=0.7)
    axes[0, 1].set_yticks(range(len(names)))
    axes[0, 1].set_yticklabels(names, fontsize=8)
    axes[0, 1].set_xlabel('Average Degree')
    axes[0, 1].set_title('Connectivity')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    densities = [s['density'] for s in stats]
    axes[0, 2].barh(range(len(names)), densities, color='lightgreen', alpha=0.7)
    axes[0, 2].set_yticks(range(len(names)))
    axes[0, 2].set_yticklabels(names, fontsize=8)
    axes[0, 2].set_xlabel('Density')
    axes[0, 2].set_title('Edge Density')
    axes[0, 2].grid(True, alpha=0.3, axis='x')
    
    components = [s['components'] for s in stats]
    axes[1, 0].barh(range(len(names)), components, color='gold', alpha=0.7)
    axes[1, 0].set_yticks(range(len(names)))
    axes[1, 0].set_yticklabels(names, fontsize=8)
    axes[1, 0].set_xlabel('Connected Components')
    axes[1, 0].set_title('Connectivity')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    clustering = [s['clustering'] for s in stats]
    axes[1, 1].barh(range(len(names)), clustering, color='mediumpurple', alpha=0.7)
    axes[1, 1].set_yticks(range(len(names)))
    axes[1, 1].set_yticklabels(names, fontsize=8)
    axes[1, 1].set_xlabel('Clustering Coefficient')
    axes[1, 1].set_title('Local Connectivity')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    selected = ['Square Lattice', 'Honeycomb', 'Random Network', 'Plain Weave']
    for sname in selected:
        if sname in structures:
            G = structures[sname]
            degrees = [d for _, d in G.degree()]
            axes[1, 2].hist(degrees, bins=range(max(degrees)+2), alpha=0.5, label=sname, density=True)
    
    axes[1, 2].set_xlabel('Degree')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].set_title('Degree Distribution')
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)

plt.suptitle('Structure Statistics Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "08_structure_statistics.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved 08_structure_statistics.png")

# ============================================================
# Figure 9: Woven Detail
# ============================================================
print("Generating Figure 9: Woven Detail...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

weave_configs = [
    ('Plain Weave (4x4)', plain_weave_2d(grid_size=(4,4), spacing=2.0, amplitude=0.3)),
    ('Plain Weave (8x8)', plain_weave_2d(grid_size=(8,8), spacing=2.0, amplitude=0.3)),
    ('Plain Weave (12x12)', plain_weave_2d(grid_size=(12,12), spacing=2.0, amplitude=0.3)),
    ('Twill Weave (4x4)', twill_weave_2d(grid_size=(4,4), spacing=2.0)),
    ('Twill Weave (8x8)', twill_weave_2d(grid_size=(8,8), spacing=2.0)),
    ('Twill Weave (12x12)', twill_weave_2d(grid_size=(12,12), spacing=2.0)),
]

for idx, (name, net) in enumerate(weave_configs):
    G = _to_nx(net)
    plot_graph_structure(G, axes[idx], title=name, linewidth=2.0)

plt.suptitle('Woven Structure Variants', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "09_woven_detail.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved 09_woven_detail.png")

# ============================================================
# Figure 10: Metamaterial Detail
# ============================================================
print("Generating Figure 10: Metamaterial Detail...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

meta_configs = [
    ('Re-entrant 3x3', reentrant_honeycomb_2d(grid_size=(3,3), cell_height=5, cell_width=5)),
    ('Re-entrant 5x5', reentrant_honeycomb_2d(grid_size=(5,5), cell_height=5, cell_width=5)),
    ('Re-entrant 7x7', reentrant_honeycomb_2d(grid_size=(7,7), cell_height=5, cell_width=5)),
    ('Star 2x2', star_honeycomb_2d(grid_size=(2,2), star_arm_length=3.0, num_arms=4)),
    ('Star 4x4', star_honeycomb_2d(grid_size=(4,4), star_arm_length=3.0, num_arms=4)),
    ('Arrowhead 3x3', arrowhead_auxetic_2d(grid_size=(3,3))),
]

for idx, (name, net) in enumerate(meta_configs):
    G = _to_nx(net)
    plot_graph_structure(G, axes[idx], title=name, linewidth=2.0, color='darkred')

plt.suptitle('Metamaterial Structure Variants', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "10_metamaterial_detail.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved 10_metamaterial_detail.png")

print("\n✓ All visualizations generated successfully!")
print(f"Output directory: {OUTPUT_DIR}")
