#!/usr/bin/env python3
"""
Generate comprehensive showcase images for FiberNet v3.

Includes:
- 2D unit gallery with n_pts_per_side
- Detail views with larger displacement
- 3D structures
- FEM simulation results
- ML dataset visualization
- RL environment visualization
"""

import sys
import os
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from fibernet import (
    pattern_2d, pattern_3d, list_units,
    render_graph, render_graph_3d, render_gallery,
    BeamFEM,
    generate_dataset,
    FiberNetworkEnv,
)

OUTPUT_DIR = Path("output_viz")
OUTPUT_DIR.mkdir(exist_ok=True)

def save_fig(fig, name):
    """Save figure with consistent settings."""
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ {path}")

# ============================================================================
# 1. 2D Gallery - All units with n_pts_per_side=3
# ============================================================================
print("1. Generating 2D unit gallery...")

units = list_units()
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.patch.set_facecolor('#1a1a2e')

for idx, unit in enumerate(units):
    ax = axes[idx // 4, idx % 4]
    g = pattern_2d(unit=unit, box=(10, 10), grid=(4, 4), 
                   n_pts_per_side=3, seed=42, n_internal=8)
    render_graph(g, ax=ax, theme="dark", color_by="orientation", 
                line_width=1.2, show_nodes=False)
    ax.set_title(f"{unit}\n({g.num_nodes} nodes)", color='white', fontsize=10)

# Hide extra axes
for idx in range(len(units), 12):
    axes[idx // 4, idx % 4].set_visible(False)

fig.suptitle("FiberNet 2D Metamaterials (n_pts_per_side=3)", 
             fontsize=16, color='white', y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.99])
save_fig(fig, "01_2d_gallery")

# ============================================================================
# 2. Honeycomb Detail - n_pts_per_side=5, larger displacement visible
# ============================================================================
print("2. Generating honeycomb detail...")

g = pattern_2d(unit="honeycomb", box=(10, 10), grid=(5, 5),
               n_pts_per_side=5, seed=42, n_internal=10)

fig = render_graph(g, figsize=(12, 12), theme="dark", 
                   color_by="orientation", colormap="coolwarm",
                   line_width=1.5, show_nodes=False,
                   title="Honeycomb Detail (n_pts_per_side=5)",
                   subtitle=f"{g.num_nodes} nodes, {g.num_edges} edges")
save_fig(fig, "02_honeycomb_detail")

# ============================================================================
# 3. Kagome Blueprint - n_pts_per_side=4
# ============================================================================
print("3. Generating kagome blueprint...")

g = pattern_2d(unit="kagome", box=(10, 10), grid=(4, 4),
               n_pts_per_side=4, seed=42, n_internal=8)

fig = render_graph(g, figsize=(12, 12), theme="blueprint",
                   color_by="orientation", colormap="coolwarm",
                   line_width=1.8, show_nodes=False,
                   title="Kagome Lattice (n_pts_per_side=4)",
                   subtitle=f"{g.num_nodes} nodes, {g.num_edges} edges")
save_fig(fig, "03_kagome_blueprint")

# ============================================================================
# 4. Voronoi Tessellation - n_pts_per_side=3
# ============================================================================
print("4. Generating Voronoi tessellation...")

g = pattern_2d(unit="voronoi", box=(10, 10), grid=(3, 3),
               n_pts_per_side=3, seed=42, n_internal=8,
               unit_kwargs={"n_seeds": 25})

fig = render_graph(g, figsize=(12, 12), theme="dark",
                   color_by="fiber", colormap="Set2",
                   line_width=1.5, show_nodes=False,
                   title="Voronoi Tessellation (n_pts_per_side=3)",
                   subtitle=f"{g.num_nodes} nodes, {g.num_edges} edges, 25 seeds")
save_fig(fig, "04_voronoi")

# ============================================================================
# 5. Auxetic Comparison - Honeycomb vs Reentrant
# ============================================================================
print("5. Generating auxetic comparison...")

g1 = pattern_2d(unit="honeycomb", box=(10, 10), grid=(4, 4),
                n_pts_per_side=4, seed=42, n_internal=8)
g2 = pattern_2d(unit="reentrant", box=(10, 10), grid=(4, 4),
                n_pts_per_side=4, seed=42, n_internal=8,
                unit_kwargs={"angle": 15})

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.patch.set_facecolor('#1a1a2e')

render_graph(g1, ax=axes[0], theme="dark", color_by="orientation",
            colormap="coolwarm", line_width=1.5, show_nodes=False)
axes[0].set_title("Regular Honeycomb", color='white', fontsize=14)
axes[0].text(0.5, -0.05, f"{g1.num_nodes} nodes", transform=axes[0].transAxes,
            ha='center', color='white', fontsize=11)

render_graph(g2, ax=axes[1], theme="dark", color_by="orientation",
            colormap="coolwarm", line_width=1.5, show_nodes=False)
axes[1].set_title("Reentrant (Auxetic)", color='white', fontsize=14)
axes[1].text(0.5, -0.05, f"{g2.num_nodes} nodes, angle=15°", 
            transform=axes[1].transAxes, ha='center', color='white', fontsize=11)

fig.suptitle("Regular vs Auxetic Structures", fontsize=16, color='white', y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])
save_fig(fig, "05_auxetic_comparison")

# ============================================================================
# 6. 3D Cubic - n_pts_per_side=3
# ============================================================================
print("6. Generating 3D cubic...")

g = pattern_3d(unit="cubic", box=(10, 10, 10), grid=(3, 3, 3),
               n_pts_per_side=3, seed=42, n_internal=8)

fig = render_graph_3d(g, figsize=(12, 12), theme="dark",
                      line_width=1.5, depth_alpha=True,
                      title="Cubic 3D (n_pts_per_side=3)",
                      )
save_fig(fig, "06_3d_cubic")

# ============================================================================
# 7. 3D Octet - n_pts_per_side=3
# ============================================================================
print("7. Generating 3D octet...")

g = pattern_3d(unit="octet", box=(10, 10, 10), grid=(2, 2, 2),
               n_pts_per_side=3, seed=42, n_internal=8)

fig = render_graph_3d(g, figsize=(12, 12), theme="dark",
                      line_width=1.5, depth_alpha=True,
                      title="Octet 3D (n_pts_per_side=3)",
                      )
save_fig(fig, "07_3d_octet")

# ============================================================================
# 8. FEM Simulation - Deformation visualization
# ============================================================================
print("8. Generating FEM deformation...")

g = pattern_2d(unit="honeycomb", box=(10, 10), grid=(4, 4),
               n_pts_per_side=4, seed=42, n_internal=8)

fem = BeamFEM(g, default_E=1e9, default_nu=0.3)
result = fem.uniaxial_tension(strain=0.02, deformation_scale=20)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.patch.set_facecolor('#1a1a2e')

# Original structure
render_graph(g, ax=axes[0], theme="dark", color_by="uniform",
            line_width=1.5, show_nodes=False)
axes[0].set_title("Original Structure", color='white', fontsize=14)

# Deformed structure
deformed = result.deformed_graph
render_graph(deformed, ax=axes[1], theme="dark", color_by="stress",
            color_data=result.stresses, colormap="inferno",
            line_width=1.5, show_nodes=False)
axes[1].set_title(f"Deformed (ε=0.02, E*={result.effective_youngs_modulus:.2e} Pa)", 
                 color='white', fontsize=12)

fig.suptitle("FEM Simulation: Uniaxial Tension", fontsize=16, color='white', y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])
save_fig(fig, "08_fem_deformation")

# ============================================================================
# 9. FEM Stress Field - Color by stress
# ============================================================================
print("9. Generating FEM stress field...")

g = pattern_2d(unit="reentrant", box=(10, 10), grid=(3, 3),
               n_pts_per_side=4, seed=42, n_internal=8,
               unit_kwargs={"angle": 20})

fem = BeamFEM(g, default_E=1e9, default_nu=0.3)
result = fem.uniaxial_tension(strain=0.015, deformation_scale=30)

fig = render_graph(result.deformed_graph, figsize=(12, 12), theme="dark",
                   color_by="stress", color_data=result.stresses,
                   colormap="inferno", line_width=1.8, show_nodes=False,
                   title="Reentrant Stress Field",
                   subtitle=f"ν*={result.effective_poissons_ratio:.3f} (auxetic)")
save_fig(fig, "09_fem_stress")

# ============================================================================
# 10. ML Dataset - Multiple structures with features
# ============================================================================
print("10. Generating ML dataset visualization...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.patch.set_facecolor('#1a1a2e')

configs = [
    ("honeycomb", {"n_pts_per_side": 4, "seed": 42}),
    ("kagome", {"n_pts_per_side": 4, "seed": 42}),
    ("reentrant", {"n_pts_per_side": 4, "seed": 42, "angle": 20}),
    ("chiral", {"n_pts_per_side": 4, "seed": 42}),
    ("voronoi", {"n_pts_per_side": 3, "seed": 42, "n_seeds": 20}),
    ("star", {"n_pts_per_side": 5, "seed": 42}),
]

for idx, (unit, kwargs) in enumerate(configs):
    ax = axes[idx // 3, idx % 3]
    g = pattern_2d(unit=unit, box=(10, 10), grid=(4, 4),
                   n_internal=8, unit_kwargs=kwargs)
    
    # Run FEM to get properties
    fem = BeamFEM(g, default_E=1e9, default_nu=0.3)
    result = fem.uniaxial_tension(strain=0.01)
    
    render_graph(g, ax=ax, theme="dark", color_by="orientation",
                colormap="coolwarm", line_width=1.2, show_nodes=False)
    
    title = f"{unit}\nE*={result.effective_youngs_modulus:.2e} Pa"
    ax.set_title(title, color='white', fontsize=11)

fig.suptitle("ML Training Dataset: Structures with FEM Properties", 
             fontsize=16, color='white', y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.99])
save_fig(fig, "10_ml_dataset")

# ============================================================================
# 11. RL Environment - Action space visualization
# ============================================================================
print("11. Generating RL environment visualization...")

env = FiberNetworkEnv(target_E=1e6, target_nu=-0.3)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.patch.set_facecolor('#1a1a2e')

# Sample different actions
actions = [
    ("honeycomb", 3, 0.1),
    ("kagome", 4, 0.1),
    ("reentrant", 3, 0.1),
    ("chiral", 4, 0.1),
    ("voronoi", 3, 0.1),
    ("star", 4, 0.1),
]

for idx, (unit, grid_size, radius) in enumerate(actions):
    ax = axes[idx // 3, idx % 3]
    
    action = {
        "unit_idx": list_units().index(unit) if unit in list_units() else 0,
        "grid_x": grid_size - 2,
        "grid_y": grid_size - 2,
        "radius": np.array([radius]),
    }
    
    obs, reward, terminated, truncated, info = env.step(action)
    g = info["graph"]
    
    render_graph(g, ax=ax, theme="dark", color_by="orientation",
                colormap="coolwarm", line_width=1.2, show_nodes=False)
    
    E_star = info["E_star"]
    nu_star = info["nu_star"]
    title = f"{unit}\nE*={E_star:.2e}, ν*={nu_star:.2f}\nR={reward:.2f}"
    ax.set_title(title, color='white', fontsize=10)

env.close()

fig.suptitle("RL Environment: Action Space Exploration", 
             fontsize=16, color='white', y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.99])
save_fig(fig, "11_rl_environment")

# ============================================================================
# 12. Chiral with statistics
# ============================================================================
print("12. Generating chiral with statistics...")

g = pattern_2d(unit="chiral", box=(10, 10), grid=(4, 4),
               n_pts_per_side=4, seed=42, n_internal=8)

from fibernet.viz.render import render_with_stats

fig = render_with_stats(g, figsize=(12, 12), theme="dark",
                        color_by="orientation", colormap="coolwarm",
                        line_width=1.5, show_nodes=False,
                        title="Chiral Honeycomb (n_pts_per_side=4)",
                        save_path=str(OUTPUT_DIR / "12_chiral_stats.png"))
print(f"  ✓ {OUTPUT_DIR / '12_chiral_stats.png'}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("✓ Generated 12 showcase images in output_viz/")
print("=" * 70)
print("\nKey improvements:")
print("  • Displacement magnitude: 0.3 * edge_length (was 0.05)")
print("  • Unified color scheme per structure (coolwarm colormap)")
print("  • Added Voronoi tessellation")
print("  • Added FEM deformation and stress visualization")
print("  • Added ML dataset visualization")
print("  • Added RL environment visualization")
print("  • Clean fiber rendering (no scattering)")

