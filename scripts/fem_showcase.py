#!/usr/bin/env python3
"""
FiberNet FEM Showcase — Publication-quality visualization
Generates dark + light theme FEM stress showcase images.

Usage:
    cd fibernet && source .venv/bin/activate
    python scripts/fem_showcase.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from pathlib import Path
import gc

import fibernet as fn
from fibernet.ml.beam_frame_fem_v6 import BeamFrameFEM_v6
from fibernet.sim.accelerated import _graph_to_arrays, _get_boundary_indices
from fibernet.viz.render import _get_theme

OUTPUT_DIR = Path(__file__).parent.parent / "output_data" / "deformation_test" / "viz"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_fem_stretch(unit, grid=(4,4), box=(2.5, 2.5), radius=0.05,
                    target_stretch=2.0, E=1e9, nu=0.3, seed=42):
    """Run FEM stretch test and return (graph, fem_result, info_dict)."""
    g = fn.pattern_2d(unit=unit, box=box, grid=grid,
                      n_pts_per_side=5, perturbation=0.40,
                      radius=radius, seed=seed)
    
    solver = BeamFrameFEM_v6(E=E, nu=nu)
    res = solver.stretch_test(g, target_stretch=target_stretch)
    
    info = {
        "unit": unit,
        "nodes": g.num_nodes,
        "edges": g.num_edges,
        "max_stress_MPa": float(np.max(res['sigma_total']) / 1e6),
        "max_axial_MPa": float(np.max(np.abs(res['sigma_axial'])) / 1e6),
        "max_bending_MPa": float(np.max(np.abs(res['sigma_bending'])) / 1e6),
    }
    
    return g, res, info


def draw_fem_panel(ax, graph, fem_result, theme_colors, cmap, norm,
                   title="", linewidth=0.8):
    """Draw a single FEM stress panel on the given axes."""
    ax.set_facecolor(theme_colors['bg'])
    ax.tick_params(colors=theme_colors['text'], labelsize=5)
    for spine in ax.spines.values():
        spine.set_color(theme_colors['grid'])
    
    pos, elements, _, _ = _graph_to_arrays(graph)
    u = fem_result['u']
    deformed = pos[:, :2] + u[:, :2]
    
    stresses = fem_result.get('sigma_total', np.array([]))
    edge_list = fem_result.get('edge_list', np.arange(len(stresses)))
    
    if len(stresses) == 0:
        return
    
    s_max = norm.vmax
    
    segments = []
    seg_colors = []
    
    for idx, e in enumerate(edge_list):
        i, j = int(elements[e, 0]), int(elements[e, 1])
        if 0 <= i < len(deformed) and 0 <= j < len(deformed):
            segments.append([deformed[i], deformed[j]])
            s = stresses[idx] if idx < len(stresses) else 0
            seg_colors.append(cmap(norm(np.clip(s, -s_max, s_max))))
    
    if segments:
        lc = LineCollection(segments, colors=seg_colors, linewidths=linewidth)
        ax.add_collection(lc)
        ax.autoscale()
    
    if title:
        ax.set_title(title, color=theme_colors['text'], fontsize=7,
                    fontweight='bold', pad=4)
    
    ax.set_aspect('equal')
    ax.axis('off')


def generate_showcase(theme_name="dark"):
    """Generate the FEM showcase image."""
    colors = _get_theme(theme_name)
    
    # ── Structure configurations ──
    units_2d = [
        ("honeycomb", 2.0),
        ("reentrant", 2.0),
        ("kagome", 2.0),
        ("triangle", 2.0),
        ("square", 2.0),
        ("diamond", 2.0),
        ("chiral", 2.0),
        ("star", 2.0),
    ]
    
    # Run all FEM simulations
    print(f"Running FEM simulations ({theme_name} theme)...")
    results_2d = []
    for unit, stretch in units_2d:
        g, res, info = run_fem_stretch(unit, target_stretch=stretch)
        results_2d.append((g, res, info))
        gc.collect()
        print(f"  {unit:12s}: σ={info['max_stress_MPa']:.0f} MPa, "
              f"axial={info['max_axial_MPa']:.0f}, bend={info['max_bending_MPa']:.0f}")
    
    # Also run compression cases for bottom row
    compress_units = [
        ("honeycomb", 0.5),
        ("reentrant", 0.5),
        ("kagome", 0.5),
        ("triangle", 0.5),
    ]
    results_compress = []
    for unit, stretch in compress_units:
        g, res, info = run_fem_stretch(unit, target_stretch=stretch)
        results_compress.append((g, res, info))
        gc.collect()
        print(f"  {unit:12s} (compress): σ={info['max_stress_MPa']:.0f} MPa")
    
    # ── Figure layout ──
    # 3 rows: stretch(8) | compress(4) + 3D(2) | analysis
    n_cols = 4
    n_rows = 3
    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows + 1.5))
    
    # Title
    fig.suptitle(
        "FiberNet v4.1 — BeamFrameFEM Showcase\n"
        f"E = 1 GPa, ν = 0.3 | Deformed structures colored by total stress",
        fontsize=14, fontweight='bold',
        color=colors['text'],
        y=0.98)
    fig.patch.set_facecolor(colors['bg'])
    
    # Global color normalization
    all_stresses = []
    for g, res, info in results_2d:
        all_stresses.extend(res['sigma_total'].tolist())
    for g, res, info in results_compress:
        all_stresses.extend(res['sigma_total'].tolist())
    
    s_max = np.percentile(np.abs(all_stresses), 95)
    norm = Normalize(vmin=0, vmax=s_max)
    cmap = cm.inferno
    
    # ── Row 1: Stretch 2x (8 structures) ──
    for idx, (g, res, info) in enumerate(results_2d):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        draw_fem_panel(ax, g, res, colors, cmap, norm,
                      title=f"{info['unit']}\nσ={info['max_stress_MPa']:.0f} MPa",
                      linewidth=0.6)
    
    # ── Row 2: Compress 0.5x (4) + extra stretch variations (4) ──
    # Compress cases
    cmap_comp = cm.coolwarm
    norm_comp = Normalize(vmin=-s_max*0.5, vmax=s_max*0.5)
    
    for idx, (g, res, info) in enumerate(results_compress):
        ax = fig.add_subplot(n_rows, n_cols, n_cols + idx + 1)
        draw_fem_panel(ax, g, res, colors, cmap_comp, norm_comp,
                      title=f"{info['unit']} (0.5x)\nσ={info['max_stress_MPa']:.0f} MPa",
                      linewidth=0.6)
    
    # Fill remaining row 2 with different radii
    for idx, radius in enumerate([0.02, 0.10, 0.20]):
        g, res, info = run_fem_stretch("honeycomb", radius=radius, target_stretch=2.0)
        ax = fig.add_subplot(n_rows, n_cols, n_cols + 4 + idx + 1)
        draw_fem_panel(ax, g, res, colors, cmap, norm,
                      title=f"honeycomb r={radius:.2f}\nσ={info['max_stress_MPa']:.0f} MPa",
                      linewidth=0.6)
        gc.collect()
    
    # Extra: nonlinear case
    g, res, info = run_fem_stretch("diamond", radius=0.10, target_stretch=2.0)
    ax = fig.add_subplot(n_rows, n_cols, n_cols + 8)
    draw_fem_panel(ax, g, res, colors, cmap, norm,
                  title=f"diamond r=0.10\nσ={info['max_stress_MPa']:.0f} MPa",
                  linewidth=0.6)
    gc.collect()
    
    # ── Row 3: Analysis plots ──
    # Plot 1: Stress comparison bar chart
    ax_bar = fig.add_subplot(n_rows, n_cols, 2*n_cols + 1)
    ax_bar.set_facecolor(colors['bg'])
    units_list = [info['unit'] for _, _, info in results_2d]
    axial_vals = [info['max_axial_MPa'] for _, _, info in results_2d]
    bending_vals = [info['max_bending_MPa'] for _, _, info in results_2d]
    x_pos = np.arange(len(units_list))
    ax_bar.bar(x_pos - 0.2, axial_vals, 0.4, label='Axial', color='#3498db', alpha=0.8)
    ax_bar.bar(x_pos + 0.2, bending_vals, 0.4, label='Bending', color='#e74c3c', alpha=0.8)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(units_list, rotation=45, ha='right', fontsize=6,
                          color=colors['text'])
    ax_bar.set_ylabel('Max Stress (MPa)', color=colors['text'], fontsize=8)
    ax_bar.set_title('Axial vs Bending Stress', color=colors['text'], fontsize=9)
    ax_bar.legend(fontsize=7, facecolor=colors['bg'],
                 labelcolor=colors['text'], edgecolor=colors['grid'])
    ax_bar.tick_params(colors=colors['text'])
    for spine in ax_bar.spines.values():
        spine.set_color(colors['grid'])
    ax_bar.grid(True, alpha=0.2, color=colors['grid'])
    
    # Plot 2: Radius effect
    ax_rad = fig.add_subplot(n_rows, n_cols, 2*n_cols + 2)
    ax_rad.set_facecolor(colors['bg'])
    radii_test = [0.02, 0.05, 0.10, 0.20]
    force_by_radius = []
    stress_by_radius = []
    for r in radii_test:
        g, res, info = run_fem_stretch("honeycomb", radius=r, target_stretch=2.0)
        stress_by_radius.append(info['max_stress_MPa'])
        gc.collect()
    
    ax_rad.plot(radii_test, stress_by_radius, 'o-', color='#e74c3c', linewidth=2, markersize=6)
    ax_rad.set_xlabel('Fiber Radius', color=colors['text'], fontsize=8)
    ax_rad.set_ylabel('Max Stress (MPa)', color=colors['text'], fontsize=8)
    ax_rad.set_title('Radius Effect on Stress', color=colors['text'], fontsize=9)
    ax_rad.tick_params(colors=colors['text'])
    for spine in ax_rad.spines.values():
        spine.set_color(colors['grid'])
    ax_rad.grid(True, alpha=0.2, color=colors['grid'])
    
    # Plot 3: Stretch ratio effect
    ax_str = fig.add_subplot(n_rows, n_cols, 2*n_cols + 3)
    ax_str.set_facecolor(colors['bg'])
    stretches = [0.5, 0.7, 1.5, 2.0]
    stress_by_stretch = []
    for s in stretches:
        g, res, info = run_fem_stretch("honeycomb", target_stretch=s)
        stress_by_stretch.append(info['max_stress_MPa'])
        gc.collect()
    ax_str.plot(stretches, stress_by_stretch, 's-', color='#9b59b6', linewidth=2, markersize=6)
    ax_str.set_xlabel('Stretch Ratio (L/L₀)', color=colors['text'], fontsize=8)
    ax_str.set_ylabel('Max Stress (MPa)', color=colors['text'], fontsize=8)
    ax_str.set_title('Stretch Effect on Stress', color=colors['text'], fontsize=9)
    ax_str.axvline(1.0, color=colors['grid'], ls='--', alpha=0.5)
    ax_str.tick_params(colors=colors['text'])
    for spine in ax_str.spines.values():
        spine.set_color(colors['grid'])
    ax_str.grid(True, alpha=0.2, color=colors['grid'])
    
    # Plot 4: Colorbar legend
    ax_cb = fig.add_subplot(n_rows, n_cols, 2*n_cols + 4)
    ax_cb.set_facecolor(colors['bg'])
    ax_cb.axis('off')
    
    # Stretch colorbar
    sm1 = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm1.set_array([])
    cb1 = plt.colorbar(sm1, ax=ax_cb, shrink=0.5, pad=0.15, location='top')
    cb1.set_label('Total Stress (Pa) — Stretch', color=colors['text'], fontsize=7)
    cb1.ax.tick_params(colors=colors['text'], labelsize=6)
    
    # Compress colorbar
    sm2 = cm.ScalarMappable(cmap=cmap_comp, norm=norm_comp)
    sm2.set_array([])
    cb2 = plt.colorbar(sm2, ax=ax_cb, shrink=0.5, pad=0.15, location='bottom')
    cb2.set_label('Total Stress (Pa) — Compress', color=colors['text'], fontsize=7)
    cb2.ax.tick_params(colors=colors['text'], labelsize=6)
    
    # Legend info text
    ax_cb.text(0.5, 0.1,
              "n_pts_per_side=5, perturbation=±0.40\n"
              "Boundary: 10% each side (rigid plate)\n"
              "BeamFrameFEM_v6 (Euler-Bernoulli beams)",
              transform=ax_cb.transAxes, ha='center', va='center',
              fontsize=6, color=colors['text'], alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    out_path = OUTPUT_DIR / f"fem_showcase_{theme_name}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=colors['bg'])
    plt.close(fig)
    print(f"  ✓ Saved: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")
    return str(out_path)


if __name__ == "__main__":
    print("=" * 60)
    print("FiberNet FEM Showcase Generator")
    print("=" * 60)
    
    dark_path = generate_showcase("dark")
    light_path = generate_showcase("light")
    
    print(f"\nDone!")
    print(f"  Dark:  {dark_path}")
    print(f"  Light: {light_path}")
