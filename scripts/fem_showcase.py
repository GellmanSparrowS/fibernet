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
from matplotlib.colors import Normalize, LinearSegmentedColormap
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from pathlib import Path
import gc

import fibernet as fn
from fibernet.ml.beam_frame_fem_v6 import BeamFrameFEM_v6
from fibernet.sim.accelerated import _graph_to_arrays, _get_boundary_indices
from fibernet.viz.render import _get_theme

OUTPUT_DIR = Path(__file__).parent.parent / "output_data" / "deformation_test" / "viz"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def make_bright_colormap(theme_name):
    """Create colormaps where even the lowest value is bright and visible."""
    if theme_name == "dark":
        # Stretch: bright cyan -> bright yellow -> bright orange/red
        # No black anywhere in the range
        colors_stretch = [
            (0.0, 0.85, 1.0),   # bright cyan (lowest stress)
            (0.3, 1.0, 0.5),    # bright green
            (1.0, 1.0, 0.2),    # bright yellow
            (1.0, 0.5, 0.1),    # bright orange
            (1.0, 0.15, 0.15),  # bright red (highest stress)
        ]
        cmap_stretch = LinearSegmentedColormap.from_list('bright_stretch', colors_stretch)

        # Compress: bright magenta -> bright yellow -> bright cyan (diverging)
        colors_compress = [
            (1.0, 0.2, 0.8),    # bright magenta (compression)
            (1.0, 0.7, 1.0),    # light pink
            (1.0, 1.0, 0.3),    # bright yellow (zero)
            (0.5, 1.0, 0.7),    # light mint
            (0.2, 0.8, 1.0),    # bright cyan (tension)
        ]
        cmap_compress = LinearSegmentedColormap.from_list('bright_compress', colors_compress)
    else:
        # Light theme: use standard colormaps
        cmap_stretch = cm.inferno
        cmap_compress = cm.coolwarm

    return cmap_stretch, cmap_compress


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
        "radius": radius,
        "stretch": target_stretch,
        "max_stress_MPa": float(np.max(res['sigma_total']) / 1e6),
        "max_axial_MPa": float(np.max(np.abs(res['sigma_axial'])) / 1e6),
        "max_bending_MPa": float(np.max(np.abs(res['sigma_bending'])) / 1e6),
    }

    return g, res, info


def draw_fem_panel(ax, graph, fem_result, theme_colors, cmap, norm,
                   title="", linewidth=2.0):
    """Draw a single FEM stress panel on the given axes.

    Uses deformed positions and colors edges by stress.
    """
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
    s_min = norm.vmin

    segments = []
    seg_colors = []

    for idx, e in enumerate(edge_list):
        i, j = int(elements[e, 0]), int(elements[e, 1])
        if 0 <= i < len(deformed) and 0 <= j < len(deformed):
            segments.append([deformed[i], deformed[j]])
            s = stresses[idx] if idx < len(stresses) else 0
            seg_colors.append(cmap(norm(np.clip(s, s_min, s_max))))

    if segments:
        lc = LineCollection(segments, color=seg_colors, linewidths=linewidth,
                           capstyle='round')
        ax.add_collection(lc)
        ax.autoscale()

    if title:
        ax.set_title(title, color=theme_colors['text'], fontsize=7,
                    fontweight='bold', pad=4)

    ax.set_aspect('equal')
    ax.axis('off')


def generate_showcase(theme_name="dark"):
    """Generate the FEM showcase image — 4 rows, no title, bright colors."""
    colors = _get_theme(theme_name)
    cmap_stretch, cmap_compress = make_bright_colormap(theme_name)

    # ── Structure configurations ──
    units_stretch = [
        ("honeycomb", 2.0),
        ("reentrant", 2.0),
        ("kagome", 2.0),
        ("triangle", 2.0),
        ("square", 2.0),
        ("diamond", 2.0),
        ("chiral", 2.0),
        ("star", 2.0),
    ]

    units_compress = [
        ("honeycomb", 0.5),
        ("reentrant", 0.5),
        ("kagome", 0.5),
        ("triangle", 0.5),
    ]

    # ── Run all FEM simulations ──
    print(f"Running FEM simulations ({theme_name} theme)...")

    results_stretch = []
    for unit, stretch in units_stretch:
        g, res, info = run_fem_stretch(unit, target_stretch=stretch)
        results_stretch.append((g, res, info))
        gc.collect()
        print(f"  stretch {unit:12s}: sigma={info['max_stress_MPa']:.0f} MPa")

    results_compress = []
    for unit, stretch in units_compress:
        g, res, info = run_fem_stretch(unit, target_stretch=stretch)
        results_compress.append((g, res, info))
        gc.collect()
        print(f"  compress {unit:12s}: sigma={info['max_stress_MPa']:.0f} MPa")

    # Radius variation
    radius_list = [0.02, 0.05, 0.10, 0.20]
    results_radius = []
    for r in radius_list:
        g, res, info = run_fem_stretch("honeycomb", radius=r, target_stretch=2.0)
        results_radius.append((g, res, info))
        gc.collect()
        print(f"  radius r={r:.2f}: sigma={info['max_stress_MPa']:.0f} MPa")

    # ── Figure layout: 4 rows × 4 cols, no title ──
    n_cols = 4
    n_rows = 4
    fig = plt.figure(figsize=(5 * n_cols, 4.5 * n_rows))
    fig.patch.set_facecolor(colors['bg'])

    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           hspace=0.08, wspace=0.08)

    # Global color normalization from stretch + compress data
    all_stresses = []
    for g, res, info in results_stretch:
        all_stresses.extend(res['sigma_total'].tolist())
    for g, res, info in results_compress:
        all_stresses.extend(res['sigma_total'].tolist())
    s_max = np.percentile(np.abs(all_stresses), 95)

    norm_stretch = Normalize(vmin=0, vmax=s_max)
    norm_compress = Normalize(vmin=-s_max * 0.5, vmax=s_max * 0.5)

    lw = 2.0 if theme_name == "dark" else 1.5

    # ── Row 1 & 2: Stretch 2x (8 structures) ──
    for idx, (g, res, info) in enumerate(results_stretch):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        draw_fem_panel(ax, g, res, colors, cmap_stretch, norm_stretch,
                      title=f"{info['unit']}\n{info['max_stress_MPa']:.0f} MPa",
                      linewidth=lw)

    # ── Row 3: Compress 0.5x (4 structures) ──
    for idx, (g, res, info) in enumerate(results_compress):
        ax = fig.add_subplot(gs[2, idx])
        draw_fem_panel(ax, g, res, colors, cmap_compress, norm_compress,
                      title=f"{info['unit']} (0.5x)\n{info['max_stress_MPa']:.0f} MPa",
                      linewidth=lw)

    # ── Row 4: Radius variation (4 honeycomb) ──
    radius_stresses = []
    for g, res, info in results_radius:
        radius_stresses.extend(res['sigma_total'].tolist())
    r_s_max = np.percentile(np.abs(radius_stresses), 95)
    norm_radius = Normalize(vmin=0, vmax=r_s_max)

    for idx, (g, res, info) in enumerate(results_radius):
        ax = fig.add_subplot(gs[3, idx])
        draw_fem_panel(ax, g, res, colors, cmap_stretch, norm_radius,
                      title=f"r={info['radius']:.2f}\n{info['max_stress_MPa']:.0f} MPa",
                      linewidth=lw)

    out_path = OUTPUT_DIR / f"fem_showcase_{theme_name}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=colors['bg'])
    plt.close(fig)
    print(f"  Saved: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")
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
