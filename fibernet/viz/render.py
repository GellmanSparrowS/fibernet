"""
Publication-quality visualization for StructureGraph.

Features:
- Dark/light themes with glow effects
- Edge discretization rendering (internal points visible as smooth curves)
- Deformation visualization with displacement coloring
- Stress/strain color maps
- 2D and 3D rendering
- Comparison panels (side-by-side structures)
- Statistics overlay (node/edge count, density, etc.)
- GitHub-ready showcase quality (300 DPI, anti-aliased)

Examples
--------
>>> from fibernet.viz.render import render_graph, render_deformation, render_gallery
>>> fig = render_graph(g, theme="dark", color_by="orientation", save_path="honeycomb.png")
>>> fig = render_deformation(g, result, save_path="deformed.png")
>>> fig = render_gallery(graphs, titles, save_path="gallery.png")
"""

from __future__ import annotations

import os
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from fibernet.core.structure_graph import StructureGraph


# ======================================================================
# Theme presets
# ======================================================================

THEMES = {
    "dark": {
        "bg": "#0a0a0f",
        "fiber": "#b388ff",  # purple
        "fiber_alt": "#7c4dff",
        "node": "#ff6644",
        "text": "#d0d0d0",
        "grid": "#1a1a2a",
        "accent": "#ff4488",
        "glow": False,
    },
    "light": {
        "bg": "#fafafa",
        "fiber": "#5e35b1",  # dark purple on white
        "fiber_alt": "#7c4dff",
        "node": "#e74c3c",
        "text": "#2c3e50",
        "grid": "#eeeeee",
        "accent": "#e74c3c",
        "glow": False,
    },
    "blueprint": {
        "bg": "#0a1628",
        "fiber": "#b388ff",  # purple
        "fiber_alt": "#7c4dff",
        "node": "#ff6644",
        "text": "#6ab0ff",
        "grid": "#1a3050",
        "accent": "#ffaa00",
        "glow": False,
    },
    "publication": {
        "bg": "#ffffff",
        "fiber": "#5e35b1",  # dark purple
        "fiber_alt": "#7c4dff",
        "node": "#cc0000",
        "text": "#333333",
        "grid": "#cccccc",
        "accent": "#0066cc",
        "glow": False,
    },
}


def _get_theme(name: str) -> dict:
    if name not in THEMES:
        raise ValueError(f"Unknown theme '{name}'. Available: {list(THEMES.keys())}")
    return THEMES[name]


# ======================================================================
# Core 2D rendering
# ======================================================================

def render_graph(
    graph: StructureGraph,
    *,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10, 10),
    theme: str = "dark",
    color_by: str = "uniform",
    colormap: str = "coolwarm",
    color_data: Optional[np.ndarray] = None,
    line_width: float = 1.5,
    show_nodes: bool = False,
    node_size: float = 10,
    show_boundary: bool = False,
    title: str = "",
    subtitle: str = "",
    save_path: Optional[str] = None,
    dpi: int = 200,
    tight: bool = True,
) -> plt.Figure:
    """Render a StructureGraph as a publication-quality 2D image.

    Parameters
    ----------
    graph : StructureGraph
        The structure to render.
    ax : matplotlib Axes, optional
        Draw on existing axes. If None, creates new figure.
    figsize : tuple
        Figure size in inches.
    theme : str
        Color theme: "dark", "light", "blueprint", "publication".
    color_by : str
        Edge coloring mode:
        - "uniform": Single color
        - "orientation": Color by edge angle
        - "length": Color by edge length
        - "stress": Color by stress (requires color_data)
        - "strain": Color by strain (requires color_data)
        - "custom": User-provided color_data array
    colormap : str
        Matplotlib colormap name for non-uniform coloring.
    color_data : np.ndarray, optional
        Per-edge scalar data for coloring (M,).
    line_width : float
        Base line width for edges.
    show_nodes : bool
        Draw node markers.
    node_size : float
        Node marker size.
    show_boundary : bool
        Highlight boundary nodes.
    title : str
        Figure title.
    subtitle : str
        Subtitle text.
    save_path : str, optional
        Save figure to this path.
    dpi : int
        Resolution for saved figure.
    tight : bool
        Use tight layout.

    Returns
    -------
    matplotlib.figure.Figure
    """
    t = _get_theme(theme)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    ax.set_facecolor(t["bg"])
    fig.patch.set_facecolor(t["bg"])
    ax.set_aspect("equal")

    # Collect edge segments with internal points
    segments = []
    edge_colors = []
    cmap = plt.get_cmap(colormap)

    edges_list = list(graph.edges.values())
    n_edges = len(edges_list)

    # Compute color values
    if color_by == "orientation":
        cdata = np.array([
            np.arctan2(
                graph.nodes[e.node_j].position[1] - graph.nodes[e.node_i].position[1],
                graph.nodes[e.node_j].position[0] - graph.nodes[e.node_i].position[0],
            )
            for e in edges_list
        ])
        cdata = (cdata + np.pi) / (2 * np.pi)  # normalize to [0, 1]
    elif color_by == "length":
        cdata = graph.edge_lengths()
        cdata = (cdata - cdata.min()) / (cdata.max() - cdata.min() + 1e-12)
    elif color_by == "fiber":
        # Group edges into fibers (connected chains) and assign colors per fiber
        # Simple approach: use edge index modulo palette size
        palette_size = 5
        cdata = np.array([i % palette_size / palette_size for i in range(n_edges)])
    elif color_by in ("stress", "strain", "custom") and color_data is not None:
        cdata = color_data
        vmin, vmax = cdata.min(), cdata.max()
        if vmax - vmin > 1e-12:
            cdata = (cdata - vmin) / (vmax - vmin)
        else:
            cdata = np.zeros_like(cdata)
    else:
        cdata = None

    for i, edge in enumerate(edges_list):
        pi = graph.nodes[edge.node_i].position[:2]
        pj = graph.nodes[edge.node_j].position[:2]

        # Build polyline through internal points if available
        if edge.internal_points is not None and len(edge.internal_points) > 0:
            pts = np.vstack([pi[None, :2], edge.internal_points[:, :2], pj[None, :2]])
        else:
            pts = np.vstack([pi[None, :2], pj[None, :2]])

        # Create segments from consecutive point pairs
        for k in range(len(pts) - 1):
            segments.append([pts[k], pts[k + 1]])

        # Color
        if cdata is not None:
            color = cmap(cdata[i])
        else:
            color = mcolors.to_rgba(t["fiber"])
        edge_colors.extend([color] * (len(pts) - 1))

    if segments:
        # Main lines
        lc = LineCollection(
            segments, colors=edge_colors,
            linewidths=line_width, capstyle="round",
        )
        ax.add_collection(lc)

    # Nodes
    if show_nodes:
        pos = graph.node_positions()[:, :2]
        ax.scatter(pos[:, 0], pos[:, 1], c=t["node"], s=node_size, zorder=5, alpha=0.8)

    # Boundary highlight
    if show_boundary:
        bnd_pos = []
        for nid, node in graph.nodes.items():
            if any(node.boundary):
                bnd_pos.append(node.position[:2])
        if bnd_pos:
            bnd_pos = np.array(bnd_pos)
            ax.scatter(bnd_pos[:, 0], bnd_pos[:, 1], c=t["accent"],
                       s=node_size * 2, zorder=6, marker="o", alpha=0.6)

    # Auto-scale
    if graph.num_nodes > 0:
        bb_min, bb_max = graph.bounding_box()
        margin = (bb_max - bb_min).max() * 0.05
        ax.set_xlim(bb_min[0] - margin, bb_max[0] + margin)
        ax.set_ylim(bb_min[1] - margin, bb_max[1] + margin)

    # Styling
    ax.tick_params(colors=t["text"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(t["grid"])
        spine.set_linewidth(0.5)
    ax.grid(False)

    # Title
    if title:
        ax.set_title(title, color=t["text"], fontsize=14, fontweight="bold",
                     pad=12, fontfamily="sans-serif")
    if subtitle:
        ax.text(0.5, -0.02, subtitle, transform=ax.transAxes,
                ha="center", va="top", color=t["text"], fontsize=9,
                alpha=0.7, fontfamily="sans-serif")

    if tight:
        fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor(),
                    bbox_inches="tight", pad_inches=0.1)

    return fig


# ======================================================================
# Deformation visualization
# ======================================================================

def render_deformation(
    original: StructureGraph,
    deformed: StructureGraph,
    *,
    figsize: Tuple[float, float] = (14, 7),
    theme: str = "dark",
    color_by: str = "displacement",
    displacement_data: Optional[np.ndarray] = None,
    line_width: float = 1.5,
    title: str = "",
    save_path: Optional[str] = None,
    dpi: int = 200,
) -> plt.Figure:
    """Render side-by-side original and deformed structure.

    Parameters
    ----------
    original : StructureGraph
        Undeformed structure.
    deformed : StructureGraph
        Deformed structure (from FEM result).
    color_by : str
        "displacement" or "stress" for the deformed view.
    displacement_data : np.ndarray, optional
        (N, 3) displacement vectors for coloring.
    """
    t = _get_theme(theme)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Original (left panel)
    render_graph(original, ax=ax1, theme=theme, color_by="uniform",
                 line_width=line_width, title="Original", tight=False)

    # Deformed (right panel)
    if displacement_data is not None and color_by == "displacement":
        # Compute per-edge average displacement magnitude
        edge_data = []
        for edge in deformed.edges.values():
            ni_idx = list(deformed.nodes.keys()).index(edge.node_i)
            nj_idx = list(deformed.nodes.keys()).index(edge.node_j)
            di = np.linalg.norm(displacement_data[ni_idx, :2])
            dj = np.linalg.norm(displacement_data[nj_idx, :2])
            edge_data.append((di + dj) / 2)
        edge_data = np.array(edge_data)
    else:
        edge_data = None

    render_graph(deformed, ax=ax2, theme=theme,
                 color_by="custom" if edge_data is not None else "uniform",
                 color_data=edge_data, colormap="viridis",
                 line_width=line_width, title="Deformed", tight=False)

    if title:
        fig.suptitle(title, color=t["text"], fontsize=16, fontweight="bold", y=0.98)

    fig.patch.set_facecolor(t["bg"])
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor(),
                    bbox_inches="tight", pad_inches=0.1)

    return fig


# ======================================================================
# Gallery / comparison
# ======================================================================


def render_trajectory(
    original: 'StructureGraph',
    positions_trajectory: List[np.ndarray],
    edge_stretches: np.ndarray,
    *,
    n_frames: int = 6,
    figsize: Tuple[float, float] = (18, 4),
    theme: str = "dark",
    title: str = "",
    line_width: float = 1.2,
    save_path: Optional[str] = None,
) -> 'plt.Figure':
    """Render multi-frame trajectory visualization with stress distribution.

    Creates a grid of subplots showing the structure at different time steps,
    colored by edge stretch (stress proxy).

    Parameters
    ----------
    original : StructureGraph
        Reference undeformed structure (for node/edge topology).
    positions_trajectory : list of np.ndarray
        List of (N, 3) position arrays per frame.
    edge_stretches : np.ndarray
        Edge stretch ratio for coloring (len = num_edges).
    n_frames : int
        Number of frames to display.
    theme : str
        Color theme name.
    title : str
        Plot title.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    th = _get_theme(theme)
    n_traj = len(positions_trajectory)
    n_display = min(n_frames, n_traj)

    # Select frame indices
    if n_traj > n_display:
        frame_idx = np.linspace(0, n_traj - 1, n_display, dtype=int)
    else:
        frame_idx = np.arange(n_traj)

    ncols = min(n_display, 3)
    nrows = (n_display + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0] * ncols / 3, figsize[1] * nrows),
                              squeeze=False)
    fig.patch.set_facecolor(th['bg'])

    # Get edge topology from original
    edge_pairs = [(e.node_i, e.node_j) for e in original.edges.values()]

    # Compute stress colors from edge_stretches
    min_stretch = float(np.percentile(edge_stretches, 1))
    max_stretch = float(np.percentile(edge_stretches, 99))
    norm = Normalize(vmin=min_stretch, vmax=max_stretch)

    # Use a colormap that looks good on dark bg
    cmap = plt.cm.viridis

    all_x, all_y = [], []
    for idx in frame_idx:
        pos = positions_trajectory[idx]
        for u, v in edge_pairs:
            all_x.extend([pos[u, 0], pos[v, 0]])
            all_y.extend([pos[u, 1], pos[v, 1]])

    x_all = np.array(all_x)
    y_all = np.array(all_y)
    pad = max((np.max(x_all) - np.min(x_all)), (np.max(y_all) - np.min(y_all))) * 0.05
    x_min, x_max = np.min(x_all) - pad, np.max(x_all) + pad
    y_min, y_max = np.min(y_all) - pad, np.max(y_all) + pad

    for plot_idx, frame_i in enumerate(frame_idx):
        ax = axes[plot_idx // ncols][plot_idx % ncols]
        ax.set_facecolor(th['bg'])

        pos = positions_trajectory[frame_i]
        segments = []
        stretch_vals = []
        for ei, (u, v) in enumerate(edge_pairs):
            segments.append([[pos[u, 0], pos[u, 1]], [pos[v, 0], pos[v, 1]]])
            stretch_vals.append(edge_stretches[ei])

        lc = LineCollection(segments, cmap=cmap, norm=norm,
                           linewidths=line_width, edgecolors='none')
        lc.set_array(np.array(stretch_vals))
        ax.add_collection(lc)

        # Draw nodes as small dots
        ax.scatter(pos[:, 0], pos[:, 1], c=th['node'], s=2, zorder=2, edgecolors='none')

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_title(f"Frame {frame_i + 1}/{n_traj}", color=th['text'], fontsize=10)
        ax.tick_params(colors='#888', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#333')

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Edge Stretch", color=th['text'], fontsize=10)
    cbar.ax.tick_params(colors='#888')

    if title:
        fig.suptitle(title, color=th['text'], fontsize=14, y=1.01)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=th['bg'])
    return fig


def render_gallery(
    graphs: List[StructureGraph],
    titles: Optional[List[str]] = None,
    *,
    ncols: int = 4,
    figsize_per_cell: Tuple[float, float] = (5, 5),
    theme: str = "dark",
    color_by: str = "uniform",
    line_width: float = 1.2,
    show_nodes: bool = False,
    suptitle: str = "",
    save_path: Optional[str] = None,
    dpi: int = 200,
) -> plt.Figure:
    """Render a gallery of structures in a grid layout.

    Parameters
    ----------
    graphs : list of StructureGraph
        Structures to render.
    titles : list of str, optional
        Title for each structure.
    ncols : int
        Number of columns in the grid.
    figsize_per_cell : tuple
        Size of each cell in inches.
    """
    t = _get_theme(theme)
    n = len(graphs)
    nrows = (n + ncols - 1) // ncols

    fw = figsize_per_cell[0] * ncols
    fh = figsize_per_cell[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh))

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[None, :]
    elif ncols == 1:
        axes = axes[:, None]

    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            ax = axes[i, j]
            ax.set_facecolor(t["bg"])
            ax.set_aspect("equal")

            if idx < n:
                title = titles[idx] if titles and idx < len(titles) else f"#{idx}"
                render_graph(
                    graphs[idx], ax=ax, theme=theme, color_by=color_by,
                    line_width=line_width, show_nodes=show_nodes,
                    title=title, tight=False,
                )
            else:
                ax.axis("off")

    for ax_row in axes:
        for ax in ax_row:
            ax.tick_params(colors=t["text"], labelsize=6)
            for spine in ax.spines.values():
                spine.set_color(t["grid"])

    if suptitle:
        fig.suptitle(suptitle, color=t["text"], fontsize=18, fontweight="bold", y=1.01)

    fig.patch.set_facecolor(t["bg"])
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor(),
                    bbox_inches="tight", pad_inches=0.2)

    return fig


# ======================================================================
# 3D rendering
# ======================================================================

def render_graph_3d(
    graph: StructureGraph,
    *,
    figsize: Tuple[float, float] = (10, 10),
    theme: str = "dark",
    line_width: float = 1.0,
    depth_alpha: bool = True,
    title: str = "",
    save_path: Optional[str] = None,
    dpi: int = 200,
    elevation: float = 25,
    azimuth: float = -60,
) -> plt.Figure:
    """Render a 3D StructureGraph.

    Parameters
    ----------
    graph : StructureGraph
        3D structure.
    depth_alpha : bool
        Apply depth-based opacity for 3D perception.
    elevation, azimuth : float
        Camera angle in degrees.
    """
    t = _get_theme(theme)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(t["bg"])
    fig.patch.set_facecolor(t["bg"])

    # Collect 3D segments
    segments_3d = []
    for edge in graph.edges.values():
        pi = graph.nodes[edge.node_i].position
        pj = graph.nodes[edge.node_j].position
        if edge.internal_points is not None and len(edge.internal_points) > 0:
            pts = np.vstack([pi[None, :], edge.internal_points, pj[None, :]])
        else:
            pts = np.vstack([pi[None, :], pj[None, :]])
        for k in range(len(pts) - 1):
            segments_3d.append([pts[k], pts[k + 1]])

    if segments_3d:
        # Depth-based coloring
        if depth_alpha:
            z_vals = [np.mean([s[0][2], s[1][2]]) for s in segments_3d]
            z_arr = np.array(z_vals)
            if z_arr.max() - z_arr.min() > 1e-12:
                z_norm = (z_arr - z_arr.min()) / (z_arr.max() - z_arr.min())
            else:
                z_norm = np.ones_like(z_arr) * 0.5
            alphas = 0.3 + 0.7 * z_norm
            colors = [(*mcolors.to_rgba(t["fiber"])[:3], a) for a in alphas]
        else:
            colors = [mcolors.to_rgba(t["fiber"])] * len(segments_3d)

        lc3d = Line3DCollection(segments_3d, colors=colors, linewidths=line_width)
        ax.add_collection3d(lc3d)

    # Auto-scale
    if graph.num_nodes > 0:
        pos = graph.node_positions()
        for ax_obj, dim in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
            mn, mx = pos[:, dim].min(), pos[:, dim].max()
            margin = (mx - mn) * 0.05
            ax_obj(mn - margin, mx + margin)

    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_axis_off()

    if title:
        ax.set_title(title, color=t["text"], fontsize=14, fontweight="bold", pad=10)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor(),
                    bbox_inches="tight", pad_inches=0.1)

    return fig


# ======================================================================
# Statistics overlay
# ======================================================================

def render_with_stats(
    graph: StructureGraph,
    *,
    theme: str = "dark",
    figsize: Tuple[float, float] = (10, 10),
    save_path: Optional[str] = None,
    dpi: int = 200,
    **kwargs,
) -> plt.Figure:
    """Render structure with statistics text overlay."""
    t = _get_theme(theme)
    fig = render_graph(graph, theme=theme, figsize=figsize, **kwargs)

    # Compute stats
    bb_min, bb_max = graph.bounding_box()
    span = bb_max - bb_min
    total_len = graph.total_edge_length()
    n_comp = len(graph.connected_components())

    stats = (
        f"Nodes: {graph.num_nodes}  |  Edges: {graph.num_edges}\n"
        f"Components: {n_comp}  |  Total length: {total_len:.1f}\n"
        f"Box: {span[0]:.1f} × {span[1]:.1f}"
    )
    if graph.dimension == 3:
        stats += f" × {span[2]:.1f}"

    fig.text(0.02, 0.02, stats, color=t["text"], fontsize=8,
             fontfamily="monospace", alpha=0.8,
             bbox=dict(boxstyle="round,pad=0.3", facecolor=t["bg"], edgecolor=t["grid"], alpha=0.9))

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor(),
                    bbox_inches="tight", pad_inches=0.1)

    return fig


# ======================================================================
# 3D Visualization Enhancements
# ======================================================================


def render_deformation_3d(
    graph,
    sim_result=None,
    *,
    figsize=(14, 7),
    theme="dark",
    line_width=1.0,
    displacement_scale=1.0,
    title="",
    save_path=None,
    dpi=200,
    elevation=25,
    azimuth=-60,
):
    """Render 3D structure before and after deformation.

    Parameters
    ----------
    graph : StructureGraph
        Original 3D structure.
    sim_result : SimResult, optional
        Simulation result with displacements.
    displacement_scale : float
        Scale factor for displacement visualization.
    """
    t = _get_theme(theme)
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(t["bg"])

    # Left: Original
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.set_facecolor(t["bg"])
    _render_3d_segments(ax1, graph, t, line_width, elevation, azimuth, "Original")

    # Right: Deformed
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.set_facecolor(t["bg"])

    if sim_result is not None and hasattr(sim_result, 'displacements'):
        disp = sim_result.displacements
        # Create deformed graph
        deformed = graph.copy()
        node_ids = sorted(deformed.nodes.keys())
        for idx, nid in enumerate(node_ids):
            if idx < len(disp):
                pos = deformed.nodes[nid].position.copy()
                pos += displacement_scale * disp[idx]
                deformed.nodes[nid].position = pos

        # Color by displacement magnitude
        disp_mags = np.linalg.norm(disp[:, :3], axis=1) if disp.shape[1] >= 3 else np.linalg.norm(disp, axis=1)
        _render_3d_colored(ax2, deformed, disp_mags, t, line_width, elevation, azimuth, "Deformed")
    else:
        _render_3d_segments(ax2, graph, t, line_width, elevation, azimuth, "Deformed")

    # Add colorbar for displacement
    if sim_result is not None and hasattr(sim_result, "displacements"):
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize as _Norm
        disp_mags = np.linalg.norm(sim_result.displacements[:, :3], axis=1)
        d_vmin = float(np.min(disp_mags)) + 0.05 * (float(np.max(disp_mags)) - float(np.min(disp_mags)))
        d_vmax = float(np.max(disp_mags))
        if d_vmax - d_vmin < 1e-12:
            d_vmax = d_vmin + 1
        sm = ScalarMappable(cmap=plt.cm.viridis, norm=_Norm(vmin=d_vmin, vmax=d_vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=[ax1, ax2], shrink=0.6, pad=0.02, aspect=30)
        cbar.set_label("Displacement", color=t["text"], fontsize=11)
        cbar.ax.yaxis.label.set_color(t["text"])
        cbar.ax.tick_params(colors=t["text"])
        cbar.outline.set_edgecolor(t["text"])

    if title:
        fig.suptitle(title, color=t["text"], fontsize=14, fontweight="bold", y=0.98)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor(),
                    bbox_inches="tight", pad_inches=0.1)
    return fig


def _render_3d_segments(ax, graph, theme, line_width, elev, azim, title):
    """Helper to render 3D segments on a given axes."""
    segments_3d = []
    for edge in graph.edges.values():
        pi = graph.nodes[edge.node_i].position
        pj = graph.nodes[edge.node_j].position
        if edge.internal_points is not None and len(edge.internal_points) > 0:
            pts = np.vstack([pi[None, :], edge.internal_points, pj[None, :]])
        else:
            pts = np.vstack([pi[None, :], pj[None, :]])
        for k in range(len(pts) - 1):
            segments_3d.append([pts[k], pts[k + 1]])

    if segments_3d:
        z_vals = [np.mean([s[0][2], s[1][2]]) for s in segments_3d]
        z_arr = np.array(z_vals)
        if z_arr.max() - z_arr.min() > 1e-12:
            z_norm = (z_arr - z_arr.min()) / (z_arr.max() - z_arr.min())
        else:
            z_norm = np.ones_like(z_arr) * 0.5
        alphas = 0.3 + 0.7 * z_norm
        colors = [(*mcolors.to_rgba(theme["fiber"])[:3], a) for a in alphas]
        lc3d = Line3DCollection(segments_3d, colors=colors, linewidths=line_width)
        ax.add_collection3d(lc3d)

    if graph.num_nodes > 0:
        pos = graph.node_positions()
        for setter, dim in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
            mn, mx = pos[:, dim].min(), pos[:, dim].max()
            margin = (mx - mn) * 0.05
            setter(mn - margin, mx + margin)

    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    if title:
        ax.set_title(title, color=theme["text"], fontsize=12, pad=5)


def _render_3d_colored(ax, graph, node_values, theme, line_width, elev, azim, title):
    """Helper to render 3D structure colored by node values."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    cmap = plt.cm.viridis
    if len(node_values) > 0:
        vmin, vmax = np.min(node_values), np.max(node_values)
    else:
        vmin, vmax = 0, 1
    if vmax - vmin < 1e-12:
        vmax = vmin + 1
    # Offset vmin by 5% to skip dark end of colormap on dark backgrounds
    vmin = vmin + 0.05 * (vmax - vmin)
    norm = Normalize(vmin=vmin, vmax=vmax)

    node_ids = sorted(graph.nodes.keys())
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    segments_3d = []
    colors = []
    for edge in graph.edges.values():
        pi = graph.nodes[edge.node_i].position
        pj = graph.nodes[edge.node_j].position
        pts = np.vstack([pi[None, :], pj[None, :]])
        
        i_idx = node_to_idx.get(edge.node_i, 0)
        j_idx = node_to_idx.get(edge.node_j, 0)
        val = (node_values[i_idx] + node_values[j_idx]) / 2 if i_idx < len(node_values) and j_idx < len(node_values) else 0
        color = cmap(norm(val))
        
        for k in range(len(pts) - 1):
            segments_3d.append([pts[k], pts[k + 1]])
            colors.append(color)

    if segments_3d:
        lc3d = Line3DCollection(segments_3d, colors=colors, linewidths=line_width)
        ax.add_collection3d(lc3d)

    if graph.num_nodes > 0:
        pos = graph.node_positions()
        for setter, dim in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
            mn, mx = pos[:, dim].min(), pos[:, dim].max()
            margin = (mx - mn) * 0.05
            setter(mn - margin, mx + margin)

    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    if title:
        ax.set_title(title, color=theme["text"], fontsize=12, pad=5)
    
    # Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.05)
    cbar.ax.yaxis.label.set_color(theme["text"])


def render_trajectory_3d(
    graph,
    trajectory=None,
    *,
    sim_result=None,
    save_dir=None,
    save_path=None,
    n_frames=None,
    figsize=None,
    ncols=None,
    theme="dark",
    line_width=1.0,
    elevation=25,
    azimuth=-60,
    dpi=150,
    color_by="displacement",
):
    """Render multi-frame 3D trajectory in ONE combined figure.

    Parameters
    ----------
    graph : StructureGraph
        Original 3D structure.
    trajectory : list of np.ndarray, or None
        Position arrays per frame. If None, uses sim_result.positions_trajectory.
    sim_result : SimResult, optional
        If provided and trajectory is None, extracts positions_trajectory.
    save_dir : str, optional
        If given, also save individual frames to this directory.
    save_path : str, optional
        Save the combined figure to this path.
    n_frames : int, optional
        Number of frames to show (subsample if more available).
    ncols : int, optional
        Number of columns. Default: min(n_frames, 4).
    figsize : tuple, optional
        Figure size. Auto-calculated if None.
    color_by : str
        "displacement" or "stretch".

    Returns
    -------
    plt.Figure
        One combined figure with all frames as subplots.
    """
    t = _get_theme(theme)

    # Build frame list
    frames = []
    if trajectory is not None:
        frames = list(trajectory)
    elif sim_result is not None:
        if hasattr(sim_result, 'positions_trajectory') and sim_result.positions_trajectory:
            frames = list(sim_result.positions_trajectory)
        elif hasattr(sim_result, 'deformed_positions') and sim_result.deformed_positions is not None:
            frames = [sim_result.deformed_positions]

    if not frames:
        fig = plt.figure(figsize=(8, 6))
        fig.patch.set_facecolor(t["bg"])
        fig.text(0.5, 0.5, "No trajectory data", ha="center", va="center",
                color=t["text"], fontsize=16)
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor())
        return fig

    # Subsample
    if n_frames and len(frames) > n_frames:
        indices = np.linspace(0, len(frames) - 1, n_frames, dtype=int)
    else:
        indices = list(range(len(frames)))

    n_show = len(indices)
    if ncols is None:
        ncols = min(n_show, 4)
    nrows = (n_show + ncols - 1) // ncols

    if figsize is None:
        figsize = (5 * ncols, 4.5 * nrows)

    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(t["bg"])

    # Original positions for displacement coloring
    node_ids = sorted(graph.nodes.keys())
    pos_orig = np.array([graph.nodes[nid].position for nid in node_ids])

    # Compute global displacement range for consistent coloring
    all_disp_mags = []
    for frame_idx in indices:
        cur_pos = frames[frame_idx]
        if isinstance(cur_pos, np.ndarray) and cur_pos.shape[0] == len(node_ids):
            dm = np.linalg.norm(cur_pos - pos_orig, axis=1)
            all_disp_mags.extend(dm.tolist())
    global_vmax = max(all_disp_mags) if all_disp_mags else 1.0
    if global_vmax < 1e-12:
        global_vmax = 1.0

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=0, vmax=global_vmax)

    for plot_idx, frame_idx in enumerate(indices):
        cur_pos = frames[frame_idx]
        ax = fig.add_subplot(nrows, ncols, plot_idx + 1, projection="3d")
        ax.set_facecolor(t["bg"])

        # Build deformed graph
        deformed = graph.copy()
        if isinstance(cur_pos, np.ndarray) and cur_pos.shape[0] == len(node_ids):
            for idx, nid in enumerate(node_ids):
                deformed.nodes[nid].position = cur_pos[idx].copy()
            disp_mags = np.linalg.norm(cur_pos - pos_orig, axis=1)
        else:
            disp_mags = np.zeros(len(node_ids))

        # Render with global color scale
        _render_3d_colored_global(ax, deformed, disp_mags, norm, t, line_width,
                                  elevation, azimuth, f"Frame {frame_idx}")

        # Save individual frame if requested
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            fp = os.path.join(save_dir, f"frame_{frame_idx:04d}.png")
            # Save just this subplot as a separate figure
            frame_fig = plt.figure(figsize=(8, 8))
            frame_fig.patch.set_facecolor(t["bg"])
            fax = frame_fig.add_subplot(111, projection="3d")
            fax.set_facecolor(t["bg"])
            _render_3d_colored_global(fax, deformed, disp_mags, norm, t, line_width,
                                      elevation, azimuth, f"Frame {frame_idx}")
            frame_fig.savefig(fp, dpi=dpi, facecolor=frame_fig.get_facecolor(),
                            bbox_inches="tight", pad_inches=0.1)
            plt.close(frame_fig)

    # Hide empty subplots
    for i in range(n_show, nrows * ncols):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        ax.set_visible(False)

    # Add shared colorbar
    cmap_obj = plt.cm.viridis
    sm = ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes[:n_show], shrink=0.6, pad=0.02, aspect=30)
    cbar.set_label("Displacement", color=t["text"], fontsize=11)
    cbar.ax.yaxis.label.set_color(t["text"])
    cbar.ax.tick_params(colors=t["text"])

    fig.suptitle("Trajectory", color=t["text"], fontsize=14, fontweight="bold", y=0.98)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor(),
                    bbox_inches="tight", pad_inches=0.1)

    return fig


def _render_3d_colored_global(ax, graph, node_values, norm, theme, line_width, elev, azim, title):
    """Render 3D structure with a shared global color normalization."""
    from matplotlib.cm import ScalarMappable
    cmap_obj = plt.cm.viridis

    node_ids = sorted(graph.nodes.keys())
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    segments_3d = []
    colors = []
    for edge in graph.edges.values():
        pi = graph.nodes[edge.node_i].position
        pj = graph.nodes[edge.node_j].position
        pts = np.vstack([pi[None, :], pj[None, :]])

        i_idx = node_to_idx.get(edge.node_i, 0)
        j_idx = node_to_idx.get(edge.node_j, 0)
        val = (node_values[i_idx] + node_values[j_idx]) / 2 if (
            i_idx < len(node_values) and j_idx < len(node_values)
        ) else 0
        color = cmap_obj(norm(val))

        for k in range(len(pts) - 1):
            segments_3d.append([pts[k], pts[k + 1]])
            colors.append(color)

    if segments_3d:
        lc3d = Line3DCollection(segments_3d, colors=colors, linewidths=line_width)
        ax.add_collection3d(lc3d)

    if graph.num_nodes > 0:
        pos = graph.node_positions()
        for setter, dim in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
            mn, mx = pos[:, dim].min(), pos[:, dim].max()
            margin = (mx - mn) * 0.05
            setter(mn - margin, mx + margin)

    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    if title:
        ax.set_title(title, color=theme["text"], fontsize=10, pad=3)






def render_gallery_3d(
    graphs,
    titles=None,
    *,
    ncols=3,
    figsize=None,
    theme="dark",
    line_width=1.0,
    elevation=25,
    azimuth=-60,
    save_path=None,
    dpi=200,
):
    """Render multiple 3D structures in a gallery layout.

    Parameters
    ----------
    graphs : list of StructureGraph
        3D structures to display.
    titles : list of str, optional
        Title for each structure.
    ncols : int
        Number of columns in the gallery.
    """
    t = _get_theme(theme)
    n = len(graphs)
    nrows = (n + ncols - 1) // ncols

    if figsize is None:
        figsize = (5 * ncols, 5 * nrows)

    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(t["bg"])

    for idx, graph in enumerate(graphs):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")
        ax.set_facecolor(t["bg"])
        title = titles[idx] if titles and idx < len(titles) else f"Structure {idx + 1}"
        _render_3d_segments(ax, graph, t, line_width, elevation, azimuth, title)

    # Hide empty subplots
    for idx in range(n, nrows * ncols):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")
        ax.set_visible(False)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor(),
                    bbox_inches="tight", pad_inches=0.1)
    return fig

def render_stress_3d(
    graph,
    sim_result,
    *,
    figsize=(10, 10),
    theme="dark",
    line_width=1.5,
    color_by="force",
    cmap="RdYlBu_r",
    title=None,
    save_path=None,
    dpi=200,
    elevation=25,
    azimuth=-60,
):
    """Render 3D structure colored by stress/force/stretch.

    Parameters
    ----------
    graph : StructureGraph
        Original 3D structure.
    sim_result : SimResult
        Simulation result with edge_forces and/or edge_stretches.
    color_by : str
        "force" (edge axial force), "stretch" (L/L0 ratio), or "displacement".
    cmap : str
        Matplotlib colormap name.
    """
    t = _get_theme(theme)
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(t["bg"])
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(t["bg"])

    node_ids = sorted(graph.nodes.keys())
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    # Determine per-edge values for coloring
    if color_by == "force" and sim_result.edge_forces is not None:
        edge_values = np.abs(sim_result.edge_forces)
        label = "Axial Force |F|"
    elif color_by == "stretch" and sim_result.edge_stretches is not None:
        edge_values = sim_result.edge_stretches
        label = "Stretch Ratio L/L₀"
    elif sim_result.displacements is not None:
        edge_values = None  # use displacement magnitude
        label = "Displacement"
    else:
        edge_values = None
        label = "Displacement"

    colormap = plt.get_cmap(cmap)

    # Get deformed positions
    if sim_result.deformed_positions is not None:
        deformed = graph.copy()
        for idx, nid in enumerate(node_ids):
            if idx < len(sim_result.deformed_positions):
                deformed.nodes[nid].position = sim_result.deformed_positions[idx].copy()
    else:
        deformed = graph

    segments_3d = []
    colors = []

    edges_list = list(deformed.edges.values())
    for ei, edge in enumerate(edges_list):
        pi = deformed.nodes[edge.node_i].position
        pj = deformed.nodes[edge.node_j].position

        if edge.internal_points is not None and len(edge.internal_points) > 0:
            pts = np.vstack([pi[None, :], edge.internal_points, pj[None, :]])
        else:
            pts = np.vstack([pi[None, :], pj[None, :]])

        if edge_values is not None and ei < len(edge_values):
            val = edge_values[ei]
        elif sim_result.displacements is not None:
            i_idx = node_to_idx.get(edge.node_i, 0)
            j_idx = node_to_idx.get(edge.node_j, 0)
            di = np.linalg.norm(sim_result.displacements[i_idx, :3])
            dj = np.linalg.norm(sim_result.displacements[j_idx, :3])
            val = (di + dj) / 2
        else:
            val = 0.0

        for k in range(len(pts) - 1):
            segments_3d.append((pts[k], pts[k + 1], val))

    if not segments_3d:
        if save_path:
            fig.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor())
        return fig

    vals = np.array([s[2] for s in segments_3d])
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if vmax - vmin < 1e-12:
        vmax = vmin + 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)

    for p0, p1, val in segments_3d:
        c = colormap(norm(val))
        segments_3d_only = [[p0, p1]]
        lc = Line3DCollection(segments_3d_only, colors=[c], linewidths=line_width)
        ax.add_collection3d(lc)

    # Auto-scale
    if deformed.num_nodes > 0:
        pos = deformed.node_positions()
        for setter, dim in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
            mn, mx = pos[:, dim].min(), pos[:, dim].max()
            margin = (mx - mn) * 0.05
            setter(mn - margin, mx + margin)

    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_axis_off()

    # Colorbar
    sm = ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.05)
    cbar.set_label(label, color=t["text"], fontsize=10)
    cbar.ax.yaxis.label.set_color(t["text"])
    cbar.ax.tick_params(colors=t["text"])
    # Ensure outline is visible on dark backgrounds
    cbar.outline.set_edgecolor(t["text"])

    if title:
        ax.set_title(title, color=t["text"], fontsize=14, fontweight="bold", pad=10)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor(),
                    bbox_inches="tight", pad_inches=0.1)
    return fig


def render_comparison_3d(
    graph,
    sim_result,
    *,
    figsize=(16, 8),
    theme="dark",
    line_width=1.0,
    displacement_scale=1.0,
    title=None,
    save_path=None,
    dpi=200,
    elevation=25,
    azimuth=-60,
):
    """Render 3-panel comparison: original, deformed overlay, stress.

    Parameters
    ----------
    graph : StructureGraph
        Original 3D structure.
    sim_result : SimResult
        Simulation result.
    displacement_scale : float
        Scale factor for deformed view (1.0 = actual deformation).
    """
    t = _get_theme(theme)
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(t["bg"])

    # Panel 1: Original
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.set_facecolor(t["bg"])
    _render_3d_segments(ax1, graph, t, line_width, elevation, azimuth, "Original")

    # Panel 2: Deformed (displacement colored)
    ax2 = fig.add_subplot(132, projection="3d")
    ax2.set_facecolor(t["bg"])
    if sim_result.deformed_positions is not None:
        deformed = graph.copy()
        node_ids = sorted(graph.nodes.keys())
        for idx, nid in enumerate(node_ids):
            if idx < len(sim_result.deformed_positions):
                orig = graph.nodes[nid].position
                new = sim_result.deformed_positions[idx]
                # Scale displacement
                deformed.nodes[nid].position = orig + displacement_scale * (new - orig)

        disp_mags = np.linalg.norm(sim_result.displacements[:, :3], axis=1)
        _render_3d_colored(ax2, deformed, disp_mags, t, line_width, elevation, azimuth,
                          "Deformed (displacement)")
    else:
        _render_3d_segments(ax2, graph, t, line_width, elevation, azimuth, "Deformed")

    # Panel 3: Stress
    ax3 = fig.add_subplot(133, projection="3d")
    ax3.set_facecolor(t["bg"])
    if sim_result.edge_forces is not None:
        # Render stress on deformed shape
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        colormap = plt.get_cmap("RdYlBu_r")
        edge_forces = np.abs(sim_result.edge_forces)
        vmin, vmax = float(np.min(edge_forces)), float(np.max(edge_forces))
        if vmax - vmin < 1e-12:
            vmax = vmin + 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)

        deformed = graph.copy()
        node_ids = sorted(graph.nodes.keys())
        if sim_result.deformed_positions is not None:
            for idx, nid in enumerate(node_ids):
                if idx < len(sim_result.deformed_positions):
                    deformed.nodes[nid].position = sim_result.deformed_positions[idx].copy()

        edges_list = list(deformed.edges.values())
        segments_3d = []
        seg_colors = []
        for ei, edge in enumerate(edges_list):
            pi = deformed.nodes[edge.node_i].position
            pj = deformed.nodes[edge.node_j].position
            val = edge_forces[ei] if ei < len(edge_forces) else 0
            c = colormap(norm(val))
            segments_3d.append([pi, pj])
            seg_colors.append(c)

        if segments_3d:
            lc = Line3DCollection(segments_3d, colors=seg_colors, linewidths=line_width * 1.2)
            ax3.add_collection3d(lc)

        if deformed.num_nodes > 0:
            pos = deformed.node_positions()
            for setter, dim in [(ax3.set_xlim, 0), (ax3.set_ylim, 1), (ax3.set_zlim, 2)]:
                mn, mx = pos[:, dim].min(), pos[:, dim].max()
                margin = (mx - mn) * 0.05
                setter(mn - margin, mx + margin)

        ax3.view_init(elev=elevation, azim=azimuth)
        ax3.set_axis_off()
        ax3.set_title("Stress |F|", color=t["text"], fontsize=12, pad=5)

        sm = ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax3, shrink=0.6, pad=0.05)
        cbar.ax.yaxis.label.set_color(t["text"])
        cbar.ax.tick_params(colors=t["text"])
        cbar.outline.set_edgecolor(t["text"])
    else:
        _render_3d_segments(ax3, graph, t, line_width, elevation, azimuth, "Stress (N/A)")

    if title:
        fig.suptitle(title, color=t["text"], fontsize=14, fontweight="bold", y=0.98)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor(),
                    bbox_inches="tight", pad_inches=0.1)
    return fig


def render_multi_angle_3d(
    graph,
    sim_result=None,
    *,
    figsize=(16, 12),
    theme="dark",
    line_width=1.0,
    angles=None,
    title=None,
    save_path=None,
    dpi=200,
):
    """Render 3D structure from multiple viewing angles.

    Parameters
    ----------
    graph : StructureGraph
        3D structure.
    sim_result : SimResult, optional
        If provided, colors by displacement magnitude on deformed shape.
    angles : list of (elevation, azimuth), optional
        Viewing angles. Default: 6 views (front, back, left, right, top, iso).
    """
    t = _get_theme(theme)

    if angles is None:
        angles = [
            (25, -60, "Isometric"),
            (0, 0, "Front (XY)"),
            (0, 90, "Side (YZ)"),
            (90, 0, "Top (XZ)"),
            (25, 120, "Back-Right"),
            (25, -150, "Back-Left"),
        ]

    n = len(angles)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(t["bg"])

    node_ids = sorted(graph.nodes.keys())
    pos_orig = np.array([graph.nodes[nid].position for nid in node_ids])

    # Use deformed shape if available
    if sim_result is not None and sim_result.deformed_positions is not None:
        deformed = graph.copy()
        for idx, nid in enumerate(node_ids):
            if idx < len(sim_result.deformed_positions):
                deformed.nodes[nid].position = sim_result.deformed_positions[idx].copy()
        disp_mags = np.linalg.norm(sim_result.displacements[:, :3], axis=1)
    else:
        deformed = graph
        disp_mags = None

    for i, angle_info in enumerate(angles):
        elev = angle_info[0]
        azim = angle_info[1]
        label = angle_info[2] if len(angle_info) > 2 else f"View {i+1}"

        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        ax.set_facecolor(t["bg"])

        if disp_mags is not None:
            _render_3d_colored(ax, deformed, disp_mags, t, line_width, elev, azim, label)
        else:
            _render_3d_segments(ax, deformed, t, line_width, elev, azim, label)

    # Hide empty subplots
    for i in range(n, nrows * ncols):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        ax.set_visible(False)

    if title:
        fig.suptitle(title, color=t["text"], fontsize=14, fontweight="bold", y=0.98)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor(),
                    bbox_inches="tight", pad_inches=0.1)
    return fig


def render_fem_stress(graph, fem_result, *,
                      theme="dark",
                      stress_type="total",
                      title="",
                      figsize=(8, 8),
                      linewidth=1.0,
                      show_colorbar=True,
                      show_boundary=False,
                      ax=None,
                      **kwargs):
    """Render FEM stress distribution on structure edges.
    
    Parameters
    ----------
    graph : StructureGraph
        The structure graph.
    fem_result : dict
        Result from BeamFrameFEM_v6 solve methods.
    stress_type : str
        "total", "axial", or "bending".
    theme : str
        "dark", "light", or custom theme name.
    title : str
        Figure title.
    figsize : tuple
        Figure size.
    linewidth : float
        Edge line width.
    show_colorbar : bool
        Show stress colorbar.
    show_boundary : bool
        Highlight boundary nodes.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    
    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    
    from fibernet.sim.accelerated import _graph_to_arrays
    from fibernet.viz.render import _get_theme
    
    colors = _get_theme(theme)
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure
    
    ax.set_facecolor(colors['bg'])
    ax.tick_params(colors=colors['text'])
    for spine in ax.spines.values():
        spine.set_color(colors['grid'])
    
    pos, elements, _, _ = _graph_to_arrays(graph)
    u = fem_result['u']
    
    # Deformed positions (2D)
    if u.shape[1] >= 3:
        deformed = pos[:, :2] + u[:, :2]
    else:
        deformed = pos[:, :2] + u
    
    # Select stress type
    stress_map = {
        "total": fem_result.get('sigma_total', np.zeros(len(fem_result.get('edge_list', [])))),
        "axial": fem_result.get('sigma_axial', np.zeros(len(fem_result.get('edge_list', [])))),
        "bending": fem_result.get('sigma_bending', np.zeros(len(fem_result.get('edge_list', [])))),
    }
    stresses = stress_map.get(stress_type, stress_map["total"])
    edge_list = fem_result.get('edge_list', np.arange(len(stresses)))
    
    if len(stresses) == 0:
        return fig
    
    # Color scale
    s_max = np.percentile(np.abs(stresses), 95) if len(stresses) > 0 else 1.0
    if s_max < 1e-10:
        s_max = 1.0
    norm = Normalize(vmin=-s_max, vmax=s_max)
    cmap = cm.coolwarm if stress_type == "axial" else cm.inferno
    
    # Build line segments
    segments = []
    seg_colors = []
    seg_widths = []
    
    for idx, e in enumerate(edge_list):
        i, j = int(elements[e, 0]), int(elements[e, 1])
        if 0 <= i < len(deformed) and 0 <= j < len(deformed):
            segments.append([deformed[i], deformed[j]])
            s = stresses[idx] if idx < len(stresses) else 0
            seg_colors.append(cmap(norm(np.clip(s, -s_max, s_max))))
            # Scale linewidth by absolute stress
            w = linewidth * (0.5 + 1.5 * abs(s) / s_max)
            seg_widths.append(w)
    
    if segments:
        lc = LineCollection(segments, colors=seg_colors, linewidths=seg_widths)
        ax.add_collection(lc)
        ax.autoscale()
    
    # Boundary nodes
    if show_boundary:
        left_nodes = fem_result.get('left_nodes', [])
        right_nodes = fem_result.get('right_nodes', [])
        for ni in left_nodes:
            if 0 <= ni < len(deformed):
                ax.plot(deformed[ni, 0], deformed[ni, 1], 's',
                       color='#e74c3c', markersize=4, zorder=5)
        for ni in right_nodes:
            if 0 <= ni < len(deformed):
                ax.plot(deformed[ni, 0], deformed[ni, 1], '^',
                       color='#3498db', markersize=4, zorder=5)
    
    # Colorbar
    if show_colorbar:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label(f'{stress_type.capitalize()} Stress (Pa)',
                    color=colors['text'], fontsize=9)
        cb.ax.tick_params(colors=colors['text'], labelsize=7)
    
    # Title
    if title:
        ax.set_title(title, color=colors['text'], fontsize=11, fontweight='bold', pad=10)
    
    ax.set_aspect('equal')
    ax.axis('off')
    
    return fig
