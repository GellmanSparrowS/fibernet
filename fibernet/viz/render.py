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
                 color_data=edge_data, colormap="hot",
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
