"""
Advanced visualization module for FiberNet.

Provides publication-quality rendering of fiber networks with:
- No node markers (clean fiber-only display)
- Variable line width based on fiber radius
- Color mapping by orientation, length, stress, or custom data
- Dark/light themes with minimal axis styling
- 3D rendering with depth-based alpha
- Structure comparison and statistics panels

All functions follow a consistent API:
    fig, ax = fn.plot(net, color_by="orientation", theme="dark")
    fn.save_figure(fig, "output.png", dpi=300)
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union
from pathlib import Path

from ..core import FiberNetwork


# ============================================================================
# Theme Presets
# ============================================================================

_THEMES = {
    "light": {
        "bg_color": "#FAFAFA",
        "fiber_color": "#2C3E50",
        "crosslink_color": "#E74C3C",
        "axis_color": "#BDC3C7",
        "text_color": "#2C3E50",
        "grid_alpha": 0.0,
    },
    "dark": {
        "bg_color": "#1A1A2E",
        "fiber_color": "#E0E0E0",
        "crosslink_color": "#FF6B6B",
        "axis_color": "#444444",
        "text_color": "#E0E0E0",
        "grid_alpha": 0.0,
    },
    "publication": {
        "bg_color": "#FFFFFF",
        "fiber_color": "#333333",
        "crosslink_color": "#CC0000",
        "axis_color": "#CCCCCC",
        "text_color": "#333333",
        "grid_alpha": 0.0,
    },
    "blueprint": {
        "bg_color": "#0A1628",
        "fiber_color": "#4A9EFF",
        "crosslink_color": "#FF8844",
        "axis_color": "#1A3050",
        "text_color": "#6AB0FF",
        "grid_alpha": 0.15,
    },
}


def _get_colormap(name: str = "viridis"):
    """Get a matplotlib colormap, falling back gracefully."""
    try:
        import matplotlib.pyplot as plt
        return plt.get_cmap(name)
    except Exception:
        import matplotlib.cm as cm
        return cm.get_cmap(name)


# ============================================================================
# Core 2D Plotting
# ============================================================================

def plot(
    network: FiberNetwork,
    ax=None,
    color_by: str = "uniform",
    color_data: Optional[np.ndarray] = None,
    colormap: str = "viridis",
    theme: str = "light",
    show_crosslinks: bool = False,
    linewidth_scale: float = 1.0,
    linewidth_base: float = 0.8,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 10),
    save_path: Optional[str] = None,
    dpi: int = 200,
    **kwargs,
):
    """Plot a 2D or 3D fiber network (projected to 2D) with publication quality.
    
    Parameters
    ----------
    network : FiberNetwork
        The fiber network to plot.
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates new figure.
    color_by : str
        How to color fibers. Options:
        - "uniform" : single color for all fibers
        - "orientation" : color by fiber angle
        - "length" : color by fiber length
        - "radius" : color by fiber radius
        - "material" : color by material type
        - "custom" : use color_data array
    color_data : array-like, optional
        Per-fiber scalar values (used when color_by="custom").
    colormap : str
        Matplotlib colormap name for non-uniform coloring.
    theme : str
        Visual theme: "light", "dark", "publication", "blueprint".
    show_crosslinks : bool
        Whether to show crosslink positions (as subtle dots).
    linewidth_scale : float
        Multiplier for line widths.
    linewidth_base : float
        Base line width in points.
    title : str, optional
        Plot title.
    figsize : tuple
        Figure size in inches.
    save_path : str, optional
        Path to save figure.
    dpi : int
        Resolution for saved figure.
    
    Returns
    -------
    fig : matplotlib Figure
    ax : matplotlib Axes
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
        
    t = _THEMES.get(theme, _THEMES["light"])
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        fig.patch.set_facecolor(t["bg_color"])
    else:
        fig = ax.figure
    
    ax.set_facecolor(t["bg_color"])
    
    # Determine if 2D or 3D projection
    is_2d = network.dimension == 2
    is_3d_fallback = not is_2d and kwargs.get("project_3d", True)
    
    # Compute per-fiber colors
    n_fibers = network.num_fibers
    if n_fibers == 0:
        ax.text(0.5, 0.5, "Empty Network", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color=t["text_color"])
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                       facecolor=t["bg_color"])
        return fig, ax
    
    # Compute colors
    if color_by == "uniform":
        colors = [mcolors.to_rgba(t["fiber_color"])] * n_fibers
    elif color_by == "orientation":
        dirs = network.fiber_orientations()
        if len(dirs) > 0 and is_2d:
            angles = np.arctan2(dirs[:, 1], dirs[:, 0])
            # Normalize to [0, pi] (undirected)
            angles = np.mod(angles, np.pi)
            cmap = _get_colormap(colormap)
            norm = plt.Normalize(0, np.pi)
            colors = [cmap(norm(a)) for a in angles]
        else:
            colors = [mcolors.to_rgba(t["fiber_color"])] * n_fibers
    elif color_by == "length":
        lengths = np.array([f.length for f in network.fibers])
        cmap = _get_colormap(colormap)
        norm = plt.Normalize(lengths.min(), lengths.max())
        colors = [cmap(norm(f.length)) for f in network.fibers]
    elif color_by == "radius":
        radii = np.array([f.radius for f in network.fibers])
        cmap = _get_colormap(colormap)
        norm = plt.Normalize(radii.min(), radii.max())
        colors = [cmap(norm(f.radius)) for f in network.fibers]
    elif color_by == "custom" and color_data is not None:
        cmap = _get_colormap(colormap)
        norm = plt.Normalize(np.min(color_data), np.max(color_data))
        colors = [cmap(norm(v)) for v in color_data]
    else:
        colors = [mcolors.to_rgba(t["fiber_color"])] * n_fibers
    
    # Draw fibers
    for i, fiber in enumerate(network.fibers):
            cl = fiber.centerline
            c = colors[i] if i < len(colors) else mcolors.to_rgba(t["fiber_color"])
            lw = linewidth_base * linewidth_scale * max(fiber.radius * 5, 0.3)
            
            if is_2d:
                ax.plot(cl[:, 0], cl[:, 1], color=c, linewidth=lw,
                       solid_capstyle="round", zorder=2)
            else:
                ax.plot(cl[:, 0], cl[:, 1], color=c, linewidth=lw,
                       solid_capstyle="round", zorder=2, alpha=0.85)
    
    # Show crosslinks (subtle)
    if show_crosslinks and network.crosslinks:
        cl_pos = np.array([cl.position for cl in network.crosslinks])
        if is_2d:
            ax.scatter(cl_pos[:, 0], cl_pos[:, 1], s=2,
                      c=t["crosslink_color"], alpha=0.4, zorder=3,
                      edgecolors="none")
        else:
            ax.scatter(cl_pos[:, 0], cl_pos[:, 1], s=2,
                      c=t["crosslink_color"], alpha=0.3, zorder=3,
                      edgecolors="none")
    
    # Style axes
    _style_axes(ax, t, is_2d, title)
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                   facecolor=t["bg_color"])
    
    return fig, ax


def _style_axes(ax, theme, is_2d, title=None):
    """Apply minimal axis styling."""
    ax.set_aspect("equal" if is_2d else "auto")
    ax.tick_params(colors=theme["axis_color"], labelsize=8, width=0.5)
    
    for spine in ax.spines.values():
        spine.set_color(theme["axis_color"])
        spine.set_linewidth(0.5)
    
    if title:
        ax.set_title(title, color=theme["text_color"], fontsize=12, pad=10)
    
    if theme["grid_alpha"] > 0:
        ax.grid(True, alpha=theme["grid_alpha"], color=theme["axis_color"])
    else:
        ax.grid(False)
    
    # Minimal labels
    ax.set_xlabel("x", color=theme["text_color"], fontsize=9)
    ax.set_ylabel("y", color=theme["text_color"], fontsize=9)


# ============================================================================
# 3D Visualization
# ============================================================================

def plot_3d(
    network: FiberNetwork,
    color: str = "#4A90D9",
    linewidth_scale: float = 1.0,
    alpha: float = 0.9,
    show_crosslinks: bool = False,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
    save_path: Optional[str] = None,
    dpi: int = 200,
    theme: str = "light",
    elevation: float = 25,
    azimuth: float = -60,
    **kwargs,
):
    """Plot 3D fiber network with matplotlib 3D projection.
    
    Parameters
    ----------
    network : FiberNetwork
        3D fiber network to visualize.
    color : str
        Base color for fibers (hex or named).
    linewidth_scale : float
        Width multiplier.
    alpha : float
        Transparency (0-1).
    show_crosslinks : bool
        Show crosslink positions.
    title : str, optional
        Plot title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    dpi : int
        Output resolution.
    theme : str
        Visual theme.
    elevation : float
        Camera elevation angle.
    azimuth : float
        Camera azimuth angle.
    
    Returns
    -------
    fig : matplotlib Figure
    ax : Axes3D
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    
    t = _THEMES.get(theme, _THEMES["light"])
    
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(t["bg_color"])
    ax = fig.add_subplot(111, projection="3d")
    
    ax.set_facecolor(t["bg_color"])
    ax.view_init(elev=elevation, azim=azimuth)
    
    # Plot fibers
    for fiber in network.fibers:
        cl = fiber.centerline
        lw = max(linewidth_scale * fiber.radius * 3, 0.2)
        ax.plot(cl[:, 0], cl[:, 1], cl[:, 2],
               color=color, linewidth=lw, alpha=alpha,
               solid_capstyle="round")
    
    # Crosslinks
    if show_crosslinks and network.crosslinks:
        cl_pos = np.array([cl.position for cl in network.crosslinks])
        ax.scatter(cl_pos[:, 0], cl_pos[:, 1], cl_pos[:, 2],
                  c=t["crosslink_color"], s=3, alpha=0.4)
    
    # Style
    ax.set_xlabel("X", color=t["text_color"], fontsize=9)
    ax.set_ylabel("Y", color=t["text_color"], fontsize=9)
    ax.set_zlabel("Z", color=t["text_color"], fontsize=9)
    ax.tick_params(colors=theme_axis_color(t), labelsize=7)
    
    if title:
        ax.set_title(title, color=t["text_color"], fontsize=12)
    
    # Remove panes for cleaner look
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(t["axis_color"])
    ax.yaxis.pane.set_edgecolor(t["axis_color"])
    ax.zaxis.pane.set_edgecolor(t["axis_color"])
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                   facecolor=t["bg_color"])
    
    return fig, ax


def theme_axis_color(t):
    return t.get("axis_color", "#CCCCCC")


# ============================================================================
# Comparison and Statistics
# ============================================================================

def plot_comparison(
    networks: List[FiberNetwork],
    labels: Optional[List[str]] = None,
    color_by: str = "uniform",
    theme: str = "light",
    ncols: int = 2,
    figsize_per: Tuple[float, float] = (6, 6),
    save_path: Optional[str] = None,
    dpi: int = 200,
    **kwargs,
):
    """Side-by-side comparison of multiple fiber networks.
    
    Parameters
    ----------
    networks : list of FiberNetwork
        Networks to compare.
    labels : list of str, optional
        Labels for each network.
    color_by : str
        Color mode for all subplots.
    theme : str
        Visual theme.
    ncols : int
        Number of columns in grid.
    figsize_per : tuple
        Size per subplot.
    save_path : str, optional
        Path to save figure.
    
    Returns
    -------
    fig : matplotlib Figure
    axes : list of Axes
    """
    import matplotlib.pyplot as plt
    
    n = len(networks)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows))
    
    t = _THEMES.get(theme, _THEMES["light"])
    fig.patch.set_facecolor(t["bg_color"])
    
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    
    for i, net in enumerate(networks):
        label = labels[i] if labels and i < len(labels) else f"Network {i}"
        ax = axes_flat[i]
        ax.set_facecolor(t["bg_color"])
        plot(net, ax=ax, color_by=color_by, theme=theme, title=label, **kwargs)
    
    # Hide empty subplots
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                   facecolor=t["bg_color"])
    
    return fig, axes_flat[:n]


def plot_statistics(
    network: FiberNetwork,
    theme: str = "light",
    figsize: Tuple[float, float] = (14, 5),
    save_path: Optional[str] = None,
    dpi: int = 200,
):
    """Plot network statistics: degree distribution, edge lengths, orientations.
    
    Parameters
    ----------
    network : FiberNetwork
        Network to analyze and plot.
    theme : str
        Visual theme.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    
    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt
    
    t = _THEMES.get(theme, _THEMES["light"])
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.patch.set_facecolor(t["bg_color"])
    
    # 1. Degree distribution (from crosslinks)
    ax = axes[0]
    degrees = {}
    for cl in network.crosslinks:
        degrees[cl.fiber_i] = degrees.get(cl.fiber_i, 0) + 1
        degrees[cl.fiber_j] = degrees.get(cl.fiber_j, 0) + 1
    
    if degrees:
        deg_vals = list(degrees.values())
        ax.hist(deg_vals, bins=max(5, max(deg_vals) - min(deg_vals) + 1),
               color=t["fiber_color"], alpha=0.7, edgecolor="none")
    ax.set_xlabel("Degree", color=t["text_color"], fontsize=9)
    ax.set_ylabel("Count", color=t["text_color"], fontsize=9)
    ax.set_title("Degree Distribution", color=t["text_color"], fontsize=11)
    ax.tick_params(colors=t["axis_color"], labelsize=8)
    ax.set_facecolor(t["bg_color"])
    for s in ax.spines.values():
        s.set_color(t["axis_color"])
    
    # 2. Edge length histogram
    ax = axes[1]
    lengths = [f.length for f in network.fibers]
    n_bins = min(30, max(5, int(np.ptp(lengths) * 10))) if np.ptp(lengths) > 1e-10 else 1
    ax.hist(lengths, bins=n_bins, color=t["fiber_color"], alpha=0.7, edgecolor="none")
    ax.set_xlabel("Fiber Length", color=t["text_color"], fontsize=9)
    ax.set_ylabel("Count", color=t["text_color"], fontsize=9)
    ax.set_title("Length Distribution", color=t["text_color"], fontsize=11)
    ax.tick_params(colors=t["axis_color"], labelsize=8)
    ax.set_facecolor(t["bg_color"])
    for s in ax.spines.values():
        s.set_color(t["axis_color"])
    
    # 3. Orientation histogram (rose diagram)
    ax = axes[2]
    if network.dimension == 2:
        dirs = network.fiber_orientations()
        if len(dirs) > 0:
            angles = np.arctan2(dirs[:, 1], dirs[:, 0])
            ax.hist(angles, bins=36, color=t["fiber_color"], alpha=0.7,
                   edgecolor="none", range=(-np.pi, np.pi))
    ax.set_xlabel("Angle (rad)", color=t["text_color"], fontsize=9)
    ax.set_ylabel("Count", color=t["text_color"], fontsize=9)
    ax.set_title("Orientation Distribution", color=t["text_color"], fontsize=11)
    ax.tick_params(colors=t["axis_color"], labelsize=8)
    ax.set_facecolor(t["bg_color"])
    for s in ax.spines.values():
        s.set_color(t["axis_color"])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                   facecolor=t["bg_color"])
    
    return fig


# ============================================================================
# Convenience Wrappers (backward compatibility)
# ============================================================================

def visualize_3d_matplotlib(network, **kwargs):
    """Alias for plot_3d."""
    return plot_3d(network, **kwargs)


def visualize_3d_pyvista(network, **kwargs):
    """Interactive 3D using pyvista (if available)."""
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("pyvista required: pip install pyvista")
    
    plotter = pv.Plotter(window_size=kwargs.get("window_size", (1024, 768)),
                        off_screen=kwargs.get("off_screen", False))
    plotter.set_background(kwargs.get("background", "white"))
    
    for fiber in network.fibers:
        cl = fiber.centerline
        if len(cl) >= 2:
            line = pv.Spline(cl, n_points=len(cl))
            tube = line.tube(radius=kwargs.get("fiber_radius", fiber.radius))
            plotter.add_mesh(tube, color=kwargs.get("color", "blue"),
                           smooth_shading=True)
    
    if not plotter.off_screen:
        plotter.show()
    elif kwargs.get("save_path"):
        plotter.screenshot(kwargs["save_path"])
    
    return plotter


def plot_network_2d(network, **kwargs):
    """Alias for plot() for backward compatibility."""
    return plot(network, **kwargs)


def plot_network_3d(network, **kwargs):
    """Alias for plot_3d() for backward compatibility."""
    return plot_3d(network, **kwargs)


def plot_graph(G, **kwargs):
    """Plot a NetworkX graph as fiber network."""
    net = FiberNetwork.from_networkx(G) if hasattr(FiberNetwork, "from_networkx") else None
    if net is not None:
        return plot(net, **kwargs)
    raise TypeError("Cannot plot NetworkX graph directly. Convert to FiberNetwork first.")


def plot_graph_comparison(graphs, labels=None, **kwargs):
    """Compare multiple NetworkX graphs."""
    nets = []
    for G in graphs:
        if hasattr(FiberNetwork, "from_networkx"):
            nets.append(FiberNetwork.from_networkx(G))
    return plot_comparison(nets, labels=labels, **kwargs)


# Aliases
render_network_3d = plot_network_3d


def save_figure(fig, path: str, dpi: int = 300, transparent: bool = False):
    """Save a matplotlib figure with high quality settings.
    
    Parameters
    ----------
    fig : matplotlib Figure
        Figure to save.
    path : str
        Output file path.
    dpi : int
        Resolution.
    transparent : bool
        Transparent background.
    """
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
               transparent=transparent, pad_inches=0.1)


# ============================================================================
# Dynamics and Damage Visualization
# ============================================================================

def visualize_deformation(
    network: FiberNetwork,
    displacement_history: List[np.ndarray],
    interval: int = 50,
    save_path: Optional[str] = None,
    theme: str = "light",
    **kwargs,
):
    """Animate deformation of a fiber network over time.
    
    Parameters
    ----------
    network : FiberNetwork
        Original (undeformed) network.
    displacement_history : list of ndarray
        List of Nx3 displacement arrays (one per timestep).
    interval : int
        Animation interval in ms.
    save_path : str, optional
        Path to save animation (.mp4 or .gif).
    
    Returns
    -------
    anim : FuncAnimation
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    t = _THEMES.get(theme, _THEMES["light"])
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor(t["bg_color"])
    ax.set_facecolor(t["bg_color"])
    
    lines = []
    for fiber in network.fibers:
        cl = fiber.centerline
        line, = ax.plot(cl[:, 0], cl[:, 1], color=t["fiber_color"],
                       linewidth=max(fiber.radius * 5, 0.3),
                       solid_capstyle="round")
        lines.append(line)
    
    def update(frame):
        disp = displacement_history[frame]
        for i, (fiber, line) in enumerate(zip(network.fibers, lines)):
            cl = fiber.centerline
            # Displaced positions
            if len(cl) == len(disp):
                displaced = cl + disp
            else:
                # Fallback
                displaced = cl + disp[:len(cl)] if len(disp) > len(cl) else cl
            line.set_data(displaced[:, 0], displaced[:, 1])
        return lines
    
    anim = FuncAnimation(fig, update, frames=len(displacement_history),
                        interval=interval, blit=True)
    
    if save_path:
        if save_path.endswith(".mp4"):
            anim.save(save_path, writer="ffmpeg", fps=1000 // interval)
        elif save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=1000 // interval)
    
    return anim


def visualize_damage_evolution(damage_result: dict, save_path=None, theme="light"):
    """Visualize progressive damage: stress-strain, damage, broken elements."""
    import matplotlib.pyplot as plt
    
    t = _THEMES.get(theme, _THEMES["light"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor(t["bg_color"])
    
    plots = [
        (axes[0, 0], "strain", "stress", "Stress-Strain"),
        (axes[0, 1], "strain", "damage", "Damage Evolution"),
        (axes[1, 0], "strain", "broken_elements", "Element Failure"),
    ]
    
    for ax, xkey, ykey, title in plots:
        ax.set_facecolor(t["bg_color"])
        ax.plot(damage_result[xkey], damage_result[ykey],
               color=t["fiber_color"], linewidth=2)
        ax.set_xlabel(xkey.capitalize(), color=t["text_color"], fontsize=9)
        ax.set_ylabel(ykey.replace("_", " ").capitalize(),
                     color=t["text_color"], fontsize=9)
        ax.set_title(title, color=t["text_color"], fontsize=11)
        ax.tick_params(colors=t["axis_color"], labelsize=8)
        ax.grid(True, alpha=0.2)
        for s in ax.spines.values():
            s.set_color(t["axis_color"])
    
    # Stiffness degradation
    ax = axes[1, 1]
    ax.set_facecolor(t["bg_color"])
    strain = damage_result["strain"]
    stress = damage_result["stress"]
    if len(strain) > 1:
        stiffness = np.abs(np.diff(stress) / np.diff(strain))
        ax.plot(strain[1:], stiffness, color=t["fiber_color"], linewidth=2)
    ax.set_xlabel("Strain", color=t["text_color"], fontsize=9)
    ax.set_ylabel("Tangent Stiffness", color=t["text_color"], fontsize=9)
    ax.set_title("Stiffness Degradation", color=t["text_color"], fontsize=11)
    ax.tick_params(colors=t["axis_color"], labelsize=8)
    ax.grid(True, alpha=0.2)
    for s in ax.spines.values():
        s.set_color(t["axis_color"])
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight",
                   facecolor=t["bg_color"])
    return fig
