"""
2D visualization for fiber networks using matplotlib.

Provides:
- Fiber network plots
- Orientation distribution plots
- Stress-strain curves
- Histograms and statistical plots
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    LineCollection = None
    Normalize = None
    ScalarMappable = None
from fibernet.core.network import FiberNetwork


def plot_network_2d(
    network: FiberNetwork,
    ax: Optional[plt.Axes] = None,
    color_by: str = "uniform",
    colormap: str = "viridis",
    show_crosslinks: bool = True,
    line_width: float = 1.0,
    title: str = "",
    figsize: Tuple[float, float] = (8, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot 2D fiber network.
    
    Parameters
    ----------
    color_by : str
        'uniform', 'length', 'orientation', 'curvature', 'stress'.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure
    
    if network.num_fibers == 0:
        ax.set_title("Empty network")
        return fig
    
    lines = []
    colors = []
    
    for fiber in network.fibers:
        pts = fiber.centerline[:, :2]
        lines.append(pts)
        
        if color_by == "length":
            colors.append(fiber.length)
        elif color_by == "orientation":
            d = fiber.direction
            colors.append(np.arctan2(d[1], d[0]))
        elif color_by == "curvature":
            colors.append(np.max(fiber.curvature()))
        else:
            colors.append(0.5)
    
    if color_by == "uniform":
        lc = LineCollection(lines, linewidths=line_width, colors='steelblue')
    else:
        cmap = plt.get_cmap(colormap)
        norm = Normalize(vmin=min(colors), vmax=max(colors))
        lc = LineCollection(
            lines, linewidths=line_width,
            cmap=cmap, norm=norm, array=np.array(colors),
        )
        plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=color_by)
    
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Fiber Network ({network.num_fibers} fibers)")
    
    if show_crosslinks and network.crosslinks:
        cl_x = [cl.position[0] for cl in network.crosslinks]
        cl_y = [cl.position[1] for cl in network.crosslinks]
        ax.scatter(cl_x, cl_y, c='red', s=10, zorder=5, alpha=0.6)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_orientation_distribution(
    network: FiberNetwork,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 6),
) -> plt.Figure:
    """Plot polar histogram of fiber orientations."""
    from fibernet.analysis.morphology import MorphologyAnalyzer
    
    analyzer = MorphologyAnalyzer(network)
    angles, counts = analyzer.orientation_distribution()
    
    if len(angles) == 0:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})
        ax.set_title("No fibers")
        return fig
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})
    
    bars = ax.bar(angles, counts, width=np.pi / len(counts), alpha=0.7, color='steelblue')
    ax.set_title("Fiber Orientation Distribution", pad=20)
    ax.set_rlabel_position(45)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_stress_strain(
    strains: np.ndarray,
    stresses: np.ndarray,
    title: str = "Stress-Strain Curve",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 5),
) -> plt.Figure:
    """Plot stress-strain curve."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(strains, stresses, 'b-', linewidth=2)
    ax.set_xlabel("Strain")
    ax.set_ylabel("Stress (Pa)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_length_distribution(
    network: FiberNetwork,
    num_bins: int = 20,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 5),
) -> plt.Figure:
    """Plot fiber length distribution histogram."""
    fig, ax = plt.subplots(figsize=figsize)
    
    lengths = network.fiber_lengths()
    if len(lengths) == 0:
        return fig
    
    l_min, l_max = lengths.min(), lengths.max()
    if l_max - l_min < 1e-12:
        num_bins = 1
    else:
        num_bins = min(num_bins, max(1, int(np.sqrt(len(lengths)))))
    
    ax.hist(lengths, bins=num_bins, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(np.mean(lengths), color='red', linestyle='--', label=f'Mean={np.mean(lengths):.2f}')
    ax.set_xlabel("Fiber Length")
    ax.set_ylabel("Density")
    ax.set_title("Fiber Length Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
