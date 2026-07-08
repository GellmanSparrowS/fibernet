"""
2D visualization for fiber networks using matplotlib.

Provides:
- Fiber network plots
- Orientation distribution plots
- Stress-strain curves
- Histograms and statistical plots
"""

from __future__ import annotations

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


def plot_dynamics_result(
    result: Dict,
    show_forces: bool = False,
    colormap: str = "viridis",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Visualize mass-spring dynamics results.
    
    Parameters
    ----------
    result : dict
        Output from ``fn.simulate_dynamics()``.
    show_forces : bool
        If True, overlay force vectors on nodes.
    colormap : str
        Colormap for spring strain visualization.
    save_path : str, optional
        Path to save figure.
    
    Returns
    -------
    plt.Figure
        Matplotlib figure with trajectory visualization.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for visualization")
    
    trajectory = result.get('trajectory', [])
    edges = result.get('edges', [])
    rest_lengths = result.get('rest_lengths', [])
    initial_positions = result.get('initial_positions', [])
    
    if len(trajectory) == 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.text(0.5, 0.5, "No trajectory data", ha='center', va='center')
        return fig
    
    # Select frames to show
    n_frames = min(6, len(trajectory))
    frame_indices = np.linspace(0, len(trajectory) - 1, n_frames, dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, frame_idx in enumerate(frame_indices):
        ax = axes[i]
        positions = trajectory[frame_idx]
        
        # Compute spring strains
        strains = []
        for e_idx, (node_i, node_j) in enumerate(edges):
            pos_i = positions[node_i]
            pos_j = positions[node_j]
            L = np.linalg.norm(pos_j - pos_i)
            strain = (L - rest_lengths[e_idx]) / rest_lengths[e_idx]
            strains.append(strain)
        
        strains = np.array(strains)
        
        # Draw springs with strain colormap
        lines = []
        for node_i, node_j in edges:
            lines.append([positions[node_i][:2], positions[node_j][:2]])
        
        cmap = matplotlib.colormaps.get_cmap(colormap)
        norm = Normalize(vmin=strains.min(), vmax=strains.max())
        lc = LineCollection(
            lines, linewidths=1.5, cmap=cmap, norm=norm,
            array=strains, alpha=0.8
        )
        ax.add_collection(lc)
        
        # Draw nodes
        ax.scatter(positions[:, 0], positions[:, 1], c='black', s=20, zorder=5)
        
        # Show forces if requested
        if show_forces and 'forces' in result and frame_idx == len(trajectory) - 1:
            forces = result['forces']
            force_scale = 0.1
            ax.quiver(
                positions[:, 0], positions[:, 1],
                forces[:, 0] * force_scale, forces[:, 1] * force_scale,
                alpha=0.5, color='red'
            )
        
        ax.autoscale()
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f"Frame {frame_idx} (t = {frame_idx * result.get('dt', 1e-7):.2e} s)")
        ax.grid(True, alpha=0.2)
    
    # Add colorbar
    cbar = plt.colorbar(lc, ax=axes[-1], label='Spring Strain')
    
    plt.suptitle('Mass-Spring Dynamics Trajectory', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    
    return fig


def plot_metamaterial(
    network: FiberNetwork,
    show_unit_cells: bool = True,
    show_crosslinks: bool = True,
    colormap: str = "viridis",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Professional visualization for metamaterial structures.
    
    Parameters
    ----------
    network : FiberNetwork
        Metamaterial network (from create_metamaterial).
    show_unit_cells : bool
        If True, draw unit cell boundaries.
    show_crosslinks : bool
        If True, show crosslink points.
    colormap : str
        Colormap for fiber coloring.
    save_path : str, optional
        Path to save figure.
    
    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for visualization")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Draw fibers with orientation-based coloring
    lines = []
    orientations = []
    
    for fiber in network.fibers:
        pts = fiber.centerline[:, :2]
        lines.append(pts)
        
        # Compute orientation angle
        direction = fiber.direction
        angle = np.arctan2(direction[1], direction[0])
        orientations.append(angle)
    
    orientations = np.array(orientations)
    cmap = matplotlib.colormaps.get_cmap(colormap)
    norm = Normalize(vmin=-np.pi, vmax=np.pi)
    
    lc = LineCollection(
        lines, linewidths=2.0, cmap=cmap, norm=norm,
        array=orientations, alpha=0.9
    )
    ax.add_collection(lc)
    
    # Draw crosslinks
    if show_crosslinks and network.crosslinks:
        cl_positions = np.array([cl.position[:2] for cl in network.crosslinks])
        ax.scatter(
            cl_positions[:, 0], cl_positions[:, 1],
            c='red', s=30, marker='o', zorder=5, alpha=0.6,
            label=f'Crosslinks ({len(network.crosslinks)})'
        )
    
    # Draw unit cell boundaries if available
    if show_unit_cells and hasattr(network, 'metadata'):
        meta = network.metadata
        if 'array_size' in meta:
            bb_min, bb_max = network.bounding_box()
            cell_size = (bb_max - bb_min)[:2] / np.array(meta['array_size'])
            
            for i in range(meta['array_size'][0] + 1):
                x = bb_min[0] + i * cell_size[0]
                ax.axvline(x, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
            
            for j in range(meta['array_size'][1] + 1):
                y = bb_min[1] + j * cell_size[1]
                ax.axhline(y, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(lc, ax=ax, label='Fiber Orientation (rad)')
    
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    
    # Title with metadata
    title = f"Metamaterial Structure\n"
    if hasattr(network, 'metadata'):
        meta = network.metadata
        title += f"Unit Cell: {meta.get('unit_cell', 'unknown')} | "
        title += f"Array: {meta.get('array_size', 'unknown')}\n"
    title += f"Fibers: {network.num_fibers} | Crosslinks: {network.num_crosslinks}"
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    
    return fig


def plot_stress_strain(
    result: Dict,
    show_modulus: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot stress-strain curve from mechanics simulation.
    
    Parameters
    ----------
    result : dict
        Output from ``fn.simulate_mechanics()``.
    show_modulus : bool
        If True, annotate Young's modulus.
    save_path : str, optional
        Path to save figure.
    
    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for visualization")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    if 'stress_strain' in result:
        strains, stresses = result['stress_strain']
        ax.plot(strains, stresses, 'b-', linewidth=2, label='Stress-Strain')
        ax.set_xlabel('Strain (ε)')
        ax.set_ylabel('Stress (σ) [Pa]')
        ax.set_title('Stress-Strain Curve')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if show_modulus and 'modulus' in result:
            E = result['modulus']
            ax.annotate(
                f'E = {E:.2e} Pa',
                xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
    else:
        ax.text(0.5, 0.5, "No stress-strain data available",
                ha='center', va='center', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    
    return fig
