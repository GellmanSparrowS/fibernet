"""
Visualization Module for Fiber Networks

Provides publication-quality visualization tools for fiber networks:
- 2D and 3D network plots
- Stress/strain field visualization
- Structure-property relationship plots
- Animation capabilities
- Customizable styling for publications

References:
- Matplotlib: https://matplotlib.org/
- Mayavi (3D): https://docs.enthought.com/mayavi/mayavi/
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings

try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize, LinearSegmentedColormap
    import matplotlib.cm as cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("Matplotlib not available. Install with: pip install matplotlib")

try:
    from mpl_toolkits.mplot3d import Axes3D
    HAS_3D = True
except ImportError:
    HAS_3D = False

from fibernet.core.network import FiberNetwork


@dataclass
class PlotStyle:
    """Plot styling configuration."""
    fiber_color: str = 'blue'
    fiber_linewidth: float = 1.0
    fiber_alpha: float = 0.7
    crosslink_color: str = 'red'
    crosslink_marker: str = 'o'
    crosslink_size: float = 5.0
    crosslink_alpha: float = 0.8
    background_color: str = 'white'
    grid: bool = True
    grid_alpha: float = 0.3
    axis_labels: bool = True
    title_fontsize: int = 14
    label_fontsize: int = 12
    
    def to_dict(self) -> Dict:
        return {
            'fiber_color': self.fiber_color,
            'fiber_linewidth': self.fiber_linewidth,
            'fiber_alpha': self.fiber_alpha,
            'crosslink_color': self.crosslink_color,
            'crosslink_marker': self.crosslink_marker,
            'crosslink_size': self.crosslink_size,
            'crosslink_alpha': self.crosslink_alpha,
            'background_color': self.background_color,
            'grid': self.grid,
            'grid_alpha': self.grid_alpha,
            'axis_labels': self.axis_labels,
            'title_fontsize': self.title_fontsize,
            'label_fontsize': self.label_fontsize,
        }


class NetworkVisualizer:
    """Visualize fiber networks with publication-quality plots.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to visualize.
    style : PlotStyle, optional
        Plot styling configuration.
    
    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.visualization import NetworkVisualizer
    >>> net = gen.random_straight_2d(num_fibers=50, seed=42)
    >>> viz = NetworkVisualizer(net)
    >>> fig = viz.plot_2d(title="Random Fiber Network")
    >>> viz.save("network_2d.png", dpi=300)
    """
    
    def __init__(
        self,
        network: FiberNetwork,
        style: Optional[PlotStyle] = None,
    ):
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib required for visualization")
        
        self.network = network
        self.style = style or PlotStyle()
        self.fig = None
        self.ax = None
    
    def plot_2d(
        self,
        show_fibers: bool = True,
        show_crosslinks: bool = True,
        color_by: Optional[str] = None,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (10, 8),
    ) -> plt.Figure:
        """Create 2D plot of fiber network.
        
        Parameters
        ----------
        show_fibers : bool
            Show fiber centerlines.
        show_crosslinks : bool
            Show crosslink points.
        color_by : str, optional
            Color fibers by property: 'length', 'orientation', 'stress'.
        title : str, optional
            Plot title.
        figsize : tuple
            Figure size (width, height) in inches.
        
        Returns
        -------
        fig : matplotlib.Figure
            Figure object.
        """
        self.fig, self.ax = plt.subplots(figsize=figsize)
        
        # Set background
        self.ax.set_facecolor(self.style.background_color)
        
        if show_fibers:
            self._plot_fibers_2d(color_by)
        
        if show_crosslinks:
            self._plot_crosslinks_2d()
        
        # Formatting
        if self.style.axis_labels:
            self.ax.set_xlabel('X', fontsize=self.style.label_fontsize)
            self.ax.set_ylabel('Y', fontsize=self.style.label_fontsize)
        
        if title:
            self.ax.set_title(title, fontsize=self.style.title_fontsize)
        
        if self.style.grid:
            self.ax.grid(True, alpha=self.style.grid_alpha)
        
        self.ax.set_aspect('equal')
        
        return self.fig
    
    def plot_3d(
        self,
        show_fibers: bool = True,
        show_crosslinks: bool = True,
        color_by: Optional[str] = None,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (10, 8),
        elevation: float = 30,
        azimuth: float = 45,
    ) -> plt.Figure:
        """Create 3D plot of fiber network.
        
        Parameters
        ----------
        show_fibers : bool
            Show fiber centerlines.
        show_crosslinks : bool
            Show crosslink points.
        color_by : str, optional
            Color fibers by property.
        title : str, optional
            Plot title.
        figsize : tuple
            Figure size.
        elevation : float
            Viewing elevation angle (degrees).
        azimuth : float
            Viewing azimuth angle (degrees).
        
        Returns
        -------
        fig : matplotlib.Figure
            Figure object.
        """
        if not HAS_3D:
            warnings.warn("3D plotting not available")
            return self.plot_2d(title=title, figsize=figsize)
        
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set background
        self.ax.set_facecolor(self.style.background_color)
        
        if show_fibers:
            self._plot_fibers_3d(color_by)
        
        if show_crosslinks:
            self._plot_crosslinks_3d()
        
        # Formatting
        if self.style.axis_labels:
            self.ax.set_xlabel('X', fontsize=self.style.label_fontsize)
            self.ax.set_ylabel('Y', fontsize=self.style.label_fontsize)
            self.ax.set_zlabel('Z', fontsize=self.style.label_fontsize)
        
        if title:
            self.ax.set_title(title, fontsize=self.style.title_fontsize)
        
        # Set viewing angle
        self.ax.view_init(elev=elevation, azim=azimuth)
        
        return self.fig
    
    def _plot_fibers_2d(self, color_by: Optional[str] = None):
        """Plot fiber centerlines in 2D."""
        if color_by == 'length':
            lengths = [f.length for f in self.network.fibers]
            norm = Normalize(vmin=min(lengths), vmax=max(lengths))
            colors = cm.viridis(norm(lengths))
        elif color_by == 'orientation':
            orientations = []
            for f in self.network.fibers:
                direction = f.centerline[-1] - f.centerline[0]
                angle = np.arctan2(direction[1], direction[0])
                orientations.append(angle)
            norm = Normalize(vmin=-np.pi, vmax=np.pi)
            colors = cm.hsv(norm(orientations))
        else:
            colors = [self.style.fiber_color] * len(self.network.fibers)
        
        # Plot each fiber
        for i, fiber in enumerate(self.network.fibers):
            x = fiber.centerline[:, 0]
            y = fiber.centerline[:, 1]
            color = colors[i] if len(colors) > 1 else colors[0]
            self.ax.plot(
                x, y,
                color=color,
                linewidth=self.style.fiber_linewidth,
                alpha=self.style.fiber_alpha,
            )
        
        # Add colorbar if colored
        if color_by:
            sm = cm.ScalarMappable(cmap=cm.viridis if color_by == 'length' else cm.hsv, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=self.ax, label=color_by.capitalize())
    
    def _plot_fibers_3d(self, color_by: Optional[str] = None):
        """Plot fiber centerlines in 3D."""
        if color_by == 'length':
            lengths = [f.length for f in self.network.fibers]
            norm = Normalize(vmin=min(lengths), vmax=max(lengths))
            colors = cm.viridis(norm(lengths))
        else:
            colors = [self.style.fiber_color] * len(self.network.fibers)
        
        for i, fiber in enumerate(self.network.fibers):
            x = fiber.centerline[:, 0]
            y = fiber.centerline[:, 1]
            z = fiber.centerline[:, 2]
            color = colors[i] if len(colors) > 1 else colors[0]
            self.ax.plot(
                x, y, z,
                color=color,
                linewidth=self.style.fiber_linewidth,
                alpha=self.style.fiber_alpha,
            )
        
        if color_by == 'length':
            sm = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=self.ax, label='Length')
    
    def _plot_crosslinks_2d(self):
        """Plot crosslink points in 2D."""
        if not self.network.crosslinks:
            return
        
        x = [cl.position[0] for cl in self.network.crosslinks]
        y = [cl.position[1] for cl in self.network.crosslinks]
        
        self.ax.scatter(
            x, y,
            c=self.style.crosslink_color,
            marker=self.style.crosslink_marker,
            s=self.style.crosslink_size,
            alpha=self.style.crosslink_alpha,
            zorder=5,
        )
    
    def _plot_crosslinks_3d(self):
        """Plot crosslink points in 3D."""
        if not self.network.crosslinks:
            return
        
        x = [cl.position[0] for cl in self.network.crosslinks]
        y = [cl.position[1] for cl in self.network.crosslinks]
        z = [cl.position[2] for cl in self.network.crosslinks]
        
        self.ax.scatter(
            x, y, z,
            c=self.style.crosslink_color,
            marker=self.style.crosslink_marker,
            s=self.style.crosslink_size,
            alpha=self.style.crosslink_alpha,
        )
    
    def plot_stress_field(
        self,
        stress_field: np.ndarray,
        title: Optional[str] = "Stress Distribution",
        figsize: Tuple[float, float] = (10, 8),
    ) -> plt.Figure:
        """Visualize stress field on network.
        
        Parameters
        ----------
        stress_field : np.ndarray
            Stress values for each fiber.
        title : str
            Plot title.
        figsize : tuple
            Figure size.
        
        Returns
        -------
        fig : matplotlib.Figure
            Figure object.
        """
        self.fig, self.ax = plt.subplots(figsize=figsize)
        
        # Normalize stress
        norm = Normalize(vmin=stress_field.min(), vmax=stress_field.max())
        
        # Plot fibers colored by stress
        for i, fiber in enumerate(self.network.fibers):
            x = fiber.centerline[:, 0]
            y = fiber.centerline[:, 1]
            color = cm.plasma(norm(stress_field[i]))
            self.ax.plot(x, y, color=color, linewidth=2, alpha=0.8)
        
        # Colorbar
        sm = cm.ScalarMappable(cmap=cm.plasma, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=self.ax, label='Stress (Pa)')
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title(title)
        self.ax.set_aspect('equal')
        
        return self.fig
    
    def save(
        self,
        filename: str,
        dpi: int = 300,
        bbox_inches: str = 'tight',
    ):
        """Save figure to file.
        
        Parameters
        ----------
        filename : str
            Output filename (png, pdf, svg).
        dpi : int
            Resolution in dots per inch.
        bbox_inches : str
            Bounding box setting.
        """
        if self.fig is None:
            warnings.warn("No figure to save. Call plot_2d() or plot_3d() first.")
            return
        
        self.fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Figure saved to {filename}")
    
    def show(self):
        """Display figure."""
        if self.fig is None:
            warnings.warn("No figure to show. Call plot_2d() or plot_3d() first.")
            return
        
        plt.show()


def visualize_network(
    network: FiberNetwork,
    dimension: int = 2,
    **kwargs,
) -> plt.Figure:
    """Convenience function for network visualization.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to visualize.
    dimension : int
        Plot dimension (2 or 3).
    **kwargs
        Additional arguments passed to NetworkVisualizer.
    
    Returns
    -------
    fig : matplotlib.Figure
        Figure object.
    
    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.visualization import visualize_network
    >>> net = gen.random_straight_2d(num_fibers=50, seed=42)
    >>> fig = visualize_network(net, dimension=2, title="My Network")
    >>> plt.show()
    """
    viz = NetworkVisualizer(network)
    
    if dimension == 2:
        return viz.plot_2d(**kwargs)
    elif dimension == 3:
        return viz.plot_3d(**kwargs)
    else:
        raise ValueError(f"dimension must be 2 or 3, got {dimension}")
