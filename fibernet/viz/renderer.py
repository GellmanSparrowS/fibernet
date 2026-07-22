"""
FiberNet Publication-Quality Renderer

High-quality 2D/3D fiber network visualization with:
- Glow effects for 2D networks
- Depth-based opacity for 3D networks
- Adaptive line widths
- Consistent publication style
"""

import numpy as np
from typing import Optional, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import LinearSegmentedColormap

from fibernet.core.network import FiberNetwork


# Publication color scheme
BG_COLOR = '#0a0a0a'
FIBER_COLOR_2D = '#00ff88'
FIBER_COLOR_3D = '#00ccff'
CROSSLINK_COLOR = '#ff6600'
GLOW_COLOR = '#00ff88'


def render_network_2d(
    net: FiberNetwork,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8, 8),
    show_crosslinks: bool = False,
    glow_intensity: float = 0.3,
    line_width_scale: float = 1.0,
    normalize: bool = True,
    title: str = '',
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """Render 2D fiber network with glow effect.
    
    Parameters
    ----------
    net : FiberNetwork
        Network to render.
    ax : matplotlib Axes, optional
        Axes to draw on. If None, creates new figure.
    figsize : tuple
        Figure size (width, height) in inches.
    show_crosslinks : bool
        Draw crosslink markers.
    glow_intensity : float
        Intensity of glow effect (0-1).
    line_width_scale : float
        Scale factor for line widths.
    normalize : bool
        Normalize coordinates to [0, 1].
    title : str
        Title text.
    save_path : str, optional
        Save figure to this path.
    dpi : int
        Resolution for saved figure.
    
    Returns
    -------
    fig : matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure
    
    ax.set_facecolor(BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)
    ax.axis('off')
    
    if net.num_fibers == 0:
        ax.text(0.5, 0.5, 'Empty Network', ha='center', va='center',
               color='white', fontsize=12, transform=ax.transAxes)
        if title:
            ax.text(0.5, 1.02, title, ha='center', va='bottom',
                   color='white', fontsize=11, fontweight='bold',
                   transform=ax.transAxes)
        return fig
    
    # Extract 2D centerlines
    lines = []
    for fiber in net.fibers:
        pts = fiber.centerline[:, :2]
        if len(pts) >= 2:
            lines.append(pts)
    
    if not lines:
        ax.text(0.5, 0.5, 'No 2D data', ha='center', va='center',
               color='white', fontsize=12, transform=ax.transAxes)
        if title:
            ax.text(0.5, 1.02, title, ha='center', va='bottom',
                   color='white', fontsize=11, fontweight='bold',
                   transform=ax.transAxes)
        return fig
    
    # Normalize coordinates
    all_pts = np.vstack(lines)
    if normalize:
        xmin, ymin = all_pts.min(axis=0)
        xmax, ymax = all_pts.max(axis=0)
        scale = max(xmax - xmin, ymax - ymin) or 1.0
        pad = 0.05
        lines = [(pts - np.array([xmin, ymin])) / scale * (1 - 2*pad) + pad 
                 for pts in lines]
    
    # Adaptive line width based on fiber count
    base_lw = max(0.3, min(2.0, 100.0 / net.num_fibers)) * line_width_scale
    
    # Draw glow layer (thicker, more transparent)
    if glow_intensity > 0:
        glow_lw = base_lw * 3.0
        glow_alpha = glow_intensity * 0.4
        glow_lines = [pts for pts in lines]
        glow_lc = LineCollection(glow_lines, linewidths=glow_lw,
                                colors=GLOW_COLOR, alpha=glow_alpha,
                                antialiased=True)
        ax.add_collection(glow_lc)
    
    # Draw main fibers
    lc = LineCollection(lines, linewidths=base_lw, colors=FIBER_COLOR_2D,
                       alpha=0.85, antialiased=True)
    ax.add_collection(lc)
    
    # Draw crosslinks
    if show_crosslinks and net.num_crosslinks > 0:
        cl_pts = np.array([cl.position[:2] for cl in net.crosslinks])
        if normalize:
            cl_pts = (cl_pts - np.array([xmin, ymin])) / scale * (1 - 2*pad) + pad
        cl_size = max(5, min(20, 500.0 / net.num_crosslinks))
        ax.scatter(cl_pts[:, 0], cl_pts[:, 1], s=cl_size, c=CROSSLINK_COLOR,
                  alpha=0.7, edgecolors='none', zorder=10)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    
    if title:
        ax.text(0.5, 1.02, title, ha='center', va='bottom',
               color='white', fontsize=11, fontweight='bold',
               transform=ax.transAxes)
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                   facecolor=BG_COLOR, pad_inches=0.1)
    
    return fig


def render_network_3d(
    net: FiberNetwork,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8, 8),
    elevation: float = 25,
    azimuth: float = 45,
    depth_shading: bool = True,
    line_width_scale: float = 1.0,
    normalize: bool = True,
    title: str = '',
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """Render 3D fiber network with depth shading.
    
    Parameters
    ----------
    net : FiberNetwork
        Network to render.
    ax : matplotlib Axes3D, optional
        Axes to draw on. If None, creates new figure.
    figsize : tuple
        Figure size.
    elevation : float
        Camera elevation angle (degrees).
    azimuth : float
        Camera azimuth angle (degrees).
    depth_shading : bool
        Apply depth-based opacity.
    line_width_scale : float
        Scale factor for line widths.
    normalize : bool
        Normalize coordinates to [0, 1].
    title : str
        Title text.
    save_path : str, optional
        Save figure to this path.
    dpi : int
        Resolution.
    
    Returns
    -------
    fig : matplotlib Figure
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
    
    ax.set_facecolor(BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)
    ax.axis('off')
    
    if net.num_fibers == 0:
        ax.text2D(0.5, 0.5, 'Empty Network', ha='center', va='center',
                 color='white', fontsize=12, transform=ax.transAxes)
        if title:
            ax.text2D(0.5, 1.02, title, ha='center', va='bottom',
                     color='white', fontsize=11, fontweight='bold',
                     transform=ax.transAxes)
        return fig
    
    # Extract 3D centerlines
    lines = [fiber.centerline for fiber in net.fibers if len(fiber.centerline) >= 2]
    
    if not lines:
        ax.text2D(0.5, 0.5, 'No 3D data', ha='center', va='center',
                 color='white', fontsize=12, transform=ax.transAxes)
        if title:
            ax.text2D(0.5, 1.02, title, ha='center', va='bottom',
                     color='white', fontsize=11, fontweight='bold',
                     transform=ax.transAxes)
        return fig
    
    # Normalize coordinates
    all_pts = np.vstack(lines)
    if normalize:
        mins = all_pts.min(axis=0)
        maxs = all_pts.max(axis=0)
        scale = max(maxs - mins) or 1.0
        pad = 0.05
        lines = [(pts - mins) / scale * (1 - 2*pad) + pad for pts in lines]
    
    # Adaptive line width
    base_lw = max(0.2, min(1.5, 80.0 / net.num_fibers)) * line_width_scale
    
    # Depth-based alpha (closer = more opaque)
    if depth_shading:
        # Compute z-center of each fiber
        z_centers = [np.mean(pts[:, 2]) for pts in lines]
        z_min, z_max = min(z_centers), max(z_centers)
        z_range = z_max - z_min or 1.0
        
        # Normalize z to [0.3, 1.0] for alpha
        alphas = [0.3 + 0.7 * (z - z_min) / z_range for z in z_centers]
        
        # Create line collection with per-line alpha
        lc = Line3DCollection(lines, linewidths=base_lw, colors=FIBER_COLOR_3D)
        lc.set_alpha(alphas)
    else:
        lc = Line3DCollection(lines, linewidths=base_lw, colors=FIBER_COLOR_3D,
                             alpha=0.6)
    
    ax.add_collection3d(lc)
    
    # Set limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    
    # Camera angle
    ax.view_init(elev=elevation, azim=azimuth)
    
    if title:
        ax.text2D(0.5, 1.02, title, ha='center', va='bottom',
                 color='white', fontsize=11, fontweight='bold',
                 transform=ax.transAxes)
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                   facecolor=BG_COLOR, pad_inches=0.1)
    
    return fig


def render_comparison_grid(
    networks: List[FiberNetwork],
    titles: List[str],
    is_3d: bool = False,
    ncols: int = 5,
    figsize_per_cell: Tuple[float, float] = (4, 4),
    save_path: str = 'comparison.png',
    dpi: int = 150,
    **render_kwargs,
) -> plt.Figure:
    """Render multiple networks in a grid for comparison.
    
    Parameters
    ----------
    networks : list of FiberNetwork
        Networks to render.
    titles : list of str
        Title for each network.
    is_3d : bool
        Whether networks are 3D.
    ncols : int
        Number of columns in grid.
    figsize_per_cell : tuple
        Size of each cell in inches.
    save_path : str
        Output file path.
    dpi : int
        Resolution.
    **render_kwargs
        Additional arguments passed to render function.
    
    Returns
    -------
    fig : matplotlib Figure
    """
    n = len(networks)
    nrows = (n + ncols - 1) // ncols
    
    cell_w, cell_h = figsize_per_cell
    figsize = (cell_w * ncols, cell_h * nrows)
    
    if is_3d:
        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor(BG_COLOR)
        
        for i, (net, title) in enumerate(zip(networks, titles)):
            ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')
            render_network_3d(net, ax=ax, title=title, **render_kwargs)
    else:
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        fig.patch.set_facecolor(BG_COLOR)
        
        if nrows == 1:
            axes = [axes]
        
        axes_flat = [ax for row in axes for ax in row]
        
        for i, (net, title) in enumerate(zip(networks, titles)):
            render_network_2d(net, ax=axes_flat[i], title=title, **render_kwargs)
        
        # Hide empty cells
        for i in range(n, len(axes_flat)):
            axes_flat[i].axis('off')
            axes_flat[i].set_facecolor(BG_COLOR)
    
    plt.tight_layout(pad=0.5)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
               facecolor=BG_COLOR, pad_inches=0.1)
    
    return fig


def render_parametric_study(
    generator_name: str,
    param_name: str,
    param_values: List,
    base_params: dict,
    is_3d: bool = False,
    save_path: str = 'parametric_study.png',
    **kwargs,
) -> plt.Figure:
    """Generate parametric study visualization.
    
    Parameters
    ----------
    generator_name : str
        Name of generator (e.g., 'lattice_2d').
    param_name : str
        Parameter to vary (e.g., 'cell_size').
    param_values : list
        Values to test.
    base_params : dict
        Base parameters for generator.
    is_3d : bool
        Whether generator produces 3D networks.
    save_path : str
        Output path.
    **kwargs
        Additional render arguments.
    
    Returns
    -------
    fig : matplotlib Figure
    """
    import fibernet as fn
    
    networks = []
    titles = []
    
    for val in param_values:
        params = {**base_params, param_name: val}
        try:
            net = fn.create(generator_name, **params)
            networks.append(net)
            titles.append(f'{param_name}={val}')
        except Exception as e:
            print(f"Warning: {generator_name}({param_name}={val}) failed: {e}")
    
    return render_comparison_grid(
        networks, titles, is_3d=is_3d,
        save_path=save_path, **kwargs
    )
