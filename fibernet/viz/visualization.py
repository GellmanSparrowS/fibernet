"""
3D visualization module for fiber networks.

Provides interactive visualization using matplotlib and pyvista (optional).
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List, Tuple
from ..core import FiberNetwork


def visualize_3d_matplotlib(
    network: FiberNetwork,
    ax=None,
    color: str = 'blue',
    linewidth: float = 1.0,
    alpha: float = 1.0,
    show_crosslinks: bool = True,
    crosslink_color: str = 'red',
    crosslink_size: float = 5,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Visualize fiber network in 3D using matplotlib.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to visualize
    ax : matplotlib Axes3D, optional
        3D axes to plot on
    color : str
        Color for fibers
    linewidth : float
        Line width for fibers
    alpha : float
        Transparency
    show_crosslinks : bool
        Whether to show crosslinks
    crosslink_color : str
        Color for crosslinks
    crosslink_size : float
        Size of crosslink markers
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib Figure
    ax : matplotlib Axes3D
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
    
    # Plot fibers using centerline
    for fiber in network.fibers:
        centerline = fiber.centerline
        ax.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2],
                color=color, linewidth=linewidth, alpha=alpha)
    
    # Plot crosslinks
    if show_crosslinks and hasattr(network, 'crosslinks'):
        crosslink_positions = []
        for crosslink in network.crosslinks:
            crosslink_positions.append(crosslink.position)
        
        if len(crosslink_positions) > 0:
            crosslink_positions = np.array(crosslink_positions)
            ax.scatter(crosslink_positions[:, 0],
                      crosslink_positions[:, 1],
                      crosslink_positions[:, 2],
                      c=crosslink_color, s=crosslink_size, alpha=alpha)
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Fiber Network ({network.num_fibers} fibers, {network.num_crosslinks} crosslinks)')
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def visualize_3d_pyvista(
    network: FiberNetwork,
    color: str = 'blue',
    fiber_radius: Optional[float] = None,
    show_crosslinks: bool = True,
    crosslink_color: str = 'red',
    crosslink_radius: float = 0.1,
    background: str = 'white',
    window_size: Tuple[int, int] = (1024, 768),
    save_path: Optional[str] = None,
    off_screen: bool = False,
):
    """
    Visualize fiber network in 3D using pyvista (interactive).
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to visualize
    color : str
        Color for fibers
    fiber_radius : float, optional
        Radius for fiber tubes (auto if None)
    show_crosslinks : bool
        Whether to show crosslinks
    crosslink_color : str
        Color for crosslinks
    crosslink_radius : float
        Radius for crosslink spheres
    background : str
        Background color
    window_size : tuple
        Window size (width, height)
    save_path : str, optional
        Path to save screenshot
    off_screen : bool
        Render off-screen (for headless environments)
    
    Returns
    -------
    plotter : pyvista.Plotter
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("pyvista is required for interactive 3D visualization. "
                         "Install with: pip install pyvista")
    
    # Create plotter
    plotter = pv.Plotter(window_size=window_size, off_screen=off_screen)
    plotter.set_background(background)
    
    # Auto-detect fiber radius
    if fiber_radius is None:
        if hasattr(network, 'fibers') and len(network.fibers) > 0:
            fiber_radius = getattr(network.fibers[0], 'radius', 0.05)
        else:
            fiber_radius = 0.05
    
    # Plot fibers as tubes using centerline
    for fiber in network.fibers:
        centerline = fiber.centerline
        if len(centerline) >= 2:
            # Create line
            line = pv.Spline(centerline, n_points=len(centerline))
            # Extrude to tube
            tube = line.tube(radius=fiber_radius)
            plotter.add_mesh(tube, color=color, smooth_shading=True)
    
    # Plot crosslinks as spheres
    if show_crosslinks and hasattr(network, 'crosslinks'):
        crosslink_positions = []
        for crosslink in network.crosslinks:
            crosslink_positions.append(crosslink.position)
        
        if len(crosslink_positions) > 0:
            crosslink_positions = np.array(crosslink_positions)
            # Create spheres
            spheres = pv.Sphere(radius=crosslink_radius, center=crosslink_positions[0])
            for pos in crosslink_positions[1:]:
                spheres = spheres + pv.Sphere(radius=crosslink_radius, center=pos)
            plotter.add_mesh(spheres, color=crosslink_color, smooth_shading=True)
    
    # Add coordinate axes
    plotter.add_axes()
    
    # Add title
    plotter.add_title(f'Fiber Network ({network.num_fibers} fibers, {network.num_crosslinks} crosslinks)')
    
    if save_path:
        plotter.screenshot(save_path)
    
    if not off_screen:
        plotter.show()
    
    return plotter


def visualize_network_stress(
    network: FiberNetwork,
    stress_values: np.ndarray,
    ax=None,
    cmap: str = 'coolwarm',
    linewidth: float = 2.0,
    colorbar: bool = True,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Visualize fiber network with stress coloring.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to visualize
    stress_values : np.ndarray
        Stress values for each fiber
    ax : matplotlib Axes3D, optional
        3D axes to plot on
    cmap : str
        Colormap name
    linewidth : float
        Line width for fibers
    colorbar : bool
        Whether to show colorbar
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib Figure
    ax : matplotlib Axes3D
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.colors import Normalize
    
    if ax is None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
    
    # Normalize stress for coloring
    norm = Normalize(vmin=stress_values.min(), vmax=stress_values.max())
    colors = cm.get_cmap(cmap) if hasattr(cm, "get_cmap") else plt.get_cmap(cmap)(norm(stress_values))
    
    # Plot fibers with stress coloring using centerline
    for i, fiber in enumerate(network.fibers):
        centerline = fiber.centerline
        if i < len(colors):
            color = colors[i]
        else:
            color = 'gray'
        
        ax.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2],
                color=color, linewidth=linewidth)
    
    # Add colorbar
    if colorbar:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=10)
        cbar.set_label('Stress')
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Fiber Network Stress Distribution')
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def animate_deformation(
    network: FiberNetwork,
    displacement_history: List[np.ndarray],
    interval: int = 100,
    save_path: Optional[str] = None,
):
    """
    Create animation of network deformation.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to visualize
    displacement_history : list of np.ndarray
        List of displacement arrays for each timestep
    interval : int
        Interval between frames in milliseconds
    save_path : str, optional
        Path to save animation (mp4 or gif)
    
    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get original centerlines
    original_centerlines = []
    for fiber in network.fibers:
        original_centerlines.append(fiber.centerline.copy())
    
    # Initialize lines
    lines = []
    for fiber in network.fibers:
        line, = ax.plot([], [], [], 'b-', linewidth=1)
        lines.append(line)
    
    def init():
        for i, fiber in enumerate(network.fibers):
            centerline = fiber.centerline
            lines[i].set_data_3d(centerline[:, 0], centerline[:, 1], centerline[:, 2])
        return lines
    
    def update(frame):
        # Apply displacement
        displacement = displacement_history[frame]
        
        # Update fiber positions
        node_idx = 0
        for i, fiber in enumerate(network.fibers):
            centerline = original_centerlines[i]
            # Apply displacement to centerline points
            displaced_centerline = centerline + displacement[node_idx:node_idx+len(centerline)]
            lines[i].set_data_3d(displaced_centerline[:, 0], 
                                displaced_centerline[:, 1], 
                                displaced_centerline[:, 2])
            node_idx += len(centerline)
        
        return lines
    
    anim = FuncAnimation(fig, update, frames=len(displacement_history),
                        init_func=init, blit=False, interval=interval)
    
    if save_path:
        if save_path.endswith('.mp4'):
            anim.save(save_path, writer='ffmpeg', fps=1000//interval)
        elif save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=1000//interval)
    
    return anim


def visualize_damage_evolution(
    damage_result: dict,
    save_path: Optional[str] = None,
):
    """
    Visualize damage evolution from progressive damage simulation.
    
    Parameters
    ----------
    damage_result : dict
        Result from progressive_damage() simulation
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Stress-strain curve
    ax = axes[0, 0]
    ax.plot(damage_result['strain'], damage_result['stress'], 'b-', linewidth=2)
    ax.set_xlabel('Strain')
    ax.set_ylabel('Stress')
    ax.set_title('Stress-Strain Curve')
    ax.grid(True, alpha=0.3)
    
    # Damage evolution
    ax = axes[0, 1]
    ax.plot(damage_result['strain'], damage_result['damage'], 'r-', linewidth=2)
    ax.set_xlabel('Strain')
    ax.set_ylabel('Damage (D)')
    ax.set_title('Damage Evolution')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # Broken elements
    ax = axes[1, 0]
    ax.plot(damage_result['strain'], damage_result['broken_elements'], 'k-', linewidth=2)
    ax.set_xlabel('Strain')
    ax.set_ylabel('Broken Elements')
    ax.set_title('Element Failure')
    ax.grid(True, alpha=0.3)
    
    # Stiffness degradation
    ax = axes[1, 1]
    strain = damage_result['strain']
    stress = damage_result['stress']
    if len(strain) > 1:
        stiffness = np.diff(stress) / np.diff(strain)
        stiffness = np.abs(stiffness)
        ax.plot(strain[1:], stiffness, 'g-', linewidth=2)
    ax.set_xlabel('Strain')
    ax.set_ylabel('Tangent Stiffness')
    ax.set_title('Stiffness Degradation')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_network_2d(
    network,
    color_by: str = 'fiber_id',
    show_crosslinks: bool = True,
    ax=None,
    **kwargs
):
    """
    Plot 2D fiber network.
    
    Parameters
    ----------
    network : FiberNetwork
        2D fiber network
    color_by : str
        Color fibers by 'fiber_id', 'material', 'stress', or 'uniform'
    show_crosslinks : bool
        Whether to show crosslinks
    ax : matplotlib Axes, optional
        Axes to plot on
    **kwargs
        Additional plotting options
    
    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure
    
    # Plot fibers
    for i, fiber in enumerate(network.fibers):
        centerline = fiber.centerline
        ax.plot(centerline[:, 0], centerline[:, 1], linewidth=kwargs.get('linewidth', 1.0))
    
    # Plot crosslinks
    if show_crosslinks and hasattr(network, 'crosslinks'):
        crosslink_positions = []
        for crosslink in network.crosslinks:
            crosslink_positions.append(crosslink.position)
        
        if len(crosslink_positions) > 0:
            crosslink_positions = np.array(crosslink_positions)
            ax.scatter(crosslink_positions[:, 0], crosslink_positions[:, 1],
                      c='red', s=20, zorder=5, label='Crosslinks')
            ax.legend()
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_network_3d(
    network,
    background: str = 'white',
    window_size: tuple = (1024, 768),
    save_path: str = None,
    **kwargs
):
    """
    Plot 3D fiber network.
    
    Parameters
    ----------
    network : FiberNetwork
        3D fiber network
    background : str
        Background color
    window_size : tuple
        Window size (width, height)
    save_path : str, optional
        Path to save screenshot
    **kwargs
        Additional plotting options
    """
    # Use pyvista if available, otherwise matplotlib
    try:
        return visualize_3d_pyvista(
            network,
            background=background,
            window_size=window_size,
            save_path=save_path,
            off_screen=kwargs.get('off_screen', True)
        )
    except ImportError:
        # Fall back to matplotlib
        return visualize_3d_matplotlib(network, **kwargs)


# Alias for compatibility
render_network_3d = plot_network_3d
