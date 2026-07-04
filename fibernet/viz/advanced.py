"""
Advanced visualization utilities.

Provides:
- Stress field visualization
- Temperature field visualization
- Displacement field visualization
- Cross-section views
- Animation creation
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from fibernet.core.network import FiberNetwork


def plot_stress_field(
    network: FiberNetwork,
    stresses: np.ndarray,
    filename: str = None,
    colormap: str = "viridis",
    vmin: float = None,
    vmax: float = None,
):
    """Plot stress field on fiber network.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network.
    stresses : np.ndarray
        Stress values for each element/fiber.
    filename : str, optional
        Output filename for saving.
    colormap : str
        Matplotlib colormap name.
    vmin, vmax : float, optional
        Color scale limits.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
    except ImportError:
        print("matplotlib required for stress visualization")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    segments = []
    colors = []
    
    for i, fiber in enumerate(network.fibers):
        pts = fiber.centerline
        for j in range(len(pts) - 1):
            segments.append([pts[j][:2], pts[j+1][:2]])
            if i < len(stresses):
                colors.append(stresses[i])
            else:
                colors.append(0.0)
    
    if not segments:
        print("No segments to plot")
        return
    
    lc = LineCollection(segments, cmap=colormap, linewidths=2)
    lc.set_array(np.array(colors))
    
    if vmin is not None:
        lc.set_clim(vmin, vmax)
    
    line = ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Stress Field')
    
    cbar = plt.colorbar(lc, ax=ax)
    cbar.set_label('Stress (Pa)')
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    return fig, ax


def plot_temperature_field(
    network: FiberNetwork,
    temperatures: np.ndarray,
    filename: str = None,
    colormap: str = "hot",
):
    """Plot temperature field on fiber network.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network.
    temperatures : np.ndarray
        Temperature values at nodes.
    filename : str, optional
        Output filename.
    colormap : str
        Matplotlib colormap name.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
    except ImportError:
        print("matplotlib required")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    segments = []
    colors = []
    
    node_id = 0
    for fiber in network.fibers:
        pts = fiber.centerline
        for j in range(len(pts) - 1):
            segments.append([pts[j][:2], pts[j+1][:2]])
            # Average temperature of two nodes
            if node_id < len(temperatures) and node_id + 1 < len(temperatures):
                colors.append((temperatures[node_id] + temperatures[node_id + 1]) / 2)
            else:
                colors.append(0.0)
            node_id += 1
    
    if not segments:
        return
    
    lc = LineCollection(segments, cmap=colormap, linewidths=2)
    lc.set_array(np.array(colors))
    
    line = ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Temperature Field')
    
    cbar = plt.colorbar(lc, ax=ax)
    cbar.set_label('Temperature (K)')
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    return fig, ax


def plot_displacement_field(
    network: FiberNetwork,
    displacements: np.ndarray,
    filename: str = None,
    scale: float = 1.0,
    colormap: str = "coolwarm",
):
    """Plot displacement field on fiber network.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network.
    displacements : np.ndarray
        Displacement vectors (N, 3).
    filename : str, optional
        Output filename.
    scale : float
        Displacement scaling factor.
    colormap : str
        Matplotlib colormap name.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot original network (gray)
    for fiber in network.fibers:
        pts = fiber.centerline
        ax.plot(pts[:, 0], pts[:, 1], 'gray', alpha=0.3, linewidth=1)
    
    # Plot displaced network (colored by magnitude)
    node_id = 0
    for fiber in network.fibers:
        pts = fiber.centerline
        displaced = []
        mags = []
        
        for pt in pts:
            if node_id < len(displacements):
                disp = displacements[node_id]
                new_pt = pt + disp * scale
                displaced.append(new_pt[:2])
                mags.append(np.linalg.norm(disp))
            else:
                displaced.append(pt[:2])
                mags.append(0.0)
            node_id += 1
        
        displaced = np.array(displaced)
        scatter = ax.scatter(displaced[:, 0], displaced[:, 1], 
                           c=mags, cmap=colormap, s=10)
        ax.plot(displaced[:, 0], displaced[:, 1], 'b-', alpha=0.5, linewidth=1)
    
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Displacement Field (scale={scale})')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Displacement Magnitude')
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    return fig, ax


def plot_cross_section(
    network: FiberNetwork,
    axis: int = 2,
    position: float = None,
    thickness: float = 1.0,
    filename: str = None,
):
    """Plot cross-section view of fiber network.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network (3D).
    axis : int
        Normal axis for cross-section (0=x, 1=y, 2=z).
    position : float, optional
        Position along axis. Defaults to center.
    thickness : float
        Slice thickness.
    filename : str, optional
        Output filename.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
    except ImportError:
        print("matplotlib required")
        return
    
    if position is None:
        bb_min, bb_max = network.bounding_box()
        position = (bb_min[axis] + bb_max[axis]) / 2
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Find fibers that intersect the slice
    for fiber in network.fibers:
        pts = fiber.centerline
        for pt in pts:
            if abs(pt[axis] - position) < thickness / 2:
                # Plot cross-section as circle
                if axis == 0:
                    center = (pt[1], pt[2])
                elif axis == 1:
                    center = (pt[0], pt[2])
                else:
                    center = (pt[0], pt[1])
                
                circle = Circle(center, fiber.radius, fill=True, 
                               facecolor='steelblue', edgecolor='black', linewidth=0.5)
                ax.add_patch(circle)
    
    ax.set_aspect('equal')
    labels = ['X', 'Y', 'Z']
    if axis == 0:
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
    elif axis == 1:
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
    else:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    ax.set_title(f'Cross-section at {labels[axis]}={position:.2f}')
    ax.autoscale()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    return fig, ax


def create_animation(
    frames: List[Dict],
    filename: str,
    fps: int = 10,
):
    """Create animation from simulation frames.
    
    Parameters
    ----------
    frames : list of dict
        Each frame contains 'network', 'data', etc.
    filename : str
        Output filename (.gif or .mp4).
    fps : int
        Frames per second.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
    except ImportError:
        print("matplotlib required")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def animate(i):
        ax.clear()
        frame = frames[i]
        
        network = frame.get('network')
        if network:
            for fiber in network.fibers:
                pts = fiber.centerline
                ax.plot(pts[:, 0], pts[:, 1], 'b-', linewidth=1)
        
        ax.set_aspect('equal')
        ax.set_title(f'Frame {i+1}/{len(frames)}')
    
    anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=1000/fps)
    
    if filename.endswith('.gif'):
        anim.save(filename, writer='pillow', fps=fps)
    else:
        anim.save(filename, writer='ffmpeg', fps=fps)
    
    plt.close()
