"""
Publication-quality visualization for fiber networks.

Design principles:
- Square canvas, no axes/frames/ticks
- Dark background with bright fibers
- Anti-aliased lines
- Consistent style across all visualizations
- 3D rendering with pyvista
"""

import numpy as np
from typing import Optional, List, Tuple, Union
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    plt = None

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    pv = None

from fibernet.core.network import FiberNetwork


class ShowcaseStyle:
    """Centralized style configuration for showcase visualizations."""
    
    bg_color = '#0a0a0a'
    bg_color_light = '#f5f5f5'
    fiber_color = '#00ff88'
    fiber_color_alt = '#00ccff'
    fiber_color_warm = '#ff6644'
    crosslink_color = '#ff3366'
    
    line_width_base = 0.8
    line_width_min = 0.2
    line_width_max = 2.0
    
    figsize = (8, 8)
    dpi = 150
    colormap = 'viridis'
    
    @classmethod
    def compute_line_width(cls, num_fibers: int) -> float:
        if num_fibers <= 50:
            return cls.line_width_max
        elif num_fibers >= 1000:
            return cls.line_width_min
        else:
            t = (num_fibers - 50) / (1000 - 50)
            return cls.line_width_max - t * (cls.line_width_max - cls.line_width_min)


def render_2d(
    network: FiberNetwork,
    ax: Optional['plt.Axes'] = None,
    color_by: str = 'uniform',
    uniform_color: Optional[str] = None,
    colormap: str = 'viridis',
    show_crosslinks: bool = False,
    crosslink_size: float = 2.0,
    line_width: Optional[float] = None,
    background: str = 'dark',
    title: Optional[str] = None,
    title_fontsize: int = 10,
    save_path: Optional[Union[str, Path]] = None,
) -> 'plt.Figure':
    """Render 2D fiber network with publication quality."""
    if not HAS_MPL:
        raise ImportError("matplotlib required for 2D visualization")
    
    if background == 'dark':
        bg = ShowcaseStyle.bg_color
        fiber_c = uniform_color or ShowcaseStyle.fiber_color
    else:
        bg = ShowcaseStyle.bg_color_light
        fiber_c = uniform_color or '#333333'
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=ShowcaseStyle.figsize)
    else:
        fig = ax.figure
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    
    if network.num_fibers == 0:
        if title:
            ax.text(0.5, 0.5, title, ha='center', va='center',
                   color='white' if background == 'dark' else 'black',
                   fontsize=title_fontsize, transform=ax.transAxes)
        return fig
    
    lines = []
    colors = []
    
    for fiber in network.fibers:
        pts = fiber.centerline[:, :2]
        if len(pts) >= 2:
            lines.append(pts)
            if color_by == 'length':
                colors.append(fiber.length)
            elif color_by == 'orientation':
                d = fiber.direction
                colors.append(np.arctan2(d[1], d[0]) / np.pi)
            elif color_by == 'curvature':
                curv = fiber.curvature()
                colors.append(np.mean(curv) if len(curv) > 0 else 0)
            else:
                colors.append(None)
    
    if not lines:
        return fig
    
    all_pts = np.vstack(lines)
    xmin, ymin = all_pts.min(axis=0)
    xmax, ymax = all_pts.max(axis=0)
    
    pad = 0.05
    scale = max(xmax - xmin, ymax - ymin) or 1.0
    
    normalized_lines = []
    for pts in lines:
        norm_pts = (pts - np.array([xmin, ymin])) / scale
        norm_pts = norm_pts * (1 - 2 * pad) + pad
        normalized_lines.append(norm_pts)
    
    lw = line_width if line_width is not None else ShowcaseStyle.compute_line_width(network.num_fibers)
    
    if color_by == 'uniform':
        lc = LineCollection(
            normalized_lines,
            linewidths=lw,
            colors=fiber_c,
            alpha=0.9,
            antialiased=True,
        )
    else:
        cmap = plt.get_cmap(colormap)
        vmin = min(colors) if colors else 0
        vmax = max(colors) if colors else 1
        if vmax == vmin:
            vmax = vmin + 1
        
        normalized_colors = [(c - vmin) / (vmax - vmin) for c in colors]
        lc = LineCollection(
            normalized_lines,
            linewidths=lw,
            cmap=cmap,
            array=np.array(normalized_colors),
            alpha=0.9,
            antialiased=True,
        )
    
    ax.add_collection(lc)
    
    if show_crosslinks and network.crosslinks:
        cl_pts = np.array([cl.position[:2] for cl in network.crosslinks[:2000]])
        if len(cl_pts) > 0:
            cl_norm = (cl_pts - np.array([xmin, ymin])) / scale
            cl_norm = cl_norm * (1 - 2 * pad) + pad
            ax.scatter(
                cl_norm[:, 0], cl_norm[:, 1],
                c=ShowcaseStyle.crosslink_color,
                s=crosslink_size,
                alpha=0.7,
                zorder=10,
            )
    
    if title:
        text_color = 'white' if background == 'dark' else 'black'
        ax.text(0.5, 1.02, title, ha='center', va='bottom',
               color=text_color, fontsize=title_fontsize,
               transform=ax.transAxes, fontweight='bold')
    
    if save_path:
        fig.savefig(save_path, dpi=ShowcaseStyle.dpi, 
                   bbox_inches='tight', facecolor=bg, pad_inches=0)
    
    return fig


def render_2d_grid(
    networks: List[FiberNetwork],
    titles: Optional[List[str]] = None,
    subtitles: Optional[List[str]] = None,
    ncols: int = 5,
    figsize_per_cell: Tuple[float, float] = (4, 4),
    background: str = 'dark',
    save_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> 'plt.Figure':
    """Render multiple 2D networks in a grid layout."""
    if not HAS_MPL:
        raise ImportError("matplotlib required for 2D visualization")
    
    n = len(networks)
    nrows = (n + ncols - 1) // ncols
    
    bg = ShowcaseStyle.bg_color if background == 'dark' else ShowcaseStyle.bg_color_light
    
    figsize = (figsize_per_cell[0] * ncols, figsize_per_cell[1] * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    if nrows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in row]
    
    fig.patch.set_facecolor(bg)
    
    for i, (ax, net) in enumerate(zip(axes_flat, networks)):
        title = titles[i] if titles and i < len(titles) else None
        render_2d(net, ax=ax, background=background, title=title, **kwargs)
        
        if subtitles and i < len(subtitles):
            text_color = 'white' if background == 'dark' else 'black'
            ax.text(0.5, -0.05, subtitles[i], ha='center', va='top',
                   color=text_color, fontsize=8, transform=ax.transAxes,
                   style='italic')
    
    for i in range(n, len(axes_flat)):
        axes_flat[i].axis('off')
        axes_flat[i].set_facecolor(bg)
    
    plt.tight_layout(pad=0.5)
    
    if save_path:
        fig.savefig(save_path, dpi=ShowcaseStyle.dpi,
                   bbox_inches='tight', facecolor=bg, pad_inches=0.1)
    
    return fig


def render_3d(
    network: FiberNetwork,
    color_by: str = 'uniform',
    uniform_color: Optional[str] = None,
    colormap: str = 'viridis',
    tube_radius: Optional[float] = None,
    show_crosslinks: bool = False,
    background: str = 'dark',
    camera_position: Optional[List] = None,
    window_size: Tuple[int, int] = (1024, 1024),
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Optional['pv.Plotter']:
    """Render 3D fiber network with pyvista."""
    if not HAS_PYVISTA:
        raise ImportError("pyvista required for 3D visualization")
    
    if background == 'dark':
        bg = ShowcaseStyle.bg_color
        fiber_c = uniform_color or ShowcaseStyle.fiber_color
    else:
        bg = '#ffffff'
        fiber_c = uniform_color or '#333333'
    
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.background_color = bg
    
    if network.num_fibers == 0:
        if title:
            plotter.add_text(title, font_size=12, color='white' if background == 'dark' else 'black')
        if save_path:
            plotter.screenshot(str(save_path))
        return plotter
    
    for fiber in network.fibers:
        pts = fiber.centerline
        if len(pts) < 2:
            continue
        
        spline = pv.Spline(pts, n_points=max(len(pts), 10))
        r = tube_radius if tube_radius is not None else fiber.radius
        tube = spline.tube(radius=r, capping=True)
        
        if color_by == 'uniform':
            plotter.add_mesh(tube, color=fiber_c, opacity=0.9, smooth_shading=True)
        else:
            plotter.add_mesh(tube, color=fiber_c, opacity=0.9, smooth_shading=True)
    
    if show_crosslinks and network.crosslinks:
        cl_pts = np.array([cl.position for cl in network.crosslinks])
        if len(cl_pts) > 0:
            cl_cloud = pv.PolyData(cl_pts)
            plotter.add_mesh(
                cl_cloud,
                color=ShowcaseStyle.crosslink_color,
                point_size=5,
                render_points_as_spheres=True,
            )
    
    if camera_position:
        plotter.camera_position = camera_position
    else:
        plotter.camera.azimuth = 30
        plotter.camera.elevation = 20
    
    if title:
        text_color = 'white' if background == 'dark' else 'black'
        plotter.add_text(title, font_size=14, color=text_color, position='upper_edge')
    
    if save_path:
        plotter.screenshot(str(save_path))
    
    return plotter


def render_3d_grid(
    networks: List[FiberNetwork],
    titles: Optional[List[str]] = None,
    ncols: int = 5,
    window_size_per_cell: Tuple[int, int] = (400, 400),
    background: str = 'dark',
    save_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Optional['plt.Figure']:
    """Render multiple 3D networks in a grid layout."""
    if not HAS_MPL or not HAS_PYVISTA:
        raise ImportError("matplotlib and pyvista required for 3D grid")
    
    import tempfile
    import os
    
    n = len(networks)
    nrows = (n + ncols - 1) // ncols
    
    temp_images = []
    for i, net in enumerate(networks):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
        
        title = titles[i] if titles and i < len(titles) else None
        render_3d(net, title=title, background=background,
                 window_size=window_size_per_cell, save_path=temp_path, **kwargs)
        temp_images.append(temp_path)
    
    bg = ShowcaseStyle.bg_color if background == 'dark' else ShowcaseStyle.bg_color_light
    
    figsize = (4 * ncols, 4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    if nrows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in row]
    
    fig.patch.set_facecolor(bg)
    
    for i, (ax, img_path) in enumerate(zip(axes_flat, temp_images)):
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.axis('off')
        ax.set_facecolor(bg)
    
    for i in range(n, len(axes_flat)):
        axes_flat[i].axis('off')
        axes_flat[i].set_facecolor(bg)
    
    plt.tight_layout(pad=0.5)
    
    if save_path:
        fig.savefig(save_path, dpi=ShowcaseStyle.dpi,
                   bbox_inches='tight', facecolor=bg, pad_inches=0.1)
    
    for path in temp_images:
        try:
            os.unlink(path)
        except:
            pass
    
    return fig
