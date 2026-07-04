"""
3D visualization for fiber networks using PyVista.

Provides:
- 3D fiber rendering with tube geometry
- Cross-section visualization
- Deformation animation
- Multi-material color mapping
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from fibernet.core.network import FiberNetwork


def render_network_3d(
    network: FiberNetwork,
    color_by: str = "uniform",
    colormap: str = "viridis",
    tube_radius: Optional[float] = None,
    show_crosslinks: bool = True,
    background: str = "white",
    window_size: Tuple[int, int] = (1024, 768),
    save_path: Optional[str] = None,
    show_axes: bool = True,
    camera_position: Optional[List] = None,
    off_screen: bool = True,
) -> Optional[object]:
    """Render fiber network in 3D using PyVista.
    
    Parameters
    ----------
    color_by : str
        'uniform', 'material', 'orientation', 'stress', 'length'.
    tube_radius : float, optional
        Override tube radius for rendering. Uses actual fiber radius if None.
    off_screen : bool
        If True, renders without display window (for saving images).
    
    Returns
    -------
    PyVista Plotter or None if PyVista not available.
    """
    try:
        import pyvista as pv
    except ImportError:
        print("PyVista not installed. Install with: pip install pyvista")
        return None
    
    plotter = pv.Plotter(off_screen=off_screen, window_size=window_size)
    plotter.background_color = background
    
    if network.num_fibers == 0:
        plotter.add_text("Empty network", font_size=20)
        if save_path:
            plotter.screenshot(save_path)
        return plotter
    
    for f_idx, fiber in enumerate(network.fibers):
        pts = fiber.centerline
        if len(pts) < 2:
            continue
        
        spline = pv.Spline(pts, n_points=max(len(pts), 10))
        
        r = tube_radius if tube_radius is not None else fiber.radius
        
        tube = spline.tube(radius=r, capping=True)
        
        if color_by == "uniform":
            plotter.add_mesh(tube, color="steelblue", opacity=0.9)
        elif color_by == "material":
            mat_colors = {
                "nylon": "orange", "carbon_fiber": "darkgray",
                "steel": "silver", "collagen": "pink",
                "silk": "gold", "generic_fiber": "steelblue",
            }
            c = mat_colors.get(fiber.material.name, "steelblue")
            plotter.add_mesh(tube, color=c, opacity=0.9)
        elif color_by == "orientation":
            d = fiber.direction
            angle = np.arctan2(d[1], d[0])
            from matplotlib.cm import get_cmap
            cmap = get_cmap(colormap)
            norm_val = (angle + np.pi) / (2 * np.pi)
            color = [int(c * 255) for c in cmap(norm_val)[:3]]
            plotter.add_mesh(tube, color=color, opacity=0.9)
        elif color_by == "length":
            from matplotlib.cm import get_cmap
            cmap = get_cmap(colormap)
            lengths = network.fiber_lengths()
            if len(lengths) > 0 and lengths.max() > lengths.min():
                norm_val = (fiber.length - lengths.min()) / (lengths.max() - lengths.min())
            else:
                norm_val = 0.5
            color = [int(c * 255) for c in cmap(norm_val)[:3]]
            plotter.add_mesh(tube, color=color, opacity=0.9)
        else:
            plotter.add_mesh(tube, color="steelblue", opacity=0.9)
    
    if show_crosslinks and network.crosslinks:
        cl_pts = np.array([cl.position for cl in network.crosslinks])
        if len(cl_pts) > 0:
            cl_cloud = pv.PolyData(cl_pts)
            plotter.add_mesh(cl_cloud, color="red", point_size=5, render_points_as_spheres=True)
    
    if show_axes:
        plotter.add_axes()
    
    if camera_position:
        plotter.camera_position = camera_position
    
    if save_path:
        plotter.screenshot(save_path)
    
    if not off_screen:
        plotter.show()
    
    return plotter


def render_deformation(
    network: FiberNetwork,
    displacements: np.ndarray,
    scale_factor: float = 1.0,
    save_path: Optional[str] = None,
    off_screen: bool = True,
) -> Optional[object]:
    """Render deformed fiber network."""
    try:
        import pyvista as pv
    except ImportError:
        return None
    
    plotter = pv.Plotter(off_screen=off_screen)
    plotter.background_color = "white"
    
    offset = 0
    for fiber in network.fibers:
        n_pts = len(fiber.centerline)
        pts = fiber.centerline.copy()
        
        for p_idx in range(n_pts):
            node_idx = offset + p_idx
            if node_idx * 6 + 2 < len(displacements):
                pts[p_idx, 0] += scale_factor * displacements[node_idx * 6]
                pts[p_idx, 1] += scale_factor * displacements[node_idx * 6 + 1]
                pts[p_idx, 2] += scale_factor * displacements[node_idx * 6 + 2]
        
        spline = pv.Spline(pts, n_points=max(len(pts), 10))
        tube = spline.tube(radius=fiber.radius, capping=True)
        plotter.add_mesh(tube, color="orangered", opacity=0.9)
        
        spline_orig = pv.Spline(fiber.centerline, n_points=max(len(fiber.centerline), 10))
        tube_orig = spline_orig.tube(radius=fiber.radius * 0.3, capping=True)
        plotter.add_mesh(tube_orig, color="lightgray", opacity=0.3)
        
        offset += n_pts
    
    plotter.add_axes()
    
    if save_path:
        plotter.screenshot(save_path)
    
    return plotter
