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
            from matplotlib import colormaps
            cmap = colormaps.get_cmap(colormap)
            norm_val = (angle + np.pi) / (2 * np.pi)
            color = [int(c * 255) for c in cmap(norm_val)[:3]]
            plotter.add_mesh(tube, color=color, opacity=0.9)
        elif color_by == "length":
            from matplotlib import colormaps
            cmap = colormaps.get_cmap(colormap)
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


def render_structure_graph_3d(
    graph,
    sim_result=None,
    *,
    color_by="uniform",
    colormap="viridis",
    tube_radius=None,
    show_nodes=True,
    background="white",
    window_size=(1280, 960),
    save_path=None,
    off_screen=True,
    camera_position=None,
):
    """Render StructureGraph in 3D using PyVista with tube geometry.

    Parameters
    ----------
    graph : StructureGraph
        3D structure graph.
    sim_result : SimResult, optional
        Simulation result for stress/displacement coloring.
    color_by : str
        "uniform", "stress" (edge force), "stretch", "displacement", "orientation".
    tube_radius : float, optional
        Override tube radius. Default: uses edge.radius or 0.05.
    show_nodes : bool
        Render node junctions as spheres.
    """
    try:
        import pyvista as pv
    except ImportError:
        print("PyVista not installed. Install with: pip install pyvista")
        return None

    plotter = pv.Plotter(off_screen=off_screen, window_size=window_size)
    plotter.background_color = background

    if graph.num_edges == 0:
        plotter.add_text("Empty graph", font_size=20)
        if save_path:
            plotter.screenshot(save_path)
        return plotter

    # Use deformed positions if available
    node_ids = sorted(graph.nodes.keys())
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    if sim_result is not None and sim_result.deformed_positions is not None:
        positions = sim_result.deformed_positions
    else:
        positions = np.array([graph.nodes[nid].position for nid in node_ids])

    # Determine edge colors
    edges_list = list(graph.edges.values())
    from matplotlib import colormaps

    if color_by == "stress" and sim_result is not None and sim_result.edge_forces is not None:
        edge_vals = np.abs(sim_result.edge_forces)
        cmap = colormaps.get_cmap("RdYlBu_r")
        vmin, vmax = edge_vals.min(), edge_vals.max()
        if vmax - vmin < 1e-12:
            vmax = vmin + 1
    elif color_by == "stretch" and sim_result is not None and sim_result.edge_stretches is not None:
        edge_vals = sim_result.edge_stretches
        cmap = colormaps.get_cmap("hot")
        vmin, vmax = edge_vals.min(), edge_vals.max()
        if vmax - vmin < 1e-12:
            vmax = vmin + 1
    elif color_by == "displacement" and sim_result is not None and sim_result.displacements is not None:
        edge_vals = None  # computed per-edge from displacement
        cmap = colormaps.get_cmap("coolwarm")
        disp_mags = np.linalg.norm(sim_result.displacements[:, :3], axis=1)
        vmin, vmax = disp_mags.min(), disp_mags.max()
        if vmax - vmin < 1e-12:
            vmax = vmin + 1
    elif color_by == "orientation":
        edge_vals = None
        cmap = colormaps.get_cmap("hsv")
        vmin, vmax = 0, 1
    else:
        edge_vals = None
        cmap = None
        vmin, vmax = 0, 1

    for ei, edge in enumerate(edges_list):
        i_idx = node_to_idx.get(edge.node_i, 0)
        j_idx = node_to_idx.get(edge.node_j, 0)
        pi = positions[i_idx]
        pj = positions[j_idx]

        if edge.internal_points is not None and len(edge.internal_points) > 0:
            pts = np.vstack([pi[None, :], edge.internal_points, pj[None, :]])
        else:
            pts = np.vstack([pi[None, :], pj[None, :]])

        r = tube_radius if tube_radius is not None else edge.radius

        try:
            spline = pv.Spline(pts, n_points=max(len(pts) * 3, 6))
            tube = spline.tube(radius=r, capping=True)

            # Determine color
            if cmap is not None:
                if edge_vals is not None and ei < len(edge_vals):
                    val = (edge_vals[ei] - vmin) / max(vmax - vmin, 1e-12)
                elif color_by == "displacement":
                    di = disp_mags[i_idx] if i_idx < len(disp_mags) else 0
                    dj = disp_mags[j_idx] if j_idx < len(disp_mags) else 0
                    val = ((di + dj) / 2 - vmin) / max(vmax - vmin, 1e-12)
                elif color_by == "orientation":
                    d = pj - pi
                    val = (np.arctan2(d[1], d[0]) + np.pi) / (2 * np.pi)
                else:
                    val = 0.5
                color = [int(c * 255) for c in cmap(val)[:3]]
                plotter.add_mesh(tube, color=color, opacity=0.9)
            else:
                plotter.add_mesh(tube, color="steelblue", opacity=0.9)
        except Exception:
            pass

    # Show nodes as spheres
    if show_nodes and len(positions) > 0:
        node_cloud = pv.PolyData(positions)
        plotter.add_mesh(node_cloud, color="red", point_size=4,
                         render_points_as_spheres=True, opacity=0.7)

    plotter.add_axes()

    if camera_position:
        plotter.camera_position = camera_position

    if save_path:
        plotter.screenshot(save_path)

    if not off_screen:
        plotter.show()

    return plotter


def render_stress_pyvista(
    graph,
    sim_result,
    *,
    save_path=None,
    window_size=(1280, 960),
    background="white",
    off_screen=True,
):
    """Convenience: PyVista stress rendering on deformed 3D structure."""
    return render_structure_graph_3d(
        graph, sim_result,
        color_by="stress",
        save_path=save_path,
        window_size=window_size,
        background=background,
        off_screen=off_screen,
    )
