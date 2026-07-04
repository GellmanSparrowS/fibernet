"""
Interactive visualization using Plotly.

Provides web-based interactive 3D visualization of fiber networks
with features like rotation, zoom, and hover information.
"""

import numpy as np
from typing import Optional, Dict, List
from ..core import FiberNetwork


def visualize_interactive(
    network: FiberNetwork,
    color_by: str = 'fiber_index',
    show_crosslinks: bool = True,
    show_bounding_box: bool = True,
    title: str = "Fiber Network",
    width: int = 800,
    height: int = 600,
    fiber_opacity: float = 0.8,
    crosslink_size: float = 3.0,
    background_color: str = 'white',
) -> 'go.Figure':
    """
    Create interactive 3D visualization using Plotly.
    
    Parameters
    ----------
    network : FiberNetwork
        The fiber network to visualize
    color_by : str
        Coloring scheme: 'fiber_index', 'length', 'orientation', 'stress'
    show_crosslinks : bool
        Whether to show crosslink points
    show_bounding_box : bool
        Whether to show bounding box
    title : str
        Plot title
    width : int
        Figure width in pixels
    height : int
        Figure height in pixels
    fiber_opacity : float
        Fiber opacity (0-1)
    crosslink_size : float
        Crosslink marker size
    background_color : str
        Background color
    
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive figure
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Plotly is required for interactive visualization. "
                         "Install with: pip install plotly")
    
    fig = go.Figure()
    
    # Add fibers
    for i, fiber in enumerate(network.fibers):
        centerline = fiber.centerline
        
        # Compute color
        if color_by == 'fiber_index':
            color = i / len(network.fibers)
            colorscale = 'Viridis'
        elif color_by == 'length':
            color = fiber.length
            colorscale = 'Hot'
        elif color_by == 'orientation':
            direction = centerline[-1] - centerline[0]
            direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else direction
            color = np.arctan2(direction[1], direction[0]) / np.pi + 0.5
            colorscale = 'HSV'
        else:
            color = 0.5
            colorscale = 'Blues'
        
        fig.add_trace(go.Scatter3d(
            x=centerline[:, 0],
            y=centerline[:, 1],
            z=centerline[:, 2] if centerline.shape[1] > 2 else np.zeros(len(centerline)),
            mode='lines',
            line=dict(
                color=color,
                width=fiber.radius * 50,
                colorscale=colorscale,
            ),
            opacity=fiber_opacity,
            name=f'Fiber {i}',
            hovertemplate=f'Fiber {i}<br>Length: {fiber.length:.2f}<br>Radius: {fiber.radius:.3f}<extra></extra>',
            showlegend=False,
        ))
    
    # Add crosslinks
    if show_crosslinks and len(network.crosslinks) > 0:
        cl_x = []
        cl_y = []
        cl_z = []
        
        for cl in network.crosslinks:
            pos = cl.position
            cl_x.append(pos[0])
            cl_y.append(pos[1])
            cl_z.append(pos[2] if len(pos) > 2 else 0)
        
        fig.add_trace(go.Scatter3d(
            x=cl_x,
            y=cl_y,
            z=cl_z,
            mode='markers',
            marker=dict(
                size=crosslink_size,
                color='red',
                opacity=0.8,
            ),
            name='Crosslinks',
            hovertemplate='Crosslink<br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>',
        ))
    
    # Add bounding box
    if show_bounding_box:
        all_points = np.vstack([f.centerline for f in network.fibers])
        min_pt = np.min(all_points, axis=0)
        max_pt = np.max(all_points, axis=0)
        
        # 12 edges of the box
        edges = [
            ([min_pt[0], max_pt[0]], [min_pt[1], min_pt[1]], [min_pt[2], min_pt[2]]),
            ([min_pt[0], max_pt[0]], [max_pt[1], max_pt[1]], [min_pt[2], min_pt[2]]),
            ([min_pt[0], min_pt[0]], [min_pt[1], max_pt[1]], [min_pt[2], min_pt[2]]),
            ([max_pt[0], max_pt[0]], [min_pt[1], max_pt[1]], [min_pt[2], min_pt[2]]),
            ([min_pt[0], max_pt[0]], [min_pt[1], min_pt[1]], [max_pt[2], max_pt[2]]),
            ([min_pt[0], max_pt[0]], [max_pt[1], max_pt[1]], [max_pt[2], max_pt[2]]),
            ([min_pt[0], min_pt[0]], [min_pt[1], max_pt[1]], [max_pt[2], max_pt[2]]),
            ([max_pt[0], max_pt[0]], [min_pt[1], max_pt[1]], [max_pt[2], max_pt[2]]),
            ([min_pt[0], min_pt[0]], [min_pt[1], min_pt[1]], [min_pt[2], max_pt[2]]),
            ([max_pt[0], max_pt[0]], [min_pt[1], min_pt[1]], [min_pt[2], max_pt[2]]),
            ([min_pt[0], min_pt[0]], [max_pt[1], max_pt[1]], [min_pt[2], max_pt[2]]),
            ([max_pt[0], max_pt[0]], [max_pt[1], max_pt[1]], [min_pt[2], max_pt[2]]),
        ]
        
        for edge in edges:
            z_vals = edge[2] if len(edge[2]) == 2 else [0, 0]
            fig.add_trace(go.Scatter3d(
                x=edge[0],
                y=edge[1],
                z=z_vals,
                mode='lines',
                line=dict(color='gray', width=1),
                opacity=0.3,
                showlegend=False,
                hoverinfo='skip',
            ))
    
    # Layout
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            bgcolor=background_color,
        ),
        showlegend=show_crosslinks,
    )
    
    return fig


def visualize_stress_field(
    network: FiberNetwork,
    stress_values: np.ndarray,
    title: str = "Stress Distribution",
    colorscale: str = 'RdBu',
    width: int = 800,
    height: int = 600,
) -> 'go.Figure':
    """
    Visualize stress distribution on fiber network.
    
    Parameters
    ----------
    network : FiberNetwork
        The fiber network
    stress_values : np.ndarray
        Stress values for each fiber
    title : str
        Plot title
    colorscale : str
        Plotly colorscale
    width : int
        Figure width
    height : int
        Figure height
    
    Returns
    -------
    plotly.graph_objects.Figure
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Plotly is required. Install with: pip install plotly")
    
    fig = go.Figure()
    
    for i, fiber in enumerate(network.fibers):
        centerline = fiber.centerline
        stress = stress_values[i] if i < len(stress_values) else 0.0
        
        fig.add_trace(go.Scatter3d(
            x=centerline[:, 0],
            y=centerline[:, 1],
            z=centerline[:, 2] if centerline.shape[1] > 2 else np.zeros(len(centerline)),
            mode='lines',
            line=dict(
                color=stress,
                width=fiber.radius * 50,
                colorscale=colorscale,
                cmin=np.min(stress_values),
                cmax=np.max(stress_values),
                colorbar=dict(title='Stress (Pa)'),
            ),
            name=f'Fiber {i}',
            hovertemplate=f'Stress: {stress:.2e} Pa<extra></extra>',
            showlegend=False,
        ))
    
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
        ),
    )
    
    return fig


def visualize_comparison(
    networks: List[FiberNetwork],
    labels: List[str],
    width: int = 1200,
    height: int = 400,
) -> 'go.Figure':
    """
    Visualize multiple networks side-by-side for comparison.
    
    Parameters
    ----------
    networks : List[FiberNetwork]
        List of networks to compare
    labels : List[str]
        Labels for each network
    width : int
        Figure width
    height : int
        Figure height
    
    Returns
    -------
    plotly.graph_objects.Figure
    """
    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Plotly is required. Install with: pip install plotly")
    
    fig = make_subplots(
        rows=1,
        cols=len(networks),
        specs=[[{'type': 'scatter3d'}] * len(networks)],
        subplot_titles=labels,
    )
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (network, label) in enumerate(zip(networks, labels)):
        col = i + 1
        color = colors[i % len(colors)]
        
        for fiber in network.fibers:
            centerline = fiber.centerline
            fig.add_trace(
                go.Scatter3d(
                    x=centerline[:, 0],
                    y=centerline[:, 1],
                    z=centerline[:, 2] if centerline.shape[1] > 2 else np.zeros(len(centerline)),
                    mode='lines',
                    line=dict(color=color, width=2),
                    opacity=0.7,
                    showlegend=False,
                    hoverinfo='skip',
                ),
                row=1,
                col=col,
            )
    
    fig.update_layout(
        width=width,
        height=height,
        title_text="Network Comparison",
    )
    
    return fig


def export_html(
    figure: 'go.Figure',
    filename: str = "network.html",
    auto_open: bool = False,
):
    """
    Export Plotly figure to HTML file.
    
    Parameters
    ----------
    figure : plotly.graph_objects.Figure
        Figure to export
    filename : str
        Output HTML filename
    auto_open : bool
        Whether to open in browser
    """
    figure.write_html(filename, auto_open=auto_open)
