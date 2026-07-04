"""
PyVista Integration for Advanced 3D Visualization

Provides high-quality 3D visualization and rendering of fiber networks using PyVista.

Features:
- Interactive 3D visualization
- Color coding by properties (orientation, length, stress, etc.)
- Cross-section views
- Animation support
- Export to various formats (VTK, STL, OBJ, etc.)

References:
- PyVista: https://docs.pyvista.org/
"""

import numpy as np
from typing import Optional, Tuple, Union
from pathlib import Path

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    pv = None

from fibernet.core.network import FiberNetwork


class PyVistaVisualizer:
    """
    Advanced 3D visualization for fiber networks using PyVista.
    
    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.pyvista_viz import PyVistaVisualizer
    >>> 
    >>> # Generate network
    >>> net = gen.random_3d(num_fibers=100, box_size=(50, 50, 50))
    >>> 
    >>> # Create visualizer
    >>> viz = PyVistaVisualizer(net)
    >>> 
    >>> # Show in Jupyter
    >>> viz.show_jupyter()
    >>> 
    >>> # Save screenshot
    >>> viz.save_screenshot('network.png')
    """
    
    def __init__(self, network: FiberNetwork):
        """
        Initialize visualizer with a fiber network.
        
        Parameters
        ----------
        network : FiberNetwork
            Network to visualize
        """
        if not PYVISTA_AVAILABLE:
            raise ImportError(
                "PyVista is not available. Install with: pip install pyvista"
            )
        
        self.network = network
        self.mesh = None
        self.plotter = None
        
        # Build mesh
        self._build_mesh()
    
    def _build_mesh(self):
        """Build PyVista mesh from network."""
        points = []
        lines = []
        lengths = []
        
        point_offset = 0
        for fiber in self.network.fibers:
            # Add line segment
            centerline = fiber.centerline
            num_points = len(centerline)
            
            # Add points
            points.extend(centerline)
            
            # Line connectivity
            line = [num_points] + list(range(point_offset, point_offset + num_points))
            lines.extend(line)
            
            point_offset += num_points
            
            # Store fiber properties for coloring
            lengths.append(fiber.length)
        
        # Convert to numpy arrays
        points = np.array(points)
        lines = np.array(lines)
        
        # Create PolyData
        self.mesh = pv.PolyData(points, lines=lines)
        
        # Add length as scalar
        n_cells = len(self.network.fibers)
        self.mesh.cell_data['length'] = np.array(lengths)[:n_cells]
    
    def show(
        self,
        color: str = 'lightblue',
        background: str = 'white',
        show_edges: bool = False,
        line_width: float = 3.0,
        window_size: Tuple[int, int] = (1024, 768),
        **kwargs
    ):
        """
        Display interactive 3D visualization.
        
        Parameters
        ----------
        color : str
            Fiber color
        background : str
            Background color
        show_edges : bool
            Show edges
        line_width : float
            Line width
        window_size : tuple
            Window size (width, height)
        **kwargs
            Additional arguments passed to PyVista plotter
        """
        self.plotter = pv.Plotter(window_size=window_size)
        self.plotter.set_background(background)
        
        self.plotter.add_mesh(
            self.mesh,
            color=color,
            line_width=line_width,
            show_edges=show_edges,
            **kwargs
        )
        
        self.plotter.add_axes()
        self.plotter.show()
    
    def show_jupyter(
        self,
        color: str = 'lightblue',
        background: str = 'white',
        line_width: float = 3.0,
        window_size: Tuple[int, int] = (800, 600),
        **kwargs
    ):
        """
        Display visualization in Jupyter notebook.
        
        Parameters
        ----------
        color : str
            Fiber color
        background : str
            Background color
        line_width : float
            Line width
        window_size : tuple
            Window size
        **kwargs
            Additional arguments
        """
        # Set Jupyter backend
        pv.set_jupyter_backend('static')
        
        self.plotter = pv.Plotter(window_size=window_size)
        self.plotter.set_background(background)
        
        self.plotter.add_mesh(
            self.mesh,
            color=color,
            line_width=line_width,
            **kwargs
        )
        
        self.plotter.add_axes()
        return self.plotter.show()
    
    def color_by_property(
        self,
        property_name: str,
        colormap: str = 'viridis',
        clim: Optional[Tuple[float, float]] = None,
        **kwargs
    ):
        """
        Color fibers by a property.
        
        Parameters
        ----------
        property_name : str
            Property to color by ('length', 'orientation', 'radius', etc.)
        colormap : str
            Colormap name
        clim : tuple, optional
            Color limits (min, max)
        **kwargs
            Additional arguments
        """
        if property_name == 'length':
            values = np.array([f.length for f in self.network.fibers])
        elif property_name == 'radius':
            values = np.array([f.radius for f in self.network.fibers])
        elif property_name == 'orientation':
            # Orientation angle with respect to z-axis
            z_axis = np.array([0, 0, 1])
            values = []
            for fiber in self.network.fibers:
                direction = fiber.end_point - fiber.start_point
                direction = direction / np.linalg.norm(direction)
                angle = np.degrees(np.arccos(np.abs(np.dot(direction, z_axis))))
                values.append(angle)
            values = np.array(values)
        else:
            raise ValueError(f"Unknown property: {property_name}")
        
        self.mesh.cell_data[property_name] = values
        
        if self.plotter:
            self.plotter.clear()
            self.plotter.add_mesh(
                self.mesh,
                scalars=property_name,
                cmap=colormap,
                clim=clim,
                **kwargs
            )
    
    def save_screenshot(
        self,
        filename: Union[str, Path],
        color: str = 'lightblue',
        background: str = 'white',
        line_width: float = 3.0,
        window_size: Tuple[int, int] = (1920, 1080),
        transparent_background: bool = False,
        **kwargs
    ):
        """
        Save visualization as image.
        
        Parameters
        ----------
        filename : str or Path
            Output filename (png, jpg, etc.)
        color : str
            Fiber color
        background : str
            Background color
        line_width : float
            Line width
        window_size : tuple
            Window size
        transparent_background : bool
            Use transparent background
        **kwargs
            Additional arguments
        """
        plotter = pv.Plotter(off_screen=True, window_size=window_size)
        plotter.set_background(background)
        
        plotter.add_mesh(
            self.mesh,
            color=color,
            line_width=line_width,
            **kwargs
        )
        
        plotter.add_axes()
        plotter.screenshot(str(filename), transparent_background=transparent_background)
        plotter.close()
    
    def export_vtk(self, filename: Union[str, Path]):
        """
        Export mesh to VTK format.
        
        Parameters
        ----------
        filename : str or Path
            Output filename (.vtk or .vtu)
        """
        self.mesh.save(str(filename))
    
    def add_cross_section(
        self,
        normal: Tuple[float, float, float] = (0, 0, 1),
        origin: Tuple[float, float, float] = (0, 0, 0),
        color: str = 'red',
        opacity: float = 0.3,
        **kwargs
    ):
        """
        Add a cross-section plane to visualization.
        
        Parameters
        ----------
        normal : tuple
            Normal vector of the plane
        origin : tuple
            Origin point of the plane
        color : str
            Plane color
        opacity : float
            Plane opacity (0-1)
        **kwargs
            Additional arguments
        """
        if not self.plotter:
            self.plotter = pv.Plotter()
        
        # Create plane
        plane = pv.Plane(center=origin, direction=normal, i_size=100, j_size=100)
        
        self.plotter.add_mesh(
            plane,
            color=color,
            opacity=opacity,
            **kwargs
        )
    
    def animate_rotation(
        self,
        filename: Union[str, Path],
        n_frames: int = 36,
        color: str = 'lightblue',
        background: str = 'white',
        line_width: float = 3.0,
        window_size: Tuple[int, int] = (800, 600),
        **kwargs
    ):
        """
        Create rotation animation.
        
        Parameters
        ----------
        filename : str or Path
            Output filename (.gif or .mp4)
        n_frames : int
            Number of frames
        color : str
            Fiber color
        background : str
            Background color
        line_width : float
            Line width
        window_size : tuple
            Window size
        **kwargs
            Additional arguments
        """
        plotter = pv.Plotter(off_screen=True, window_size=window_size)
        plotter.set_background(background)
        
        plotter.add_mesh(
            self.mesh,
            color=color,
            line_width=line_width,
            **kwargs
        )
        
        plotter.add_axes()
        
        # Open animation
        plotter.open_gif(str(filename))
        
        # Rotate and capture frames
        for i in range(n_frames):
            plotter.camera.azimuth(360 / n_frames)
            plotter.write_frame()
        
        plotter.close()


def visualize_network_3d(
    network: FiberNetwork,
    color: str = 'lightblue',
    background: str = 'white',
    line_width: float = 3.0,
    window_size: Tuple[int, int] = (1024, 768),
    **kwargs
) -> Optional[pv.Plotter]:
    """
    Quick visualization of a fiber network.
    
    Parameters
    ----------
    network : FiberNetwork
        Network to visualize
    color : str
        Fiber color
    background : str
        Background color
    line_width : float
        Line width
    window_size : tuple
        Window size
    **kwargs
        Additional arguments
    
    Returns
    -------
    plotter : pv.Plotter or None
        PyVista plotter (if available)
    
    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.pyvista_viz import visualize_network_3d
    >>> net = gen.random_3d(num_fibers=100, box_size=(50, 50, 50))
    >>> visualize_network_3d(net)
    """
    viz = PyVistaVisualizer(network)
    viz.show(color=color, background=background, line_width=line_width, 
             window_size=window_size, **kwargs)
    return viz.plotter


