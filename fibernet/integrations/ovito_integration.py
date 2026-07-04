"""
OVITO Integration for FiberNet

Provides integration with OVITO for advanced visualization and analysis:
- Convert FiberNetwork to OVITO DataCollection
- Common neighbor analysis
- Coordination analysis
- Render high-quality images

OVITO is GPL v3 licensed: https://github.com/ovito-org/ovito
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import warnings

from fibernet.core.network import FiberNetwork


@dataclass
class OVITOAnalysisResult:
    """Result of OVITO analysis."""
    coordination_numbers: np.ndarray = None
    common_neighbors: Dict = None
    structure_types: Dict = None
    rendered_image: Optional[str] = None
    
    def to_dict(self) -> Dict:
        result = {
            'mean_coordination': float(np.mean(self.coordination_numbers)) if self.coordination_numbers is not None else 0.0,
        }
        if self.structure_types:
            result['structure_types'] = self.structure_types
        return result


class OVITOBridge:
    """Bridge between FiberNetwork and OVITO.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to visualize/analyze.
    """
    
    def __init__(self, network: FiberNetwork):
        self.network = network
        self._ovito_available = False
        try:
            import ovito
            self.ovito = ovito
            self._ovito_available = True
        except ImportError:
            warnings.warn(
                "OVITO Python bindings not available. "
                "Install with: pip install ovito"
            )
    
    def to_data_collection(self):
        """Convert FiberNetwork to OVITO DataCollection.
        
        Returns
        -------
        data : ovito.DataCollection or dict
            OVITO data collection, or dict if OVITO not available.
        """
        # Collect all coordinates
        coords_list = []
        fiber_ids = []
        
        for i, fiber in enumerate(self.network.fibers):
            cl = fiber.centerline
            n_pts = len(cl)
            coords_list.append(cl)
            fiber_ids.extend([i] * n_pts)
        
        all_coords = np.vstack(coords_list)
        fiber_ids = np.array(fiber_ids)
        
        if self._ovito_available:
            from ovito.data import Particles, DataCollection
            
            particles = Particles()
            particles.add_property('Position', all_coords)
            particles.add_property('Fiber ID', fiber_ids)
            
            data = DataCollection()
            data.add_particles(particles)
            return data
        else:
            return {
                'positions': all_coords,
                'fiber_ids': fiber_ids,
                'num_particles': len(all_coords),
            }
    
    def render(
        self,
        output_file: str = "fiber_network.png",
        viewport: str = 'perspective',
        resolution: tuple = (1920, 1080),
        color_scheme: str = 'fiber_id',
    ) -> str:
        """Render high-quality image of fiber network.
        
        Parameters
        ----------
        output_file : str
            Output image filename.
        viewport : str
            Viewport type: 'perspective', 'orthographic'
        resolution : tuple
            Image resolution (width, height).
        color_scheme : str
            Color scheme: 'fiber_id', 'radius', 'length'
        
        Returns
        -------
        output_file : str
            Path to rendered image.
        """
        if not self._ovito_available:
            warnings.warn("OVITO not available for rendering")
            return None
        
        try:
            data = self.to_data_collection()
            
            # Create pipeline
            from ovito.vis import Viewport, Pipeline
            
            pipeline = Pipeline(source=data)
            
            # Set up viewport
            vp = Viewport()
            vp.type = Viewport.Type.Perspective if viewport == 'perspective' else Viewport.Type.Ortho
            vp.camera_dir = (-1, -1, -0.5)
            vp.fov = 0.5
            
            # Render
            vp.render_image(
                size=resolution,
                filename=output_file,
            )
            
            return output_file
            
        except Exception as e:
            warnings.warn(f"OVITO rendering failed: {e}")
            return None
    
    def coordination_analysis(self, distance_threshold: float = None) -> np.ndarray:
        """Compute coordination numbers (number of nearby fibers).
        
        Parameters
        ----------
        distance_threshold : float, optional
            Distance threshold for coordination. Default: 2x average fiber radius.
        
        Returns
        -------
        coordination : np.ndarray
            Coordination number for each fiber.
        """
        if distance_threshold is None:
            avg_radius = np.mean([f.radius for f in self.network.fibers])
            distance_threshold = avg_radius * 4
        
        n = self.network.num_fibers
        coord = np.zeros(n, dtype=int)
        
        # Compute fiber centers
        centers = np.array([f.centerline.mean(axis=0) for f in self.network.fibers])
        
        # Count neighbors within threshold
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < distance_threshold:
                    coord[i] += 1
                    coord[j] += 1
        
        return coord
    
    def write_xyz(self, filename: str) -> str:
        """Export network as XYZ file for OVITO.
        
        Parameters
        ----------
        filename : str
            Output filename.
        
        Returns
        -------
        filename : str
            Path to written file.
        """
        coords_list = []
        for fiber in self.network.fibers:
            coords_list.append(fiber.centerline)
        
        all_coords = np.vstack(coords_list)
        n_atoms = len(all_coords)
        
        with open(filename, 'w') as f:
            f.write(f"{n_atoms}\n")
            f.write("FiberNet export\n")
            for coord in all_coords:
                f.write(f"C {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")
        
        return filename
    
    def write_vtk(self, filename: str) -> str:
        """Export network as VTK file for OVITO/ParaView.
        
        Parameters
        ----------
        filename : str
            Output filename.
        
        Returns
        -------
        filename : str
            Path to written file.
        """
        try:
            from fibernet.io.formats import to_vtk
            return to_vtk(self.network, filename)
        except ImportError:
            # Simple VTK legacy format
            coords_list = []
            line_cells = []
            offset = 0
            
            for fiber in self.network.fibers:
                cl = fiber.centerline
                n_pts = len(cl)
                coords_list.append(cl)
                line_cells.append((offset, offset + n_pts))
                offset += n_pts
            
            all_coords = np.vstack(coords_list)
            n_points = len(all_coords)
            n_lines = len(line_cells)
            
            with open(filename, 'w') as f:
                f.write("# vtk DataFile Version 3.0\n")
                f.write("FiberNet export\n")
                f.write("ASCII\n")
                f.write("DATASET UNSTRUCTURED_GRID\n")
                f.write(f"POINTS {n_points} float\n")
                for c in all_coords:
                    f.write(f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
                
                total_size = sum(end - start + 1 for start, end in line_cells)
                f.write(f"\nCELLS {n_lines} {total_size}\n")
                for start, end in line_cells:
                    n = end - start
                    indices = ' '.join(str(i) for i in range(start, end))
                    f.write(f"{n} {indices}\n")
                
                f.write(f"\nCELL_TYPES {n_lines}\n")
                for _ in range(n_lines):
                    f.write("4\n")  # VTK_POLY_LINE
            
            return filename


def render_network_ovito(
    network: FiberNetwork,
    output_file: str = "fiber_network.png",
    **kwargs,
) -> str:
    """Convenience function for OVITO rendering.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to render.
    output_file : str
        Output image filename.
    
    Returns
    -------
    output_file : str
        Path to rendered image.
    """
    bridge = OVITOBridge(network)
    return bridge.render(output_file, **kwargs)
