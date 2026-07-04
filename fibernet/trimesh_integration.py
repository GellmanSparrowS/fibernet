"""
Trimesh Integration for Mesh Operations

Provides advanced mesh operations and analysis for fiber networks using Trimesh.

Features:
- Convert fiber networks to meshes
- Boolean operations (union, intersection, difference)
- Mesh analysis (volume, surface area, etc.)
- Mesh repair and simplification
- Export to various formats

References:
- Trimesh: https://trimesh.org/
"""

import numpy as np
from typing import Optional, Tuple, List, Union
from pathlib import Path

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    trimesh = None

from fibernet.core.network import FiberNetwork
from fibernet.core.fiber import Fiber


class TrimeshConverter:
    """
    Convert fiber networks to trimesh objects.
    
    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.trimesh_integration import TrimeshConverter
    >>> 
    >>> # Generate network
    >>> net = gen.random_straight_3d(num_fibers=50, box_size=(50, 50, 50))
    >>> 
    >>> # Convert to mesh
    >>> converter = TrimeshConverter(net)
    >>> mesh = converter.to_mesh()
    >>> 
    >>> # Analyze mesh
    >>> print(f"Volume: {mesh.volume:.2f}")
    >>> print(f"Surface area: {mesh.area:.2f}")
    """
    
    def __init__(
        self,
        network: FiberNetwork,
        segments_per_fiber: int = 8,
        radial_segments: int = 6
    ):
        """
        Initialize converter.
        
        Parameters
        ----------
        network : FiberNetwork
            Network to convert
        segments_per_fiber : int
            Number of longitudinal segments per fiber
        radial_segments : int
            Number of radial segments (polygon approximation of circle)
        """
        if not TRIMESH_AVAILABLE:
            raise ImportError(
                "Trimesh is not available. Install with: pip install trimesh"
            )
        
        self.network = network
        self.segments_per_fiber = segments_per_fiber
        self.radial_segments = radial_segments
    
    def _create_cylinder_mesh(
        self,
        start: np.ndarray,
        end: np.ndarray,
        radius: float
    ) -> trimesh.Trimesh:
        """
        Create a cylinder mesh between two points.
        
        Parameters
        ----------
        start : np.ndarray
            Start point (3D)
        end : np.ndarray
            End point (3D)
        radius : float
            Cylinder radius
        
        Returns
        -------
        mesh : trimesh.Trimesh
            Cylinder mesh
        """
        # Compute length and direction
        direction = end - start
        length = np.linalg.norm(direction)
        
        if length < 1e-10:
            return trimesh.Trimesh()
        
        # Create cylinder using trimesh
        cylinder = trimesh.creation.cylinder(
            radius=radius,
            height=length,
            sections=self.radial_segments
        )
        
        # Align cylinder with direction
        # Trimesh cylinders are aligned with Z-axis by default
        z_axis = np.array([0, 0, 1])
        direction_norm = direction / length
        
        # Compute rotation
        rotation = trimesh.geometry.align_vectors(z_axis, direction_norm)
        
        # Apply rotation
        cylinder.apply_transform(rotation)
        
        # Translate to correct position
        midpoint = (start + end) / 2
        cylinder.apply_translation(midpoint)
        
        return cylinder
    
    def to_mesh(self, merge: bool = True) -> trimesh.Trimesh:
        """
        Convert network to a single mesh.
        
        Parameters
        ----------
        merge : bool
            If True, merge all fibers into single mesh.
            If False, return list of meshes.
        
        Returns
        -------
        mesh : trimesh.Trimesh or list
            Merged mesh or list of meshes
        """
        meshes = []
        
        for fiber in self.network.fibers:
            start = fiber.start_point
            end = fiber.end_point
            
            cylinder = self._create_cylinder_mesh(start, end, fiber.radius)
            meshes.append(cylinder)
        
        if merge:
            # Merge all meshes
            merged = trimesh.util.concatenate(meshes)
            return merged
        else:
            return meshes
    
    def to_scene(self) -> trimesh.Scene:
        """
        Convert network to a trimesh scene.
        
        Returns
        -------
        scene : trimesh.Scene
            Scene containing all fibers
        """
        scene = trimesh.Scene()
        
        for i, fiber in enumerate(self.network.fibers):
            start = fiber.start_point
            end = fiber.end_point
            
            cylinder = self._create_cylinder_mesh(start, end, fiber.radius)
            scene.add_geometry(cylinder, node_name=f'fiber_{i}')
        
        return scene


def network_to_trimesh(
    network: FiberNetwork,
    segments_per_fiber: int = 8,
    radial_segments: int = 6,
    merge: bool = True
) -> Union[trimesh.Trimesh, List[trimesh.Trimesh]]:
    """
    Convert fiber network to trimesh.
    
    Parameters
    ----------
    network : FiberNetwork
        Network to convert
    segments_per_fiber : int
        Number of longitudinal segments per fiber
    radial_segments : int
        Number of radial segments
    merge : bool
        If True, merge all fibers into single mesh
    
    Returns
    -------
    mesh : trimesh.Trimesh or list
        Merged mesh or list of meshes
    
    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.trimesh_integration import network_to_trimesh
    >>> net = gen.random_straight_3d(num_fibers=50, box_size=(50, 50, 50))
    >>> mesh = network_to_trimesh(net)
    >>> print(f"Volume: {mesh.volume:.2f}")
    """
    converter = TrimeshConverter(
        network,
        segments_per_fiber=segments_per_fiber,
        radial_segments=radial_segments
    )
    return converter.to_mesh(merge=merge)


def analyze_mesh_properties(mesh: trimesh.Trimesh) -> dict:
    """
    Analyze mesh properties.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to analyze
    
    Returns
    -------
    properties : dict
        Dictionary of mesh properties
    
    Examples
    --------
    >>> mesh = network_to_trimesh(net)
    >>> props = analyze_mesh_properties(mesh)
    >>> print(f"Volume: {props['volume']:.2f}")
    >>> print(f"Surface area: {props['surface_area']:.2f}")
    """
    properties = {
        'volume': float(mesh.volume),
        'surface_area': float(mesh.area),
        'bounds': mesh.bounds.tolist(),
        'centroid': mesh.centroid.tolist(),
        'is_watertight': mesh.is_watertight,
        'is_winding_consistent': mesh.is_winding_consistent,
        'num_vertices': len(mesh.vertices),
        'num_faces': len(mesh.faces),
    }
    
    return properties


def boolean_operation(
    mesh1: trimesh.Trimesh,
    mesh2: trimesh.Trimesh,
    operation: str = 'union'
) -> trimesh.Trimesh:
    """
    Perform boolean operation between two meshes.
    
    Parameters
    ----------
    mesh1 : trimesh.Trimesh
        First mesh
    mesh2 : trimesh.Trimesh
        Second mesh
    operation : str
        Operation type: 'union', 'intersection', 'difference'
    
    Returns
    -------
    result : trimesh.Trimesh
        Result mesh
    
    Raises
    ------
    ImportError
        If boolean backend (manifold3d or blender) is not available
    
    Examples
    --------
    >>> mesh1 = network_to_trimesh(net1)
    >>> mesh2 = network_to_trimesh(net2)
    >>> union = boolean_operation(mesh1, mesh2, 'union')
    """
    if operation == 'union':
        return mesh1.union(mesh2)
    elif operation == 'intersection':
        return mesh1.intersection(mesh2)
    elif operation == 'difference':
        return mesh1.difference(mesh2)
    else:
        raise ValueError(f"Unknown operation: {operation}")


def repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Repair mesh (fix normals, remove degenerates, etc.).
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to repair
    
    Returns
    -------
    repaired : trimesh.Trimesh
        Repaired mesh
    
    Examples
    --------
    >>> mesh = network_to_trimesh(net)
    >>> repaired = repair_mesh(mesh)
    >>> print(f"Is watertight: {repaired.is_watertight}")
    """
    # Fix normals
    mesh.fix_normals()
    
    # Remove degenerate faces (use nondegenerate_faces mask)
    good_faces = mesh.nondegenerate_faces()
    if good_faces is not None and len(good_faces) > 0:
        mesh.update_faces(good_faces)
    
    # Remove duplicate faces
    unique_faces = mesh.unique_faces()
    if unique_faces is not None:
        mesh.update_faces(unique_faces)
    
    # Fill holes (if method exists)
    if hasattr(mesh, 'fill_holes'):
        mesh.fill_holes()
    
    return mesh


def simplify_mesh(
    mesh: trimesh.Trimesh,
    face_count: int
) -> trimesh.Trimesh:
    """
    Simplify mesh by reducing face count.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to simplify
    face_count : int
        Target face count
    
    Returns
    -------
    simplified : trimesh.Trimesh
        Simplified mesh
    
    Raises
    ------
    ImportError
        If fast_simplification is not available
    
    Examples
    --------
    >>> mesh = network_to_trimesh(net)
    >>> print(f"Original faces: {len(mesh.faces)}")
    >>> simplified = simplify_mesh(mesh, face_count=1000)
    >>> print(f"Simplified faces: {len(simplified.faces)}")
    """
    try:
        simplified = mesh.simplify_quadric_decimation(face_count)
    except ModuleNotFoundError:
        raise ImportError(
            "fast_simplification is required for mesh simplification. "
            "Install with: pip install fast-simplification"
        )
    return simplified


