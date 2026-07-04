"""
Mesh Export Module

Export fiber networks as mesh files for external FEM solvers:
- STL (stereolithography)
- OBJ (Wavefront OBJ)
- PLY (Stanford PLY)
- GMSH (already in io module)

References:
- STL format: https://www.fabbers.com/tech/STL_Format
- OBJ format: https://en.wikipedia.org/wiki/Wavefront_.obj_file
"""

import numpy as np
from typing import Optional, Tuple
import warnings

from fibernet.core.network import FiberNetwork


def _generate_cylinder_mesh(
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    n_sides: int = 8,
    n_caps: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate mesh for a cylindrical fiber segment.
    
    Parameters
    ----------
    start : np.ndarray
        Start point (3D).
    end : np.ndarray
        End point (3D).
    radius : float
        Cylinder radius.
    n_sides : int
        Number of sides for polygon approximation.
    n_caps : bool
        Whether to include end caps.
    
    Returns
    -------
    vertices : np.ndarray
        Nx3 array of vertex positions.
    faces : np.ndarray
        Mx3 array of face indices.
    """
    # Direction vector
    direction = end - start
    length = np.linalg.norm(direction)
    
    if length < 1e-10:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    
    # Create perpendicular vectors
    direction = direction / length
    
    # Find perpendicular vector
    if abs(direction[0]) < 0.9:
        perp = np.cross(direction, np.array([1, 0, 0]))
    else:
        perp = np.cross(direction, np.array([0, 1, 0]))
    perp = perp / np.linalg.norm(perp)
    
    # Second perpendicular
    perp2 = np.cross(direction, perp)
    
    # Generate circle points
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    circle = np.zeros((n_sides, 3))
    for i, angle in enumerate(angles):
        circle[i] = radius * (np.cos(angle) * perp + np.sin(angle) * perp2)
    
    # Generate vertices
    vertices = []
    
    # Bottom circle
    for i in range(n_sides):
        vertices.append(start + circle[i])
    
    # Top circle
    for i in range(n_sides):
        vertices.append(end + circle[i])
    
    if n_caps:
        # Center points for caps
        vertices.append(start)  # Bottom center
        vertices.append(end)    # Top center
    
    vertices = np.array(vertices)
    
    # Generate faces
    faces = []
    
    # Side faces
    for i in range(n_sides):
        i_next = (i + 1) % n_sides
        # Two triangles per side quad
        faces.append([i, i_next, i + n_sides])
        faces.append([i + n_sides, i_next, i_next + n_sides])
    
    if n_caps:
        bottom_center = 2 * n_sides
        top_center = 2 * n_sides + 1
        
        # Bottom cap
        for i in range(n_sides):
            i_next = (i + 1) % n_sides
            faces.append([bottom_center, i_next, i])
        
        # Top cap
        for i in range(n_sides):
            i_next = (i + 1) % n_sides
            faces.append([top_center, i + n_sides, i_next + n_sides])
    
    faces = np.array(faces, dtype=int)
    
    return vertices, faces


def export_stl(
    network: FiberNetwork,
    filename: str,
    n_sides: int = 8,
    binary: bool = False
) -> None:
    """
    Export network as STL file.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to export.
    filename : str
        Output filename.
    n_sides : int
        Number of sides for cylinder approximation.
    binary : bool
        If True, export binary STL. Otherwise ASCII.
    
    Examples
    --------
    >>> from fibernet.io.mesh_export import export_stl
    >>> net = gen.random_straight_2d(num_fibers=50)
    >>> export_stl(net, 'network.stl')
    """
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    
    for fiber in network.fibers:
        start = fiber.centerline[0]
        end = fiber.centerline[-1]
        
        vertices, faces = _generate_cylinder_mesh(
            start, end, fiber.radius, n_sides=n_sides
        )
        
        if len(vertices) > 0:
            all_vertices.append(vertices)
            all_faces.append(faces + vertex_offset)
            vertex_offset += len(vertices)
    
    if not all_vertices:
        warnings.warn("No fibers to export")
        return
    
    vertices = np.vstack(all_vertices)
    faces = np.vstack(all_faces)
    
    if binary:
        _write_binary_stl(filename, vertices, faces)
    else:
        _write_ascii_stl(filename, vertices, faces)


def _write_ascii_stl(filename: str, vertices: np.ndarray, faces: np.ndarray):
    """Write ASCII STL file."""
    with open(filename, 'w') as f:
        f.write("solid fibernet\n")
        
        for face in faces:
            # Compute face normal
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]
            
            normal = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            
            f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n")
            f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
            f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        
        f.write("endsolid fibernet\n")


def _write_binary_stl(filename: str, vertices: np.ndarray, faces: np.ndarray):
    """Write binary STL file."""
    import struct
    
    with open(filename, 'wb') as f:
        # Header (80 bytes)
        header = b'FiberNet STL Export' + b'\0' * 61
        f.write(header)
        
        # Number of triangles
        f.write(struct.pack('<I', len(faces)))
        
        for face in faces:
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]
            
            normal = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            
            # Normal
            f.write(struct.pack('<3f', *normal.astype(np.float32)))
            # Vertices
            f.write(struct.pack('<3f', *v0.astype(np.float32)))
            f.write(struct.pack('<3f', *v1.astype(np.float32)))
            f.write(struct.pack('<3f', *v2.astype(np.float32)))
            # Attribute byte count
            f.write(struct.pack('<H', 0))


def export_obj(
    network: FiberNetwork,
    filename: str,
    n_sides: int = 8
) -> None:
    """
    Export network as Wavefront OBJ file.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to export.
    filename : str
        Output filename.
    n_sides : int
        Number of sides for cylinder approximation.
    
    Examples
    --------
    >>> from fibernet.io.mesh_export import export_obj
    >>> net = gen.random_straight_2d(num_fibers=50)
    >>> export_obj(net, 'network.obj')
    """
    all_vertices = []
    all_faces = []
    vertex_offset = 1  # OBJ uses 1-based indexing
    
    for fiber in network.fibers:
        start = fiber.centerline[0]
        end = fiber.centerline[-1]
        
        vertices, faces = _generate_cylinder_mesh(
            start, end, fiber.radius, n_sides=n_sides
        )
        
        if len(vertices) > 0:
            all_vertices.append(vertices)
            all_faces.append(faces + vertex_offset)
            vertex_offset += len(vertices)
    
    if not all_vertices:
        warnings.warn("No fibers to export")
        return
    
    vertices = np.vstack(all_vertices)
    faces = np.vstack(all_faces)
    
    with open(filename, 'w') as f:
        f.write("# FiberNet OBJ Export\n")
        f.write(f"# Vertices: {len(vertices)}, Faces: {len(faces)}\n\n")
        
        # Vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        f.write("\n")
        
        # Faces
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def export_ply(
    network: FiberNetwork,
    filename: str,
    n_sides: int = 8
) -> None:
    """
    Export network as Stanford PLY file.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to export.
    filename : str
        Output filename.
    n_sides : int
        Number of sides for cylinder approximation.
    
    Examples
    --------
    >>> from fibernet.io.mesh_export import export_ply
    >>> net = gen.random_straight_2d(num_fibers=50)
    >>> export_ply(net, 'network.ply')
    """
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    
    for fiber in network.fibers:
        start = fiber.centerline[0]
        end = fiber.centerline[-1]
        
        vertices, faces = _generate_cylinder_mesh(
            start, end, fiber.radius, n_sides=n_sides
        )
        
        if len(vertices) > 0:
            all_vertices.append(vertices)
            all_faces.append(faces + vertex_offset)
            vertex_offset += len(vertices)
    
    if not all_vertices:
        warnings.warn("No fibers to export")
        return
    
    vertices = np.vstack(all_vertices)
    faces = np.vstack(all_faces)
    
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


