"""
Ordered fiber network generators.

All generators use the FiberGraph abstraction to ensure proper
connectivity and crosslink creation.

Available generators:
- square_lattice_2d: Square grid (coordination 4)
- triangular_lattice_2d: Triangular grid (coordination 6)
- honeycomb_lattice_2d: Hexagonal honeycomb (coordination 3)
- kagome_lattice_2d: Kagome/trihexagonal (coordination 4)
- cubic_lattice_3d: Simple cubic (coordination 6)
- octet_truss_3d: Octet truss (coordination 12)
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from fibernet.core.network import FiberNetwork
from fibernet.core.material import Material
from fibernet.gen._graph_builder import FiberGraph


# ============================================================================
# 2D Lattices
# ============================================================================

def square_lattice_2d(
    spacing: float = 5.0,
    grid_size: Tuple[int, int] = (10, 10),
    radius: float = 0.1,
    material: Optional[Material] = None,
    **kwargs,
) -> FiberNetwork:
    """Generate a 2D square lattice.
    
    Parameters
    ----------
    spacing : float
        Distance between adjacent nodes.
    grid_size : tuple
        (nx, ny) number of cells in each direction.
    radius : float
        Fiber radius.
    material : Material, optional
        Fiber material.
    """
    mat = material or Material(name="square_lattice")
    nx, ny = grid_size
    
    g = FiberGraph(dimension=2, tolerance=spacing * 0.01)
    
    # Create grid of edges
    for i in range(nx + 1):
        for j in range(ny + 1):
            x, y = i * spacing, j * spacing
            # Horizontal edge to the right
            if i < nx:
                g.add_edge_by_pos(
                    np.array([x, y, 0.0]),
                    np.array([x + spacing, y, 0.0]),
                    radius=radius, material=mat,
                )
            # Vertical edge upward
            if j < ny:
                g.add_edge_by_pos(
                    np.array([x, y, 0.0]),
                    np.array([x, y + spacing, 0.0]),
                    radius=radius, material=mat,
                )
    
    return g.to_network(
        material=mat,
        box_size=np.array([nx * spacing, ny * spacing, 0.0]),
        metadata={"generator": "square_lattice_2d", "spacing": spacing, "grid_size": grid_size},
    )


def triangular_lattice_2d(
    spacing: float = 5.0,
    grid_size: Tuple[int, int] = (10, 10),
    radius: float = 0.1,
    material: Optional[Material] = None,
    **kwargs,
) -> FiberNetwork:
    """Generate a 2D triangular lattice (coordination 6).
    
    Properly tiles with alternating up/down triangles sharing all edges.
    """
    mat = material or Material(name="triangular_lattice")
    nx, ny = grid_size
    h = spacing * np.sqrt(3) / 2
    
    g = FiberGraph(dimension=2, tolerance=spacing * 0.01)
    
    # Collect all unique edges from the triangular tessellation
    edge_set = set()
    
    for ci in range(nx):
        for cj in range(ny):
            # Cell origin (offset for odd rows)
            ox = ci * spacing + (cj % 2) * spacing / 2
            oy = cj * h
            
            # Up-triangle vertices
            p1 = (ox, oy)
            p2 = (ox + spacing, oy)
            p3 = (ox + spacing / 2, oy + h)
            
            # Down-triangle (shares edges with adjacent up-triangles)
            p4 = (ox + spacing / 2, oy + h)
            p5 = (ox + spacing * 3 / 2, oy + h)
            p6 = (ox + spacing, oy)
            
            # Add edges as sorted position tuples
            for e in [(p1, p2), (p2, p3), (p3, p1),
                      (p4, p5), (p5, p6), (p6, p4)]:
                key = tuple(sorted([
                    tuple(np.round(e[0], 8)),
                    tuple(np.round(e[1], 8)),
                ]))
                edge_set.add(key)
    
    # Add all unique edges to graph
    for (p1, p2) in edge_set:
        g.add_edge_by_pos(
            np.array([p1[0], p1[1], 0.0]),
            np.array([p2[0], p2[1], 0.0]),
            radius=radius, material=mat,
        )
    
    bb_min, bb_max = g.bounding_box()
    box = bb_max - bb_min
    box[2] = 0.0
    
    return g.to_network(
        material=mat,
        box_size=box,
        metadata={"generator": "triangular_lattice_2d", "spacing": spacing, "grid_size": grid_size},
    )


def honeycomb_lattice_2d(
    cell_size: float = 10.0,
    grid_size: Tuple[int, int] = (10, 10),
    radius: float = 0.1,
    material: Optional[Material] = None,
    **kwargs,
) -> FiberNetwork:
    """Generate a 2D honeycomb (hexagonal) lattice.
    
    Parameters
    ----------
    cell_size : float
        Side length of hexagonal cells.
    grid_size : tuple
        (nx, ny) number of cells in each direction.
    """
    mat = material or Material(name="honeycomb")
    nx, ny = grid_size
    a = cell_size  # hexagon side length
    
    g = FiberGraph(dimension=2, tolerance=a * 0.01)
    
    # Honeycomb unit cell dimensions
    dx = a * np.sqrt(3)  # horizontal spacing between cell centers
    dy = a * 1.5          # vertical spacing between cell centers
    
    for ci in range(nx):
        for cj in range(ny):
            cx = ci * dx + (cj % 2) * dx / 2
            cy = cj * dy
            
            # 6 vertices of the hexagon
            vertices = []
            for k in range(6):
                angle = np.pi / 6 + k * np.pi / 3
                px = cx + a * np.cos(angle)
                py = cy + a * np.sin(angle)
                vertices.append(np.array([px, py, 0.0]))
            
            # Add 6 edges of the hexagon
            for k in range(6):
                g.add_edge_by_pos(
                    vertices[k],
                    vertices[(k + 1) % 6],
                    radius=radius, material=mat,
                )
    
    bb_min, bb_max = g.bounding_box()
    box = bb_max - bb_min
    box[2] = 0.0
    
    return g.to_network(
        material=mat,
        box_size=box,
        metadata={"generator": "honeycomb_lattice_2d", "cell_size": cell_size, "grid_size": grid_size},
    )


def kagome_lattice_2d(
    spacing: float = 5.0,
    grid_size: Tuple[int, int] = (8, 8),
    radius: float = 0.1,
    material: Optional[Material] = None,
    **kwargs,
) -> FiberNetwork:
    """Generate a 2D Kagome lattice (trihexagonal tiling, coordination 4).
    
    The Kagome lattice consists of corner-sharing triangles arranged
    on a triangular lattice. Properly connected via shared vertices.
    """
    mat = material or Material(name="kagome")
    nx, ny = grid_size
    a = spacing
    
    g = FiberGraph(dimension=2, tolerance=a * 0.01)
    
    # Strategy: place triangular lattice vertices, then for each vertex
    # find the midpoints of its incident edges and connect them.
    
    # Step 1: Build triangular lattice vertices
    tri_nodes: Dict[Tuple[int, int], np.ndarray] = {}
    for i in range(nx + 2):
        for j in range(ny + 2):
            x = i * a + (j % 2) * a / 2
            y = j * a * np.sqrt(3) / 2
            tri_nodes[(i, j)] = np.array([x, y, 0.0])
    
    # Step 2: For each triangular lattice vertex, collect midpoints
    # of all its incident edges and connect them to form the Kagome pattern
    midpoint_cache: Dict[Tuple, int] = {}  # edge_key -> node_id
    
    def get_midpoint(key_a, key_b):
        """Get or create a midpoint node for a triangular lattice edge."""
        edge_key = (min(key_a, key_b), max(key_a, key_b))
        if edge_key not in midpoint_cache:
            p1 = tri_nodes[key_a]
            p2 = tri_nodes[key_b]
            mid = (p1 + p2) / 2
            nid = g.add_node(mid)
            midpoint_cache[edge_key] = nid
        return midpoint_cache[edge_key]
    
    # Step 3: For each triangular lattice vertex, connect midpoints
    # of its incident edges to form the Kagome pattern
    neighbor_offsets_even = [
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, -1)
    ]
    neighbor_offsets_odd = [
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1)
    ]
    
    for (i, j) in tri_nodes:
        offsets = neighbor_offsets_even if j % 2 == 0 else neighbor_offsets_odd
        
        # Collect midpoints of all edges incident to this vertex
        vertex_mids = []
        for di, dj in offsets:
            nk = (i + di, j + dj)
            if nk in tri_nodes:
                mid_nid = get_midpoint((i, j), nk)
                vertex_mids.append(mid_nid)
        
        # Connect consecutive midpoints (forms the Kagome pattern)
        for k in range(len(vertex_mids)):
            g.add_edge(vertex_mids[k], vertex_mids[(k + 1) % len(vertex_mids)],
                      radius=radius, material=mat)
    
    bb_min, bb_max = g.bounding_box()
    box = bb_max - bb_min
    box[2] = 0.0
    
    return g.to_network(
        material=mat,
        box_size=box,
        metadata={"generator": "kagome_lattice_2d", "spacing": spacing, "grid_size": grid_size},
    )


# ============================================================================
# 3D Lattices
# ============================================================================

def cubic_lattice_3d(
    spacing: float = 5.0,
    grid_size: Tuple[int, int, int] = (3, 3, 3),
    radius: float = 0.15,
    material: Optional[Material] = None,
    **kwargs,
) -> FiberNetwork:
    """Generate a 3D simple cubic lattice."""
    mat = material or Material(name="cubic")
    nx, ny, nz = grid_size
    a = spacing
    
    g = FiberGraph(dimension=3, tolerance=a * 0.01)
    
    node_map = {}
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                pos = np.array([i * a, j * a, k * a])
                nid = g.add_node(pos)
                node_map[(i, j, k)] = nid
    
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                if i < nx:
                    g.add_edge(node_map[(i, j, k)], node_map[(i+1, j, k)],
                              radius=radius, material=mat)
                if j < ny:
                    g.add_edge(node_map[(i, j, k)], node_map[(i, j+1, k)],
                              radius=radius, material=mat)
                if k < nz:
                    g.add_edge(node_map[(i, j, k)], node_map[(i, j, k+1)],
                              radius=radius, material=mat)
    
    return g.to_network(
        material=mat,
        box_size=np.array([nx * a, ny * a, nz * a]),
        metadata={"generator": "cubic_lattice_3d", "spacing": spacing, "grid_size": grid_size},
    )


def octet_truss_3d(
    spacing: float = 5.0,
    grid_size: Tuple[int, int, int] = (3, 3, 3),
    radius: float = 0.15,
    material: Optional[Material] = None,
    **kwargs,
) -> FiberNetwork:
    """Generate a 3D octet truss lattice (12 struts per unit cell).
    
    The octet truss is one of the stiffest lattice structures
    for a given density.
    """
    mat = material or Material(name="octet")
    nx, ny, nz = grid_size
    a = spacing
    
    g = FiberGraph(dimension=3, tolerance=a * 0.01)
    
    node_map = {}
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                pos = np.array([i * a, j * a, k * a])
                nid = g.add_node(pos)
                node_map[(i, j, k)] = nid
    
    # All 12 edge directions of the octet truss
    edge_dirs = [
        (1, 0, 0), (0, 1, 0), (0, 0, 1),      # cube edges (3)
        (1, 1, 0), (1, 0, 1), (0, 1, 1),       # face diagonals + (3)
        (1, -1, 0), (1, 0, -1), (0, 1, -1),    # face diagonals - (3)
        (1, 1, 1), (1, 1, -1), (1, -1, 1),      # body diagonals (3)
    ]
    
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                if (i, j, k) not in node_map:
                    continue
                for di, dj, dk in edge_dirs:
                    ni, nj, nk = i + di, j + dj, k + dk
                    if (ni, nj, nk) in node_map:
                        g.add_edge(node_map[(i, j, k)], node_map[(ni, nj, nk)],
                                  radius=radius, material=mat)
    
    return g.to_network(
        material=mat,
        box_size=np.array([nx * a, ny * a, nz * a]),
        metadata={"generator": "octet_truss_3d", "spacing": spacing, "grid_size": grid_size},
    )
