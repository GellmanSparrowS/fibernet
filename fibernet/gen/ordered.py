"""
Ordered fiber network generators.

Generates ordered/periodic fiber networks including:
- Square and triangular lattices (2D)
- Cubic, BCC, FCC, and octet lattices (3D)
- Kagome lattice
- Honeycomb lattice
- Custom periodic structures
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from itertools import product

from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork, Crosslink
from fibernet.core.material import Material


def square_lattice_2d(
    spacing: float = 5.0,
    grid_size: Tuple[int, int] = (10, 10),
    radius: float = 0.1,
    material: Optional[Material] = None,
    periodic: bool = False,
) -> FiberNetwork:
    """Generate a 2D square lattice fiber network.
    
    Creates horizontal and vertical fibers forming a square grid.
    """
    mat = material or Material(name="lattice_fiber")
    nx, ny = grid_size
    Lx = nx * spacing
    Ly = ny * spacing
    
    net = FiberNetwork(
        dimension=2,
        box_size=np.array([Lx, Ly, 0.0]),
        periodic=periodic,
        metadata={"generator": "square_lattice_2d", "spacing": spacing, "grid_size": grid_size},
    )
    
    fid = 0
    # Horizontal fibers
    for j in range(ny + 1):
        y = j * spacing
        start = np.array([0.0, y, 0.0])
        end = np.array([Lx, y, 0.0])
        net.add_fiber(Fiber.straight(start, end, radius=radius, material=mat, fiber_id=fid, segments=nx * 2))
        fid += 1
    
    # Vertical fibers
    for i in range(nx + 1):
        x = i * spacing
        start = np.array([x, 0.0, 0.0])
        end = np.array([x, Ly, 0.0])
        net.add_fiber(Fiber.straight(start, end, radius=radius, material=mat, fiber_id=fid, segments=ny * 2))
        fid += 1
    
    # Add crosslinks at intersections
    for i in range(nx + 1):
        for j in range(ny + 1):
            h_idx = j
            v_idx = nx + 1 + i
            pos = np.array([i * spacing, j * spacing, 0.0])
            net.add_crosslink(Crosslink(
                fiber_i=h_idx, fiber_j=v_idx,
                param_i=i / nx if nx > 0 else 0,
                param_j=j / ny if ny > 0 else 0,
                position=pos, crosslink_type="welded",
            ))
    
    return net


def triangular_lattice_2d(
    spacing: float = 5.0,
    grid_size: Tuple[int, int] = (10, 10),
    radius: float = 0.1,
    material: Optional[Material] = None,
    periodic: bool = False,
) -> FiberNetwork:
    """Generate a 2D triangular lattice fiber network.
    
    Three families of fibers at 0°, 60°, and 120°.
    """
    mat = material or Material(name="lattice_fiber")
    nx, ny = grid_size
    Lx = nx * spacing
    Ly = ny * spacing * np.sqrt(3) / 2
    
    net = FiberNetwork(
        dimension=2,
        box_size=np.array([Lx, Ly, 0.0]),
        periodic=periodic,
        metadata={"generator": "triangular_lattice_2d", "spacing": spacing},
    )
    
    fid = 0
    directions = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.5, np.sqrt(3) / 2, 0.0]),
        np.array([-0.5, np.sqrt(3) / 2, 0.0]),
    ]
    
    for d_idx, d in enumerate(directions):
        if d_idx == 0:
            for j in range(ny + 1):
                y = j * spacing * np.sqrt(3) / 2
                start = np.array([0.0, y, 0.0])
                end = start + Lx * d
                net.add_fiber(Fiber.straight(start, end, radius=radius, material=mat, fiber_id=fid))
                fid += 1
        elif d_idx == 1:
            num_lines = int(Lx / spacing) + ny + 2
            for k in range(-ny, num_lines):
                start = np.array([k * spacing, 0.0, 0.0])
                diag_len = np.sqrt(Lx**2 + Ly**2)
                end = start + diag_len * d
                net.add_fiber(Fiber.straight(start, end, radius=radius, material=mat, fiber_id=fid))
                fid += 1
        else:
            num_lines = int(Lx / spacing) + ny + 2
            for k in range(num_lines + ny):
                start = np.array([k * spacing, 0.0, 0.0])
                diag_len = np.sqrt(Lx**2 + Ly**2)
                end = start + diag_len * d
                net.add_fiber(Fiber.straight(start, end, radius=radius, material=mat, fiber_id=fid))
                fid += 1
    
    net.auto_crosslink(threshold=2.5 * radius)
    return net


def honeycomb_lattice_2d(
    cell_size: float = 5.0,
    grid_size: Tuple[int, int] = (10, 10),
    radius: float = 0.1,
    material: Optional[Material] = None,
) -> FiberNetwork:
    """Generate a 2D honeycomb (hexagonal) lattice fiber network."""
    mat = material or Material(name="lattice_fiber")
    nx, ny = grid_size
    a = cell_size
    dx = a * 1.5
    dy = a * np.sqrt(3)
    
    Lx = nx * dx + a
    Ly = ny * dy + dy
    
    net = FiberNetwork(
        dimension=2,
        box_size=np.array([Lx, Ly, 0.0]),
        metadata={"generator": "honeycomb_lattice_2d", "cell_size": cell_size},
    )
    
    fid = 0
    for i in range(nx):
        for j in range(ny):
            cx = i * dx
            cy = j * dy
            if i % 2 == 1:
                cy += dy / 2
            
            vertices = []
            for k in range(6):
                angle = np.pi / 3 * k + np.pi / 6
                vx = cx + a * np.cos(angle)
                vy = cy + a * np.sin(angle)
                vertices.append(np.array([vx, vy, 0.0]))
            
            for k in range(6):
                v1 = vertices[k]
                v2 = vertices[(k + 1) % 6]
                net.add_fiber(Fiber.straight(v1, v2, radius=radius, material=mat, fiber_id=fid, segments=4))
                fid += 1
    
    net.auto_crosslink(threshold=2.5 * radius)
    return net


def cubic_lattice_3d(
    spacing: float = 5.0,
    grid_size: Tuple[int, int, int] = (5, 5, 5),
    radius: float = 0.1,
    material: Optional[Material] = None,
    periodic: bool = False,
) -> FiberNetwork:
    """Generate a 3D simple cubic lattice fiber network."""
    mat = material or Material(name="lattice_fiber")
    nx, ny, nz = grid_size
    Lx = nx * spacing
    Ly = ny * spacing
    Lz = nz * spacing
    
    net = FiberNetwork(
        dimension=3,
        box_size=np.array([Lx, Ly, Lz]),
        periodic=periodic,
        metadata={"generator": "cubic_lattice_3d", "spacing": spacing, "grid_size": grid_size},
    )
    
    fid = 0
    # X-direction fibers
    for j in range(ny + 1):
        for k in range(nz + 1):
            start = np.array([0.0, j * spacing, k * spacing])
            end = np.array([Lx, j * spacing, k * spacing])
            net.add_fiber(Fiber.straight(start, end, radius=radius, material=mat, fiber_id=fid, segments=nx * 2))
            fid += 1
    
    # Y-direction fibers
    for i in range(nx + 1):
        for k in range(nz + 1):
            start = np.array([i * spacing, 0.0, k * spacing])
            end = np.array([i * spacing, Ly, k * spacing])
            net.add_fiber(Fiber.straight(start, end, radius=radius, material=mat, fiber_id=fid, segments=ny * 2))
            fid += 1
    
    # Z-direction fibers
    for i in range(nx + 1):
        for j in range(ny + 1):
            start = np.array([i * spacing, j * spacing, 0.0])
            end = np.array([i * spacing, j * spacing, Lz])
            net.add_fiber(Fiber.straight(start, end, radius=radius, material=mat, fiber_id=fid, segments=nz * 2))
            fid += 1
    
    net.auto_crosslink(threshold=2.5 * radius)
    return net


def octet_truss_3d(
    spacing: float = 5.0,
    grid_size: Tuple[int, int, int] = (3, 3, 3),
    radius: float = 0.1,
    material: Optional[Material] = None,
) -> FiberNetwork:
    """Generate a 3D octet truss (tetrahedral-octahedral honeycomb).
    
    This is a well-known lightweight structural configuration.
    """
    mat = material or Material(name="lattice_fiber")
    nx, ny, nz = grid_size
    a = spacing
    
    net = FiberNetwork(
        dimension=3,
        box_size=np.array([nx * a, ny * a, nz * a]),
        metadata={"generator": "octet_truss_3d", "spacing": spacing},
    )
    
    nodes = {}
    nid = 0
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                pos = np.array([i * a, j * a, k * a])
                nodes[(i, j, k)] = pos
    
    edge_set = set()
    
    def add_edge(n1, n2):
        key = tuple(sorted([n1, n2]))
        if key not in edge_set:
            edge_set.add(key)
            if n1 in nodes and n2 in nodes:
                p1, p2 = nodes[n1], nodes[n2]
                net.add_fiber(Fiber.straight(p1, p2, radius=radius, material=mat, fiber_id=net.num_fibers))
    
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                n = (i, j, k)
                if i < nx:
                    add_edge(n, (i + 1, j, k))
                if j < ny:
                    add_edge(n, (i, j + 1, k))
                if k < nz:
                    add_edge(n, (i, j, k + 1))
                
                if i < nx and j < ny:
                    add_edge(n, (i + 1, j + 1, k))
                if i < nx and k < nz:
                    add_edge(n, (i + 1, j, k + 1))
                if j < ny and k < nz:
                    add_edge(n, (i, j + 1, k + 1))
                if i < nx and j < ny and k < nz:
                    add_edge(n, (i + 1, j + 1, k + 1))
    
    net.auto_crosslink(threshold=2.5 * radius)
    return net


def kagome_lattice_2d(
    spacing: float = 5.0,
    grid_size: Tuple[int, int] = (8, 8),
    radius: float = 0.1,
    material: Optional[Material] = None,
) -> FiberNetwork:
    """Generate a 2D Kagome lattice (trihexagonal tiling)."""
    mat = material or Material(name="lattice_fiber")
    nx, ny = grid_size
    a = spacing
    
    net = FiberNetwork(
        dimension=2,
        box_size=np.array([nx * a * 3, ny * a * np.sqrt(3), 0.0]),
        metadata={"generator": "kagome_lattice_2d", "spacing": spacing},
    )
    
    fid = 0
    for i in range(nx):
        for j in range(ny):
            cx = i * 3 * a
            cy = j * np.sqrt(3) * a
            if i % 2 == 1:
                cy += np.sqrt(3) * a / 2
            
            tri_offsets = [
                np.array([0, 0, 0]),
                np.array([a, 0, 0]),
                np.array([0.5 * a, a * np.sqrt(3) / 2, 0]),
            ]
            
            for k in range(3):
                p1 = cx + tri_offsets[k]
                p2 = cx + tri_offsets[(k + 1) % 3]
                net.add_fiber(Fiber.straight(p1, p2, radius=radius, material=mat, fiber_id=fid, segments=4))
                fid += 1
            
            cx2 = cx + np.array([1.5 * a, a * np.sqrt(3) / 2, 0])
            for k in range(3):
                angle = np.pi / 3 * k + np.pi
                p1 = cx2 + a * np.array([np.cos(angle), np.sin(angle), 0])
                p2 = cx2 + a * np.array([np.cos(angle + np.pi / 3), np.sin(angle + np.pi / 3), 0])
                net.add_fiber(Fiber.straight(p1, p2, radius=radius, material=mat, fiber_id=fid, segments=4))
                fid += 1
    
    net.auto_crosslink(threshold=2.5 * radius)
    return net
