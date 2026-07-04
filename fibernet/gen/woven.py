"""
Woven and textile fiber network generators.

Generates woven structures including:
- Plain weave
- Twill weave
- Satin weave
- 3D orthogonal woven
- Triaxial braid
"""

import numpy as np
from typing import Optional, Tuple, List
from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork
from fibernet.core.material import Material


def plain_weave_2d(
    spacing: float = 2.0,
    grid_size: Tuple[int, int] = (20, 20),
    amplitude: float = 0.5,
    radius: float = 0.1,
    material_warp: Optional[Material] = None,
    material_weft: Optional[Material] = None,
) -> FiberNetwork:
    """Generate a 2D plain weave (1/1) fabric structure.
    
    Warp and weft fibers alternate over-under at each crossing.
    
    Parameters
    ----------
    spacing : float
        Distance between parallel yarns.
    grid_size : tuple
        (num_warp, num_weft) yarns.
    amplitude : float
        Out-of-plane crimp amplitude.
    """
    mat_warp = material_warp or Material(name="warp")
    mat_weft = material_weft or Material(name="weft")
    num_warp, num_weft = grid_size
    Lx = num_warp * spacing
    Ly = num_weft * spacing
    
    net = FiberNetwork(
        dimension=3,
        box_size=np.array([Lx, Ly, 4 * amplitude]),
        metadata={"generator": "plain_weave_2d", "pattern": "plain", "spacing": spacing},
    )
    
    num_pts = 200
    
    # Warp fibers (along x)
    for j in range(num_weft):
        y = j * spacing
        x_vals = np.linspace(0, Lx, num_pts)
        z_vals = np.zeros(num_pts)
        
        for k, x in enumerate(x_vals):
            cross_idx = int(x / spacing)
            if (j + cross_idx) % 2 == 0:
                z_vals[k] = amplitude * np.sin(np.pi * (x % spacing) / spacing)
            else:
                z_vals[k] = -amplitude * np.sin(np.pi * (x % spacing) / spacing)
        
        points = np.column_stack([x_vals, np.full(num_pts, y), z_vals])
        fiber = Fiber(centerline=points, radius=radius, material=mat_warp, fiber_id=net.num_fibers)
        net.add_fiber(fiber)
    
    # Weft fibers (along y)
    for i in range(num_warp):
        x = i * spacing
        y_vals = np.linspace(0, Ly, num_pts)
        z_vals = np.zeros(num_pts)
        
        for k, y in enumerate(y_vals):
            cross_idx = int(y / spacing)
            if (i + cross_idx) % 2 == 0:
                z_vals[k] = -amplitude * np.sin(np.pi * (y % spacing) / spacing)
            else:
                z_vals[k] = amplitude * np.sin(np.pi * (y % spacing) / spacing)
        
        points = np.column_stack([np.full(num_pts, x), y_vals, z_vals])
        fiber = Fiber(centerline=points, radius=radius, material=mat_weft, fiber_id=net.num_fibers)
        net.add_fiber(fiber)
    
    net.auto_crosslink(threshold=3.0 * radius)
    return net


def twill_weave_2d(
    spacing: float = 2.0,
    grid_size: Tuple[int, int] = (20, 20),
    amplitude: float = 0.5,
    radius: float = 0.1,
    twill_pattern: Tuple[int, int] = (2, 2),
    material: Optional[Material] = None,
) -> FiberNetwork:
    """Generate a 2D twill weave structure.
    
    Parameters
    ----------
    twill_pattern : tuple
        (over, under) count, e.g., (2, 1) for 2/1 twill, (2, 2) for 2/2 twill.
    """
    mat = material or Material(name="twill_fiber")
    num_warp, num_weft = grid_size
    Lx = num_warp * spacing
    Ly = num_weft * spacing
    over, under = twill_pattern
    repeat = over + under
    
    net = FiberNetwork(
        dimension=3,
        box_size=np.array([Lx, Ly, 4 * amplitude]),
        metadata={"generator": "twill_weave_2d", "pattern": f"{over}/{under}", "spacing": spacing},
    )
    
    num_pts = 200
    
    for j in range(num_weft):
        y = j * spacing
        x_vals = np.linspace(0, Lx, num_pts)
        z_vals = np.zeros(num_pts)
        
        for k, x in enumerate(x_vals):
            cross_idx = int(x / spacing)
            phase = (cross_idx + j) % repeat
            if phase < over:
                z_vals[k] = amplitude * np.sin(np.pi * (x % spacing) / spacing)
            else:
                z_vals[k] = -amplitude * np.sin(np.pi * (x % spacing) / spacing)
        
        points = np.column_stack([x_vals, np.full(num_pts, y), z_vals])
        net.add_fiber(Fiber(centerline=points, radius=radius, material=mat, fiber_id=net.num_fibers))
    
    for i in range(num_warp):
        x = i * spacing
        y_vals = np.linspace(0, Ly, num_pts)
        z_vals = np.zeros(num_pts)
        
        for k, y in enumerate(y_vals):
            cross_idx = int(y / spacing)
            phase = (i + cross_idx) % repeat
            if phase < over:
                z_vals[k] = -amplitude * np.sin(np.pi * (y % spacing) / spacing)
            else:
                z_vals[k] = amplitude * np.sin(np.pi * (y % spacing) / spacing)
        
        points = np.column_stack([np.full(num_pts, x), y_vals, z_vals])
        net.add_fiber(Fiber(centerline=points, radius=radius, material=mat, fiber_id=net.num_fibers))
    
    net.auto_crosslink(threshold=3.0 * radius)
    return net


def satin_weave_2d(
    spacing: float = 2.0,
    grid_size: Tuple[int, int] = (20, 20),
    amplitude: float = 0.5,
    radius: float = 0.1,
    satin_step: int = 3,
    material: Optional[Material] = None,
) -> FiberNetwork:
    """Generate a 2D satin weave (e.g., 5-harness satin).
    
    Parameters
    ----------
    satin_step : int
        Step offset for interlacement pattern (typically 2 or 3).
    """
    mat = material or Material(name="satin_fiber")
    num_warp, num_weft = grid_size
    Lx = num_warp * spacing
    Ly = num_weft * spacing
    repeat = max(5, satin_step + 2)
    
    net = FiberNetwork(
        dimension=3,
        box_size=np.array([Lx, Ly, 4 * amplitude]),
        metadata={"generator": "satin_weave_2d", "satin_step": satin_step},
    )
    
    num_pts = 200
    interlace = [(k * satin_step) % repeat for k in range(repeat)]
    
    for j in range(num_weft):
        y = j * spacing
        x_vals = np.linspace(0, Lx, num_pts)
        z_vals = np.zeros(num_pts)
        
        for k, x in enumerate(x_vals):
            cross_idx = int(x / spacing) % repeat
            is_over = any((j * satin_step + cross_idx) % repeat == p for p in interlace[:1])
            if is_over:
                z_vals[k] = amplitude * np.sin(np.pi * (x % spacing) / spacing)
            else:
                z_vals[k] = -amplitude * 0.3 * np.sin(np.pi * (x % spacing) / spacing)
        
        points = np.column_stack([x_vals, np.full(num_pts, y), z_vals])
        net.add_fiber(Fiber(centerline=points, radius=radius, material=mat, fiber_id=net.num_fibers))
    
    for i in range(num_warp):
        x = i * spacing
        y_vals = np.linspace(0, Ly, num_pts)
        z_vals = np.zeros(num_pts)
        
        for k, y in enumerate(y_vals):
            cross_idx = int(y / spacing) % repeat
            is_over = any((i * satin_step + cross_idx) % repeat == p for p in interlace[:1])
            if not is_over:
                z_vals[k] = amplitude * np.sin(np.pi * (y % spacing) / spacing)
            else:
                z_vals[k] = -amplitude * 0.3 * np.sin(np.pi * (y % spacing) / spacing)
        
        points = np.column_stack([np.full(num_pts, x), y_vals, z_vals])
        net.add_fiber(Fiber(centerline=points, radius=radius, material=mat, fiber_id=net.num_fibers))
    
    net.auto_crosslink(threshold=3.0 * radius)
    return net


def woven_3d_orthogonal(
    spacing: float = 3.0,
    grid_size: Tuple[int, int, int] = (5, 5, 3),
    radius: float = 0.2,
    material: Optional[Material] = None,
) -> FiberNetwork:
    """Generate a 3D orthogonal woven structure.
    
    Three mutually orthogonal yarn systems (warp, weft, z-binder).
    """
    mat = material or Material(name="3d_woven")
    nx, ny, nz = grid_size
    Lx = nx * spacing
    Ly = ny * spacing
    Lz = nz * spacing
    
    net = FiberNetwork(
        dimension=3,
        box_size=np.array([Lx, Ly, Lz]),
        metadata={"generator": "woven_3d_orthogonal"},
    )
    
    # Warp (x-direction)
    for j in range(ny):
        for k in range(nz):
            start = np.array([0.0, j * spacing + spacing / 2, k * spacing + spacing / 2])
            end = np.array([Lx, j * spacing + spacing / 2, k * spacing + spacing / 2])
            net.add_fiber(Fiber.straight(start, end, radius=radius, material=mat, fiber_id=net.num_fibers))
    
    # Weft (y-direction)
    for i in range(nx):
        for k in range(nz):
            start = np.array([i * spacing + spacing / 2, 0.0, k * spacing + spacing / 2])
            end = np.array([i * spacing + spacing / 2, Ly, k * spacing + spacing / 2])
            net.add_fiber(Fiber.straight(start, end, radius=radius, material=mat, fiber_id=net.num_fibers))
    
    # Z-binder
    for i in range(nx):
        for j in range(ny):
            start = np.array([i * spacing + spacing / 2, j * spacing + spacing / 2, 0.0])
            end = np.array([i * spacing + spacing / 2, j * spacing + spacing / 2, Lz])
            net.add_fiber(Fiber.straight(start, end, radius=radius, material=mat, fiber_id=net.num_fibers))
    
    net.auto_crosslink(threshold=3.0 * radius)
    return net
