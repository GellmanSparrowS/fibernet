"""
Fractal and Self-Similar Network Generator Module

Provides generators for fractal and self-similar fiber networks:
- Sierpinski triangle networks
- Koch curve networks
- Fractal tree networks
- Self-similar hierarchical structures

References:
- Mandelbrot, B.B., "The Fractal Geometry of Nature", Freeman, 1982
- Falconer, K., "Fractal Geometry: Mathematical Foundations and Applications", Wiley, 2014
"""

import numpy as np
from typing import Tuple, Optional, List
import warnings

from fibernet.core.network import FiberNetwork
from fibernet.core.fiber import Fiber
from fibernet.core.material import Material


def sierpinski_triangle(
    iterations: int = 3,
    size: float = 10.0,
    origin: Tuple[float, float] = (0.0, 0.0),
    radius: float = 0.1,
    material: Optional[Material] = None,
) -> FiberNetwork:
    """Generate a Sierpinski triangle fiber network.
    
    Parameters
    ----------
    iterations : int
        Number of iterations (recursion depth).
    size : float
        Initial triangle side length.
    origin : tuple
        (x, y) position of bottom-left corner.
    radius : float
        Fiber radius.
    material : Material, optional
        Fiber material.
    
    Returns
    -------
    network : FiberNetwork
        Sierpinski triangle network.
    
    Examples
    --------
    >>> from fibernet.gen.fractal import sierpinski_triangle
    >>> net = sierpinski_triangle(iterations=3, size=10.0)
    >>> print(f"Fibers: {net.num_fibers}")
    """
    net = FiberNetwork(dimension=2)
    
    # Triangle vertices
    h = size * np.sqrt(3) / 2  # Height
    p1 = np.array([origin[0], origin[1], 0.0])
    p2 = np.array([origin[0] + size, origin[1], 0.0])
    p3 = np.array([origin[0] + size/2, origin[1] + h, 0.0])
    
    fiber_id = 0
    
    # Recursive function
    def add_triangle(p1, p2, p3, depth):
        nonlocal fiber_id
        if depth == 0:
            # Add edges
            net.add_fiber(Fiber.straight(p1, p2, radius=radius, material=material, fiber_id=fiber_id))
            fiber_id += 1
            net.add_fiber(Fiber.straight(p2, p3, radius=radius, material=material, fiber_id=fiber_id))
            fiber_id += 1
            net.add_fiber(Fiber.straight(p3, p1, radius=radius, material=material, fiber_id=fiber_id))
            fiber_id += 1
        else:
            # Midpoints
            m1 = (p1 + p2) / 2
            m2 = (p2 + p3) / 2
            m3 = (p3 + p1) / 2
            
            # Recurse
            add_triangle(p1, m1, m3, depth - 1)
            add_triangle(m1, p2, m2, depth - 1)
            add_triangle(m3, m2, p3, depth - 1)
    
    add_triangle(p1, p2, p3, iterations)
    
    return net


def koch_curve(
    iterations: int = 3,
    start: Tuple[float, float] = (0.0, 0.0),
    end: Tuple[float, float] = (10.0, 0.0),
    radius: float = 0.1,
    material: Optional[Material] = None,
) -> FiberNetwork:
    """Generate a Koch curve fiber network.
    
    Parameters
    ----------
    iterations : int
        Number of iterations.
    start : tuple
        (x, y) start point.
    end : tuple
        (x, y) end point.
    radius : float
        Fiber radius.
    material : Material, optional
        Fiber material.
    
    Returns
    -------
    network : FiberNetwork
        Koch curve network.
    
    Examples
    --------
    >>> from fibernet.gen.fractal import koch_curve
    >>> net = koch_curve(iterations=3)
    >>> print(f"Fibers: {net.num_fibers}")
    """
    net = FiberNetwork(dimension=2)
    
    # Convert to numpy arrays
    p_start = np.array([start[0], start[1], 0.0])
    p_end = np.array([end[0], end[1], 0.0])
    
    # Recursive function
    def koch_points(p1, p2, depth):
        if depth == 0:
            return [p1, p2]
        
        # Divide into thirds
        v = p2 - p1
        p3 = p1 + v / 3
        p5 = p1 + 2 * v / 3
        
        # Peak point
        angle = np.pi / 3  # 60 degrees
        rot = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        p4 = p3 + rot @ (v / 3)
        
        # Recurse
        points = []
        points.extend(koch_points(p1, p3, depth - 1)[:-1])
        points.extend(koch_points(p3, p4, depth - 1)[:-1])
        points.extend(koch_points(p4, p5, depth - 1)[:-1])
        points.extend(koch_points(p5, p2, depth - 1))
        
        return points
    
    points = koch_points(p_start, p_end, iterations)
    
    # Add fibers between consecutive points
    for i in range(len(points) - 1):
        net.add_fiber(Fiber.straight(points[i], points[i+1], radius=radius, material=material, fiber_id=i))
    
    return net


def fractal_tree(
    iterations: int = 5,
    trunk_length: float = 10.0,
    branch_ratio: float = 0.7,
    branch_angle: float = np.pi / 6,
    origin: Tuple[float, float] = (0.0, 0.0),
    radius: float = 0.1,
    material: Optional[Material] = None,
) -> FiberNetwork:
    """Generate a fractal tree fiber network.
    
    Parameters
    ----------
    iterations : int
        Number of iterations (branching depth).
    trunk_length : float
        Length of initial trunk.
    branch_ratio : float
        Length ratio for child branches (0.5-0.9).
    branch_angle : float
        Angle between branches (radians).
    origin : tuple
        (x, y) position of tree base.
    radius : float
        Fiber radius (decreases with depth).
    material : Material, optional
        Fiber material.
    
    Returns
    -------
    network : FiberNetwork
        Fractal tree network.
    
    Examples
    --------
    >>> from fibernet.gen.fractal import fractal_tree
    >>> net = fractal_tree(iterations=5, trunk_length=10.0)
    >>> print(f"Fibers: {net.num_fibers}")
    """
    net = FiberNetwork(dimension=2)
    
    # Start point
    p_start = np.array([origin[0], origin[1], 0.0])
    
    fiber_id = 0
    
    # Recursive function
    def add_branch(start, angle, length, depth, current_radius):
        nonlocal fiber_id
        if depth == 0:
            return
        
        # End point
        end = start + length * np.array([np.cos(angle), np.sin(angle), 0.0])
        
        # Add fiber
        net.add_fiber(Fiber.straight(start, end, radius=current_radius, material=material, fiber_id=fiber_id))
        fiber_id += 1
        
        # Recurse for left and right branches
        child_length = length * branch_ratio
        child_radius = current_radius * 0.8
        
        add_branch(end, angle + branch_angle, child_length, depth - 1, child_radius)
        add_branch(end, angle - branch_angle, child_length, depth - 1, child_radius)
    
    # Start with trunk pointing up
    add_branch(p_start, np.pi / 2, trunk_length, iterations, radius)
    
    return net


def hilbert_curve(
    order: int = 3,
    size: float = 10.0,
    origin: Tuple[float, float] = (0.0, 0.0),
    radius: float = 0.1,
    material: Optional[Material] = None,
) -> FiberNetwork:
    """Generate a Hilbert curve fiber network.
    
    Parameters
    ----------
    order : int
        Order of the Hilbert curve (2^order segments per side).
    size : float
        Total size of the curve.
    origin : tuple
        (x, y) position of bottom-left corner.
    radius : float
        Fiber radius.
    material : Material, optional
        Fiber material.
    
    Returns
    -------
    network : FiberNetwork
        Hilbert curve network.
    
    Examples
    --------
    >>> from fibernet.gen.fractal import hilbert_curve
    >>> net = hilbert_curve(order=3, size=10.0)
    >>> print(f"Fibers: {net.num_fibers}")
    """
    net = FiberNetwork(dimension=2)
    
    # Generate Hilbert curve points
    n = 2 ** order
    points = []
    
    for i in range(n * n):
        # Convert index to (x, y) using Hilbert curve algorithm
        x, y = 0, 0
        t = i
        
        for s in range(order):
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            
            if ry == 0:
                if rx == 1:
                    x = (1 << s) - 1 - x
                    y = (1 << s) - 1 - y
                x, y = y, x
            
            x += rx << s
            y += ry << s
            t //= 4
        
        # Scale to size
        px = origin[0] + size * x / (n - 1)
        py = origin[1] + size * y / (n - 1)
        points.append(np.array([px, py, 0.0]))
    
    # Add fibers between consecutive points
    for i in range(len(points) - 1):
        net.add_fiber(Fiber.straight(points[i], points[i+1], radius=radius, material=material, fiber_id=i))
    
    return net


