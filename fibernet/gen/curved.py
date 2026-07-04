"""
Curved Fiber Generators

Generate fibers with various curved geometries:
- Sinusoidal (crimped/wavy)
- Helical (coiled/spring)
- Arc (circular)
- Bezier (smooth curves)
- Random curves

These are important for:
- Biological fibers (collagen crimp, actin helices)
- Natural fibers (wool crimp, cotton twist)
- Engineered materials (spring composites, coil actuators)

References:
- Silver et al., "Collagen fibril structure", J Struct Biol, 1995
- Hearle, "Fiber structure and mechanics", Textile Res J, 1978
"""

import numpy as np
from math import comb
from typing import Tuple, Optional, List, Union
from fibernet.core.network import FiberNetwork
from fibernet.core.fiber import Fiber
from fibernet.core.material import Material


def sinusoidal_fiber_2d(
    length: float = 50.0,
    amplitude: float = 2.0,
    wavelength: float = 10.0,
    radius: float = 0.5,
    num_segments: int = 50,
    orientation: float = 0.0,
    start: Tuple[float, float] = (0.0, 0.0),
    material: Optional[Material] = None,
    seed: Optional[int] = None,
) -> Fiber:
    """
    Generate a 2D sinusoidal (crimped) fiber.
    
    Parameters
    ----------
    length : float
        Total length along main axis
    amplitude : float
        Wave amplitude
    wavelength : float
        Wave wavelength
    radius : float
        Fiber radius
    num_segments : int
        Number of centerline points
    orientation : float
        Orientation angle (radians)
    start : tuple
        Start position (x, y)
    material : Material, optional
        Fiber material
    seed : int, optional
        Random seed (for phase)
    
    Returns
    -------
    Fiber
        Sinusoidal fiber
    
    Examples
    --------
    >>> fiber = sinusoidal_fiber_2d(length=50, amplitude=2, wavelength=10)
    """
    if seed is not None:
        np.random.seed(seed)
    
    phase = np.random.uniform(0, 2 * np.pi) if seed is not None else 0.0
    
    # Generate centerline
    t = np.linspace(0, length, num_segments)
    
    # Direction and perpendicular
    direction = np.array([np.cos(orientation), np.sin(orientation), 0.0])
    perp = np.array([-np.sin(orientation), np.cos(orientation), 0.0])
    
    centerline = np.zeros((num_segments, 3))
    for i, ti in enumerate(t):
        offset = amplitude * np.sin(2 * np.pi * ti / wavelength + phase)
        centerline[i] = (
            np.array([start[0], start[1], 0.0]) +
            ti * direction +
            offset * perp
        )
    
    # Create fiber
    fiber = Fiber(
        centerline=centerline,
        radius=radius,
        material=material or Material(),
        segments=num_segments
    )
    
    return fiber


def helical_fiber_3d(
    length: float = 50.0,
    radius_helix: float = 3.0,
    pitch: float = 10.0,
    fiber_radius: float = 0.5,
    num_turns: float = 5.0,
    num_segments: int = 100,
    center: Tuple[float, float, float] = (25.0, 25.0, 25.0),
    material: Optional[Material] = None,
) -> Fiber:
    """
    Generate a 3D helical (spring/coil) fiber.
    
    Parameters
    ----------
    length : float
        Total length along helix axis
    radius_helix : float
        Helix radius
    pitch : float
        Distance per turn
    fiber_radius : float
        Fiber cross-section radius
    num_turns : float
        Number of turns
    num_segments : int
        Number of centerline points
    center : tuple
        Center position (x, y, z)
    material : Material, optional
        Fiber material
    
    Returns
    -------
    Fiber
        Helical fiber
    
    Examples
    --------
    >>> fiber = helical_fiber_3d(length=50, radius_helix=3, pitch=10)
    """
    # Generate helix centerline
    t = np.linspace(0, 2 * np.pi * num_turns, num_segments)
    
    centerline = np.zeros((num_segments, 3))
    center = np.array(center)
    
    for i, ti in enumerate(t):
        x = center[0] + radius_helix * np.cos(ti)
        y = center[1] + radius_helix * np.sin(ti)
        z = center[2] + (ti / (2 * np.pi)) * pitch - length / 2
        
        centerline[i] = [x, y, z]
    
    # Create fiber
    fiber = Fiber(
        centerline=centerline,
        radius=fiber_radius,
        material=material or Material(),
        segments=num_segments
    )
    
    return fiber


def arc_fiber_2d(
    radius_arc: float = 20.0,
    angle: float = np.pi / 2,
    fiber_radius: float = 0.5,
    num_segments: int = 50,
    center: Tuple[float, float] = (25.0, 25.0),
    start_angle: float = 0.0,
    material: Optional[Material] = None,
) -> Fiber:
    """
    Generate a 2D arc (circular) fiber.
    
    Parameters
    ----------
    radius_arc : float
        Arc radius
    angle : float
        Arc angle (radians)
    fiber_radius : float
        Fiber cross-section radius
    num_segments : int
        Number of centerline points
    center : tuple
        Arc center (x, y)
    start_angle : float
        Starting angle (radians)
    material : Material, optional
        Fiber material
    
    Returns
    -------
    Fiber
        Arc fiber
    
    Examples
    --------
    >>> fiber = arc_fiber_2d(radius_arc=20, angle=np.pi/2)
    """
    # Generate arc centerline
    t = np.linspace(start_angle, start_angle + angle, num_segments)
    
    centerline = np.zeros((num_segments, 3))
    center = np.array([center[0], center[1], 0.0])
    
    for i, ti in enumerate(t):
        x = center[0] + radius_arc * np.cos(ti)
        y = center[1] + radius_arc * np.sin(ti)
        centerline[i] = [x, y, 0.0]
    
    # Create fiber
    fiber = Fiber(
        centerline=centerline,
        radius=fiber_radius,
        material=material or Material(),
        segments=num_segments
    )
    
    return fiber


def bezier_fiber_3d(
    control_points: List[Tuple[float, float, float]],
    fiber_radius: float = 0.5,
    num_segments: int = 50,
    material: Optional[Material] = None,
) -> Fiber:
    """
    Generate a 3D Bezier curve fiber.
    
    Parameters
    ----------
    control_points : list of tuples
        Bezier control points [(x,y,z), ...]
    fiber_radius : float
        Fiber cross-section radius
    num_segments : int
        Number of centerline points
    material : Material, optional
        Fiber material
    
    Returns
    -------
    Fiber
        Bezier fiber
    
    Examples
    --------
    >>> control = [(0, 0, 0), (10, 20, 0), (20, -10, 0), (30, 0, 0)]
    >>> fiber = bezier_fiber_3d(control)
    """
    # Evaluate Bezier curve
    control_points = np.array(control_points)
    n = len(control_points) - 1
    
    t_values = np.linspace(0, 1, num_segments)
    centerline = np.zeros((num_segments, 3))
    
    for i, t in enumerate(t_values):
        point = np.zeros(3)
        for j, cp in enumerate(control_points):
            # Bernstein polynomial
            binom = comb(n, j)
            weight = binom * (t ** j) * ((1 - t) ** (n - j))
            point += weight * cp
        centerline[i] = point
    
    # Create fiber
    fiber = Fiber(
        centerline=centerline,
        radius=fiber_radius,
        material=material or Material(),
        segments=num_segments
    )
    
    return fiber


def random_curved_network_3d(
    num_fibers: int = 20,
    box_size: Tuple[float, float, float] = (50.0, 50.0, 50.0),
    min_length: float = 10.0,
    max_length: float = 30.0,
    fiber_radius: float = 0.5,
    curvature: float = 0.3,
    num_control_points: int = 4,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """
    Generate a 3D network of randomly curved fibers.
    
    Parameters
    ----------
    num_fibers : int
        Number of fibers
    box_size : tuple
        Simulation box size (x, y, z)
    min_length : float
        Minimum fiber length
    max_length : float
        Maximum fiber length
    fiber_radius : float
        Fiber cross-section radius
    curvature : float
        Curvature magnitude (0 = straight, 1 = highly curved)
    num_control_points : int
        Number of control points for Bezier curves
    material : Material, optional
        Fiber material
    seed : int, optional
        Random seed
    
    Returns
    -------
    FiberNetwork
        Network with curved fibers
    
    Examples
    --------
    >>> net = random_curved_network_3d(num_fibers=50, curvature=0.5)
    """
    if seed is not None:
        np.random.seed(seed)
    
    net = FiberNetwork(dimension=3)
    box_size = np.array(box_size)
    
    for i in range(num_fibers):
        # Random start point
        start = np.random.uniform(0, 1, 3) * box_size
        
        # Random direction and length
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
        length = np.random.uniform(min_length, max_length)
        
        # Generate control points along direction with random perturbations
        control_points = []
        for j in range(num_control_points):
            t = j / (num_control_points - 1)
            base_point = start + t * length * direction
            
            # Add random perturbation (curvature)
            if 0 < j < num_control_points - 1:
                perturbation = np.random.randn(3) * curvature * length
                base_point += perturbation
            
            control_points.append(tuple(base_point))
        
        # Create Bezier fiber
        fiber = bezier_fiber_3d(
            control_points=control_points,
            fiber_radius=fiber_radius,
            material=material,
            num_segments=30
        )
        fiber.fiber_id = i
        
        net.add_fiber(fiber)
    
    return net


def crimped_network_2d(
    num_fibers: int = 30,
    box_size: Tuple[float, float] = (100.0, 100.0),
    fiber_length: float = 30.0,
    crimp_amplitude: float = 2.0,
    crimp_wavelength: float = 10.0,
    fiber_radius: float = 0.5,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """
    Generate a 2D network of crimped (wavy) fibers.
    
    Common in biological tissues and non-woven mats.
    
    Parameters
    ----------
    num_fibers : int
        Number of fibers
    box_size : tuple
        Simulation box size (x, y)
    fiber_length : float
        Fiber length along main axis
    crimp_amplitude : float
        Crimp amplitude
    crimp_wavelength : float
        Crimp wavelength
    fiber_radius : float
        Fiber cross-section radius
    material : Material, optional
        Fiber material
    seed : int, optional
        Random seed
    
    Returns
    -------
    FiberNetwork
        Network with crimped fibers
    
    Examples
    --------
    >>> net = crimped_network_2d(num_fibers=50, crimp_amplitude=3.0)
    """
    if seed is not None:
        np.random.seed(seed)
    
    net = FiberNetwork(dimension=2)
    box_size = np.array(box_size)
    
    for i in range(num_fibers):
        # Random position and orientation
        start = np.random.uniform(0, 1, 2) * box_size
        orientation = np.random.uniform(0, np.pi)
        
        # Create sinusoidal fiber
        fiber = sinusoidal_fiber_2d(
            length=fiber_length,
            amplitude=crimp_amplitude,
            wavelength=crimp_wavelength,
            radius=fiber_radius,
            orientation=orientation,
            start=tuple(start),
            material=material,
            seed=i
        )
        fiber.fiber_id = i
        
        net.add_fiber(fiber)
    
    return net


