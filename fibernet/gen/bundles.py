"""
Fiber Bundle Generators

Generate fiber bundles - groups of fibers that act together as a unit.
Common in biological tissues (tendons, muscles, ligaments) and composite materials.

Types of bundles:
- Parallel bundles: fibers aligned in parallel
- Twisted bundles: helical twist around central axis
- Braided bundles: interwoven fiber groups
- Random bundles: loosely grouped fibers

References:
- Fratzl, P. "Cellulosic and collagenous materials", Springer, 2008
- Silver, F.H. "Biomaterials", Chapman & Hall, 1994
"""

import numpy as np
from typing import Tuple, Optional
from fibernet.core.network import FiberNetwork
from fibernet.core.fiber import Fiber
from fibernet.core.material import Material


def parallel_bundle_2d(
    num_fibers: int = 10,
    bundle_length: float = 50.0,
    bundle_width: float = 5.0,
    fiber_radius: float = 0.5,
    orientation: float = 0.0,
    center: Tuple[float, float] = (25.0, 25.0),
    material: Optional[Material] = None,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """
    Generate a 2D parallel fiber bundle.
    
    Parameters
    ----------
    num_fibers : int
        Number of fibers in bundle
    bundle_length : float
        Length of the bundle
    bundle_width : float
        Width of the bundle (perpendicular to fiber direction)
    fiber_radius : float
        Radius of individual fibers
    orientation : float
        Bundle orientation angle in radians (0 = horizontal)
    center : tuple
        Center position (x, y)
    material : Material, optional
        Fiber material properties
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    FiberNetwork
        Network containing the parallel bundle
    
    Examples
    --------
    >>> net = parallel_bundle_2d(num_fibers=10, bundle_length=50.0)
    >>> print(f"Created bundle with {len(net.fibers)} fibers")
    """
    if seed is not None:
        np.random.seed(seed)
    
    net = FiberNetwork(dimension=2)
    
    # Direction vectors
    direction = np.array([np.cos(orientation), np.sin(orientation), 0.0])
    perpendicular = np.array([-np.sin(orientation), np.cos(orientation), 0.0])
    
    # Generate fibers
    fiber_id = 0
    for i in range(num_fibers):
        # Position across bundle width
        offset = (i / (num_fibers - 1) - 0.5) * bundle_width if num_fibers > 1 else 0.0
        
        # Small random perturbation
        perturbation = np.random.normal(0, 0.05 * bundle_width) if seed is not None else 0.0
        offset += perturbation
        
        # Fiber endpoints
        center_pos = np.array([center[0], center[1], 0.0]) + offset * perpendicular
        start = center_pos - 0.5 * bundle_length * direction
        end = center_pos + 0.5 * bundle_length * direction
        
        # Add fiber
        fiber = Fiber.straight(start, end, radius=fiber_radius, material=material, fiber_id=fiber_id)
        net.add_fiber(fiber)
        fiber_id += 1
    
    return net


def twisted_bundle_2d(
    num_fibers: int = 8,
    bundle_length: float = 50.0,
    twist_pitch: float = 20.0,
    bundle_radius: float = 3.0,
    fiber_radius: float = 0.5,
    center: Tuple[float, float] = (25.0, 25.0),
    material: Optional[Material] = None,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """
    Generate a 2D twisted (helical) fiber bundle.
    
    Fibers follow helical paths around a central axis, creating a rope-like structure.
    
    Parameters
    ----------
    num_fibers : int
        Number of fibers in bundle
    bundle_length : float
        Length of the bundle along axis
    twist_pitch : float
        Distance for one complete twist
    bundle_radius : float
        Radius of the bundle cross-section
    fiber_radius : float
        Radius of individual fibers
    center : tuple
        Center position (x, y)
    material : Material, optional
        Fiber material properties
    seed : int, optional
        Random seed
    
    Returns
    -------
    FiberNetwork
        Network containing the twisted bundle
    
    Examples
    --------
    >>> net = twisted_bundle_2d(num_fibers=8, bundle_length=50.0, twist_pitch=20.0)
    """
    if seed is not None:
        np.random.seed(seed)
    
    net = FiberNetwork(dimension=2)
    
    # Generate helical paths
    fiber_id = 0
    for i in range(num_fibers):
        # Angular offset for this fiber
        angle_offset = 2 * np.pi * i / num_fibers
        
        # Start and end positions
        z_start = -bundle_length / 2
        z_end = bundle_length / 2
        
        # Start position
        angle_start = angle_offset + 2 * np.pi * z_start / twist_pitch
        x_start = center[0] + bundle_radius * np.cos(angle_start)
        y_start = center[1] + z_start
        start = np.array([x_start, y_start, 0.0])
        
        # End position
        angle_end = angle_offset + 2 * np.pi * z_end / twist_pitch
        x_end = center[0] + bundle_radius * np.cos(angle_end)
        y_end = center[1] + z_end
        end = np.array([x_end, y_end, 0.0])
        
        # Create fiber
        fiber = Fiber.straight(start, end, radius=fiber_radius, material=material, fiber_id=fiber_id)
        net.add_fiber(fiber)
        fiber_id += 1
    
    return net


def random_bundle_3d(
    num_fibers: int = 20,
    bundle_length: float = 50.0,
    bundle_radius: float = 5.0,
    fiber_radius: float = 0.5,
    orientation_variance: float = 0.2,
    center: Tuple[float, float, float] = (25.0, 25.0, 25.0),
    direction: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    material: Optional[Material] = None,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """
    Generate a 3D random fiber bundle with orientation spread.
    
    Fibers are roughly aligned with a main direction but have random angular spread.
    
    Parameters
    ----------
    num_fibers : int
        Number of fibers
    bundle_length : float
        Mean fiber length
    bundle_radius : float
        Radius of bundle cross-section
    fiber_radius : float
        Radius of individual fibers
    orientation_variance : float
        Standard deviation of angular deviation from main direction (radians)
    center : tuple
        Center position (x, y, z)
    direction : tuple
        Main bundle direction (will be normalized)
    material : Material, optional
        Fiber material
    seed : int, optional
        Random seed
    
    Returns
    -------
    FiberNetwork
        3D network containing the bundle
    
    Examples
    --------
    >>> net = random_bundle_3d(num_fibers=20, bundle_length=50.0, orientation_variance=0.2)
    """
    if seed is not None:
        np.random.seed(seed)
    
    net = FiberNetwork(dimension=3)
    
    # Normalize main direction
    direction = np.array(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    
    # Create orthonormal basis
    # Find a vector not parallel to direction
    if abs(direction[0]) < 0.9:
        perp1 = np.cross(direction, [1, 0, 0])
    else:
        perp1 = np.cross(direction, [0, 1, 0])
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)
    
    center = np.array(center)
    
    fiber_id = 0
    for i in range(num_fibers):
        # Random position in bundle cross-section
        r = bundle_radius * np.sqrt(np.random.uniform(0, 1))
        theta = np.random.uniform(0, 2 * np.pi)
        offset = r * (np.cos(theta) * perp1 + np.sin(theta) * perp2)
        
        # Random orientation (Gaussian angular spread)
        angle_x = np.random.normal(0, orientation_variance)
        angle_y = np.random.normal(0, orientation_variance)
        
        # Perturbed direction
        fiber_dir = (
            direction + 
            angle_x * perp1 + 
            angle_y * perp2
        )
        fiber_dir = fiber_dir / np.linalg.norm(fiber_dir)
        
        # Random length
        length = bundle_length * np.random.uniform(0.8, 1.2)
        
        # Fiber endpoints
        start = center + offset - 0.5 * length * fiber_dir
        end = center + offset + 0.5 * length * fiber_dir
        
        # Add fiber
        fiber = Fiber.straight(start, end, radius=fiber_radius, material=material, fiber_id=fiber_id)
        net.add_fiber(fiber)
        fiber_id += 1
    
    return net


def braided_bundle_3d(
    num_strands: int = 6,
    bundle_length: float = 50.0,
    braid_radius: float = 5.0,
    fibers_per_strand: int = 3,
    strand_radius: float = 0.5,
    center: Tuple[float, float, float] = (25.0, 25.0, 25.0),
    material: Optional[Material] = None,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """
    Generate a 3D braided fiber bundle.
    
    Creates a braided structure where multiple strands interweave around a central axis.
    
    Parameters
    ----------
    num_strands : int
        Number of braided strands (should be even for proper braiding)
    bundle_length : float
        Length of the braid
    braid_radius : float
        Radius of the braid pattern
    fibers_per_strand : int
        Number of fibers in each strand
    strand_radius : float
        Radius of each strand
    center : tuple
        Center position (x, y, z)
    material : Material, optional
        Fiber material
    seed : int, optional
        Random seed
    
    Returns
    -------
    FiberNetwork
        3D network containing the braided bundle
    
    Examples
    --------
    >>> net = braided_bundle_3d(num_strands=6, bundle_length=50.0)
    """
    if seed is not None:
        np.random.seed(seed)
    
    net = FiberNetwork(dimension=3)
    center = np.array(center)
    
    fiber_id = 0
    
    # Generate braided strands
    for strand_idx in range(num_strands):
        # Determine braiding pattern (alternating over/under)
        is_clockwise = (strand_idx % 2 == 0)
        angular_offset = 2 * np.pi * (strand_idx // 2) / (num_strands // 2)
        
        # Generate strand start and end points
        t_start = 0.0
        t_end = 1.0
        
        # Start position
        z_start = t_start * bundle_length - bundle_length / 2
        angle_start = angular_offset + 2 * np.pi * t_start * (1.0 if is_clockwise else -1.0)
        r_start = braid_radius * (1.0 + 0.3 * np.sin(4 * np.pi * t_start))
        x_start = center[0] + r_start * np.cos(angle_start)
        y_start = center[1] + r_start * np.sin(angle_start)
        strand_start = np.array([x_start, y_start, center[2] + z_start])
        
        # End position
        z_end = t_end * bundle_length - bundle_length / 2
        angle_end = angular_offset + 2 * np.pi * t_end * (1.0 if is_clockwise else -1.0)
        r_end = braid_radius * (1.0 + 0.3 * np.sin(4 * np.pi * t_end))
        x_end = center[0] + r_end * np.cos(angle_end)
        y_end = center[1] + r_end * np.sin(angle_end)
        strand_end = np.array([x_end, y_end, center[2] + z_end])
        
        # Strand direction
        strand_dir = strand_end - strand_start
        strand_dir = strand_dir / np.linalg.norm(strand_dir)
        
        # Create orthonormal basis for strand cross-section
        if abs(strand_dir[0]) < 0.9:
            perp1 = np.cross(strand_dir, [1, 0, 0])
        else:
            perp1 = np.cross(strand_dir, [0, 1, 0])
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(strand_dir, perp1)
        
        strand_center = (strand_start + strand_end) / 2
        
        # Add fibers within this strand
        for fiber_idx in range(fibers_per_strand):
            # Position within strand
            r = strand_radius * np.sqrt(np.random.uniform(0.3, 1.0))
            theta = np.random.uniform(0, 2 * np.pi)
            offset = r * (np.cos(theta) * perp1 + np.sin(theta) * perp2)
            
            # Fiber endpoints
            start = strand_center + offset - 0.4 * bundle_length * strand_dir
            end = strand_center + offset + 0.4 * bundle_length * strand_dir
            
            fiber = Fiber.straight(start, end, radius=strand_radius / 3, material=material, fiber_id=fiber_id)
            net.add_fiber(fiber)
            fiber_id += 1
    
    return net


def tendon_like_bundle_3d(
    num_fibers: int = 30,
    bundle_length: float = 80.0,
    bundle_radius: float = 8.0,
    fiber_radius: float = 0.8,
    crimp_amplitude: float = 1.0,
    crimp_wavelength: float = 10.0,
    center: Tuple[float, float, float] = (40.0, 40.0, 40.0),
    material: Optional[Material] = None,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """
    Generate a tendon-like wavy (crimped) fiber bundle.
    
    Biological tendons have fibers with characteristic crimp patterns.
    This generator creates fibers with sinusoidal waviness.
    
    Parameters
    ----------
    num_fibers : int
        Number of fibers
    bundle_length : float
        Length of the bundle
    bundle_radius : float
        Radius of bundle cross-section
    fiber_radius : float
        Radius of individual fibers
    crimp_amplitude : float
        Amplitude of fiber crimp (waviness)
    crimp_wavelength : float
        Wavelength of crimp pattern
    center : tuple
        Center position (x, y, z)
    material : Material, optional
        Fiber material
    seed : int, optional
        Random seed
    
    Returns
    -------
    FiberNetwork
        3D network with crimped fiber bundle
    
    Examples
    --------
    >>> net = tendon_like_bundle_3d(num_fibers=30, bundle_length=80.0, crimp_amplitude=1.0)
    """
    if seed is not None:
        np.random.seed(seed)
    
    net = FiberNetwork(dimension=3)
    center = np.array(center)
    
    # Main direction (along z-axis)
    main_dir = np.array([0.0, 0.0, 1.0])
    
    # Orthonormal basis
    perp1 = np.array([1.0, 0.0, 0.0])
    perp2 = np.array([0.0, 1.0, 0.0])
    
    fiber_id = 0
    for i in range(num_fibers):
        # Random position in cross-section
        r = bundle_radius * np.sqrt(np.random.uniform(0, 1))
        theta = np.random.uniform(0, 2 * np.pi)
        offset = r * (np.cos(theta) * perp1 + np.sin(theta) * perp2)
        
        # Random phase for crimp
        phase = np.random.uniform(0, 2 * np.pi)
        
        # Start position
        z_start = -bundle_length / 2
        crimp_start = crimp_amplitude * np.sin(2 * np.pi * z_start / crimp_wavelength + phase)
        crimp_dir_start = np.random.uniform(0, 2 * np.pi)
        crimp_offset_start = crimp_start * (np.cos(crimp_dir_start) * perp1 + np.sin(crimp_dir_start) * perp2)
        start = center + offset + z_start * main_dir + crimp_offset_start
        
        # End position
        z_end = bundle_length / 2
        crimp_end = crimp_amplitude * np.sin(2 * np.pi * z_end / crimp_wavelength + phase)
        crimp_dir_end = np.random.uniform(0, 2 * np.pi)
        crimp_offset_end = crimp_end * (np.cos(crimp_dir_end) * perp1 + np.sin(crimp_dir_end) * perp2)
        end = center + offset + z_end * main_dir + crimp_offset_end
        
        # Create fiber
        fiber = Fiber.straight(start, end, radius=fiber_radius, material=material, fiber_id=fiber_id)
        net.add_fiber(fiber)
        fiber_id += 1
    
    return net


