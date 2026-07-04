"""
Disordered fiber network generators.

Generates random/disordered fiber networks in 2D and 3D, including:
- Random deposition (Mikado model)
- Random walk fibers
- Cross-linked random networks
- Fiber networks with controlled orientation distribution
- Poisson line processes
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from scipy.spatial import cKDTree

from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork, Crosslink
from fibernet.core.material import Material


def random_straight_2d(
    num_fibers: int = 100,
    fiber_length: float = 10.0,
    box_size: Tuple[float, float] = (50.0, 50.0),
    radius: float = 0.1,
    length_std: float = 0.0,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """Generate a 2D random straight fiber network (Mikado model).
    
    Parameters
    ----------
    num_fibers : int
        Number of fibers.
    fiber_length : float
        Mean fiber length.
    box_size : tuple
        (Lx, Ly) domain size.
    radius : float
        Fiber radius.
    length_std : float
        Standard deviation of fiber length (Gaussian).
    material : Material, optional
        Fiber material.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    FiberNetwork
    """
    rng = np.random.default_rng(seed)
    mat = material or Material(name="generic_fiber")
    net = FiberNetwork(
        dimension=2,
        box_size=np.array([box_size[0], box_size[1], 0.0]),
        metadata={
            "generator": "random_straight_2d",
            "num_fibers": num_fibers,
            "fiber_length": fiber_length,
            "box_size": box_size,
        },
    )
    
    Lx, Ly = box_size
    
    for i in range(num_fibers):
        length = fiber_length + rng.normal(0, length_std) if length_std > 0 else fiber_length
        length = max(length, 0.1)
        
        cx = rng.uniform(0, Lx)
        cy = rng.uniform(0, Ly)
        angle = rng.uniform(0, np.pi)
        
        dx = 0.5 * length * np.cos(angle)
        dy = 0.5 * length * np.sin(angle)
        
        start = np.array([cx - dx, cy - dy, 0.0])
        end = np.array([cx + dx, cy + dy, 0.0])
        
        fiber = Fiber.straight(start, end, radius=radius, material=mat, fiber_id=i)
        net.add_fiber(fiber)
    
    net.auto_crosslink(threshold=2.0 * radius)
    return net


def random_straight_3d(
    num_fibers: int = 100,
    fiber_length: float = 10.0,
    box_size: Tuple[float, float, float] = (50.0, 50.0, 50.0),
    radius: float = 0.1,
    length_std: float = 0.0,
    orientation_bias: Optional[np.ndarray] = None,
    orientation_spread: float = 0.5,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """Generate a 3D random straight fiber network.
    
    Parameters
    ----------
    num_fibers : int
        Number of fibers.
    fiber_length : float
        Mean fiber length.
    box_size : tuple
        (Lx, Ly, Lz) domain size.
    radius : float
        Fiber radius.
    length_std : float
        Standard deviation of fiber length.
    orientation_bias : array-like, optional
        Preferred direction vector. If None, fully isotropic.
    orientation_spread : float
        Angular spread around bias direction (radians). Only used with orientation_bias.
    material : Material, optional
        Fiber material.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    FiberNetwork
    """
    rng = np.random.default_rng(seed)
    mat = material or Material(name="generic_fiber")
    net = FiberNetwork(
        dimension=3,
        box_size=np.array(box_size),
        metadata={
            "generator": "random_straight_3d",
            "num_fibers": num_fibers,
            "fiber_length": fiber_length,
            "box_size": box_size,
        },
    )
    
    Lx, Ly, Lz = box_size
    
    for i in range(num_fibers):
        length = fiber_length + rng.normal(0, length_std) if length_std > 0 else fiber_length
        length = max(length, 0.1)
        
        cx = rng.uniform(0, Lx)
        cy = rng.uniform(0, Ly)
        cz = rng.uniform(0, Lz)
        center = np.array([cx, cy, cz])
        
        if orientation_bias is not None:
            bias = np.asarray(orientation_bias, dtype=np.float64)
            bias = bias / np.linalg.norm(bias)
            phi = rng.normal(0, orientation_spread)
            theta = rng.uniform(0, 2 * np.pi)
            
            if abs(bias[0]) < 0.9:
                perp1 = np.cross(bias, [1, 0, 0])
            else:
                perp1 = np.cross(bias, [0, 1, 0])
            perp1 /= np.linalg.norm(perp1)
            perp2 = np.cross(bias, perp1)
            
            direction = (
                np.cos(phi) * bias
                + np.sin(phi) * np.cos(theta) * perp1
                + np.sin(phi) * np.sin(theta) * perp2
            )
            direction /= np.linalg.norm(direction)
        else:
            theta = rng.uniform(0, 2 * np.pi)
            cos_phi = rng.uniform(-1, 1)
            sin_phi = np.sqrt(1 - cos_phi**2)
            direction = np.array([
                sin_phi * np.cos(theta),
                sin_phi * np.sin(theta),
                cos_phi,
            ])
        
        start = center - 0.5 * length * direction
        end = center + 0.5 * length * direction
        
        fiber = Fiber.straight(start, end, radius=radius, material=mat, fiber_id=i)
        net.add_fiber(fiber)
    
    net.auto_crosslink(threshold=2.0 * radius)
    return net


def random_walk_fibers(
    num_fibers: int = 50,
    num_steps: int = 100,
    step_length: float = 0.5,
    box_size: Tuple[float, float, float] = (50.0, 50.0, 50.0),
    radius: float = 0.1,
    persistence_length: float = 5.0,
    dimension: int = 3,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """Generate fibers using random walks with persistence.
    
    Models semi-flexible polymers with worm-like chain behavior.
    
    Parameters
    ----------
    num_fibers : int
        Number of fibers.
    num_steps : int
        Steps per fiber.
    step_length : float
        Length of each step.
    box_size : tuple
        Domain size.
    radius : float
        Fiber radius.
    persistence_length : float
        Persistence length controlling fiber stiffness.
        Higher = straighter fibers.
    dimension : int
        2 or 3.
    material : Material, optional
        Fiber material.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    FiberNetwork
    """
    rng = np.random.default_rng(seed)
    mat = material or Material(name="generic_fiber")
    net = FiberNetwork(
        dimension=dimension,
        box_size=np.array(box_size),
        metadata={
            "generator": "random_walk_fibers",
            "num_fibers": num_fibers,
            "persistence_length": persistence_length,
        },
    )
    
    Lx, Ly, Lz = box_size
    kappa = step_length / persistence_length  # bending flexibility
    
    for i in range(num_fibers):
        points = np.zeros((num_steps + 1, 3))
        points[0] = rng.uniform([0, 0, 0], [Lx, Ly, Lz])
        
        if dimension == 2:
            angle = rng.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle), 0.0])
        else:
            theta = rng.uniform(0, 2 * np.pi)
            cos_phi = rng.uniform(-1, 1)
            sin_phi = np.sqrt(1 - cos_phi**2)
            direction = np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
        
        for s in range(1, num_steps + 1):
            if dimension == 2:
                d_angle = rng.normal(0, kappa)
                angle = np.arctan2(direction[1], direction[0]) + d_angle
                direction = np.array([np.cos(angle), np.sin(angle), 0.0])
            else:
                bend_angle = rng.normal(0, kappa)
                twist_angle = rng.uniform(0, 2 * np.pi)
                
                if abs(direction[0]) < 0.9:
                    perp1 = np.cross(direction, [1, 0, 0])
                else:
                    perp1 = np.cross(direction, [0, 1, 0])
                perp1 /= np.linalg.norm(perp1)
                perp2 = np.cross(direction, perp1)
                
                direction = (
                    np.cos(bend_angle) * direction
                    + np.sin(bend_angle) * np.cos(twist_angle) * perp1
                    + np.sin(bend_angle) * np.sin(twist_angle) * perp2
                )
                direction /= np.linalg.norm(direction)
            
            points[s] = points[s - 1] + step_length * direction
        
        fiber = Fiber(
            centerline=points, radius=radius, material=mat,
            segments=num_steps, fiber_id=i,
        )
        net.add_fiber(fiber)
    
    net.auto_crosslink(threshold=2.5 * radius)
    return net


def oriented_random_2d(
    num_fibers: int = 100,
    fiber_length: float = 10.0,
    box_size: Tuple[float, float] = (50.0, 50.0),
    radius: float = 0.1,
    preferred_angle: float = 0.0,
    angular_spread: float = 0.3,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """Generate a 2D fiber network with preferred orientation.
    
    Parameters
    ----------
    preferred_angle : float
        Preferred fiber angle in radians.
    angular_spread : float
        Standard deviation of angular distribution (radians).
    """
    rng = np.random.default_rng(seed)
    mat = material or Material(name="generic_fiber")
    net = FiberNetwork(
        dimension=2,
        box_size=np.array([box_size[0], box_size[1], 0.0]),
        metadata={
            "generator": "oriented_random_2d",
            "preferred_angle": preferred_angle,
            "angular_spread": angular_spread,
        },
    )
    
    Lx, Ly = box_size
    
    for i in range(num_fibers):
        length = fiber_length + rng.normal(0, fiber_length * 0.1)
        length = max(length, 0.1)
        
        cx = rng.uniform(0, Lx)
        cy = rng.uniform(0, Ly)
        angle = preferred_angle + rng.normal(0, angular_spread)
        
        dx = 0.5 * length * np.cos(angle)
        dy = 0.5 * length * np.sin(angle)
        
        start = np.array([cx - dx, cy - dy, 0.0])
        end = np.array([cx + dx, cy + dy, 0.0])
        
        fiber = Fiber.straight(start, end, radius=radius, material=mat, fiber_id=i)
        net.add_fiber(fiber)
    
    net.auto_crosslink(threshold=2.0 * radius)
    return net


def poisson_line_network_2d(
    line_density: float = 0.5,
    box_size: Tuple[float, float] = (50.0, 50.0),
    radius: float = 0.1,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """Generate a 2D Poisson line process fiber network.
    
    Parameters
    ----------
    line_density : float
        Expected number of lines per unit length.
    box_size : tuple
        (Lx, Ly) domain size.
    """
    rng = np.random.default_rng(seed)
    mat = material or Material(name="generic_fiber")
    net = FiberNetwork(
        dimension=2,
        box_size=np.array([box_size[0], box_size[1], 0.0]),
        metadata={"generator": "poisson_line_network_2d", "line_density": line_density},
    )
    
    Lx, Ly = box_size
    diagonal = np.sqrt(Lx**2 + Ly**2)
    expected_lines = int(line_density * diagonal * np.pi)
    num_lines = rng.poisson(expected_lines)
    
    for i in range(num_lines):
        angle = rng.uniform(0, np.pi)
        dist = rng.uniform(-diagonal / 2, diagonal / 2)
        
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        if abs(cos_a) > abs(sin_a):
            y0 = dist / cos_a
            p1 = np.array([0.0, y0, 0.0])
            p2 = np.array([Lx, y0 + Lx * np.tan(angle + np.pi / 2), 0.0])
        else:
            x0 = dist / sin_a
            p1 = np.array([x0, 0.0, 0.0])
            p2 = np.array([x0 + Ly / np.tan(angle + np.pi / 2), Ly, 0.0])
        
        fiber = Fiber.straight(p1, p2, radius=radius, material=mat, fiber_id=i)
        net.add_fiber(fiber)
    
    net.auto_crosslink(threshold=2.5 * radius)
    return net


def oriented_random_3d(
    num_fibers: int = 100,
    fiber_length: float = 10.0,
    box_size: Tuple[float, float, float] = (50.0, 50.0, 50.0),
    radius: float = 0.1,
    preferred_direction: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    angular_spread: float = 0.3,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """Generate a 3D fiber network with preferred orientation.

    Fibers are distributed randomly in 3D space but tend to align
    with the preferred direction, with angular spread controlling
    the degree of alignment.

    Parameters
    ----------
    num_fibers : int
        Number of fibers.
    fiber_length : float
        Length of each fiber.
    box_size : tuple of float
        (Lx, Ly, Lz) box dimensions.
    radius : float
        Fiber radius.
    preferred_direction : tuple of float
        Preferred alignment direction (will be normalized).
    angular_spread : float
        Standard deviation of angular distribution (radians).
        0 = perfect alignment, pi/2 = isotropic.
    material : Material, optional
        Material properties.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    FiberNetwork
        3D oriented fiber network.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    if material is None:
        material = Material()

    # Normalize preferred direction
    pref_dir = np.array(preferred_direction, dtype=float)
    pref_dir /= np.linalg.norm(pref_dir)

    # Create orthogonal basis
    if abs(pref_dir[0]) < 0.9:
        up = np.array([1.0, 0.0, 0.0])
    else:
        up = np.array([0.0, 1.0, 0.0])
    u = np.cross(pref_dir, up)
    u /= np.linalg.norm(u)
    v = np.cross(pref_dir, u)

    network = FiberNetwork(dimension=3)

    for i in range(num_fibers):
        # Sample orientation using von Mises-Fisher-like distribution
        theta = rng.normal(0, angular_spread)
        phi = rng.uniform(0, 2 * np.pi)

        # Direction in local frame
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        local_dir = np.array([cos_theta, sin_theta * np.cos(phi), sin_theta * np.sin(phi)])

        # Transform to global frame
        direction = local_dir[0] * pref_dir + local_dir[1] * u + local_dir[2] * v
        direction /= np.linalg.norm(direction)

        # Random center
        center = np.array([
            rng.uniform(0, box_size[0]),
            rng.uniform(0, box_size[1]),
            rng.uniform(0, box_size[2]),
        ])

        # Create fiber centerline
        start = center - 0.5 * fiber_length * direction
        end = center + 0.5 * fiber_length * direction
        centerline = np.array([start, end])

        fiber = Fiber(centerline=centerline, radius=radius, material=material, fiber_id=i)
        network.add_fiber(fiber)

    network.detect_contacts()
    return network


def random_curved_fibers_3d(
    num_fibers: int = 50,
    fiber_length: float = 10.0,
    box_size: Tuple[float, float, float] = (50.0, 50.0, 50.0),
    radius: float = 0.1,
    curvature: float = 0.5,
    num_segments: int = 20,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """Generate a 3D network of curved fibers.

    Fibers are generated as curved paths with controlled curvature.

    Parameters
    ----------
    num_fibers : int
        Number of fibers.
    fiber_length : float
        Arc length of each fiber.
    box_size : tuple of float
        (Lx, Ly, Lz) box dimensions.
    radius : float
        Fiber radius.
    curvature : float
        Curvature magnitude (higher = more curved).
    num_segments : int
        Number of segments per fiber.
    material : Material, optional
        Material properties.
    seed : int, optional
        Random seed.

    Returns
    -------
    FiberNetwork
        3D curved fiber network.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    if material is None:
        material = Material()

    network = FiberNetwork(dimension=3)
    step_length = fiber_length / num_segments

    for i in range(num_fibers):
        # Random starting point and direction
        pos = np.array([
            rng.uniform(0, box_size[0]),
            rng.uniform(0, box_size[1]),
            rng.uniform(0, box_size[2]),
        ])

        # Random initial direction
        direction = rng.randn(3)
        direction /= np.linalg.norm(direction)

        centerline = [pos.copy()]

        for _ in range(num_segments - 1):
            # Random curvature perturbation
            perturbation = rng.randn(3) * curvature
            # Remove component along current direction (keep orthogonal)
            perturbation -= np.dot(perturbation, direction) * direction
            direction = direction + perturbation
            direction /= np.linalg.norm(direction)

            pos = pos + step_length * direction
            centerline.append(pos.copy())

        centerline = np.array(centerline)

        fiber = Fiber(centerline=centerline, radius=radius, material=material, fiber_id=i)
        network.add_fiber(fiber)

    network.detect_contacts()
    return network
