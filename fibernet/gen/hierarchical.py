"""
Hierarchical and super-structure fiber network generators.

Generates multi-scale fiber structures:
- Hierarchical bundles (fibers within bundles within bundles)
- Fractal-like networks
- Gradient density networks
- Core-shell fiber arrangements
- Random multi-scale networks
"""

import numpy as np
from typing import Optional, Tuple, List
from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork
from fibernet.core.material import Material
from fibernet.gen.chiral import _hexagonal_pack


def hierarchical_bundle(
    levels: int = 3,
    fibers_per_level: List[int] = None,
    radii_per_level: List[float] = None,
    total_length: float = 100.0,
    twist_per_level: List[float] = None,
    material: Optional[Material] = None,
    num_points: int = 200,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """Generate a hierarchical fiber bundle with multiple packing levels.
    
    Each level wraps sub-bundles into a larger bundle with twist.
    
    Parameters
    ----------
    levels : int
        Number of hierarchical levels.
    fibers_per_level : list of int
        Number of sub-elements at each level.
    radii_per_level : list of float
        Bundle radius at each level.
    twist_per_level : list of float
        Twist angle at each level (radians).
    """
    rng = np.random.default_rng(seed)
    mat = material or Material(name="hierarchical")
    
    if fibers_per_level is None:
        fibers_per_level = [7] * levels
    if radii_per_level is None:
        radii_per_level = [2.0 * (0.5 ** i) for i in range(levels)]
    if twist_per_level is None:
        twist_per_level = [np.pi / 6] * levels
    
    net = FiberNetwork(
        dimension=3,
        box_size=np.array([2 * radii_per_level[0], 2 * radii_per_level[0], total_length]),
        metadata={
            "generator": "hierarchical_bundle",
            "levels": levels,
            "fibers_per_level": fibers_per_level,
        },
    )
    
    base_radius = radii_per_level[-1] / max(fibers_per_level[-1], 1)
    z_vals = np.linspace(0, total_length, num_points)
    
    def _generate_level(center_x, center_y, level, cumulative_twist=0.0):
        if level >= levels:
            points = np.zeros((num_points, 3))
            total_twist = cumulative_twist
            for k, z in enumerate(z_vals):
                angle = total_twist * z / total_length
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                rx = center_x * cos_a - center_y * sin_a
                ry = center_x * sin_a + center_y * cos_a
                points[k] = [rx, ry, z]
            fiber = Fiber(centerline=points, radius=base_radius, material=mat, fiber_id=net.num_fibers)
            net.add_fiber(fiber)
            return
        
        n_sub = fibers_per_level[level]
        r_sub = radii_per_level[level]
        offsets = _hexagonal_pack(n_sub, r_sub / max(n_sub, 1))
        
        for ox, oy in offsets:
            sub_cx = center_x + ox
            sub_cy = center_y + oy
            
            # Twist is applied during fiber generation (see _generate_level base case)
            
            next_twist = cumulative_twist + (twist_per_level[level] if level < levels else 0)
            _generate_level(sub_cx, sub_cy, level + 1, next_twist)
    
    _generate_level(0.0, 0.0, 0, 0.0)
    return net


def gradient_density_network(
    box_size: Tuple[float, float, float] = (50.0, 50.0, 50.0),
    density_gradient: str = "linear",
    num_fibers: int = 200,
    fiber_length: float = 10.0,
    radius: float = 0.1,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """Generate a fiber network with spatially varying density.
    
    Parameters
    ----------
    density_gradient : str
        'linear', 'exponential', 'gaussian', or 'step'.
    """
    rng = np.random.default_rng(seed)
    mat = material or Material(name="gradient_fiber")
    Lx, Ly, Lz = box_size
    
    net = FiberNetwork(
        dimension=3,
        box_size=np.array(box_size),
        metadata={"generator": "gradient_density_network", "gradient": density_gradient},
    )
    
    for i in range(num_fibers):
        x = rng.uniform(0, Lx)
        
        if density_gradient == "linear":
            prob = x / Lx
        elif density_gradient == "exponential":
            prob = np.exp(x / Lx) / np.e
        elif density_gradient == "gaussian":
            prob = np.exp(-((x - Lx / 2) / (Lx / 4))**2)
        elif density_gradient == "step":
            prob = 1.0 if x < Lx / 2 else 0.3
        else:
            prob = 1.0
        
        if rng.uniform() > prob:
            continue
        
        length = fiber_length * (0.8 + 0.4 * rng.uniform())
        y = rng.uniform(0, Ly)
        z = rng.uniform(0, Lz)
        center = np.array([x, y, z])
        
        theta = rng.uniform(0, 2 * np.pi)
        cos_phi = rng.uniform(-1, 1)
        sin_phi = np.sqrt(1 - cos_phi**2)
        direction = np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
        
        start = center - 0.5 * length * direction
        end = center + 0.5 * length * direction
        
        net.add_fiber(Fiber.straight(start, end, radius=radius, material=mat, fiber_id=net.num_fibers))
    
    net.auto_crosslink(threshold=2.5 * radius)
    return net


def core_shell_fiber(
    core_radius: float = 0.5,
    shell_thickness: float = 0.3,
    length: float = 50.0,
    material_core: Optional[Material] = None,
    material_shell: Optional[Material] = None,
    num_shell_fibers: int = 12,
    num_points: int = 100,
) -> FiberNetwork:
    """Generate a core-shell fiber arrangement.
    
    One central fiber surrounded by shell fibers arranged around it.
    """
    mat_core = material_core or Material(name="core")
    mat_shell = material_shell or Material(name="shell")
    
    net = FiberNetwork(
        dimension=3,
        box_size=np.array([2 * (core_radius + shell_thickness), 2 * (core_radius + shell_thickness), length]),
        metadata={"generator": "core_shell_fiber"},
    )
    
    z_vals = np.linspace(0, length, num_points)
    core_points = np.column_stack([np.zeros(num_points), np.zeros(num_points), z_vals])
    net.add_fiber(Fiber(centerline=core_points, radius=core_radius, material=mat_core, fiber_id=0))
    
    shell_radius = core_radius + shell_thickness
    for i in range(num_shell_fibers):
        angle = 2 * np.pi * i / num_shell_fibers
        x = shell_radius * np.cos(angle)
        y = shell_radius * np.sin(angle)
        shell_points = np.column_stack([np.full(num_points, x), np.full(num_points, y), z_vals])
        net.add_fiber(Fiber(
            centerline=shell_points, radius=shell_thickness / 2,
            material=mat_shell, fiber_id=i + 1,
        ))
    
    return net


def fractal_network(
    iterations: int = 3,
    initial_length: float = 50.0,
    branch_factor: int = 3,
    branch_angle: float = np.pi / 4,
    radius: float = 0.5,
    radius_decay: float = 0.7,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """Generate a fractal-like branching fiber network (dendritic).
    
    Parameters
    ----------
    iterations : int
        Number of branching iterations.
    branch_factor : int
        Number of branches per split.
    branch_angle : float
        Angle of branch divergence.
    radius_decay : float
        Radius reduction factor per iteration.
    """
    rng = np.random.default_rng(seed)
    mat = material or Material(name="fractal_fiber")
    
    net = FiberNetwork(
        dimension=3,
        metadata={"generator": "fractal_network", "iterations": iterations},
    )
    
    def _branch(start, direction, length, rad, depth):
        if depth >= iterations:
            return
        
        end = start + length * direction
        net.add_fiber(Fiber.straight(start, end, radius=rad, material=mat, fiber_id=net.num_fibers))
        
        child_length = length * 0.7
        child_radius = rad * radius_decay
        
        for b in range(branch_factor):
            angle = branch_angle * (b - (branch_factor - 1) / 2) / max(1, (branch_factor - 1) / 2)
            
            if abs(direction[0]) < 0.9:
                perp = np.cross(direction, [1, 0, 0])
            else:
                perp = np.cross(direction, [0, 1, 0])
            perp /= np.linalg.norm(perp)
            perp2 = np.cross(direction, perp)
            
            rot_angle = rng.uniform(0, 2 * np.pi)
            bend = np.cos(angle) * direction + np.sin(angle) * (np.cos(rot_angle) * perp + np.sin(rot_angle) * perp2)
            bend /= np.linalg.norm(bend)
            
            _branch(end, bend, child_length, child_radius, depth + 1)
    
    _branch(
        start=np.array([0.0, 0.0, 0.0]),
        direction=np.array([0.0, 0.0, 1.0]),
        length=initial_length,
        rad=radius,
        depth=0,
    )
    
    bb_min, bb_max = net.bounding_box()
    net.box_size = bb_max - bb_min + 1e-6
    return net
