"""
Specialized Fiber Network Generators

Provides generators for specific applications:
- Carbon nanotube (CNT) networks
- Paper/cellulose fiber networks
- Textile weave structures
- Fiber-reinforced composites
- Electrospun nanofiber mats
"""

import numpy as np
from typing import Optional, Tuple, List
from fibernet.core.network import FiberNetwork
from fibernet.core.fiber import Fiber
from fibernet.core.material import Material


def cnt_network_2d(
    num_tubes: int = 100,
    tube_length: float = 5.0,
    box_size: Tuple[float, float] = (50, 50),
    diameter: float = 0.01,
    bundle_size: int = 1,
    orientation_bias: float = 0.0,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """Generate 2D carbon nanotube network.
    
    Parameters
    ----------
    num_tubes : int
        Number of CNTs.
    tube_length : float
        CNT length.
    box_size : tuple
        Box dimensions (Lx, Ly).
    diameter : float
        CNT diameter (typically 1-10 nm).
    bundle_size : int
        Number of tubes per bundle (1 = individual tubes).
    orientation_bias : float
        Orientation bias (0 = random, 1 = fully aligned).
    seed : int, optional
        Random seed.
    
    Returns
    -------
    FiberNetwork
        CNT network.
    """
    rng = np.random.RandomState(seed)
    
    # CNT material properties
    cnt = Material(
        name="CNT",
        youngs_modulus=1e12,  # 1 TPa
        density=2200,  # kg/m³
        poissons_ratio=0.2,
    )
    
    net = FiberNetwork(dimension=2)
    
    for i in range(num_tubes):
        # Random position
        x = rng.uniform(0, box_size[0])
        y = rng.uniform(0, box_size[1])
        
        # Orientation with optional bias
        if orientation_bias > 0:
            theta = rng.normal(0, np.pi * (1 - orientation_bias))
        else:
            theta = rng.uniform(0, 2 * np.pi)
        
        # Create tube
        dx = tube_length * np.cos(theta) / 2
        dy = tube_length * np.sin(theta) / 2
        
        start = np.array([x - dx, y - dy, 0.0])
        end = np.array([x + dx, y + dy, 0.0])
        
        # Add individual tube or bundle
        for j in range(bundle_size):
            offset = rng.normal(0, diameter * 2, size=2)
            s = start.copy()
            e = end.copy()
            s[:2] += offset
            e[:2] += offset
            
            tube = Fiber(
                centerline=np.array([s, e]),
                radius=diameter / 2,
                material=cnt,
                fiber_id=len(net.fibers),
            )
            net.add_fiber(tube)
    
    net.auto_crosslink(threshold=diameter * 3)
    return net


def cnt_network_3d(
    num_tubes: int = 100,
    tube_length: float = 5.0,
    box_size: Tuple[float, float, float] = (50, 50, 50),
    diameter: float = 0.01,
    bundle_size: int = 1,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """Generate 3D carbon nanotube network.
    
    Parameters
    ----------
    num_tubes : int
        Number of CNTs.
    tube_length : float
        CNT length.
    box_size : tuple
        Box dimensions (Lx, Ly, Lz).
    diameter : float
        CNT diameter.
    bundle_size : int
        Number of tubes per bundle.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    FiberNetwork
        3D CNT network.
    """
    rng = np.random.RandomState(seed)
    
    cnt = Material(
        name="CNT",
        youngs_modulus=1e12,
        density=2200,
        poissons_ratio=0.2,
    )
    
    net = FiberNetwork(dimension=3)
    
    for i in range(num_tubes):
        # Random position in 3D
        pos = rng.uniform([0, 0, 0], box_size)
        
        # Random 3D orientation
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0, np.pi)
        
        direction = np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi),
        ])
        
        half_len = tube_length / 2
        start = pos - half_len * direction
        end = pos + half_len * direction
        
        for j in range(bundle_size):
            offset = rng.normal(0, diameter * 2, size=3)
            s = start + offset
            e = end + offset
            
            tube = Fiber(
                centerline=np.array([s, e]),
                radius=diameter / 2,
                material=cnt,
                fiber_id=len(net.fibers),
            )
            net.add_fiber(tube)
    
    net.auto_crosslink(threshold=diameter * 3)
    return net


def paper_network(
    num_fibers: int = 200,
    fiber_length: float = 10.0,
    box_size: Tuple[float, float] = (50, 50),
    fiber_width: float = 0.05,
    fiber_thickness: float = 0.01,
    curliness: float = 0.3,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """Generate paper/cellulose fiber network.
    
    Parameters
    ----------
    num_fibers : int
        Number of cellulose fibers.
    fiber_length : float
        Fiber length (typically 1-5 mm).
    box_size : tuple
        Box dimensions.
    fiber_width : float
        Fiber width (typically 20-50 µm).
    fiber_thickness : float
        Fiber thickness (typically 5-10 µm).
    curliness : float
        Curl factor (0 = straight, 1 = very curly).
    seed : int, optional
        Random seed.
    
    Returns
    -------
    FiberNetwork
        Paper fiber network.
    """
    rng = np.random.RandomState(seed)
    
    cellulose = Material(
        name="Cellulose",
        youngs_modulus=10e9,  # 10 GPa
        density=1500,  # kg/m³
        poissons_ratio=0.3,
    )
    
    net = FiberNetwork(dimension=2)
    
    for i in range(num_fibers):
        # Random position and orientation
        x = rng.uniform(0, box_size[0])
        y = rng.uniform(0, box_size[1])
        theta = rng.uniform(0, 2 * np.pi)
        
        # Generate curved fiber path
        num_segments = max(3, int(fiber_length / 1.0))
        pts = []
        
        current_pos = np.array([x, y, 0.0])
        current_dir = np.array([np.cos(theta), np.sin(theta), 0.0])
        
        pts.append(current_pos.copy())
        
        for j in range(num_segments - 1):
            # Add some curl
            if curliness > 0:
                curl_angle = rng.normal(0, np.pi * curliness / 2)
                cos_c = np.cos(curl_angle)
                sin_c = np.sin(curl_angle)
                new_dir = np.array([
                    current_dir[0] * cos_c - current_dir[1] * sin_c,
                    current_dir[0] * sin_c + current_dir[1] * cos_c,
                    0.0,
                ])
            else:
                new_dir = current_dir
            
            step = fiber_length / num_segments
            current_pos = current_pos + step * new_dir
            pts.append(current_pos.copy())
            current_dir = new_dir
        
        pts = np.array(pts)
        
        fiber = Fiber(
            centerline=pts,
            radius=fiber_width / 2,
            material=cellulose,
            fiber_id=len(net.fibers),
        )
        net.add_fiber(fiber)
    
    net.auto_crosslink(threshold=fiber_width * 2)
    return net


def textile_weave(
    warp_count: int = 10,
    weft_count: int = 10,
    spacing: float = 2.0,
    fiber_diameter: float = 0.1,
    weave_pattern: str = "plain",
    crimp_amplitude: float = 0.5,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """Generate textile weave structure.
    
    Parameters
    ----------
    warp_count : int
        Number of warp (lengthwise) fibers.
    weft_count : int
        Number of weft (crosswise) fibers.
    spacing : float
        Spacing between fibers.
    fiber_diameter : float
        Fiber diameter.
    weave_pattern : str
        Weave pattern: "plain", "twill", "satin".
    crimp_amplitude : float
        Amplitude of fiber crimp (waviness).
    seed : int, optional
        Random seed.
    
    Returns
    -------
    FiberNetwork
        Textile weave network.
    """
    rng = np.random.RandomState(seed)
    
    textile = Material(
        name="Textile",
        youngs_modulus=5e9,  # 5 GPa (typical for cotton/polyester)
        density=1400,
        poissons_ratio=0.35,
    )
    
    net = FiberNetwork(dimension=2)
    
    # Warp fibers (vertical)
    for i in range(warp_count):
        x = i * spacing
        num_pts = 20
        
        pts = []
        for j in range(num_pts):
            y = j * spacing * weft_count / num_pts
            
            # Add crimp based on weave pattern
            if weave_pattern == "plain":
                # Alternate up/down every crossing
                crimp = crimp_amplitude * np.sin(np.pi * y / spacing)
            elif weave_pattern == "twill":
                # Diagonal pattern
                crimp = crimp_amplitude * np.sin(np.pi * y / (2 * spacing))
            else:  # satin
                crimp = crimp_amplitude * 0.5 * np.sin(np.pi * y / spacing)
            
            pts.append([x, y, crimp])
        
        pts = np.array(pts)
        fiber = Fiber(
            centerline=pts,
            radius=fiber_diameter / 2,
            material=textile,
            fiber_id=len(net.fibers),
        )
        net.add_fiber(fiber)
    
    # Weft fibers (horizontal)
    for i in range(weft_count):
        y = i * spacing
        num_pts = 20
        
        pts = []
        for j in range(num_pts):
            x = j * spacing * warp_count / num_pts
            
            # Crimp for weft
            if weave_pattern == "plain":
                crimp = -crimp_amplitude * np.sin(np.pi * x / spacing)
            elif weave_pattern == "twill":
                crimp = -crimp_amplitude * np.sin(np.pi * x / (2 * spacing))
            else:
                crimp = -crimp_amplitude * 0.5 * np.sin(np.pi * x / spacing)
            
            pts.append([x, y, crimp])
        
        pts = np.array(pts)
        fiber = Fiber(
            centerline=pts,
            radius=fiber_diameter / 2,
            material=textile,
            fiber_id=len(net.fibers),
        )
        net.add_fiber(fiber)
    
    return net


def electrospun_mat(
    num_fibers: int = 500,
    fiber_diameter: float = 0.001,
    box_size: Tuple[float, float] = (50, 50),
    fiber_length_mean: float = 20.0,
    fiber_length_std: float = 5.0,
    alignment: float = 0.0,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """Generate electrospun nanofiber mat.
    
    Parameters
    ----------
    num_fibers : int
        Number of electrospun fibers.
    fiber_diameter : float
        Fiber diameter (typically 100 nm - 1 µm).
    box_size : tuple
        Mat dimensions.
    fiber_length_mean : float
        Mean fiber length.
    fiber_length_std : float
        Standard deviation of fiber length.
    alignment : float
        Alignment degree (0 = random, 1 = fully aligned).
    seed : int, optional
        Random seed.
    
    Returns
    -------
    FiberNetwork
        Electrospun fiber mat.
    """
    rng = np.random.RandomState(seed)
    
    polymer = Material(
        name="Polymer",
        youngs_modulus=2e9,  # 2 GPa (typical for PCL/PLA)
        density=1200,
        poissons_ratio=0.35,
    )
    
    net = FiberNetwork(dimension=2)
    
    for i in range(num_fibers):
        # Random position
        x = rng.uniform(0, box_size[0])
        y = rng.uniform(0, box_size[1])
        
        # Orientation with alignment
        if alignment > 0:
            theta = rng.normal(0, np.pi * (1 - alignment) / 2)
        else:
            theta = rng.uniform(0, 2 * np.pi)
        
        # Random length
        length = max(1.0, rng.normal(fiber_length_mean, fiber_length_std))
        
        # Create slightly curved fiber
        num_pts = max(5, int(length / 2.0))
        pts = []
        
        start = np.array([x, y, 0.0])
        direction = np.array([np.cos(theta), np.sin(theta), 0.0])
        
        for j in range(num_pts):
            t = j / (num_pts - 1)
            pos = start + t * length * direction
            
            # Add small random curvature
            if j > 0 and j < num_pts - 1:
                curl = rng.normal(0, 0.1)
                perp = np.array([-direction[1], direction[0], 0.0])
                pos = pos + curl * perp
            
            pts.append(pos)
        
        pts = np.array(pts)
        
        fiber = Fiber(
            centerline=pts,
            radius=fiber_diameter / 2,
            material=polymer,
            fiber_id=len(net.fibers),
        )
        net.add_fiber(fiber)
    
    net.auto_crosslink(threshold=fiber_diameter * 3)
    return net


def fiber_reinforced_composite(
    matrix_size: Tuple[float, float, float] = (50, 50, 10),
    fiber_volume_fraction: float = 0.6,
    fiber_diameter: float = 0.01,
    fiber_length: float = None,
    fiber_orientation: str = "unidirectional",
    seed: Optional[int] = None,
) -> FiberNetwork:
    """Generate fiber-reinforced composite structure.
    
    Parameters
    ----------
    matrix_size : tuple
        Composite dimensions (Lx, Ly, Lz).
    fiber_volume_fraction : float
        Volume fraction of fibers (typically 0.3-0.7).
    fiber_diameter : float
        Fiber diameter.
    fiber_length : float, optional
        Fiber length (None = continuous fibers).
    fiber_orientation : str
        Fiber orientation: "unidirectional", "random", "woven".
    seed : int, optional
        Random seed.
    
    Returns
    -------
    FiberNetwork
        Composite fiber network.
    """
    rng = np.random.RandomState(seed)
    
    # Fiber material (carbon or glass)
    fiber_mat = Material(
        name="Carbon_Fiber",
        youngs_modulus=230e9,  # 230 GPa
        density=1800,
        poissons_ratio=0.2,
    )
    
    # Calculate number of fibers needed
    Lx, Ly, Lz = matrix_size
    
    if fiber_length is None:
        # Continuous fibers
        fiber_length = Lx if fiber_orientation == "unidirectional" else max(Lx, Ly, Lz)
    
    fiber_area = np.pi * (fiber_diameter / 2)**2
    total_volume = Lx * Ly * Lz
    fiber_volume = total_volume * fiber_volume_fraction
    
    if fiber_orientation == "unidirectional":
        num_fibers = int(fiber_volume / (fiber_area * fiber_length))
    else:
        num_fibers = int(fiber_volume / (fiber_area * fiber_length))
    
    net = FiberNetwork(dimension=3)
    
    for i in range(num_fibers):
        if fiber_orientation == "unidirectional":
            # All fibers along x-axis
            y = rng.uniform(0, Ly)
            z = rng.uniform(0, Lz)
            start = np.array([0, y, z])
            end = np.array([Lx, y, z])
        
        elif fiber_orientation == "random":
            # Random 3D orientation
            pos = rng.uniform([0, 0, 0], matrix_size)
            theta = rng.uniform(0, 2 * np.pi)
            phi = rng.uniform(0, np.pi)
            
            direction = np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi),
            ])
            
            half_len = fiber_length / 2
            start = pos - half_len * direction
            end = pos + half_len * direction
        
        else:  # woven
            # Alternate between x and y directions
            if i % 2 == 0:
                y = (i // 2) * (Ly / (num_fibers // 2))
                z = Lz / 2
                start = np.array([0, y, z])
                end = np.array([Lx, y, z])
            else:
                x = (i // 2) * (Lx / (num_fibers // 2))
                z = Lz / 2
                start = np.array([x, 0, z])
                end = np.array([x, Ly, z])
        
        fiber = Fiber(
            centerline=np.array([start, end]),
            radius=fiber_diameter / 2,
            material=fiber_mat,
            fiber_id=len(net.fibers),
        )
        net.add_fiber(fiber)
    
    return net
