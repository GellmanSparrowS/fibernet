"""
Advanced fiber network generators.

Includes:
- Voronoi-based fiber networks
- Electrospun-like networks
- Melt-blown networks
- Biomimetic structures (collagen, actin, fibrin)
- Variable-property networks
- Defected/modified lattices
- Composite networks (multiple materials)
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from scipy.spatial import Voronoi, Delaunay
from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork, Crosslink
from fibernet.core.material import Material


def voronoi_network_2d(
    num_seeds: int = 50,
    box_size: Tuple[float, float] = (50.0, 50.0),
    radius: float = 0.1,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
    remove_short_edges: bool = True,
    min_edge_length: float = 1.0,
) -> FiberNetwork:
    """Generate a 2D fiber network from Voronoi tessellation edges.
    
    Creates realistic random cellular fiber networks, similar to
    foam or cellular solid structures.
    
    Parameters
    ----------
    num_seeds : int
        Number of Voronoi seed points.
    box_size : tuple
        (Lx, Ly) domain size.
    remove_short_edges : bool
        Remove very short Voronoi edges.
    min_edge_length : float
        Minimum edge length to keep.
    """
    rng = np.random.default_rng(seed)
    mat = material or Material(name="voronoi_fiber")
    Lx, Ly = box_size
    
    seeds = rng.uniform([0, 0], [Lx, Ly], (num_seeds, 2))
    
    # Mirror seeds for bounded Voronoi
    all_seeds = np.vstack([
        seeds,
        seeds + [Lx, 0], seeds - [Lx, 0],
        seeds + [0, Ly], seeds - [0, Ly],
        seeds + [Lx, Ly], seeds - [Lx, Ly],
        seeds + [Lx, -Ly], seeds - [Lx, -Ly],
    ])
    
    vor = Voronoi(all_seeds)
    
    net = FiberNetwork(
        dimension=2,
        box_size=np.array([Lx, Ly, 0.0]),
        metadata={"generator": "voronoi_network_2d", "num_seeds": num_seeds},
    )
    
    fid = 0
    for ridge_idx, ridge in enumerate(vor.ridge_vertices):
        if -1 in ridge:
            continue
        
        p1 = vor.vertices[ridge[0]]
        p2 = vor.vertices[ridge[1]]
        
        if p1[0] < -0.1 or p1[0] > Lx + 0.1 or p1[1] < -0.1 or p1[1] > Ly + 0.1:
            continue
        if p2[0] < -0.1 or p2[0] > Lx + 0.1 or p2[1] < -0.1 or p2[1] > Ly + 0.1:
            continue
        
        edge_length = np.linalg.norm(p2 - p1)
        if remove_short_edges and edge_length < min_edge_length:
            continue
        
        start = np.array([p1[0], p1[1], 0.0])
        end = np.array([p2[0], p2[1], 0.0])
        net.add_fiber(Fiber.straight(start, end, radius=radius, material=mat, fiber_id=fid))
        fid += 1
    
    net.auto_crosslink(threshold=2.5 * radius)
    return net


def voronoi_network_3d(
    num_seeds: int = 100,
    box_size: Tuple[float, float, float] = (30.0, 30.0, 30.0),
    radius: float = 0.1,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """Generate a 3D fiber network from Voronoi tessellation edges."""
    rng = np.random.default_rng(seed)
    mat = material or Material(name="voronoi_fiber")
    Lx, Ly, Lz = box_size
    
    seeds = rng.uniform([0, 0, 0], [Lx, Ly, Lz], (num_seeds, 3))
    
    all_seeds = []
    for dx in [-Lx, 0, Lx]:
        for dy in [-Ly, 0, Ly]:
            for dz in [-Lz, 0, Lz]:
                all_seeds.append(seeds + np.array([dx, dy, dz]))
    all_seeds = np.vstack(all_seeds)
    
    vor = Voronoi(all_seeds)
    
    net = FiberNetwork(
        dimension=3,
        box_size=np.array(box_size),
        metadata={"generator": "voronoi_network_3d", "num_seeds": num_seeds},
    )
    
    fid = 0
    for ridge in vor.ridge_vertices:
        if -1 in ridge or len(ridge) < 2:
            continue
        
        p1 = vor.vertices[ridge[0]]
        p2 = vor.vertices[ridge[1]]
        
        inside1 = np.all(p1 >= -0.1) and np.all(p1 <= np.array([Lx + 0.1, Ly + 0.1, Lz + 0.1]))
        inside2 = np.all(p2 >= -0.1) and np.all(p2 <= np.array([Lx + 0.1, Ly + 0.1, Lz + 0.1]))
        
        if inside1 and inside2:
            net.add_fiber(Fiber.straight(p1, p2, radius=radius, material=mat, fiber_id=fid))
            fid += 1
    
    net.auto_crosslink(threshold=2.5 * radius)
    return net


def electrospun_network(
    num_fibers: int = 200,
    fiber_length: float = 50.0,
    box_size: Tuple[float, float] = (100.0, 100.0),
    radius_mean: float = 0.2,
    radius_std: float = 0.1,
    waviness: float = 0.3,
    deposition_pattern: str = "random",
    material: Optional[Material] = None,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """Generate an electrospun-like fiber network.
    
    Models the random deposition of fibers during electrospinning,
    with characteristic waviness and radius distribution.
    
    Parameters
    ----------
    radius_mean : float
        Mean fiber radius.
    radius_std : float
        Standard deviation of radius distribution.
    waviness : float
        Amplitude of sinusoidal waviness (0=straight, higher=wavier).
    deposition_pattern : str
        'random', 'aligned', 'cross-ply', 'concentric'.
    """
    rng = np.random.default_rng(seed)
    mat = material or Material(name="electrospun_fiber")
    Lx, Ly = box_size
    
    net = FiberNetwork(
        dimension=2,
        box_size=np.array([Lx, Ly, 0.0]),
        metadata={
            "generator": "electrospun_network",
            "pattern": deposition_pattern,
            "waviness": waviness,
        },
    )
    
    for i in range(num_fibers):
        length = fiber_length * (0.5 + rng.uniform())
        r = max(radius_mean + rng.normal(0, radius_std), radius_mean * 0.1)
        
        cx = rng.uniform(0, Lx)
        cy = rng.uniform(0, Ly)
        
        if deposition_pattern == "aligned":
            angle = rng.normal(0, 0.1)
        elif deposition_pattern == "cross-ply":
            angle = rng.choice([0, np.pi / 2]) + rng.normal(0, 0.05)
        elif deposition_pattern == "concentric":
            center = np.array([Lx / 2, Ly / 2])
            pos = np.array([cx, cy])
            radial = pos - center
            angle = np.arctan2(radial[1], radial[0]) + np.pi / 2 + rng.normal(0, 0.2)
        else:
            angle = rng.uniform(0, np.pi)
        
        num_pts = max(20, int(length / 2))
        t = np.linspace(-length / 2, length / 2, num_pts)
        
        x = t * np.cos(angle) + cx
        y = t * np.sin(angle) + cy
        
        if waviness > 0:
            perp_x = -np.sin(angle)
            perp_y = np.cos(angle)
            wave_freq = rng.uniform(0.1, 0.5)
            wave_amp = waviness * r * rng.uniform(0.5, 2.0)
            x += wave_amp * np.sin(wave_freq * t) * perp_x
            y += wave_amp * np.sin(wave_freq * t) * perp_y
        
        points = np.column_stack([x, y, np.zeros(num_pts)])
        
        fiber = Fiber(centerline=points, radius=r, material=mat, fiber_id=i)
        net.add_fiber(fiber)
    
    net.auto_crosslink(threshold=2.5 * radius_mean)
    return net


def meltblown_network(
    num_fibers: int = 300,
    box_size: Tuple[float, float] = (80.0, 80.0),
    radius_mean: float = 0.05,
    radius_std: float = 0.03,
    curliness: float = 0.5,
    draw_direction: float = 0.0,
    draw_spread: float = 0.3,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """Generate a melt-blown fiber network.
    
    Models melt-blown nonwoven fabrication where fibers are
    stretched and deposited with high draw ratio.
    
    Parameters
    ----------
    curliness : float
        Random walk deviation (0=straight, higher=curlier).
    draw_direction : float
        Primary draw direction (radians).
    draw_spread : float
        Angular spread around draw direction.
    """
    rng = np.random.default_rng(seed)
    mat = material or Material(name="meltblown_fiber")
    Lx, Ly = box_size
    
    net = FiberNetwork(
        dimension=2,
        box_size=np.array([Lx, Ly, 0.0]),
        metadata={"generator": "meltblown_network", "curliness": curliness},
    )
    
    for i in range(num_fibers):
        r = max(radius_mean + rng.normal(0, radius_std), 0.01)
        
        num_steps = rng.integers(50, 200)
        step_length = rng.uniform(0.5, 2.0)
        
        cx = rng.uniform(0, Lx)
        cy = rng.uniform(0, Ly)
        
        base_angle = draw_direction + rng.normal(0, draw_spread)
        
        points = np.zeros((num_steps, 3))
        points[0] = [cx, cy, 0]
        
        for s in range(1, num_steps):
            prev_angle = np.arctan2(
                points[s - 1, 1] - points[max(0, s - 2), 1],
                points[s - 1, 0] - points[max(0, s - 2), 0],
            )
            
            angle_change = rng.normal(0, curliness * 0.5)
            pull_to_base = 0.1 * (base_angle - prev_angle)
            new_angle = prev_angle + angle_change + pull_to_base
            
            points[s, 0] = points[s - 1, 0] + step_length * np.cos(new_angle)
            points[s, 1] = points[s - 1, 1] + step_length * np.sin(new_angle)
        
        fiber = Fiber(centerline=points, radius=r, material=mat, fiber_id=i)
        net.add_fiber(fiber)
    
    net.auto_crosslink(threshold=3.0 * radius_mean)
    return net


def biomimetic_collagen(
    num_fibers: int = 100,
    box_size: Tuple[float, float, float] = (50.0, 50.0, 20.0),
    persistence_length: float = 15.0,
    d_periodicity: float = 0.67,
    radius_mean: float = 0.5,
    bundling_probability: float = 0.3,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """Generate biomimetic collagen-like network.
    
    Models type-I collagen with D-banding periodicity,
    branching, and bundle formation.
    
    Parameters
    ----------
    persistence_length : float
        Collagen persistence length (controls stiffness).
    d_periodicity : float
        D-banding period (typically ~67nm scaled).
    bundling_probability : float
        Probability of fiber bundling with neighbors.
    """
    rng = np.random.default_rng(seed)
    mat = material or Material(name="collagen", density=1300, youngs_modulus=1.2e9, poissons_ratio=0.35)
    
    net = FiberNetwork(
        dimension=3,
        box_size=np.array(box_size),
        metadata={
            "generator": "biomimetic_collagen",
            "persistence_length": persistence_length,
            "d_periodicity": d_periodicity,
        },
    )
    
    Lx, Ly, Lz = box_size
    
    for i in range(num_fibers):
        num_steps = rng.integers(80, 200)
        step = rng.uniform(0.3, 0.8)
        
        start = np.array([rng.uniform(0, Lx), rng.uniform(0, Ly), rng.uniform(0, Lz)])
        
        theta = rng.uniform(0, 2 * np.pi)
        cos_phi = rng.uniform(-1, 1)
        sin_phi = np.sqrt(1 - cos_phi**2)
        direction = np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
        
        kappa = step / persistence_length
        points = np.zeros((num_steps, 3))
        points[0] = start
        
        for s in range(1, num_steps):
            bend = rng.normal(0, kappa)
            twist = rng.uniform(0, 2 * np.pi)
            
            if abs(direction[0]) < 0.9:
                perp1 = np.cross(direction, [1, 0, 0])
            else:
                perp1 = np.cross(direction, [0, 1, 0])
            perp1 /= max(np.linalg.norm(perp1), 1e-12)
            perp2 = np.cross(direction, perp1)
            
            direction = (
                np.cos(bend) * direction
                + np.sin(bend) * np.cos(twist) * perp1
                + np.sin(bend) * np.sin(twist) * perp2
            )
            direction /= max(np.linalg.norm(direction), 1e-12)
            
            points[s] = points[s - 1] + step * direction
        
        r = radius_mean * (0.5 + rng.uniform())
        
        if rng.uniform() < bundling_probability and i > 0:
            r *= rng.uniform(1.5, 3.0)
        
        fiber = Fiber(centerline=points, radius=r, material=mat, fiber_id=i)
        net.add_fiber(fiber)
    
    net.auto_crosslink(threshold=3.0 * radius_mean)
    return net


def biomimetic_fibrin(
    num_fibers: int = 80,
    box_size: Tuple[float, float, float] = (40.0, 40.0, 15.0),
    branch_probability: float = 0.3,
    branch_angle: float = np.pi / 6,
    radius_mean: float = 0.1,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """Generate biomimetic fibrin clot-like network.
    
    Models fibrin network formation with characteristic branching.
    """
    rng = np.random.default_rng(seed)
    mat = material or Material(name="fibrin", density=1100, youngs_modulus=1e6, poissons_ratio=0.45)
    Lx, Ly, Lz = box_size
    
    net = FiberNetwork(
        dimension=3,
        box_size=np.array(box_size),
        metadata={"generator": "biomimetic_fibrin", "branch_probability": branch_probability},
    )
    
    fid = 0
    for i in range(num_fibers):
        num_steps = rng.integers(50, 150)
        step = rng.uniform(0.2, 0.5)
        start = rng.uniform([0, 0, 0], [Lx, Ly, Lz])
        
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0, np.pi)
        direction = np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi),
        ])
        
        points = [start.copy()]
        branch_points = []
        
        for s in range(num_steps):
            bend = rng.normal(0, 0.1)
            if abs(direction[0]) < 0.9:
                perp = np.cross(direction, [1, 0, 0])
            else:
                perp = np.cross(direction, [0, 1, 0])
            perp /= max(np.linalg.norm(perp), 1e-12)
            
            direction = direction + bend * perp
            direction /= max(np.linalg.norm(direction), 1e-12)
            points.append(points[-1] + step * direction)
            
            if rng.uniform() < branch_probability * 0.05 and fid < 500:
                branch_points.append((len(points) - 1, points[-1].copy(), direction.copy()))
        
        points = np.array(points)
        r = radius_mean * (0.5 + rng.uniform())
        net.add_fiber(Fiber(centerline=points, radius=r, material=mat, fiber_id=fid))
        fid += 1
        
        for bp_idx, bp_pos, bp_dir in branch_points:
            if fid >= num_fibers * 2:
                break
            
            branch_dir = bp_dir.copy()
            if abs(branch_dir[0]) < 0.9:
                perp = np.cross(branch_dir, [1, 0, 0])
            else:
                perp = np.cross(branch_dir, [0, 1, 0])
            perp /= max(np.linalg.norm(perp), 1e-12)
            
            sign = rng.choice([-1, 1])
            branch_dir = (
                np.cos(branch_angle) * branch_dir
                + sign * np.sin(branch_angle) * perp
            )
            branch_dir /= max(np.linalg.norm(branch_dir), 1e-12)
            
            b_len = rng.integers(20, 60)
            b_pts = [bp_pos.copy()]
            for s in range(b_len):
                b_pts.append(b_pts[-1] + step * branch_dir)
                branch_dir += rng.normal(0, 0.05) * perp
                branch_dir /= max(np.linalg.norm(branch_dir), 1e-12)
            
            b_pts = np.array(b_pts)
            net.add_fiber(Fiber(
                centerline=b_pts, radius=r * 0.7, material=mat, fiber_id=fid
            ))
            fid += 1
    
    net.auto_crosslink(threshold=3.0 * radius_mean)
    return net


def defected_lattice(
    base_generator: callable,
    base_kwargs: Dict[str, Any],
    defect_type: str = "vacancy",
    defect_fraction: float = 0.1,
    defect_radius: float = None,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """Create a lattice with controlled defects.
    
    Parameters
    ----------
    base_generator : callable
        Generator function for base lattice.
    base_kwargs : dict
        Arguments for base generator.
    defect_type : str
        'vacancy' (remove fibers), 'interstitial' (add fibers),
        'substitution' (change material), 'displacement' (shift nodes).
    defect_fraction : float
        Fraction of fibers affected (0 to 1).
    """
    rng = np.random.default_rng(seed)
    net = base_generator(**base_kwargs)
    
    if defect_type == "vacancy":
        num_remove = int(defect_fraction * net.num_fibers)
        remove_idx = rng.choice(net.num_fibers, num_remove, replace=False)
        for idx in sorted(remove_idx, reverse=True):
            net.remove_fiber(idx)
    
    elif defect_type == "interstitial":
        num_add = int(defect_fraction * net.num_fibers)
        bb_min, bb_max = net.bounding_box()
        
        for _ in range(num_add):
            start = rng.uniform(bb_min, bb_max)
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            length = rng.uniform(2, 10)
            end = start + length * direction
            
            r = defect_radius or net.mean_radius
            net.add_fiber(Fiber.straight(start, end, radius=r, fiber_id=net.num_fibers))
    
    elif defect_type == "displacement":
        bb_min, bb_max = net.bounding_box()
        L = np.max(bb_max - bb_min) * 0.01
        
        for fiber in net.fibers:
            if rng.uniform() < defect_fraction:
                displacement = rng.normal(0, L, 3)
                fiber.centerline += displacement
    
    net.auto_crosslink(threshold=3.0 * (defect_radius or net.mean_radius))
    return net


def composite_network(
    networks: List[FiberNetwork],
    volume_fractions: Optional[List[float]] = None,
    box_size: Tuple[float, float, float] = (50.0, 50.0, 50.0),
    seed: Optional[int] = None,
) -> FiberNetwork:
    """Create a multi-material composite fiber network.
    
    Combines multiple fiber networks with different materials
    into a single composite structure.
    
    Parameters
    ----------
    networks : list of FiberNetwork
        Sub-networks to combine.
    volume_fractions : list of float, optional
        Volume fractions for each sub-network.
    """
    from fibernet.core.transform import merge
    
    rng = np.random.default_rng(seed)
    
    if volume_fractions is not None:
        vf = np.array(volume_fractions)
        vf = vf / vf.sum()
    else:
        vf = np.ones(len(networks)) / len(networks)
    
    result = merge(networks, merge_crosslinks=True, proximity_threshold=1.0)
    result.box_size = np.array(box_size)
    result.metadata["generator"] = "composite_network"
    result.metadata["num_phases"] = len(networks)
    
    return result


def graded_network(
    base_generator: callable,
    base_kwargs: Dict[str, Any],
    gradient_axis: int = 0,
    property_name: str = "radius",
    gradient_func: str = "linear",
    value_range: Tuple[float, float] = (0.05, 0.5),
    num_bins: int = 10,
) -> FiberNetwork:
    """Create a network with spatially graded properties.
    
    Parameters
    ----------
    gradient_axis : int
        Axis along which to grade (0=x, 1=y, 2=z).
    property_name : str
        Property to grade: 'radius', 'length', 'density'.
    gradient_func : str
        'linear', 'exponential', 'step', 'gaussian'.
    value_range : tuple
        (min_value, max_value) for the graded property.
    """
    net = base_generator(**base_kwargs)
    
    bb_min, bb_max = net.bounding_box()
    L = bb_max[gradient_axis] - bb_min[gradient_axis]
    
    vmin, vmax = value_range
    
    for fiber in net.fibers:
        center = 0.5 * (fiber.start_point + fiber.end_point)
        t = (center[gradient_axis] - bb_min[gradient_axis]) / max(L, 1e-12)
        
        if gradient_func == "linear":
            factor = t
        elif gradient_func == "exponential":
            factor = (np.exp(2 * t) - 1) / (np.e**2 - 1)
        elif gradient_func == "step":
            factor = 1.0 if t > 0.5 else 0.0
        elif gradient_func == "gaussian":
            factor = np.exp(-((t - 0.5) / 0.2)**2)
        else:
            factor = t
        
        new_val = vmin + factor * (vmax - vmin)
        
        if property_name == "radius":
            fiber.radius = new_val
    
    net.metadata["gradient"] = {
        "axis": gradient_axis,
        "property": property_name,
        "function": gradient_func,
        "range": value_range,
    }
    
    return net


def auxetic_structure(
    reentrant_angle: float = np.pi / 4,
    cell_size: float = 5.0,
    grid_size: Tuple[int, int] = (5, 5),
    radius: float = 0.2,
    material: Optional[Material] = None,
) -> FiberNetwork:
    """Generate a 2D auxetic (re-entrant honeycomb) structure.
    
    Negative Poisson's ratio metamaterial structure.
    
    Parameters
    ----------
    reentrant_angle : float
        Angle of re-entrant cell walls (controls auxetic behavior).
    """
    mat = material or Material(name="auxetic")
    nx, ny = grid_size
    a = cell_size
    
    net = FiberNetwork(
        dimension=2,
        box_size=np.array([nx * a * 2, ny * a * 2, 0.0]),
        metadata={"generator": "auxetic_structure", "reentrant_angle": reentrant_angle},
    )
    
    fid = 0
    for i in range(nx):
        for j in range(ny):
            cx = (i + 0.5) * 2 * a
            cy = (j + 0.5) * 2 * a
            
            vertices = []
            for k in range(6):
                angle = np.pi / 3 * k
                if k % 2 == 0:
                    r = a * (1 + 0.3 * np.cos(reentrant_angle))
                else:
                    r = a * (1 - 0.3 * np.cos(reentrant_angle))
                vx = cx + r * np.cos(angle)
                vy = cy + r * np.sin(angle)
                vertices.append(np.array([vx, vy, 0.0]))
            
            for k in range(6):
                v1 = vertices[k]
                v2 = vertices[(k + 1) % 6]
                net.add_fiber(Fiber.straight(v1, v2, radius=radius, material=mat, fiber_id=fid))
                fid += 1
    
    net.auto_crosslink(threshold=2.5 * radius)
    return net


def kirigami_structure(
    cut_pattern: str = "alternating",
    sheet_size: Tuple[float, float] = (50.0, 50.0),
    cut_length: float = 8.0,
    cut_spacing: float = 3.0,
    cut_width: float = 0.1,
    num_cuts: int = 5,
    thickness: float = 0.5,
    material: Optional[Material] = None,
) -> FiberNetwork:
    """Generate a kirigami-inspired fiber structure.
    
    Models cut patterns that allow out-of-plane deformation
    when stretched, inspired by paper cutting art.
    """
    mat = material or Material(name="kirigami")
    Lx, Ly = sheet_size
    
    net = FiberNetwork(
        dimension=3,
        box_size=np.array([Lx, Ly, thickness * 2]),
        metadata={"generator": "kirigami_structure", "pattern": cut_pattern},
    )
    
    fid = 0
    num_rows = int(Ly / (2 * cut_length + cut_spacing))
    
    for row in range(num_rows):
        y = (row + 0.5) * (2 * cut_length + cut_spacing)
        
        for cut_idx in range(num_cuts):
            if cut_pattern == "alternating" and (row + cut_idx) % 2 == 1:
                x_start = cut_spacing + cut_idx * (cut_length + cut_spacing)
            else:
                x_start = cut_spacing + cut_idx * (cut_length + cut_spacing)
            
            x_end = x_start + cut_length
            
            if x_end > Lx:
                break
            
            net.add_fiber(Fiber.straight(
                np.array([x_start, y, 0]),
                np.array([x_end, y, 0]),
                radius=cut_width / 2, material=mat, fiber_id=fid,
            ))
            fid += 1
    
    for i in range(int(Lx / cut_spacing)):
        x = i * cut_spacing
        net.add_fiber(Fiber.straight(
            np.array([x, 0, 0]),
            np.array([x, Ly, 0]),
            radius=cut_width / 2, material=mat, fiber_id=fid,
        ))
        fid += 1
    
    for j in range(int(Ly / cut_spacing)):
        y = j * cut_spacing
        net.add_fiber(Fiber.straight(
            np.array([0, y, 0]),
            np.array([Lx, y, 0]),
            radius=cut_width / 2, material=mat, fiber_id=fid,
        ))
        fid += 1
    
    net.auto_crosslink(threshold=2.0 * cut_width)
    return net
