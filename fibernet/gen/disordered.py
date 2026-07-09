"""
Disordered fiber network generators.

Generates random/disordered fiber networks with proper geometric
intersection detection for crosslink creation.

Key fix: 3D random networks now use segment intersection detection
to create crosslinks at actual crossing points.

Available generators:
- random_2d: Random straight fibers in 2D
- random_3d: Random straight fibers in 3D (with intersection detection)
- random_walk: Random walk fibers
- oriented_2d: Oriented random fibers in 2D
- oriented_3d: Oriented random fibers in 3D (with intersection detection)
- electrospun: Electrospun-like random deposition
- voronoi_2d: Voronoi tessellation edges
- voronoi_3d: 3D Voronoi tessellation edges
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork, Crosslink
from fibernet.core.material import Material


def _compute_percolation_box_2d(num_fibers, fiber_length, threshold_factor=2.0):
    """Compute box size for 2D percolation.
    
    Percolation threshold for 2D sticks: ρ_c * L² ≈ 5.71
    We target threshold_factor × threshold for reliable connectivity.
    """
    target = 5.71 * threshold_factor
    # ρ * L² > target → A = N * L² / target
    area = num_fibers * fiber_length**2 / target
    box = np.sqrt(area)
    return max(box, fiber_length * 2)


def _compute_percolation_box_3d(num_fibers, fiber_length, threshold_factor=2.0):
    """Compute box size for 3D percolation.
    
    Percolation threshold for 3D sticks: ρ_c * L³ ≈ 2.53
    We target threshold_factor × threshold.
    """
    target = 2.53 * threshold_factor
    volume = num_fibers * fiber_length**3 / target
    box = volume ** (1/3)
    return max(box, fiber_length * 2)


def _count_components(net):
    """Count connected components in a FiberNetwork."""
    from collections import defaultdict
    adj = defaultdict(set)
    for cl in net.crosslinks:
        adj[cl.fiber_i].add(cl.fiber_j)
        adj[cl.fiber_j].add(cl.fiber_i)
    
    visited = set()
    n_comp = 0
    for s in range(net.num_fibers):
        if s not in visited:
            n_comp += 1
            q = [s]
            while q:
                n = q.pop(0)
                if n in visited: continue
                visited.add(n)
                q.extend(adj[n] - visited)
    
    return n_comp


def _ensure_connected(net, max_gap_factor=3.0):
    """Ensure network is connected, bridging components if needed."""
    n_comp = _count_components(net)
    if n_comp > 1:
        max_gap = max_gap_factor * net.mean_fiber_length if net.mean_fiber_length > 0 else 50.0
        net.connect_components(max_gap=max_gap)
    return net



def _segment_min_distance_3d(p1, p2, p3, p4):
    """Minimum distance between two 3D line segments (p1-p2 and p3-p4).
    
    Returns (min_dist, t1, t2) where t1, t2 are parametric positions
    on segments 1 and 2 respectively (0 to 1).
    """
    d1 = p2 - p1  # direction of segment 1
    d2 = p4 - p3  # direction of segment 2
    r = p1 - p3
    
    a = np.dot(d1, d1)  # |d1|^2
    e = np.dot(d2, d2)  # |d2|^2
    f = np.dot(d2, r)
    
    eps = 1e-12
    
    if a <= eps and e <= eps:
        # Both segments degenerate to points
        return np.linalg.norm(r), 0.0, 0.0
    
    if a <= eps:
        # First segment is a point
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = np.dot(d1, r)
        if e <= eps:
            # Second segment is a point
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            # General case
            b = np.dot(d1, d2)
            denom = a * e - b * b
            
            if abs(denom) > eps:
                s = np.clip((b * f - c * e) / denom, 0.0, 1.0)
            else:
                # Parallel segments
                s = 0.0
            
            t = (b * s + f) / e
            
            # Clamp t and recompute s if needed
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)
    
    closest1 = p1 + s * d1
    closest2 = p3 + t * d2
    dist = np.linalg.norm(closest1 - closest2)
    
    return dist, s, t


def _detect_intersections_3d(fibers, threshold_factor=2.0):
    """Detect intersections between 3D fibers based on segment proximity.
    
    Uses spatial hashing to reduce O(n^2) to O(n*k) complexity.
    Returns list of (fiber_i, fiber_j, param_i, param_j, position).
    """
    from collections import defaultdict
    
    crosslinks = []
    n = len(fibers)
    
    # Build spatial hash grid
    # Use cell size based on typical fiber length
    if n == 0:
        return crosslinks
    
    # Estimate cell size from fiber lengths
    avg_length = np.mean([f.length for f in fibers])
    cell_size = max(avg_length / 2, 5.0)
    
    # Map segments to grid cells
    grid = defaultdict(list)  # cell_key -> [(fiber_idx, seg_idx)]
    
    for i, fiber in enumerate(fibers):
        for si in range(fiber.num_points - 1):
            p1 = fiber.centerline[si]
            p2 = fiber.centerline[si + 1]
            # Use midpoint for cell assignment
            mid = (p1 + p2) / 2
            cell_key = (int(mid[0] / cell_size), int(mid[1] / cell_size), int(mid[2] / cell_size))
            grid[cell_key].append((i, si))
    
    # Check pairs within nearby cells
    checked_pairs = set()
    
    for cell_key, segments in grid.items():
        # Check all segments in this cell
        for idx1 in range(len(segments)):
            i1, si1 = segments[idx1]
            for idx2 in range(idx1 + 1, len(segments)):
                i2, si2 = segments[idx2]
                
                # Skip same fiber
                if i1 == i2:
                    continue
                
                # Skip already checked pair
                pair_key = (min(i1, i2), max(i1, i2))
                if pair_key in checked_pairs:
                    continue
                
                fi = fibers[i1]
                fj = fibers[i2]
                
                p1 = fi.centerline[si1]
                p2 = fi.centerline[si1 + 1]
                p3 = fj.centerline[si2]
                p4 = fj.centerline[si2 + 1]
                
                dist, t1, t2 = _segment_min_distance_3d(p1, p2, p3, p4)
                
                # Check if close enough
                threshold = (fi.radius + fj.radius) * threshold_factor
                if dist < threshold:
                    param_i = (si1 + t1) / (fi.num_points - 1)
                    param_j = (si2 + t2) / (fj.num_points - 1)
                    
                    closest1 = p1 + t1 * (p2 - p1)
                    closest2 = p3 + t2 * (p4 - p3)
                    pos = (closest1 + closest2) / 2
                    
                    crosslinks.append((i1, i2, param_i, param_j, pos))
        
        # Mark pair as checked after processing this cell
        for idx1 in range(len(segments)):
            i1, _ = segments[idx1]
            for idx2 in range(idx1 + 1, len(segments)):
                i2, _ = segments[idx2]
                if i1 != i2:
                    pair_key = (min(i1, i2), max(i1, i2))
                    checked_pairs.add(pair_key)
    
    return crosslinks


def random_straight_2d(
    num_fibers: int = 100,
    fiber_length: float = 10.0,
    box_size: Optional[float] = None,
    radius: float = 0.1,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
    ensure_connected: bool = True,
    **kwargs,
) -> FiberNetwork:
    """Generate random straight fibers in 2D with crosslinks at intersections.
    
    Parameters
    ----------
    num_fibers : int
        Number of fibers.
    fiber_length : float
        Length of each fiber.
    box_size : float
        Size of the square domain.
    radius : float
        Fiber radius.
    """
    mat = material or Material(name="random_2d")
    rng = np.random.RandomState(seed)
    
    # Auto-compute box_size for percolation if not specified
    if box_size is None:
        bx = by = _compute_percolation_box_2d(num_fibers, fiber_length)
    elif isinstance(box_size, (list, tuple, np.ndarray)):
        bx = box_size[0] if len(box_size) > 0 else 50.0
        by = box_size[1] if len(box_size) > 1 else bx
    else:
        bx = by = float(box_size)
    
    net = FiberNetwork(
        dimension=2,
        box_size=np.array([bx, by, 0.0]),
        metadata={"generator": "random_2d", "num_fibers": num_fibers},
    )
    
    # Generate random fibers
    fibers = []
    for i in range(num_fibers):
        # Random center
        cx = rng.uniform(0, bx)
        cy = rng.uniform(0, by)
        # Random angle
        angle = rng.uniform(0, 2 * np.pi)
        
        # Fiber endpoints
        dx = fiber_length / 2 * np.cos(angle)
        dy = fiber_length / 2 * np.sin(angle)
        
        start = np.array([cx - dx, cy - dy, 0.0])
        end = np.array([cx + dx, cy + dy, 0.0])
        
        fiber = Fiber.straight(start, end, radius=radius, material=mat, fiber_id=i)
        net.add_fiber(fiber)
        fibers.append(fiber)
    
    # Detect 2D intersections (line segment crossing)
    for i in range(len(fibers)):
        fi = fibers[i]
        p1, p2 = fi.centerline[0], fi.centerline[-1]
        
        for j in range(i + 1, len(fibers)):
            fj = fibers[j]
            p3, p4 = fj.centerline[0], fj.centerline[-1]
            
            # 2D line intersection
            d1 = p2 - p1
            d2 = p4 - p3
            cross = d1[0] * d2[1] - d1[1] * d2[0]
            
            if abs(cross) < 1e-10:
                continue  # Parallel
            
            dp = p3 - p1
            t = (dp[0] * d2[1] - dp[1] * d2[0]) / cross
            u = (dp[0] * d1[1] - dp[1] * d1[0]) / cross
            
            if 0 <= t <= 1 and 0 <= u <= 1:
                pos = p1 + t * d1
                net.add_crosslink(Crosslink(
                    fiber_i=i, fiber_j=j,
                    param_i=t, param_j=u,
                    position=pos,
                    crosslink_type="welded",
                ))
    
    if ensure_connected:
        _ensure_connected(net)
    
    return net


def random_straight_3d(
    num_fibers: int = 100,
    fiber_length: float = 10.0,
    box_size: Optional[float] = None,
    radius: float = 0.5,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
    threshold_factor: float = 2.0,
    ensure_connected: bool = True,
    **kwargs,
) -> FiberNetwork:
    """Generate random straight fibers in 3D with geometric intersection detection.
    
    Uses minimum segment distance to detect crossing points and create
    crosslinks where fibers pass near each other.
    
    Parameters
    ----------
    num_fibers : int
        Number of fibers.
    fiber_length : float
        Length of each fiber.
    box_size : float
        Size of the cubic domain.
    radius : float
        Fiber radius.
    threshold_factor : float
        Crosslink threshold as multiple of sum of radii.
    """
    mat = material or Material(name="random_3d")
    rng = np.random.RandomState(seed)
    
    # Auto-compute box_size for percolation if not specified
    if box_size is None:
        box_size = _compute_percolation_box_3d(num_fibers, fiber_length, threshold_factor)
    box_size = float(box_size)
    
    net = FiberNetwork(
        dimension=3,
        box_size=np.array([box_size, box_size, box_size]),
        metadata={"generator": "random_3d", "num_fibers": num_fibers},
    )
    
    # Generate random fibers
    fibers = []
    for i in range(num_fibers):
        # Random center
        center = rng.uniform(0, box_size, size=3)
        # Random direction (uniform on sphere)
        theta = rng.uniform(0, 2 * np.pi)
        phi = np.arccos(rng.uniform(-1, 1))
        direction = np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi),
        ])
        
        start = center - direction * fiber_length / 2
        end = center + direction * fiber_length / 2
        
        fiber = Fiber.straight(start, end, radius=radius, material=mat, fiber_id=i)
        net.add_fiber(fiber)
        fibers.append(fiber)
    
    # Detect 3D intersections using segment proximity
    intersections = _detect_intersections_3d(fibers, threshold_factor)
    
    for (fi, fj, pi, pj, pos) in intersections:
        net.add_crosslink(Crosslink(
            fiber_i=fi, fiber_j=fj,
            param_i=pi, param_j=pj,
            position=pos,
            crosslink_type="welded",
        ))
    
    if ensure_connected:
        _ensure_connected(net, max_gap_factor=threshold_factor)
    
    return net


def random_walk_fibers(
    num_fibers: int = 50,
    num_steps: int = 30,
    step_size: float = 1.0,
    box_size: Optional[float] = None,
    radius: float = 0.1,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
    threshold_factor: float = 2.0,
    ensure_connected: bool = True,
    **kwargs,
) -> FiberNetwork:
    """Generate random walk fibers with intersection detection.
    
    Each fiber is a random walk in 2D or 3D space.
    """
    mat = material or Material(name="random_walk")
    
    # Auto-compute box_size if not specified
    if box_size is None:
        # Estimate effective fiber length from random walk
        eff_length = step_size * np.sqrt(num_steps)
        box_size = max(_compute_percolation_box_3d(num_fibers, eff_length, threshold_factor), 
                       eff_length * 2)
    rng = np.random.RandomState(seed)
    
    net = FiberNetwork(
        dimension=3,
        box_size=np.array([box_size, box_size, box_size]),
        metadata={"generator": "random_walk", "num_fibers": num_fibers},
    )
    
    # Generate random walk fibers
    fibers = []
    for i in range(num_fibers):
        # Start position
        pos = rng.uniform(0, box_size, size=3)
        points = [pos.copy()]
        
        for _ in range(num_steps):
            # Random step
            direction = rng.randn(3)
            direction /= np.linalg.norm(direction) + 1e-12
            pos = pos + direction * step_size
            
            # Reflect at boundaries
            for d in range(3):
                if pos[d] < 0:
                    pos[d] = -pos[d]
                elif pos[d] > box_size:
                    pos[d] = 2 * box_size - pos[d]
            
            points.append(pos.copy())
        
        centerline = np.array(points)
        fiber = Fiber(centerline=centerline, radius=radius, material=mat, fiber_id=i)
        net.add_fiber(fiber)
        fibers.append(fiber)
    
    # Detect intersections
    intersections = _detect_intersections_3d(fibers, threshold_factor)
    for (fi, fj, pi, pj, pos) in intersections:
        net.add_crosslink(Crosslink(
            fiber_i=fi, fiber_j=fj,
            param_i=pi, param_j=pj,
            position=pos,
            crosslink_type="welded",
        ))
    
    if ensure_connected:
        _ensure_connected(net, max_gap_factor=threshold_factor)
    
    return net


def oriented_random_2d(
    num_fibers: int = 100,
    fiber_length: float = 10.0,
    box_size: Optional[float] = None,
    mean_angle: float = 0.0,
    angle_std: float = 0.2,
    radius: float = 0.1,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
    ensure_connected: bool = True,
    **kwargs,
) -> FiberNetwork:
    """Generate oriented random fibers in 2D.
    
    Fibers are aligned around a mean direction with some angular spread.
    """
    mat = material or Material(name="oriented_2d")
    rng = np.random.RandomState(seed)
    
    # Auto-compute box_size for percolation if not specified
    # Oriented fibers percolate easier, use factor 1.5
    if box_size is None:
        box_size = _compute_percolation_box_2d(num_fibers, fiber_length, threshold_factor=1.5)
    box_size = float(box_size)
    
    net = FiberNetwork(
        dimension=2,
        box_size=np.array([box_size, box_size, 0.0]),
        metadata={"generator": "oriented_2d", "num_fibers": num_fibers},
    )
    
    fibers = []
    for i in range(num_fibers):
        cx = rng.uniform(0, box_size)
        cy = rng.uniform(0, box_size)
        angle = mean_angle + rng.randn() * angle_std
        
        dx = fiber_length / 2 * np.cos(angle)
        dy = fiber_length / 2 * np.sin(angle)
        
        start = np.array([cx - dx, cy - dy, 0.0])
        end = np.array([cx + dx, cy + dy, 0.0])
        
        fiber = Fiber.straight(start, end, radius=radius, material=mat, fiber_id=i)
        net.add_fiber(fiber)
        fibers.append(fiber)
    
    # Detect 2D intersections
    for i in range(len(fibers)):
        fi = fibers[i]
        p1, p2 = fi.centerline[0], fi.centerline[-1]
        
        for j in range(i + 1, len(fibers)):
            fj = fibers[j]
            p3, p4 = fj.centerline[0], fj.centerline[-1]
            
            d1 = p2 - p1
            d2 = p4 - p3
            cross = d1[0] * d2[1] - d1[1] * d2[0]
            
            if abs(cross) < 1e-10:
                continue
            
            dp = p3 - p1
            t = (dp[0] * d2[1] - dp[1] * d2[0]) / cross
            u = (dp[0] * d1[1] - dp[1] * d1[0]) / cross
            
            if 0 <= t <= 1 and 0 <= u <= 1:
                pos = p1 + t * d1
                net.add_crosslink(Crosslink(
                    fiber_i=i, fiber_j=j,
                    param_i=t, param_j=u,
                    position=pos,
                    crosslink_type="welded",
                ))
    
    if ensure_connected:
        _ensure_connected(net)
    
    return net


def oriented_random_3d(
    num_fibers: int = 100,
    fiber_length: float = 10.0,
    box_size: Optional[float] = None,
    mean_direction: Optional[np.ndarray] = None,
    angle_std: float = 0.2,
    radius: float = 0.5,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
    threshold_factor: float = 2.0,
    ensure_connected: bool = True,
    **kwargs,
) -> FiberNetwork:
    """Generate oriented random fibers in 3D with intersection detection."""
    mat = material or Material(name="oriented_3d")
    rng = np.random.RandomState(seed)
    
    if mean_direction is None:
        mean_direction = np.array([1.0, 0.0, 0.0])
    mean_direction = np.array(mean_direction, dtype=float)
    mean_direction /= np.linalg.norm(mean_direction)
    
    # Auto-compute box_size for percolation if not specified
    if box_size is None:
        box_size = _compute_percolation_box_3d(num_fibers, fiber_length, threshold_factor)
    box_size = float(box_size)
    
    net = FiberNetwork(
        dimension=3,
        box_size=np.array([box_size, box_size, box_size]),
        metadata={"generator": "oriented_3d", "num_fibers": num_fibers},
    )
    
    fibers = []
    for i in range(num_fibers):
        center = rng.uniform(0, box_size, size=3)
        
        # Perturb direction
        perturbation = rng.randn(3) * angle_std
        direction = mean_direction + perturbation
        direction /= np.linalg.norm(direction) + 1e-12
        
        start = center - direction * fiber_length / 2
        end = center + direction * fiber_length / 2
        
        fiber = Fiber.straight(start, end, radius=radius, material=mat, fiber_id=i)
        net.add_fiber(fiber)
        fibers.append(fiber)
    
    # Detect 3D intersections
    intersections = _detect_intersections_3d(fibers, threshold_factor)
    for (fi, fj, pi, pj, pos) in intersections:
        net.add_crosslink(Crosslink(
            fiber_i=fi, fiber_j=fj,
            param_i=pi, param_j=pj,
            position=pos,
            crosslink_type="welded",
        ))
    
    if ensure_connected:
        _ensure_connected(net, max_gap_factor=threshold_factor)
    
    return net


def electrospun_random(
    num_fibers: int = 200,
    fiber_length_mean: float = 50.0,
    fiber_length_std: float = 10.0,
    box_size: float = 100.0,
    radius_mean: float = 0.1,
    radius_std: float = 0.03,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> FiberNetwork:
    """Generate electrospun-like random fiber mat in 2D."""
    mat = material or Material(name="electrospun")
    rng = np.random.RandomState(seed)
    
    net = FiberNetwork(
        dimension=2,
        box_size=np.array([box_size, box_size, 0.0]),
        metadata={"generator": "electrospun", "num_fibers": num_fibers},
    )
    
    fibers = []
    for i in range(num_fibers):
        flen = max(5.0, rng.normal(fiber_length_mean, fiber_length_std))
        frad = max(0.01, rng.normal(radius_mean, radius_std))
        
        cx = rng.uniform(0, box_size)
        cy = rng.uniform(0, box_size)
        angle = rng.uniform(0, 2 * np.pi)
        
        dx = flen / 2 * np.cos(angle)
        dy = flen / 2 * np.sin(angle)
        
        start = np.array([cx - dx, cy - dy, 0.0])
        end = np.array([cx + dx, cy + dy, 0.0])
        
        fiber = Fiber.straight(start, end, radius=frad, material=mat, fiber_id=i)
        net.add_fiber(fiber)
        fibers.append(fiber)
    
    # Detect 2D intersections
    for i in range(len(fibers)):
        fi = fibers[i]
        p1, p2 = fi.centerline[0], fi.centerline[-1]
        
        for j in range(i + 1, len(fibers)):
            fj = fibers[j]
            p3, p4 = fj.centerline[0], fj.centerline[-1]
            
            d1 = p2 - p1
            d2 = p4 - p3
            cross = d1[0] * d2[1] - d1[1] * d2[0]
            
            if abs(cross) < 1e-10:
                continue
            
            dp = p3 - p1
            t = (dp[0] * d2[1] - dp[1] * d2[0]) / cross
            u = (dp[0] * d1[1] - dp[1] * d1[0]) / cross
            
            if 0 <= t <= 1 and 0 <= u <= 1:
                pos = p1 + t * d1
                net.add_crosslink(Crosslink(
                    fiber_i=i, fiber_j=j,
                    param_i=t, param_j=u,
                    position=pos,
                    crosslink_type="welded",
                ))
    
    # Ensure connected
    from fibernet.gen.disordered import _ensure_connected
    _ensure_connected(net, max_gap_factor=5.0)
    
    return net


def voronoi_2d(
    num_seeds: int = 50,
    box_size: float = 50.0,
    radius: float = 0.1,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> FiberNetwork:
    """Generate 2D fiber network from Voronoi tessellation edges."""
    mat = material or Material(name="voronoi_2d")
    rng = np.random.RandomState(seed)
    
    # Generate random seed points
    points = rng.uniform(0, box_size, size=(num_seeds, 2))
    
    # Use scipy Voronoi
    try:
        from scipy.spatial import Voronoi
        vor = Voronoi(points)
    except ImportError:
        raise ImportError("scipy required for Voronoi tessellation")
    
    g = __import__('fibernet.gen._graph_builder', fromlist=['FiberGraph']).FiberGraph(
        dimension=2, tolerance=box_size * 0.001
    )
    
    # Add edges from Voronoi ridges
    for ridge_vertices in vor.ridge_vertices:
        if -1 in ridge_vertices:
            continue  # Skip infinite ridges
        p1 = vor.vertices[ridge_vertices[0]]
        p2 = vor.vertices[ridge_vertices[1]]
        
        # Check bounds
        if (0 <= p1[0] <= box_size and 0 <= p1[1] <= box_size and
            0 <= p2[0] <= box_size and 0 <= p2[1] <= box_size):
            g.add_edge_by_pos(
                np.array([p1[0], p1[1], 0.0]),
                np.array([p2[0], p2[1], 0.0]),
                radius=radius, material=mat,
            )
    
    return g.to_network(
        material=mat,
        box_size=np.array([box_size, box_size, 0.0]),
        metadata={"generator": "voronoi_2d", "num_seeds": num_seeds},
    )


def voronoi_3d(
    num_seeds: int = 50,
    box_size: float = 50.0,
    radius: float = 0.1,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> FiberNetwork:
    """Generate 3D fiber network from 3D Voronoi tessellation edges."""
    mat = material or Material(name="voronoi_3d")
    rng = np.random.RandomState(seed)
    
    points = rng.uniform(0, box_size, size=(num_seeds, 3))
    
    try:
        from scipy.spatial import Voronoi
        vor = Voronoi(points)
    except ImportError:
        raise ImportError("scipy required for Voronoi tessellation")
    
    g = __import__('fibernet.gen._graph_builder', fromlist=['FiberGraph']).FiberGraph(
        dimension=3, tolerance=box_size * 0.001
    )
    
    # Add edges from Voronoi ridges
    for ridge_vertices in vor.ridge_vertices:
        if -1 in ridge_vertices:
            continue
        for i in range(len(ridge_vertices)):
            for j in range(i + 1, len(ridge_vertices)):
                p1 = vor.vertices[ridge_vertices[i]]
                p2 = vor.vertices[ridge_vertices[j]]
                
                # Check bounds
                if (np.all(p1 >= 0) and np.all(p1 <= box_size) and
                    np.all(p2 >= 0) and np.all(p2 <= box_size)):
                    g.add_edge_by_pos(p1, p2, radius=radius, material=mat)
    
    return g.to_network(
        material=mat,
        box_size=np.array([box_size, box_size, box_size]),
        metadata={"generator": "voronoi_3d", "num_seeds": num_seeds},
    )


def _bridge_if_needed(net: FiberNetwork, max_gap_factor: float = 3.0):
    """Bridge disconnected components if needed."""
    from collections import defaultdict
    adj = defaultdict(set)
    for cl in net.crosslinks:
        adj[cl.fiber_i].add(cl.fiber_j)
        adj[cl.fiber_j].add(cl.fiber_i)
    
    visited = set()
    n_comp = 0
    for s in range(net.num_fibers):
        if s not in visited:
            n_comp += 1
            q = [s]
            while q:
                n = q.pop(0)
                if n in visited: continue
                visited.add(n)
                q.extend(adj[n] - visited)
    
    if n_comp > 1:
        max_gap = max_gap_factor * net.mean_fiber_length if net.mean_fiber_length > 0 else 50.0
        net.connect_components(max_gap=max_gap)


# ============================================================================
# Aliases and additional generators
# ============================================================================

def poisson_line_network_2d(
    intensity: float = 0.5,
    box_size: float = 50.0,
    radius: float = 0.1,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> FiberNetwork:
    """Generate a Poisson line process fiber network in 2D.
    
    The number of lines follows a Poisson distribution with parameter
    intensity * box_size^2.
    """
    rng = np.random.RandomState(seed)
    num_fibers = rng.poisson(intensity * box_size * box_size / (10.0 * 10.0))
    num_fibers = max(10, num_fibers)
    
    return random_straight_2d(
        num_fibers=num_fibers,
        fiber_length=10.0,
        box_size=box_size,
        radius=radius,
        material=material,
        seed=seed,
    )


def random_curved_fibers_3d(
    num_fibers: int = 50,
    fiber_length: float = 20.0,
    box_size: float = 50.0,
    curvature: float = 0.3,
    num_segments: int = 10,
    radius: float = 0.3,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
    threshold_factor: float = 2.0,
    **kwargs,
) -> FiberNetwork:
    """Generate random curved fibers in 3D with intersection detection.
    
    Each fiber is a random walk with controlled curvature.
    """
    mat = material or Material(name="curved_3d")
    rng = np.random.RandomState(seed)
    
    net = FiberNetwork(
        dimension=3,
        box_size=np.array([box_size, box_size, box_size]),
        metadata={"generator": "random_curved_fibers_3d", "num_fibers": num_fibers},
    )
    
    seg_len = fiber_length / num_segments
    fibers = []
    
    for i in range(num_fibers):
        # Start position
        pos = rng.uniform(0, box_size, size=3)
        # Initial direction
        theta = rng.uniform(0, 2 * np.pi)
        phi = np.arccos(rng.uniform(-1, 1))
        direction = np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi),
        ])
        
        points = [pos.copy()]
        for _ in range(num_segments):
            # Perturb direction (curvature control)
            perturbation = rng.randn(3) * curvature
            direction = direction + perturbation
            direction /= np.linalg.norm(direction) + 1e-12
            
            pos = pos + direction * seg_len
            
            # Reflect at boundaries
            for d in range(3):
                if pos[d] < 0: pos[d] = -pos[d]
                elif pos[d] > box_size: pos[d] = 2 * box_size - pos[d]
            
            points.append(pos.copy())
        
        centerline = np.array(points)
        fiber = Fiber(centerline=centerline, radius=radius, material=mat, fiber_id=i)
        net.add_fiber(fiber)
        fibers.append(fiber)
    
    # Detect intersections
    intersections = _detect_intersections_3d(fibers, threshold_factor)
    for (fi, fj, pi, pj, pos) in intersections:
        net.add_crosslink(Crosslink(
            fiber_i=fi, fiber_j=fj,
            param_i=pi, param_j=pj,
            position=pos,
            crosslink_type="welded",
        ))
    
    # Note: network may have multiple components. Use net.connect_components() to bridge if needed.
    # Ensure connected
    from fibernet.gen.disordered import _ensure_connected
    _ensure_connected(net, max_gap_factor=5.0)
    
    return net
