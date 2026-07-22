"""
Chiral fiber network generators.

Generates chiral/helical fiber structures including:
- Single helices (left/right handed)
- Double helices (DNA-like)
- Braided ropes
- Twisted yarns
- Chiral metamaterials
"""

import numpy as np
from typing import Optional, Tuple, List
from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork, Crosslink
from fibernet.core.material import Material
from fibernet.gen.disordered import _detect_intersections_3d, _bridge_if_needed


def single_helix(
    helix_radius: float = 5.0,
    pitch: float = 2.0,
    num_turns: float = 5.0,
    fiber_radius: float = 0.2,
    handedness: str = "right",
    material: Optional[Material] = None,
    segments_per_turn: int = 50,
) -> FiberNetwork:
    """Generate a single helical fiber.
    
    Parameters
    ----------
    helix_radius : float
        Radius from axis center.
    pitch : float
        Axial distance per turn.
    num_turns : float
        Number of complete turns.
    fiber_radius : float
        Cross-section radius of the fiber.
    handedness : str
        'right' or 'left'.
    """
    mat = material or Material(name="helical_fiber")
    sign = 1.0 if handedness == "right" else -1.0
    
    fiber = Fiber.helical(
        axis_direction=np.array([0, 0, 1]),
        center=np.array([0, 0, 0]),
        helix_radius=helix_radius,
        pitch=pitch,
        num_turns=num_turns,
        fiber_radius=fiber_radius,
        material=mat,
        segments_per_turn=segments_per_turn,
    )
    
    net = FiberNetwork(
        dimension=3,
        box_size=np.array([2 * helix_radius, 2 * helix_radius, num_turns * pitch]),
        metadata={"generator": "single_helix", "handedness": handedness, "pitch": pitch},
    )
    net.add_fiber(fiber)
    return net


def double_helix(
    helix_radius: float = 5.0,
    pitch: float = 3.0,
    num_turns: float = 5.0,
    fiber_radius: float = 0.2,
    phase_offset: float = np.pi,
    material: Optional[Material] = None,
    segments_per_turn: int = 50,
    add_crosslinks: bool = True,
    crosslink_interval: int = 5,
) -> FiberNetwork:
    """Generate a double helix (DNA-like structure).
    
    Parameters
    ----------
    phase_offset : float
        Angular offset between the two strands (radians).
    add_crosslinks : bool
        Whether to add cross-links between strands (like base pairs).
    crosslink_interval : int
        Add a crosslink every N points along the helix.
    """
    mat = material or Material(name="helical_fiber")
    net = FiberNetwork(
        dimension=3,
        box_size=np.array([2 * helix_radius, 2 * helix_radius, num_turns * pitch]),
        metadata={"generator": "double_helix", "phase_offset": phase_offset},
    )
    
    for strand in range(2):
        phase = strand * phase_offset
        n_pts = int(num_turns * segments_per_turn) + 1
        t = np.linspace(0, num_turns * 2 * np.pi, n_pts)
        
        x = helix_radius * np.cos(t + phase)
        y = helix_radius * np.sin(t + phase)
        z = pitch * t / (2 * np.pi)
        
        points = np.column_stack([x, y, z])
        fiber = Fiber(centerline=points, radius=fiber_radius, material=mat, fiber_id=strand)
        net.add_fiber(fiber)
    
    if add_crosslinks and len(net.fibers) == 2:
        n_pts = net.fibers[0].num_points
        step = max(1, crosslink_interval)
        for i in range(0, min(n_pts, net.fibers[1].num_points), step):
            pos1 = net.fibers[0].centerline[i]
            pos2 = net.fibers[1].centerline[min(i, net.fibers[1].num_points - 1)]
            mid = 0.5 * (pos1 + pos2)
            net.add_crosslink(Crosslink(
                fiber_i=0, fiber_j=1,
                param_i=i / (n_pts - 1),
                param_j=i / (net.fibers[1].num_points - 1),
                position=mid,
                crosslink_type="bonded",
            ))
    
    return net


def braided_rope(
    num_strands: int = 3,
    rope_radius: float = 3.0,
    pitch: float = 10.0,
    num_turns: float = 3.0,
    fiber_radius: float = 0.3,
    material: Optional[Material] = None,
    segments_per_turn: int = 40,
) -> FiberNetwork:
    """Generate a braided rope structure.
    
    Parameters
    ----------
    num_strands : int
        Number of strands in the braid (typically 3, 4, or more).
    rope_radius : float
        Radius of the braid from center axis.
    pitch : float
        Axial distance per turn.
    """
    mat = material or Material(name="braid_fiber")
    net = FiberNetwork(
        dimension=3,
        box_size=np.array([2 * rope_radius, 2 * rope_radius, num_turns * pitch]),
        metadata={"generator": "braided_rope", "num_strands": num_strands},
    )
    
    for s in range(num_strands):
        phase = 2 * np.pi * s / num_strands
        n_pts = int(num_turns * segments_per_turn) + 1
        t = np.linspace(0, num_turns * 2 * np.pi, n_pts)
        
        sign = 1.0 if s % 2 == 0 else -1.0
        x = rope_radius * np.cos(sign * t + phase)
        y = rope_radius * np.sin(sign * t + phase)
        z = pitch * t / (2 * np.pi)
        
        points = np.column_stack([x, y, z])
        fiber = Fiber(centerline=points, radius=fiber_radius, material=mat, fiber_id=s)
        net.add_fiber(fiber)
    
    # Detect intersections between strands
    intersections = _detect_intersections_3d(net.fibers, threshold_factor=3.0)
    for (fi, fj, pi, pj, pos) in intersections:
        net.add_crosslink(Crosslink(
            fiber_i=fi, fiber_j=fj,
            param_i=pi, param_j=pj,
            position=pos,
            crosslink_type="welded",
        ))
    
    return net


def twisted_bundle(
    num_fibers: int = 7,
    bundle_radius: float = 2.0,
    fiber_radius: float = 0.3,
    twist_angle: float = np.pi / 4,
    total_length: float = 50.0,
    packing: str = "hexagonal",
    material: Optional[Material] = None,
    num_points: int = 100,
    seed: Optional[int] = None,
) -> FiberNetwork:
    """Generate a twisted fiber bundle.
    
    Parameters
    ----------
    num_fibers : int
        Number of fibers in the bundle.
    bundle_radius : float
        Overall bundle radius.
    twist_angle : float
        Total twist angle over the bundle length.
    packing : str
        Initial cross-section packing: 'hexagonal' or 'circular'.
    """
    rng = np.random.default_rng(seed)
    mat = material or Material(name="bundle_fiber")
    net = FiberNetwork(
        dimension=3,
        box_size=np.array([2 * bundle_radius, 2 * bundle_radius, total_length]),
        metadata={"generator": "twisted_bundle", "num_fibers": num_fibers, "twist_angle": twist_angle},
    )
    
    if packing == "hexagonal":
        offsets = _hexagonal_pack(num_fibers, fiber_radius)
    else:
        offsets = _circular_pack(num_fibers, bundle_radius)
    
    z_vals = np.linspace(0, total_length, num_points)
    
    for idx, (ox, oy) in enumerate(offsets[:num_fibers]):
        r = np.sqrt(ox**2 + oy**2)
        phi0 = np.arctan2(oy, ox)
        
        points = np.zeros((num_points, 3))
        for k, z in enumerate(z_vals):
            twist = twist_angle * z / total_length
            phi = phi0 + twist
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            points[k] = [x, y, z]
        
        fiber = Fiber(centerline=points, radius=fiber_radius, material=mat, fiber_id=idx)
        net.add_fiber(fiber)
    
    # Detect intersections between twisted fibers
    intersections = _detect_intersections_3d(net.fibers, threshold_factor=1.0)
    for (fi, fj, pi, pj, pos) in intersections:
        net.add_crosslink(Crosslink(
            fiber_i=fi, fiber_j=fj,
            param_i=pi, param_j=pj,
            position=pos,
            crosslink_type="welded",
        ))
    
    net.auto_crosslink(threshold=1.0)

    return net


def chiral_metamaterial(
    unit_cell_size: float = 10.0,
    grid_size: Tuple[int, int, int] = (3, 3, 3),
    helix_radius: float = 2.0,
    fiber_radius: float = 0.2,
    turns_per_cell: float = 1.0,
    material: Optional[Material] = None,
) -> FiberNetwork:
    """Generate a 3D chiral metamaterial structure.
    
    Each unit cell contains a helical inclusion, creating a chiral
    architecture useful for studying auxetic and wave propagation behavior.
    """
    mat = material or Material(name="chiral_meta")
    nx, ny, nz = grid_size
    a = unit_cell_size
    
    net = FiberNetwork(
        dimension=3,
        box_size=np.array([nx * a, ny * a, nz * a]),
        metadata={"generator": "chiral_metamaterial", "unit_cell_size": unit_cell_size},
    )
    
    fid = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cx = (i + 0.5) * a
                cy = (j + 0.5) * a
                cz = k * a
                
                center = np.array([cx, cy, cz])
                fiber = Fiber.helical(
                    axis_direction=np.array([0, 0, 1]),
                    center=center,
                    helix_radius=helix_radius,
                    pitch=a / turns_per_cell,
                    num_turns=turns_per_cell,
                    fiber_radius=fiber_radius,
                    material=mat,
                    fiber_id=fid,
                )
                net.add_fiber(fiber)
                fid += 1
    
    # Detect intersections between helices in different cells
    intersections = _detect_intersections_3d(net.fibers, threshold_factor=2.0)
    for (fi, fj, pi, pj, pos) in intersections:
        net.add_crosslink(Crosslink(
            fiber_i=fi, fiber_j=fj,
            param_i=pi, param_j=pj,
            position=pos,
            crosslink_type="welded",
        ))
    
    return net


def _hexagonal_pack(n: int, radius: float) -> List[Tuple[float, float]]:
    """Pack n circles in hexagonal arrangement, return (x, y) offsets."""
    if n <= 1:
        return [(0.0, 0.0)]
    
    offsets = [(0.0, 0.0)]
    d = 2.0 * radius * 1.05  # 5% gap
    
    ring = 1
    while len(offsets) < n:
        for side in range(6):
            for pos in range(ring):
                if len(offsets) >= n:
                    break
                angle = np.pi / 3 * side + np.pi / 6
                next_angle = np.pi / 3 * (side + 1) + np.pi / 6
                
                frac = pos / ring
                x_start = ring * d * np.cos(angle)
                y_start = ring * d * np.sin(angle)
                x_end = ring * d * np.cos(next_angle)
                y_end = ring * d * np.sin(next_angle)
                
                x = x_start + frac * (x_end - x_start)
                y = y_start + frac * (y_end - y_start)
                offsets.append((x, y))
            if len(offsets) >= n:
                break
        ring += 1
    
    return offsets[:n]


def _circular_pack(n: int, bundle_radius: float) -> List[Tuple[float, float]]:
    """Pack n fibers in circular arrangement."""
    if n <= 1:
        return [(0.0, 0.0)]
    
    offsets = [(0.0, 0.0)]
    for i in range(1, n):
        angle = 2 * np.pi * i / (n - 1)
        r = bundle_radius * 0.8
        offsets.append((r * np.cos(angle), r * np.sin(angle)))
    
    return offsets[:n]
