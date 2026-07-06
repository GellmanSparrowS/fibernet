"""
Mechanics metamaterial structure generators.

Generates architected materials for mechanical metamaterial design:
- Re-entrant honeycomb (2D/3D auxetic)
- Chiral honeycomb (node-ligament)
- Star-shaped auxetic
- Hierarchical lattices (self-similar)
- Proper octet truss (all 12 struts per unit cell)
- Diamond lattice (tetrahedral coordination)
- TPMS-inspired structures (gyroid, Schwarz-P)
- Plate/Shell lattices
- Arrowhead auxetic
- Missing-rib auxetic

References:
- Gibson & Ashby, Cellular Solids (1997)
- Evans et al., Nature Materials 2001 (auxetic)
- Lakes, Science 1987 (re-entrant foam)
- Spadoni & Ruzzene, J Intell Mat Syst Str 2012 (chiral honeycomb)
"""

import numpy as np
from typing import Optional, Tuple, List
from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork, Crosslink
from fibernet.core.material import Material


# ===========================================================================
# Re-entrant Honeycomb (2D Auxetic)
# ===========================================================================

def reentrant_honeycomb_2d(
    cell_height: float = 10.0,
    cell_width: float = 10.0,
    reentrant_angle: float = 150.0,
    grid_size: Tuple[int, int] = (5, 5),
    radius: float = 0.2,
    material: Optional[Material] = None,
    vertical_thickness: float = 1.0,
) -> FiberNetwork:
    """Generate a 2D re-entrant honeycomb auxetic structure.

    The re-entrant angle controls the inward-pointing cell walls that
    produce a negative Poisson's ratio under tension.

    Parameters
    ----------
    cell_height : float
        Vertical dimension of unit cell (h in Gibson-Ashby notation).
    cell_width : float
        Horizontal dimension of unit cell (l in Gibson-Ashby notation).
    reentrant_angle : float
        Angle of inclined cell walls in degrees.
        150° = strongly re-entrant (auxetic), 120° = hexagonal, <120° = concave.
    grid_size : tuple
        (nx, ny) number of unit cells.
    vertical_thickness : float
        Relative thickness of vertical struts vs inclined (1.0 = same).

    Returns
    -------
    FiberNetwork with re-entrant honeycomb topology.
    """
    mat = material or Material(name="reentrant")
    theta = np.radians(reentrant_angle)
    nx, ny = grid_size

    l = cell_width / 2
    h = cell_height / 2
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    unit_w = 2 * l * abs(cos_t)
    unit_h = h + l * sin_t

    Lx = nx * unit_w
    Ly = ny * unit_h

    net = FiberNetwork(
        dimension=2,
        box_size=np.array([Lx, Ly, 0.0]),
        metadata={
            "generator": "reentrant_honeycomb_2d",
            "reentrant_angle_deg": reentrant_angle,
            "cell_height": cell_height,
            "cell_width": cell_width,
        },
    )

    for ci in range(nx):
        for cj in range(ny):
            ox = ci * unit_w
            oy = cj * unit_h

            n1 = np.array([ox, oy, 0.0])
            n2 = np.array([ox, oy + h, 0.0])
            n3 = np.array([ox + l * cos_t, oy + h + l * sin_t, 0.0])
            n4 = np.array([ox + unit_w, oy + h, 0.0])
            n5 = np.array([ox + unit_w, oy, 0.0])
            n6 = np.array([ox + unit_w - l * cos_t, oy + l * sin_t, 0.0])
            n7 = np.array([ox + l * cos_t, oy + l * sin_t, 0.0])

            edges = [
                (n1, n2, vertical_thickness * radius),
                (n2, n3, radius),
                (n3, n4, radius),
                (n4, n5, vertical_thickness * radius),
                (n5, n6, radius),
                (n6, n7, radius),
                (n7, n1, radius),
            ]

            for p1, p2, r in edges:
                if np.linalg.norm(p2 - p1) > 1e-10:
                    net.add_fiber(Fiber.straight(
                        p1, p2, radius=r, material=mat, fiber_id=net.num_fibers
                    ))

    net.auto_crosslink(threshold=3.0 * radius)
    return net


def reentrant_honeycomb_3d(
    cell_size: float = 10.0,
    reentrant_angle: float = 150.0,
    grid_size: Tuple[int, int, int] = (3, 3, 3),
    radius: float = 0.2,
    material: Optional[Material] = None,
) -> FiberNetwork:
    """Generate a 3D re-entrant auxetic lattice.

    Extends the 2D re-entrant concept into 3D with cell walls
    pointing inward in all three orthogonal planes.

    Parameters
    ----------
    cell_size : float
        Unit cell edge length.
    reentrant_angle : float
        Inward strut angle in degrees (150° default for strong auxetic).
    grid_size : tuple
        (nx, ny, nz) number of unit cells.
    """
    mat = material or Material(name="reentrant_3d")
    theta = np.radians(reentrant_angle)
    nx, ny, nz = grid_size
    a = cell_size

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    d = a * abs(cos_t)
    unit_h = a * (1 + sin_t)

    Lx = nx * 2 * d
    Ly = ny * 2 * d
    Lz = nz * unit_h

    net = FiberNetwork(
        dimension=3,
        box_size=np.array([Lx, Ly, Lz]),
        metadata={
            "generator": "reentrant_honeycomb_3d",
            "reentrant_angle_deg": reentrant_angle,
        },
    )

    for ci in range(nx):
        for cj in range(ny):
            for ck in range(nz):
                ox = ci * 2 * d
                oy = cj * 2 * d
                oz = ck * unit_h

                corners = []
                for di in [0, 1]:
                    for dj in [0, 1]:
                        for dk in [0, 1]:
                            x = ox + di * 2 * d
                            y = oy + dj * 2 * d
                            z = oz + dk * unit_h
                            inward_x = ox + d if di == 0 else ox + d
                            inward_y = oy + d if dj == 0 else oy + d
                            inward_z = oz + unit_h / 2 if dk == 0 else oz + unit_h / 2
                            frac = 0.3 * (1 - sin_t)
                            px = x + frac * (inward_x - x)
                            py = y + frac * (inward_y - y)
                            pz = z + frac * (inward_z - z)
                            corners.append(np.array([px, py, pz]))

                edges_3d = [
                    (0, 1), (2, 3), (4, 5), (6, 7),
                    (0, 2), (1, 3), (4, 6), (5, 7),
                    (0, 4), (1, 5), (2, 6), (3, 7),
                ]

                for ei, ej in edges_3d:
                    p1, p2 = corners[ei], corners[ej]
                    if np.linalg.norm(p2 - p1) > 1e-10:
                        net.add_fiber(Fiber.straight(
                            p1, p2, radius=radius, material=mat,
                            fiber_id=net.num_fibers
                        ))

    net.auto_crosslink(threshold=3.0 * radius)
    return net


# ===========================================================================
# Chiral Honeycomb (Node-Ligament)
# ===========================================================================

def chiral_honeycomb_2d(
    node_radius: float = 3.0,
    ligament_length: float = 8.0,
    grid_size: Tuple[int, int] = (5, 5),
    fiber_radius: float = 0.2,
    num_node_points: int = 24,
    material: Optional[Material] = None,
) -> FiberNetwork:
    """Generate a 2D chiral honeycomb (node-ligament structure).

    Each unit cell has a circular node connected to neighbors via
    tangential ligaments, producing in-plane auxetic behavior
    and negative Poisson's ratio.

    Parameters
    ----------
    node_radius : float
        Radius of circular nodes.
    ligament_length : float
        Length of connecting ligaments between node surfaces.
    grid_size : tuple
        (nx, ny) number of unit cells.
    num_node_points : int
        Number of discretization points per circular node.

    Reference: Spadoni & Ruzzene, 2012.
    """
    mat = material or Material(name="chiral_honeycomb")
    nx, ny = grid_size
    R = node_radius
    ll = ligament_length
    unit_spacing = 2 * R + ll

    Lx = nx * unit_spacing
    Ly = ny * unit_spacing

    net = FiberNetwork(
        dimension=2,
        box_size=np.array([Lx, Ly, 0.0]),
        metadata={
            "generator": "chiral_honeycomb_2d",
            "node_radius": node_radius,
            "ligament_length": ligament_length,
        },
    )

    node_centers = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            cx = i * unit_spacing
            cy = j * unit_spacing
            if j % 2 == 1:
                cx += unit_spacing / 2
            node_centers.append((cx, cy))

    for cx, cy in node_centers:
            angles = np.linspace(0, 2 * np.pi, num_node_points, endpoint=False)
            ring_pts = np.column_stack([
                cx + R * np.cos(angles),
                cy + R * np.sin(angles),
                np.zeros(num_node_points),
            ])
            for k in range(num_node_points):
                p1 = ring_pts[k]
                p2 = ring_pts[(k + 1) % num_node_points]
                net.add_fiber(Fiber.straight(
                    p1, p2, radius=fiber_radius, material=mat,
                    fiber_id=net.num_fibers
                ))

    for idx_a, (cx_a, cy_a) in enumerate(node_centers):
        for idx_b, (cx_b, cy_b) in enumerate(node_centers):
            if idx_b <= idx_a:
                continue
            dist = np.sqrt((cx_b - cx_a)**2 + (cy_b - cy_a)**2)
            if abs(dist - unit_spacing) > unit_spacing * 0.1:
                continue

            angle_ab = np.arctan2(cy_b - cy_a, cx_b - cx_a)
            tangent_a = angle_ab + np.pi / 2
            tangent_b = angle_ab + np.pi / 2

            p1 = np.array([
                cx_a + R * np.cos(tangent_a),
                cy_a + R * np.sin(tangent_a),
                0.0,
            ])
            p2 = np.array([
                cx_b + R * np.cos(tangent_b),
                cy_b + R * np.sin(tangent_b),
                0.0,
            ])

            net.add_fiber(Fiber.straight(
                p1, p2, radius=fiber_radius, material=mat,
                fiber_id=net.num_fibers
            ))

    net.auto_crosslink(threshold=3.0 * fiber_radius)
    return net


# ===========================================================================
# Star-Shaped Auxetic
# ===========================================================================

def star_honeycomb_2d(
    star_arm_length: float = 5.0,
    star_inner_angle: float = 60.0,
    grid_size: Tuple[int, int] = (4, 4),
    radius: float = 0.2,
    num_arms: int = 4,
    material: Optional[Material] = None,
) -> FiberNetwork:
    """Generate a 2D star-shaped auxetic honeycomb.

    Star-shaped unit cells with inward-pointing arms produce
    negative Poisson's ratio. The geometry is controlled by
    arm length and the inner angle between arms.

    Parameters
    ----------
    star_arm_length : float
        Length of each star arm.
    star_inner_angle : float
        Angle between adjacent arms (degrees). Smaller = more auxetic.
    num_arms : int
        Number of arms per star (typically 4, 6, or 8).
    """
    mat = material or Material(name="star_auxetic")
    nx, ny = grid_size
    a = star_arm_length
    alpha = np.radians(star_inner_angle)

    unit_size = 2 * a * np.cos(alpha / 2) + a

    Lx = nx * unit_size
    Ly = ny * unit_size

    net = FiberNetwork(
        dimension=2,
        box_size=np.array([Lx, Ly, 0.0]),
        metadata={
            "generator": "star_honeycomb_2d",
            "star_arm_length": star_arm_length,
            "star_inner_angle": star_inner_angle,
            "num_arms": num_arms,
        },
    )

    for ci in range(nx):
        for cj in range(ny):
            cx = (ci + 0.5) * unit_size
            cy = (cj + 0.5) * unit_size
            center = np.array([cx, cy, 0.0])

            for k in range(num_arms):
                arm_angle = 2 * np.pi * k / num_arms

                tip = center + a * np.array([
                    np.cos(arm_angle), np.sin(arm_angle), 0.0
                ])
                net.add_fiber(Fiber.straight(
                    center, tip, radius=radius, material=mat,
                    fiber_id=net.num_fibers
                ))

                bend_angle = alpha / 2
                for sign in [-1, 1]:
                    bent = tip + a * 0.5 * np.array([
                        np.cos(arm_angle + np.pi + sign * bend_angle),
                        np.sin(arm_angle + np.pi + sign * bend_angle),
                        0.0,
                    ])
                    net.add_fiber(Fiber.straight(
                        tip, bent, radius=radius, material=mat,
                        fiber_id=net.num_fibers
                    ))

    net.auto_crosslink(threshold=3.0 * radius)
    return net


# ===========================================================================
# Arrowhead Auxetic
# ===========================================================================

def arrowhead_auxetic_2d(
    arm_length: float = 8.0,
    arm_angle: float = 60.0,
    grid_size: Tuple[int, int] = (5, 5),
    radius: float = 0.2,
    material: Optional[Material] = None,
) -> FiberNetwork:
    """Generate a 2D arrowhead (fishbone) auxetic structure.

    The arrowhead geometry produces strong auxetic behavior with
    a large negative Poisson's ratio range.

    Parameters
    ----------
    arm_length : float
        Length of each arrowhead arm.
    arm_angle : float
        Half-angle of the arrowhead tip (degrees).
    """
    mat = material or Material(name="arrowhead")
    nx, ny = grid_size
    l = arm_length
    theta = np.radians(arm_angle)

    unit_w = 2 * l * np.sin(theta)
    unit_h = 2 * l * np.cos(theta)

    Lx = nx * unit_w
    Ly = ny * unit_h

    net = FiberNetwork(
        dimension=2,
        box_size=np.array([Lx, Ly, 0.0]),
        metadata={
            "generator": "arrowhead_auxetic_2d",
            "arm_length": arm_length,
            "arm_angle": arm_angle,
        },
    )

    for ci in range(nx):
        for cj in range(ny):
            ox = ci * unit_w
            oy = cj * unit_h

            tip = np.array([ox + unit_w / 2, oy, 0.0])
            left = np.array([ox, oy + l * np.cos(theta), 0.0])
            right = np.array([ox + unit_w, oy + l * np.cos(theta), 0.0])
            bottom = np.array([ox + unit_w / 2, oy + unit_h, 0.0])

            net.add_fiber(Fiber.straight(tip, left, radius=radius, material=mat, fiber_id=net.num_fibers))
            net.add_fiber(Fiber.straight(tip, right, radius=radius, material=mat, fiber_id=net.num_fibers))
            net.add_fiber(Fiber.straight(left, bottom, radius=radius, material=mat, fiber_id=net.num_fibers))
            net.add_fiber(Fiber.straight(right, bottom, radius=radius, material=mat, fiber_id=net.num_fibers))

    net.auto_crosslink(threshold=3.0 * radius)
    return net


# ===========================================================================
# Hierarchical Lattice
# ===========================================================================

def hierarchical_lattice_2d(
    base_type: str = "triangular",
    levels: int = 2,
    cell_size: float = 50.0,
    radius: float = 0.2,
    radius_decay: float = 0.7,
    material: Optional[Material] = None,
) -> FiberNetwork:
    """Generate a 2D hierarchical lattice (self-similar at multiple scales).

    Each strut of the parent lattice is replaced by a smaller copy
    of the same lattice pattern, creating fractal-like structural
    hierarchy. Useful for studying scale-dependent mechanics.

    Parameters
    ----------
    base_type : str
        Base lattice: 'triangular', 'kagome', or 'square'.
    levels : int
        Number of hierarchical levels (1 = no hierarchy).
    cell_size : float
        Size of the largest (level-0) cell.
    radius_decay : float
        Radius reduction factor per level.
    """
    mat = material or Material(name="hierarchical_lattice")

    net = FiberNetwork(
        dimension=2,
        metadata={
            "generator": "hierarchical_lattice_2d",
            "base_type": base_type,
            "levels": levels,
        },
    )

    def _make_edges(p1, p2, level):
        if level >= levels:
            r = radius * (radius_decay ** level)
            net.add_fiber(Fiber.straight(
                p1, p2, radius=r, material=mat, fiber_id=net.num_fibers
            ))
            return

        n_sub = 3 if base_type == "triangular" else 2
        sub_pts = [p1 + (p2 - p1) * i / n_sub for i in range(n_sub + 1)]

        if base_type == "triangular":
            for i in range(n_sub):
                _make_edges(sub_pts[i], sub_pts[i + 1], level + 1)
            mid = 0.5 * (p1 + p2)
            h = np.linalg.norm(p2 - p1) * np.sqrt(3) / 2 / n_sub
            d = (p2 - p1) / np.linalg.norm(p2 - p1)
            perp = np.array([-d[1], d[0], 0.0])
            for i in range(n_sub):
                apex = 0.5 * (sub_pts[i] + sub_pts[i + 1]) + h * perp
                _make_edges(sub_pts[i], apex, level + 1)
                _make_edges(sub_pts[i + 1], apex, level + 1)
        else:
            for i in range(n_sub):
                _make_edges(sub_pts[i], sub_pts[i + 1], level + 1)

    if base_type in ("triangular", "kagome"):
        s = cell_size
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([s, 0.0, 0.0])
        p3 = np.array([s / 2, s * np.sqrt(3) / 2, 0.0])
        _make_edges(p1, p2, 0)
        _make_edges(p2, p3, 0)
        _make_edges(p3, p1, 0)
    else:
        s = cell_size
        corners = [
            np.array([0, 0, 0]), np.array([s, 0, 0]),
            np.array([s, s, 0]), np.array([0, s, 0]),
        ]
        for i in range(4):
            _make_edges(corners[i], corners[(i + 1) % 4], 0)

    bb_min, bb_max = net.bounding_box()
    net.box_size = bb_max - bb_min + 1e-6
    net.auto_crosslink(threshold=3.0 * radius)
    return net


# ===========================================================================
# Proper Octet Truss (all 12 struts)
# ===========================================================================

def proper_octet_truss_3d(
    spacing: float = 10.0,
    grid_size: Tuple[int, int, int] = (3, 3, 3),
    radius: float = 0.3,
    material: Optional[Material] = None,
) -> FiberNetwork:
    """Generate a 3D octet truss with correct topology.

    Each octahedral cell has 12 struts connecting nearest-neighbor
    nodes on the FCC lattice. This is the canonical stretch-dominated
    lattice (Deshpande, Fleck, Ashby 2001).

    Parameters
    ----------
    spacing : float
        Edge length of the cubic unit cell.
    grid_size : tuple
        (nx, ny, nz) number of unit cells.
    """
    mat = material or Material(name="octet_truss")
    nx, ny, nz = grid_size
    a = spacing
    half = a / 2

    net = FiberNetwork(
        dimension=3,
        box_size=np.array([nx * a, ny * a, nz * a]),
        metadata={"generator": "proper_octet_truss_3d", "spacing": spacing},
    )

    nodes = {}

    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                nodes[(2 * i, 2 * j, 2 * k)] = np.array([i * a, j * a, k * a])

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                bi = 2 * i + 1
                bj = 2 * j + 1
                bk = 2 * k + 1

                face_centers = [
                    (bi, bj, 2 * k),
                    (bi, bj, 2 * k + 2),
                    (bi, 2 * j, bk),
                    (bi, 2 * j + 2, bk),
                    (2 * i, bj, bk),
                    (2 * i + 2, bj, bk),
                ]
                for fc in face_centers:
                    pos = np.array([fc[0] * half, fc[1] * half, fc[2] * half])
                    nodes[fc] = pos

    edge_set = set()
    directions = [
        (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
        (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
        (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
    ]

    for key, pos in nodes.items():
        for di, dj, dk in directions:
            nb = (key[0] + di, key[1] + dj, key[2] + dk)
            if nb in nodes:
                edge_key = tuple(sorted([key, nb]))
                if edge_key not in edge_set:
                    edge_set.add(edge_key)
                    net.add_fiber(Fiber.straight(
                        pos, nodes[nb], radius=radius, material=mat,
                        fiber_id=net.num_fibers
                    ))

    net.auto_crosslink(threshold=3.0 * radius)
    return net


# ===========================================================================
# Diamond Lattice (tetrahedral coordination)
# ===========================================================================

def diamond_lattice_3d(
    spacing: float = 10.0,
    grid_size: Tuple[int, int, int] = (2, 2, 2),
    radius: float = 0.3,
    material: Optional[Material] = None,
) -> FiberNetwork:
    """Generate a 3D diamond lattice (tetrahedrally coordinated).

    Each node has exactly 4 neighbors in tetrahedral coordination,
    producing a bending-dominated isotropic lattice.

    Parameters
    ----------
    spacing : float
        Cubic unit cell edge length.
    grid_size : tuple
        (nx, ny, nz) number of unit cells.
    """
    mat = material or Material(name="diamond")
    nx, ny, nz = grid_size
    a = spacing

    net = FiberNetwork(
        dimension=3,
        box_size=np.array([nx * a, ny * a, nz * a]),
        metadata={"generator": "diamond_lattice_3d", "spacing": spacing},
    )

    basis = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.25, 0.25, 0.25]) * a,
    ]
    fcc_vecs = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.5, 0.5, 0.0]) * a,
        np.array([0.5, 0.0, 0.5]) * a,
        np.array([0.0, 0.5, 0.5]) * a,
    ]

    nodes = {}
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cell_origin = np.array([i * a, j * a, k * a])
                for fv in fcc_vecs:
                    for bv in basis:
                        pos = cell_origin + fv + bv
                        key = tuple(np.round(pos, 8))
                        nodes[key] = pos

    bond_length = a * np.sqrt(3) / 4
    tol = bond_length * 0.1

    edge_set = set()
    keys = list(nodes.keys())
    positions = np.array([nodes[k] for k in keys])
    key_to_idx = {k: i for i, k in enumerate(keys)}

    for i, k1 in enumerate(keys):
        for j in range(i + 1, len(keys)):
            dist = np.linalg.norm(positions[i] - positions[j])
            if abs(dist - bond_length) < tol:
                edge_set.add((i, j))

    for i, j in edge_set:
        net.add_fiber(Fiber.straight(
            positions[i], positions[j], radius=radius, material=mat,
            fiber_id=net.num_fibers
        ))

    net.auto_crosslink(threshold=3.0 * radius)
    return net


# ===========================================================================
# TPMS-Inspired Lattice (Gyroid)
# ===========================================================================

def gyroid_lattice_3d(
    cell_size: float = 10.0,
    grid_size: Tuple[int, int, int] = (3, 3, 3),
    resolution: int = 20,
    threshold: float = 0.0,
    radius: float = 0.15,
    material: Optional[Material] = None,
) -> FiberNetwork:
    """Generate a 3D gyroid-inspired lattice.

    Samples the gyroid implicit surface sin(x)cos(y) + sin(y)cos(z) +
    sin(z)cos(x) = threshold and creates a fiber network along
    the surface edges. Gyroid structures have isotropic mechanics
    and high strength-to-weight ratio.

    Parameters
    ----------
    cell_size : float
        Unit cell size (one gyroid period).
    resolution : int
        Grid points per unit cell for surface sampling.
    threshold : float
        Level-set value (0 = exact gyroid, ±0.5 = shifted).
    """
    mat = material or Material(name="gyroid")
    nx, ny, nz = grid_size
    a = cell_size
    n = resolution

    net = FiberNetwork(
        dimension=3,
        box_size=np.array([nx * a, ny * a, nz * a]),
        metadata={"generator": "gyroid_lattice_3d", "cell_size": cell_size},
    )

    total_n = n * max(nx, ny, nz)
    xs = np.linspace(0, nx * 2 * np.pi, total_n + 1)
    ys = np.linspace(0, ny * 2 * np.pi, total_n + 1)
    zs = np.linspace(0, nz * 2 * np.pi, total_n + 1)

    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    dz = zs[1] - zs[0]

    scale = a / (2 * np.pi)

    for ix in range(len(xs) - 1):
        for iy in range(len(ys) - 1):
            for iz in range(len(zs) - 1):
                x0, y0, z0 = xs[ix], ys[iy], zs[iz]
                x1, y1, z1 = xs[ix + 1], ys[iy + 1], zs[iz + 1]

                def gyroid(x, y, z):
                    return (np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) +
                            np.sin(z) * np.cos(x))

                v000 = gyroid(x0, y0, z0) - threshold
                v100 = gyroid(x1, y0, z0) - threshold
                v010 = gyroid(x0, y1, z0) - threshold
                v001 = gyroid(x0, y0, z1) - threshold

                sign_changes = []
                if v000 * v100 < 0:
                    t = v000 / (v000 - v100)
                    px = x0 + t * dx
                    sign_changes.append(np.array([px * scale, y0 * scale, z0 * scale]))
                if v000 * v010 < 0:
                    t = v000 / (v000 - v010)
                    py = y0 + t * dy
                    sign_changes.append(np.array([x0 * scale, py * scale, z0 * scale]))
                if v000 * v001 < 0:
                    t = v000 / (v000 - v001)
                    pz = z0 + t * dz
                    sign_changes.append(np.array([x0 * scale, y0 * scale, pz * scale]))

                if len(sign_changes) >= 2:
                    for k in range(len(sign_changes) - 1):
                        p1 = sign_changes[k]
                        p2 = sign_changes[k + 1]
                        if np.linalg.norm(p2 - p1) > 1e-10:
                            net.add_fiber(Fiber.straight(
                                p1, p2, radius=radius, material=mat,
                                fiber_id=net.num_fibers
                            ))

    net.auto_crosslink(threshold=3.0 * radius)
    return net


# ===========================================================================
# Missing-Rib Auxetic
# ===========================================================================

def missing_rib_auxetic_2d(
    cell_size: float = 10.0,
    grid_size: Tuple[int, int] = (5, 5),
    radius: float = 0.2,
    rib_angle: float = 45.0,
    material: Optional[Material] = None,
) -> FiberNetwork:
    """Generate a 2D missing-rib auxetic structure.

    A square grid with alternating ribs removed, creating a
    structure with negative Poisson's ratio. Based on the
    model by Gaspar et al. (2005).

    Parameters
    ----------
    cell_size : float
        Unit cell edge length.
    rib_angle : float
        Angle of diagonal ribs (degrees).
    """
    mat = material or Material(name="missing_rib")
    nx, ny = grid_size
    a = cell_size
    theta = np.radians(rib_angle)

    Lx = nx * a
    Ly = ny * a

    net = FiberNetwork(
        dimension=2,
        box_size=np.array([Lx, Ly, 0.0]),
        metadata={
            "generator": "missing_rib_auxetic_2d",
            "cell_size": cell_size,
            "rib_angle": rib_angle,
        },
    )

    for i in range(nx + 1):
        for j in range(ny + 1):
            x, y = i * a, j * a

            if i < nx:
                net.add_fiber(Fiber.straight(
                    np.array([x, y, 0]), np.array([x + a, y, 0]),
                    radius=radius, material=mat, fiber_id=net.num_fibers
                ))
            if j < ny:
                net.add_fiber(Fiber.straight(
                    np.array([x, y, 0]), np.array([x, y + a, 0]),
                    radius=radius, material=mat, fiber_id=net.num_fibers
                ))

    for i in range(nx):
        for j in range(ny):
            if (i + j) % 2 == 0:
                cx = (i + 0.5) * a
                cy = (j + 0.5) * a
                half_diag = a / (2 * np.cos(theta))

                p1 = np.array([cx - half_diag * np.cos(theta), cy - half_diag * np.sin(theta), 0])
                p2 = np.array([cx + half_diag * np.cos(theta), cy + half_diag * np.sin(theta), 0])
                net.add_fiber(Fiber.straight(
                    p1, p2, radius=radius * 0.8, material=mat,
                    fiber_id=net.num_fibers
                ))

    net.auto_crosslink(threshold=3.0 * radius)
    return net


# ===========================================================================
# Shell/Plate Lattice
# ===========================================================================

def plate_lattice_3d(
    spacing: float = 10.0,
    grid_size: Tuple[int, int, int] = (3, 3, 3),
    plate_thickness: float = 0.5,
    material: Optional[Material] = None,
) -> FiberNetwork:
    """Generate a 3D plate lattice (cubic + octahedral plates).

    Plate lattices achieve near-optimal stiffness at low relative
    density, outperforming beam-based lattices (Tancogne-Dejean 2018).
    Each plate is represented as a dense fiber sheet.

    Parameters
    ----------
    spacing : float
        Unit cell edge length.
    plate_thickness : float
        Number of fiber layers per plate (controls effective thickness).
    """
    mat = material or Material(name="plate_lattice")
    nx, ny, nz = grid_size
    a = spacing
    n_layers = max(1, int(plate_thickness))
    layer_spacing = a * 0.02

    net = FiberNetwork(
        dimension=3,
        box_size=np.array([nx * a, ny * a, nz * a]),
        metadata={"generator": "plate_lattice_3d", "spacing": spacing},
    )

    r = a * 0.01

    for ci in range(nx):
        for cj in range(ny):
            for ck in range(nz):
                ox, oy, oz = ci * a, cj * a, ck * a

                for layer in range(n_layers):
                    offset = (layer - n_layers / 2) * layer_spacing

                    for n_seg in range(4):
                        t1 = n_seg / 4
                        t2 = (n_seg + 1) / 4
                        x1, x2 = ox + t1 * a, ox + t2 * a

                        net.add_fiber(Fiber.straight(
                            np.array([x1, oy + offset, oz]),
                            np.array([x2, oy + offset, oz]),
                            radius=r, material=mat, fiber_id=net.num_fibers
                        ))
                        net.add_fiber(Fiber.straight(
                            np.array([ox + offset, oy + t1 * a, oz]),
                            np.array([ox + offset, oy + t2 * a, oz]),
                            radius=r, material=mat, fiber_id=net.num_fibers
                        ))
                        net.add_fiber(Fiber.straight(
                            np.array([ox, oy + offset, oz + t1 * a]),
                            np.array([ox, oy + offset, oz + t2 * a]),
                            radius=r, material=mat, fiber_id=net.num_fibers
                        ))

    net.auto_crosslink(threshold=3.0 * r)
    return net
