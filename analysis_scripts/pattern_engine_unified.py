#!/usr/bin/env python3
"""
Pattern Engine — Unified API v7
================================

Design principles:
- box=(w, h): cell dimensions
- points: absolute coordinates in box space
- fit_to_box=True: auto-scale/center custom points to fit box
- boundary_mode: 'error' | 'extend' | 'none'
- No random perturbation unless explicitly set (deterministic with seed)
- Grid size is always an API parameter (default 3×3)
- Full programmability: user controls positions, boundary contact, complexity
"""

import sys, os, numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Union
from collections import defaultdict
import hashlib

sys.path.insert(0, str(Path(__file__).parent.parent))

from fibernet.gen._graph_builder import FiberGraph
from fibernet.core.material import Material

try:
    from shapely.geometry import LineString, Point as ShapelyPoint
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False


# ══════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════

def _fit_points_to_box(
    points: List[Tuple[float, float]],
    box: Tuple[float, float],
    pad: float = 0.0,
) -> List[Tuple[float, float]]:
    """Scale and translate points to fit within [0+pad, w-pad]×[0+pad, h-pad].
    
    Maintains aspect ratio. Centers the shape within the box.
    """
    w, h = box
    pts = np.array(points, dtype=float)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    span = mx - mn
    if span[0] < 1e-12:
        span[0] = 1.0
    if span[1] < 1e-12:
        span[1] = 1.0

    scale_x = (w - 2 * pad) / span[0]
    scale_y = (h - 2 * pad) / span[1]
    scale = min(scale_x, scale_y)

    scaled = (pts - mn) * scale
    scaled_w = span[0] * scale
    scaled_h = span[1] * scale
    offset_x = (w - scaled_w) / 2
    offset_y = (h - scaled_h) / 2
    scaled[:, 0] += offset_x
    scaled[:, 1] += offset_y

    return [(float(x), float(y)) for x, y in scaled]


def _generate_polygon(box: Tuple[float, float], polygon_type: str, n_pts: int) -> List[Tuple[float, float]]:
    """Generate polygon perimeter in box coordinates."""
    w, h = box

    if polygon_type == 'square':
        corners = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
    elif polygon_type == 'triangle':
        corners = [(w/2, h), (0.0, 0.0), (w, 0.0)]
    elif polygon_type == 'hexagon':
        corners = [
            (w/2, 0),
            (w, h/4),
            (w, 3*h/4),
            (w/2, h),
            (0, 3*h/4),
            (0, h/4),
        ]
    else:
        raise ValueError(f"Unknown polygon_type: {polygon_type}")

    pts = []
    n_sides = len(corners)
    for i in range(n_sides):
        p1 = corners[i]
        p2 = corners[(i + 1) % n_sides]
        if i == 0:
            pts.append(p1)
        for j in range(1, n_pts + 1):
            t = j / (n_pts + 1)
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            pts.append((x, y))
        pts.append(p2)

    return pts


def _cn_symmetric_perturbation(
    pts: List[Tuple[float, float]],
    polygon_type: str,
    n_pts_per_side: int,
    perturbation: float,
    seed: int,
) -> List[Tuple[float, float]]:
    """Cn symmetric perturbation: one (dx,dy) rotated to all N sides."""
    rng = np.random.default_rng(seed)
    n_sides = {'square': 4, 'triangle': 3, 'hexagon': 6}.get(polygon_type, 4)

    side_indices = []
    idx = 1
    for s in range(n_sides):
        side = list(range(idx, idx + n_pts_per_side))
        side_indices.append(side)
        idx += n_pts_per_side + 1

    pts = list(pts)

    p0 = pts[0]
    p1 = pts[1] if len(pts) > 1 else pts[0]
    edge_len = np.sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)
    noise = perturbation * edge_len if edge_len > 0 else perturbation

    for k in range(n_pts_per_side):
        dx = rng.uniform(-noise, noise)
        dy = rng.uniform(-noise, noise)
        for s in range(n_sides):
            angle = 2 * np.pi * s / n_sides
            rdx = dx * np.cos(angle) - dy * np.sin(angle)
            rdy = dx * np.sin(angle) + dy * np.cos(angle)
            idx = side_indices[s][k]
            if idx < len(pts):
                x, y = pts[idx]
                pts[idx] = (x + rdx, y + rdy)

    return pts


def _intersection_weld(g: FiberGraph, material: Material, radius: float) -> int:
    """Detect edge crossings and add junction nodes."""
    if not HAS_SHAPELY:
        return 0

    edge_lines = []
    for eid, edge_obj in g.edges.items():
        ni, nj = edge_obj.node_i, edge_obj.node_j
        p1 = g.nodes[ni].position[:2]
        p2 = g.nodes[nj].position[:2]
        edge_lines.append((LineString([p1, p2]), ni, nj, edge_obj.radius))

    intersections = {}
    for i, (line1, n1a, n1b, r1) in enumerate(edge_lines):
        for j, (line2, n2a, n2b, r2) in enumerate(edge_lines[i+1:], i+1):
            if {n1a, n1b} & {n2a, n2b}:
                continue
            if line1.intersects(line2):
                isect = line1.intersection(line2)
                if isinstance(isect, ShapelyPoint):
                    intersections.setdefault(i, []).append((isect.x, isect.y, r1, r2))
                    intersections.setdefault(j, []).append((isect.x, isect.y, r1, r2))

    n_added = 0
    for edge_idx, isect_list in intersections.items():
        _, ni, nj, r = edge_lines[edge_idx]
        p1 = g.nodes[ni].position[:2]
        p2 = g.nodes[nj].position[:2]
        line = LineString([p1, p2])
        sorted_isects = sorted(isect_list, key=lambda pt: line.project(ShapelyPoint(pt[0], pt[1])))
        prev_nid = ni
        for (ix, iy, r1, r2) in sorted_isects:
            nid = g.add_node(np.array([ix, iy, 0.0]))
            avg_r = (r1 + r2) / 2
            if prev_nid != nid:
                g.add_edge(prev_nid, nid, radius=avg_r, material=material)
            prev_nid = nid
            n_added += 1
        if prev_nid != nj:
            g.add_edge(prev_nid, nj, radius=r, material=material)

    return n_added


def _check_cell_boundary_contact(
    base_pts: List[Tuple[float, float]],
    w: float,
    h: float,
    tol: float,
) -> dict:
    """Check which cell boundaries the base polyline touches."""
    xs = np.array([p[0] for p in base_pts])
    ys = np.array([p[1] for p in base_pts])
    return {
        'left':   bool(np.any(np.abs(xs) < tol)),
        'right':  bool(np.any(np.abs(xs - w) < tol)),
        'bottom': bool(np.any(np.abs(ys) < tol)),
        'top':    bool(np.any(np.abs(ys - h) < tol)),
    }


def _compute_boundary_extensions(
    base_pts: List[Tuple[float, float]],
    w: float,
    h: float,
) -> List[Tuple[int, Tuple[float, float]]]:
    """Compute bridge segments from polyline vertices to missing cell boundaries.
    
    Returns list of (vertex_index, boundary_point) pairs.
    Each bridge creates a fiber from the closest vertex to the boundary.
    
    Bridge endpoints use centroid coordinates to ensure adjacent cells
    share the same boundary point and merge correctly.
    """
    xs = np.array([p[0] for p in base_pts])
    ys = np.array([p[1] for p in base_pts])
    tol = min(w, h) * 0.01
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))

    extensions = []

    # Left boundary: bridge to (0, centroid_y)
    if not np.any(np.abs(xs) < tol):
        idx = int(np.argmin(xs))
        extensions.append((idx, (0.0, cy)))

    # Right boundary: bridge to (w, centroid_y)
    if not np.any(np.abs(xs - w) < tol):
        idx = int(np.argmax(xs))
        extensions.append((idx, (float(w), cy)))

    # Bottom boundary: bridge to (centroid_x, 0)
    if not np.any(np.abs(ys) < tol):
        idx = int(np.argmin(ys))
        extensions.append((idx, (cx, 0.0)))

    # Top boundary: bridge to (centroid_x, h)
    if not np.any(np.abs(ys - h) < tol):
        idx = int(np.argmax(ys))
        extensions.append((idx, (cx, float(h))))

    return extensions


# ══════════════════════════════════════════════════════════════
#  Main API
# ══════════════════════════════════════════════════════════════

def pattern_2d(
    box: Tuple[float, float] = (10.0, 10.0),
    points: Optional[List[Tuple[float, float]]] = None,
    closed: bool = True,
    polygon_type: Optional[str] = None,
    n_pts_per_side: int = 5,
    perturbation: float = 0.0,
    seed: int = 42,
    grid: Union[Tuple[int, int], int] = (3, 3),
    stagger: str = 'auto',
    mirror_x: bool = False,
    mirror_y: bool = False,
    rotation: float = 0,
    detect_intersections: bool = True,
    fit_to_box: bool = False,
    boundary_mode: str = 'none',
    radius: float = 0.1,
    material: Optional[Material] = None,
    name: str = 'pattern',
):
    """
    Unified 2D pattern generator (v7).

    Parameters
    ----------
    box : (w, h)
        Cell dimensions. All coordinates are in [0,w]×[0,h] space.
    points : [(x,y), ...] or None
        Custom polyline vertices in box coordinates.
        No auto-transform unless fit_to_box=True.
    closed : bool
        Whether to close the polyline into a loop.
    polygon_type : 'square' | 'triangle' | 'hexagon' or None
        Preset polygon (overrides points).
    n_pts_per_side : int
        Intermediate points per edge for polygon presets.
    perturbation : float
        Cn-symmetric perturbation magnitude (0=no perturbation).
        Only applies to polygon presets, not custom points.
    seed : int
        Random seed (deterministic output).
    grid : (nx, ny) or int
        Tiling grid. If int, uses (grid, grid). Default 3×3.
    stagger : 'auto' | 'none' | 'hex'
        Tiling mode. 'auto' selects based on polygon type.
    mirror_x, mirror_y : bool
        Per-cell mirror flip.
    rotation : float
        Per-cell rotation in degrees.
    detect_intersections : bool
        Edge crossing detection and welding (default True).
    fit_to_box : bool
        If True, auto-scale/center custom points to fit box.
        Maintains aspect ratio.
    boundary_mode : 'error' | 'extend' | 'none'
        'error': raise error if base polyline doesn't touch all 4 cell boundaries.
        'extend': auto-add bridge segments from polyline to missing boundaries.
        'none': leave as is (default).
    """
    mat = material or Material(name=name)
    if isinstance(grid, int):
        nx_grid, ny_grid = grid, grid
    else:
        nx_grid, ny_grid = grid
    w, h = box

    # ── Build base polyline ──
    if points is not None:
        base_pts = list(points)
        polygon_type_str = 'custom'
        if fit_to_box:
            base_pts = _fit_points_to_box(base_pts, box)
    elif polygon_type is not None:
        base_pts = _generate_polygon(box, polygon_type, n_pts_per_side)
        polygon_type_str = polygon_type
    else:
        raise ValueError("Must specify either 'points' or 'polygon_type'")

    # ── Apply perturbation (polygon presets only) ──
    if perturbation > 0 and polygon_type_str != 'custom':
        base_pts = _cn_symmetric_perturbation(base_pts, polygon_type_str, n_pts_per_side, perturbation, seed)

    # ── Boundary contact check ──
    tolerance = min(w, h) * 0.01
    contact = _check_cell_boundary_contact(base_pts, w, h, tolerance)
    missing = [k for k, v in contact.items() if not v]

    if boundary_mode == 'error' and missing:
        raise ValueError(
            f"Base polyline does not touch cell boundary: {missing}. "
            f"Contact: {contact}. "
            f"Use boundary_mode='extend' to auto-add bridge segments, "
            f"or adjust points to touch cell edges."
        )

    # Compute extension bridges if needed
    extensions = []
    if boundary_mode == 'extend' and missing:
        extensions = _compute_boundary_extensions(base_pts, w, h)

    # ── Build graph ──
    g = FiberGraph(dimension=2, tolerance=tolerance)

    if stagger == 'auto':
        if polygon_type_str == 'triangle':
            stagger = 'hex'
        else:
            stagger = 'none'

    if stagger == 'hex':
        sdx, sdy = 0.5 * w, h
    else:
        sdx, sdy = 0.0, h

    def transform_point(x, y, col, row):
        if rotation != 0:
            angle_rad = np.radians(rotation)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            cx, cy = w / 2, h / 2
            xr, yr = x - cx, y - cy
            x = cx + xr * cos_a - yr * sin_a
            y = cy + xr * sin_a + yr * cos_a
        px, py = x, y
        if mirror_x and col % 2 == 1:
            px = w - x
        if mirror_y and row % 2 == 1:
            py = h - y
        return px + col * w + (sdx if row % 2 == 1 else 0), py + row * sdy

    # ── Tile base polyline ──
    for col in range(nx_grid):
        for row in range(ny_grid):
            prev_nid = None
            for i, (x, y) in enumerate(base_pts):
                tx, ty = transform_point(x, y, col, row)
                nid = g.add_node(np.array([tx, ty, 0.0]))
                if prev_nid is not None and prev_nid != nid:
                    g.add_edge(prev_nid, nid, radius=radius, material=mat)
                prev_nid = nid
            if closed and len(base_pts) > 2:
                tx0, ty0 = transform_point(base_pts[0][0], base_pts[0][1], col, row)
                nid0 = g.add_node(np.array([tx0, ty0, 0.0]))
                if prev_nid != nid0:
                    g.add_edge(prev_nid, nid0, radius=radius, material=mat)

            # ── Tile extension bridges ──
            for (vert_idx, (bx, by)) in extensions:
                vx, vy = base_pts[vert_idx]
                tvx, tvy = transform_point(vx, vy, col, row)
                tbx, tby = transform_point(bx, by, col, row)
                vid = g.add_node(np.array([tvx, tvy, 0.0]))
                bid = g.add_node(np.array([tbx, tby, 0.0]))
                if vid != bid:
                    g.add_edge(vid, bid, radius=radius, material=mat)

    # ── Edge intersection welding ──
    n_isect = 0
    if detect_intersections:
        n_isect = _intersection_weld(g, mat, radius)

    total_w = nx_grid * w + (sdx if ny_grid > 1 and sdx > 0 else 0)
    total_h = ny_grid * sdy
    box_size = np.array([total_w, total_h, 0.0])

    hash_input = f"{seed}_{perturbation}_{nx_grid}_{ny_grid}_{w}_{h}"
    seed_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]

    return g.to_network(
        material=mat,
        box_size=box_size,
        metadata={
            'generator': 'pattern_2d',
            'version': 'v7',
            'polygon_type': polygon_type_str,
            'n_pts_per_side': n_pts_per_side if polygon_type_str != 'custom' else len(base_pts),
            'box': box,
            'perturbation': perturbation,
            'grid': (nx_grid, ny_grid),
            'stagger': stagger,
            'mirror_x': mirror_x,
            'mirror_y': mirror_y,
            'rotation': rotation,
            'boundary_mode': boundary_mode,
            'boundary_missing': missing,
            'n_extensions': len(extensions),
            'n_intersections': n_isect,
            'seed_hash': seed_hash,
        },
    )


def check_connectivity(net):
    """Check number of connected components."""
    adj = defaultdict(set)
    for cl in net.crosslinks:
        adj[cl.fiber_i].add(cl.fiber_j)
        adj[cl.fiber_j].add(cl.fiber_i)
    visited, comps = set(), 0
    for s in range(net.num_fibers):
        if s not in visited:
            comps += 1
            q = [s]
            while q:
                n = q.pop(0)
                if n in visited:
                    continue
                visited.add(n)
                q.extend(adj[n] - visited)
    return comps


if __name__ == "__main__":
    print("=" * 60)
    print("Pattern Engine v7 — Unified API")
    print("=" * 60)

    print("\n1. POLYGON PRESETS (all connected):")
    for shape in ['square', 'triangle', 'hexagon']:
        for n in [5, 8]:
            for pert in [0.0, 0.2]:
                net = pattern_2d(polygon_type=shape, n_pts_per_side=n,
                                perturbation=pert, seed=42, grid=(3,3))
                c = check_connectivity(net)
                print(f"  {shape:<10} n={n:>2} pert={pert:.1f}: {net.num_fibers:>5}F {net.num_crosslinks:>5}CL {'✅' if c==1 else f'❌ {c}c'}")

    print("\n2. CUSTOM SHAPES (boundary-touching):")
    shapes = [
        ('Diamond', [(5,0),(10,5),(5,10),(0,5)], True),
        ('Cross', [(3,0),(7,0),(7,3),(10,3),(10,7),(7,7),(7,10),(3,10),(3,7),(0,7),(0,3),(3,3)], True),
        ('L-shape', [(0,0),(10,0),(10,3),(3,3),(3,10),(0,10)], True),
        ('H-shape', [(0,0),(3,0),(3,3.5),(7,3.5),(7,0),(10,0),(10,10),(7,10),(7,6.5),(3,6.5),(3,10),(0,10)], True),
    ]
    for name, pts, closed in shapes:
        net = pattern_2d(box=(10,10), points=pts, closed=closed, grid=(3,3),
                        mirror_x=True, mirror_y=True)
        c = check_connectivity(net)
        print(f"  {name:<12}: {net.num_fibers:>5}F {net.num_crosslinks:>5}CL {'✅' if c==1 else f'❌ {c}c'}")

    print("\n3. CUSTOM + fit_to_box (arbitrary coords → box):")
    arbitrary = [
        ('Star_arb', [(50,0),(65,35),(100,50),(65,65),(50,100),(35,65),(0,50),(35,35)], True),
        ('Arrow_arb', [(0,40),(60,40),(60,0),(100,50),(60,100),(60,60),(0,60)], True),
    ]
    for name, pts, closed in arbitrary:
        net = pattern_2d(box=(10,10), points=pts, closed=closed, fit_to_box=True,
                        grid=(3,3), mirror_x=True, mirror_y=True)
        c = check_connectivity(net)
        print(f"  {name:<12}: {net.num_fibers:>5}F {net.num_crosslinks:>5}CL {'✅' if c==1 else f'❌ {c}c'}")

    print("\n4. BOUNDARY MODE TESTS:")
    pts_interior = [(3,3),(7,3),(7,7),(3,7)]

    # Error mode
    try:
        net = pattern_2d(box=(10,10), points=pts_interior, closed=True, grid=(3,3),
                        boundary_mode='error')
        print(f"  Interior shape: no error (unexpected)")
    except ValueError as e:
        print(f"  Interior shape → error: ✅ ({str(e)[:60]}...)")

    # Extend mode
    net = pattern_2d(box=(10,10), points=pts_interior, closed=True, grid=(3,3),
                    boundary_mode='extend')
    c = check_connectivity(net)
    print(f"  Interior + extend: {net.num_fibers:>5}F {net.num_crosslinks:>5}CL {'✅' if c==1 else f'❌ {c}c'}")

    # None mode (default)
    net = pattern_2d(box=(10,10), points=pts_interior, closed=True, grid=(3,3),
                    boundary_mode='none')
    c = check_connectivity(net)
    print(f"  Interior + none:   {net.num_fibers:>5}F {net.num_crosslinks:>5}CL {'✅' if c==1 else f'❌ {c}c'}")

    print("\n5. GRID SIZE VARIATIONS:")
    for gs in [2, 3, 4, 5]:
        net = pattern_2d(polygon_type='square', n_pts_per_side=5, perturbation=0.2,
                        seed=42, grid=gs)
        c = check_connectivity(net)
        print(f"  grid={gs}×{gs}: {net.num_fibers:>5}F {net.num_crosslinks:>5}CL {'✅' if c==1 else f'❌ {c}c'}")

    print("\n" + "=" * 60)
