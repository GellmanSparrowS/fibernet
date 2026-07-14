"""
Pattern Engine — Unified structure generation for FiberNet.

The Pattern Engine implements a single, coherent paradigm for generating
all periodic 2D/3D structures:

    **Base Unit + Transform + Tiling + Welding**

Design Principles
-----------------
- **Deterministic**: No randomness unless explicitly set via ``seed``.
- **Programmable**: Full control over positions, transforms, boundary behavior.
- **Unified API**: ``pattern_2d()`` and ``pattern_3d()`` for all structures.
- **Edge discretization**: Every edge can carry N internal points for deformation.
- **Intermediate points**: Each polygon edge gets ``n_pts_per_side`` graph nodes
  with individually programmable (dx, dy) displacements.
- **Boundary-aware**: Nodes on cell boundaries are correctly welded during tiling.

Intermediate Point Programmability
-----------------------------------
Each polygon edge can have ``n_pts_per_side`` intermediate graph nodes inserted.
Each intermediate node has a (dx, dy) displacement that is:
- Explicitly set via ``point_displacements`` list, OR
- Auto-generated deterministically from ``seed`` (default non-zero), OR
- Applied via Cn-symmetric perturbation preserving rotational symmetry.

This gives full control over beam geometry — straight, curved, wavy, or custom.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from fibernet.core.structure_graph import StructureGraph
from fibernet.core.material import Material
from fibernet.core.transforms import (
    translate, rotate, mirror_x as _mirror_x, mirror_y as _mirror_y,
    mirror as _mirror, scale as _scale, compose,
)
from fibernet.core.tiling import tile_2d, tile_3d, fit_unit_to_box


# ======================================================================
# Intermediate point helpers
# ======================================================================

def _interpolate_edge_points(
    p1: Sequence[float],
    p2: Sequence[float],
    n_pts: int,
) -> List[Tuple[float, ...]]:
    """Generate n_pts equally-spaced intermediate points between p1 and p2.

    Returns list of n_pts points (does NOT include p1 or p2).
    """
    if n_pts <= 0:
        return []
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    pts = []
    for i in range(1, n_pts + 1):
        t = i / (n_pts + 1)
        pt = p1 + t * (p2 - p1)
        pts.append(tuple(pt.tolist()))
    return pts


def _apply_displacements(
    pts: List[Tuple[float, ...]],
    displacements: Sequence[Tuple[float, ...]],
) -> List[Tuple[float, ...]]:
    """Apply (dx, dy) displacements to a list of points.

    Parameters
    ----------
    pts : list of (x, y) or (x, y, z)
        Points to displace.
    displacements : list of (dx, dy)
        Displacements. Length must match pts.
    """
    if len(pts) != len(displacements):
        raise ValueError(
            f"displacements length ({len(displacements)}) != pts length ({len(pts)})"
        )
    result = []
    for pt, dp in zip(pts, displacements):
        x = pt[0] + dp[0]
        y = pt[1] + dp[1]
        if len(pt) > 2:
            result.append((x, y, pt[2]))
        else:
            result.append((x, y))
    return result


def _generate_polygon_perimeter(
    box: Tuple[float, float],
    polygon_type: str,
    n_pts_per_side: int = 0,
    point_displacements: Optional[Sequence[Tuple[float, float]]] = None,
    perturbation: float = 0.0,
    seed: Optional[int] = None,
) -> List[Tuple[float, float]]:
    """Generate polygon perimeter with intermediate points per edge.

    Parameters
    ----------
    box : (w, h)
        Cell dimensions.
    polygon_type : str
        'square', 'triangle', or 'hexagon'.
    n_pts_per_side : int
        Number of intermediate nodes per edge (0 = corners only).
    point_displacements : list of (dx, dy), optional
        Explicit displacement for each intermediate point (flat list).
        Total length = n_sides * n_pts_per_side.
    perturbation : float
        Cn symmetric perturbation magnitude (fraction of edge length).
    seed : int, optional
        Random seed for auto-generated displacements.
        If n_pts_per_side > 0 and no displacements given,
        defaults to seed=0 with magnitude=0.3*edge_length.

    Returns
    -------
    list of (x, y)
        Ordered perimeter points (corners + intermediate, closed loop).
    """
    w, h = box

    if polygon_type == 'square':
        corners = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
        cn_order = 4
    elif polygon_type == 'triangle':
        corners = [(w / 2, h), (0.0, 0.0), (w, 0.0)]
        cn_order = 3
    elif polygon_type == 'hexagon':
        corners = [
            (w / 2, 0), (w, h / 4), (w, 3 * h / 4),
            (w / 2, h), (0, 3 * h / 4), (0, h / 4),
        ]
        cn_order = 6
    else:
        raise ValueError(f"Unknown polygon_type: {polygon_type}")

    n_sides = len(corners)

    # Build perimeter: corners + intermediate points per edge
    perimeter = []
    for i in range(n_sides):
        p1 = corners[i]
        p2 = corners[(i + 1) % n_sides]
        if i == 0:
            perimeter.append(p1)
        # Add intermediate points
        if n_pts_per_side > 0:
            intermediates = _interpolate_edge_points(p1, p2, n_pts_per_side)
            perimeter.extend(intermediates)
        perimeter.append(p2)

    # Apply displacements to intermediate points only (not corners)
    n_intermediate = n_sides * n_pts_per_side
    if n_intermediate > 0:
        # Determine displacements
        if point_displacements is not None:
            # Explicit displacements
            if len(point_displacements) != n_intermediate:
                raise ValueError(
                    f"Expected {n_intermediate} displacements "
                    f"({n_sides} sides × {n_pts_per_side} pts), "
                    f"got {len(point_displacements)}"
                )
            disp = list(point_displacements)
        elif perturbation > 0:
            # Cn symmetric perturbation
            effective_seed = seed if seed is not None else 0
            disp = _cn_symmetric_displacements(
                perimeter, polygon_type, n_pts_per_side, perturbation, effective_seed,
            )
        elif seed is not None:
            # Auto-generate deterministic displacements
            edge_len = np.sqrt((corners[1][0] - corners[0][0])**2 +
                             (corners[1][1] - corners[0][1])**2)
            magnitude = 0.3 * edge_len  # 5% of edge length
            disp = _cn_symmetric_displacements(
                perimeter, polygon_type, n_pts_per_side, magnitude, seed,
            )
        else:
            # Default: seed=0, 5% edge length
            edge_len = np.sqrt((corners[1][0] - corners[0][0])**2 +
                             (corners[1][1] - corners[0][1])**2)
            magnitude = 0.3 * edge_len
            disp = _cn_symmetric_displacements(
                perimeter, polygon_type, n_pts_per_side, magnitude, 0,
            )

        # Find intermediate point indices in perimeter
        intermediate_indices = []
        idx = 1  # Skip first corner
        for s in range(n_sides):
            for k in range(n_pts_per_side):
                intermediate_indices.append(idx)
                idx += 1
            idx += 1  # Skip next corner

        # Apply displacements
        for i, pidx in enumerate(intermediate_indices):
            if pidx < len(perimeter) and i < len(disp):
                x, y = perimeter[pidx]
                perimeter[pidx] = (x + disp[i][0], y + disp[i][1])

    return perimeter


def _cn_symmetric_displacements(
    perimeter: List[Tuple[float, float]],
    polygon_type: str,
    n_pts_per_side: int,
    magnitude: float,
    seed: int,
) -> List[Tuple[float, float]]:
    """Generate Cn-symmetric displacements: one (dx,dy) rotated to all N sides.

    This preserves the rotational symmetry of the polygon.
    One random displacement is generated per intermediate position k,
    then rotated to all N sides.

    Parameters
    ----------
    perimeter : list of (x, y)
        Full perimeter points.
    polygon_type : str
        'square' (C4), 'triangle' (C3), 'hexagon' (C6).
    n_pts_per_side : int
        Number of intermediate points per edge.
    magnitude : float
        Maximum displacement magnitude.
    seed : int
        Random seed.

    Returns
    -------
    list of (dx, dy)
        Flat list of displacements, one per intermediate point.
    """
    rng = np.random.default_rng(seed)
    n_sides = {'square': 4, 'triangle': 3, 'hexagon': 6}.get(polygon_type, 4)

    # Build side indices: which intermediate indices belong to each side
    side_indices = []
    idx = 1  # Skip first corner
    for s in range(n_sides):
        side = list(range(idx, idx + n_pts_per_side))
        side_indices.append(side)
        idx += n_pts_per_side + 1  # Skip next corner

    # Generate one (dx, dy) per intermediate position, rotated to all sides
    result = [(0.0, 0.0)] * (n_sides * n_pts_per_side)
    flat_idx = 0
    for k in range(n_pts_per_side):
        dx = rng.uniform(-magnitude, magnitude)
        dy = rng.uniform(-magnitude, magnitude)
        # Ensure non-zero displacement
        if abs(dx) < 1e-10 and abs(dy) < 1e-10:
            dx = magnitude * 0.5
            dy = magnitude * 0.3
        for s in range(n_sides):
            angle = 2 * np.pi * s / n_sides
            rdx = dx * np.cos(angle) - dy * np.sin(angle)
            rdy = dx * np.sin(angle) + dy * np.cos(angle)
            result[s * n_pts_per_side + k] = (rdx, rdy)

    return result


def _displace_intermediate_nodes(
    graph: StructureGraph,
    edge_node_pairs: List[Tuple[int, int]],
    n_pts_per_side: int,
    point_displacements: Optional[Sequence[Tuple[float, float]]] = None,
    perturbation: float = 0.0,
    seed: Optional[int] = None,
) -> None:
    """Apply displacements to intermediate nodes on edges in-place.

    Parameters
    ----------
    graph : StructureGraph
        The graph to modify.
    edge_node_pairs : list of (node_i, node_j)
        Edge endpoints defining which edges have intermediate nodes.
    n_pts_per_side : int
        Number of intermediate nodes per edge.
    point_displacements : list of (dx, dy), optional
        Explicit displacements (flat list).
    perturbation : float
        Auto-generated perturbation magnitude.
    seed : int, optional
        Random seed for auto-generated displacements.
    """
    if n_pts_per_side <= 0:
        return

    n_edges = len(edge_node_pairs)
    n_total = n_edges * n_pts_per_side

    # Determine displacements
    if point_displacements is not None:
        if len(point_displacements) != n_total:
            raise ValueError(
                f"Expected {n_total} displacements ({n_edges} edges × {n_pts_per_side} pts), "
                f"got {len(point_displacements)}"
            )
        disp = list(point_displacements)
    else:
        # Auto-generate
        effective_seed = seed if seed is not None else 0
        rng = np.random.default_rng(effective_seed)
        # Compute mean edge length for magnitude
        edge_lens = []
        for ni, nj in edge_node_pairs:
            pi = graph.nodes[ni].position
            pj = graph.nodes[nj].position
            edge_lens.append(np.linalg.norm(pj - pi))
        mean_len = np.mean(edge_lens) if edge_lens else 1.0
        magnitude = perturbation * mean_len if perturbation > 0 else 0.05 * mean_len
        disp = []
        for _ in range(n_total):
            dx = rng.uniform(-magnitude, magnitude)
            dy = rng.uniform(-magnitude, magnitude)
            if abs(dx) < 1e-10 and abs(dy) < 1e-10:
                dx = magnitude * 0.5
                dy = magnitude * 0.3
            disp.append((dx, dy))

    # Apply displacements to intermediate nodes
    # Find intermediate nodes by checking which nodes are not endpoints
    endpoint_ids = set()
    for ni, nj in edge_node_pairs:
        endpoint_ids.add(ni)
        endpoint_ids.add(nj)

    disp_idx = 0
    for ni, nj in edge_node_pairs:
        pi = graph.nodes[ni].position[:2]
        pj = graph.nodes[nj].position[:2]
        for k in range(n_pts_per_side):
            t = (k + 1) / (n_pts_per_side + 1)
            target = pi + t * (pj - pi)
            # Find the node closest to target
            for nid, node in graph.nodes.items():
                if nid in endpoint_ids:
                    continue
                if np.linalg.norm(node.position[:2] - target) < graph.tolerance * 10:
                    node.position[0] += disp[disp_idx][0]
                    node.position[1] += disp[disp_idx][1]
                    break
            disp_idx += 1


def _add_edge_with_intermediates(
    graph: StructureGraph,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    n_pts_per_side: int = 0,
    displacements: Optional[List[Tuple[float, float]]] = None,
    radius: float = 0.1,
    material: Material = None,
    n_internal: int = 0,
) -> List[int]:
    """Add a line segment with intermediate graph nodes.

    Creates n_pts_per_side intermediate nodes between p1 and p2,
    optionally displaced, and connects them as a chain of edges.

    Parameters
    ----------
    graph : StructureGraph
        Graph to add to.
    p1, p2 : (x, y)
        Endpoints.
    n_pts_per_side : int
        Number of intermediate nodes (0 = direct edge).
    displacements : list of (dx, dy), optional
        Displacement for each intermediate node.
    radius : float
        Beam radius.
    material : Material, optional
        Beam material.
    n_internal : int
        Internal points per edge (for deformation visualization).

    Returns
    -------
    list of int
        Node IDs of the chain (including endpoints).
    """
    n1 = graph.add_node(p1)
    chain = [n1]

    if n_pts_per_side > 0:
        p1_arr = np.array(p1[:2], dtype=float)
        p2_arr = np.array(p2[:2], dtype=float)
        for k in range(n_pts_per_side):
            t = (k + 1) / (n_pts_per_side + 1)
            pt = p1_arr + t * (p2_arr - p1_arr)
            if displacements and k < len(displacements):
                pt[0] += displacements[k][0]
                pt[1] += displacements[k][1]
            nid = graph.add_node(pt.tolist())
            chain.append(nid)

    n2 = graph.add_node(p2)
    chain.append(n2)

    # Connect chain
    for i in range(len(chain) - 1):
        graph.add_edge(chain[i], chain[i + 1], radius=radius,
                      material=material, n_internal=n_internal)

    return chain


def _auto_displacements(n_total: int, magnitude: float, seed: int = 0) -> List[Tuple[float, float]]:
    """Generate deterministic non-zero displacements.

    Parameters
    ----------
    n_total : int
        Number of displacements to generate.
    magnitude : float
        Maximum displacement magnitude.
    seed : int
        Random seed.

    Returns
    -------
    list of (dx, dy)
    """
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(n_total):
        dx = rng.uniform(-magnitude, magnitude)
        dy = rng.uniform(-magnitude, magnitude)
        if abs(dx) < 1e-10 and abs(dy) < 1e-10:
            dx = magnitude * 0.5
            dy = magnitude * 0.3
        result.append((float(dx), float(dy)))
    return result


# ======================================================================
# Built-in unit factories
# ======================================================================

def _unit_square(
    box: Tuple[float, float],
    n_internal: int = 0,
    radius: float = 0.1,
    material: Material = None,
    n_pts_per_side: int = 0,
    point_displacements: Optional[Sequence[Tuple[float, float]]] = None,
    perturbation: float = 0.0,
    seed: Optional[int] = None,
) -> StructureGraph:
    """Square frame unit cell with programmable intermediate points."""
    w, h = box
    pts = _generate_polygon_perimeter(
        box, 'square', n_pts_per_side, point_displacements, perturbation, seed,
    )
    g = StructureGraph(dimension=2, box_size=[w, h])
    nids = [g.add_node(p) for p in pts]
    for i in range(len(nids)):
        g.add_edge(nids[i], nids[(i + 1) % len(nids)],
                  radius=radius, material=material, n_internal=n_internal)
    g._metadata["unit_type"] = "square"
    g._metadata["n_pts_per_side"] = n_pts_per_side
    return g


def _unit_triangle(
    box: Tuple[float, float],
    n_internal: int = 0,
    radius: float = 0.1,
    material: Material = None,
    n_pts_per_side: int = 0,
    point_displacements: Optional[Sequence[Tuple[float, float]]] = None,
    perturbation: float = 0.0,
    seed: Optional[int] = None,
) -> StructureGraph:
    """Triangular lattice unit cell — rhombus of two triangles."""
    w, h = box
    g = StructureGraph(dimension=2, box_size=[w, h])

    # Four corners
    n00 = g.add_node([0, 0])
    n10 = g.add_node([w, 0])
    n01 = g.add_node([0, h])
    n11 = g.add_node([w, h])

    # 5 edges: 4 boundary + 1 diagonal
    edge_pairs = [(n00, n10), (n10, n01), (n01, n00), (n10, n11), (n11, n01)]

    # Generate displacements for all edges
    n_total_pts = len(edge_pairs) * n_pts_per_side
    if n_pts_per_side > 0:
        if point_displacements is not None:
            all_disp = list(point_displacements)
        else:
            edge_len = w  # approximate
            mag = perturbation * edge_len if perturbation > 0 else 0.3 * edge_len
            effective_seed = seed if seed is not None else 0
            all_disp = _auto_displacements(n_total_pts, mag, effective_seed)
    else:
        all_disp = []

    for idx, (ni, nj) in enumerate(edge_pairs):
        start = idx * n_pts_per_side
        end = start + n_pts_per_side
        edge_disp = all_disp[start:end] if n_pts_per_side > 0 else None

        pi = g.nodes[ni].position[:2]
        pj = g.nodes[nj].position[:2]
        _add_edge_with_intermediates(
            g, tuple(pi), tuple(pj), n_pts_per_side, edge_disp,
            radius, material, n_internal,
        )

    g._metadata["unit_type"] = "triangle"
    g._metadata["n_pts_per_side"] = n_pts_per_side
    return g


def _unit_hexagon(
    box: Tuple[float, float],
    n_internal: int = 0,
    radius: float = 0.1,
    material: Material = None,
    n_pts_per_side: int = 0,
    point_displacements: Optional[Sequence[Tuple[float, float]]] = None,
    perturbation: float = 0.0,
    seed: Optional[int] = None,
) -> StructureGraph:
    """Regular hexagon unit cell with programmable intermediate points."""
    w, h = box
    pts = _generate_polygon_perimeter(
        box, 'hexagon', n_pts_per_side, point_displacements, perturbation, seed,
    )
    g = StructureGraph(dimension=2, box_size=[w, h])
    nids = [g.add_node(p) for p in pts]
    for i in range(len(nids)):
        g.add_edge(nids[i], nids[(i + 1) % len(nids)],
                  radius=radius, material=material, n_internal=n_internal)
    g._metadata["unit_type"] = "hexagon"
    g._metadata["n_pts_per_side"] = n_pts_per_side
    return g


def _unit_honeycomb(
    box: Tuple[float, float],
    n_internal: int = 0,
    radius: float = 0.1,
    material: Material = None,
    n_pts_per_side: int = 0,
    point_displacements: Optional[Sequence[Tuple[float, float]]] = None,
    perturbation: float = 0.0,
    seed: Optional[int] = None,
) -> StructureGraph:
    """Honeycomb unit cell with programmable intermediate points on each segment."""
    w, h = box
    g = StructureGraph(dimension=2, box_size=[w, h])

    a = w / 2
    b = h / 4

    # 6 segments of the honeycomb cell
    segments = [
        ((0, b), (0, 3 * b)),        # left vertical
        ((w, b), (w, 3 * b)),        # right vertical
        ((0, b), (a, 0)),            # top-left diagonal
        ((w, b), (a, 0)),            # top-right diagonal
        ((0, 3 * b), (a, h)),        # bottom-left diagonal
        ((w, 3 * b), (a, h)),        # bottom-right diagonal
    ]

    n_total_pts = len(segments) * n_pts_per_side
    if n_pts_per_side > 0:
        if point_displacements is not None:
            all_disp = list(point_displacements)
        else:
            edge_len = np.sqrt((a)**2 + b**2)  # diagonal length
            mag = perturbation * edge_len if perturbation > 0 else 0.3 * edge_len
            effective_seed = seed if seed is not None else 0
            all_disp = _auto_displacements(n_total_pts, mag, effective_seed)
    else:
        all_disp = []

    for idx, (p1, p2) in enumerate(segments):
        start = idx * n_pts_per_side
        end = start + n_pts_per_side
        edge_disp = all_disp[start:end] if n_pts_per_side > 0 else None
        _add_edge_with_intermediates(
            g, p1, p2, n_pts_per_side, edge_disp,
            radius, material, n_internal,
        )

    g._metadata["unit_type"] = "honeycomb"
    g._metadata["n_pts_per_side"] = n_pts_per_side
    return g


def _unit_kagome(
    box: Tuple[float, float],
    n_internal: int = 0,
    radius: float = 0.1,
    material: Material = None,
    n_pts_per_side: int = 0,
    point_displacements: Optional[Sequence[Tuple[float, float]]] = None,
    perturbation: float = 0.0,
    seed: Optional[int] = None,
) -> StructureGraph:
    """Kagome lattice with programmable intermediate points."""
    w, h = box
    g = StructureGraph(dimension=2, box_size=[w, h])

    # Corner nodes
    n00 = g.add_node([0, 0])
    n10 = g.add_node([w, 0])
    n01 = g.add_node([0, h])
    n11 = g.add_node([w, h])
    # Edge midpoints
    nm_b = g.add_node([w / 2, 0])
    nm_t = g.add_node([w / 2, h])
    nm_l = g.add_node([0, h / 2])
    nm_r = g.add_node([w, h / 2])
    # Center
    nc = g.add_node([w / 2, h / 2])

    # 12 edges
    edge_node_pairs = [
        (n00, nm_b), (n10, nm_b),
        (n00, nm_l), (n01, nm_l),
        (n10, nm_r), (n11, nm_r),
        (n01, nm_t), (n11, nm_t),
        (nm_b, nc), (nm_t, nc), (nm_l, nc), (nm_r, nc),
    ]

    n_total_pts = len(edge_node_pairs) * n_pts_per_side
    if n_pts_per_side > 0:
        if point_displacements is not None:
            all_disp = list(point_displacements)
        else:
            edge_len = w / 2
            mag = perturbation * edge_len if perturbation > 0 else 0.3 * edge_len
            effective_seed = seed if seed is not None else 0
            all_disp = _auto_displacements(n_total_pts, mag, effective_seed)
    else:
        all_disp = []

    for idx, (ni, nj) in enumerate(edge_node_pairs):
        start = idx * n_pts_per_side
        end = start + n_pts_per_side
        edge_disp = all_disp[start:end] if n_pts_per_side > 0 else None
        pi = g.nodes[ni].position[:2]
        pj = g.nodes[nj].position[:2]
        _add_edge_with_intermediates(
            g, tuple(pi), tuple(pj), n_pts_per_side, edge_disp,
            radius, material, n_internal,
        )

    g._metadata["unit_type"] = "kagome"
    g._metadata["n_pts_per_side"] = n_pts_per_side
    return g


def _unit_reentrant(
    box: Tuple[float, float],
    n_internal: int = 0,
    radius: float = 0.1,
    material: Material = None,
    angle: float = 15.0,
    n_pts_per_side: int = 0,
    point_displacements: Optional[Sequence[Tuple[float, float]]] = None,
    perturbation: float = 0.0,
    seed: Optional[int] = None,
) -> StructureGraph:
    """Reentrant (arrowhead) honeycomb — auxetic structure."""
    w, h = box
    g = StructureGraph(dimension=2, box_size=[w, h])

    b = h / 4

    # Boundary nodes
    n_l_lo = g.add_node([0, b])
    n_l_hi = g.add_node([0, 3 * b])
    n_r_lo = g.add_node([w, b])
    n_r_hi = g.add_node([w, 3 * b])
    n_b = g.add_node([w / 2, 0])
    n_t = g.add_node([w / 2, h])

    # Internal indentation nodes
    indent = (w / 2) * np.tan(np.radians(min(angle, 40)))
    n_indent_lo = g.add_node([w / 2, b + indent])
    n_indent_hi = g.add_node([w / 2, 3 * b - indent])

    # 8 edges
    edge_pairs = [
        (n_l_lo, n_l_hi),       # left wall
        (n_r_lo, n_r_hi),       # right wall
        (n_l_lo, n_indent_lo),  # lower-left diagonal
        (n_r_lo, n_indent_lo),  # lower-right diagonal
        (n_indent_lo, n_b),     # indent to bottom
        (n_l_hi, n_indent_hi),  # upper-left diagonal
        (n_r_hi, n_indent_hi),  # upper-right diagonal
        (n_indent_hi, n_t),     # indent to top
    ]

    n_total_pts = len(edge_pairs) * n_pts_per_side
    if n_pts_per_side > 0:
        if point_displacements is not None:
            all_disp = list(point_displacements)
        else:
            edge_len = h / 2
            mag = perturbation * edge_len if perturbation > 0 else 0.3 * edge_len
            effective_seed = seed if seed is not None else 0
            all_disp = _auto_displacements(n_total_pts, mag, effective_seed)
    else:
        all_disp = []

    for idx, (ni, nj) in enumerate(edge_pairs):
        start = idx * n_pts_per_side
        end = start + n_pts_per_side
        edge_disp = all_disp[start:end] if n_pts_per_side > 0 else None
        pi = g.nodes[ni].position[:2]
        pj = g.nodes[nj].position[:2]
        _add_edge_with_intermediates(
            g, tuple(pi), tuple(pj), n_pts_per_side, edge_disp,
            radius, material, n_internal,
        )

    g._metadata["unit_type"] = "reentrant"
    g._metadata["reentrant_angle"] = angle
    g._metadata["n_pts_per_side"] = n_pts_per_side
    return g


def _unit_chiral(
    box: Tuple[float, float],
    n_internal: int = 0,
    radius: float = 0.1,
    material: Material = None,
    node_radius: float = None,
    n_pts_per_side: int = 0,
    point_displacements: Optional[Sequence[Tuple[float, float]]] = None,
    perturbation: float = 0.0,
    seed: Optional[int] = None,
) -> StructureGraph:
    """Chiral honeycomb with programmable intermediate points."""
    w, h = box
    g = StructureGraph(dimension=2, box_size=[w, h])

    if node_radius is None:
        node_radius = min(w, h) * 0.15

    cx, cy = w / 2, h / 2
    n_ring_pts = 8
    ring_nodes = []
    for i in range(n_ring_pts):
        theta = 2 * np.pi * i / n_ring_pts
        px = cx + node_radius * np.cos(theta)
        py = cy + node_radius * np.sin(theta)
        nid = g.add_node([px, py])
        ring_nodes.append(nid)

    # Collect all edge pairs
    edge_pairs = []
    # Ring edges
    for i in range(n_ring_pts):
        edge_pairs.append((ring_nodes[i], ring_nodes[(i + 1) % n_ring_pts]))

    # Corner nodes and tangent ligaments
    corners = [(0, 0), (w, 0), (w, h), (0, h)]
    corner_nodes = []
    for i, (cx_c, cy_c) in enumerate(corners):
        cn = g.add_node([cx_c, cy_c])
        corner_nodes.append(cn)
        ring_idx = (i * n_ring_pts) // 4
        rn = ring_nodes[ring_idx]
        rn2 = ring_nodes[(ring_idx + 1) % n_ring_pts]
        edge_pairs.append((cn, rn))
        edge_pairs.append((cn, rn2))

    n_total_pts = len(edge_pairs) * n_pts_per_side
    if n_pts_per_side > 0:
        if point_displacements is not None:
            all_disp = list(point_displacements)
        else:
            edge_len = node_radius
            mag = perturbation * edge_len if perturbation > 0 else 0.3 * edge_len
            effective_seed = seed if seed is not None else 0
            all_disp = _auto_displacements(n_total_pts, mag, effective_seed)
    else:
        all_disp = []

    for idx, (ni, nj) in enumerate(edge_pairs):
        start = idx * n_pts_per_side
        end = start + n_pts_per_side
        edge_disp = all_disp[start:end] if n_pts_per_side > 0 else None
        pi = g.nodes[ni].position[:2]
        pj = g.nodes[nj].position[:2]
        _add_edge_with_intermediates(
            g, tuple(pi), tuple(pj), n_pts_per_side, edge_disp,
            radius, material, n_internal,
        )

    g._metadata["unit_type"] = "chiral"
    g._metadata["n_pts_per_side"] = n_pts_per_side
    return g


def _unit_star(
    box: Tuple[float, float],
    n_internal: int = 0,
    radius: float = 0.1,
    material: Material = None,
    n_arms: int = 4,
    n_pts_per_side: int = 0,
    point_displacements: Optional[Sequence[Tuple[float, float]]] = None,
    perturbation: float = 0.0,
    seed: Optional[int] = None,
) -> StructureGraph:
    """Star-shaped unit cell with programmable intermediate points."""
    w, h = box
    g = StructureGraph(dimension=2, box_size=[w, h])
    cx, cy = w / 2, h / 2
    r_outer = min(w, h) / 2
    r_inner = r_outer * 0.4

    corners = []
    for i in range(2 * n_arms):
        theta = np.pi * i / n_arms - np.pi / 2
        r = r_outer if i % 2 == 0 else r_inner
        px = cx + r * np.cos(theta)
        py = cy + r * np.sin(theta)
        corners.append((px, py))

    # Build perimeter with intermediate points
    n_sides = len(corners)
    perimeter = []
    for i in range(n_sides):
        p1 = corners[i]
        p2 = corners[(i + 1) % n_sides]
        if i == 0:
            perimeter.append(p1)
        if n_pts_per_side > 0:
            intermediates = _interpolate_edge_points(p1, p2, n_pts_per_side)
            perimeter.extend(intermediates)
        perimeter.append(p2)

    # Apply displacements
    n_intermediate = n_sides * n_pts_per_side
    if n_intermediate > 0:
        if point_displacements is not None:
            all_disp = list(point_displacements)
        else:
            edge_len = r_outer - r_inner
            mag = perturbation * edge_len if perturbation > 0 else 0.3 * edge_len
            effective_seed = seed if seed is not None else 0
            all_disp = _auto_displacements(n_intermediate, mag, effective_seed)

        idx = 1
        disp_idx = 0
        for s in range(n_sides):
            for k in range(n_pts_per_side):
                if idx < len(perimeter) and disp_idx < len(all_disp):
                    x, y = perimeter[idx]
                    perimeter[idx] = (x + all_disp[disp_idx][0], y + all_disp[disp_idx][1])
                idx += 1
                disp_idx += 1
            idx += 1

    nids = [g.add_node(p) for p in perimeter]
    for i in range(len(nids)):
        g.add_edge(nids[i], nids[(i + 1) % len(nids)],
                  radius=radius, material=material, n_internal=n_internal)

    g._metadata["unit_type"] = "star"
    g._metadata["n_arms"] = n_arms
    g._metadata["n_pts_per_side"] = n_pts_per_side
    return g


def _unit_cross(
    box: Tuple[float, float],
    n_internal: int = 0,
    radius: float = 0.1,
    material: Material = None,
    arm_width: float = 0.3,
    n_pts_per_side: int = 0,
    point_displacements: Optional[Sequence[Tuple[float, float]]] = None,
    perturbation: float = 0.0,
    seed: Optional[int] = None,
) -> StructureGraph:
    """Plus/cross shape with programmable intermediate points."""
    w, h = box
    g = StructureGraph(dimension=2, box_size=[w, h])
    aw = arm_width * min(w, h) / 2
    cx, cy = w / 2, h / 2

    corners = [
        (cx - aw, 0), (cx + aw, 0),
        (cx + aw, cy - aw), (w, cy - aw),
        (w, cy + aw), (cx + aw, cy + aw),
        (cx + aw, h), (cx - aw, h),
        (cx - aw, cy + aw), (0, cy + aw),
        (0, cy - aw), (cx - aw, cy - aw),
    ]

    # Build perimeter with intermediate points
    n_sides = len(corners)
    perimeter = []
    for i in range(n_sides):
        p1 = corners[i]
        p2 = corners[(i + 1) % n_sides]
        if i == 0:
            perimeter.append(p1)
        if n_pts_per_side > 0:
            intermediates = _interpolate_edge_points(p1, p2, n_pts_per_side)
            perimeter.extend(intermediates)
        perimeter.append(p2)

    n_intermediate = n_sides * n_pts_per_side
    if n_intermediate > 0:
        if point_displacements is not None:
            all_disp = list(point_displacements)
        else:
            edge_len = aw
            mag = perturbation * edge_len if perturbation > 0 else 0.3 * edge_len
            effective_seed = seed if seed is not None else 0
            all_disp = _auto_displacements(n_intermediate, mag, effective_seed)

        idx = 1
        disp_idx = 0
        for s in range(n_sides):
            for k in range(n_pts_per_side):
                if idx < len(perimeter) and disp_idx < len(all_disp):
                    x, y = perimeter[idx]
                    perimeter[idx] = (x + all_disp[disp_idx][0], y + all_disp[disp_idx][1])
                idx += 1
                disp_idx += 1
            idx += 1

    nids = [g.add_node(p) for p in perimeter]
    for i in range(len(nids)):
        g.add_edge(nids[i], nids[(i + 1) % len(nids)],
                  radius=radius, material=material, n_internal=n_internal)

    g._metadata["unit_type"] = "cross"
    g._metadata["n_pts_per_side"] = n_pts_per_side
    return g


def _unit_missing_rib(
    box: Tuple[float, float],
    n_internal: int = 0,
    radius: float = 0.1,
    material: Material = None,
    n_pts_per_side: int = 0,
    point_displacements: Optional[Sequence[Tuple[float, float]]] = None,
    perturbation: float = 0.0,
    seed: Optional[int] = None,
) -> StructureGraph:
    """Missing-rib honeycomb with programmable intermediate points."""
    w, h = box
    g = StructureGraph(dimension=2, box_size=[w, h])
    a = w / 2
    b = h / 4

    # 5 segments (left vertical removed)
    segments = [
        ((w, b), (w, 3 * b)),       # right vertical
        ((0, b), (a, 0)),            # top-left diagonal
        ((w, b), (a, 0)),            # top-right diagonal
        ((0, 3 * b), (a, h)),        # bottom-left diagonal
        ((w, 3 * b), (a, h)),        # bottom-right diagonal
    ]

    n_total_pts = len(segments) * n_pts_per_side
    if n_pts_per_side > 0:
        if point_displacements is not None:
            all_disp = list(point_displacements)
        else:
            edge_len = np.sqrt(a**2 + b**2)
            mag = perturbation * edge_len if perturbation > 0 else 0.3 * edge_len
            effective_seed = seed if seed is not None else 0
            all_disp = _auto_displacements(n_total_pts, mag, effective_seed)
    else:
        all_disp = []

    for idx, (p1, p2) in enumerate(segments):
        start = idx * n_pts_per_side
        end = start + n_pts_per_side
        edge_disp = all_disp[start:end] if n_pts_per_side > 0 else None
        _add_edge_with_intermediates(
            g, p1, p2, n_pts_per_side, edge_disp,
            radius, material, n_internal,
        )

    g._metadata["unit_type"] = "missing_rib"
    g._metadata["n_pts_per_side"] = n_pts_per_side
    return g


def _unit_diamond(
    box: Tuple[float, float],
    n_internal: int = 0,
    radius: float = 0.1,
    material: Material = None,
    n_pts_per_side: int = 0,
    point_displacements: Optional[Sequence[Tuple[float, float]]] = None,
    perturbation: float = 0.0,
    seed: Optional[int] = None,
) -> StructureGraph:
    """Diamond (rhombus) unit cell with programmable intermediate points."""
    w, h = box
    corners = [(w / 2, 0), (w, h / 2), (w / 2, h), (0, h / 2)]

    n_sides = len(corners)
    perimeter = []
    for i in range(n_sides):
        p1 = corners[i]
        p2 = corners[(i + 1) % n_sides]
        if i == 0:
            perimeter.append(p1)
        if n_pts_per_side > 0:
            intermediates = _interpolate_edge_points(p1, p2, n_pts_per_side)
            perimeter.extend(intermediates)
        perimeter.append(p2)

    n_intermediate = n_sides * n_pts_per_side
    if n_intermediate > 0:
        if point_displacements is not None:
            all_disp = list(point_displacements)
        else:
            edge_len = np.sqrt((w/2)**2 + (h/2)**2)
            mag = perturbation * edge_len if perturbation > 0 else 0.3 * edge_len
            effective_seed = seed if seed is not None else 0
            all_disp = _auto_displacements(n_intermediate, mag, effective_seed)

        idx = 1
        disp_idx = 0
        for s in range(n_sides):
            for k in range(n_pts_per_side):
                if idx < len(perimeter) and disp_idx < len(all_disp):
                    x, y = perimeter[idx]
                    perimeter[idx] = (x + all_disp[disp_idx][0], y + all_disp[disp_idx][1])
                idx += 1
                disp_idx += 1
            idx += 1

    g = StructureGraph(dimension=2, box_size=[w, h])
    nids = [g.add_node(p) for p in perimeter]
    for i in range(len(nids)):
        g.add_edge(nids[i], nids[(i + 1) % len(nids)],
                  radius=radius, material=material, n_internal=n_internal)

    g._metadata["unit_type"] = "diamond"
    g._metadata["n_pts_per_side"] = n_pts_per_side
    return g


# ======================================================================
# Unit registry
# ======================================================================

_UNIT_FACTORIES = {
    "square": _unit_square,
    "triangle": _unit_triangle,
    "hexagon": _unit_hexagon,
    "honeycomb": _unit_honeycomb,
    "kagome": _unit_kagome,
    "reentrant": _unit_reentrant,
    "chiral": _unit_chiral,
    "star": _unit_star,
    "cross": _unit_cross,
    "missing_rib": _unit_missing_rib,
    "diamond": _unit_diamond,
}


def list_units() -> List[str]:
    """Return list of available built-in unit types."""
    return sorted(_UNIT_FACTORIES.keys())


def register_unit(name: str, factory):
    """Register a custom unit factory."""
    _UNIT_FACTORIES[name.lower()] = factory

# ======================================================================
# 3D Unit Factories
# ======================================================================

def _unit_cubic_3d(
    box=(10.0, 10.0, 10.0),
    radius=0.1,
    material=None,
    n_pts_per_side=0,
    point_displacements=None,
    seed=None,
    n_internal=0,
    **kwargs,
):
    """Cubic lattice unit cell."""
    mat = material or Material()
    w, h, d = box
    g = StructureGraph(dimension=3, box_size=[w, h, d])
    corners = [
        (0, 0, 0), (w, 0, 0), (w, h, 0), (0, h, 0),
        (0, 0, d), (w, 0, d), (w, h, d), (0, h, d),
    ]
    nids = [g.add_node(c) for c in corners]
    edges_12 = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    _add_3d_edges(g, nids, edges_12, n_pts_per_side, point_displacements,
                  radius, mat, n_internal, seed)
    g._metadata["unit_type"] = "cubic"
    return g


def _unit_octet_3d(
    box=(10.0, 10.0, 10.0),
    radius=0.1,
    material=None,
    n_pts_per_side=0,
    point_displacements=None,
    seed=None,
    n_internal=0,
    **kwargs,
):
    """Octet truss unit cell (cubic + body center)."""
    mat = material or Material()
    w, h, d = box
    g = StructureGraph(dimension=3, box_size=[w, h, d])
    corners = [
        (0, 0, 0), (w, 0, 0), (w, h, 0), (0, h, 0),
        (0, 0, d), (w, 0, d), (w, h, d), (0, h, d),
    ]
    nids = [g.add_node(c) for c in corners]
    edges_12 = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    _add_3d_edges(g, nids, edges_12, n_pts_per_side, point_displacements,
                  radius, mat, n_internal, seed)
    center = g.add_node([w / 2, h / 2, d / 2])
    center_edges = [(i, len(nids)) for i in range(len(nids))]
    nids_with_center = nids + [center]
    _add_3d_edges(g, nids_with_center, center_edges, n_pts_per_side,
                  point_displacements, radius * 0.7, mat, n_internal, seed)
    g._metadata["unit_type"] = "octet"
    return g


def _unit_diamond_3d(
    box=(10.0, 10.0, 10.0),
    radius=0.1,
    material=None,
    n_pts_per_side=0,
    point_displacements=None,
    seed=None,
    n_internal=0,
    **kwargs,
):
    """Diamond lattice unit cell (cubic + face centers)."""
    mat = material or Material()
    w, h, d = box
    g = StructureGraph(dimension=3, box_size=[w, h, d])
    corners = [
        (0, 0, 0), (w, 0, 0), (w, h, 0), (0, h, 0),
        (0, 0, d), (w, 0, d), (w, h, d), (0, h, d),
    ]
    nids = [g.add_node(c) for c in corners]
    fc = [
        g.add_node([w / 2, h / 2, 0]),
        g.add_node([w / 2, h / 2, d]),
        g.add_node([w / 2, 0, d / 2]),
        g.add_node([w / 2, h, d / 2]),
        g.add_node([0, h / 2, d / 2]),
        g.add_node([w, h / 2, d / 2]),
    ]
    face_to_corners = [
        (0, [0, 1, 2, 3]),
        (1, [4, 5, 6, 7]),
        (2, [0, 1, 4, 5]),
        (3, [2, 3, 6, 7]),
        (4, [0, 3, 4, 7]),
        (5, [1, 2, 5, 6]),
    ]
    all_edges = []
    all_nids = nids + fc
    for fc_idx, c_list in face_to_corners:
        for c_idx in c_list:
            all_edges.append((len(nids) + fc_idx, c_idx))
    _add_3d_edges(g, all_nids, all_edges, n_pts_per_side, point_displacements,
                  radius, mat, n_internal, seed)
    g._metadata["unit_type"] = "diamond_3d"
    return g




def _unit_bcc_3d(
    box=(10.0, 10.0, 10.0),
    radius=0.1,
    material=None,
    n_pts_per_side=0,
    point_displacements=None,
    seed=None,
    n_internal=0,
    **kwargs,
):
    """Body-Centered Cubic (BCC) unit cell.
    
    8 corner atoms + 1 body center. Each corner connected to center
    (8 nearest-neighbor bonds). Plus 12 cube edges for structural integrity.
    """
    mat = material or Material()
    w, h, d = box
    g = StructureGraph(dimension=3, box_size=[w, h, d])
    corners = [
        (0, 0, 0), (w, 0, 0), (w, h, 0), (0, h, 0),
        (0, 0, d), (w, 0, d), (w, h, d), (0, h, d),
    ]
    nids = [g.add_node(c) for c in corners]
    center = g.add_node([w / 2, h / 2, d / 2])
    center_nid = len(nids)
    # Corner-to-center edges (8 nearest-neighbor bonds)
    corner_center_edges = [(i, center_nid) for i in range(8)]
    # Cube edges (12)
    cube_edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    all_edges = corner_center_edges + cube_edges
    all_nids = nids + [center]
    _add_3d_edges(g, all_nids, all_edges, n_pts_per_side, point_displacements,
                  radius, mat, n_internal, seed)
    g._metadata["unit_type"] = "bcc"
    return g


def _unit_fcc_3d(
    box=(10.0, 10.0, 10.0),
    radius=0.1,
    material=None,
    n_pts_per_side=0,
    point_displacements=None,
    seed=None,
    n_internal=0,
    **kwargs,
):
    """Face-Centered Cubic (FCC) unit cell.
    
    8 corner atoms + 6 face center atoms.
    Face centers connect to their 4 face corners + adjacent face centers.
    """
    mat = material or Material()
    w, h, d = box
    g = StructureGraph(dimension=3, box_size=[w, h, d])
    corners = [
        (0, 0, 0), (w, 0, 0), (w, h, 0), (0, h, 0),
        (0, 0, d), (w, 0, d), (w, h, d), (0, h, d),
    ]
    nids = [g.add_node(c) for c in corners]
    # Face centers: index 8-13
    fc = [
        g.add_node([w / 2, h / 2, 0]),     # 8: z=0 face
        g.add_node([w / 2, h / 2, d]),     # 9: z=d face
        g.add_node([w / 2, 0, d / 2]),     # 10: y=0 face
        g.add_node([w / 2, h, d / 2]),     # 11: y=h face
        g.add_node([0, h / 2, d / 2]),     # 12: x=0 face
        g.add_node([w, h / 2, d / 2]),     # 13: x=w face
    ]
    # Face center to corner edges (24 edges)
    fc_corner_edges = [
        (8, 0), (8, 1), (8, 2), (8, 3),   # z=0 face
        (9, 4), (9, 5), (9, 6), (9, 7),   # z=d face
        (10, 0), (10, 1), (10, 4), (10, 5),  # y=0 face
        (11, 2), (11, 3), (11, 6), (11, 7),  # y=h face
        (12, 0), (12, 3), (12, 4), (12, 7),  # x=0 face
        (13, 1), (13, 2), (13, 5), (13, 6),  # x=w face
    ]
    # Face center to adjacent face center edges (12, one per cube edge)
    # Each cube edge is shared by 2 faces → their face centers connect
    fc_fc_edges = [
        (8, 10), (8, 11), (8, 12), (8, 13),   # z=0 face ↔ 4 side faces
        (9, 10), (9, 11), (9, 12), (9, 13),   # z=d face ↔ 4 side faces
        (10, 12), (10, 13),                     # y=0 ↔ x=0, x=w
        (11, 12), (11, 13),                     # y=h ↔ x=0, x=w
    ]
    all_nids = nids + fc
    all_edges = fc_corner_edges + fc_fc_edges
    _add_3d_edges(g, all_nids, all_edges, n_pts_per_side, point_displacements,
                  radius, mat, n_internal, seed)
    g._metadata["unit_type"] = "fcc"
    return g


def _unit_hcp_3d(
    box=(10.0, 10.0, 10.0),
    radius=0.1,
    material=None,
    n_pts_per_side=0,
    point_displacements=None,
    seed=None,
    n_internal=0,
    **kwargs,
):
    """Hexagonal Close-Packed (HCP) unit cell.
    
    Bottom hexagonal layer (z=0) + middle offset layer (z=d/2) +
    top hexagonal layer (z=d). ABAB stacking.
    """
    mat = material or Material()
    w, h, d = box
    g = StructureGraph(dimension=3, box_size=[w, h, d])
    
    cx, cy = w / 2, h / 2
    rx, ry = w / 2, h / 2  # radii for hexagonal arrangement
    
    # Bottom layer (z=0): 6 hexagonal corners + center
    bottom_corners = []
    for i in range(6):
        angle = i * np.pi / 3
        x = cx + rx * np.cos(angle)
        y = cy + ry * np.sin(angle)
        bottom_corners.append((x, y, 0.0))
    bottom_center = (cx, cy, 0.0)
    
    # Top layer (z=d): same positions
    top_corners = [(x, y, d) for (x, y, _) in bottom_corners]
    top_center = (cx, cy, d)
    
    # Middle layer (z=d/2): 3 atoms offset by 30° from bottom
    mid_atoms = []
    for i in range(3):
        angle = (2 * i + 1) * np.pi / 6  # 30°, 150°, 270°
        x = cx + rx * 0.577 * np.cos(angle)  # 1/√3 offset for close packing
        y = cy + ry * 0.577 * np.sin(angle)
        mid_atoms.append((x, y, d / 2))
    
    # Add all nodes
    bottom_nids = [g.add_node(p) for p in bottom_corners]
    bottom_center_nid = g.add_node(bottom_center)
    top_nids = [g.add_node(p) for p in top_corners]
    top_center_nid = g.add_node(top_center)
    mid_nids = [g.add_node(p) for p in mid_atoms]
    
    all_nids = bottom_nids + [bottom_center_nid] + top_nids + [top_center_nid] + mid_nids
    
    edges = []
    # Bottom hexagonal edges (6)
    for i in range(6):
        edges.append((bottom_nids[i], bottom_nids[(i + 1) % 6]))
    # Bottom center to corners (6)
    for i in range(6):
        edges.append((bottom_center_nid, bottom_nids[i]))
    # Top hexagonal edges (6)
    for i in range(6):
        edges.append((top_nids[i], top_nids[(i + 1) % 6]))
    # Top center to corners (6)
    for i in range(6):
        edges.append((top_center_nid, top_nids[i]))
    # Middle to bottom: each mid atom connects to 2 nearest bottom corners
    for mi in range(3):
        # Each middle atom at angle (2i+1)*30° connects to corners i and (i+1)%6
        c1 = 2 * mi
        c2 = (2 * mi + 1) % 6
        edges.append((mid_nids[mi], bottom_nids[c1]))
        edges.append((mid_nids[mi], bottom_nids[c2]))
        edges.append((mid_nids[mi], bottom_center_nid))
    # Middle to top: each mid atom connects to 2 nearest top corners
    for mi in range(3):
        c1 = (2 * mi + 2) % 6
        c2 = (2 * mi + 3) % 6
        edges.append((mid_nids[mi], top_nids[c1]))
        edges.append((mid_nids[mi], top_nids[c2]))
        edges.append((mid_nids[mi], top_center_nid))
    # Vertical center-to-center connection
    edges.append((bottom_center_nid, top_center_nid))
    
    _add_3d_edges(g, all_nids, edges, n_pts_per_side, point_displacements,
                  radius, mat, n_internal, seed)
    g._metadata["unit_type"] = "hcp"
    return g


# ======================================================================
# TPMS (Triply Periodic Minimal Surface) Helpers
# ======================================================================

def _tpms_field(kind, X, Y, Z):
    """Compute TPMS level-set field for given kind."""
    k = kind.lower()
    if k in ('gyroid', 'g'):
        return (np.sin(X) * np.cos(Y) +
                np.sin(Y) * np.cos(Z) +
                np.sin(Z) * np.cos(X))
    elif k in ('schwarz_p', 'primitive', 'p'):
        return np.cos(X) + np.cos(Y) + np.cos(Z)
    elif k in ('schwarz_d', 'diamond', 'd'):
        return (np.sin(X) * np.sin(Y) * np.sin(Z) +
                np.sin(X) * np.cos(Y) * np.cos(Z) +
                np.cos(X) * np.sin(Y) * np.cos(Z) +
                np.cos(X) * np.cos(Y) * np.sin(Z))
    elif k in ('iwp', 'i-wp'):
        return (2.0 * (np.cos(X) * np.cos(Y) +
                       np.cos(Y) * np.cos(Z) +
                       np.cos(Z) * np.cos(X)) -
                (np.cos(2*X) + np.cos(2*Y) + np.cos(2*Z)))
    elif k == 'neovius':
        return (3.0 * (np.cos(X) + np.cos(Y) + np.cos(Z)) +
                4.0 * np.cos(X) * np.cos(Y) * np.cos(Z))
    elif k == 'lidinoid':
        return (np.cos(2*X) * np.sin(Y) * np.cos(Z) +
                np.cos(2*Y) * np.sin(Z) * np.cos(X) +
                np.cos(2*Z) * np.sin(X) * np.cos(Y) -
                np.cos(2*X) * np.cos(2*Y) -
                np.cos(2*Y) * np.cos(2*Z) -
                np.cos(2*Z) * np.cos(2*X) + 0.3)
    else:
        raise ValueError(f"Unknown TPMS type: {kind}")


def _build_tpms_unit(
    kind,
    box=(10.0, 10.0, 10.0),
    num_periods=(1, 1, 1),
    resolution=12,
    voxel_factor=2.0,
    radius=0.05,
    material=None,
    **kwargs,
):
    """Build TPMS unit cell StructureGraph via marching_cubes + voxel downsampling.
    
    Parameters
    ----------
    kind : str
        TPMS type: gyroid, schwarz_p, schwarz_d, iwp, neovius, lidinoid.
    box : tuple
        Unit cell dimensions.
    num_periods : tuple
        Number of TPMS periods in each direction.
    resolution : int
        Marching cubes grid points per period.
    voxel_factor : float
        Voxel grid cells per period (controls downsampling density).
    radius : float
        Beam radius.
    material : Material, optional
        Beam material.
    """
    try:
        from skimage import measure
    except ImportError:
        raise ImportError("TPMS generation requires scikit-image: pip install scikit-image")
    
    try:
        import networkx as _nx
    except ImportError:
        raise ImportError("TPMS generation requires networkx: pip install networkx")
    
    mat = material or Material(name=f"tpms_{kind}")
    w, h, d = box
    npx, npy, npz = num_periods
    
    res_x = max(resolution * npx, 8)
    res_y = max(resolution * npy, 8)
    res_z = max(resolution * npz, 8)
    
    x = np.linspace(0, 2 * np.pi * npx, res_x)
    y = np.linspace(0, 2 * np.pi * npy, res_y)
    z = np.linspace(0, 2 * np.pi * npz, res_z)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    field = _tpms_field(kind, X, Y, Z)
    
    verts, faces, _, _ = measure.marching_cubes(field, level=0.0)
    
    # Scale to physical coordinates
    sx = w / (2 * np.pi * npx)
    sy = h / (2 * np.pi * npy)
    sz = d / (2 * np.pi * npz)
    verts_phys = verts.copy()
    verts_phys[:, 0] *= sx
    verts_phys[:, 1] *= sy
    verts_phys[:, 2] *= sz
    
    # Extract unique edges from triangular faces
    edge_set = set()
    for face in faces:
        for i in range(3):
            edge_set.add(tuple(sorted([face[i], face[(i + 1) % 3]])))
    
    # Voxel grid downsampling
    vs_x = w / (npx * voxel_factor)
    vs_y = h / (npy * voxel_factor)
    vs_z = d / (npz * voxel_factor)
    
    voxel_idx = np.column_stack([
        np.floor(verts_phys[:, 0] / vs_x).astype(int),
        np.floor(verts_phys[:, 1] / vs_y).astype(int),
        np.floor(verts_phys[:, 2] / vs_z).astype(int),
    ])
    
    voxel_groups = {}
    for i in range(len(verts_phys)):
        key = tuple(voxel_idx[i])
        if key not in voxel_groups:
            voxel_groups[key] = []
        voxel_groups[key].append(i)
    
    centroids = {}
    vert_to_cid = {}
    for key, indices in voxel_groups.items():
        centroid = np.mean(verts_phys[indices], axis=0)
        cid = len(centroids)
        centroids[cid] = centroid
        for idx in indices:
            vert_to_cid[idx] = cid
    
    centroid_edges = set()
    for e in edge_set:
        c0 = vert_to_cid.get(e[0])
        c1 = vert_to_cid.get(e[1])
        if c0 is not None and c1 is not None and c0 != c1:
            centroid_edges.add(tuple(sorted([c0, c1])))
    
    # Build StructureGraph
    g = StructureGraph(dimension=3, box_size=[w, h, d])
    cid_to_nid = {}
    for cid, pos in centroids.items():
        nid = g.add_node(pos.tolist(), merge=True)
        cid_to_nid[cid] = nid
    for c0, c1 in centroid_edges:
        ni, nj = cid_to_nid[c0], cid_to_nid[c1]
        if ni != nj:
            g.add_edge(ni, nj, radius=radius, material=mat)
    
    # Connectivity repair: merge small components into largest
    G_nx = _nx.Graph()
    for nid in g.nodes:
        G_nx.add_node(nid)
    for eid in g.edges:
        e = g.edges[eid]
        G_nx.add_edge(e.node_i, e.node_j)
    comps = sorted(_nx.connected_components(G_nx), key=len, reverse=True)
    
    if len(comps) > 1:
        main = set(comps[0])
        for small_comp in comps[1:]:
            if len(small_comp) < 2:
                continue
            best_dist = float('inf')
            best_pair = None
            for snid in small_comp:
                spos = g.nodes[snid].position
                for mnid in list(main)[:100]:
                    mpos = g.nodes[mnid].position
                    dd = np.linalg.norm(spos - mpos)
                    if dd < best_dist:
                        best_dist = dd
                        best_pair = (snid, mnid)
            if best_pair:
                g.add_edge(best_pair[0], best_pair[1], radius=radius, material=mat)
                main = main | set(small_comp)
    
    g._metadata["unit_type"] = kind
    g._metadata["tpms_params"] = {
        "num_periods": list(num_periods),
        "resolution": resolution,
        "voxel_factor": voxel_factor,
    }
    return g


def _make_tpms_factory(tpms_kind):
    """Create a factory function for a specific TPMS type."""
    def factory(
        box=(10.0, 10.0, 10.0),
        radius=0.05,
        material=None,
        n_pts_per_side=0,
        point_displacements=None,
        seed=None,
        n_internal=0,
        num_periods=(1, 1, 1),
        resolution=12,
        voxel_factor=2.0,
        **kwargs,
    ):
        g = _build_tpms_unit(
            tpms_kind,
            box=box,
            num_periods=num_periods,
            resolution=resolution,
            voxel_factor=voxel_factor,
            radius=radius,
            material=material,
        )
        return g
    factory.__name__ = f"_unit_{tpms_kind}_3d"
    factory.__doc__ = f"TPMS {tpms_kind} unit cell factory."
    return factory



def _unit_chiral_3d(
    box=(10.0, 10.0, 10.0),
    radius=0.1,
    material=None,
    n_pts_per_side=0,
    point_displacements=None,
    seed=None,
    n_internal=0,
    node_radius=None,
    chirality=0.3,
    **kwargs,
):
    """3D chiral metamaterial unit cell.
    
    Central ring of nodes on a sphere connected to 8 corner nodes
    via tangent (chiral) ligaments. The ``chirality`` parameter controls
    the tangent offset angle, creating auxetic twist behavior.
    
    Parameters
    ----------
    chirality : float
        Tangent offset fraction (0=radial, 1=full tangent). Default 0.3.
    node_radius : float, optional
        Radius of the central node ring. Default = min(box)/3.
    """
    mat = material or Material()
    w, h, d = box
    g = StructureGraph(dimension=3, box_size=[w, h, d])
    
    if node_radius is None:
        node_radius = min(w, h, d) / 3.0
    
    cx, cy, cz = w / 2, h / 2, d / 2
    
    # 8 ring nodes on a sphere (vertices of a cube inscribed in the sphere)
    s = node_radius / np.sqrt(3)  # scale so corners are at node_radius
    ring_positions = [
        (cx + s, cy + s, cz + s), (cx + s, cy + s, cz - s),
        (cx + s, cy - s, cz + s), (cx + s, cy - s, cz - s),
        (cx - s, cy + s, cz + s), (cx - s, cy + s, cz - s),
        (cx - s, cy - s, cz + s), (cx - s, cy - s, cz - s),
    ]
    ring_nids = [g.add_node(p) for p in ring_positions]
    
    # 8 corner nodes
    corners = [
        (0, 0, 0), (w, 0, 0), (w, h, 0), (0, h, 0),
        (0, 0, d), (w, 0, d), (w, h, d), (0, h, d),
    ]
    corner_nids = [g.add_node(c) for c in corners]
    
    edge_pairs = []
    
    # Ring-to-ring edges (cube edges connecting the 8 ring nodes)
    ring_edges = [
        (0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),
        (4,5),(4,6),(5,7),(6,7),
    ]
    for i, j in ring_edges:
        edge_pairs.append((ring_nids[i], ring_nids[j]))
    
    # Chiral (tangent) ligaments from corners to ring nodes
    # Each corner connects to its 3 nearest ring nodes, but offset by chirality
    for ci in range(8):
        # Corner binary representation (which octant)
        cx_b = (ci >> 2) & 1  # x bit
        cy_b = (ci >> 1) & 1  # y bit  
        cz_b = ci & 1         # z bit
        
        # Find the 3 nearest ring nodes (same octant neighbors)
        for axis in range(3):
            # Flip one bit to get the neighbor ring node
            ri = ci ^ (1 << axis)
            if ri < 8:
                # Create a chiral (tangent) connection point
                # Instead of connecting corner directly to ring node,
                # offset the ring endpoint by chirality * tangent direction
                rp = np.array(ring_positions[ri])
                cp = np.array(corners[ci])
                
                # Tangent direction = perpendicular to radial direction
                radial = rp - np.array([cx, cy, cz])
                tangent = np.cross(radial, np.array([0, 0, 1]) if axis == 0 else
                                  (np.array([1, 0, 0]) if axis == 1 else np.array([0, 1, 0])))
                tangent_norm = np.linalg.norm(tangent)
                if tangent_norm > 1e-10:
                    tangent = tangent / tangent_norm * node_radius * chirality
                    target = rp + tangent
                else:
                    target = rp
                
                # Add a tangent node
                tan_nid = g.add_node(target.tolist())
                edge_pairs.append((corner_nids[ci], tan_nid))
                edge_pairs.append((tan_nid, ring_nids[ri]))
    
    _add_3d_edges(g, list(range(g.num_nodes)), edge_pairs, n_pts_per_side,
                  point_displacements, radius, mat, n_internal, seed)
    g._metadata["unit_type"] = "chiral_3d"
    g._metadata["chirality"] = chirality
    return g


def _unit_reentrant_3d(
    box=(10.0, 10.0, 10.0),
    radius=0.1,
    material=None,
    n_pts_per_side=0,
    point_displacements=None,
    seed=None,
    n_internal=0,
    angle=15.0,
    **kwargs,
):
    """3D reentrant (auxetic) metamaterial unit cell.
    
    Extends the 2D reentrant arrowhead concept to 3D.
    Internal indentation nodes create negative Poisson's ratio
    when the structure is stretched.
    
    Parameters
    ----------
    angle : float
        Reentrant angle in degrees (0-40). Controls indentation depth.
    """
    mat = material or Material()
    w, h, d = box
    g = StructureGraph(dimension=3, box_size=[w, h, d])
    
    cx, cy, cz = w / 2, h / 2, d / 2
    
    # 6 boundary nodes at face centers
    face_centers = [
        (cx, cy, 0),    # bottom
        (cx, cy, d),    # top
        (cx, 0, cz),    # front
        (cx, h, cz),    # back
        (0, cy, cz),    # left
        (w, cy, cz),    # right
    ]
    face_nids = [g.add_node(p) for p in face_centers]
    
    # 8 corner nodes
    corners = [
        (0, 0, 0), (w, 0, 0), (w, h, 0), (0, h, 0),
        (0, 0, d), (w, 0, d), (w, h, d), (0, h, d),
    ]
    corner_nids = [g.add_node(c) for c in corners]
    
    # Indentation: compute inward offset based on reentrant angle
    max_indent = min(w, h, d) / 4
    indent = max_indent * np.tan(np.radians(min(angle, 40)))
    
    # 2 indentation nodes along z-axis (above and below center)
    indent_lo = g.add_node([cx, cy, cz - indent])  # lower indent
    indent_hi = g.add_node([cx, cy, cz + indent])  # upper indent
    
    edge_pairs = []
    
    # Corner-to-face edges (12 cube edges)
    cube_edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7),
    ]
    for i, j in cube_edges:
        edge_pairs.append((corner_nids[i], corner_nids[j]))
    
    # Face center to adjacent corners (structural support)
    face_corner_map = [
        (0, [0,1,2,3]),   # bottom face → bottom corners
        (1, [4,5,6,7]),   # top face → top corners
        (2, [0,1,4,5]),   # front face → front corners
        (3, [2,3,6,7]),   # back face → back corners
        (4, [0,3,4,7]),   # left face → left corners
        (5, [1,2,5,6]),   # right face → right corners
    ]
    for fi, c_list in face_corner_map:
        for ci in c_list:
            edge_pairs.append((face_nids[fi], corner_nids[ci]))
    
    # Reentrant diagonals: each face center connects to both indent nodes
    for fi in range(6):
        edge_pairs.append((face_nids[fi], indent_lo))
        edge_pairs.append((face_nids[fi], indent_hi))
    
    # Indent-to-indent connection (vertical spine)
    edge_pairs.append((indent_lo, indent_hi))
    
    _add_3d_edges(g, list(range(g.num_nodes)), edge_pairs, n_pts_per_side,
                  point_displacements, radius, mat, n_internal, seed)
    g._metadata["unit_type"] = "reentrant_3d"
    g._metadata["reentrant_angle"] = angle
    return g


_UNIT_FACTORIES_3D = {
    "cubic": _unit_cubic_3d,
    "octet": _unit_octet_3d,
    "diamond_3d": _unit_diamond_3d,
    "bcc": _unit_bcc_3d,
    "fcc": _unit_fcc_3d,
    "hcp": _unit_hcp_3d,
    # Chiral and reentrant
    "chiral_3d": _unit_chiral_3d,
    "reentrant_3d": _unit_reentrant_3d,
    # TPMS types
    "gyroid": _make_tpms_factory("gyroid"),
    "schwarz_p": _make_tpms_factory("schwarz_p"),
    "schwarz_d": _make_tpms_factory("schwarz_d"),
    "iwp": _make_tpms_factory("iwp"),
    "neovius": _make_tpms_factory("neovius"),
    "lidinoid": _make_tpms_factory("lidinoid"),
}


def list_units_3d():
    """Return list of available built-in 3D unit types."""
    return sorted(_UNIT_FACTORIES_3D.keys())


def register_unit_3d(name, factory):
    """Register a custom 3D unit factory."""
    _UNIT_FACTORIES_3D[name.lower()] = factory



# ======================================================================
# Main pattern_2d / pattern_3d functions
# ======================================================================

def pattern_2d(
    *,
    unit: Optional[str] = None,
    points: Optional[Sequence[Sequence[float]]] = None,
    closed: bool = True,
    box: Tuple[float, float] = (10.0, 10.0),
    grid: Union[int, Tuple[int, int]] = (3, 3),
    n_internal: int = 0,
    n_pts_per_side: int = 0,
    point_displacements: Optional[Sequence[Tuple[float, float]]] = None,
    radius: float = 0.1,
    material: Optional[Material] = None,
    mirror_x: bool = False,
    mirror_y: bool = False,
    rotation: float = 0.0,
    perturbation: float = 0.0,
    seed: Optional[int] = None,
    boundary_mode: str = "none",
    fit_to_box: bool = False,
    unit_kwargs: Optional[Dict[str, Any]] = None,
) -> StructureGraph:
    """Generate a 2D periodic structure using Base Unit + Transform + Tiling.

    Parameters
    ----------
    unit : str, optional
        Built-in unit type. See ``list_units()``.
    points : sequence of (x, y), optional
        Custom polyline points. Mutually exclusive with ``unit``.
    closed : bool
        Whether the custom polyline is closed.
    box : (w, h)
        Unit cell dimensions.
    grid : int or (nx, ny)
        Tiling grid size.
    n_internal : int
        Internal points per edge (for FEM deformation visualization).
    n_pts_per_side : int
        Intermediate graph nodes per polygon edge (affects structure topology).
        0 = corners only. Higher values create more complex beam geometry.
    point_displacements : list of (dx, dy), optional
        Explicit displacement for each intermediate node.
        Total length = n_sides * n_pts_per_side (varies by unit type).
        If None and n_pts_per_side > 0, auto-generates deterministic non-zero
        displacements using ``seed`` (default seed=0).
    radius : float
        Edge/beam radius.
    material : Material, optional
        Beam material.
    mirror_x : bool
        Mirror the unit across the y-axis before tiling.
    mirror_y : bool
        Mirror the unit across the x-axis before tiling.
    rotation : float
        Rotate the unit (degrees) before tiling.
    perturbation : float
        Cn-symmetric perturbation magnitude (fraction of edge length).
    seed : int, optional
        Random seed for deterministic displacements.
    boundary_mode : str
        ``"none"``, ``"extend"``, or ``"error"``.
    fit_to_box : bool
        Scale custom points to fit within the box.
    unit_kwargs : dict, optional
        Extra kwargs for the unit factory (e.g., ``angle=20`` for reentrant).

    Returns
    -------
    StructureGraph
    """
    if unit is None and points is None:
        raise ValueError("Must specify either 'unit' or 'points'")
    if unit is not None and points is not None:
        raise ValueError("Cannot specify both 'unit' and 'points'")

    mat = material or Material()
    ukw = unit_kwargs or {}

    # Step 1: Create unit cell
    if unit is not None:
        unit_name = unit.lower()
        if unit_name not in _UNIT_FACTORIES:
            raise ValueError(f"Unknown unit '{unit}'. Available: {list_units()}")
        # Pass n_pts_per_side + displacement params to factory
        factory_kwargs = {
            "box": box,
            "n_internal": n_internal,
            "radius": radius,
            "material": mat,
            "n_pts_per_side": n_pts_per_side,
            "point_displacements": point_displacements,
            "perturbation": perturbation,
            "seed": seed,
        }
        factory_kwargs.update(ukw)
        unit_cell = _UNIT_FACTORIES[unit_name](**factory_kwargs)
    else:
        # Custom points with intermediate node support
        pts = [tuple(p) for p in points]
        if fit_to_box:
            raw = StructureGraph(dimension=2, box_size=list(box) + [0.0])
            raw.add_polyline(pts, closed=closed, radius=radius, material=mat,
                             n_internal=n_internal)
            unit_cell = fit_unit_to_box(raw, target_box=list(box) + [0.0])
        else:
            unit_cell = StructureGraph(dimension=2, box_size=list(box) + [0.0])
            if n_pts_per_side > 0:
                # Build polyline with intermediate nodes on each segment
                pairs = list(zip(pts[:-1], pts[1:]))
                if closed and len(pts) > 2:
                    pairs.append((pts[-1], pts[0]))
                n_total_pts = len(pairs) * n_pts_per_side
                if point_displacements is not None:
                    all_disp = list(point_displacements)
                else:
                    edge_lens = [np.linalg.norm(np.array(p2) - np.array(p1)) for p1, p2 in pairs]
                    mean_len = np.mean(edge_lens) if edge_lens else min(box) / 5
                    mag = perturbation * mean_len if perturbation > 0 else 0.3 * mean_len
                    effective_seed = seed if seed is not None else 0
                    all_disp = _auto_displacements(n_total_pts, mag, effective_seed)
                for idx, (p1, p2) in enumerate(pairs):
                    start = idx * n_pts_per_side
                    end = start + n_pts_per_side
                    edge_disp = all_disp[start:end]
                    _add_edge_with_intermediates(
                        unit_cell, p1, p2, n_pts_per_side, edge_disp,
                        radius, mat, n_internal,
                    )
            else:
                unit_cell.add_polyline(pts, closed=closed, radius=radius, material=mat,
                                       n_internal=n_internal)

    # Step 2: Apply mirror transforms (applied before tiling, preserves boundary alignment)
    if mirror_x:
        unit_cell = _mirror_x(unit_cell, origin=box[0] / 2)
    if mirror_y:
        unit_cell = _mirror_y(unit_cell, origin=box[1] / 2)

    # Note: rotation is applied AFTER tiling to preserve boundary welding.
    # Rotating the unit cell before tiling breaks periodic boundary alignment.

    # Step 3: Boundary mode check
    if boundary_mode == "error":
        _check_boundary_contact(unit_cell, box)
    elif boundary_mode == "extend":
        unit_cell = _extend_to_boundary(unit_cell, box, radius, mat, n_internal)

    # Step 4: Tile
    result = tile_2d(unit_cell, grid=grid, box_size=list(box) + [0.0])

    # Step 5: Apply rotation to the tiled result (preserves connectivity)
    if rotation != 0.0:
        grid_tuple = grid if isinstance(grid, (list, tuple)) else (grid, grid)
        center = np.array([box[0] * grid_tuple[0] / 2, box[1] * grid_tuple[1] / 2, 0.0])
        result = rotate(result, rotation, center=center)

    # Step 6: Bridge small disconnected components (Voronoi artifacts)
    if unit == "voronoi":
        result = _bridge_small_components(result, min_component_size=5,
                                         radius=radius, material=mat)

    result._metadata["pattern"] = {
        "unit": unit or "custom",
        "box": list(box),
        "grid": list(grid) if isinstance(grid, (list, tuple)) else [grid, grid],
        "n_internal": n_internal,
        "n_pts_per_side": n_pts_per_side,
        "mirror_x": mirror_x,
        "mirror_y": mirror_y,
        "rotation": rotation,
    }

    return result



def _post_tile_connectivity_repair(
    graph,
    max_bridge_dist=3.0,
    radius=None,
    material=None,
):
    """Repair disconnected components after tiling by adding bridge edges.
    
    Uses spatial proximity to find the closest node pairs between
    disconnected components and adds bridging edges.
    
    Parameters
    ----------
    graph : StructureGraph
        Tiled structure (modified in-place).
    max_bridge_dist : float
        Maximum distance for bridge edges.
    radius : float, optional
        Bridge edge radius. Default: median of existing edge radii.
    material : Material, optional
        Bridge material. Default: first edge's material.
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return graph
    
    try:
        import networkx as _nx
    except ImportError:
        return graph
    
    # Check if already connected
    G_nx = _nx.Graph()
    for nid in graph.nodes:
        G_nx.add_node(nid)
    for eid in graph.edges:
        e = graph.edges[eid]
        G_nx.add_edge(e.node_i, e.node_j)
    
    comps = list(_nx.connected_components(G_nx))
    if len(comps) <= 1:
        return graph
    
    # Default radius/material from existing edges
    if radius is None:
        radii = [graph.edges[eid].radius for eid in graph.edges]
        radius = float(np.median(radii)) if radii else 0.05
    if material is None:
        first_eid = next(iter(graph.edges))
        material = graph.edges[first_eid].material
    
    # Get all positions
    node_ids = sorted(graph.nodes.keys())
    positions = np.array([graph.nodes[nid].position for nid in node_ids])
    nid_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    
    # Build KDTree
    tree = cKDTree(positions)
    
    # Find close pairs that aren't already connected
    existing_edges = set()
    for eid in graph.edges:
        e = graph.edges[eid]
        existing_edges.add(frozenset([e.node_i, e.node_j]))
    
    # Merge components iteratively
    max_iterations = len(comps) * 3
    for iteration in range(max_iterations):
        # Rebuild connectivity
        G_nx = _nx.Graph()
        for nid in graph.nodes:
            G_nx.add_node(nid)
        for eid in graph.edges:
            e = graph.edges[eid]
            G_nx.add_edge(e.node_i, e.node_j)
        
        comps = sorted(_nx.connected_components(G_nx), key=len, reverse=True)
        if len(comps) <= 1:
            break
        
        # Find closest pair between largest component and next largest
        main_comp = set(comps[0])
        for other_comp in comps[1:]:
            other_set = set(other_comp)
            if len(other_set) < 1:
                continue
            
            # Get node IDs
            main_nids = list(main_comp)
            other_nids = list(other_set)
            
            # Use KDTree for efficiency
            main_indices = [nid_to_idx[nid] for nid in main_nids if nid in nid_to_idx]
            other_indices = [nid_to_idx[nid] for nid in other_nids if nid in nid_to_idx]
            
            if not main_indices or not other_indices:
                continue
            
            main_pos = positions[main_indices]
            other_pos = positions[other_indices]
            
            # Find closest pair using cKDTree
            main_tree = cKDTree(main_pos)
            dists, idxs = main_tree.query(other_pos, k=1)
            
            # Filter by max distance
            valid = dists < max_bridge_dist
            if not np.any(valid):
                # Try with larger distance
                continue
            
            # Add bridge for the closest valid pair
            best_other = np.argmin(dists[valid])
            best_main_idx = idxs[valid][best_other]
            best_other_idx = np.where(valid)[0][best_other]
            
            main_nid = main_nids[best_main_idx]
            other_nid = other_nids[best_other_idx]
            
            # Check not already connected
            if frozenset([main_nid, other_nid]) not in existing_edges:
                graph.add_edge(main_nid, other_nid, radius=radius, material=material)
                existing_edges.add(frozenset([main_nid, other_nid]))
                main_comp = main_comp | other_set
    
    # Repair dangling (degree-1) nodes by connecting to nearest non-boundary neighbor
    G_check = _nx.Graph()
    for nid in graph.nodes:
        G_check.add_node(nid)
    for eid in graph.edges:
        e = graph.edges[eid]
        G_check.add_edge(e.node_i, e.node_j)
    
    dangling = [nid for nid, d in G_check.degree() if d == 1]
    if dangling:
        node_ids = sorted(graph.nodes.keys())
        nid_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        positions = np.array([graph.nodes[nid].position for nid in node_ids])
        
        for d_nid in dangling:
            d_pos = graph.nodes[d_nid].position
            d_idx = nid_to_idx[d_nid]
            
            # Find nearest node that isn't the same dangling node's only neighbor
            neighbors = set(G_check.neighbors(d_nid))
            best_dist = float('inf')
            best_nid = None
            
            for other_nid in node_ids:
                if other_nid == d_nid or other_nid in neighbors:
                    continue
                dist = np.linalg.norm(graph.nodes[other_nid].position - d_pos)
                if dist < best_dist and dist < max_bridge_dist * 2:
                    best_dist = dist
                    best_nid = other_nid
            
            if best_nid is not None:
                graph.add_edge(d_nid, best_nid, radius=radius, material=material)

    return graph

def pattern_3d(
    *,
    unit="cubic",
    box=(10.0, 10.0, 10.0),
    grid=(3, 3, 3),
    n_internal=0,
    n_pts_per_side=0,
    point_displacements=None,
    radius=0.1,
    material=None,
    seed=None,
    unit_kwargs=None,
):
    """Generate a 3D periodic structure.

    Parameters
    ----------
    unit : str
        Built-in 3D unit type. Use ``list_units_3d()`` to see available types.
    box : (w, h, d)
        Unit cell dimensions.
    grid : int or (nx, ny, nz)
        Grid dimensions.
    n_internal : int
        Internal points per edge (for FEM deformation).
    n_pts_per_side : int
        Intermediate graph nodes per edge (affects structure topology).
    point_displacements : list of (dx, dy, dz), optional
        Explicit displacement per intermediate node.
    radius : float
        Beam radius.
    material : Material, optional
        Beam material.
    seed : int, optional
        Random seed for auto-generated displacements.
    unit_kwargs : dict, optional
        Extra keyword arguments passed to the unit factory.

    Returns
    -------
    StructureGraph
    """
    if isinstance(grid, int):
        grid = (grid, grid, grid)

    unit_name = unit.lower()
    if unit_name not in _UNIT_FACTORIES_3D:
        available = ", ".join(sorted(_UNIT_FACTORIES_3D.keys()))
        raise ValueError(
            f"Unknown 3D unit \'{unit}\'. Available: {available}"
        )

    factory = _UNIT_FACTORIES_3D[unit_name]
    fkwargs = dict(unit_kwargs) if unit_kwargs else {}
    fkwargs.update(
        box=box,
        radius=radius,
        material=material,
        n_pts_per_side=n_pts_per_side,
        point_displacements=point_displacements,
        seed=seed,
        n_internal=n_internal,
    )

    g = factory(**fkwargs)
    g._metadata["n_pts_per_side"] = n_pts_per_side

    w, h, d = box
    result = tile_3d(g, grid=grid, box_size=[w, h, d])
    # Post-tiling connectivity repair for TPMS and other mesh-based types
    tpms_types = {"gyroid", "schwarz_p", "schwarz_d", "iwp", "neovius", "lidinoid"}
    if unit_name in tpms_types or unit_name == "hcp":
        result = _post_tile_connectivity_repair(result, max_bridge_dist=3.0, radius=radius)
        # Update component count in metadata
        try:
            import networkx as _nx
            G_check = _nx.Graph()
            for nid in result.nodes:
                G_check.add_node(nid)
            for eid in result.edges:
                e = result.edges[eid]
                G_check.add_edge(e.node_i, e.node_j)
            n_comps = _nx.number_connected_components(G_check)
            result._metadata["n_components_post_repair"] = n_comps
        except ImportError:
            pass

    result._metadata["pattern"] = {
        "unit": unit_name,
        "box": list(box),
        "grid": list(grid),
        "n_pts_per_side": n_pts_per_side,
    }
    return result

def _add_3d_edges(
    g: StructureGraph,
    nids: List[int],
    edge_pairs: List[Tuple[int, int]],
    n_pts_per_side: int,
    point_displacements: Optional[Sequence],
    radius: float,
    material: Material,
    n_internal: int,
    seed: Optional[int],
):
    """Add 3D edges with optional intermediate nodes."""
    n_total_pts = len(edge_pairs) * n_pts_per_side
    if n_pts_per_side > 0:
        if point_displacements is not None and len(point_displacements) >= n_total_pts:
            all_disp = list(point_displacements[:n_total_pts])
        else:
            # Auto-generate 3D displacements
            effective_seed = seed if seed is not None else 0
            rng = np.random.default_rng(effective_seed)
            mag = 0.3  # 30% default
            all_disp = []
            for _ in range(n_total_pts):
                dx = rng.uniform(-mag, mag)
                dy = rng.uniform(-mag, mag)
                dz = rng.uniform(-mag, mag)
                if abs(dx) < 1e-10 and abs(dy) < 1e-10 and abs(dz) < 1e-10:
                    dx = mag * 0.5
                all_disp.append((float(dx), float(dy), float(dz)))
    else:
        all_disp = []

    for idx, (i, j) in enumerate(edge_pairs):
        if i >= len(nids) or j >= len(nids):
            continue
        ni, nj = nids[i], nids[j]
        pi = g.nodes[ni].position
        pj = g.nodes[nj].position

        if n_pts_per_side > 0:
            start = idx * n_pts_per_side
            end = start + n_pts_per_side
            edge_disp = all_disp[start:end]

            # Add intermediate nodes
            chain = [ni]
            for k in range(n_pts_per_side):
                t = (k + 1) / (n_pts_per_side + 1)
                pt = pi + t * (pj - pi)
                if k < len(edge_disp):
                    d = edge_disp[k]
                    pt = pt.copy()
                    pt[0] += d[0]
                    pt[1] += d[1]
                    if len(d) > 2:
                        pt[2] += d[2]
                mid_nid = g.add_node(pt.tolist())
                chain.append(mid_nid)
            chain.append(nj)

            for c in range(len(chain) - 1):
                g.add_edge(chain[c], chain[c + 1], radius=radius,
                          material=material, n_internal=n_internal)
        else:
            g.add_edge(ni, nj, radius=radius, material=material,
                      n_internal=n_internal)



# ======================================================================
# Post-tiling connectivity repair
# ======================================================================

def _weld_nearby_nodes(graph: StructureGraph, tolerance: float = 1e-4) -> StructureGraph:
    """Merge nodes that are close but not merged (e.g., after rotation + tiling).

    This handles cases where the standard spatial-hash merge misses nodes
    because rotation moved boundary nodes slightly off expected positions.
    """
    g = graph.copy()
    pos = g.node_positions()
    nids = sorted(g.nodes.keys())

    if len(nids) < 2:
        return g

    # Use scipy cKDTree for efficient proximity search, fall back to brute force
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(pos)
        pairs = tree.query_pairs(tolerance)
    except ImportError:
        pairs = set()
        for i in range(len(nids)):
            for j in range(i + 1, len(nids)):
                if np.linalg.norm(pos[i] - pos[j]) < tolerance:
                    pairs.add((i, j))

    # Build union-find for node merging
    parent = {nid: nid for nid in nids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            # Keep the smaller ID as representative
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    # Convert pair indices to node IDs and union them
    idx_to_nid = {i: nids[i] for i in range(len(nids))}
    for i, j in pairs:
        union(idx_to_nid[i], idx_to_nid[j])

    # Build mapping from old node ID to merged node ID
    id_map = {nid: find(nid) for nid in nids}

    # Check if any merges happened
    if all(k == v for k, v in id_map.items()):
        return g  # No merges needed

    # Rebuild graph with merged nodes
    merged = StructureGraph(dimension=g.dimension, tolerance=g.tolerance,
                           box_size=g.box_size)
    merged._metadata = copy.deepcopy(g.metadata)

    # Add unique nodes (using representative positions)
    new_id_map = {}
    for nid in nids:
        rep = id_map[nid]
        if rep not in new_id_map:
            new_nid = merged.add_node(g.nodes[rep].position, merge=False,
                                      **g.nodes[rep].metadata)
            new_id_map[rep] = new_nid
        new_id_map[nid] = new_id_map[rep]

    # Add edges (skip self-loops from merged nodes)
    seen_edges = set()
    for eid in sorted(g.edges.keys()):
        edge = g.edges[eid]
        new_i = new_id_map[edge.node_i]
        new_j = new_id_map[edge.node_j]
        if new_i == new_j:
            continue  # Skip self-loops
        edge_key = (min(new_i, new_j), max(new_i, new_j))
        if edge_key in seen_edges:
            continue  # Skip duplicate edges
        seen_edges.add(edge_key)
        merged.add_edge(new_i, new_j, radius=edge.radius,
                       material=copy.deepcopy(edge.material),
                       internal_points=edge.internal_points,
                       segments=edge.segments, **edge.metadata)

    return merged


# ======================================================================
# Boundary helpers
# ======================================================================

def _check_boundary_contact(graph: StructureGraph, box: Tuple[float, float],
                            tol: float = 1e-4):
    """Raise ValueError if no node touches any cell boundary."""
    w, h = box
    touches = {"x_min": False, "x_max": False, "y_min": False, "y_max": False}
    for node in graph.nodes.values():
        p = node.position
        if abs(p[0]) < tol:
            touches["x_min"] = True
        if abs(p[0] - w) < tol:
            touches["x_max"] = True
        if abs(p[1]) < tol:
            touches["y_min"] = True
        if abs(p[1] - h) < tol:
            touches["y_max"] = True
    missing = [k for k, v in touches.items() if not v]
    if missing:
        raise ValueError(
            f"Unit does not touch cell boundaries: {missing}. "
            f"Use boundary_mode='extend' to auto-connect."
        )


def _extend_to_boundary(
    graph: StructureGraph,
    box: Tuple[float, float],
    radius: float,
    material: Material,
    n_internal: int,
) -> StructureGraph:
    """Add bridge edges to connect interior unit to cell boundaries."""
    g = graph.copy()
    w, h = box

    if not g.nodes:
        return g
    pos = g.node_positions()
    centroid = pos.mean(axis=0)

    targets = [
        (0, centroid[1], 0),
        (w, centroid[1], 0),
        (centroid[0], 0, 0),
        (centroid[0], h, 0),
    ]

    for tx, ty, tz in targets:
        target = np.array([tx, ty, tz])
        already = False
        for node in g.nodes.values():
            if np.linalg.norm(node.position - target) < graph.tolerance * 10:
                already = True
                break
        if already:
            continue

        min_dist = float("inf")
        nearest_nid = None
        for nid, node in g.nodes.items():
            d = np.linalg.norm(node.position - target)
            if d < min_dist:
                min_dist = d
                nearest_nid = nid

        if nearest_nid is not None and min_dist > graph.tolerance:
            bn = g.add_node(target, merge=False)
            g.add_edge(nearest_nid, bn, radius=radius, material=material,
                       n_internal=n_internal)

    g.box_size = np.array([w, h, 0.0])
    return g




def _bridge_small_components(
    graph: StructureGraph,
    min_component_size: int = 5,
    radius: float = 0.1,
    material: Optional[Material] = None,
) -> StructureGraph:
    """Connect small isolated components to the largest component via bridge edges.

    After Voronoi clipping, a few edges may end up disconnected at tile
    boundaries. This finds components smaller than ``min_component_size``
    and adds a bridge edge from their nearest node to the main component.
    """
    g = graph.copy()
    if not g.nodes:
        return g

    # Build adjacency
    adj = {nid: set() for nid in g.nodes}
    for e in g.edges.values():
        adj[e.node_i].add(e.node_j)
        adj[e.node_j].add(e.node_i)

    # Find connected components via BFS
    visited = set()
    components = []
    for nid in g.nodes:
        if nid in visited:
            continue
        comp = set()
        queue = [nid]
        visited.add(nid)
        while queue:
            n = queue.pop(0)
            comp.add(n)
            for nb in adj.get(n, set()):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        components.append(comp)

    if len(components) <= 1:
        return g  # Already connected

    # Find the main (largest) component
    components.sort(key=len, reverse=True)
    main_comp = components[0]
    main_positions = np.array([g.nodes[nid].position for nid in main_comp])

    for small_comp in components[1:]:
        if len(small_comp) >= min_component_size:
            continue  # Skip large disconnected components (might be intentional)

        # Find the closest pair between small comp and main comp
        best_dist = float("inf")
        best_small_nid = None
        best_main_nid = None

        for snid in small_comp:
            spos = g.nodes[snid].position
            for mnid in main_comp:
                mpos = g.nodes[mnid].position
                d = np.linalg.norm(spos - mpos)
                if d < best_dist:
                    best_dist = d
                    best_small_nid = snid
                    best_main_nid = mnid

        if best_small_nid is not None and best_main_nid is not None:
            g.add_edge(best_small_nid, best_main_nid, radius=radius,
                       material=material or Material())

    return g


def _clip_segment_to_box(
    p0: np.ndarray,
    p1: np.ndarray,
    w: float,
    h: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Clip a line segment to the box [0, w] x [0, h].

    Returns clipped (p0, p1) clamped to exact boundary, or None if
    the segment is entirely outside the box.
    """
    p0 = np.asarray(p0, dtype=float)[:2]
    p1 = np.asarray(p1, dtype=float)[:2]
    d = p1 - p0
    t_min, t_max = 0.0, 1.0

    for axis, lo, hi in [(0, 0.0, w), (1, 0.0, h)]:
        if abs(d[axis]) < 1e-14:
            if p0[axis] < lo or p0[axis] > hi:
                return None
        else:
            t1 = (lo - p0[axis]) / d[axis]
            t2 = (hi - p0[axis]) / d[axis]
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)

    if t_min > t_max + 1e-14:
        return None

    cp0 = np.clip(p0 + t_min * d, [0, 0], [w, h])
    cp1 = np.clip(p0 + t_max * d, [0, 0], [w, h])
    return cp0, cp1


def _unit_voronoi(
    box: Tuple[float, float],
    n_internal: int = 0,
    radius: float = 0.1,
    material: Material = None,
    n_seeds: int = 20,
    seed: int = 42,
    n_pts_per_side: int = 0,
    point_displacements: Optional[Sequence[Tuple[float, float]]] = None,
    perturbation: float = 0.0,
) -> StructureGraph:
    """Voronoi tessellation unit cell with periodic seeding for tiled connectivity.

    Uses periodic (mirror) seeding to ensure Voronoi edges match at cell
    boundaries when tiled, producing connected structures.

    Parameters
    ----------
    n_seeds : int
        Number of Voronoi seed points (before periodic replication)
    seed : int
        Random seed for seed point generation
    """
    w, h = box
    g = StructureGraph(dimension=2, box_size=[w, h])

    # Generate seed points
    rng = np.random.default_rng(seed)
    seeds = rng.uniform(low=[0, 0], high=[w, h], size=(n_seeds, 2))

    # Periodic replication: mirror seeds across all 8 neighbors
    # This ensures Voronoi edges at boundaries match when tiled
    periodic_seeds = []
    for dx in [-w, 0, w]:
        for dy in [-h, 0, h]:
            shifted = seeds + np.array([dx, dy])
            periodic_seeds.append(shifted)
    periodic_seeds = np.vstack(periodic_seeds)

    # Compute Voronoi on periodic seeds
    try:
        from scipy.spatial import Voronoi
        vor = Voronoi(periodic_seeds)
    except ImportError:
        raise ImportError("Voronoi requires scipy: pip install scipy")

    # Clip all Voronoi ridges to the primary cell [0, w] x [0, h]
    edge_tol = 1e-8
    edge_pairs = []
    for ridge in vor.ridge_vertices:
        if -1 not in ridge:  # Skip infinite ridges
            v0, v1 = vor.vertices[ridge[0]][:2], vor.vertices[ridge[1]][:2]
            clipped = _clip_segment_to_box(v0, v1, w, h)
            if clipped is not None:
                p0, p1 = clipped
                if np.linalg.norm(p1 - p0) > edge_tol:
                    edge_pairs.append((p0, p1))

    # Generate displacements for all edges
    n_total_pts = len(edge_pairs) * n_pts_per_side
    if n_pts_per_side > 0:
        if point_displacements is not None:
            all_disp = list(point_displacements)
        else:
            edge_lens = [np.linalg.norm(p2 - p1) for p1, p2 in edge_pairs]
            mean_len = np.mean(edge_lens) if edge_lens else min(w, h) / 5
            mag = perturbation * mean_len if perturbation > 0 else 0.3 * mean_len
            effective_seed = seed if seed is not None else 0
            all_disp = _auto_displacements(n_total_pts, mag, effective_seed)
    else:
        all_disp = []

    # Add edges with intermediates
    for idx, (p1, p2) in enumerate(edge_pairs):
        start = idx * n_pts_per_side
        end = start + n_pts_per_side
        edge_disp = all_disp[start:end] if n_pts_per_side > 0 else None
        _add_edge_with_intermediates(
            g, tuple(p1), tuple(p2), n_pts_per_side, edge_disp,
            radius, material, n_internal,
        )

    g._metadata["unit_type"] = "voronoi"
    g._metadata["n_seeds"] = n_seeds
    g._metadata["n_pts_per_side"] = n_pts_per_side
    return g


# Register Voronoi
_UNIT_FACTORIES["voronoi"] = _unit_voronoi
