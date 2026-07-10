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
        defaults to seed=0 with magnitude=0.05*edge_length.

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
            magnitude = 0.05 * edge_len  # 5% of edge length
            disp = _cn_symmetric_displacements(
                perimeter, polygon_type, n_pts_per_side, magnitude, seed,
            )
        else:
            # Default: seed=0, 5% edge length
            edge_len = np.sqrt((corners[1][0] - corners[0][0])**2 +
                             (corners[1][1] - corners[0][1])**2)
            magnitude = 0.05 * edge_len
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
            mag = perturbation * edge_len if perturbation > 0 else 0.05 * edge_len
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
            mag = perturbation * edge_len if perturbation > 0 else 0.05 * edge_len
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
            mag = perturbation * edge_len if perturbation > 0 else 0.05 * edge_len
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
            mag = perturbation * edge_len if perturbation > 0 else 0.05 * edge_len
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
            mag = perturbation * edge_len if perturbation > 0 else 0.05 * edge_len
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
            mag = perturbation * edge_len if perturbation > 0 else 0.05 * edge_len
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
            mag = perturbation * edge_len if perturbation > 0 else 0.05 * edge_len
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
            mag = perturbation * edge_len if perturbation > 0 else 0.05 * edge_len
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
            mag = perturbation * edge_len if perturbation > 0 else 0.05 * edge_len
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
        # Custom points
        pts = [tuple(p) for p in points]
        if fit_to_box:
            raw = StructureGraph(dimension=2, box_size=list(box) + [0.0])
            raw.add_polyline(pts, closed=closed, radius=radius, material=mat,
                             n_internal=n_internal)
            unit_cell = fit_unit_to_box(raw, target_box=list(box) + [0.0])
        else:
            unit_cell = StructureGraph(dimension=2, box_size=list(box) + [0.0])
            unit_cell.add_polyline(pts, closed=closed, radius=radius, material=mat,
                                   n_internal=n_internal)

    # Step 2: Apply transforms
    if rotation != 0.0:
        center = np.array([box[0] / 2, box[1] / 2, 0.0])
        unit_cell = rotate(unit_cell, rotation, center=center)
    if mirror_x:
        unit_cell = _mirror_x(unit_cell, origin=box[0] / 2)
    if mirror_y:
        unit_cell = _mirror_y(unit_cell, origin=box[1] / 2)

    # Step 3: Boundary mode check
    if boundary_mode == "error":
        _check_boundary_contact(unit_cell, box)
    elif boundary_mode == "extend":
        unit_cell = _extend_to_boundary(unit_cell, box, radius, mat, n_internal)

    # Step 4: Tile
    result = tile_2d(unit_cell, grid=grid, box_size=list(box) + [0.0])
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


def pattern_3d(
    *,
    unit: str = "cubic",
    box: Tuple[float, float, float] = (10.0, 10.0, 10.0),
    grid: Union[int, Tuple[int, int, int]] = (3, 3, 3),
    n_internal: int = 0,
    n_pts_per_side: int = 0,
    point_displacements: Optional[Sequence[Tuple[float, float, float]]] = None,
    radius: float = 0.1,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
    unit_kwargs: Optional[Dict[str, Any]] = None,
) -> StructureGraph:
    """Generate a 3D periodic structure.

    Parameters
    ----------
    unit : str
        Built-in 3D unit: "cubic", "octet", "diamond_3d".
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

    Returns
    -------
    StructureGraph
    """
    mat = material or Material()
    w, h, d = box

    if unit == "cubic":
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

    elif unit == "octet":
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

    elif unit == "diamond_3d":
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
    else:
        raise ValueError(f"Unknown 3D unit '{unit}'. Available: cubic, octet, diamond_3d")

    g._metadata["unit_type"] = unit
    g._metadata["n_pts_per_side"] = n_pts_per_side
    result = tile_3d(g, grid=grid, box_size=[w, h, d])
    result._metadata["pattern"] = {
        "unit": unit, "box": list(box),
        "grid": list(grid) if isinstance(grid, (list, tuple)) else [grid, grid, grid],
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
            mag = 0.05  # 5% default
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
