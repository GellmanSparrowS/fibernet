"""
Pattern Engine — Unified structure generation for FiberNet.

The Pattern Engine implements a single, coherent paradigm for generating
all periodic 2D/3D structures:

    **Base Unit + Transform + Tiling + Welding**

This covers the vast majority of known metamaterial and lattice structures.

Design Principles
-----------------
- **Deterministic**: No randomness unless explicitly set via ``seed`` + ``perturbation``.
- **Programmable**: Full control over positions, transforms, boundary behavior.
- **Unified API**: ``pattern_2d()`` and ``pattern_3d()`` for all structures.
- **Edge discretization**: Every edge can carry N internal points for deformation.
- **Boundary-aware**: Nodes on cell boundaries are correctly welded during tiling.

Unit Types
----------
Built-in presets:
- ``"square"``: Square frame (4 edges)
- ``"triangle"``: Equilateral triangle (3 edges)
- ``"hexagon"``: Regular hexagon (6 edges)
- ``"honeycomb"``: Honeycomb cell (hexagonal with proper tiling)
- ``"kagome"``: Kagome lattice (triangle + hexagon)
- ``"diamond"``: Diamond/triangular dual
- ``"reentrant"``: Reentrant honeycomb (auxetic)
- ``"chiral"``: Chiral honeycomb
- ``"star"``: Star-shaped unit
- ``"cross"``: Plus/cross shape
- ``"missing_rib"``: Missing-rib honeycomb

Custom units:
- ``points=[...]``: List of (x,y) or (x,y,z) coordinates
- ``closed=True/False``: Whether to close the polyline

API
---
>>> from fibernet.gen.pattern import pattern_2d, pattern_3d
>>> # Built-in
>>> g = pattern_2d(unit="honeycomb", box=(10, 10), grid=(3, 3))
>>> # Custom
>>> g = pattern_2d(points=[(0,0),(5,0),(5,5),(0,5)], closed=True, box=(10,10), grid=(3,3))
>>> # With transforms
>>> g = pattern_2d(unit="square", box=(10,10), grid=(3,3), mirror_x=True, mirror_y=True)
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
# Built-in unit factories
# ======================================================================

def _unit_square(box: Tuple[float, float], n_internal: int = 0,
                 radius: float = 0.1, material: Material = None) -> StructureGraph:
    """Square frame unit cell."""
    w, h = box
    g = StructureGraph(dimension=2, box_size=[w, h])
    g.add_polyline([(0, 0), (w, 0), (w, h), (0, h)], closed=True,
                   radius=radius, material=material, n_internal=n_internal)
    g._metadata["unit_type"] = "square"
    return g


def _unit_triangle(box: Tuple[float, float], n_internal: int = 0,
                   radius: float = 0.1, material: Material = None) -> StructureGraph:
    """Triangular lattice unit cell — two triangles forming a rhombus.

    The unit cell contains two triangles sharing a diagonal edge,
    creating a rhombus that tiles into the triangular lattice.
    """
    w, h = box
    g = StructureGraph(dimension=2, box_size=[w, h])
    # Four corners + center diagonal
    n00 = g.add_node([0, 0])
    n10 = g.add_node([w, 0])
    n01 = g.add_node([0, h])
    n11 = g.add_node([w, h])
    # Bottom triangle
    g.add_edge(n00, n10, radius=radius, material=material, n_internal=n_internal)
    g.add_edge(n10, n01, radius=radius, material=material, n_internal=n_internal)
    g.add_edge(n01, n00, radius=radius, material=material, n_internal=n_internal)
    # Top triangle (sharing the diagonal n10-n01)
    g.add_edge(n10, n11, radius=radius, material=material, n_internal=n_internal)
    g.add_edge(n11, n01, radius=radius, material=material, n_internal=n_internal)
    g._metadata["unit_type"] = "triangle"
    return g


def _unit_hexagon(box: Tuple[float, float], n_internal: int = 0,
                  radius: float = 0.1, material: Material = None) -> StructureGraph:
    """Regular hexagon unit cell."""
    w, h = box
    g = StructureGraph(dimension=2, box_size=[w, h])
    pts = [
        (w / 2, 0), (w, h / 4), (w, 3 * h / 4),
        (w / 2, h), (0, 3 * h / 4), (0, h / 4),
    ]
    g.add_polyline(pts, closed=True,
                   radius=radius, material=material, n_internal=n_internal)
    g._metadata["unit_type"] = "hexagon"
    return g


def _unit_honeycomb(box: Tuple[float, float], n_internal: int = 0,
                    radius: float = 0.1, material: Material = None) -> StructureGraph:
    """Honeycomb unit cell — hexagonal lattice with proper periodicity.

    The unit cell is a rectangular region containing one hexagon and
    half-struts connecting to adjacent cells. When tiled, the half-struts
    from adjacent cells merge into full struts, forming the honeycomb.
    """
    w, h = box
    g = StructureGraph(dimension=2, box_size=[w, h])
    # Hexagonal unit with vertices touching cell boundaries
    # Standard honeycomb: vertical walls + diagonal connections
    # Cell width = w, cell height = h
    # The hexagon is oriented with flat sides top/bottom
    a = w / 2  # half-width
    b = h / 4  # quarter-height

    # Vertical left wall
    g.add_polyline([(0, b), (0, 3 * b)], radius=radius, material=material,
                   n_internal=n_internal)
    # Vertical right wall
    g.add_polyline([(w, b), (w, 3 * b)], radius=radius, material=material,
                   n_internal=n_internal)
    # Top-left diagonal
    g.add_polyline([(0, b), (a, 0)], radius=radius, material=material,
                   n_internal=n_internal)
    # Top-right diagonal
    g.add_polyline([(w, b), (a, 0)], radius=radius, material=material,
                   n_internal=n_internal)
    # Bottom-left diagonal
    g.add_polyline([(0, 3 * b), (a, h)], radius=radius, material=material,
                   n_internal=n_internal)
    # Bottom-right diagonal
    g.add_polyline([(w, 3 * b), (a, h)], radius=radius, material=material,
                   n_internal=n_internal)
    g._metadata["unit_type"] = "honeycomb"
    return g


def _unit_kagome(box: Tuple[float, float], n_internal: int = 0,
                 radius: float = 0.1, material: Material = None) -> StructureGraph:
    """Kagome lattice — triangles + hexagons with proper tiling connectivity.

    The unit cell contains midpoints on all edges connected in a star pattern
    through the center, plus corner-to-midpoint connections.
    """
    w, h = box
    g = StructureGraph(dimension=2, box_size=[w, h])
    # Corner nodes (shared with adjacent cells)
    n00 = g.add_node([0, 0])
    n10 = g.add_node([w, 0])
    n01 = g.add_node([0, h])
    n11 = g.add_node([w, h])
    # Edge midpoints (shared with adjacent cells)
    nm_b = g.add_node([w / 2, 0])
    nm_t = g.add_node([w / 2, h])
    nm_l = g.add_node([0, h / 2])
    nm_r = g.add_node([w, h / 2])
    # Center node
    nc = g.add_node([w / 2, h / 2])

    # Kagome connectivity:
    # 1. Corner-to-midpoint edges (boundary edges that tile properly)
    g.add_edge(n00, nm_b, radius=radius, material=material, n_internal=n_internal)
    g.add_edge(n10, nm_b, radius=radius, material=material, n_internal=n_internal)
    g.add_edge(n00, nm_l, radius=radius, material=material, n_internal=n_internal)
    g.add_edge(n01, nm_l, radius=radius, material=material, n_internal=n_internal)
    g.add_edge(n10, nm_r, radius=radius, material=material, n_internal=n_internal)
    g.add_edge(n11, nm_r, radius=radius, material=material, n_internal=n_internal)
    g.add_edge(n01, nm_t, radius=radius, material=material, n_internal=n_internal)
    g.add_edge(n11, nm_t, radius=radius, material=material, n_internal=n_internal)
    # 2. Midpoint-to-center (inner star)
    g.add_edge(nm_b, nc, radius=radius, material=material, n_internal=n_internal)
    g.add_edge(nm_t, nc, radius=radius, material=material, n_internal=n_internal)
    g.add_edge(nm_l, nc, radius=radius, material=material, n_internal=n_internal)
    g.add_edge(nm_r, nc, radius=radius, material=material, n_internal=n_internal)
    g._metadata["unit_type"] = "kagome"
    return g


def _unit_reentrant(box: Tuple[float, float], n_internal: int = 0,
                    radius: float = 0.1, material: Material = None,
                    angle: float = 15.0) -> StructureGraph:
    """Reentrant (arrowhead) honeycomb — auxetic structure.

    The diagonals bend inward through an indentation point at the cell center,
    creating the concave cell shape that produces negative Poisson ratio.
    Angle parameter controls indentation depth.
    """
    w, h = box
    g = StructureGraph(dimension=2, box_size=[w, h])

    b = h / 4
    # Indentation depth: larger angle = more reentrant = more auxetic
    indent = (w / 2) * np.tan(np.radians(min(angle, 40)))

    # Boundary nodes (for tiling connectivity)
    n_l_lo = g.add_node([0, b])
    n_l_hi = g.add_node([0, 3 * b])
    n_r_lo = g.add_node([w, b])
    n_r_hi = g.add_node([w, 3 * b])
    n_b = g.add_node([w / 2, 0])
    n_t = g.add_node([w / 2, h])

    # Internal indentation nodes
    n_indent_lo = g.add_node([w / 2, b + indent])
    n_indent_hi = g.add_node([w / 2, 3 * b - indent])

    # Vertical walls
    g.add_edge(n_l_lo, n_l_hi, radius=radius, material=material, n_internal=n_internal)
    g.add_edge(n_r_lo, n_r_hi, radius=radius, material=material, n_internal=n_internal)

    # Lower diagonals: wall → indent → bottom (V-shape bending inward)
    g.add_edge(n_l_lo, n_indent_lo, radius=radius, material=material, n_internal=n_internal)
    g.add_edge(n_r_lo, n_indent_lo, radius=radius, material=material, n_internal=n_internal)
    g.add_edge(n_indent_lo, n_b, radius=radius, material=material, n_internal=n_internal)

    # Upper diagonals: wall → indent → top (V-shape bending inward)
    g.add_edge(n_l_hi, n_indent_hi, radius=radius, material=material, n_internal=n_internal)
    g.add_edge(n_r_hi, n_indent_hi, radius=radius, material=material, n_internal=n_internal)
    g.add_edge(n_indent_hi, n_t, radius=radius, material=material, n_internal=n_internal)

    g._metadata["unit_type"] = "reentrant"
    g._metadata["reentrant_angle"] = angle
    return g


def _unit_chiral(box: Tuple[float, float], n_internal: int = 0,
                 radius: float = 0.1, material: Material = None,
                 node_radius: float = None) -> StructureGraph:
    """Chiral honeycomb — nodes connected by tangential ligaments.

    Creates a unit with a central ring node connected to surrounding
    nodes by tangent ligaments, producing chirality.
    """
    w, h = box
    g = StructureGraph(dimension=2, box_size=[w, h])

    if node_radius is None:
        node_radius = min(w, h) * 0.15

    cx, cy = w / 2, h / 2
    # Central ring: represented as a small polygon
    n_ring_pts = 8
    ring_nodes = []
    for i in range(n_ring_pts):
        theta = 2 * np.pi * i / n_ring_pts
        px = cx + node_radius * np.cos(theta)
        py = cy + node_radius * np.sin(theta)
        nid = g.add_node([px, py])
        ring_nodes.append(nid)
    # Connect ring
    for i in range(n_ring_pts):
        g.add_edge(ring_nodes[i], ring_nodes[(i + 1) % n_ring_pts],
                   radius=radius, material=material, n_internal=n_internal)

    # Corner nodes and tangent ligaments
    corners = [(0, 0), (w, 0), (w, h), (0, h)]
    for i, (cx_c, cy_c) in enumerate(corners):
        cn = g.add_node([cx_c, cy_c])
        # Connect to nearest ring point with tangent ligament
        ring_idx = (i * n_ring_pts) // 4
        rn = ring_nodes[ring_idx]
        g.add_edge(cn, rn, radius=radius, material=material, n_internal=n_internal)
        # Also connect to adjacent ring point for chirality
        rn2 = ring_nodes[(ring_idx + 1) % n_ring_pts]
        g.add_edge(cn, rn2, radius=radius * 0.8, material=material, n_internal=n_internal)

    g._metadata["unit_type"] = "chiral"
    return g


def _unit_star(box: Tuple[float, float], n_internal: int = 0,
               radius: float = 0.1, material: Material = None,
               n_arms: int = 4) -> StructureGraph:
    """Star-shaped unit cell."""
    w, h = box
    g = StructureGraph(dimension=2, box_size=[w, h])
    cx, cy = w / 2, h / 2
    r_outer = min(w, h) / 2
    r_inner = r_outer * 0.4

    pts = []
    for i in range(2 * n_arms):
        theta = np.pi * i / n_arms - np.pi / 2
        r = r_outer if i % 2 == 0 else r_inner
        px = cx + r * np.cos(theta)
        py = cy + r * np.sin(theta)
        pts.append((px, py))

    g.add_polyline(pts, closed=True, radius=radius, material=material,
                   n_internal=n_internal)
    g._metadata["unit_type"] = "star"
    g._metadata["n_arms"] = n_arms
    return g


def _unit_cross(box: Tuple[float, float], n_internal: int = 0,
                radius: float = 0.1, material: Material = None,
                arm_width: float = 0.3) -> StructureGraph:
    """Plus/cross shape unit cell."""
    w, h = box
    g = StructureGraph(dimension=2, box_size=[w, h])
    aw = arm_width * min(w, h) / 2
    cx, cy = w / 2, h / 2

    # Cross outline: 12 vertices
    pts = [
        (cx - aw, 0), (cx + aw, 0),
        (cx + aw, cy - aw), (w, cy - aw),
        (w, cy + aw), (cx + aw, cy + aw),
        (cx + aw, h), (cx - aw, h),
        (cx - aw, cy + aw), (0, cy + aw),
        (0, cy - aw), (cx - aw, cy - aw),
    ]
    g.add_polyline(pts, closed=True, radius=radius, material=material,
                   n_internal=n_internal)
    g._metadata["unit_type"] = "cross"
    return g


def _unit_missing_rib(box: Tuple[float, float], n_internal: int = 0,
                      radius: float = 0.1, material: Material = None) -> StructureGraph:
    """Missing-rib honeycomb — honeycomb with one horizontal rib removed.

    Produces auxetic behavior through rib removal.
    """
    g = _unit_honeycomb(box, n_internal, radius, material)
    # Remove the middle horizontal edge if present
    # In our honeycomb, we don't have explicit horizontal ribs,
    # so we modify the diagonal angles instead
    # Alternative: create honeycomb with a missing vertical wall segment
    w, h = box
    g2 = StructureGraph(dimension=2, box_size=[w, h])
    a = w / 2
    b = h / 4
    # Only right vertical wall (left removed)
    g2.add_polyline([(w, b), (w, 3 * b)], radius=radius, material=material,
                    n_internal=n_internal)
    # All diagonals
    g2.add_polyline([(0, b), (a, 0)], radius=radius, material=material, n_internal=n_internal)
    g2.add_polyline([(w, b), (a, 0)], radius=radius, material=material, n_internal=n_internal)
    g2.add_polyline([(0, 3 * b), (a, h)], radius=radius, material=material, n_internal=n_internal)
    g2.add_polyline([(w, 3 * b), (a, h)], radius=radius, material=material, n_internal=n_internal)
    g2._metadata["unit_type"] = "missing_rib"
    return g2


def _unit_diamond(box: Tuple[float, float], n_internal: int = 0,
                  radius: float = 0.1, material: Material = None) -> StructureGraph:
    """Diamond (rhombus) unit cell."""
    w, h = box
    g = StructureGraph(dimension=2, box_size=[w, h])
    g.add_polyline([(w / 2, 0), (w, h / 2), (w / 2, h), (0, h / 2)],
                   closed=True, radius=radius, material=material,
                   n_internal=n_internal)
    g._metadata["unit_type"] = "diamond"
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
    """Register a custom unit factory.

    Parameters
    ----------
    name : str
        Unit name (lowercase).
    factory : callable
        Factory function: (box, n_internal, radius, material, **kwargs) → StructureGraph.
    """
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
    """Generate a 2D periodic structure using the Base Unit + Transform + Tiling paradigm.

    Parameters
    ----------
    unit : str, optional
        Built-in unit type name. See ``list_units()``.
    points : sequence of (x, y), optional
        Custom polyline points. Mutually exclusive with ``unit``.
    closed : bool
        Whether the custom polyline is closed.
    box : (w, h)
        Unit cell dimensions.
    grid : int or (nx, ny)
        Tiling grid size.
    n_internal : int
        Number of internal points per edge (for deformation/visualization).
    radius : float
        Edge/beam radius.
    material : Material, optional
        Beam material.
    mirror_x : bool
        Mirror the unit across the y-axis before tiling.
    mirror_y : bool
        Mirror the unit across the x-axis before tiling.
    rotation : float
        Rotate the unit by this angle (degrees) before tiling.
    perturbation : float
        Random perturbation magnitude (fraction of edge length).
        Only applied when ``seed`` is set.
    seed : int, optional
        Random seed for perturbation. If None, no perturbation.
    boundary_mode : str
        How to handle units that don't touch cell boundaries:
        - ``"none"``: No special handling.
        - ``"extend"``: Add bridge segments to connect to cell boundaries.
        - ``"error"``: Raise ValueError if unit doesn't touch boundaries.
    fit_to_box : bool
        Scale custom points to fit within the box (maintaining aspect ratio).
    unit_kwargs : dict, optional
        Extra kwargs passed to the unit factory (e.g., ``angle=20`` for reentrant).

    Returns
    -------
    StructureGraph

    Examples
    --------
    >>> g = pattern_2d(unit="honeycomb", box=(10, 10), grid=(3, 3), n_internal=8)
    >>> g = pattern_2d(points=[(0,0),(5,8.66),(10,0)], closed=True, box=(10,10), grid=(4,4))
    >>> g = pattern_2d(unit="reentrant", box=(10,10), grid=(3,3), unit_kwargs={"angle": 20})
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
        unit_cell = _UNIT_FACTORIES[unit_name](
            box=box, n_internal=n_internal, radius=radius, material=mat, **ukw,
        )
    else:
        # Custom points
        pts = [tuple(p) for p in points]
        if fit_to_box:
            # Create raw unit, then fit to box
            raw = StructureGraph(dimension=2, box_size=list(box) + [0.0])
            raw.add_polyline(pts, closed=closed, radius=radius, material=mat,
                             n_internal=n_internal)
            unit_cell = fit_unit_to_box(raw, target_box=list(box) + [0.0])
        else:
            unit_cell = StructureGraph(dimension=2, box_size=list(box) + [0.0])
            unit_cell.add_polyline(pts, closed=closed, radius=radius, material=mat,
                                   n_internal=n_internal)

    # Step 2: Apply perturbation (if seed is set)
    if perturbation > 0 and seed is not None:
        rng = np.random.default_rng(seed)
        edge_lens = unit_cell.edge_lengths()
        mean_len = edge_lens.mean() if len(edge_lens) > 0 else 1.0
        noise = perturbation * mean_len
        for nid, node in unit_cell.nodes.items():
            if not any(node.boundary):  # Don't perturb boundary nodes
                node.position[:2] += rng.uniform(-noise, noise, size=2)

    # Step 3: Apply transforms
    if rotation != 0.0:
        center = np.array([box[0] / 2, box[1] / 2, 0.0])
        unit_cell = rotate(unit_cell, rotation, center=center)
    if mirror_x:
        unit_cell = _mirror_x(unit_cell, origin=box[0] / 2)
    if mirror_y:
        unit_cell = _mirror_y(unit_cell, origin=box[1] / 2)

    # Step 4: Boundary mode check
    if boundary_mode == "error":
        _check_boundary_contact(unit_cell, box)
    elif boundary_mode == "extend":
        unit_cell = _extend_to_boundary(unit_cell, box, radius, mat, n_internal)

    # Step 5: Tile
    result = tile_2d(unit_cell, grid=grid, box_size=list(box) + [0.0])
    result._metadata["pattern"] = {
        "unit": unit or "custom",
        "box": list(box),
        "grid": list(grid) if isinstance(grid, (list, tuple)) else [grid, grid],
        "n_internal": n_internal,
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
    radius: float = 0.1,
    material: Optional[Material] = None,
    unit_kwargs: Optional[Dict[str, Any]] = None,
) -> StructureGraph:
    """Generate a 3D periodic structure.

    Parameters
    ----------
    unit : str
        Built-in 3D unit: "cubic", "octet", "diamond_3d", "bcc", "fcc".
    box : (w, h, d)
        Unit cell dimensions.
    grid : int or (nx, ny, nz)
        Grid dimensions.
    n_internal : int
        Internal points per edge.
    radius : float
        Beam radius.
    material : Material, optional
        Beam material.

    Returns
    -------
    StructureGraph
    """
    mat = material or Material()
    w, h, d = box

    if unit == "cubic":
        g = StructureGraph(dimension=3, box_size=[w, h, d])
        # 8 corners
        corners = [
            (0, 0, 0), (w, 0, 0), (w, h, 0), (0, h, 0),
            (0, 0, d), (w, 0, d), (w, h, d), (0, h, d),
        ]
        nids = [g.add_node(c) for c in corners]
        # 12 edges of cube
        edges_12 = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
            (4, 5), (5, 6), (6, 7), (7, 4),  # top
            (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
        ]
        for i, j in edges_12:
            g.add_edge(nids[i], nids[j], radius=radius, material=mat,
                       n_internal=n_internal)

    elif unit == "octet":
        g = StructureGraph(dimension=3, box_size=[w, h, d])
        # Octet truss: cube + face diagonals + body diagonal
        corners = [
            (0, 0, 0), (w, 0, 0), (w, h, 0), (0, h, 0),
            (0, 0, d), (w, 0, d), (w, h, d), (0, h, d),
        ]
        nids = [g.add_node(c) for c in corners]
        # 12 cube edges
        for i, j in [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]:
            g.add_edge(nids[i], nids[j], radius=radius, material=mat,
                       n_internal=n_internal)
        # 12 face diagonals
        center = g.add_node([w / 2, h / 2, d / 2])
        for nid in nids:
            g.add_edge(nid, center, radius=radius * 0.7, material=mat,
                       n_internal=n_internal)

    elif unit == "diamond_3d":
        g = StructureGraph(dimension=3, box_size=[w, h, d])
        # Diamond lattice: FCC-like with tetrahedral bonds
        corners = [
            (0, 0, 0), (w, 0, 0), (w, h, 0), (0, h, 0),
            (0, 0, d), (w, 0, d), (w, h, d), (0, h, d),
        ]
        nids = [g.add_node(c) for c in corners]
        # Face centers
        fc = [
            g.add_node([w / 2, h / 2, 0]),   # bottom
            g.add_node([w / 2, h / 2, d]),   # top
            g.add_node([w / 2, 0, d / 2]),   # front
            g.add_node([w / 2, h, d / 2]),   # back
            g.add_node([0, h / 2, d / 2]),   # left
            g.add_node([w, h / 2, d / 2]),   # right
        ]
        # Connect face centers to adjacent corners
        face_to_corners = [
            (0, [0, 1, 2, 3]),   # bottom face
            (1, [4, 5, 6, 7]),   # top face
            (2, [0, 1, 4, 5]),   # front face
            (3, [2, 3, 6, 7]),   # back face
            (4, [0, 3, 4, 7]),   # left face
            (5, [1, 2, 5, 6]),   # right face
        ]
        for fc_idx, c_list in face_to_corners:
            for c_idx in c_list:
                g.add_edge(fc[fc_idx], nids[c_idx], radius=radius, material=mat,
                           n_internal=n_internal)
    else:
        raise ValueError(f"Unknown 3D unit '{unit}'. Available: cubic, octet, diamond_3d")

    g._metadata["unit_type"] = unit
    result = tile_3d(g, grid=grid, box_size=[w, h, d])
    result._metadata["pattern"] = {
        "unit": unit, "box": list(box),
        "grid": list(grid) if isinstance(grid, (list, tuple)) else [grid, grid, grid],
    }
    return result


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

    # Find centroid of existing nodes
    if not g.nodes:
        return g
    pos = g.node_positions()
    centroid = pos.mean(axis=0)

    # Find boundary target points
    targets = [
        (0, centroid[1], 0),       # x_min
        (w, centroid[1], 0),       # x_max
        (centroid[0], 0, 0),       # y_min
        (centroid[0], h, 0),       # y_max
    ]

    # For each boundary, find nearest node and add bridge
    for tx, ty, tz in targets:
        target = np.array([tx, ty, tz])
        # Check if any node already touches this boundary
        already = False
        for node in g.nodes.values():
            if np.linalg.norm(node.position - target) < graph.tolerance * 10:
                already = True
                break
        if already:
            continue

        # Find nearest node to target
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
