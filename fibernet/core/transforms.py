"""
Geometric transforms for StructureGraph.

All transforms are **functional** — they return a new StructureGraph
without modifying the input. Boundary flags are updated correctly
(e.g., mirror_x swaps x_min and x_max flags).

Supported transforms:
- translate: Shift all nodes by an offset vector
- rotate: Rotate around a point (2D) or axis (3D)
- mirror: Reflect across an axis (2D) or plane (3D)
- scale: Uniform or non-uniform scaling around a center point
- compose: Chain multiple transforms into a single operation

Examples
--------
>>> from fibernet.core.structure_graph import StructureGraph
>>> from fibernet.core.transforms import translate, rotate, mirror_x, compose
>>> g = StructureGraph(dimension=2)
>>> g.add_polyline([(0,0), (10,0), (10,10), (0,10)], closed=True)
>>> g2 = translate(g, [20, 0])
>>> g3 = rotate(g, angle=45)
>>> g4 = mirror_x(g)
>>> g5 = compose(g, translate([5,0]), rotate(90), mirror_x())
"""

from __future__ import annotations

import copy
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from fibernet.core.structure_graph import StructureGraph, SNode, SEdge


# ---------------------------------------------------------------------------
# Low-level matrix helpers
# ---------------------------------------------------------------------------

def _rotation_matrix_2d(angle_deg: float) -> np.ndarray:
    """2D rotation matrix for angle in degrees."""
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rotation_matrix_3d(angle_deg: float, axis: np.ndarray) -> np.ndarray:
    """3D rotation matrix (Rodrigues) for angle in degrees around axis."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    x, y, z = axis
    C = 1 - c
    return np.array([
        [c + x*x*C,    x*y*C - z*s,  x*z*C + y*s],
        [y*x*C + z*s,  c + y*y*C,    y*z*C - x*s],
        [z*x*C - y*s,  z*y*C + x*s,  c + z*z*C  ],
    ])


def _apply_transform_to_graph(
    graph: StructureGraph,
    transform_node: Callable[[np.ndarray], np.ndarray],
    transform_internal: Callable[[np.ndarray], np.ndarray],
    boundary_transform: Callable[[tuple], tuple],
) -> StructureGraph:
    """Apply a node-position transform + boundary transform to create new graph."""
    g = StructureGraph(
        dimension=graph.dimension,
        tolerance=graph.tolerance,
        box_size=graph.box_size,
    )
    g._metadata = copy.deepcopy(graph.metadata)

    id_map = {}
    for nid in sorted(graph.nodes.keys()):
        node = graph.nodes[nid]
        new_pos = transform_node(node.position.copy())
        new_bnd = boundary_transform(node.boundary)
        new_nid = g.add_node(new_pos, boundary=new_bnd, merge=False, **node.metadata)
        id_map[nid] = new_nid

    for eid in sorted(graph.edges.keys()):
        edge = graph.edges[eid]
        new_ip = None
        if edge.internal_points is not None:
            new_ip = np.array([transform_internal(pt.copy()) for pt in edge.internal_points])
        g.add_edge(
            id_map[edge.node_i], id_map[edge.node_j],
            radius=edge.radius,
            material=copy.deepcopy(edge.material),
            internal_points=new_ip,
            segments=edge.segments,
            **edge.metadata,
        )
    return g


# ---------------------------------------------------------------------------
# Public transforms
# ---------------------------------------------------------------------------

def translate(
    graph: StructureGraph,
    offset: Sequence[float],
) -> StructureGraph:
    """Translate all nodes by an offset vector.

    Parameters
    ----------
    graph : StructureGraph
    offset : (dx, dy[, dz])

    Returns
    -------
    StructureGraph
    """
    off = np.asarray(offset, dtype=np.float64).ravel()
    if len(off) == 2:
        off = np.append(off, 0.0)

    def t_node(pos):
        return pos + off

    def t_bnd(b):
        return b

    return _apply_transform_to_graph(graph, t_node, t_node, t_bnd)


def rotate(
    graph: StructureGraph,
    angle: float,
    center: Optional[Sequence[float]] = None,
    axis: Optional[Sequence[float]] = None,
) -> StructureGraph:
    """Rotate the graph around a center point (2D) or axis (3D).

    Parameters
    ----------
    graph : StructureGraph
    angle : float
        Rotation angle in degrees (counter-clockwise for 2D).
    center : (cx, cy[, cz]), optional
        Center of rotation. Defaults to origin.
    axis : (ax, ay, az), optional
        Rotation axis for 3D. Defaults to z-axis (0,0,1).

    Returns
    -------
    StructureGraph
    """
    ctr = np.zeros(3)
    if center is not None:
        ctr = np.asarray(center, dtype=np.float64).ravel()
        if len(ctr) == 2:
            ctr = np.append(ctr, 0.0)

    if graph.dimension == 2 or axis is None:
        R = _rotation_matrix_2d(angle)
    else:
        R = _rotation_matrix_3d(angle, np.asarray(axis, dtype=np.float64))

    def t_node(pos):
        return R @ (pos - ctr) + ctr

    def t_bnd(b):
        return b  # rotation doesn't change boundary flags semantically

    return _apply_transform_to_graph(graph, t_node, t_node, t_bnd)


def mirror(
    graph: StructureGraph,
    axis: str = "x",
    origin: float = 0.0,
) -> StructureGraph:
    """Reflect the graph across a line (2D) or plane (3D).

    Parameters
    ----------
    graph : StructureGraph
    axis : str
        Mirror axis: 'x', 'y', or 'z'. Reflects the coordinate along that axis.
        E.g., mirror(axis='x') flips x-coordinates: x → 2*origin - x.
    origin : float
        Position of the mirror line/plane along the axis.

    Returns
    -------
    StructureGraph
    """
    ax_idx = {"x": 0, "y": 1, "z": 2}[axis.lower()]

    def t_node(pos):
        pos[ax_idx] = 2 * origin - pos[ax_idx]
        return pos

    # Boundary flag swap
    def t_bnd(b):
        b = list(b)
        # boundary = (x_min, x_max, y_min, y_max, z_min, z_max)
        idx_lo = ax_idx * 2
        idx_hi = idx_lo + 1
        b[idx_lo], b[idx_hi] = b[idx_hi], b[idx_lo]
        return tuple(b)

    return _apply_transform_to_graph(graph, t_node, t_node, t_bnd)


def mirror_x(graph: StructureGraph, origin: float = 0.0) -> StructureGraph:
    """Mirror across the y-axis (flip x). Shortcut for mirror(axis='x')."""
    return mirror(graph, axis="x", origin=origin)


def mirror_y(graph: StructureGraph, origin: float = 0.0) -> StructureGraph:
    """Mirror across the x-axis (flip y). Shortcut for mirror(axis='y')."""
    return mirror(graph, axis="y", origin=origin)


def mirror_z(graph: StructureGraph, origin: float = 0.0) -> StructureGraph:
    """Mirror across the xy-plane (flip z). Shortcut for mirror(axis='z')."""
    return mirror(graph, axis="z", origin=origin)


def scale(
    graph: StructureGraph,
    factor: Union[float, Sequence[float]],
    center: Optional[Sequence[float]] = None,
) -> StructureGraph:
    """Scale the graph uniformly or non-uniformly.

    Parameters
    ----------
    graph : StructureGraph
    factor : float or (sx, sy[, sz])
        Scale factor(s). If float, uniform scaling.
    center : (cx, cy[, cz]), optional
        Center of scaling. Defaults to origin.

    Returns
    -------
    StructureGraph
    """
    ctr = np.zeros(3)
    if center is not None:
        ctr = np.asarray(center, dtype=np.float64).ravel()
        if len(ctr) == 2:
            ctr = np.append(ctr, 0.0)

    if isinstance(factor, (int, float)):
        sf = np.array([factor, factor, factor], dtype=np.float64)
    else:
        sf = np.asarray(factor, dtype=np.float64).ravel()
        if len(sf) == 2:
            sf = np.append(sf, 1.0)

    def t_node(pos):
        return (pos - ctr) * sf + ctr

    # Scale radius of edges if uniform
    g = _apply_transform_to_graph(graph, t_node, t_node, lambda b: b)

    # Adjust edge radii for uniform scaling
    if isinstance(factor, (int, float)):
        for eid, edge in g._edges.items():
            edge.radius = abs(factor) * edge.radius

    # Update box_size if present
    if g.box_size is not None:
        g.box_size = g.box_size * sf

    return g


def compose(
    graph: StructureGraph,
    *transforms: Callable[[StructureGraph], StructureGraph],
) -> StructureGraph:
    """Apply a sequence of transforms in order.

    Parameters
    ----------
    graph : StructureGraph
    *transforms : callables
        Each callable takes a StructureGraph and returns a StructureGraph.

    Returns
    -------
    StructureGraph

    Examples
    --------
    >>> g2 = compose(g, translate([5, 0]), rotate(45), mirror_x())
    """
    result = graph
    for t in transforms:
        result = t(result)
    return result


# ---------------------------------------------------------------------------
# Convenience: transform factory functions (for use with compose)
# ---------------------------------------------------------------------------

def make_translate(offset: Sequence[float]) -> Callable:
    """Create a translate transform function."""
    return lambda g: translate(g, offset)


def make_rotate(angle: float, center=None, axis=None) -> Callable:
    """Create a rotate transform function."""
    return lambda g: rotate(g, angle, center=center, axis=axis)


def make_mirror(axis: str = "x", origin: float = 0.0) -> Callable:
    """Create a mirror transform function."""
    return lambda g: mirror(g, axis=axis, origin=origin)


def make_scale(factor, center=None) -> Callable:
    """Create a scale transform function."""
    return lambda g: scale(g, factor, center=center)
