"""
Tiling and welding engine for StructureGraph.

Tiles a unit cell into an NxM (2D) or NxMxK (3D) grid, automatically
welding shared nodes at cell boundaries.

Design
------
- Unit cell must have ``box_size`` set (the cell dimensions).
- Nodes on cell boundaries are automatically detected and merged.
- The result has correctly welded nodes where adjacent cells share edges.
- Boundary flags on outer-boundary nodes are set for downstream use.

The welding algorithm:
1. For each grid cell (i, j[, k]), copy the unit cell and translate it.
2. Add all nodes from all copies to a new graph (with merge=True).
   The StructureGraph's spatial hashing merges coincident nodes.
3. Set boundary flags on the outer boundary of the tiled structure.

Examples
--------
>>> from fibernet.core.structure_graph import StructureGraph
>>> from fibernet.core.tiling import tile_2d, tile_3d
>>> unit = make_square_unit(cell_size=10, n_internal=4)
>>> g = tile_2d(unit, grid=(3, 3))
>>> print(g)
StructureGraph(dim=2, nodes=..., edges=...)
"""

from __future__ import annotations

import copy
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from fibernet.core.structure_graph import StructureGraph
from fibernet.core.transforms import translate


def _detect_boundary_nodes(
    graph: StructureGraph,
    box_size: np.ndarray,
    tolerance: float,
) -> dict:
    """Detect which nodes lie on cell boundaries.

    Returns
    -------
    dict
        Maps node_id → tuple of booleans (x_min, x_max, y_min, y_max, z_min, z_max).
    """
    boundaries = {}
    w, h = box_size[0], box_size[1]
    d = box_size[2] if len(box_size) > 2 else 0.0

    for nid, node in graph.nodes.items():
        pos = node.position
        bnd = (
            abs(pos[0]) < tolerance,          # x_min
            abs(pos[0] - w) < tolerance,      # x_max
            abs(pos[1]) < tolerance,          # y_min
            abs(pos[1] - h) < tolerance,      # y_max
            abs(pos[2]) < tolerance if d > 0 else False,  # z_min
            abs(pos[2] - d) < tolerance if d > 0 else False,  # z_max
        )
        boundaries[nid] = bnd
    return boundaries


def tile_2d(
    unit_cell: StructureGraph,
    grid: Union[int, Tuple[int, int]] = (3, 3),
    box_size: Optional[Sequence[float]] = None,
    tolerance: Optional[float] = None,
) -> StructureGraph:
    """Tile a 2D unit cell into an NxM grid.

    Parameters
    ----------
    unit_cell : StructureGraph
        The unit cell. Must have ``box_size`` set, or pass ``box_size`` explicitly.
    grid : int or (nx, ny)
        Grid dimensions. If int, uses (grid, grid).
    box_size : (w, h), optional
        Override the unit cell's box_size.
    tolerance : float, optional
        Node merging tolerance. Defaults to unit_cell.tolerance.

    Returns
    -------
    StructureGraph
        The tiled structure with welded nodes.
    """
    if isinstance(grid, int):
        grid = (grid, grid)
    nx, ny = grid

    if box_size is not None:
        bs = np.asarray(box_size, dtype=np.float64)
        if len(bs) == 2:
            bs = np.append(bs, 0.0)
    elif unit_cell.box_size is not None:
        bs = unit_cell.box_size
    else:
        # Auto-detect from bounding box
        bb_min, bb_max = unit_cell.bounding_box()
        bs = bb_max - bb_min
        bs[2] = 0.0

    tol = tolerance or unit_cell.tolerance
    w, h = bs[0], bs[1]

    if w < 1e-12 or h < 1e-12:
        raise ValueError(f"box_size must be positive, got ({w}, {h})")

    # Detect boundary nodes in the unit cell
    cell_boundaries = _detect_boundary_nodes(unit_cell, bs, tol)

    # Create the tiled graph
    result = StructureGraph(dimension=2, tolerance=tol)
    result.box_size = np.array([w * nx, h * ny, 0.0])
    result._metadata = {
        "tiling": {"grid": list(grid), "unit_box": bs[:2].tolist()},
        **copy.deepcopy(unit_cell.metadata),
    }

    # Track node ID mapping for each cell copy
    for iy in range(ny):
        for ix in range(nx):
            offset = np.array([ix * w, iy * h, 0.0])

            # Map from unit cell node IDs to tiled graph node IDs
            id_map = {}
            for nid in sorted(unit_cell.nodes.keys()):
                node = unit_cell.nodes[nid]
                new_pos = node.position + offset

                # Compute boundary flags for the tiled structure
                abs_pos = new_pos
                bnd = (
                    abs(abs_pos[0]) < tol,                        # x_min of tiled
                    abs(abs_pos[0] - w * nx) < tol,               # x_max of tiled
                    abs(abs_pos[1]) < tol,                        # y_min of tiled
                    abs(abs_pos[1] - h * ny) < tol,               # y_max of tiled
                    False, False,
                )

                new_nid = result.add_node(
                    new_pos, boundary=bnd, merge=True,
                    **node.metadata,
                )
                id_map[nid] = new_nid

            # Add edges
            for eid in sorted(unit_cell.edges.keys()):
                edge = unit_cell.edges[eid]
                new_ip = None
                if edge.internal_points is not None:
                    new_ip = edge.internal_points + offset[None, :]
                result.add_edge(
                    id_map[edge.node_i], id_map[edge.node_j],
                    radius=edge.radius,
                    material=copy.deepcopy(edge.material),
                    internal_points=new_ip,
                    segments=edge.segments,
                    **edge.metadata,
                )

    return result


def tile_3d(
    unit_cell: StructureGraph,
    grid: Union[int, Tuple[int, int, int]] = (3, 3, 3),
    box_size: Optional[Sequence[float]] = None,
    tolerance: Optional[float] = None,
) -> StructureGraph:
    """Tile a 3D unit cell into an NxMxK grid.

    Parameters
    ----------
    unit_cell : StructureGraph
        The unit cell with ``box_size`` set.
    grid : int or (nx, ny, nz)
        Grid dimensions.
    box_size : (w, h, d), optional
        Override the unit cell's box_size.
    tolerance : float, optional
        Node merging tolerance.

    Returns
    -------
    StructureGraph
    """
    if isinstance(grid, int):
        grid = (grid, grid, grid)
    nx, ny, nz = grid

    if box_size is not None:
        bs = np.asarray(box_size, dtype=np.float64)
        if len(bs) == 2:
            bs = np.append(bs, 0.0)
    elif unit_cell.box_size is not None:
        bs = unit_cell.box_size
    else:
        bb_min, bb_max = unit_cell.bounding_box()
        bs = bb_max - bb_min

    tol = tolerance or unit_cell.tolerance
    w, h, d = bs[0], bs[1], bs[2]

    result = StructureGraph(dimension=3, tolerance=tol)
    result.box_size = np.array([w * nx, h * ny, d * nz])
    result._metadata = {
        "tiling": {"grid": list(grid), "unit_box": bs.tolist()},
        **copy.deepcopy(unit_cell.metadata),
    }

    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                offset = np.array([ix * w, iy * h, iz * d])

                id_map = {}
                for nid in sorted(unit_cell.nodes.keys()):
                    node = unit_cell.nodes[nid]
                    new_pos = node.position + offset

                    bnd = (
                        abs(new_pos[0]) < tol,
                        abs(new_pos[0] - w * nx) < tol,
                        abs(new_pos[1]) < tol,
                        abs(new_pos[1] - h * ny) < tol,
                        abs(new_pos[2]) < tol,
                        abs(new_pos[2] - d * nz) < tol,
                    )

                    new_nid = result.add_node(
                        new_pos, boundary=bnd, merge=True,
                        **node.metadata,
                    )
                    id_map[nid] = new_nid

                for eid in sorted(unit_cell.edges.keys()):
                    edge = unit_cell.edges[eid]
                    new_ip = None
                    if edge.internal_points is not None:
                        new_ip = edge.internal_points + offset[None, :]
                    result.add_edge(
                        id_map[edge.node_i], id_map[edge.node_j],
                        radius=edge.radius,
                        material=copy.deepcopy(edge.material),
                        internal_points=new_ip,
                        segments=edge.segments,
                        **edge.metadata,
                    )

    return result


def tile_with_transforms(
    unit_cell: StructureGraph,
    grid: Union[int, Tuple[int, int]] = (3, 3),
    box_size: Optional[Sequence[float]] = None,
    cell_transforms: Optional[dict] = None,
    tolerance: Optional[float] = None,
) -> StructureGraph:
    """Tile with per-cell transforms (e.g., alternating rotations).

    Parameters
    ----------
    unit_cell : StructureGraph
    grid : int or tuple
    box_size : optional
    cell_transforms : dict, optional
        Maps (ix, iy) → callable(StructureGraph) → StructureGraph.
        Applied to the unit cell before placing it at that grid position.
    tolerance : float, optional

    Returns
    -------
    StructureGraph

    Examples
    --------
    >>> transforms = {
    ...     (1, 0): lambda g: rotate(g, 90),
    ...     (0, 1): lambda g: mirror_x(g),
    ... }
    >>> g = tile_with_transforms(unit, grid=(2,2), cell_transforms=transforms)
    """
    if isinstance(grid, int):
        grid = (grid, grid)
    nx, ny = grid

    if box_size is not None:
        bs = np.asarray(box_size, dtype=np.float64)
        if len(bs) == 2:
            bs = np.append(bs, 0.0)
    elif unit_cell.box_size is not None:
        bs = unit_cell.box_size
    else:
        bb_min, bb_max = unit_cell.bounding_box()
        bs = bb_max - bb_min

    tol = tolerance or unit_cell.tolerance
    w, h = bs[0], bs[1]

    result = StructureGraph(dimension=unit_cell.dimension, tolerance=tol)
    result.box_size = np.array([w * nx, h * ny, 0.0])

    cell_transforms = cell_transforms or {}

    for iy in range(ny):
        for ix in range(nx):
            offset = np.array([ix * w, iy * h, 0.0])

            # Apply per-cell transform if specified
            cell = unit_cell
            if (ix, iy) in cell_transforms:
                cell = cell_transforms[(ix, iy)](cell)

            id_map = {}
            for nid in sorted(cell.nodes.keys()):
                node = cell.nodes[nid]
                new_pos = node.position + offset
                new_nid = result.add_node(new_pos, merge=True, **node.metadata)
                id_map[nid] = new_nid

            for eid in sorted(cell.edges.keys()):
                edge = cell.edges[eid]
                new_ip = None
                if edge.internal_points is not None:
                    new_ip = edge.internal_points + offset[None, :]
                result.add_edge(
                    id_map[edge.node_i], id_map[edge.node_j],
                    radius=edge.radius,
                    material=copy.deepcopy(edge.material),
                    internal_points=new_ip,
                    segments=edge.segments,
                    **edge.metadata,
                )

    return result


# ---------------------------------------------------------------------------
# Helper: fit unit cell to target box
# ---------------------------------------------------------------------------

def fit_unit_to_box(
    graph: StructureGraph,
    target_box: Sequence[float],
    padding: float = 0.0,
) -> StructureGraph:
    """Scale and center a unit cell to fit within a target box.

    Maintains aspect ratio. Useful for normalizing arbitrary point inputs
    to a standard cell size.

    Parameters
    ----------
    graph : StructureGraph
    target_box : (w, h[, d])
    padding : float
        Inset from box edges.

    Returns
    -------
    StructureGraph
    """
    from fibernet.core.transforms import scale as g_scale, translate as g_translate

    tb = np.asarray(target_box, dtype=np.float64)
    if len(tb) == 2:
        tb = np.append(tb, 0.0)

    bb_min, bb_max = graph.bounding_box()
    span = bb_max - bb_min

    # Avoid zero spans
    for i in range(3):
        if span[i] < 1e-12:
            span[i] = 1.0

    usable = tb - 2 * padding
    # Only consider active dimensions for scale factor (2D ignores z)
    ndim = graph.dimension if hasattr(graph, 'dimension') else (3 if tb[2] > 0 else 2)
    factors = usable[:ndim] / span[:ndim]
    f = float(min(factors))

    # Center
    scaled = g_scale(graph, f)
    bb_min2, bb_max2 = scaled.bounding_box()
    span2 = bb_max2 - bb_min2
    offset = (tb - span2) / 2 - bb_min2
    offset[2] = 0 if graph.dimension == 2 else offset[2]

    result = g_translate(scaled, offset)
    result.box_size = tb
    return result
