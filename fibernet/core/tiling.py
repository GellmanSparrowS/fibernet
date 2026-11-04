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
from collections import deque
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from fibernet.core.material import Material
from fibernet.core.structure_graph import StructureGraph
from fibernet.core.transforms import translate
from fibernet.core._grid_contract import normalize_grid


_NUMPY_SCALARS = tuple(np.dtype(name).type for name in (
    'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32',
    'uint64', 'float16', 'float32', 'float64', 'complex64', 'complex128',
))


def _known_equal(left, right, depth=0):
    """Compare plain data without consulting arbitrary user equality hooks.

    Unrecognized values and cyclic/deep containers are not evidence that two
    physical fibers are the same, so they conservatively prevent deduplication.
    """
    if depth > 32:
        return False
    numeric = (int, float, complex) + _NUMPY_SCALARS
    if (any(type(left) is known for known in numeric)
            and any(type(right) is known for known in numeric)):
        # Built-in int/float comparison retains exact integer identity, unlike
        # NumPy promotion of a large integer and a float to the same float64.
        a = left.item() if isinstance(left, np.generic) else left
        b = right.item() if isinstance(right, np.generic) else right
        return bool(a == b)
    if type(left) is not type(right):
        return False
    kind = type(left)
    if left is None:
        return True
    if any(kind is known for known in (bool, np.bool_, str, bytes)):
        return bool(left == right)
    if kind is np.ndarray:
        if left.shape != right.shape or left.dtype.kind not in 'biufcSU' or right.dtype.kind not in 'biufcSU':
            return False
        if left.dtype == right.dtype:
            return bool(np.array_equal(left, right))
        if left.dtype.kind in 'iufc' and right.dtype.kind in 'iufc':
            return all(_known_equal(a, b, depth + 1) for a, b in zip(left.flat, right.flat))
        return False
    if kind is list or kind is tuple:
        return len(left) == len(right) and all(
            _known_equal(a, b, depth + 1) for a, b in zip(left, right)
        )
    if kind is dict:
        if not all(type(key) is str for key in left) or not all(type(key) is str for key in right):
            return False
        return left.keys() == right.keys() and all(
            _known_equal(left[key], right[key], depth + 1) for key in left
        )
    if kind is Material:
        return _known_equal(vars(left), vars(right), depth + 1)
    return False


def _canonical_curve(node_i, node_j, internal_points):
    """Orient stored intermediate points by the welded endpoint IDs."""
    if internal_points is None:
        return np.empty((0, 3))
    return internal_points if node_i < node_j else internal_points[::-1]


def _same_curve(left, right, tolerance):
    if type(left) is not np.ndarray or type(right) is not np.ndarray:
        return False
    if left.shape != right.shape or left.dtype.kind not in 'iuf' or right.dtype.kind not in 'iuf':
        return False
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return False
    if np.array_equal(left, right):
        return True
    with np.errstate(over='ignore', invalid='ignore'):
        return bool(np.all(np.linalg.norm(left - right, axis=1) < tolerance))


def _place_node(result, node, new_position, boundary=None):
    """Keep metadata separate from constructor arguments and source ownership."""
    count = result.num_nodes
    node_id = result.add_node(new_position, boundary=boundary, merge=True)
    if result.num_nodes != count:
        result.nodes[node_id].metadata = copy.deepcopy(node.metadata)
    return node_id


def _maximum_shared_matching(compatible, candidate_count):
    """Match current fibers to distinct historical fibers by augmenting paths.

    Tolerance-based curve compatibility is not transitive. A maximum bipartite
    matching avoids increasing multiplicity merely because a greedy first match
    took the only candidate available to a later fiber.
    """
    owners = [None] * candidate_count
    for start in range(len(compatible)):
        queue = deque([start])
        parents = {start: None}
        seen_candidates = set()
        end = None
        while queue and end is None:
            current = queue.popleft()
            for candidate in compatible[current]:
                if candidate in seen_candidates:
                    continue
                seen_candidates.add(candidate)
                owner = owners[candidate]
                if owner is None:
                    end = (current, candidate)
                    break
                if owner not in parents:
                    parents[owner] = (current, candidate)
                    queue.append(owner)
        while end is not None:
            current, candidate = end
            owners[candidate] = current
            end = parents[current]
    return {current for current in owners if current is not None}


def _place_cell_edges(result, cell, id_map, offset, edge_groups):
    """Match shared fibers across cells while retaining cell multiplicity.

    Independent current-cell fibers are never compared with one another. For
    each welded endpoint pair, match as many compatible historical fibers as
    possible, then append unmatched fibers in their original source-edge order.
    """
    records = []
    current_groups = {}
    for edge_id in sorted(cell.edges):
        edge = cell.edges[edge_id]
        node_i, node_j = id_map[edge.node_i], id_map[edge.node_j]
        if node_i == node_j:
            continue  # Preserve the existing welded zero-length edge policy.
        internal = (None if edge.internal_points is None else
                    edge.internal_points + offset[None, :])
        curve = _canonical_curve(node_i, node_j, internal)
        pair = (min(node_i, node_j), max(node_i, node_j))
        current_groups.setdefault(pair, []).append(len(records))
        records.append((edge, node_i, node_j, internal, curve, pair))

    matched = set()
    for pair, indices in current_groups.items():
        candidates = edge_groups.get(pair, [])
        compatible = []
        for index in indices:
            edge, _, _, _, curve, _ = records[index]
            options = []
            for candidate_index, existing_id in enumerate(candidates):
                existing = result.edges[existing_id]
                if (_known_equal(edge.radius, existing.radius)
                        and _known_equal(edge.segments, existing.segments)
                        and _known_equal(edge.material, existing.material)
                        and _known_equal(edge.metadata, existing.metadata)
                        and _same_curve(curve, _canonical_curve(
                            existing.node_i, existing.node_j,
                            existing.internal_points), result.tolerance)):
                    options.append(candidate_index)
            compatible.append(options)
        matched.update(indices[index] for index in
                       _maximum_shared_matching(compatible, len(candidates)))

    for index, (edge, node_i, node_j, internal, _, pair) in enumerate(records):
        if index in matched:
            continue
        new_id = result.add_edge(
            node_i, node_j, radius=edge.radius,
            material=copy.deepcopy(edge.material), internal_points=internal,
            segments=edge.segments, allow_parallel=True,
        )
        result.edges[new_id].metadata = copy.deepcopy(edge.metadata)
        edge_groups.setdefault(pair, []).append(new_id)


def _geometry_bounds(graph):
    """Bounds of nodes and stored curve points in graph coordinate units."""
    points = [graph.node_positions()]
    points.extend(edge.internal_points for edge in graph.edges.values()
                  if edge.internal_points is not None and len(edge.internal_points))
    positions = np.vstack(points)
    if not len(positions):
        return np.zeros(3), np.zeros(3)
    return positions.min(axis=0), positions.max(axis=0)


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
        Nonnegative integer counts; integer scalars broadcast to both axes.
        Python/NumPy integers are accepted, bool and floats are rejected.
        A zero count retains the empty-tiling behavior.
    box_size : (w, h), optional
        Override the unit cell's box_size.
    tolerance : float, optional
        Node merging tolerance. Defaults to unit_cell.tolerance.

    Returns
    -------
    StructureGraph
        The tiled structure with welded nodes.

    Notes
    -----
    Independent parallel fibers within a cell are retained. Across cells,
    shared fibers match only when radius, material, discretization, metadata
    and oriented curve geometry agree; curve coordinates use the node-welding
    tolerance. Matching bundles retain their maximum multiplicity. Values that
    cannot be safely compared remain separate fibers.
    """
    grid = normalize_grid(grid, 2)
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
    edge_groups = {}

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

                new_nid = _place_node(result, node, new_pos, boundary=bnd)
                id_map[nid] = new_nid

            _place_cell_edges(result, unit_cell, id_map, offset, edge_groups)

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

    Notes
    -----
    Uses the same shared-fiber matching and physical multiplicity rules as
    :func:`tile_2d`.
    """
    grid = normalize_grid(grid, 3)
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
    edge_groups = {}

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

                    new_nid = _place_node(result, node, new_pos, boundary=bnd)
                    id_map[nid] = new_nid

                _place_cell_edges(result, unit_cell, id_map, offset, edge_groups)

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

    Notes
    -----
    Shared fibers follow the matching and multiplicity rules of :func:`tile_2d`.

    Examples
    --------
    >>> transforms = {
    ...     (1, 0): lambda g: rotate(g, 90),
    ...     (0, 1): lambda g: mirror_x(g),
    ... }
    >>> g = tile_with_transforms(unit, grid=(2,2), cell_transforms=transforms)
    """
    grid = normalize_grid(grid, 2)
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
    result._metadata = copy.deepcopy(unit_cell.metadata)
    edge_groups = {}

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
                new_nid = _place_node(result, node, new_pos)
                id_map[nid] = new_nid

            _place_cell_edges(result, cell, id_map, offset, edge_groups)

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

    Maintains aspect ratio and includes stored curve points in the bounds.
    Useful for normalizing arbitrary point inputs to a standard cell size.

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

    bb_min, bb_max = _geometry_bounds(graph)
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
    bb_min2, bb_max2 = _geometry_bounds(scaled)
    span2 = bb_max2 - bb_min2
    offset = (tb - span2) / 2 - bb_min2
    offset[2] = 0 if graph.dimension == 2 else offset[2]

    result = g_translate(scaled, offset)
    result.box_size = tb
    return result
