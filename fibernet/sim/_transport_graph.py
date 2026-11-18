"""Lossless admission of graph identities and coordinate provenance for transport.

Only the fields used by the lumped electrical and thermal models are admitted.
Radii, curves and materials are not inferred as conductance or capacity.
"""

import numpy as np


def _identifier(value, label):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not 0 <= value <= np.iinfo(np.int64).max
    ):
        raise ValueError(f"{label} must be a nonnegative integer ID fitting int64")
    return int(value)


def _position(value, label):
    # Inspect before NumPy promotion can erase masks, booleans, or complex parts.
    def inspect(item):
        if np.ma.isMaskedArray(item):
            raise ValueError(f"{label} must not contain masked arrays or scalars")
        if isinstance(item, np.ndarray):
            if item.dtype.kind not in "iuf":
                raise ValueError(f"{label} must contain real numeric coordinates")
        elif isinstance(item, (list, tuple)):
            for child in item:
                inspect(child)
        elif isinstance(item, (bool, np.bool_)) or not isinstance(
            item, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"{label} must contain real numeric coordinates")

    inspect(value)
    try:
        raw = np.asarray(value)
        if (
            raw.shape != (3,)
            or raw.dtype.kind not in "iuf"
            or not np.isfinite(raw).all()
        ):
            raise ValueError(f"{label} must contain three finite real coordinates")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            position = np.array(raw, dtype=np.float64, copy=True)
        if not np.isfinite(position).all() or np.any((raw != 0) & (position == 0)):
            raise ValueError(
                f"{label} is not representable as finite nonzero float64 coordinates"
            )
    except (TypeError, OverflowError) as exc:
        raise ValueError(f"{label} must contain three finite real coordinates") from exc
    return position


def transport_graph_snapshot(graph):
    """Return detached ordered IDs, positions and oriented endpoint pairs.

    Dictionary keys and record IDs must agree. Node IDs are sorted and edge
    insertion order is retained. No graph mutation, unit conversion, geometry
    repair or inference from non-topological fields is performed.
    """
    node_ids = sorted(_identifier(k, "node ID") for k in graph.nodes)
    edge_ids = [_identifier(k, "edge ID") for k in graph.edges]
    if not node_ids:
        raise ValueError("transport requires a nonempty node set")
    positions = []
    for nid in node_ids:
        node = graph.nodes[nid]
        if _identifier(node.node_id, "node record ID") != nid:
            raise ValueError("node record ID must agree with its dictionary key")
        positions.append(_position(node.position, f"node {nid} position"))
    rows = set(node_ids)
    endpoints = []
    for eid in edge_ids:
        edge = graph.edges[eid]
        if _identifier(edge.edge_id, "edge record ID") != eid:
            raise ValueError("edge record ID must agree with its dictionary key")
        a = _identifier(edge.node_i, "endpoint ID")
        b = _identifier(edge.node_j, "endpoint ID")
        if a not in rows or b not in rows or a == b:
            raise ValueError("each edge needs two distinct existing endpoint IDs")
        endpoints.append((a, b))
    return node_ids, edge_ids, np.array(positions, dtype=np.float64), endpoints
