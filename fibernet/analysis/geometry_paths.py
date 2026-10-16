"""Bounded explicit-edge geometry and selected pair distances, schema 1."""
from dataclasses import dataclass, field
import math
from numbers import Integral
from typing import Optional, Tuple

import numpy as np
from fibernet.core.structure_graph import StructureGraph


@dataclass(frozen=True)
class GeometryPathMetricsV1:
    """Caller-unit geometry, original IDs, and explicitly weighted distances."""
    node_ids: Tuple[int, ...]
    edge_ids: Tuple[int, ...]
    length_unit: str
    coordinate_frame: str
    distance_mode: str
    chord_length_by_edge: Tuple[float, ...]
    polyline_length_by_edge: Tuple[float, ...]
    curve_straightness_by_edge: Tuple[Optional[float], ...]
    degenerate_edge_ids: Tuple[int, ...]
    pair_node_ids: Tuple[Tuple[int, int], ...]
    pair_distance: Tuple[Optional[float], ...]
    pair_chord: Tuple[float, ...]
    pair_tortuosity: Tuple[Optional[float], ...]
    pair_status: Tuple[str, ...]
    schema: str = field(default='fibernet.geometry_paths/1', init=False)


def _integer(value):
    return isinstance(value, Integral) and not isinstance(value, bool) and value >= 0


def _finite(value, label):
    if not math.isfinite(value):
        raise OverflowError(f'{label} exceeds finite float64 range')
    return value


def _ratio(numerator, denominator, label):
    value = _finite(numerator / denominator, label)
    if numerator != 0 and value == 0:
        raise OverflowError(f'{label} is below representable float64 range')
    return value


def _distance(a, b):
    delta = [_finite(float(y) - float(x), 'coordinate difference') for x, y in zip(a, b)]
    return _finite(math.hypot(*delta), 'distance')


def _geometry(value, shape, label):
    if np.ma.isMaskedArray(value):
        raise ValueError(f'{label} must not be masked')
    try:
        raw = np.asarray(value)
        if raw.shape != shape or raw.dtype.kind not in 'iuf' or not np.isfinite(raw).all():
            raise ValueError(f'{label} must be finite numeric data with shape {shape}')
        with np.errstate(over='ignore', invalid='ignore'):
            arr = raw.astype(np.float64)
        if not np.isfinite(arr).all():
            raise OverflowError(f'{label} exceeds finite float64 range')
        # Reject conversion underflow, rather than silently turning geometry zero.
        if np.any((raw != 0) & (arr == 0)):
            raise OverflowError(f'{label} is below representable float64 range')
        return arr
    except (TypeError, OverflowError):
        raise
    except ValueError as exc:
        raise ValueError(f'invalid {label}: {exc}') from exc


def analyze_geometry_paths(graph, *, length_unit, coordinate_frame, node_pairs=(),
                           distance_mode='polyline', max_pairs=256,
                           max_nodes=10000, max_edges=20000, max_points=100000,
                           max_path_work=1000000) -> GeometryPathMetricsV1:
    """Measure full XYZ chords/polylines and selected undirected shortest paths.

    Budgets are strict positive integers; excess raises MemoryError before path
    traversal. max_points counts nodes plus stored internal points, and work is
    conservatively charged as pairs * (nodes + physical edges), including repeats.
    Input errors raise ValueError/TypeError; numerical overflow raises OverflowError.
    See docs/architecture/geometry-paths.md for undefined-value semantics.
    """
    if not isinstance(graph, StructureGraph):
        raise TypeError('analyze_geometry_paths requires StructureGraph')
    for name, value in [('length_unit', length_unit), ('coordinate_frame', coordinate_frame)]:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{name} must be a nonempty caller declaration')
    if distance_mode not in ('chord', 'polyline'):
        raise ValueError('distance_mode must be chord or polyline')
    for name, value in [('max_pairs', max_pairs), ('max_nodes', max_nodes),
                        ('max_edges', max_edges), ('max_points', max_points),
                        ('max_path_work', max_path_work)]:
        if not _integer(value) or value == 0:
            raise ValueError(f'{name} must be a positive integer, not bool')
    n, e = len(graph.nodes), len(graph.edges)
    if n > max_nodes or e > max_edges:
        raise MemoryError('graph exceeds max_nodes or max_edges')
    point_count = n
    for edge in graph.edges.values():
        if edge.internal_points is not None:
            try:
                point_count += len(edge.internal_points)
            except TypeError as exc:
                raise ValueError('internal_points must be an (M, 3) array') from exc
        if point_count > max_points:
            raise MemoryError('geometry exceeds max_points')
    if point_count > max_points:
        raise MemoryError('geometry exceeds max_points')
    positions = {}
    for nid, node in graph.nodes.items():
        if not _integer(nid) or not _integer(node.node_id) or nid != node.node_id:
            raise ValueError('node IDs must be matching nonnegative integers')
        positions[int(nid)] = _geometry(node.position, (3,), 'position')
    if np.ma.isMaskedArray(node_pairs):
        raise ValueError('node_pairs must not be masked')
    pairs = []
    try:
        iterator = iter(node_pairs)
    except TypeError as exc:
        raise ValueError('node_pairs must be an iterable of ID pairs') from exc
    for pair in iterator:
        if len(pairs) >= max_pairs:
            raise MemoryError('node_pairs exceeds max_pairs')
        if np.ma.isMaskedArray(pair):
            raise ValueError('node pair must not be masked')
        try:
            if len(pair) != 2:
                raise ValueError('node pair must contain two IDs')
            a, b = pair
        except TypeError as exc:
            raise ValueError('node pair must contain two IDs') from exc
        if not _integer(a) or not _integer(b) or a not in positions or b not in positions:
            raise ValueError('node pair IDs must be existing nonnegative integers')
        pairs.append((int(a), int(b)))
    if len(pairs) * (n + e) > max_path_work:
        raise MemoryError('selected paths exceed max_path_work')

    import networkx as nx
    paths = nx.MultiGraph()
    paths.add_nodes_from(positions)
    ids, chords, polylines, straightness, degenerate = [], [], [], [], []
    for eid, edge in graph.edges.items():
        a, b = edge.node_i, edge.node_j
        if not _integer(eid) or not _integer(edge.edge_id) or eid != edge.edge_id:
            raise ValueError('edge IDs must be matching nonnegative integers')
        if not _integer(a) or not _integer(b) or a not in positions or b not in positions:
            raise ValueError('edge endpoints must be existing nonnegative integer IDs')
        if a == b:
            raise ValueError('geometry_paths v1 does not support self-loops')
        eid, a, b = int(eid), int(a), int(b)
        start, end = positions[a], positions[b]
        internal = (np.empty((0, 3)) if edge.internal_points is None else
                    _geometry(edge.internal_points, (len(edge.internal_points), 3), 'internal_points'))
        points = [start, *internal, end]
        chord = _distance(start, end)
        polyline = _finite(math.fsum(_distance(x, y) for x, y in zip(points, points[1:])), 'polyline length')
        ratio = _ratio(chord, polyline, 'straightness') if polyline else None
        ids.append(eid); chords.append(chord); polylines.append(polyline); straightness.append(ratio)
        if chord == 0:
            degenerate.append(eid)
        paths.add_edge(a, b, key=eid, distance=polyline if distance_mode == 'polyline' else chord)
    distances, pair_chords, ratios, statuses = [], [], [], []
    for a, b in pairs:
        chord = _distance(positions[a], positions[b])
        pair_chords.append(chord)
        try:
            route = nx.shortest_path(paths, a, b, weight='distance', method='dijkstra')
        except nx.NetworkXNoPath:
            distances.append(None); ratios.append(None); statuses.append('unreachable')
            continue
        # Stable sum of explicit minimum physical-edge costs along the NX route.
        distance = _finite(math.fsum(min(data['distance'] for data in paths[u][v].values())
                                     for u, v in zip(route, route[1:])), 'path distance')
        distances.append(distance)
        ratios.append(_ratio(distance, chord, 'tortuosity') if chord else None)
        statuses.append('ok' if chord else 'zero_chord')
    return GeometryPathMetricsV1(
        tuple(sorted(positions)), tuple(ids), length_unit, coordinate_frame, distance_mode,
        tuple(chords), tuple(polylines), tuple(straightness), tuple(degenerate), tuple(pairs),
        tuple(distances), tuple(pair_chords), tuple(ratios), tuple(statuses))
