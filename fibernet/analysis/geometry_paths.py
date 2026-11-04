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


def compare_lammps_node_displacements(reference_graph, initial_graph, result, *,
                                       reference_length_unit, initial_length_unit):
    """Compare final nodes to two caller-supplied geometries, in metres.

    Returns sorted stable node IDs and detached (N, 3) arrays named
    ``from_initial_m`` and ``from_reference_m``. The initial graph must exactly
    match the result's recorded graph digest and original length unit. The
    reference graph must have the same node IDs, but matching IDs alone do not
    establish physical correspondence; the caller must supply the same frame
    and meaning. Float64 conversion can round near-coincident coordinates.
    Only final positions are used. No interpolation, force,
    reaction, stress, work, or physical validation is performed. Digests detect
    disagreement, not maliciously fabricated results.
    """
    import hashlib
    import json

    from fibernet.units import LENGTH_FACTORS as scales
    if (not isinstance(reference_length_unit, str) or not isinstance(initial_length_unit, str)
            or reference_length_unit not in scales or initial_length_unit not in scales):
        raise ValueError('explicit supported graph length units are required')
    if not isinstance(reference_graph, StructureGraph) or not isinstance(initial_graph, StructureGraph):
        raise TypeError('reference and initial must be StructureGraph instances')
    if (not _integer(reference_graph.dimension) or reference_graph.dimension != 3
            or not _integer(initial_graph.dimension) or initial_graph.dimension != 3):
        raise ValueError('graphs must be 3D')

    def graph_ids(graph, label):
        ids = list(graph.nodes)
        if any(not _integer(i) for i in ids) or len(ids) != len(set(ids)):
            raise ValueError(f'{label} node IDs must be nonnegative integers')
        if any(not _integer(graph.nodes[i].node_id) or graph.nodes[i].node_id != i for i in ids):
            raise ValueError(f'{label} node records must match their IDs')
        return ids

    ref_ids = graph_ids(reference_graph, 'reference')
    initial_ids = graph_ids(initial_graph, 'initial')
    if not initial_ids or set(ref_ids) != set(initial_ids):
        raise ValueError('graphs must have the same nonempty node IDs')
    order = tuple(sorted(initial_ids))
    metadata = getattr(result, 'metadata', None)
    if not isinstance(metadata, dict) or metadata.get('backend') != 'lammps':
        raise ValueError('result must carry explicit LAMMPS metadata')
    ids = metadata.get('node_ids')
    if not isinstance(ids, (tuple, list)) or any(not _integer(i) for i in ids) or len(ids) != len(set(ids)) or set(ids) != set(order):
        raise ValueError('result node_ids must exactly cover graph IDs')
    fields = metadata.get('field_units')
    if (not isinstance(fields, dict) or fields.get('deformed_positions') != initial_length_unit
            or metadata.get('original_graph_length_unit') != initial_length_unit
            or isinstance(metadata.get('original_graph_length_to_m'), (bool, np.bool_))
            or metadata.get('original_graph_length_to_m') != scales[initial_length_unit]):
        raise ValueError('result and initial graph units must agree')
    try:
        canonical = json.dumps(initial_graph.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError('initial graph must have finite JSON representation') from exc
    identity = metadata.get('source_identity')
    if not isinstance(identity, dict) or identity.get('graph_sha256') != hashlib.sha256(canonical.encode()).hexdigest():
        raise ValueError('initial graph does not match result graph digest')

    def real_array(value, shape, label):
        pending = [value]
        seen = set()
        while pending:
            item = pending.pop()
            if (np.ma.isMaskedArray(item) or isinstance(item, (bool, np.bool_))
                    or (isinstance(item, np.ndarray) and item.dtype.kind == 'b')):
                raise ValueError(f'{label} must contain unmasked real numbers')
            if isinstance(item, (list, tuple)) and id(item) not in seen:
                seen.add(id(item)); pending.extend(item)
        return _geometry(value, shape, label)

    def graph_positions(graph, label):
        return np.stack([real_array(graph.nodes[i].position, (3,), label) for i in order])

    reference = graph_positions(reference_graph, 'reference positions')
    initial = graph_positions(initial_graph, 'initial positions')
    final = real_array(getattr(result, 'deformed_positions', None), (len(order), 3), 'final positions')
    rows = {nid: index for index, nid in enumerate(ids)}
    final = final[[rows[nid] for nid in order]]
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        reference_m = reference * scales[reference_length_unit]
        initial_m = initial * scales[initial_length_unit]
        final_m = final * scales[initial_length_unit]
    for original, converted in ((reference, reference_m), (initial, initial_m), (final, final_m)):
        if not np.isfinite(converted).all() or np.any((original != 0) & (converted == 0)):
            raise OverflowError('position conversion is not representable in metres')

    def same_unit_difference(a, b, a_m, b_m, factor):
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            delta = a - b
            result_m = delta * factor
            # Only the components whose raw subtraction overflowed use the
            # already-scaled operands; other components keep their precision.
            result_m = np.where(np.isfinite(delta), result_m, a_m - b_m)
        if np.any(np.isfinite(delta) & (delta != 0) & (result_m == 0)):
            raise OverflowError('displacement is below representable float64 range in metres')
        return result_m

    from_initial = same_unit_difference(final, initial, final_m, initial_m, scales[initial_length_unit])
    if scales[reference_length_unit] == scales[initial_length_unit]:
        from_reference = same_unit_difference(final, reference, final_m, reference_m, scales[initial_length_unit])
    else:
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            from_reference = final_m - reference_m
    if not np.isfinite(from_initial).all() or not np.isfinite(from_reference).all():
        raise OverflowError('displacement exceeds finite float64 range')
    return {'node_ids': order, 'units': 'm', 'from_initial_m': from_initial.copy(),
            'from_reference_m': from_reference.copy()}
