"""Physical-edge-once trails with explicit breaks and detached graph provenance.

This additive adapter does not repair/edit the authoritative StructureGraph or
change the existing closed-route APIs. It makes no machine/process qualification.
"""
from dataclasses import dataclass
import csv
import hashlib
import io
import json
import re

from .manufacturing_bridge import (
    _array_record, _check_conversion, _check_units, _digest, _exact_binary64,
    _fresh_paths, _json_bytes, _owned_reservation, _reserve, _snapshot_record,
)


_TRAIL_ARRAYS = ('route_nodes', 'route_edges', 'stroke_node_ptr',
                 'stroke_edge_ptr', 'closed')
_SCHEMA = 'fibernet.manufacturing_trails.v1'
_ALGORITHM = 'sorted_odd_matching_euler_split.v1'


@dataclass(frozen=True, init=False, eq=False)
class ManufacturingTrails:
    """Factory-created traversal of a single detached GraphBatch v2 snapshot.

    ``route_nodes`` contains E + S dense node rows; ``route_edges`` contains E
    dense physical-edge rows. ``stroke_node_ptr`` and ``stroke_edge_ptr`` delimit
    the S trails in these separate arrays. Each trail has one more node than
    edge, and ``closed[s]`` reports coincident endpoint node rows. Arrays are
    detached, bytes-backed and read-only; this is a result, not an editable graph.
    Stable source IDs, coordinates, units and materials live in the paired
    snapshot. Hashes detect later ordinary replacement/mutation, not hostile code.
    """
    schema_version: str
    algorithm_version: str
    route_nodes: object
    route_edges: object
    stroke_node_ptr: object
    stroke_edge_ptr: object
    closed: object
    normalized_source_sha256: str
    route_binding_sha256: str

    def __init__(self, *args, **kwargs):
        raise TypeError('use manufacturing_trails_from_graph to create ManufacturingTrails')

    @property
    def stroke_count(self):
        return len(self.closed)


def _immutable_array(values, dtype):
    import numpy as np
    # Immutable backing prevents write=True from re-enabling accidental writes.
    value = np.asarray(values, dtype=dtype)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def _trails_record(trails):
    if (type(trails) is not ManufacturingTrails
            or getattr(trails, 'schema_version', None) != _SCHEMA
            or getattr(trails, 'algorithm_version', None) != _ALGORITHM
            or not all(hasattr(trails, name) for name in _TRAIL_ARRAYS)
            or not hasattr(trails, 'normalized_source_sha256')):
        raise ValueError('mapped export requires factory-created ManufacturingTrails')
    return {'schema_version': trails.schema_version, 'algorithm_version': trails.algorithm_version,
            'normalized_source_sha256': trails.normalized_source_sha256,
            'arrays': {name: _array_record(getattr(trails, name))
                       for name in _TRAIL_ARRAYS}}


def _decompose(edges, node_count, odd):
    """Use private scratch edges, remove them all, preserve physical row identity."""
    import numpy as np
    from .manufacturing import _route
    physical_count = len(edges)
    augmented = (np.concatenate((edges, odd.reshape(-1, 2)), axis=0)
                 if len(odd) else edges)
    nodes, edge_rows = _route(augmented.tobytes(), node_count)
    if not len(odd):
        return [(nodes, edge_rows)]
    # Start immediately after a virtual edge, so the final edge is virtual too.
    cut = next(i for i, row in enumerate(edge_rows) if row >= physical_count)
    total = len(edge_rows)
    current_nodes, current_edges = [nodes[cut + 1]], []
    trails = []
    for k in range(total):
        index = (cut + 1 + k) % total
        row, endpoint = edge_rows[index], nodes[index + 1]
        if row >= physical_count:
            if not current_edges:
                raise RuntimeError('unexpected empty stroke during route decomposition')
            trails.append((tuple(current_nodes), tuple(current_edges)))
            current_nodes, current_edges = [endpoint], []
        else:
            current_edges.append(row)
            current_nodes.append(endpoint)
    return trails


def _validate_trails(trails, snapshot):
    """Check incidence, edge-once coverage, minimum count and endpoint parity."""
    import numpy as np
    nodes, rows = trails.route_nodes, trails.route_edges
    nptr, eptr, closed = trails.stroke_node_ptr, trails.stroke_edge_ptr, trails.closed
    for value in (nodes, rows, nptr, eptr):
        if value.dtype != np.dtype('int64') or value.ndim != 1:
            raise ValueError('trail indices must be one-dimensional int64 arrays')
    if closed.dtype != np.dtype('bool') or closed.ndim != 1:
        raise ValueError('trail closure flags must be a one-dimensional bool array')
    edges = snapshot.edge_index.T
    n, e, s = len(snapshot.positions), len(edges), len(closed)
    if not n or not e or not s:
        raise ValueError('manufacturing trails require a nonempty connected graph')
    if (len(nptr) != s + 1 or len(eptr) != s + 1 or nptr[0] != 0 or eptr[0] != 0
            or nptr[-1] != len(nodes) or eptr[-1] != len(rows)
            or len(nodes) != e + s or len(rows) != e
            or np.any(np.diff(eptr) <= 0)
            or not np.array_equal(np.diff(nptr), np.diff(eptr) + 1)
            or np.any(nodes < 0) or np.any(nodes >= n)
            or not np.array_equal(np.sort(rows), np.arange(e, dtype=np.int64))):
        raise ValueError('invalid stroke pointers, node rows or physical-edge coverage')
    degree = np.bincount(edges.ravel(), minlength=n)
    odd = np.flatnonzero(degree % 2)
    if s != max(1, len(odd) // 2):
        raise ValueError('stroke count is not minimal')
    endpoints = []
    for stroke in range(s):
        ns = nodes[nptr[stroke]:nptr[stroke + 1]]
        es = rows[eptr[stroke]:eptr[stroke + 1]]
        if bool(closed[stroke]) != bool(ns[0] == ns[-1]):
            raise ValueError('stroke closure flag disagrees with endpoints')
        expected = edges[es]
        forward = (expected[:, 0] == ns[:-1]) & (expected[:, 1] == ns[1:])
        reverse = (expected[:, 1] == ns[:-1]) & (expected[:, 0] == ns[1:])
        if not np.all(forward | reverse):
            raise ValueError('stroke steps do not follow their physical edges')
        endpoints.extend((int(ns[0]), int(ns[-1])))
    if len(odd):
        if np.any(closed) or sorted(endpoints) != odd.tolist():
            raise ValueError('odd vertices must occur exactly once as trail endpoints')
    elif not bool(closed[0]):
        raise ValueError('an all-even graph must have one closed trail')


def manufacturing_trails_from_graph(graph, *, graph_id, length_unit, coordinate_frame):
    """Return ``(ManufacturingTrails, GraphBatch)`` for a connected nonempty graph.

    Every physical edge appears once in max(1, odd_vertex_count / 2) trails.
    Sorted odd node rows are paired only in routing scratch storage, never in the
    source graph or output. The same ordered snapshot gives the same route; no
    canonical ordering across graph reorderings is promised. Only explicit m/mm
    source units and a nonempty frame are accepted. Nonempty internal-point
    geometry is refused. Zero-length incidence and stored self-loops are not
    physically qualified or newly prohibited by this topology-only adapter.
    """
    import numpy as np
    from fibernet.core.structure_graph import StructureGraph
    from fibernet.ml.graph_data import graph_batch
    from .cell_rules import graph_health
    from .manufacturing import MAX_MANUFACTURING_POINTS, MAX_MANUFACTURING_EDGES
    if type(graph) is not StructureGraph:
        raise TypeError('adapter requires an ordinary StructureGraph')
    if type(length_unit) is not str or length_unit not in ('m', 'mm'):
        raise ValueError('source length_unit must be explicitly m or mm')
    if not graph.nodes or not graph.edges:
        raise ValueError('manufacturing trails require a nonempty connected graph')
    if (len(graph.nodes) > MAX_MANUFACTURING_POINTS
            or len(graph.edges) > MAX_MANUFACTURING_EDGES):
        raise MemoryError('manufacturing graph budget exceeded')
    for node in graph.nodes.values():
        _exact_binary64(node.position, 'node coordinates')
    for edge in graph.edges.values():
        if edge.radius is not None:
            _exact_binary64(edge.radius, 'edge radius')
    snapshot = graph_batch([graph], graph_ids=[graph_id], length_unit=length_unit,
                           coordinate_frame=coordinate_frame, edge_features=('radius',),
                           include_provenance=True)
    if len(snapshot.internal_points):
        raise ValueError('manufacturing adapter does not support internal-point geometry')
    for values in (snapshot.graph_metadata, snapshot.node_metadata, snapshot.edge_metadata):
        _check_units(values, length_unit, coordinate_frame)
    edges = np.asarray(snapshot.edge_index.T, dtype=np.int64)
    if graph_health(snapshot.positions, edges)['components'] != 1:
        raise ValueError('manufacturing trails require a connected graph including every node')
    odd = np.flatnonzero(np.bincount(edges.ravel(), minlength=len(snapshot.positions)) % 2)
    pieces = _decompose(edges, len(snapshot.positions), odd)
    nodes, rows, nptr, eptr, closed = [], [], [0], [0], []
    for ns, es in pieces:
        nodes.extend(ns)
        rows.extend(es)
        nptr.append(len(nodes))
        eptr.append(len(rows))
        closed.append(ns[0] == ns[-1])
    result = object.__new__(ManufacturingTrails)
    object.__setattr__(result, 'schema_version', _SCHEMA)
    object.__setattr__(result, 'algorithm_version', _ALGORITHM)
    for name, value in zip(_TRAIL_ARRAYS, (nodes, rows, nptr, eptr, closed)):
        object.__setattr__(result, name, _immutable_array(value, bool if name == 'closed' else np.int64))
    object.__setattr__(result, 'normalized_source_sha256', _digest(_snapshot_record(snapshot)))
    object.__setattr__(result, 'route_binding_sha256', _digest(_trails_record(result)))
    _validate_trails(result, snapshot)
    return result, snapshot


def _prepare_export(trails, snapshot, source_input_sha256):
    if source_input_sha256 is not None and (type(source_input_sha256) is not str
            or re.fullmatch('[0-9a-f]{64}', source_input_sha256) is None):
        raise ValueError('source_input_sha256 must be lowercase SHA256 or None')
    trail_hash = _digest(_trails_record(trails))
    source = _snapshot_record(snapshot)
    source_hash = _digest(source)
    if (source_hash != trails.normalized_source_sha256
            or trail_hash != trails.route_binding_sha256):
        raise ValueError('source snapshot or manufacturing trails changed after adaptation')
    _validate_trails(trails, snapshot)
    if snapshot.length_unit not in ('m', 'mm'):
        raise ValueError('source length unit is unsupported')
    factor = 1000.0 if snapshot.length_unit == 'm' else 1.0
    originals = snapshot.positions.tolist()
    centers = [[float(value) * factor for value in row] for row in originals]
    edges = snapshot.edge_index.T.tolist()
    _check_conversion(originals, centers, edges)
    node_ids, edge_ids = snapshot.node_ids.tolist(), snapshot.edge_ids.tolist()
    nodes, edge_rows = trails.route_nodes.tolist(), trails.route_edges.tolist()
    nptr, eptr = trails.stroke_node_ptr.tolist(), trails.stroke_edge_ptr.tolist()
    route, strokes = [], []
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(['stroke', 'step', 'node_row', 'node_id', 'incoming_edge_row',
                     'incoming_edge_id', 'source_edge_direction', 'x_mm', 'y_mm', 'z_mm'])
    for stroke in range(trails.stroke_count):
        ns, es = nodes[nptr[stroke]:nptr[stroke + 1]], edge_rows[eptr[stroke]:eptr[stroke + 1]]
        strokes.append({'stroke': stroke, 'node_start': nptr[stroke], 'node_stop': nptr[stroke + 1],
                        'edge_start': eptr[stroke], 'edge_stop': eptr[stroke + 1],
                        'closed': bool(trails.closed[stroke])})
        for step, node in enumerate(ns):
            edge = None if step == 0 else es[step - 1]
            eid = None if edge is None else edge_ids[edge]
            direction = None if edge is None else (1 if edges[edge] == [ns[step - 1], node] else -1)
            writer.writerow([stroke, step, node, node_ids[node],
                             -1 if edge is None else edge, -1 if eid is None else eid,
                             0 if direction is None else direction, *centers[node]])
            route.append({'stroke': stroke, 'step': step, 'node_row': node, 'node_id': node_ids[node],
                          'incoming_edge_row': edge, 'incoming_edge_id': eid,
                          'source_edge_direction': direction})
    csv_bytes = stream.getvalue().encode('utf-8')
    mapping = {
        'schema_version': 'fibernet.manufacturing_trails_mapping.v1',
        'algorithm_version': trails.algorithm_version,
        'graph_id': snapshot.graph_ids[0], 'graph_id_origin': 'caller_asserted',
        'source_input_sha256': source_input_sha256,
        'source_input_sha256_origin': 'caller_asserted_not_file_verified' if source_input_sha256 else 'unknown',
        'normalized_source_sha256': source_hash,
        'normalized_source_hash_scope': 'detached GraphBatch v2 payload and typed array bytes',
        'trails_binding_sha256': trail_hash,
        'route_csv_sha256': hashlib.sha256(csv_bytes).hexdigest(),
        'source_length_unit': snapshot.length_unit, 'route_length_unit': 'mm',
        'coordinate_frame': snapshot.coordinate_frame,
        'conversion': {'factor': factor, 'kind': 'isotropic_unit_conversion_only',
                       'arithmetic': 'binary64; not guaranteed bit-reversible'},
        'csv_identity_semantics': {'stroke': 'zero_based_stroke', 'step': 'zero_based_within_stroke',
                                   'node_row': 'dense_snapshot_node_row', 'node_id': 'stable_source_node_id',
                                   'incoming_edge_row': 'dense_physical_edge_row',
                                   'incoming_edge_id': 'stable_source_edge_id',
                                   'initial_incoming_edge_row': -1, 'initial_incoming_edge_id': -1,
                                   'initial_source_edge_direction': 0,
                                   'source_edge_direction': '+1 source node_i to node_j; -1 reverse; self-loop +1'},
        'stroke_count': trails.stroke_count, 'physical_edge_count': len(edges),
        'odd_vertex_count': 0 if bool(trails.closed[0]) else 2 * trails.stroke_count,
        'route_row_count': len(route), 'strokes': strokes, 'route': route,
        'source_snapshot': source,
        'material_scope': 'stored records with declared or unknown origin; not effective solver material',
        'claim': 'physical_edge_once_trails_only_not_machine_motion_printability_or_predictive_evidence',
    }
    return csv_bytes, _json_bytes(mapping) + b'\n'


def export_mapped_trails(csv_path, trails, snapshot, *, sidecar_path, source_input_sha256=None):
    """Write NEW stroke-aware mm CSV and a complete stable-ID JSON sidecar.

    Each stroke starts at step 0 with CSV incoming-edge row/ID -1 and direction 0
    (JSON null). There is NO physical or travel edge between strokes. The source
    snapshot's coordinates/radii/materials keep their original units and values.
    Both destinations must be fresh with existing parents. Validate/serialize
    before exclusive reservation, atomically publish CSV, verify its bytes, then
    atomically publish sidecar last. The matching CSV SHA and sidecar together
    are the completion record; this is not an atomic pair. On failure, owned
    empty reservations or a completed CSV may remain; no automatic retry/delete.
    Concurrent destination/parent replacement or source mutation is unsupported.
    """
    from .solid_export import _owned_export_file
    csv_path, metadata_path = _fresh_paths(csv_path, sidecar_path)
    csv_bytes, raw_mapping = _prepare_export(trails, snapshot, source_input_sha256)
    csv_identity = _reserve(csv_path)
    metadata_identity = _reserve(metadata_path)
    _owned_reservation(csv_path, csv_identity)
    with _owned_export_file(csv_path) as temporary:
        temporary.write_bytes(csv_bytes)
    if csv_path.read_bytes() != csv_bytes:
        raise RuntimeError('published trail CSV differs from the complete mapped contract')
    _owned_reservation(metadata_path, metadata_identity)
    with _owned_export_file(metadata_path) as temporary:
        temporary.write_bytes(raw_mapping)
    return json.loads(raw_mapping)
