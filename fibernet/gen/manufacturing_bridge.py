"""Explicit stable-ID handoff to the existing closed-route manufacturing API.

This module has no numerical import until an adapter/export function is called.
It is a route mapping, not another graph model or a physical qualification.
"""
from collections.abc import Mapping
from copy import copy
from dataclasses import fields
from pathlib import Path
import csv
import hashlib
import io
import json
import math
import os
import re
import stat
import struct


_ARRAYS = ('reference', 'positions', 'edges', 'route_nodes', 'route_edges')
_LENGTH_KEYS = frozenset(('length_unit', 'coordinate_unit', 'coordinate_units',
                          'position_unit', 'radius_unit'))


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(',', ':')).encode('utf-8')


def _digest(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _json_value(value):
    """Detach the JSON-native/read-only containers used by GraphBatch."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('source provenance must be finite')
        return value
    if type(value) in (list, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError('source provenance keys must be strings')
        return {key: _json_value(item) for key, item in value.items()}
    raise ValueError('unsupported source provenance value')


def _array_record(value, *, readonly=True):
    import numpy as np
    if (type(value) is not np.ndarray or value.dtype.kind not in 'biuf'
            or (readonly and value.flags.writeable)):
        raise ValueError('mapping requires detached read-only numeric arrays')
    return {'dtype': value.dtype.str, 'shape': list(value.shape),
            'sha256': hashlib.sha256(value.tobytes()).hexdigest()}


def _snapshot_record(snapshot):
    import numpy as np
    from fibernet.ml.graph_data import GraphBatch, GraphBatchProvenance
    if (type(snapshot) is not GraphBatch or snapshot.schema_version != 'fibernet.graph_batch.v2'
            or type(snapshot.provenance) is not GraphBatchProvenance
            or snapshot.provenance.schema_version != 'fibernet.graph_batch.provenance.v1'
            or len(snapshot.graph_ids) != 1):
        raise ValueError('mapping requires one factory-created GraphBatch v2 snapshot')
    arrays = {}
    def record(obj, prefix=''):
        result = {}
        for field in fields(obj):
            value = getattr(obj, field.name)
            if field.name == 'provenance':
                result[field.name] = record(value, 'provenance.')
            elif type(value) is np.ndarray:
                arrays[prefix + field.name] = _array_record(value)
                result[field.name] = _json_value(value.tolist())
            else:
                result[field.name] = _json_value(value)
        return result
    payload = record(snapshot)
    return {'graph_batch': payload, 'array_storage': arrays}


def _network_record(network):
    return {'arrays': {name: _array_record(getattr(network, name)) for name in _ARRAYS},
            'topology_id': network.topology_id,
            'added_edges': network.added_edges, 'bridges': network.bridges}


def _check_units(value, length_unit, coordinate_frame):
    """Check explicit geometry unit/frame declarations, including provenance."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _LENGTH_KEYS and (type(item) is not str or item != length_unit):
                raise ValueError('stored length-unit declaration is unknown or conflicts')
            if key == 'coordinate_frame' and item != coordinate_frame:
                raise ValueError('stored coordinate-frame declaration conflicts')
            if key == 'units':
                if type(item) is str:
                    if item != length_unit:
                        raise ValueError('stored units declaration is unknown or conflicts')
                elif isinstance(item, Mapping):
                    for name in ('length', 'coordinates', 'position', 'radius'):
                        if name in item and item[name] != length_unit:
                            raise ValueError('stored length-unit declaration is unknown or conflicts')
                else:
                    raise ValueError('stored units declaration is unsupported')
            _check_units(item, length_unit, coordinate_frame)
    elif type(value) in (list, tuple):
        for item in value:
            _check_units(item, length_unit, coordinate_frame)



def _float64_identity(value):
    """Stored binary64 identity without arithmetic; signed zeros agree."""
    bits = struct.unpack('>Q', struct.pack('>d', value))[0]
    return bits if bits & ((1 << 63) - 1) else 0


def _check_conversion(originals, converted, edges):
    if not all(math.isfinite(value) for row in converted for value in row):
        raise ValueError('m-to-mm conversion produced nonfinite coordinates')
    before = [[_float64_identity(value) for value in row] for row in originals]
    after = [[_float64_identity(value) for value in row] for row in converted]
    if any(a and not b for old, new in zip(before, after) for a, b in zip(old, new)):
        raise ValueError('unit conversion lost a nonzero coordinate')
    for a, b in edges:
        if before[a] != before[b] and after[a] == after[b]:
            raise ValueError('unit conversion collapsed distinct edge endpoints')


def _exact_binary64(value, label):
    """Refuse source storage/value narrowing beyond the GraphBatch profile."""
    import numpy as np
    # Mixed Python sequences can round integers during NumPy promotion. Check
    # original elements first, including nested sequences and NumPy scalars.
    if type(value) in (list, tuple):
        for item in value:
            _exact_binary64(item, label)
    raw = np.asarray(value)
    if (raw.dtype.kind not in 'iuf' or raw.dtype.itemsize not in (1, 2, 4, 8)
            or (raw.dtype.kind == 'f' and raw.dtype.itemsize != 8)):
        raise ValueError(label + ' has unsupported numeric storage')
    for original in raw.flat:
        original = original.item() if hasattr(original, 'item') else original
        converted = float(original)
        if not math.isfinite(converted) or original != converted:
            raise ValueError(label + ' cannot be retained exactly as binary64')


def manufacturing_from_graph(graph, *, graph_id, length_unit, coordinate_frame):
    """Return ``(ManufacturingNetwork, GraphBatch)`` without topology repair.

    Only explicit source m/mm are supported. The GraphBatch v2 snapshot retains
    stable IDs, edge order/direction, radii, stored material and metadata. Nonempty
    internal-point geometry is refused rather than straightened. Network and
    snapshot arrays are detached from the caller and read-only. Private digests
    bind their values; ordinary later mutation/replacement is refused at export.
    This is not a complete StructureGraph persistence format or a security boundary.
    """
    import numpy as np
    from fibernet.core.structure_graph import StructureGraph
    from fibernet.ml.graph_data import graph_batch
    from .manufacturing import ManufacturingNetwork
    if type(graph) is not StructureGraph:
        raise TypeError('adapter requires an ordinary StructureGraph')
    if type(length_unit) is not str or length_unit not in ('m', 'mm'):
        raise ValueError('source length_unit must be explicitly m or mm')
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
    source = _snapshot_record(snapshot)
    network = ManufacturingNetwork(np.array(snapshot.positions, copy=True),
                                   np.array(snapshot.positions, copy=True),
                                   np.array(snapshot.edge_index.T, copy=True))
    for name in _ARRAYS:
        value = np.array(getattr(network, name), copy=True)
        value.setflags(write=False)
        setattr(network, name, value)
    network._manufacturing_snapshot_sha256 = _digest(source)
    network._manufacturing_network_sha256 = _digest(_network_record(network))
    return network, snapshot


def _prepare_mapping(network, snapshot, source_input_sha256):
    import numpy as np
    from .manufacturing import ManufacturingNetwork
    if (type(network) is not ManufacturingNetwork
            or not hasattr(network, '_manufacturing_snapshot_sha256')
            or not hasattr(network, '_manufacturing_network_sha256')):
        raise ValueError('mapped export requires a factory-bound network')
    if source_input_sha256 is not None and (type(source_input_sha256) is not str
            or re.fullmatch('[0-9a-f]{64}', source_input_sha256) is None):
        raise ValueError('source_input_sha256 must be lowercase SHA256 or None')
    source = _snapshot_record(snapshot)
    source_hash = _digest(source)
    network_hash = _digest(_network_record(network))
    if (source_hash != network._manufacturing_snapshot_sha256
            or network_hash != network._manufacturing_network_sha256):
        raise ValueError('source snapshot or manufacturing arrays changed after adaptation')
    if snapshot.length_unit not in ('m', 'mm'):
        raise ValueError('source length unit is unsupported')
    factor = 1000.0 if snapshot.length_unit == 'm' else 1.0
    originals = snapshot.positions.tolist()
    converted = [[float(value) * factor for value in row] for row in originals]
    _check_conversion(originals, converted, network.edges.tolist())
    # The legacy exporter sees a local detached view, not caller-replaceable attrs.
    local = copy(network)
    for name in _ARRAYS:
        value = np.array(getattr(network, name), copy=True)
        value.setflags(write=False)
        setattr(local, name, value)
    centers = np.array(converted, dtype=float)
    centers.setflags(write=False)
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(['step', 'incoming_edge_id', 'node_id', 'x_mm', 'y_mm', 'z_mm'])
    route = []
    node_ids, edge_ids = snapshot.node_ids.tolist(), snapshot.edge_ids.tolist()
    edges = network.edges.tolist()
    for step, node in enumerate(network.route_nodes.tolist()):
        edge = None if step == 0 else int(network.route_edges[step - 1])
        writer.writerow([step, -1 if edge is None else edge, node, *centers[node]])
        direction = None
        if edge is not None:
            previous = int(network.route_nodes[step - 1])
            direction = 1 if edges[edge] == [previous, node] else -1
        route.append({'step': step, 'node_row': node, 'node_id': node_ids[node],
                      'incoming_edge_row': edge,
                      'incoming_edge_id': None if edge is None else edge_ids[edge],
                      'source_edge_direction': direction})
    csv_bytes = stream.getvalue().encode('utf-8')
    mapping = {'schema_version': 'fibernet.manufacturing_route_mapping.v1',
               'graph_id': snapshot.graph_ids[0], 'graph_id_origin': 'caller_asserted',
               'source_input_sha256': source_input_sha256,
               'source_input_sha256_origin': 'caller_asserted_not_file_verified' if source_input_sha256 else 'unknown',
               'normalized_source_sha256': source_hash,
               'normalized_source_hash_scope': 'detached GraphBatch v2 payload and typed array bytes',
               'network_binding_sha256': network_hash,
               'route_csv_sha256': hashlib.sha256(csv_bytes).hexdigest(),
               'source_length_unit': snapshot.length_unit, 'route_length_unit': 'mm',
               'coordinate_frame': snapshot.coordinate_frame,
               'conversion': {'factor': factor, 'kind': 'isotropic_unit_conversion_only',
                              'arithmetic': 'binary64; not guaranteed bit-reversible'},
               'csv_identity_semantics': {'node_id': 'dense_node_row', 'incoming_edge_id': 'dense_physical_edge_row',
                                          'initial_incoming_edge_id': -1},
               'source_snapshot': source, 'route': route,
               'material_scope': 'stored records with declared or unknown origin; not effective solver material',
               'claim': 'closed_route_mapping_only_not_printability_or_predictive_evidence'}
    raw = _json_bytes(mapping) + b'\n'
    return local, centers, csv_bytes, raw


def _fresh_paths(path, sidecar_path):
    paths = (Path(path), Path(sidecar_path))
    if paths[0].resolve() == paths[1].resolve():
        raise ValueError('CSV and sidecar must be different destinations')
    for target in paths:
        if not target.parent.is_dir():
            raise ValueError('output parent directory must already exist')
        try:
            target.lstat()
        except FileNotFoundError:
            continue
        raise FileExistsError('mapped export refuses an existing destination: ' + str(target))
    return paths


def _reserve(path):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        info = os.fstat(fd)
        return info.st_dev, info.st_ino
    finally:
        os.close(fd)


def _owned_reservation(path, identity):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or (info.st_dev, info.st_ino) != identity:
        raise RuntimeError('owned output reservation was replaced')


def export_mapped_route(path, network, snapshot, *, sidecar_path,
                        source_input_sha256=None):
    """Export legacy mm CSV plus an explicit stable-ID/provenance JSON sidecar.

    Validate both fresh destinations and the complete mapping before any write.
    Both names are exclusively reserved; the existing exporter publishes the CSV,
    then the sidecar is atomically published last. This is not an atomic pair.
    A valid sidecar plus the matching CSV SHA256 is the completion record. On
    failure, owned empty reservations or completed CSV may remain: no automatic
    deletion of final paths and no retry. Pre-existing destinations are refused.
    Concurrent replacement of owned destinations or their parents is unsupported.
    Only sibling staging files owned by the legacy atomic writer are cleaned up.
    Returned metadata is a fresh JSON-native copy. No PrintSettings is applied.
    """
    from .solid_export import export_route, _owned_export_file
    csv_path, metadata_path = _fresh_paths(path, sidecar_path)
    local, centers, expected_csv, raw_mapping = _prepare_mapping(network, snapshot, source_input_sha256)
    # Reserve only after validation/serialization. A racing creation fails closed;
    # any earlier owned reservation is retained as explicit incomplete output.
    csv_identity = _reserve(csv_path)
    metadata_identity = _reserve(metadata_path)
    _owned_reservation(csv_path, csv_identity)
    export_route(csv_path, local, centers)
    if csv_path.read_bytes() != expected_csv:
        raise RuntimeError('legacy route bytes differ from the complete mapped contract')
    _owned_reservation(metadata_path, metadata_identity)
    with _owned_export_file(metadata_path) as temporary:
        temporary.write_bytes(raw_mapping)
    return json.loads(raw_mapping)
