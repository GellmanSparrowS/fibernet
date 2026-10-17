"""Versioned, detached NumPy views of authoritative StructureGraph objects.

No backend imports, graph editing, physical-unit inference or training occurs here.
Read-only flags prevent ordinary accidental writes, not adversarial mutation.
"""
import math
import struct

from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

import numpy as np



def _float64_identity(value):
    """Compare stored finite binary64 values without DAZ; signed zeros agree."""
    bits = struct.unpack('>Q', struct.pack('>d', value))[0]
    return bits if bits & ((1 << 63)-1) else 0


def _distance(a, b, name):
    value = math.dist(a, b)
    if not _float64_identity(value) and any(
            _float64_identity(x) != _float64_identity(y) for x, y in zip(a, b)):
        raise ValueError(name + ' lost a nonzero endpoint distance in float64 arithmetic')
    return value


def _array(value, dtype, shape=None):
    result = np.array(value, dtype=dtype, copy=True)
    if shape is not None:
        result = result.reshape(shape)
    result.setflags(write=False)
    return result


def _id(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(label + ' must be a nonnegative integer')
    if value < 0 or value > np.iinfo(np.int64).max:
        raise ValueError(label + ' is outside supported int64 ID range')
    return int(value)


def _freeze(value, path='metadata', active=None):
    """Copy JSON-like metadata without executing custom deepcopy methods."""
    active = set() if active is None else active
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not np.isfinite(value):
            raise ValueError(path + ' must be finite')
        return value
    if type(value) not in (dict, list):
        raise ValueError(path + ' must contain only JSON-native values')
    if id(value) in active:
        raise ValueError(path + ' contains a cycle')
    if len(active) >= 100:
        raise ValueError(path + ' exceeds 100 nesting levels')
    active.add(id(value))
    try:
        if type(value) is list:
            return tuple(_freeze(x, path, active) for x in value)
        if any(type(k) is not str for k in value):
            raise ValueError(path + ' keys must be strings')
        return MappingProxyType({k: _freeze(v, path + '.' + k, active)
                                 for k, v in value.items()})
    finally:
        active.remove(id(value))


def _metadata(value):
    if type(value) is not dict:
        raise ValueError('metadata must be a JSON-native dictionary')
    return _freeze(value)


def _coordinates(value, shape, label):
    if np.ma.isMaskedArray(value):
        raise ValueError(label + ' must not be a MaskedArray')
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in 'iuf':
        raise ValueError(label + ' must have real numeric shape ' + str(shape))
    result = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(label + ' must be finite')
    if np.any((raw != 0) & (result == 0)):
        raise ValueError(label + ' nonzero values must not underflow to zero in float64')
    return result


@dataclass(frozen=True)
class GraphBatchProvenance:
    """Detached stored provenance, not numeric model features or a graph owner.

    Rows align with the enclosing GraphBatch. Boundary flags describe geometric
    cell faces, not mechanical constraints. Available values have unknown
    explicit/default/derived origin; unavailable fields use None/zero placeholders.
    Material records use authoritative graph serialization semantics, without
    constitutive validation, unit conversion or inference from material names.
    """
    schema_version: str
    node_boundary_face_flags: np.ndarray
    node_boundary_face_available: np.ndarray
    node_boundary_face_origin: Tuple[str, ...]
    edge_material_records: Tuple[Optional[Mapping], ...]
    edge_material_available: np.ndarray
    edge_material_origin: Tuple[str, ...]
    edge_segments: np.ndarray
    edge_segments_available: np.ndarray
    edge_segments_origin: Tuple[str, ...]


@dataclass(frozen=True)
class GraphBatch:
    """Read-only numerical snapshot; IDs are local to each graph, rows global.

    Construct with :func:`graph_batch`, which validates and detaches all fields.
    Direct dataclass construction does not perform validation.
    """
    schema_version: str
    graph_ids: Tuple[str, ...]
    dimensions: np.ndarray
    length_unit: str
    coordinate_frame: str
    node_ptr: np.ndarray
    edge_ptr: np.ndarray
    node_ids: np.ndarray
    edge_ids: np.ndarray
    positions: np.ndarray
    edge_index: np.ndarray
    internal_points: np.ndarray
    internal_ptr: np.ndarray
    internal_available: np.ndarray
    message_index: np.ndarray
    message_parent_rows: np.ndarray
    message_parent_ids: np.ndarray
    message_orientation: np.ndarray
    edge_feature_names: Tuple[str, ...]
    edge_features: np.ndarray
    edge_feature_available: np.ndarray
    graph_metadata: Tuple[Mapping, ...]
    node_metadata: Tuple[Mapping, ...]
    edge_metadata: Tuple[Mapping, ...]
    provenance: Optional[GraphBatchProvenance] = None


def _stored_provenance(graphs):
    """Inspect optional source records only for the explicit v2 opt-in."""
    from fibernet.core.material import Material
    from fibernet.core.structure_graph import _material_record

    faces, face_available, face_origin = [], [], []
    materials, material_available, material_origin = [], [], []
    segments, segment_available, segment_origin = [], [], []
    for graph in graphs:
        for nid in sorted(graph.nodes):
            value = getattr(graph.nodes[nid], 'boundary', None)
            present = value is not None
            if present:
                if (np.ma.isMaskedArray(value)
                        or not isinstance(value, (tuple, list, np.ndarray))
                        or (isinstance(value, np.ndarray) and value.shape != (6,))
                        or len(value) != 6
                        or any(not isinstance(flag, (bool, np.bool_)) for flag in value)):
                    raise ValueError('node boundary face flags must be exactly six booleans or None')
                faces.append(tuple(bool(flag) for flag in value))
            else:
                faces.append((False,) * 6)
            face_available.append(present)
            face_origin.append('stored_origin_unknown' if present else 'missing')
        for edge in graph.edges.values():
            material = getattr(edge, 'material', None)
            present = material is not None
            if present:
                # This first provenance profile does not execute custom material
                # serializers or silently discard subclass-specific state.
                if type(material) is not Material:
                    raise ValueError('edge material provenance requires Material or None')
                try:
                    record = _material_record(Material.to_dict(material))
                except (AttributeError, TypeError, ValueError, OverflowError) as exc:
                    raise ValueError('edge material provenance must be a supported finite JSON record') from exc
                materials.append(_freeze(record, 'edge material'))
            else:
                materials.append(None)
            material_available.append(present)
            material_origin.append('stored_origin_unknown' if present else 'missing')

            value = getattr(edge, 'segments', None)
            present = value is not None
            if present and (isinstance(value, (bool, np.bool_))
                            or not isinstance(value, Integral)
                            or value < 1 or value > np.iinfo(np.int64).max):
                raise ValueError('edge segments provenance must be a positive int64 integer or None')
            segments.append(int(value) if present else 0)
            segment_available.append(present)
            segment_origin.append('stored_origin_unknown' if present else 'missing')
    return GraphBatchProvenance(
        'fibernet.graph_batch.provenance.v1',
        _array(faces, bool, (len(faces), 6)), _array(face_available, bool), tuple(face_origin),
        tuple(materials), _array(material_available, bool), tuple(material_origin),
        _array(segments, np.int64), _array(segment_available, bool), tuple(segment_origin),
    )


def graph_batch(graphs, *, graph_ids, length_unit, coordinate_frame,
                edge_features=(), include_provenance=False):
    """Snapshot graphs without changing IDs, geometry, units or source metadata.

    Nodes use sorted local IDs; physical edges retain insertion order and direction.
    Each physical edge produces forward/reverse messages, including parallel edges.
    Supported opt-in edge features: radius, chord_length, polyline_length.
    None/missing radius uses zero with availability=False; NaN/inf are errors.
    Polyline length follows stored internal points from node_i to node_j; absent
    internal points yield endpoint length (not an assertion about a smooth curve).
    Distance features reject a computed zero for distinct stored endpoints or
    a zero polyline sum of nonzero segments; caller floating-point mode is unchanged.
    Metadata must be JSON-native finite values; lists become tuples and dictionaries
    read-only mappings. Default v1 does not inspect material, boundary or segments.
    include_provenance=True selects v2 and snapshots stored Material records,
    geometric boundary-face flags and meshing segments in a detached payload.
    No mechanical conditions, numeric material features or targets are inferred.
    """
    from fibernet.core.structure_graph import StructureGraph

    if type(include_provenance) is not bool:
        raise ValueError('include_provenance must be a bool')
    graphs = tuple(graphs)
    if isinstance(graph_ids, str):
        raise ValueError('graph_ids must be a sequence of graph names')
    graph_ids = tuple(graph_ids)
    if len(graph_ids) != len(graphs) or any(type(x) is not str or not x.strip()
                                           for x in graph_ids):
        raise ValueError('graph_ids must name every graph with nonempty strings')
    if len(set(graph_ids)) != len(graph_ids):
        raise ValueError('graph_ids must be unique within a batch')
    for name, value in [('length_unit', length_unit), ('coordinate_frame', coordinate_frame)]:
        if type(value) is not str or not value.strip():
            raise ValueError(name + ' must be an explicit nonempty string')
    if isinstance(edge_features, str):
        raise ValueError('edge_features must be a sequence of names')
    names = tuple(edge_features)
    allowed = ('radius', 'chord_length', 'polyline_length')
    if any(type(x) is not str or x not in allowed for x in names) or len(set(names)) != len(names):
        raise ValueError('edge_features contains unknown or duplicate names')
    nids, eids, positions, endpoints, points = [], [], [], [], []
    node_ptr, edge_ptr, internal_ptr = [0], [0], [0]
    available, features, masks, dims = [], [], [], []
    gm, nm, em = [], [], []
    for g in graphs:
        if not isinstance(g, StructureGraph):
            raise TypeError('graphs must contain StructureGraph instances')
        if isinstance(g.dimension, (bool, np.bool_)) or g.dimension not in (2, 3):
            raise ValueError('graph dimension must be 2 or 3')
        dims.append(g.dimension)
        gm.append(_metadata(g.metadata))
        ids = [_id(x, 'node ID') for x in g.nodes]
        ids.sort()
        rows = {nid: len(nids) + i for i, nid in enumerate(ids)}
        local_positions = {}
        for nid in ids:
            node = g.nodes[nid]
            if _id(node.node_id, 'node record ID') != nid:
                raise ValueError('node key and record ID disagree')
            pos = _coordinates(node.position, (3,), 'node position')
            local_positions[nid] = pos
            positions.append(pos)
            nids.append(nid)
            nm.append(_metadata(node.metadata))
        for key, edge in g.edges.items():
            eid = _id(key, 'edge ID')
            if _id(edge.edge_id, 'edge record ID') != eid:
                raise ValueError('edge key and record ID disagree')
            ni, nj = _id(edge.node_i, 'node_i'), _id(edge.node_j, 'node_j')
            if ni not in rows or nj not in rows:
                raise ValueError('edge references a missing node')
            eids.append(eid)
            endpoints.append((rows[ni], rows[nj]))
            em.append(_metadata(edge.metadata))
            interior = edge.internal_points
            has_interior = interior is not None
            if has_interior:
                if np.ma.isMaskedArray(interior):
                    raise ValueError('internal_points must not be a MaskedArray')
                raw = np.asarray(interior)
                if raw.ndim != 2 or raw.shape[1] != 3:
                    raise ValueError('internal_points must have shape (M, 3)')
                interior = _coordinates(raw, raw.shape, 'internal_points')
            else:
                interior = np.empty((0, 3))
            points.extend(interior)
            internal_ptr.append(len(points))
            available.append(has_interior)
            values, valid = [], []
            for name in names:
                present = True
                if name == 'radius':
                    value = getattr(edge, 'radius', None)
                    present = value is not None
                    if not present:
                        value = 0.0
                    elif isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                        raise ValueError('radius must be a finite real number or None')
                elif name == 'chord_length':
                    value = _distance(local_positions[ni], local_positions[nj], name)
                else:
                    path = np.vstack((local_positions[ni], interior, local_positions[nj]))
                    try:
                        segments = [_distance(a, b, name) for a, b in zip(path[:-1], path[1:])]
                        value = math.fsum(segments)
                        if not _float64_identity(value) and any(_float64_identity(x) for x in segments):
                            raise ValueError('polyline_length lost a nonzero segment sum in float64 arithmetic')
                    except OverflowError as exc:
                        raise ValueError('polyline_length exceeds finite float64 range') from exc
                if not np.isfinite(value):
                    raise ValueError(name + ' must be finite')
                original_value = value
                value = float(value)
                if original_value != 0 and value == 0:
                    raise ValueError(name + ' nonzero value must not underflow to zero in float64')
                if not np.isfinite(value):
                    raise ValueError(name + ' must remain finite in float64')
                values.append(value)
                valid.append(present)
            features.append(values)
            masks.append(valid)
        node_ptr.append(len(nids))
        edge_ptr.append(len(eids))
    n, e = len(nids), len(eids)
    ei = np.array(endpoints, dtype=np.int64).reshape(e, 2).T
    messages = np.empty((2, 2 * e), dtype=np.int64)
    messages[:, 0::2], messages[:, 1::2] = ei, ei[::-1]
    return GraphBatch(
        'fibernet.graph_batch.v2' if include_provenance else 'fibernet.graph_batch.v1',
        graph_ids, _array(dims, np.int64),
        length_unit, coordinate_frame, _array(node_ptr, np.int64),
        _array(edge_ptr, np.int64), _array(nids, np.int64), _array(eids, np.int64),
        _array(positions, np.float64, (n, 3)), _array(ei, np.int64),
        _array(points, np.float64, (len(points), 3)), _array(internal_ptr, np.int64),
        _array(available, bool), _array(messages, np.int64),
        _array(np.repeat(np.arange(e), 2), np.int64),
        _array(np.repeat(eids, 2), np.int64),
        _array(np.tile([1, -1], e), np.int8), names,
        _array(features, np.float64, (e, len(names))),
        _array(masks, bool, (e, len(names))), tuple(gm), tuple(nm), tuple(em),
        _stored_provenance(graphs) if include_provenance else None)
