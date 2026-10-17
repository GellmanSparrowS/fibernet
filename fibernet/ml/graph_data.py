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


@dataclass(frozen=True)
class MaterialFeatureView:
    """Read-only scalar features referring to one existing GraphBatch snapshot.

    This is not a graph owner, Mapping, persistence format or FiberGNN input.
    Rows are physical edges; snapshot.message_parent_rows maps directed messages
    to them. Availability and property_status distinguish missing/null from zero.
    Stored origin, units and constitutive validity cannot be inferred here.
    """
    schema_version: str
    snapshot: GraphBatch
    feature_names: Tuple[str, ...]
    property_units: Tuple[str, ...]
    scales: Tuple[float, ...]
    material_assumption: str
    origin_policy: str
    values: np.ndarray
    available: np.ndarray
    property_status: Tuple[Tuple[str, ...], ...]


_MATERIAL_UNITS = MappingProxyType({
    'youngs_modulus': 'Pa', 'shear_modulus': 'Pa',
    'density': 'kg/m^3', 'poissons_ratio': '1',
})
_MATERIAL_OPTIONAL_SCALARS = (
    'yield_strength', 'tensile_strength', 'fracture_toughness',
    'thermal_conductivity', 'specific_heat', 'thermal_expansion',
    'electrical_conductivity', 'permittivity', 'permeability',
)


def _material_bits(value):
    return struct.unpack('>Q', struct.pack('>d', value))[0]


def _material_number(value, label, positive=False):
    """Convert only built-in scalars; classify zeros/signs without FP compares."""
    if type(value) not in (int, float):
        raise ValueError(label + ' must be a built-in int or float, not bool')
    if type(value) is int:
        nonzero, sign = value != 0, int(value < 0)
    else:
        original = _material_bits(value)
        nonzero, sign = bool(original & ((1 << 63) - 1)), original >> 63
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(label + ' exceeds finite float64 range') from exc
    bits = _material_bits(result)
    if (bits >> 52) & 0x7ff == 0x7ff:
        raise ValueError(label + ' must be finite')
    if nonzero and not bits & ((1 << 63) - 1) or sign != bits >> 63:
        raise ValueError(label + ' lost a nonzero value or sign in float64 conversion')
    if positive and (sign or not bits & ((1 << 63) - 1)):
        raise ValueError(label + ' must be strictly positive')
    return result


def _material_normalize(value, scale):
    before = _material_bits(value)
    try:
        result = value / scale
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError('material normalization exceeds finite float64 arithmetic') from exc
    after = _material_bits(result)
    if ((after >> 52) & 0x7ff == 0x7ff
            or (before & ((1 << 63) - 1) and not after & ((1 << 63) - 1))
            or before >> 63 != after >> 63):
        raise ValueError('material normalization lost finite range, a nonzero value or sign')
    return result


def _material_identity(batch):
    """Validate the existing identity maps used by the material table only."""
    if (type(batch) is not GraphBatch
            or type(batch.schema_version) is not str
            or batch.schema_version != 'fibernet.graph_batch.v2'
            or type(batch.provenance) is not GraphBatchProvenance
            or type(batch.provenance.schema_version) is not str
            or batch.provenance.schema_version != 'fibernet.graph_batch.provenance.v1'):
        raise ValueError('material features require GraphBatch v2 with provenance v1')

    def array(value, dtype, shape, label):
        if (type(value) is not np.ndarray or value.dtype != np.dtype(dtype)
                or value.shape != shape or value.flags.writeable):
            raise ValueError(label + ' must be a read-only base ndarray with canonical dtype/shape')
        return value

    gids = batch.graph_ids
    if (type(gids) is not tuple or any(type(x) is not str or not x.strip() for x in gids)
            or len(set(gids)) != len(gids)):
        raise ValueError('graph_ids must be unique nonempty strings')
    b = len(gids)
    if type(batch.node_ids) is not np.ndarray or type(batch.edge_ids) is not np.ndarray:
        raise ValueError('IDs must be base ndarrays')
    n, e = batch.node_ids.size, batch.edge_ids.size
    nids = array(batch.node_ids, np.int64, (n,), 'node_ids')
    eids = array(batch.edge_ids, np.int64, (e,), 'edge_ids')
    if np.any(nids < 0) or np.any(eids < 0):
        raise ValueError('IDs must be nonnegative')
    dims = array(batch.dimensions, np.int64, (b,), 'dimensions')
    if not np.isin(dims, [2, 3]).all():
        raise ValueError('dimensions must be 2 or 3')

    def pointer(value, end, label):
        p = array(value, np.int64, (b + 1,), label)
        if p[0] != 0 or p[-1] != end or np.any(p[1:] < p[:-1]):
            raise ValueError(label + ' must span its rows monotonically')
        return p

    nptr, eptr = pointer(batch.node_ptr, n, 'node_ptr'), pointer(batch.edge_ptr, e, 'edge_ptr')
    physical = array(batch.edge_index, np.int64, (2, e), 'edge_index')
    messages = array(batch.message_index, np.int64, (2, 2 * e), 'message_index')
    parents = array(batch.message_parent_rows, np.int64, (2 * e,), 'message_parent_rows')
    ids = array(batch.message_parent_ids, np.int64, (2 * e,), 'message_parent_ids')
    orientation = array(batch.message_orientation, np.int8, (2 * e,), 'message_orientation')
    if (not np.array_equal(parents, np.repeat(np.arange(e, dtype=np.int64), 2))
            or not np.array_equal(ids, np.repeat(eids, 2))
            or not np.array_equal(orientation, np.tile([1, -1], e))
            or not np.array_equal(messages[:, ::2], physical)
            or not np.array_equal(messages[:, 1::2], physical[::-1])):
        raise ValueError('messages must preserve physical parent and direction')
    for g in range(b):
        lo, hi, elo, ehi = map(int, (nptr[g], nptr[g+1], eptr[g], eptr[g+1]))
        if ((hi - lo > 1 and np.any(nids[lo+1:hi] <= nids[lo:hi-1]))
                or len(set(eids[elo:ehi])) != ehi-elo):
            raise ValueError('IDs must be sorted unique for nodes and graph-local unique for edges')
        if np.any(physical[:, elo:ehi] < lo) or np.any(physical[:, elo:ehi] >= hi):
            raise ValueError('physical endpoints must stay within their graph')
    p = batch.provenance
    available = array(p.edge_material_available, bool, (e,), 'edge_material_available')
    if (type(p.edge_material_records) is not tuple or len(p.edge_material_records) != e
            or type(p.edge_material_origin) is not tuple or len(p.edge_material_origin) != e):
        raise ValueError('material records and origins must align with physical edges')
    for row, record in enumerate(p.edge_material_records):
        present = record is not None
        origin = p.edge_material_origin[row]
        if (bool(available[row]) != present or type(origin) is not str
                or origin != ('stored_origin_unknown' if present else 'missing')):
            raise ValueError('material availability/origin must agree with its record')
    return e


def material_features_from_batch(batch, *, fields, units, scales,
                                 material_assumption, origin_policy):
    """Derive read-only physical-edge material scalars from a trusted v2 snapshot.

    Select an ordered subset of youngs_modulus/shear_modulus/density/poissons_ratio
    with explicit Pa/Pa/kg/m^3/1 declarations and positive fixed division scales.
    material_assumption='isotropic_scalar' is the caller's assertion, not physical
    validation. origin_policy='retain_stored_origin_unknown' preserves the inability
    to distinguish supplied, defaulted and derived stored values. No G is inferred.

    Non-null tensors, custom extra/unknown fields, unsupported types and new
    nonfinite/nonzero/sign losses fail closed. Missing/null uses unavailable +0;
    genuine signed zero stays available. Binary64 rounding applies; earlier
    serialization losses and general FTZ/DAZ accuracy cannot be certified.

    GraphBatch v1 and the Torch adapter are unchanged. This is not FiberGNN input.
    Geometric face flags, meshing segments and geometry remain provenance only.
    Input must be a factory-produced or equivalently trusted canonical snapshot;
    MappingProxyType does not authenticate its backing mapping. Shape/value checks
    are not hostile-object isolation. Readonly means ordinary accidental writes.
    """
    if type(material_assumption) is not str or material_assumption != 'isotropic_scalar':
        raise ValueError("material_assumption must be 'isotropic_scalar'")
    if type(origin_policy) is not str or origin_policy != 'retain_stored_origin_unknown':
        raise ValueError("origin_policy must be 'retain_stored_origin_unknown'")
    if (type(fields) not in (list, tuple) or not fields
            or any(type(x) is not str or x not in _MATERIAL_UNITS for x in fields)
            or len(set(fields)) != len(fields)):
        raise ValueError('fields must be a nonempty unique sequence of supported names')
    names = tuple(fields)
    for value, label in ((units, 'units'), (scales, 'scales')):
        if (type(value) is not dict or any(type(k) is not str for k in value)
                or set(value) != set(names)):
            raise ValueError(label + ' must be a built-in dict with exactly the selected keys')
    if any(type(units[name]) is not str or units[name] != _MATERIAL_UNITS[name] for name in names):
        raise ValueError('units must explicitly match the supported native property units')
    factors = tuple(_material_number(scales[name], name + ' scale', positive=True) for name in names)
    e = _material_identity(batch)
    allowed = set(_MATERIAL_UNITS) | set(_MATERIAL_OPTIONAL_SCALARS) | {'name', 'extra', 'stiffness_tensor'}
    rows, masks, states = [], [], []
    for record in batch.provenance.edge_material_records:
        if record is not None:
            if type(record) is not MappingProxyType:
                raise ValueError('material record must be a canonical read-only mapping')
            if any(type(k) is not str or k not in allowed for k in record):
                raise ValueError('material record contains unknown fields')
            for key, value in record.items():
                if key == 'name':
                    if type(value) is not str:
                        raise ValueError('material name must be a built-in string')
                elif key == 'extra':
                    if type(value) is not MappingProxyType or len(value):
                        raise ValueError('custom material extra is unsupported')
                elif key == 'stiffness_tensor':
                    if value is not None:
                        raise ValueError('anisotropic stiffness_tensor is unsupported')
                elif value is None:
                    if key in ('density', 'youngs_modulus', 'poissons_ratio'):
                        raise ValueError(key + ' cannot be null')
                else:
                    _material_number(value, key)
        row, mask, status = [], [], []
        for name, factor in zip(names, factors):
            state = ('material_missing' if record is None else
                     'property_missing' if name not in record else
                     'property_null' if record[name] is None else 'stored_origin_unknown')
            present = state == 'stored_origin_unknown'
            row.append(_material_normalize(_material_number(record[name], name), factor) if present else 0.0)
            mask.append(present)
            status.append(state)
        rows.append(row)
        masks.append(mask)
        states.append(tuple(status))
    values = _array(rows, np.float64, (e, len(names)))
    if values.tobytes() != b''.join(struct.pack('=d', x) for row in rows for x in row):
        raise ValueError('material array conversion changed stored binary64 values')
    return MaterialFeatureView(
        'fibernet.material_feature_view.v1', batch, names,
        tuple(units[name] for name in names), factors, material_assumption, origin_policy,
        values, _array(masks, bool, (e, len(names))), tuple(states))
