"""Bounded, explicit admission and loss records for the existing legacy bridge.

These configurations and reports do not own a graph, run a solver, qualify a
manufacturing process, or describe downstream mesh-export success.
"""
from dataclasses import dataclass, field, fields
import math
from numbers import Integral
import struct
import sys


def _ordinary_integer(value):
    return (type(value) is int or (type(value).__module__ == 'numpy'
                                  and isinstance(value, Integral)))


@dataclass(frozen=True)
class FiberNetworkProjectionLimits:
    """Payload admission bounds, not a measured peak-memory guarantee."""
    max_nodes: int = 10000
    max_edges: int = 20000
    max_source_points: int = 100000
    max_output_points: int = 1000000
    max_crosslinks: int = 100000
    max_metadata_bytes: int = 4 * 1024 * 1024
    max_report_bytes: int = 16 * 1024 * 1024

    def __post_init__(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if (not _ordinary_integer(value)
                    or value <= 0 or value > sys.maxsize):
                raise ValueError(item.name + ' must be a positive platform-sized integer')


@dataclass(frozen=True)
class FiberNetworkProfile:
    """Opt into ``(FiberNetwork, report)`` from ``to_fiber_network``.

Unit/frame labels are caller declarations covering coordinates, internal
points, radii and box dimensions together; no recognition or conversion occurs.
"""
    length_unit: str
    coordinate_frame: str
    geometry_policy: str = 'reject_internal_points'
    short_edge_policy: str = 'reject'
    missing_material_policy: str = 'reject'
    limits: FiberNetworkProjectionLimits = field(default_factory=FiberNetworkProjectionLimits)

    def __post_init__(self):
        for name in ('length_unit', 'coordinate_frame'):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError(name + ' must be an explicit nonempty string')
        choices = {'geometry_policy': ('reject_internal_points', 'endpoint_chord'),
                   'short_edge_policy': ('reject', 'omit_legacy'),
                   'missing_material_policy': ('reject', 'legacy_default')}
        for name, allowed in choices.items():
            value = getattr(self, name)
            if type(value) is not str or value not in allowed:
                raise ValueError(name + ' must be one of ' + repr(allowed))
        if type(self.limits) is not FiberNetworkProjectionLimits:
            raise TypeError('limits must be FiberNetworkProjectionLimits')


class FiberNetworkProjectionError(ValueError):
    """A well-formed input requires explicitly permitted projection losses."""
    def __init__(self, violations):
        self.violations = {key: tuple(value) for key, value in violations.items()}
        super().__init__('FiberNetwork projection requires explicit loss policies: ' +
                         ', '.join(key + ' (' + str(len(value)) + ' edges)'
                                   for key, value in self.violations.items()))


class _JsonBudget:
    """Conservative compact ensure_ascii JSON bytes, without first encoding.

Each logical occurrence is charged, even when containers are shared. Integer
digit accounting is conservative by at most one digit; no giant decimal string
or JSON result is allocated merely to determine whether it fits.
"""
    def __init__(self, limit, label):
        self.limit, self.label, self.used = int(limit), label, 0

    def add(self, count):
        self.used += count
        if self.used > self.limit:
            raise MemoryError(self.label + ' budget exceeded')

    def charge(self, value, depth=0, active=None):
        if depth > 100:
            raise ValueError(self.label + ' exceeds 100 nesting levels')
        kind = type(value)
        if value is None:
            self.add(4)
        elif kind is bool:
            self.add(4 if value else 5)
        elif kind is int:
            # 30103/100000 is an upper bound on log10(2).
            digits = max(1, (value.bit_length() * 30103 + 99999) // 100000)
            self.add(digits + (1 if value < 0 else 0))
            encoder_limit = getattr(sys, 'get_int_max_str_digits', lambda: 0)()
            if encoder_limit and digits > encoder_limit:
                raise ValueError(self.label + ' integer exceeds active JSON digit limit')
        elif kind is float:
            if not math.isfinite(value):
                raise ValueError(self.label + ' numbers must be finite')
            self.add(len(repr(value)))
        elif kind is str:
            if len(value) + 2 > self.limit - self.used:
                raise MemoryError(self.label + ' budget exceeded')
            self.add(2)
            for char in value:
                code = ord(char)
                self.add(2 if char in '\"\\\b\f\n\r\t' else
                         12 if code > 65535 else 6 if code >= 127 or code < 32 else 1)
        elif kind in (dict, list):
            active = set() if active is None else active
            if id(value) in active:
                raise ValueError(self.label + ' contains a cycle')
            if kind is dict and any(type(key) is not str for key in value):
                raise ValueError(self.label + ' keys must be strings')
            self.add(2 + max(0, len(value) - 1))
            active.add(id(value))
            try:
                if kind is dict:
                    for key, item in value.items():
                        self.charge(key, depth + 1, active)
                        self.add(1)
                        self.charge(item, depth + 1, active)
                else:
                    for item in value:
                        self.charge(item, depth + 1, active)
            finally:
                active.remove(id(value))
        else:
            raise ValueError(self.label + ' requires JSON-native values')


_LENGTH_KEYS = frozenset(('length_unit', 'coordinate_unit', 'coordinate_units',
                        'position_unit', 'radius_unit', 'internal_points_unit',
                        'curve_unit', 'box_size_unit'))
_UNIT_FIELDS = ('length', 'coordinates', 'position', 'radius', 'internal_points',
                'curve', 'box_size')


def _float64_identity(value):
    """Stored binary64 identity independent of arithmetic; signed zeros agree."""
    bits = struct.unpack('>Q', struct.pack('>d', value))[0]
    return bits if bits & ((1 << 63) - 1) else 0


def _units(value, profile, *, root=True, geometry=False):
    # Called only after the bounded, cycle/depth-checked JSON walk.
    if type(value) is dict:
        for key, item in value.items():
            if key in _LENGTH_KEYS and (root or geometry) and item != profile.length_unit:
                raise ValueError('stored geometry length declaration conflicts')
            if key == 'coordinate_frame' and (root or geometry) and item != profile.coordinate_frame:
                raise ValueError('stored geometry frame declaration conflicts')
            if key == 'units' and (root or geometry):
                if type(item) is str:
                    if item != profile.length_unit:
                        raise ValueError('stored geometry units declaration conflicts')
                elif type(item) is dict:
                    if any(name in item and item[name] != profile.length_unit
                           for name in _UNIT_FIELDS):
                        raise ValueError('stored geometry units declaration conflicts')
                else:
                    raise ValueError('stored geometry units declaration is unsupported')
            _units(item, profile, root=False,
                   geometry=key == 'geometry' and type(item) is dict)
    elif type(value) is list:
        for item in value:
            _units(item, profile, root=False, geometry=geometry)


def _preflight(graph, profile):
    import numpy as np
    from .material import Material
    from .structure_graph import StructureGraph, SNode, SEdge

    if type(graph) is not StructureGraph:
        raise TypeError('checked projection requires an ordinary StructureGraph')
    if type(profile) is not FiberNetworkProfile:
        raise TypeError('profile must be FiberNetworkProfile or None')
    profile.__post_init__()
    profile.limits.__post_init__()
    limits = profile.limits
    if type(graph.nodes) is not dict or type(graph.edges) is not dict:
        raise ValueError('checked graph node/edge containers must be ordinary dictionaries')
    if len(graph.nodes) > limits.max_nodes or len(graph.edges) > limits.max_edges:
        raise MemoryError('graph exceeds node/edge budget')
    if not _ordinary_integer(graph.dimension) or graph.dimension not in (2, 3):
        raise ValueError('dimension must be 2 or 3')
    metadata = _JsonBudget(limits.max_metadata_bytes, 'metadata/material')
    report_budget = _JsonBudget(limits.max_report_bytes, 'report')
    node_order, node_metadata, degree = [], [], {}
    records, curved, projected, omitted, missing, defaulted, unrepresented = ([] for _ in range(7))
    graph_fields = []
    counts = {'source_nodes': len(graph.nodes), 'source_edges': len(graph.edges),
              'source_points': 0, 'fibers': 0, 'centerline_points': 0, 'crosslinks': 0}
    report = {
        'schema_version': 'fibernet.fiber_network_projection.v1',
        'scope': 'conversion_time_correspondence_to_returned_network_only',
        'source_authority': 'StructureGraph; native schema 3.1, not this report',
        'representation': 'endpoint_chord',
        'geometry_policy': profile.geometry_policy,
        'short_edge_policy': profile.short_edge_policy,
        'missing_material_policy': profile.missing_material_policy,
        'length_unit': profile.length_unit, 'coordinate_frame': profile.coordinate_frame,
        'declaration_origin': 'caller_asserted_not_certified',
        'length_scope': ['node_positions', 'internal_points', 'radius', 'box_size'],
        'unit_conversion': 'none', 'frame_transform': 'none',
        'dimension': int(graph.dimension), 'coordinate_components': 'XYZ',
        'source_node_order': node_order, 'edges': records,
        'nonempty_internal_point_edge_ids': curved,
        'projected_internal_point_edge_ids': projected,
        'omitted_edge_ids': omitted, 'missing_material_edge_ids': missing,
        'defaulted_material_edge_ids': defaulted,
        'unrepresented_node_ids': unrepresented, 'counts': counts,
        'metadata': {'graph_fields': graph_fields,
                     'graph_disposition': 'copied_to_network_metadata',
                     'node_fields_not_transferred': node_metadata,
                     'edge_disposition': 'fields_listed_per_edge_not_transferred'},
        'field_disposition': {
            'stable_ids': 'report_only; graph_local_not_global',
            'internal_points': 'not_transferred',
            'node_boundary': 'not_transferred; not_solver_constraints',
            'tolerance_and_id_counters': 'not_transferred',
            'box_size': 'copied_without_activating_periodicity',
            'periodic': 'legacy_false; graph_metadata_not_interpreted',
            'segments': 'legacy_maximum_rule; source_and_output_listed_per_edge'},
        'junction_rule': 'pairwise_welded_crosslinks_at_shared_source_node_IDs_only',
        'crosslink_strength_and_stiffness': 'legacy_infinite_sentinels_not_qualification',
        'short_chord_threshold': 1e-12,
        'not_qualified': ['downstream_mesh_survival', 'manufacturability',
                          'bonded_junctions', 'mechanics', 'lossless_roundtrip'],
    }
    # Admit the fixed envelope/labels before collecting any source-sized report.
    report_budget.charge(report)

    def append_report(target, value):
        report_budget.add(1 if target else 0)
        report_budget.charge(value)
        target.append(value)

    def append_record(target, record, key_sources=()):
        # The small record has empty placeholders for variable-size field lists.
        report_budget.add(1 if target else 0)
        report_budget.charge(record)
        for name, keys in key_sources:
            for key in keys:
                append_report(record[name], key)
        target.append(record)

    def identifier(value, label):
        if not _ordinary_integer(value) or value < 0:
            raise ValueError(label + ' must be an ordinary nonnegative integer, not bool')
        return int(value)

    def number(value, label):
        if (np.ma.isMaskedArray(value) or isinstance(value, (bool, np.bool_))
                or not (type(value) in (int, float) or
                        (type(value).__module__ == 'numpy' and
                         isinstance(value, (np.integer, np.floating))))):
            raise ValueError(label + ' must be an ordinary finite real number')
        original = int(value) if isinstance(value, Integral) else value
        try:
            normalized = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(label + ' exceeds binary64 range') from exc
        if not math.isfinite(normalized) or original != normalized:
            raise ValueError(label + ' must be exactly representable as finite binary64')
        return normalized

    def array(value, shape, label):
        if (type(value) is not np.ndarray or value.shape != shape
                or value.dtype.kind not in 'iuf'):
            raise ValueError(label + ' must be an ordinary real array of shape ' + str(shape))
        for item in value.flat:
            number(item, label)

    def meta(value):
        if type(value) is not dict:
            raise ValueError('metadata must be a JSON-native dictionary')
        metadata.charge(value)
        _units(value, profile)

    material_fields = set(Material.__dataclass_fields__)
    def charge_material(material):
        if type(material) is not Material or set(vars(material)) != material_fields:
            raise ValueError('checked materials require ordinary Material fields only')
        record, absent = {}, []
        for name in Material.__dataclass_fields__:
            value = getattr(material, name)
            if name == 'name':
                if type(value) is not str:
                    raise ValueError('material name must be a string')
                record[name] = value
            elif name == 'extra':
                if type(value) is not dict:
                    raise ValueError('material extra must be a JSON-native dictionary')
                record[name] = value
            elif name == 'stiffness_tensor':
                if value is not None:
                    array(value, (6, 6), 'material stiffness_tensor')
                record[name] = None if value is None else value.tolist()
                if value is None:
                    absent.append(name)
            elif value is None:
                if name in ('density', 'youngs_modulus', 'poissons_ratio'):
                    raise ValueError('required material numbers cannot be None')
                record[name] = None
                absent.append(name)
            else:
                record[name] = number(value, 'material ' + name)
        metadata.charge(record)
        return absent

    meta(graph.metadata)
    for key in graph.metadata:
        append_report(graph_fields, key)
    if graph._box_size is not None:
        array(graph._box_size, (3,), 'box_size')
        if any(number(value, 'box_size') < 0 for value in graph._box_size):
            raise ValueError('box_size must be nonnegative')
    source_points = len(graph.nodes)
    for edge in graph.edges.values():
        if type(edge) is not SEdge:
            raise ValueError('checked edges must be ordinary SEdge records')
        points = edge.internal_points
        if points is not None:
            if type(points) is not np.ndarray or points.ndim != 2 or points.shape[1] != 3:
                raise ValueError('internal_points must have shape (M, 3)')
            source_points += len(points)
        if source_points > limits.max_source_points:
            raise MemoryError('source point budget exceeded')
    if source_points > limits.max_source_points:
        raise MemoryError('source point budget exceeded')
    for key, node in graph.nodes.items():
        if type(node) is not SNode:
            raise ValueError('checked nodes must be ordinary SNode records')
        nid = identifier(key, 'node key')
        if identifier(node.node_id, 'node ID') != nid:
            raise ValueError('node key and record ID disagree')
        array(node.position, (3,), 'node position')
        meta(node.metadata)
        append_report(node_order, nid)
        degree[nid] = 0
        if node.metadata:
            append_record(node_metadata, {'node_id': nid, 'fields': []},
                          [('fields', node.metadata)])
    output_points, fiber_count = 0, 0
    default_material = None
    for source_row, (key, edge) in enumerate(graph.edges.items()):
        eid = identifier(key, 'edge key')
        if identifier(edge.edge_id, 'edge ID') != eid:
            raise ValueError('edge key and record ID disagree')
        a, b = identifier(edge.node_i, 'node_i'), identifier(edge.node_j, 'node_j')
        if a == b or a not in degree or b not in degree:
            raise ValueError('edge endpoints must be distinct existing node IDs')
        if number(edge.radius, 'radius') <= 0:
            raise ValueError('radius must be positive')
        segments = identifier(edge.segments, 'segments')
        if not segments or segments >= sys.maxsize:
            raise ValueError('segments must be a positive platform-sized integer')
        points = edge.internal_points
        count = 0 if points is None else len(points)
        if points is not None:
            array(points, (count, 3), 'internal_points')
        if count:
            append_report(curved, eid)
        meta(edge.metadata)
        material = edge.material
        if material is None:
            append_report(missing, eid)
            absent = None
        else:
            # One charge per source edge, including shared/omitted materials.
            # Retained material copies are bounded by this same occurrence count.
            absent = charge_material(material)
        pi, pj = graph.nodes[a].position, graph.nodes[b].position
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
                delta = pj - pi
                if any(float(d) != float(y) - float(x) for d, x, y in zip(delta, pi, pj)):
                    raise OverflowError('legacy coordinate subtraction loses binary64 difference')
                # Keep NumPy scalar arithmetic identical to the legacy branch.
                dist = np.linalg.norm(delta)
                if not math.isfinite(float(dist)):
                    raise OverflowError('endpoint distance is not finite')
                if dist == 0 and any(_float64_identity(float(x)) != _float64_identity(float(y))
                                     for x, y in zip(pi, pj)):
                    raise OverflowError('endpoint distance lost distinct coordinates')
                short = dist < 1e-12
                selected = None
                if not short:
                    divisor = max(edge.radius * 2, 0.05)
                    ratio = dist / divisor
                    if not math.isfinite(float(divisor)) or not math.isfinite(float(ratio)):
                        raise OverflowError('legacy segment arithmetic is not finite')
                    selected = max(segments, max(4, int(ratio)))
        except FloatingPointError as exc:
            raise OverflowError('legacy projection arithmetic exceeds numeric range') from exc
        row = None
        if short:
            append_report(omitted, eid)
        else:
            output_points += selected + 1
            if output_points > limits.max_output_points or output_points > sys.maxsize:
                raise MemoryError('output centerline point budget exceeded')
            row = fiber_count
            fiber_count += 1
            degree[a] += 1
            degree[b] += 1
            if count:
                append_report(projected, eid)
            if material is None and profile.missing_material_policy == 'legacy_default':
                append_report(defaulted, eid)
                # Charge each retained fallback. One small ordinary prototype is
                # sufficient to measure its current real fields, without copying.
                if default_material is None:
                    default_material = Material()
                charge_material(default_material)
        record = {'source_row': source_row, 'edge_id': eid, 'node_i': a,
                  'node_j': b, 'fiber_row': row,
                  'disposition': 'omitted_short_chord' if short else 'endpoint_chord',
                  'internal_points_state': 'absent' if points is None else
                  'nonempty' if count else 'empty',
                  'internal_point_count': count, 'source_segments': segments,
                  'output_segments': selected,
                  'material_disposition': 'stored_not_transferred' if short and material is not None
                  else 'missing_not_transferred' if short else
                  'missing_rejected' if material is None and profile.missing_material_policy == 'reject'
                  else 'legacy_default' if material is None else 'stored_copy',
                  'material_origin': 'missing' if material is None else 'stored_origin_unknown',
                  'material_missing_optional_fields': absent,
                  'metadata_fields': []}
        append_record(records, record, [('metadata_fields', edge.metadata)])
    crosslinks = sum(value * (value - 1) // 2 for value in degree.values())
    if crosslinks > limits.max_crosslinks or crosslinks > sys.maxsize:
        raise MemoryError('pairwise crosslink budget exceeded')
    for nid in node_order:
        if not degree[nid]:
            append_report(unrepresented, nid)
    new_counts = {'source_points': source_points, 'fibers': fiber_count,
                  'centerline_points': output_points, 'crosslinks': crosslinks}
    for key, value in new_counts.items():
        # Replace the already charged single-digit zero with the final count.
        report_budget.add(max(0, len(str(value)) - 1))
        counts[key] = value
    # Independent final walk checks the assembled JSON envelope, not admission.
    _JsonBudget(limits.max_report_bytes, 'report').charge(report)
    violations = {}
    if curved and profile.geometry_policy == 'reject_internal_points':
        violations['internal_points'] = curved
    if omitted and profile.short_edge_policy == 'reject':
        violations['short_edges'] = omitted
    if missing and profile.missing_material_policy == 'reject':
        violations['missing_materials'] = missing
    if violations:
        raise FiberNetworkProjectionError(violations)
    return report


def _checked_to_fiber_network(graph, profile):
    report = _preflight(graph, profile)
    # Re-enter only the unchanged no-profile branch: one projection algorithm.
    network = graph.to_fiber_network()
    return network, report
