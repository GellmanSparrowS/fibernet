"""Bounded, prepare-only transport geometry from caller-prescribed translations.

The existing StructureGraph remains authoritative. This module neither accepts
mechanical results nor imports/calls a solver. Its detached report binds the
returned values at preparation time, not after subsequent caller mutation.
"""
import hashlib
import json
import math
from types import SimpleNamespace

_CEILINGS = {'max_nodes': 4096, 'max_edges': 16384,
             'max_metadata_bytes': 4 * 1024 * 1024,
             'max_report_bytes': 16 * 1024 * 1024}
_UNITS = {'m': 1.0, 'mm': 1e-3, 'um': 1e-6}
_SCHEMA = 'fibernet.deformed_transport_preparation/1'
_POLICIES = {'geometry_policy': 'endpoint_chord',
             'area_policy': 'fixed_effective_area',
             'conductivity_policy': 'frozen_reference_sigma_A'}


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('preparation requires finite JSON-serializable data') from exc


def _hash(value):
    return hashlib.sha256(_json(value).encode('utf-8')).hexdigest()


def _identifier(value, label):
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError(label + ' must be a nonnegative Python int64, not bool')
    return value


def _number(value, label, *, positive=False):
    import numpy as np
    if not (type(value) in (int, float) or
            (type(value).__module__ == 'numpy' and
             isinstance(value, (np.integer, np.floating)))):
        raise ValueError(label + ' must be an ordinary finite real number, not bool')
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(label + ' exceeds binary64 range') from exc
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(label + ' must be finite' + (' and positive' if positive else ''))
    exact = int(value) == int(result) if isinstance(value, (int, np.integer)) else value == result
    if not exact:
        raise ValueError(label + ' must be exactly representable as binary64')
    return result


def _array(value, shape, label):
    import numpy as np
    if type(value) is not np.ndarray or value.shape != shape or value.dtype.kind not in 'iuf':
        raise ValueError(label + ' requires an ordinary numeric array of shape ' + str(shape))
    for item in value.flat:
        _number(item, label)


def _frame_labels(value, *, root=True, geometry=False):
    """Mirror _units' exact metadata scope after its bounded JSON admission."""
    if type(value) is dict:
        for key, item in value.items():
            if key == 'coordinate_frame' and (root or geometry):
                if type(item) is not str or not item.strip():
                    raise ValueError('stored geometry frame must be a nonempty string')
                yield item
            yield from _frame_labels(item, root=False,
                                     geometry=key == 'geometry' and type(item) is dict)
    elif type(value) is list:
        for item in value:
            yield from _frame_labels(item, root=False, geometry=geometry)


def _index_gate(graph, max_nodes):
    index = graph._spatial_hash
    if type(index) is not dict or len(index) > 4 * max_nodes:
        raise MemoryError('spatial hash exceeds the 4 * max_nodes bucket admission limit')
    total = 0
    for values in index.values():
        if type(values) is not list:
            raise ValueError('spatial hash buckets must be ordinary lists')
        total += len(values)
        if total > len(graph.nodes):
            raise ValueError('spatial hash contains too many node memberships')
    seen = set()
    for key, values in index.items():
        if (type(key) is not tuple or len(key) != 3 or
                any(type(component) is not int for component in key)):
            raise ValueError('spatial hash keys must be integer triples')
        for nid in values:
            _identifier(nid, 'spatial node ID')
            if nid in seen or nid not in graph.nodes:
                raise ValueError('spatial membership must contain each source node exactly once')
            if key != graph._checked_position_key(graph.nodes[nid].position):
                raise ValueError('source spatial membership is stale')
            seen.add(nid)
    if seen != set(graph.nodes):
        raise ValueError('spatial membership is incomplete')
    edge_set = graph._edge_set
    if type(edge_set) is not dict or len(edge_set) > len(graph.edges):
        raise ValueError('edge lookup exceeds edge admission limit')
    expected = {frozenset((edge.node_i, edge.node_j)) for edge in graph.edges.values()}
    if set(edge_set) != expected:
        raise ValueError('edge lookup does not cover endpoint pairs')
    for key, eid in edge_set.items():
        if type(key) is not frozenset:
            raise ValueError('edge lookup key must be an ordinary frozenset')
        for nid in key:
            _identifier(nid, 'edge lookup endpoint ID')
        _identifier(eid, 'edge lookup ID')
        if (type(key) is not frozenset or eid not in graph.edges or
                frozenset((graph.edges[eid].node_i, graph.edges[eid].node_j)) != key):
            raise ValueError('edge lookup has an invalid representative')


def _preflight(graph, length_unit, limits):
    import numpy as np
    from fibernet.core.structure_graph import StructureGraph, SNode, SEdge
    from fibernet.core.material import Material
    from fibernet.core.fiber_network_profile import _JsonBudget, _units

    if type(graph) is not StructureGraph:
        raise TypeError('preparation requires an ordinary StructureGraph')
    if type(graph.nodes) is not dict or type(graph.edges) is not dict:
        raise ValueError('graph node/edge containers must be ordinary dictionaries')
    if not 0 < len(graph.nodes) <= limits['max_nodes'] or not 0 < len(graph.edges) <= limits['max_edges']:
        raise MemoryError('graph exceeds nonempty node/edge admission bounds')
    if type(graph.dimension) is not int or graph.dimension != 2:
        raise ValueError('preparation requires dimension=2')
    if set(vars(graph)) != {'_dimension', '_tolerance', '_box_size', '_nodes', '_edges',
                           '_next_node_id', '_next_edge_id', '_metadata', '_spatial_hash', '_edge_set'}:
        raise ValueError('preparation requires ordinary StructureGraph fields only')
    _number(graph.tolerance, 'tolerance', positive=True)
    for field, ids in (('_next_node_id', graph.nodes), ('_next_edge_id', graph.edges)):
        counter = _identifier(getattr(graph, field), field)
        for key in ids:
            _identifier(key, 'graph ID')
        if counter <= max(ids):
            raise ValueError('next-ID counter must exceed existing IDs')
    if graph._box_size is not None:
        _array(graph._box_size, (3,), 'box_size')
        if any(value < 0 for value in graph._box_size):
            raise ValueError('box_size must be nonnegative')
    budget = _JsonBudget(limits['max_metadata_bytes'], 'aggregate metadata/material')
    frame = [None]
    def metadata(value):
        if type(value) is not dict:
            raise ValueError('metadata must be an ordinary JSON dictionary')
        budget.charge(value)
        for label in _frame_labels(value):
            if frame[0] is not None and frame[0] != label:
                raise ValueError('stored geometry frame declarations conflict')
            frame[0] = label
        _units(value, SimpleNamespace(length_unit=length_unit, coordinate_frame=frame[0]))
    metadata(graph.metadata)
    plane_z = None
    for nid, node in graph.nodes.items():
        if type(node) is not SNode or set(vars(node)) != set(SNode.__dataclass_fields__):
            raise ValueError('nodes must be ordinary SNode records')
        if _identifier(node.node_id, 'node record ID') != nid:
            raise ValueError('node key and record ID disagree')
        _array(node.position, (3,), 'node position')
        if plane_z is None:
            plane_z = float(node.position[2])
        elif float(node.position[2]) != plane_z:
            raise ValueError('dimension=2 positions must share one exact plane z')
        if (type(node.boundary) not in (tuple, list) or len(node.boundary) != 6 or
                any(type(value) is not bool for value in node.boundary)):
            raise ValueError('boundary flags must be six booleans')
        metadata(node.metadata)
    for eid, edge in graph.edges.items():
        if type(edge) is not SEdge or set(vars(edge)) != set(SEdge.__dataclass_fields__):
            raise ValueError('edges must be ordinary SEdge records')
        if _identifier(edge.edge_id, 'edge record ID') != eid:
            raise ValueError('edge key and record ID disagree')
        a, b = _identifier(edge.node_i, 'node_i'), _identifier(edge.node_j, 'node_j')
        if a == b or a not in graph.nodes or b not in graph.nodes:
            raise ValueError('edge endpoints must be distinct existing node IDs')
        _number(edge.radius, 'radius', positive=True)
        if _identifier(edge.segments, 'segments') == 0:
            raise ValueError('segments must be positive')
        if edge.internal_points is not None:
            _array(edge.internal_points, (0, 3), 'internal_points (straight edges only)')
        metadata(edge.metadata)
        material = edge.material
        if type(material) is not Material or set(vars(material)) != set(Material.__dataclass_fields__):
            raise ValueError('preparation requires ordinary Material fields only')
        # Check/bound every field before Material.to_dict or graph.copy can copy it.
        record = {}
        for name in Material.__dataclass_fields__:
            value = getattr(material, name)
            if name == 'stiffness_tensor':
                if value is not None:
                    _array(value, (6, 6), 'material stiffness_tensor')
                    value = value.tolist()  # At most 36 already-validated elements.
            elif name == 'name':
                if type(value) is not str:
                    raise ValueError('material name must be a string')
            elif name == 'extra':
                if type(value) is not dict:
                    raise ValueError('material extra must be a JSON dictionary')
            elif value is not None:
                value = _number(value, 'material ' + name)
            elif name in ('density', 'youngs_modulus', 'poissons_ratio'):
                raise ValueError('required material numbers cannot be None')
            record[name] = value
        budget.charge(record)
    _index_gate(graph, limits['max_nodes'])
    return budget.used, frame[0]


def _complete_map(value, ids, label):
    if type(value) is not dict or len(value) != len(ids):
        raise ValueError(label + ' must be a complete ordinary stable-ID dictionary')
    for key in value:
        _identifier(key, label + ' key')
    if set(value) != set(ids):
        raise ValueError(label + ' has missing or unknown IDs')


def prepare_deformed_transport(graph, *, translations_m_by_id, reference_conductance_S,
                               length_unit,
                               max_nodes=4096, max_edges=16384,
                               max_metadata_bytes=4 * 1024 * 1024,
                               max_report_bytes=16 * 1024 * 1024):
    """Return ``(derived_graph, conductance_S, binding_report)`` without solving.

    Required declaration: graph lengths are m/mm/um. A consistent stored frame
    label is preserved, or reported as undeclared. Each complete Python-int64
    node ID maps to exactly
    ``[ux_m, uy_m]`` in the source graph x/y axes, not xy+rotation or xyz. Each
    physical edge ID has positive reference conductance in S. The ordinary
    dimension-2 graph must have constant z and no nonempty internal points.

    Coordinates use x_new=x_ref+u_m/length_scale, retaining z. Conductance uses
    actual stored endpoint chords: G_new=G_ref*L_ref/L_new, with fixed effective
    area and frozen reference sigma*A. Material/radius fields are preserved,
    never interpreted as conductivity, area or mechanical acceptance. No
    boundary condition, periodic update, frame transform or feedback is implied.

    The four positive Python-int limits can only narrow their default ceilings.
    Metadata/material bytes are aggregate, counted once per occurrence before
    copying. Spatial buckets (including empty ones) are bounded by 4*max_nodes;
    an accumulated valid lookup may therefore exceed this payload admission.
    These limits are not a physical applicability or peak-memory guarantee.
    Exact ordinary graph/record/material types are deliberately narrower than
    some other library APIs. Failed preparation never repairs the source.

    All three outputs are caller-owned. The report is a finite JSON snapshot,
    not an automatically revalidated plan. Call existing linear_transport
    separately with explicit electrical boundary conditions if a solve is wanted.
    """
    import numpy as np
    from fibernet.core.fiber_network_profile import _JsonBudget

    limits = {'max_nodes': max_nodes, 'max_edges': max_edges,
              'max_metadata_bytes': max_metadata_bytes, 'max_report_bytes': max_report_bytes}
    for name, value in limits.items():
        if type(value) is not int or not 0 < value <= _CEILINGS[name]:
            raise ValueError(name + ' must be a positive Python int no greater than its fixed ceiling')
    if type(length_unit) is not str or length_unit not in _UNITS:
        raise ValueError('length_unit must explicitly be m, mm or um')
    metadata_bytes, coordinate_frame = _preflight(graph, length_unit, limits)
    node_order, edge_order = list(graph.nodes), list(graph.edges)
    node_ids, edge_ids = sorted(node_order), sorted(edge_order)
    _complete_map(translations_m_by_id, node_ids, 'translations_m_by_id')
    _complete_map(reference_conductance_S, edge_ids, 'reference_conductance_S')
    translations = {}
    for nid in node_ids:
        row = translations_m_by_id[nid]
        if type(row) is np.ndarray:
            _array(row, (2,), 'translation')
        elif type(row) not in (list, tuple) or len(row) != 2:
            raise ValueError('translation must contain exactly ux_m, uy_m')
        translations[nid] = [_number(value, 'translation') for value in row]
    reference_g = {eid: _number(reference_conductance_S[eid], 'reference conductance', positive=True)
                   for eid in edge_ids}
    scale = _UNITS[length_unit]
    old_positions, positions, new_positions = {}, {}, {}
    for nid in node_ids:
        old = [float(value) for value in graph.nodes[nid].position]
        positions[nid] = [old[0] + translations[nid][0] / scale,
                          old[1] + translations[nid][1] / scale, old[2]]
        old_positions[nid] = [_number(value * scale, 'source SI position') for value in old]
        new_positions[nid] = [_number(value * scale, 'derived SI position') for value in positions[nid]]
        for value in positions[nid]:
            _number(value, 'derived graph position')
        graph._checked_position_key(np.asarray(positions[nid], dtype=float))
    endpoints = [[graph.edges[eid].node_i, graph.edges[eid].node_j] for eid in edge_ids]
    old_lengths, new_lengths, products, new_g = {}, {}, {}, {}
    for eid, (a, b) in zip(edge_ids, endpoints):
        old_lengths[eid] = _number(math.dist(old_positions[a], old_positions[b]), 'reference chord', positive=True)
        new_lengths[eid] = _number(math.dist(new_positions[a], new_positions[b]), 'derived chord', positive=True)
        products[eid] = _number(reference_g[eid] * old_lengths[eid], 'frozen sigma*A', positive=True)
        value = reference_g[eid] if old_lengths[eid] == new_lengths[eid] else products[eid] / new_lengths[eid]
        new_g[eid] = _number(value, 'derived conductance', positive=True)
    box = None if graph._box_size is None else [_number(float(v) * scale, 'reference SI box') for v in graph._box_size]
    report = dict(_POLICIES, schema=_SCHEMA, scope='preparation_time_correspondence_only',
                  input_kind='caller_prescribed_translations', length_unit=length_unit,
                  translation_unit='m', conductance_unit='S', coordinate_frame=coordinate_frame,
                  translation_frame='source_graph_xy_axes', declaration_origin={'length_unit': 'caller_asserted_not_certified',
                                      'coordinate_frame': 'stored_label_not_authenticated' if coordinate_frame else 'undeclared',
                                      'translations': 'caller_prescribed_in_source_graph_xy_axes'},
                  frame_transform='none', displacement_scale=1,
                  mechanical_result_admitted=False, solver_executed=False, feedback=False,
                  source_full_graph_sha256='0' * 64, derived_full_graph_sha256='0' * 64,
                  source_graph_order_envelope_sha256='0' * 64,
                  derived_graph_order_envelope_sha256='0' * 64, input_sha256='0' * 64,
                  source_node_insertion_order=node_order, source_edge_insertion_order=edge_order,
                  node_ids=node_ids, edge_ids=edge_ids, translations_m=[translations[n] for n in node_ids],
                  endpoints_by_id=endpoints, reference_lengths_m=[old_lengths[e] for e in edge_ids],
                  derived_lengths_m=[new_lengths[e] for e in edge_ids],
                  reference_conductance_S=[reference_g[e] for e in edge_ids],
                  conductance_S=[new_g[e] for e in edge_ids],
                  frozen_sigma_A_S_m=[products[e] for e in edge_ids],
                  reference_box_size_m=box,
                  derived_node_AABB_m=[[min(p[c] for p in new_positions.values()) for c in range(3)],
                                       [max(p[c] for p in new_positions.values()) for c in range(3)]],
                  box_boundary_tiling_interpretation='unchanged reference/source provenance only; not updated electrical boundaries or periodic physics',
                  graph_field_disposition='all serialized fields preserved except node positions; lookup remains consistent; materials/radii not interpreted',
                  arithmetic='binary64 actual stored coordinates and endpoint chords',
                  size_admission=dict(limits, fixed_ceilings=dict(_CEILINGS),
                                      max_spatial_buckets=4 * max_nodes,
                                      metadata_material_bytes_charged=metadata_bytes,
                                      scope='payload_bounds_not_peak_memory_or_physical_qualification'))
    _JsonBudget(max_report_bytes, 'report').charge(report)
    # All source-sized and nested inputs are bounded before graph serialization/copy.
    source_record = json.loads(_json(graph.to_dict()))
    source_envelope = {'graph': source_record, 'node_insertion_order': node_order,
                       'edge_insertion_order': edge_order}
    report['source_full_graph_sha256'] = _hash(source_record)
    report['source_graph_order_envelope_sha256'] = _hash(source_envelope)
    report['input_sha256'] = _hash({'schema': _SCHEMA, 'source_graph_order_envelope_sha256': report['source_graph_order_envelope_sha256'],
                                    'node_ids': node_ids, 'edge_ids': edge_ids,
                                    'translations_m': report['translations_m'],
                                    'reference_conductance_S': report['reference_conductance_S'],
                                    'length_unit': length_unit, 'coordinate_frame': coordinate_frame,
                                    'policies': _POLICIES})
    derived = graph.copy()
    derived.set_node_positions(positions)
    _index_gate(derived, max_nodes)
    derived_record = json.loads(_json(derived.to_dict()))
    for source_node, derived_node in zip(source_record['nodes'], derived_record['nodes']):
        if derived_node['position'] != positions[source_node['id']]:
            raise RuntimeError('derived graph did not retain the prepared positions')
        derived_node_without_position = dict(derived_node, position=source_node['position'])
        if derived_node_without_position != source_node:
            raise RuntimeError('derived graph changed non-coordinate node fields')
    comparison = dict(derived_record, nodes=source_record['nodes'])
    if (comparison != source_record or list(derived.nodes) != node_order or
            list(derived.edges) != edge_order or derived._edge_set != graph._edge_set):
        raise RuntimeError('derived graph changed non-coordinate fields or order')
    if _hash({'graph': graph.to_dict(), 'node_insertion_order': list(graph.nodes),
              'edge_insertion_order': list(graph.edges)}) != report['source_graph_order_envelope_sha256']:
        raise RuntimeError('source graph changed during preparation')
    report['derived_full_graph_sha256'] = _hash(derived_record)
    report['derived_graph_order_envelope_sha256'] = _hash({'graph': derived_record,
                                                         'node_insertion_order': node_order,
                                                         'edge_insertion_order': edge_order})
    return derived, {eid: new_g[eid] for eid in edge_order}, json.loads(_json(report))
