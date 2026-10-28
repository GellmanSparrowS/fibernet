"""Saved archived-n1 GE result projection, not a native provider capability.

Consumes the existing StructureGraph and immutable archived n1 expanded-endpoint
binding. Translations are projected only after occurrence agreement; rotations,
forces and energies are not recast as spring/axial SimResult fields.
"""
from dataclasses import dataclass
import hashlib, json, math
from fibernet.core.structure_graph import StructureGraph

__all__ = ["ArchivedGEProjection", "import_archived_ge_state"]

def _si_units():
    return {'length': 'm', 'time': 's', 'mass': 'kg', 'frame': 'global Cartesian'}
SI = _si_units()

def raw_json(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def sha(x):
    return hashlib.sha256(x).hexdigest()

def require(ok, message):
    if not ok:
        raise ValueError(message)

def finite_vector(v, n):
    return isinstance(v, list) and len(v) == n and all((type(x) in (int, float) and math.isfinite(x) for x in v))

@dataclass(frozen=True)
class ArchivedGEProjection:
    """Immutable bytes, no provider object; each decode is a detached copy."""
    translation_json: bytes
    ge_sidecar_json: bytes
    original_graph_bytes: bytes
    original_header_bytes: bytes
    original_state_bytes: bytes

    def translations(self):
        return json.loads(self.translation_json)

    def sidecar(self):
        return json.loads(self.ge_sidecar_json)

def _project_validated_records(graph, *, source_graph_bytes, header_bytes, state_bytes, expected_sha256, section_override, units, occurrence_tolerance_m):
    """Import ONE pinned accepted state, never initialize/call a native provider.

 `section_override` acknowledges that source Material may be an unassigned
 placeholder: native sectionData is authoritative, with explicit SI provenance.
 Row order is sorted physical ID, not graph insertion/native numbering. Selecting
 the lowest-ID agreeing endpoint occurrence is explicit; no rotational averaging.
 """
    require(isinstance(graph, StructureGraph), 'existing StructureGraph required')
    require(units == _si_units(), 'explicit supported SI/global Cartesian units required')
    require(type(occurrence_tolerance_m) in (int, float) and math.isfinite(occurrence_tolerance_m) and (0 <= occurrence_tolerance_m <= 1e-09), 'explicit occurrence tolerance in [0,1e-9] m')
    blobs = {'graph': source_graph_bytes, 'header': header_bytes, 'state': state_bytes}
    require(set(expected_sha256) == set(blobs), 'all three immutable byte identities required')
    for key, blob in blobs.items():
        require(type(blob) is bytes and len(blob) <= 20 * 1024 ** 2 and (sha(blob) == expected_sha256[key]), key + ' identity/size mismatch')
    source, h, s = (json.loads(blobs[k]) for k in ('graph', 'header', 'state'))
    b = h['binding']
    assembly = b['assembly']
    p = assembly['reference_plan']
    require(p['subdivisions'] == 1 and p['new_nodes'] == [], 'only archived n1 expanded endpoints supported')
    require(p['model'] == 'fibernet.new-ge-reference-lobatto-contact/1', 'unknown native model')
    require(p['source_graph'] == source, 'binding source record differs from original graph bytes')
    canonical_graph = StructureGraph.from_dict(source).to_dict()
    require(graph.to_dict() == canonical_graph, 'graph topology/material/geometry/metadata mismatch')
    nodes = {n['id']: n for n in canonical_graph['nodes']}
    edges = {e['id']: e for e in canonical_graph['edges']}
    require(all((n.node_id == i for i, n in graph.nodes.items())) and all((e.edge_id == i for i, e in graph.edges.items())), 'dictionary/physical ID mismatch')
    for e in edges.values():
        require(e['internal_points'] is None and e['segments'] == 1, 'curves/implicit subdivision unsupported')
        require(e['node_i'] != e['node_j'], 'self-loop unsupported')
    sections = [dict(parent_edge_id=v['parent_edge_id'], segment_index=v['segment_index'], section_id=v['section_id'], radius_m=v['radius_m'], sectionData=v['sectionData']) for v in p['beams']]
    required = {'policy', 'reason', 'units', 'source_graph_sha256', 'binding_sha256', 'native_sections_sha256'}
    require(isinstance(section_override, dict) and set(section_override) == required, 'explicit section override provenance required')
    require(section_override['policy'] == 'archived-native-sectionData-overrides-source-Material' and isinstance(section_override['reason'], str) and bool(section_override['reason'].strip()), 'section override acknowledgement required')
    require(section_override['units'] == _si_units() and section_override['source_graph_sha256'] == sha(source_graph_bytes) and (section_override['binding_sha256'] == sha(raw_json(b))) and (section_override['native_sections_sha256'] == sha(raw_json(sections))), 'section override provenance mismatch')
    require(s['schema'] in ('fibernet.complete-accepted-state/1', 'fibernet.linked-complete-accepted-state/1') and s['header_sha256'] == sha(header_bytes), 'accepted-state schema/header link')
    step = s['step']
    require(type(step) is int and step >= 0 and (type(s['time_s']) in (int, float)) and math.isfinite(s['time_s']) and (s['time_s'] >= 0), 'global step/time')
    anchor = s.get('anchor', False)
    require(type(anchor) is bool, 'anchor flag must be boolean')
    if s['schema'].startswith('fibernet.linked-'):
        local = s['local_step']
        initial = h['initial_global_step']
        initial_time = h['initial_absolute_time_s']
        require(type(local) is int and local >= 0 and (type(initial) is int) and (initial >= 0), 'nonnegative linked step indices')
        require(s['global_step'] == step and local == step - initial and (s['parent_state_sha256'] == h['parent_link']['state_sha256']), 'linked global/parent identity')
        require(anchor == (local == 0 and step == initial), 'anchor must identify exactly the linked initial state')
        clock = h['clock']
        dt = clock['step_size_s']
        tol = clock['absolute_time_tolerance_s']
        require(type(dt) in (int, float) and math.isfinite(dt) and (dt > 0) and (type(tol) in (int, float)) and math.isfinite(tol) and (0 <= tol <= 1e-15), 'supported finite positive clock and bounded tolerance')
        require(type(initial_time) in (int, float) and math.isfinite(initial_time) and (initial_time >= 0), 'nonnegative finite initial clock')
        require(clock['absolute_time'] is True and abs(s['time_s'] - (initial_time + local * dt)) <= tol, 'linked absolute clock')
    else:
        require(anchor is False, 'nonlinked trace cannot declare imported anchor')
    n = h['ode2_size']
    aen = h['ae_size']
    d = s['diagnostics']
    for key, size in [('q', n), ('v', n), ('a', n), ('ae', aen)]:
        require(finite_vector(s[key], size), 'complete native ' + key)
    require(finite_vector(d['algorithmic_acceleration'], n), 'complete current GA algorithmic state')
    require('start_of_step_algorithmic_acceleration' in d, 'previous algorithmic phase field required')
    previous = d['start_of_step_algorithmic_acceleration']
    require(finite_vector(previous, n) or ((step == 0 or anchor) and previous is None), 'phase-qualified previous algorithmic cache required')
    require(d['data_coordinates'] == [] and d['ode1_coordinates'] == [], 'unknown extra state unsupported')
    if step > 0 and (not anchor):
        require(d.get('newton_converged') is True and d.get('discontinuous_success') is True and (d.get('linear_solver_failed') is False), 'unaccepted native state')
    refs = b['reference_frames']
    ltg = b['layout']['node_ltg_ode2']
    occ = {i: set() for i in nodes}
    emap = {}
    native_to_physical = {}
    native_beams = {(v['parent_edge_id'], v['segment']): v['object'] for v in assembly['beams']}
    require(len(native_beams) == len(p['beams']) == len(edges), 'one native beam per physical edge required')
    for beam in p['beams']:
        eid = beam['parent_edge_id']
        require(eid in edges and eid not in emap and (beam['segment_index'] == 0) and (beam['segments'] == 1), 'stable edge/segment mapping')
        e = edges[eid]
        a, bn = beam['nodeNumbers']
        ra = refs[str(a)]
        rb = refs[str(bn)]
        require(ra['nodeType'] == rb['nodeType'] == 'RigidBodyRxyz' and ra['referenceCoordinates'][3:] == rb['referenceCoordinates'][3:], 'curved/twisted initial endpoints unsupported')
        pa, pb = (ra['referenceCoordinates'][:3], rb['referenceCoordinates'][:3])
        direct = pa == nodes[e['node_i']]['position'] and pb == nodes[e['node_j']]['position']
        reverse = pb == nodes[e['node_i']]['position'] and pa == nodes[e['node_j']]['position']
        require(direct != reverse, 'endpoint incidence/reference positions not unambiguous')
        require(math.isclose(math.dist(pa, pb), beam['length'], rel_tol=1e-12, abs_tol=1e-15) and beam['radius_m'] == e['radius'], 'straight beam length/radius mismatch')
        pair = (e['node_i'], e['node_j']) if direct else (e['node_j'], e['node_i'])
        for native, physical in zip((a, bn), pair):
            require(native not in native_to_physical or native_to_physical[native] == physical, 'native endpoint has conflicting physical IDs')
            native_to_physical[native] = physical
            occ[physical].add(native)
        emap[eid] = dict(native_object=native_beams[eid, 0], native_nodes=[a, bn], native_endpoint_physical_ids=list(pair), orientation_matches_source=direct, segment_index=0, section_id=beam['section_id'])
    require(all(occ.values()), 'unmapped physical node')
    rows = []
    max_disagreement = 0.0
    for nid in sorted(nodes):
        ids = sorted(occ[nid])
        values = []
        for native in ids:
            dofs = ltg[str(native)]
            require(len(dofs) == 6 and len(set(dofs)) == 6 and all((type(i) is int and 0 <= i < n for i in dofs)), 'Rxyz LTG layout')
            values.append([s['q'][i] for i in dofs[:3]])
        delta = max((abs(x - y) for v in values for x, y in zip(v, values[0])), default=0.0)
        max_disagreement = max(max_disagreement, delta)
        require(delta <= occurrence_tolerance_m, 'weld occurrence translations disagree; projection refused')
        rows.append(dict(node_id=nid, displacement_m=values[0], position_m=[x + y for x, y in zip(nodes[nid]['position'], values[0])], native_occurrences=ids, selected_native_occurrence=ids[0]))
    markers = {v['old_id']: v['fields'] for v in p['retained_markers']}
    clamps = []
    for item in p['retained_objects']:
        f = item['fields']
        if f['objectType'] == 'ConnectorCoordinate' and f['activeConnector']:
            mark = markers[f['markerNumbers'][1]]
            native = mark['nodeNumber']
            clamps.append(dict(native_object=assembly['retained_object_map'][str(item['old_id'])], native_node=native, physical_node_id=native_to_physical.get(native), coordinate=mark['coordinate']))
    projection = dict(schema='fibernet.archived-ge-translations/1', node_ids=sorted(nodes), edge_ids=sorted(edges), rows=rows, time_s=s['time_s'], global_step=step, units=_si_units(), occurrence_tolerance_m=occurrence_tolerance_m, max_occurrence_disagreement_m=max_disagreement, rotation_projection='none', edge_forces='not exported; no axial-force reinterpretation', energy='not exported; preserve labelled native diagnostics separately')
    sidecar = dict(schema='fibernet.archived-expanded-web-ge-sidecar/1', identity=expected_sha256, native_state=s, binding=h['binding'], runtime_identity=h['runtime_identity'], static_metadata=h['static_metadata'], parent_link=h.get('parent_link'), edge_mapping=emap, node_mapping={str(k): sorted(v) for k, v in occ.items()}, contact_stations=p['contact_sites'], clamps=clamps, section_override=section_override, native_sections=sections, not_a_simresult_dynamics_state=True, not_restart_validation=True)
    return ArchivedGEProjection(raw_json(projection), raw_json(sidecar), source_graph_bytes, header_bytes, state_bytes)

def _record_shape_check(graph_record, header, state):
    """Validate format structure, not native assembly correctness or physics."""

    def obj(value, label):
        require(type(value) is dict, label + ' must be an object')
        return value

    def count(value, label, maximum, minimum=0):
        require(type(value) is int and minimum <= value <= maximum, label + ' is outside the supported bounds')
        return value

    def number(value, label, positive=False):
        require(type(value) in (int, float) and math.isfinite(value), label + ' must be finite')
        require(not positive or value > 0, label + ' must be positive')

    def matrix(value, size, label):
        require(type(value) is list and len(value) == size and all((finite_vector(row, size) for row in value)), label + ' has an invalid shape or nonfinite entries')
    source = obj(graph_record, 'source graph')
    require(source.get('version') in ('3.0', '3.1'), 'unsupported graph schema')
    for field in ('nodes', 'edges'):
        require(type(source[field]) is list and 1 <= len(source[field]) <= 10000, 'source graph count outside supported bounds')
    h = obj(header, 'header')
    require(h['schema'] in ('fibernet.new-nodal-contact-state-trace/1', 'fibernet.linked-nodal-contact-state-trace/1'), 'unsupported native trace header schema')
    s = obj(state, 'state')
    require(s['schema'] in ('fibernet.complete-accepted-state/1', 'fibernet.linked-complete-accepted-state/1'), 'unsupported accepted state schema')
    require(('linked-' in s['schema']) == ('linked-' in h['schema']), 'header and state schema families differ')
    n = count(h['ode2_size'], 'ODE2 size', 200000, 1)
    count(h['ae_size'], 'AE size', 200000)
    obj(h['runtime_identity'], 'runtime identity')
    obj(h['static_metadata'], 'static metadata')
    obj(s['diagnostics'], 'diagnostics')
    count(s['step'], 'global step', 2**53)
    if 'linked-' in s['schema']:
        count(s['global_step'], 'linked global step', 2**53)
    b = obj(h['binding'], 'binding')
    assembly = obj(b['assembly'], 'assembly')
    plan = obj(assembly['reference_plan'], 'reference plan')
    refs = obj(b['reference_frames'], 'reference frames')
    ltg = obj(obj(b['layout'], 'layout')['node_ltg_ode2'], 'node LTG')
    native_map = obj(assembly['node_map'], 'native node map')
    require(type(plan['beams']) is list and 1 <= len(plan['beams']) <= 10000, 'native beam count outside supported bounds')
    require(type(assembly['beams']) is list and len(assembly['beams']) == len(plan['beams']), 'native object mapping coverage')
    object_ids = [row['object'] for row in assembly['beams']]
    require(all((type(i) is int and i >= 0 for i in object_ids)) and len(set(object_ids)) == len(object_ids), 'colliding native beam object IDs')
    endpoints = set()
    for beam in plan['beams']:
        obj(beam, 'beam')
        pair = beam['nodeNumbers']
        require(type(pair) is list and len(pair) == 2 and all((type(i) is int and i >= 0 for i in pair)) and (pair[0] != pair[1]), 'invalid native beam endpoint IDs')
        endpoints.update(pair)
        number(beam['length'], 'beam length', positive=True)
        number(beam['radius_m'], 'beam radius', positive=True)
        require(type(beam['section_id']) is str and bool(beam['section_id']), 'section identity required')
        section = obj(beam['sectionData'], 'sectionData')
        matrix(section['stiffnessMatrix'], 6, 'section stiffness')
        matrix(section['dampingMatrix'], 6, 'section damping')
        matrix(section['inertia'], 3, 'section inertia')
        number(section['massPerLength'], 'section mass per length', positive=True)
    all_dofs = set()
    mapped_native_ids = set()
    for native in endpoints:
        key = str(native)
        ref = obj(refs[key], 'reference frame')
        require(ref['nodeType'] == 'RigidBodyRxyz' and finite_vector(ref['referenceCoordinates'], 6), 'mapped Rxyz reference coordinates must have six finite entries')
        indices = ltg[key]
        require(type(indices) is list and len(indices) == 6 and all((type(i) is int and 0 <= i < n for i in indices)) and (len(set(indices)) == 6) and (not all_dofs.intersection(indices)), 'invalid or colliding LTG coordinates across native endpoints')
        all_dofs.update(indices)
        mapped = native_map[key]
        require(type(mapped) is int and mapped >= 0 and (mapped not in mapped_native_ids), 'invalid or colliding native node map')
        mapped_native_ids.add(mapped)
    require(type(plan['retained_markers']) is list and type(plan['retained_objects']) is list and (len(plan['retained_markers']) <= 100000) and (len(plan['retained_objects']) <= 100000), 'retained item bounds')
    markers = {row['old_id']: row['fields'] for row in plan['retained_markers']}
    require(len(markers) == len(plan['retained_markers']), 'duplicate retained marker IDs')
    obj(assembly['retained_object_map'], 'retained object map')
    for row in plan['retained_objects']:
        fields = obj(row['fields'], 'retained object fields')
        if fields['objectType'] == 'ConnectorCoordinate':
            require(type(fields['activeConnector']) is bool, 'clamp activation must be boolean')
            if fields['activeConnector']:
                pair = fields['markerNumbers']
                require(type(pair) is list and len(pair) == 2, 'clamp marker pair')
                marker = markers[pair[1]]
                require(marker['nodeNumber'] in endpoints and type(marker['coordinate']) is int and (0 <= marker['coordinate'] < 6), 'unmapped clamp occurrence or invalid coordinate')
                native_object = assembly['retained_object_map'][str(row['old_id'])]
                require(type(native_object) is int and native_object >= 0, 'invalid mapped clamp object')
    require(type(plan['contact_sites']) is list and len(plan['contact_sites']) <= 20000, 'contact station bounds')
    edge_ids = {beam['parent_edge_id'] for beam in plan['beams']}
    for site in plan['contact_sites']:
        require(site['parent_edge_id'] in edge_ids and site['node_id'] in endpoints and (type(site['station']) is int) and (site['station'] in (0, 1)), 'invalid n1 contact station identity')
        number(site['reference_weight_m'], 'reference station weight', positive=True)

def _decode_record(blob):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON object key')
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError('nonfinite JSON constant is unsupported: ' + value)

    return json.loads(blob, object_pairs_hook=unique_pairs, parse_constant=reject_constant)


def import_archived_ge_state(graph, *, source_graph_bytes, header_bytes, state_bytes, expected_sha256, section_override, units, occurrence_tolerance_m):
    """Project one previously validated archived-n1 state without a provider.

    Raises ValueError for malformed/unsupported records or mismatched identities,
    and TypeError only when graph is not an existing StructureGraph. Trusted
    manifests identify bytes; they do not certify native assembly correctness.
    No ancestor traversal, physical qualification or restart validation occurs.
    """
    if not isinstance(graph, StructureGraph):
        raise TypeError('graph must be an existing StructureGraph')
    blobs = (source_graph_bytes, header_bytes, state_bytes)
    require(all((type(blob) is bytes and len(blob) <= 20 * 1024 ** 2 for blob in blobs)), 'three byte records of at most 20 MiB each are required')
    require(type(expected_sha256) is dict and set(expected_sha256) == {'graph', 'header', 'state'}, 'trusted graph/header/state hash map required')
    for key, blob in zip(('graph', 'header', 'state'), blobs):
        require(type(expected_sha256[key]) is str and sha(blob) == expected_sha256[key], key + ' identity mismatch')
    try:
        records = tuple((_decode_record(blob) for blob in blobs))
        _record_shape_check(*records)
    except (KeyError, TypeError, IndexError, OverflowError, RecursionError) as error:
        raise ValueError('malformed archived-n1 record: missing or invalid field') from error
    try:
        return _project_validated_records(graph, source_graph_bytes=source_graph_bytes, header_bytes=header_bytes, state_bytes=state_bytes, expected_sha256=expected_sha256, section_override=section_override, units=units, occurrence_tolerance_m=occurrence_tolerance_m)
    except (KeyError, TypeError, IndexError, OverflowError, RecursionError) as error:
        raise ValueError('malformed archived-n1 record: inconsistent projection fields') from error
