"""Strict saved electrothermal fields, followed by explicit one-way mechanics.

Import is stdlib-only. Export/binding uses existing identity/law/load owners,
never a solver. Content hashes detect inconsistent data, not authenticity.
Native input_sha256 is reported, not reconstructed. No root API is extended.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

_SCHEMA = 'fibernet.saved_electrothermal/1'
_MAX_JSON_BYTES = 64 * 1024 * 1024
_MODEL = 'fixed_geometry_temperature_dependent_resistive_lumped_thermal'
_METHOD = 'backward_euler_bounded_picard'
_UNITS = {'temperature': 'K', 'electrical_conductance': 'S',
          'thermal_conductance': 'W/K', 'capacity': 'J/K', 'power': 'W', 'energy': 'J'}
_SCOPE = {
    'upstream_verified': ['node_ids', 'edge_ids', 'oriented_endpoints', 'coordinates', 'length_unit'],
    'caller_supplied_not_upstream_verified': ['dimension', 'radii', 'internal_points', 'graph_materials', 'graph_metadata'],
    'mechanically_authoritative': 'explicit scalar E_Pa and supplied edge expansion_alpha_per_K; graph materials are retained, not applied or calibrated',
    'source_input_sha256': 'reported_native_identifier_not_reconstructed_or_authenticated',
    'electrical_input_sha256': 'verified_per_frame_with_existing_electrical_binding_owner',
    'saved_content_sha256': 'canonical_strict_json_v1_excluding_content_sha256',
    'qualification': 'consistency_binding_not_independent_physics_or_stability_qualification',
}
_NODE_ARRAYS = ('temperature_K', 'reservoir_injection_W', 'node_stored_heat_change_J',
                'interval_stored_heat_J', 'cumulative_source_heat_J', 'cumulative_reservoir_heat_J')
_EDGE_ARRAYS = ('conductance_S', 'edge_heat_rate_W')
_DIAGNOSTICS = ('free_node_residual_J', 'node_acceptance_bound_J',
                'component_energy_residual_J', 'component_acceptance_bound_J')
_SOURCE_FIELDS = ('node_ids', 'edge_ids', 'endpoints_by_id', 'thermal_component_labels',
                  'capacity_J_per_K', 'conductance_W_per_K', 'fixed_node_ids',
                  'fixed_temperature_K', 'iteration_history', 'completed', 'input_sha256')


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('saved electrothermal fields require finite strict JSON') from exc


def _hash(value):
    return hashlib.sha256(_json(value).encode('utf-8')).hexdigest()


def _digest(value, name):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(name + ' requires a lowercase SHA256 digest')
    return value


def _integer(value, name, *, maximum=2**63-1):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(name + ' requires a nonnegative integer, not bool')
    return value


def _number(value, name, *, positive=False, nonnegative=False):
    if type(value) not in (int, float):
        raise ValueError(name + ' requires a finite real number, not bool')
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(name + ' is not representable') from exc
    if not math.isfinite(result) or (positive and result <= 0) or (nonnegative and result < 0):
        raise ValueError(name + ' requires a finite admissible number')
    return result


def _list(value, count, name):
    if type(value) is not list or len(value) != count:
        raise ValueError(name + ' shape mismatch')
    return value


def _ids(value, name, *, allow_empty=False):
    if type(value) is not list or (not value and not allow_empty):
        raise ValueError(name + ' requires an ID array')
    for item in value:
        _integer(item, name)
    if len(set(value)) != len(value):
        raise ValueError(name + ' contains duplicate IDs')
    return value


def _id_map(value, name):
    if type(value) is not dict:
        raise ValueError(name + ' requires a serialized ID map')
    result = {}
    for key, item in value.items():
        if type(key) is not str or not key.isascii() or not key.isdecimal() or str(int(key)) != key:
            raise ValueError(name + ' keys must be canonical nonnegative int64 decimal strings')
        result[_integer(int(key), name)] = item
    return result


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError('nonfinite JSON token: ' + value)


def _decode(raw):
    if len(raw) > _MAX_JSON_BYTES:
        raise ValueError('saved JSON exceeds 64 MiB reader ceiling')
    value = json.loads(raw, object_pairs_hook=_unique, parse_constant=_reject_constant)
    _json(value)  # Reject finite JSON syntax overflowing binary64, e.g. 1e999.
    if type(value) is not dict:
        raise ValueError('saved JSON requires an object')
    return value


def read_saved_electrothermal_json(path):
    """Strictly decode a file without changing it; binding is a separate step."""
    with Path(path).open('rb') as stream:
        return _decode(stream.read(_MAX_JSON_BYTES + 1))


def _graph_record(graph):
    from fibernet.core.structure_graph import StructureGraph
    if not isinstance(graph, StructureGraph):
        raise TypeError('graph must be the existing StructureGraph')
    data = graph.to_dict()
    for key in ('nodes', 'edges'):
        data[key] = sorted(data[key], key=lambda row: row['id'])
    return json.loads(_json(data))


def _bounded_shape(saved):
    if type(saved) is not dict or saved.get('schema') != _SCHEMA:
        raise ValueError('unsupported saved electrothermal envelope schema')
    source = saved.get('source')
    if type(source) is not dict:
        raise ValueError('source requires a native-result record')
    for key, maximum in (('node_ids', 4096), ('edge_ids', 16384), ('frames', 2049),
                         ('iteration_history', 2048 * 25)):
        value = source.get(key)
        if type(value) is not list or len(value) > maximum:
            raise ValueError(key + ' exceeds saved-state resource bounds')
    n, e, f = (len(source[key]) for key in ('node_ids', 'edge_ids', 'frames'))
    if not n or not f or f * (80*n + 80*e) > 2_000_000:
        raise ValueError('saved-state retained history exceeds source scalar bounds')
    # Bound nested input before importing owners or expensive field binding.
    budget = 2_000_000
    pending = [saved]
    while pending:
        budget -= 1
        if budget < 0:
            raise ValueError('saved-state nested scalar budget exceeded')
        item = pending.pop()
        if type(item) is dict:
            if any(type(k) is not str for k in item):
                raise ValueError('strict JSON object keys must be strings')
            pending.extend(item.values())
        elif type(item) is list:
            pending.extend(item)
        elif item is not None and type(item) not in (str, bool, int, float):
            raise ValueError('strict JSON values only')
    return source


def _law(raw, graph, expected_hash):
    from .electrothermal import TemperatureConductanceLaw
    if type(raw) is not dict or set(raw) != {'G0_S', 'alpha_per_K', 'reference_temperature_K',
                                            'endpoint_temperature_weights', 'temperature_domain_K'}:
        raise ValueError('unsupported temperature conductance law record')
    maps = {key: _id_map(raw[key], key) for key in ('G0_S', 'alpha_per_K', 'reference_temperature_K')}
    weights = {eid: _id_map(row, 'temperature endpoint weights')
               for eid, row in _id_map(raw['endpoint_temperature_weights'], 'temperature weights').items()}
    # Native digest uses integer-key ordering and DEFAULT separators. Do not
    # hash the parsed string-key object with the envelope's canonical recipe.
    payload = dict(maps, endpoint_temperature_weights=weights,
                   temperature_domain_K=raw['temperature_domain_K'])
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if digest != _digest(expected_hash, 'law_sha256'):
        raise ValueError('law_sha256 does not match native integer-key law semantics')
    law = TemperatureConductanceLaw(**payload)
    law.binding(graph)
    return law


def _endpoints(data, ids, eids, graph, name):
    if set(_ids(data['node_ids'], name + ' node_ids')) != set(ids):
        raise ValueError(name + ' node identity mismatch')
    nested_eids = _ids(data['edge_ids'], name + ' edge_ids', allow_empty=True)
    if set(nested_eids) != set(eids):
        raise ValueError(name + ' edge identity mismatch')
    pairs = _list(data['endpoints_by_id'], len(nested_eids), name + ' endpoints')
    for eid, pair in zip(nested_eids, pairs):
        _list(pair, 2, name + ' endpoint pair')
        for node in pair:
            _integer(node, name + ' endpoint')
        edge = graph.edges[eid]
        if pair != [edge.node_i, edge.node_j]:
            raise ValueError(name + ' physical endpoint orientation mismatch')
    return nested_eids


def _validate_history(source):
    frames = source['frames']
    history = source['iteration_history']
    groups = {}
    previous = (1, 0)
    for row in history:
        _list(row, 6, 'iteration history row')
        step = _integer(row[0], 'iteration step', maximum=2048)
        iteration = _integer(row[1], 'iteration count', maximum=25)
        if step < 1 or iteration < 1 or not (step == previous[0] and iteration == previous[1]+1 or
                                             step == previous[0]+1 and iteration == 1):
            raise ValueError('iteration history must be an ordered consecutive sequence')
        _number(row[2], 'iteration temperature increment', nonnegative=True)
        if type(row[3]) is not bool or type(row[4]) is not bool or (row[5] is not None and type(row[5]) is not str):
            raise ValueError('malformed iteration history flags or residual reason')
        if row[4] != (row[5] is None):
            raise ValueError('iteration residual reason and availability differ')
        groups.setdefault(step, []).append(row)
        previous = (step, iteration)
    for step, frame in enumerate(frames[1:], 1):
        group = groups.get(step, [])
        if len(group) != frame['accepted_iterations'] or not group or group[-1][3:5] != [True, True]:
            raise ValueError('accepted frame and iteration history disagree')
        if any(row[3] and row[4] for row in group[:-1]):
            raise ValueError('history continued after an accepted iteration')
    allowed_steps = set(range(1, len(frames)))
    if not source['completed']:
        allowed_steps.add(len(frames))
    if set(groups) - allowed_steps:
        raise ValueError('iteration history extends beyond the accepted prefix')
    # Native history is appended before append_frame's retention checks. A
    # terminal successful trial can therefore exist without a retained frame.
    # It stays diagnostic history, never a transferable frame or implicit restart.
    if not source['completed'] and any(row[3] and row[4] for row in groups.get(len(frames), [])[:-1]):
        raise ValueError('partial iteration history continued after a successful terminal trial')


def _validate(graph, saved, length_unit):
    source = _bounded_shape(saved)
    if length_unit not in ('m', 'mm', 'um') or saved.get('length_unit') != length_unit:
        raise ValueError('length_unit must explicitly match m, mm or um')
    if set(saved) != {'schema', 'length_unit', 'graph', 'graph_sha256', 'source',
                      'failure', 'provenance_scope', 'content_sha256'}:
        raise ValueError('unsupported saved envelope fields')
    encoded = _json(saved)
    if len(encoded.encode()) > _MAX_JSON_BYTES:
        raise ValueError('saved JSON exceeds 64 MiB ceiling')
    payload = {k: v for k, v in saved.items() if k != 'content_sha256'}
    if _hash(payload) != _digest(saved['content_sha256'], 'content_sha256'):
        raise ValueError('saved canonical content SHA256 mismatch')
    if saved['provenance_scope'] != _SCOPE:
        raise ValueError('unsupported provenance scope')
    if type(saved['graph']) is not dict or _hash(saved['graph']) != _digest(saved['graph_sha256'], 'graph_sha256'):
        raise ValueError('captured graph SHA256 mismatch')
    actual_graph = _graph_record(graph)
    if _hash(actual_graph) != saved['graph_sha256']:
        raise ValueError('supplied graph differs from captured graph record')
    if set(source) != set(_SOURCE_FIELDS) | {'frames', 'provenance'}:
        raise ValueError('unsupported native source fields')
    if type(source['completed']) is not bool:
        raise ValueError('source completed must be an exact bool')
    _digest(source['input_sha256'], 'reported native input_sha256')
    provenance = source['provenance']
    if (type(provenance) is not dict or provenance.get('model') != _MODEL or
            provenance.get('method') != _METHOD or provenance.get('units') != _UNITS or
            provenance.get('length_unit') != length_unit or
            provenance.get('validation_contract') != 'transport-roundoff-2' or
            provenance.get('geometry_inference') is not False or
            provenance.get('time_accuracy_verified') is not False or
            provenance.get('thermal_capacity_reconstruction_profile') != 'binary64-rn-capacity-rho-1'):
        raise ValueError('unsupported native model, method, provenance or units')
    ids = _ids(source['node_ids'], 'node_ids')
    eids = _ids(source['edge_ids'], 'edge_ids', allow_empty=True)
    if set(ids) != set(graph.nodes) or set(eids) != set(graph.edges):
        raise ValueError('saved source and graph require exact whole-graph stable IDs')
    _endpoints(source, ids, eids, graph, 'source')
    fixed = _ids(source['fixed_node_ids'], 'fixed_node_ids', allow_empty=True)
    if not set(fixed) <= set(ids):
        raise ValueError('unknown fixed thermal node')
    fixed_values = _list(source['fixed_temperature_K'], len(fixed), 'fixed_temperature_K')
    fixed_map = {nid: _number(v, 'fixed temperature', positive=True) for nid, v in zip(fixed, fixed_values)}
    for name, count, positive in (('capacity_J_per_K', len(ids), True),
                                  ('conductance_W_per_K', len(eids), False)):
        for value in _list(source[name], count, name):
            _number(value, name, positive=positive, nonnegative=True)
    labels = _list(source['thermal_component_labels'], len(ids), 'thermal_component_labels')
    for value in labels:
        _integer(value, 'thermal component label', maximum=len(ids)-1)
    component_count = len(set(labels))
    if set(labels) != set(range(component_count)):
        raise ValueError('thermal component labels must be contiguous')
    # Thermal connectivity is deliberately separate from electrical ledgers.
    adjacency = {nid: set() for nid in ids}
    for eid, conductance in zip(eids, source['conductance_W_per_K']):
        edge = graph.edges[eid]
        if conductance > 0:
            adjacency[edge.node_i].add(edge.node_j)
            adjacency[edge.node_j].add(edge.node_i)
    partitions, unseen = [], set(ids)
    while unseen:
        pending, members = [min(unseen)], set()
        while pending:
            nid = pending.pop()
            if nid not in members:
                members.add(nid)
                pending.extend(adjacency[nid]-members)
        unseen -= members
        partitions.append(members)
    labeled = [{nid for nid, label in zip(ids, labels) if label == k} for k in range(component_count)]
    if {frozenset(x) for x in partitions} != {frozenset(x) for x in labeled}:
        raise ValueError('thermal component labels disagree with thermal connectivity')
    law = _law(provenance['law'], graph, provenance['law_sha256'])
    from .joule_heating import _electrical_binding, _assemble_payload
    times, constant_inputs = [], None
    for index, frame in enumerate(source['frames']):
        if type(frame) is not dict or set(frame) != set(_NODE_ARRAYS + _EDGE_ARRAYS + _DIAGNOSTICS) | {
                'time_s', 'accepted_iterations', 'interval_diagnostics_available', 'electrical', 'heat_load'}:
            raise ValueError('unsupported saved frame fields')
        time = _number(frame['time_s'], 'frame time', nonnegative=True)
        times.append(time)
        accepted = _integer(frame['accepted_iterations'], 'accepted_iterations', maximum=25)
        if (index == 0 and accepted != 0) or (index > 0 and accepted == 0):
            raise ValueError('accepted iteration count inconsistent with frame')
        available = frame['interval_diagnostics_available']
        if type(available) is not bool or available != (index > 0):
            raise ValueError('interval diagnostic availability inconsistent with frame')
        for name in _NODE_ARRAYS + _EDGE_ARRAYS:
            count = len(ids) if name in _NODE_ARRAYS else len(eids)
            for value in _list(frame[name], count, name):
                _number(value, name, positive=name == 'temperature_K', nonnegative=name == 'conductance_S')
        temperatures = dict(zip(ids, frame['temperature_K']))
        if any(temperatures[nid] != value for nid, value in fixed_map.items()):
            raise ValueError('saved fixed thermal temperatures changed')
        expected_G = law.evaluate(graph, temperatures)
        actual_G = dict(zip(eids, frame['conductance_S']))
        if expected_G != actual_G:
            raise ValueError('accepted conductance differs from same-state temperature law')
        for name in _DIAGNOSTICS:
            count = 0 if not available else (len(ids) if name in _DIAGNOSTICS[:2] else component_count)
            values = _list(frame[name], count, name)
            for row, value in enumerate(values):
                absent = name in _DIAGNOSTICS[:2] and ids[row] in fixed_map
                if absent:
                    if value is not None:
                        raise ValueError('fixed-node interval diagnostics must be explicit null')
                else:
                    _number(value, name, nonnegative='bound' in name)
        for residual_name, bound_name in ((_DIAGNOSTICS[0], _DIAGNOSTICS[1]),
                                          (_DIAGNOSTICS[2], _DIAGNOSTICS[3])):
            for residual, bound in zip(frame[residual_name], frame[bound_name]):
                if residual is not None and abs(residual) > bound:
                    raise ValueError('accepted diagnostic residual exceeds its recorded bound')
        electric, heat = frame['electrical'], frame['heat_load']
        if type(electric) is not dict or type(heat) is not dict:
            raise ValueError('nested electrical and heat-load records required')
        electric_eids = _endpoints(electric, ids, eids, graph, 'electrical')
        heat_eids = _endpoints(heat, ids, eids, graph, 'heat load')
        _id_map(electric['prescribed_voltage_V'], 'prescribed_voltage_V')
        binding = _electrical_binding(graph, electric, length_unit)
        if dict(zip(electric_eids, electric['conductance_S'])) != actual_G:
            raise ValueError('frame and nested electrical conductances differ')
        if heat.get('provenance', {}).get('electrical_input_sha256') != binding[5]:
            raise ValueError('heat load electrical input SHA256 mismatch')
        if heat.get('provenance', {}).get('model') != 'one_way_constant_joule_load':
            raise ValueError('instantaneous heat-load model label must remain unchanged')
        powers = dict(zip(electric_eids, electric['edge_power_W']))
        if dict(zip(heat_eids, _list(heat['edge_power_W'], len(eids), 'heat edge powers'))) != powers:
            raise ValueError('nested heat-load edge powers differ from electrical state')
        weights = {}
        for eid, pair, row in zip(heat_eids, heat['endpoints_by_id'],
                                  _list(heat['endpoint_weights'], len(eids), 'endpoint heat weights')):
            _list(row, 2, 'endpoint heat weight row')
            weights[eid] = dict(zip(pair, row))
        external = dict(zip(heat['node_ids'], _list(heat['external_node_power_W'], len(ids), 'external heat')))
        # Reuse pure load assembly and its roundoff contracts, not a new solver
        # or copied power equations. Exact JSON equality catches bool-as-number.
        reconstructed = _assemble_payload(graph, electric, endpoint_weights=weights,
                                            external_source_W=external, length_unit=length_unit).to_dict()
        if _json(reconstructed) != _json(heat):
            raise ValueError('saved heat-load fields differ from existing load assembly contract')
        constants = (electric['prescribed_voltage_V'], weights, external)
        if constant_inputs is None:
            constant_inputs = constants
        elif constant_inputs != constants:
            raise ValueError('voltage boundaries, endpoint lumping or external source changed across frames')
    if times[0] != 0 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('times must start at zero and strictly increase')
    if len(times) == 1 and source['completed']:
        raise ValueError('a completed native run requires at least two frames')
    if len(times) > 1:
        dt = times[1]-times[0]
        tolerance = 32 * sys.float_info.epsilon * max(times[-1], dt)
        if any(abs((b-a)-dt) > tolerance for a, b in zip(times, times[1:])):
            raise ValueError('saved times do not satisfy the native uniform-grid allowance')
    _validate_history(source)
    failure = saved['failure']
    if failure is not None:
        if source['completed'] or type(failure) is not dict or set(failure) != {'reason', 'attempted_step', 'attempted_time_s'}:
            raise ValueError('failure context requires an incomplete source and exact fields')
        if type(failure['reason']) is not str:
            raise ValueError('failure reason must be text')
        if _integer(failure['attempted_step'], 'attempted_step', maximum=2048) != len(times):
            raise ValueError('failure attempted step must immediately follow accepted prefix')
        if _number(failure['attempted_time_s'], 'attempted_time_s') <= times[-1]:
            raise ValueError('failure attempted time must follow accepted prefix')
        if len(times) > 1:
            expected_time = times[-1] + (times[1] - times[0])
            allowance = 32 * sys.float_info.epsilon * max(expected_time, times[1] - times[0])
            if abs(failure['attempted_time_s'] - expected_time) > allowance:
                raise ValueError('failure attempted time must follow the saved uniform grid')
    return source


def export_saved_electrothermal(graph, result, *, length_unit, failure=None):
    """Detach a native ElectrothermalResult, preserving accepted-only diagnostics.

    Native fixed-node diagnostic NaNs alone become null; unavailable initial
    diagnostic arrays remain empty. ``failure`` is explicit caught-exception
    context, never inferred from partial_result. The graph must be the original
    upstream geometry; caller mechanical fields are recorded separately.
    """
    from .electrothermal import ElectrothermalResult
    if type(result) is not ElectrothermalResult:
        raise TypeError('result must be the native ElectrothermalResult')
    if (not 1 <= len(result.node_ids) <= 4096 or len(result.edge_ids) > 16384 or
            not 1 <= len(result.frames) <= 2049 or len(result.iteration_history) > 2048 * 25 or
            len(result.frames) * (80*len(result.node_ids) + 80*len(result.edge_ids)) > 2_000_000):
        raise ValueError('native result exceeds saved-state resource bounds')
    source = {name: getattr(result, name) for name in _SOURCE_FIELDS}
    source['provenance'] = _decode(result.provenance_json.encode())
    source['frames'] = []
    fixed = set(result.fixed_node_ids)
    for frame in result.frames:
        row = {name: list(getattr(frame, name)) for name in _NODE_ARRAYS + _EDGE_ARRAYS + _DIAGNOSTICS}
        for name in _DIAGNOSTICS[:2]:
            for index, value in enumerate(row[name]):
                if result.node_ids[index] in fixed and type(value) is float and math.isnan(value):
                    row[name][index] = None
        row.update(time_s=frame.time_s, accepted_iterations=frame.accepted_iterations,
                   interval_diagnostics_available=frame.interval_diagnostics_available,
                   electrical=_decode(frame.electrical_json.encode()), heat_load=_decode(frame.heat_load_json.encode()))
        source['frames'].append(row)
    graph_record = _graph_record(graph)
    value = {'schema': _SCHEMA, 'length_unit': length_unit, 'graph': graph_record,
             'graph_sha256': _hash(graph_record), 'source': source, 'failure': failure,
             'provenance_scope': _SCOPE}
    value = json.loads(_json(value))
    value['content_sha256'] = _hash(value)
    _validate(graph, value, length_unit)
    return value


@dataclass(frozen=True)
class SavedElectrothermalTemperatureSnapshot:
    """Immutable JSON-owned frame selection, with mutation-checked graph binding."""
    data_json: str

    def to_dict(self):
        return json.loads(self.data_json)

    def mechanical_kwargs(self, graph):
        value = self.to_dict()
        if _hash(_graph_record(graph)) != value['graph_sha256']:
            raise ValueError('graph changed since saved-electrothermal preparation')
        return {'length_unit': value['length_unit'],
                'temperature_K': dict(zip(value['node_ids'], value['temperature_K']))}

    def solve(self, graph, *, profile, E_Pa, expansion_alpha_per_K,
              stress_free_temperature_K, constraints, loads=None, size_limits=None):
        """Start NEW mechanics through the public facade; owner errors propagate."""
        if profile not in ('elementary', 'network'):
            raise ValueError("profile must explicitly be 'elementary' or 'network'")
        if size_limits is not None and profile != 'network':
            raise ValueError('size_limits is supported only for profile=network')
        kwargs = self.mechanical_kwargs(graph)
        if size_limits is not None:
            kwargs['size_limits'] = size_limits
        from . import thermoelastic
        owner = (thermoelastic.solve_thermoelastic_snapshot if profile == 'elementary'
                 else thermoelastic.solve_thermoelastic_network_snapshot)
        result = owner(graph, **kwargs, E_Pa=E_Pa, expansion_alpha_per_K=expansion_alpha_per_K,
                       stress_free_temperature_K=stress_free_temperature_K,
                       constraints=constraints, loads=loads)
        return {'schema': 'fibernet.saved_electrothermal_mechanics/1',
                'execution': 'new_mechanical_solve_from_saved_electrothermal_temperature',
                'temperature_source': self.to_dict(), 'mechanical': result.to_dict(),
                'upstream_electrical_thermal_feedback': True,
                'feedback_to_electrical_or_thermal': False,
                'three_field_reciprocal_coupling_or_stability_qualified': False}


def prepare_saved_electrothermal_temperature(graph, saved, *, frame_index,
                                             length_unit, allow_partial=False):
    """Select one exact accepted frame, without interpolation or any solve."""
    if type(allow_partial) is not bool:
        raise ValueError('allow_partial must be an exact bool')
    if type(frame_index) is not int or frame_index < 0:
        raise ValueError('frame_index must be an explicit nonnegative integer')
    try:
        source = _validate(graph, saved, length_unit)
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError('malformed saved electrothermal transfer field: ' + str(exc)) from exc
    if not source['completed'] and not allow_partial:
        raise ValueError('incomplete accepted prefix requires explicit allow_partial=True')
    if frame_index >= len(source['frames']):
        raise ValueError('frame_index does not identify an existing accepted frame')
    if graph.dimension != 2:
        raise ValueError('saved electrothermal mechanics requires dimension=2; no projection')
    z = [graph.nodes[nid].position[2] for nid in source['node_ids']]
    if any(value != z[0] for value in z):
        raise ValueError('frame geometry must lie in one XY plane')
    frame = source['frames'][frame_index]
    record = {'schema': 'fibernet.saved_electrothermal_temperature/1',
              'node_ids': source['node_ids'], 'edge_ids': source['edge_ids'],
              'endpoints_by_id': source['endpoints_by_id'], 'temperature_K': frame['temperature_K'],
              'frame_index': frame_index, 'time_s': frame['time_s'], 'length_unit': length_unit,
              'temperature_unit': 'K', 'source_completed': source['completed'],
              'source_frame_count': len(source['frames']), 'accepted_iterations': frame['accepted_iterations'],
              'source_iteration_history': source['iteration_history'], 'source_failure': saved['failure'],
              'interval_diagnostics_available': frame['interval_diagnostics_available'],
              'source_reported_input_sha256': source['input_sha256'],
              'source_content_sha256': saved['content_sha256'], 'source_law_sha256': source['provenance']['law_sha256'],
              'selected_electrical_input_sha256': frame['electrical']['provenance']['input_sha256'],
              'graph_sha256': saved['graph_sha256'], 'provenance_scope': _SCOPE,
              'interpolation': 'none', 'upstream_electrical_thermal_feedback': True,
              'feedback_to_electrical_or_thermal': False}
    return SavedElectrothermalTemperatureSnapshot(_json(record))
