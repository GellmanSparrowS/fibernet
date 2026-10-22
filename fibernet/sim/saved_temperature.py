"""Replay one saved temperature frame into existing prescribed-temperature mechanics.

No electrical, thermal or mechanical solver runs during preparation. This is
field binding, not result authentication, restart, energy validation or a new
coupling model. The existing mechanical owner retains all applicability limits.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

_MAX_JSON_BYTES = 64 * 1024 * 1024


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('saved fields must be finite strict-JSON data') from exc


def _id(value):
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError('saved IDs must be nonnegative int64 integers, not bools')
    return value


def _ids(value, label):
    if not isinstance(value, list) or not value:
        raise ValueError(label + ' requires a nonempty ID list')
    result = [_id(item) for item in value]
    if len(set(result)) != len(result):
        raise ValueError(label + ' contains duplicate IDs')
    return result


def _number(value, label, *, positive=False):
    if type(value) not in (int, float):
        raise ValueError(label + ' requires real non-bool numbers')
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(label + ' is not representable') from exc
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(label + ' must be finite' + (' and positive' if positive else ''))
    return value


def _graph_signature(graph):
    from fibernet.core.structure_graph import StructureGraph
    if not isinstance(graph, StructureGraph):
        raise TypeError('graph must be the existing StructureGraph')
    # Exact original graph data, including physical parallel edges, radii and
    # internal geometry. Record insertion order is not physical identity.
    data = graph.to_dict()
    for key in ('nodes', 'edges'):
        data[key] = sorted(data[key], key=lambda item: item['id'])
    return hashlib.sha256(_json(data).encode()).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON object key: ' + key)
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError('nonfinite JSON token: ' + value)


def read_saved_thermal_json(path):
    """Read a thermal.to_dict() object, or a bundle with a thermal member.

    Reject duplicate object keys, nonfinite JSON numbers and files over 64 MiB.
    Other bundle members are retained, not interpreted as authenticated or
    causally linked electrical evidence. No file is modified.
    """
    with Path(path).open('rb') as stream:
        raw = stream.read(_MAX_JSON_BYTES + 1)
    if len(raw) > _MAX_JSON_BYTES:
        raise ValueError('saved JSON exceeds 64 MiB reader limit')
    data = json.loads(raw, object_pairs_hook=_unique_object,
                      parse_constant=_reject_constant)
    # Also rejects valid JSON exponent syntax that overflows Python binary64.
    _json(data)
    if not isinstance(data, dict):
        raise ValueError('saved JSON must contain an object')
    return data


@dataclass(frozen=True)
class SavedTemperatureSnapshot:
    """Detached, JSON-owned selection; graph is rechecked before delegation."""
    data_json: str

    def to_dict(self):
        return json.loads(self.data_json)

    def mechanical_kwargs(self, graph):
        """Return only explicit length units and temperature keyed by stable ID.

        E_Pa, alpha, stress-free temperature, supports and loads remain separate
        caller inputs. A changed graph requires fresh preparation, never remapping.
        """
        data = self.to_dict()
        if _graph_signature(graph) != data['graph_sha256']:
            raise ValueError('graph changed since saved-temperature preparation')
        return {'length_unit': data['length_unit'],
                'temperature_K': dict(zip(data['node_ids'], data['temperature_K']))}

    def solve(self, graph, *, profile, E_Pa, expansion_alpha_per_K,
              stress_free_temperature_K, constraints, loads=None):
        """Explicit NEW mechanics solve, never a replayed displacement history.

        profile must be 'elementary' or 'network'. Delegation preserves the
        corresponding experimental owner's admission limits and errors. This
        does not certify time/spatial accuracy, stability or electrical lineage.
        """
        if profile not in ('elementary', 'network'):
            raise ValueError("profile must explicitly be 'elementary' or 'network'")
        kwargs = self.mechanical_kwargs(graph)
        from . import _thermoelastic
        owner = (_thermoelastic.solve_thermoelastic_snapshot if profile == 'elementary'
                 else _thermoelastic.solve_thermoelastic_network_snapshot)
        result = owner(graph, **kwargs, E_Pa=E_Pa,
            expansion_alpha_per_K=expansion_alpha_per_K,
            stress_free_temperature_K=stress_free_temperature_K,
            constraints=constraints, loads=loads)
        return {'schema': 'fibernet.saved_temperature_mechanics/1',
                'execution': 'new_mechanical_solve_from_saved_temperature',
                'temperature_source': self.to_dict(), 'mechanical': result.to_dict(),
                'feedback_to_electrical_or_thermal': False}


def prepare_saved_temperature(graph, saved, *, frame_index, length_unit,
                              allow_partial=False):
    """Bind one exact accepted frame of TransientThermalResult.to_dict().

    Accept a thermal object or a bundle containing 'thermal'. No interpolation,
    projection, subset selection, unit conversion or material inference occurs.
    Node and physical-edge order may differ; IDs, oriented endpoints and original
    positions must match exactly. This validates fields used for transfer, not
    all thermal diagnostics or the origin/authenticity of a saved result.
    Accepted-prefix results require allow_partial=True and remain partial.
    """
    if type(allow_partial) is not bool:
        raise ValueError('allow_partial must be bool')
    if length_unit not in ('m', 'mm', 'um'):
        raise ValueError('mechanical length_unit must explicitly be m, mm or um')
    signature = _graph_signature(graph)
    if not isinstance(saved, dict):
        raise ValueError('saved result must be a JSON object')
    data = saved.get('thermal', saved)
    if not isinstance(data, dict):
        raise ValueError('thermal member must be a JSON object')
    encoded = _json(data)
    provenance = data.get('provenance')
    if (not isinstance(provenance, dict) or provenance.get('model') != 'constant_lumped_thermal'
            or provenance.get('length_unit') != length_unit
            or not isinstance(provenance.get('units'), dict)
            or provenance['units'].get('T') != 'K'):
        raise ValueError('supported thermal provenance, matching length_unit and Kelvin units required')
    completed = data.get('completed')
    if type(completed) is not bool or (not completed and not allow_partial):
        raise ValueError('incomplete accepted prefix requires explicit allow_partial=True')
    input_hash = data.get('input_sha256')
    if (not isinstance(input_hash, str) or len(input_hash) != 64
            or any(c not in '0123456789abcdef' for c in input_hash)):
        raise ValueError('thermal input_sha256 must be a lowercase SHA-256 digest')
    try:
        ids, eids = _ids(data['node_ids'], 'node_ids'), _ids(data['edge_ids'], 'edge_ids')
        if set(ids) != set(graph.nodes) or set(eids) != set(graph.edges):
            raise ValueError('saved result and graph require exact stable-ID coverage')
        endpoints, positions = data['endpoints_by_id'], data['positions']
        if not isinstance(endpoints, list) or len(endpoints) != len(eids):
            raise ValueError('endpoints_by_id shape mismatch')
        for eid, pair in zip(eids, endpoints):
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError('endpoints_by_id shape mismatch')
            pair = [_id(value) for value in pair]
            edge = graph.edges[eid]
            if pair != [edge.node_i, edge.node_j] or pair[0] == pair[1]:
                raise ValueError('physical edge endpoints or orientation differ')
        if not isinstance(positions, list) or len(positions) != len(ids):
            raise ValueError('positions shape mismatch')
        for nid, xyz in zip(ids, positions):
            if not isinstance(xyz, list) or len(xyz) != 3:
                raise ValueError('positions shape mismatch')
            xyz = [_number(value, 'position') for value in xyz]
            if xyz != list(graph.nodes[nid].position):
                raise ValueError('saved and graph node positions differ')
        times = data['times_s']
        if not isinstance(times, list) or not times:
            raise ValueError('times_s requires a nonempty list')
        times = [_number(value, 'time_s') for value in times]
        if times[0] != 0 or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError('times_s must start at zero and strictly increase')
        if type(frame_index) is not int or not 0 <= frame_index < len(times):
            raise ValueError('frame_index must select one existing nonnegative integer row')
        temperatures = data['temperature_K']
        if not isinstance(temperatures, list) or len(temperatures) != len(times):
            raise ValueError('temperature history shape mismatch')
        selected = None
        for index, row in enumerate(temperatures):
            if not isinstance(row, list) or len(row) != len(ids):
                raise ValueError('temperature history shape mismatch')
            values = [_number(value, 'temperature_K', positive=True) for value in row]
            if index == frame_index:
                selected = values
    except KeyError as exc:
        raise ValueError('missing saved thermal transfer field: ' + str(exc)) from exc
    record = {'schema': 'fibernet.saved_temperature/1', 'node_ids': ids,
        'edge_ids': eids, 'endpoints_by_id': endpoints, 'temperature_K': selected,
        'frame_index': frame_index, 'time_s': times[frame_index], 'length_unit': length_unit,
        'temperature_unit': 'K', 'source_completed': completed,
        'source_frame_count': len(times), 'source_thermal_input_sha256': input_hash,
        'source_thermal_json_sha256': hashlib.sha256(encoded.encode()).hexdigest(),
        'graph_sha256': signature, 'interpolation': 'none',
        'validation_scope': 'field_binding_only_not_thermal_or_electrical_qualification'}
    return SavedTemperatureSnapshot(_json(record))
