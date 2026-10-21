"""Explicit, conservative one-way Joule-load assembly; no numerical solver.

Electrical conductance is supplied in S. This adapter returns heat sources in
W for the separate constant-lumped thermal API (whose conductances are W/K and
capacities J/K). Endpoint fractions are a declared lumping model, not material
calibration. Thermal boundaries are independent and are never inferred here.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import numbers
import sys

from ._roundoff import upper_sum, upper_product, evaluation_ulp
from ._roundoff import _require_gradual_for_subnormal

_EPS = sys.float_info.epsilon
_WEIGHT_SUM_ATOL = 8 * _EPS
_UNITS = {'conductance': 'S', 'voltage': 'V', 'current': 'A', 'power': 'W'}


@dataclass(frozen=True)
class JouleHeatingRun:
    """One explicit load and the existing thermal solver's result."""
    heat_load: JouleHeatingLoad
    thermal: object


@dataclass(frozen=True)
class JouleHeatingLoad:
    """Tuple-backed detached load. Source-map properties return fresh copies."""
    node_ids: tuple
    edge_ids: tuple
    endpoints_by_id: tuple
    edge_power_W: tuple
    endpoint_weights: tuple
    endpoint_heat_W: tuple
    joule_node_power_W: tuple
    external_node_power_W: tuple
    total_node_power_W: tuple
    edge_weight_sum_error: tuple
    edge_partition_residual_W: tuple
    edge_partition_bound_W: tuple
    node_aggregation_residual_W: tuple
    node_aggregation_bound_W: tuple
    node_source_addition_residual_W: tuple
    node_source_addition_bound_W: tuple
    total_joule_power_W: float
    electrical_boundary_power_W: float
    global_partition_residual_W: float
    global_partition_bound_W: float
    total_source_balance_residual_W: float
    total_source_balance_bound_W: float
    component_ledger_json: str
    provenance_json: str
    input_sha256: str

    @property
    def joule_source_W(self):
        return dict(zip(self.node_ids, self.joule_node_power_W))

    @property
    def external_source_W(self):
        return dict(zip(self.node_ids, self.external_node_power_W))

    @property
    def total_source_W(self):
        return dict(zip(self.node_ids, self.total_node_power_W))

    def to_dict(self):
        """Return a detached strict-JSON-ready ledger with explicit units."""
        value = asdict(self)
        value['schema'] = 'fibernet-joule-heating-load-1'
        value['component_ledger'] = json.loads(value.pop('component_ledger_json'))
        value['provenance'] = json.loads(value.pop('provenance_json'))
        value['total_source_W'] = {str(k): v for k, v in self.total_source_W.items()}
        # The JSON round trip deliberately detaches nested records and turns
        # tuple row arrays into lists without exposing internal mutable state.
        return json.loads(json.dumps(value, allow_nan=False))


def _id(value, label):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or not 0 <= value <= 2**63-1:
        raise ValueError(label + ' must be a nonnegative int64 ID')
    return int(value)


def _number(value, label, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(label + ' must be a finite real number, not bool')
    if nonnegative and value < 0:
        raise ValueError(label + ' must be nonnegative')
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(label + ' is not representable') from exc
    if not math.isfinite(result):
        raise ValueError(label + ' must be finite')
    if value != 0 and result == 0:
        raise ValueError(label + ' nonzero value underflows binary64')
    _require_gradual_for_subnormal(result)
    return result


def _sum(values):
    result = math.fsum(values)
    if not math.isfinite(result):
        raise ArithmeticError('power/current accumulation overflowed')
    _require_gradual_for_subnormal(result)
    return result


def _sum_evaluation_bound(values):
    """Conservative comparison with ordinary finite binary64 accumulation."""
    values = tuple(values)
    ku = upper_product(len(values)+2, _EPS)
    if ku >= .5:
        raise ArithmeticError('accumulation roundoff budget is not supported')
    gamma = math.nextafter(ku / math.nextafter(1-ku, -math.inf), math.inf)
    return upper_sum([upper_product(gamma, upper_sum(abs(v) for v in values)),
                      evaluation_ulp(_sum(values))])


def _product(a, b, label):
    value = a*b
    if not math.isfinite(value) or (a != 0 and b != 0 and value == 0):
        raise ArithmeticError(label + ' is not representable')
    _require_gradual_for_subnormal(value)
    return value


def _mapping(values, ids, label):
    if not isinstance(values, Mapping):
        raise ValueError(label + ' must be an exact stable-ID mapping')
    result = {_id(k, label+' key'): _number(v, label, nonnegative=True) for k, v in values.items()}
    if set(result) != set(ids):
        raise ValueError(label + ' requires exact stable-ID coverage')
    return result


def _electrical_binding(graph, data, length_unit):
    if not isinstance(data, dict) or data.get('schema') != 'fibernet-linear-conductance-1' or data.get('status') != 'converged':
        raise ValueError('a converged supported electrical result is required')
    provenance = data.get('provenance', {})
    if (not isinstance(provenance, dict) or provenance.get('validation_contract') != 'transport-roundoff-2' or
            provenance.get('model_kind') != 'steady-linear-electrical-conductance' or
            provenance.get('geometry_conversion') != 'none' or provenance.get('units') != _UNITS):
        raise ValueError('unsupported electrical provenance contract or units')
    if length_unit not in ('m', 'cm', 'mm', 'um', 'nm') or provenance.get('length_unit') != length_unit:
        raise ValueError('length_unit must exactly match electrical geometry provenance')
    ids = tuple(_id(i, 'node ID') for i in data['node_ids'])
    eids = tuple(_id(i, 'edge ID') for i in data['edge_ids'])
    graph_ids = sorted(_id(i, 'graph node ID') for i in graph.nodes)
    graph_eids = [_id(i, 'graph edge ID') for i in graph.edges]
    if not ids or list(ids) != graph_ids or len(set(eids)) != len(eids) or set(eids) != set(graph_eids):
        raise ValueError('electrical and target graph stable identities differ')
    rows = {n: i for i, n in enumerate(ids)}
    positions = []
    for n in ids:
        xyz = tuple(_number(v, 'coordinate') for v in graph.nodes[n].position)
        if len(xyz) != 3:
            raise ValueError('geometry positions must have exactly three coordinates')
        positions.append(xyz)
    endpoints = []
    if len(data['endpoints_by_id']) != len(eids):
        raise ValueError('electrical endpoint rows do not match edge IDs')
    for eid, pair in zip(eids, data['endpoints_by_id']):
        edge = graph.edges[eid]
        a, b = _id(edge.node_i, 'endpoint ID'), _id(edge.node_j, 'endpoint ID')
        if len(pair) != 2 or tuple(_id(v, 'result endpoint ID') for v in pair) != (a, b):
            raise ValueError('electrical result endpoint rows differ')
        if a not in rows or b not in rows or a == b:
            raise ValueError('electrical endpoint identity or orientation differs')
        endpoints.append((a, b))
    def vector(name, count, nonnegative=False):
        values = tuple(_number(v, name, nonnegative=nonnegative) for v in data[name])
        if len(values) != count:
            raise ValueError(name + ' row count mismatch')
        return values
    conductance = vector('conductance_S', len(eids), True)
    voltage = vector('voltage_V', len(ids))
    current = vector('current_A', len(eids))
    power = vector('edge_power_W', len(eids), True)
    supplied_injection = vector('node_injection_A', len(ids))
    bc = {}
    for key, raw in data['prescribed_voltage_V'].items():
        if not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key:
            raise ValueError('electrical boundary keys must be canonical serialized IDs')
        n = _id(int(key), 'boundary node ID')
        if n not in rows:
            raise ValueError('unknown electrical boundary node')
        bc[n] = _number(raw, 'prescribed voltage')
        if voltage[rows[n]] != bc[n]:
            raise ValueError('returned voltage does not satisfy its prescribed boundary')
    binding = dict(node_ids=ids, edge_ids=eids, endpoints=endpoints,
                   conductance_S=conductance, prescribed_voltage_V=sorted(bc.items()),
                   coordinates=positions, length_unit=length_unit)
    fingerprint = hashlib.sha256(json.dumps(binding, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if provenance.get('input_sha256') != fingerprint:
        raise ValueError('electrical input fingerprint does not bind this geometry/conductance/boundary state')
    # Input binding is not an output authenticity signature. Validate the actual
    # fields needed for heat transfer as well as their recorded balance limits.
    incident = {n: [] for n in ids}
    adjacency = {n: set() for n in ids}
    for j, (a, b) in enumerate(endpoints):
        difference = voltage[rows[a]] - voltage[rows[b]]
        if not math.isfinite(difference):
            raise ValueError('electrical voltage difference overflowed')
        expected_current = _product(conductance[j], difference, 'electrical current') if conductance[j] else 0.
        expected_power = _product(expected_current, difference, 'electrical power') if conductance[j] else 0.
        if current[j] != expected_current or power[j] != expected_power:
            raise ValueError('electrical current or edge power disagrees with g and returned voltage')
        incident[a].append(current[j]); incident[b].append(-current[j])
        if conductance[j] > 0:
            adjacency[a].add(b); adjacency[b].add(a)
    injection = {n: _sum(incident[n]) for n in ids}
    evaluation = {n: _sum_evaluation_bound(incident[n]) for n in ids}
    limits = provenance.get('achieved_limits', {}).get('node_current_A', ())
    if len(limits) != len(ids):
        raise ValueError('electrical node-current acceptance limits unavailable')
    for i, n in enumerate(ids):
        if abs(supplied_injection[i]-injection[n]) > upper_sum([evaluation[n], evaluation_ulp(supplied_injection[i])]):
            raise ValueError('electrical node injection is inconsistent with returned edge currents')
        limit = _number(limits[i], 'electrical node-current limit', nonnegative=True)
        if n not in bc and abs(injection[n]) > upper_sum([limit, evaluation[n]]):
            raise ValueError('electrical free-node balance exceeds its acceptance limit')
    free = [n for n in ids if n not in bc]
    reported_free = _number(data['free_node_residual_A'], 'electrical free-node residual', nonnegative=True)
    actual_free = max((abs(injection[n]) for n in free), default=0.)
    if abs(reported_free-actual_free) > upper_sum([max((evaluation[n] for n in free), default=0.), evaluation_ulp(reported_free)]):
        raise ValueError('electrical free-node residual field is inconsistent')
    reported_balance = _number(data['global_current_balance_A'], 'electrical current balance')
    balance_bound = upper_sum([upper_sum(evaluation.values()), _sum_evaluation_bound(supplied_injection), evaluation_ulp(reported_balance)])
    if abs(reported_balance-_sum(injection.values())) > balance_bound:
        raise ValueError('electrical global-current field is inconsistent')
    controls = provenance.get('tolerances', {})
    rtol = _number(controls['rtol'], 'electrical rtol', nonnegative=True)
    power_atol = _number(controls['power_atol_W'], 'electrical power_atol_W', nonnegative=True)
    components = []
    unseen = set(ids)
    while unseen:
        todo = [min(unseen)]; members = set()
        while todo:
            n = todo.pop()
            if n not in members:
                members.add(n); todo.extend(adjacency[n]-members)
        unseen -= members
        fixed = sorted(members.intersection(bc))
        if not fixed:
            raise ValueError('electrical conducting component lacks its voltage anchor')
        edge_rows = [j for j, (a, b) in enumerate(endpoints) if conductance[j] > 0 and a in members]
        component_power = _sum(power[j] for j in edge_rows)
        reference = voltage[rows[fixed[0]]]
        terms = [_product(voltage[rows[n]]-reference, injection[n], 'electrical boundary power') for n in fixed]
        boundary_power = _sum(terms)
        bound = upper_sum([power_atol, upper_product(rtol, max(abs(component_power), abs(boundary_power))),
                           _sum_evaluation_bound(terms), _sum_evaluation_bound(power[j] for j in edge_rows)])
        if abs(component_power-boundary_power) > bound:
            raise ValueError('electrical component boundary supply does not balance dissipation')
        components.append((tuple(sorted(members)), tuple(edge_rows), boundary_power))
    total_power = _sum(power)
    supplied_power = _number(data['dissipated_power_W'], 'electrical dissipated power', nonnegative=True)
    boundary_power = _number(data['boundary_power_W'], 'electrical boundary power')
    if abs(supplied_power-total_power) > upper_sum([_sum_evaluation_bound(power), evaluation_ulp(supplied_power)]):
        raise ValueError('electrical dissipated-power total is inconsistent')
    boundary_terms = [item[2] for item in components]
    if abs(boundary_power-_sum(boundary_terms)) > upper_sum([power_atol, upper_product(rtol, abs(total_power)),
                                                           _sum_evaluation_bound(boundary_terms), evaluation_ulp(boundary_power)]):
        raise ValueError('electrical boundary-power total is inconsistent')
    return ids, eids, tuple(endpoints), power, components, fingerprint, boundary_power, controls


def _assemble_payload(graph, data, *, endpoint_weights, external_source_W=None, length_unit):
    """Pure scalar implementation; public callers pass LinearTransportResult."""
    ids, eids, endpoints, power, components, fingerprint, boundary_power, controls = _electrical_binding(graph, data, length_unit)
    if not isinstance(endpoint_weights, Mapping):
        raise ValueError('endpoint_weights must map every stable edge ID')
    weight_inputs = {_id(k, 'weight edge ID'): v for k, v in endpoint_weights.items()}
    if set(weight_inputs) != set(eids):
        raise ValueError('endpoint_weights requires exact edge-ID coverage')
    external = {n: 0. for n in ids} if external_source_W is None else _mapping(external_source_W, ids, 'external_source_W')
    weights = []; allocated = []; weight_errors = []; edge_residuals = []; edge_bounds = []
    contributions = {n: [] for n in ids}
    for j, (eid, (a, b)) in enumerate(zip(eids, endpoints)):
        supplied = _mapping(weight_inputs[eid], (a, b), 'endpoint fractions')
        wi, wj = supplied[a], supplied[b]
        if wi > 1 or wj > 1:
            raise ValueError('each endpoint fraction must lie in [0, 1]')
        defect = _sum((wi, wj, -1.))
        if abs(defect) > _WEIGHT_SUM_ATOL:
            raise ValueError('endpoint fractions must sum to one within fixed 8*eps; no normalization is performed')
        qa, qb = _product(wi, power[j], 'Joule allocation'), _product(wj, power[j], 'Joule allocation')
        residual = _sum((qa, qb, -power[j]))
        bound = upper_sum([upper_product(power[j], upper_sum([abs(defect), evaluation_ulp(defect)])),
                           evaluation_ulp(qa), evaluation_ulp(qb), evaluation_ulp(residual)])
        if abs(residual) > bound:
            raise ArithmeticError('edge-local Joule allocation conservation failed')
        weights.append((wi, wj)); allocated.append((qa, qb)); weight_errors.append(defect)
        edge_residuals.append(residual); edge_bounds.append(bound)
        contributions[a].append((eid, qa)); contributions[b].append((eid, qb))
    joule = {}; totals = {}; node_residuals = []; node_bounds = []; addition_residuals = []; addition_bounds = []
    for n in ids:
        # Sort by identity to make output independent of graph/result row order.
        values = [v for _, v in sorted(contributions[n])]
        joule[n] = _sum(values)
        residual = _sum([joule[n]] + [-v for v in values])
        bound = upper_sum([evaluation_ulp(joule[n]), evaluation_ulp(residual)])
        if abs(residual) > bound:
            raise ArithmeticError('node-local Joule aggregation conservation failed')
        totals[n] = _sum((joule[n], external[n]))
        if joule[n] > 0 and external[n] > 0 and totals[n] in (joule[n], external[n]):
            raise ArithmeticError('positive source contribution is lost when combining Joule and external heating')
        addition = _sum((totals[n], -joule[n], -external[n]))
        addition_bound = upper_sum([evaluation_ulp(totals[n]), evaluation_ulp(addition)])
        if abs(addition) > addition_bound:
            raise ArithmeticError('node-local total-source conservation failed')
        node_residuals.append(residual); node_bounds.append(bound)
        addition_residuals.append(addition); addition_bounds.append(addition_bound)
    rows = {n: i for i, n in enumerate(ids)}
    component_ledgers = []
    for members, edge_rows, supply in components:
        residual = _sum([joule[n] for n in members] + [-power[j] for j in edge_rows])
        bound = upper_sum([upper_sum(edge_bounds[j] for j in edge_rows), upper_sum(node_bounds[rows[n]] for n in members), evaluation_ulp(residual)])
        if abs(residual) > bound:
            raise ArithmeticError('component-local Joule transfer conservation failed')
        component_ledgers.append({'component_kind':'positive-electrical-conductance', 'node_ids': members, 'edge_ids': [eids[j] for j in edge_rows],
            'electrical_power_W': _sum(power[j] for j in edge_rows), 'electrical_boundary_power_W': supply,
            'assigned_node_heat_W': _sum(joule[n] for n in members), 'residual_W': residual, 'bound_W': bound})
    global_residual = _sum(list(joule.values()) + [-p for p in power])
    global_bound = upper_sum([upper_sum(edge_bounds), upper_sum(node_bounds), evaluation_ulp(global_residual)])
    total_residual = _sum(list(totals.values()) + [-v for v in external.values()] + [-p for p in power])
    total_bound = upper_sum([global_bound, upper_sum(addition_bounds), evaluation_ulp(total_residual)])
    if abs(global_residual) > global_bound or abs(total_residual) > total_bound:
        raise ArithmeticError('global Joule/total-source conservation failed')
    provenance = {'schema':'fibernet-joule-heating-provenance-1', 'model':'one_way_constant_joule_load',
        'electrical_input_sha256':fingerprint, 'electrical_validation_contract':'transport-roundoff-2',
        'electrical_tolerances':controls, 'length_unit':length_unit, 'geometry_conversion':'none',
        'units':{'electrical_conductance':'S','edge_power':'W','node_source':'W','endpoint_weights':'1'},
        'weight_sum_atol':_WEIGHT_SUM_ATOL, 'weights_normalized':False,
        'partition_convention':'explicit caller endpoint fractions; not inferred or calibrated',
        'thermal_boundary_inferred':False, 'temperature_feedback':False, 'mechanical_coupling':False,
        'assumptions':['constant steady electrical state over the thermal interval',
            'all supplied resistor dissipation becomes local heat', 'negligible electrical energy storage',
            'thermal reservoirs and external heat sources are supplied independently'],
        'power_bookkeeping':'Joule dissipation is the electrical-to-thermal transfer; do not also add electrical boundary supply or fixed-node allocations a second time as thermal sources',
        'fixed_thermal_nodes':'retain the full source; thermal reservoir injection accounts for source removal',
        'scope':'input consistency and conservative finite-precision transfer; not output authenticity, material calibration, or time-accuracy certification'}
    normalized = {'electrical_input_sha256':fingerprint,'node_ids':ids,'edge_ids':eids,'endpoints_by_id':endpoints,
        'edge_power_W':power,'endpoint_weights':weights,'external_source_W':sorted(external.items()),'length_unit':length_unit}
    load_digest = hashlib.sha256(json.dumps(normalized,sort_keys=True,allow_nan=False).encode()).hexdigest()
    return JouleHeatingLoad(ids,eids,endpoints,power,tuple(weights),tuple(allocated),tuple(joule[n] for n in ids),
        tuple(external[n] for n in ids),tuple(totals[n] for n in ids),tuple(weight_errors),tuple(edge_residuals),
        tuple(edge_bounds),tuple(node_residuals),tuple(node_bounds),tuple(addition_residuals),tuple(addition_bounds),
        _sum(power),boundary_power,global_residual,global_bound,total_residual,total_bound,
        json.dumps(component_ledgers,sort_keys=True,allow_nan=False),json.dumps(provenance,sort_keys=True,allow_nan=False),load_digest)


def assemble_joule_heating(graph, electrical_result, *, endpoint_weights, external_source_W=None, length_unit):
    """Assemble explicit heat-source maps for ``solve_transient_thermal``.

    ``endpoint_weights[eid]`` maps BOTH endpoint stable IDs to finite
    nonnegative fractions whose sum differs from one by at most 8*binary64 eps.
    Every edge, including inactive and parallel edges, needs its own entry.
    Fractions are never normalized or inferred. Equal halves, when supplied,
    mean an explicit equal-endpoint lumping convention, not calibration.

    ``external_source_W`` is omitted for zero or covers all node IDs exactly
    with nonnegative powers. Use ``load.total_source_W`` for the existing
    thermal solver. Keep sources on fixed-temperature nodes: its reservoir
    reaction removes or supplies heat consistently. This function does not
    prescribe thermal boundaries, run either solver, or update conductances.
    """
    from .linear_transport import LinearTransportResult
    if not isinstance(electrical_result, LinearTransportResult):
        raise TypeError('electrical_result must be LinearTransportResult')
    return _assemble_payload(graph, electrical_result.to_dict(), endpoint_weights=endpoint_weights,
                             external_source_W=external_source_W, length_unit=length_unit)


def run_joule_heating(graph, electrical_result, capacity_J_per_K, conductance_W_per_K,
                     initial_temperature_K, times_s, *, endpoint_weights, length_unit,
                     external_source_W=None, fixed_temperature_K=(), model_metadata=None):
    """Run one-way DC heating from an existing electrical result in one call.

    This composes ``assemble_joule_heating`` and the existing
    ``solve_transient_thermal`` exactly once. It does not rerun the electrical
    solver or feed temperatures back into electrical conductances. Electrical
    S, thermal W/K and capacity J/K remain separate explicit inputs.

    Return ``JouleHeatingRun`` with ``heat_load`` and ``thermal``. Thermal input
    validation and ``ThermalSolveError`` propagate unchanged. The returned
    thermal result retains the existing backward-Euler/time-accuracy contract.
    """
    heat_load = assemble_joule_heating(graph, electrical_result,
        endpoint_weights=endpoint_weights, external_source_W=external_source_W,
        length_unit=length_unit)
    from .transient_thermal import solve_transient_thermal
    thermal = solve_transient_thermal(graph, capacity_J_per_K, conductance_W_per_K,
        initial_temperature_K, times_s, source_W=heat_load.total_source_W,
        fixed_temperature_K=fixed_temperature_K, length_unit=length_unit,
        model_metadata=model_metadata)
    return JouleHeatingRun(heat_load, thermal)
