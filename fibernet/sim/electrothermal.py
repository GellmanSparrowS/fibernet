"""Experimental fixed-geometry, voltage-driven two-way lumped electrothermal model.

No material inference, public root export, or import-time execution. Convergence
means finite-tolerance joint consistency, never an exact nonlinear solution.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .linear_transport import solve_linear_transport
from .joule_heating import assemble_joule_heating, _number, _id, _product
from .transient_thermal import _mapping, _solve_thermal_interval
from ._thermal_step import thermal_fields, evaluate_thermal_step, ThermalResidualRejection

_MAX_NODES = 4096
_MAX_EDGES = 16384
_MAX_INTERVALS = 2048
_MAX_ITERATIONS = 25
_MAX_RETAINED_SCALARS = 2_000_000


def _exact_map(value, label, nonnegative=True):
    if not isinstance(value, Mapping):
        raise ValueError(label + ' requires an exact stable-ID mapping')
    return {_id(k, label): _number(v, label, nonnegative=nonnegative)
            for k, v in value.items()}


@dataclass(frozen=True)
class TemperatureConductanceLaw:
    G0_S: Mapping
    alpha_per_K: Mapping
    reference_temperature_K: Mapping
    endpoint_temperature_weights: Mapping
    temperature_domain_K: tuple

    def __post_init__(self):
        for name in ('G0_S', 'alpha_per_K', 'reference_temperature_K'):
            object.__setattr__(self, name, MappingProxyType(_exact_map(getattr(self, name), name)))
        raw = self.temperature_domain_K
        if not isinstance(raw, (tuple, list)) or len(raw) != 2:
            raise ValueError('temperature_domain_K requires two finite Kelvin endpoints')
        lo, hi = (_number(v, 'temperature domain', nonnegative=True) for v in raw)
        if lo > hi:
            raise ValueError('temperature domain is reversed')
        object.__setattr__(self, 'temperature_domain_K', (lo, hi))
        ids = set(self.G0_S)
        if set(self.alpha_per_K) != ids or set(self.reference_temperature_K) != ids:
            raise ValueError('law maps require identical edge IDs')
        if not isinstance(self.endpoint_temperature_weights, Mapping):
            raise ValueError('explicit endpoint temperature weights required')
        weights = {_id(e, 'edge ID'): MappingProxyType(_exact_map(w, 'temperature weight'))
                   for e, w in self.endpoint_temperature_weights.items()}
        if set(weights) != ids:
            raise ValueError('temperature weights require exact edge coverage')
        for e in ids:
            if not lo <= self.reference_temperature_K[e] <= hi:
                raise ValueError('reference temperature lies outside domain')
            for bound in (lo, hi):
                self._factor(e, bound)
            if abs(math.fsum(weights[e].values()) - 1.) > 8*np.finfo(float).eps:
                raise ValueError('endpoint temperature weights must sum to one; no normalization')
        object.__setattr__(self, 'endpoint_temperature_weights', MappingProxyType(weights))

    def _factor(self, eid, temperature):
        term = _product(self.alpha_per_K[eid], temperature-self.reference_temperature_K[eid], 'temperature-law factor')
        factor = 1. + term
        if not math.isfinite(term) or not math.isfinite(factor) or factor <= 0:
            raise ValueError('temperature law denominator must be finite and positive')
        return factor

    def binding(self, graph):
        if set(graph.edges) != set(self.G0_S):
            raise ValueError('law needs exact graph edge coverage')
        for eid, edge in graph.edges.items():
            if set(self.endpoint_temperature_weights[eid]) != {edge.node_i, edge.node_j}:
                raise ValueError('temperature weights must identify both actual endpoints')

    def evaluate(self, graph, temperatures):
        lo, hi = self.temperature_domain_K
        if any(not math.isfinite(v) or not lo <= v <= hi for v in temperatures.values()):
            raise ValueError('node temperature lies outside declared law domain')
        values = {}
        for eid, edge in graph.edges.items():
            weights = self.endpoint_temperature_weights[eid]
            temperature = math.fsum(_product(weights[n], temperatures[n], 'edge temperature interpolation') for n in (edge.node_i, edge.node_j))
            if not math.isfinite(temperature) or not lo <= temperature <= hi:
                raise ValueError('edge temperature lies outside declared law domain')
            factor = self._factor(eid, temperature)
            value = self.G0_S[eid]/factor if self.G0_S[eid] else 0.
            if not math.isfinite(value) or (self.G0_S[eid] > 0 and value <= 0):
                raise ArithmeticError('active electrical conductance is not representable')
            values[eid] = value
        return values

    def to_dict(self):
        return {'G0_S': dict(self.G0_S), 'alpha_per_K': dict(self.alpha_per_K),
                'reference_temperature_K': dict(self.reference_temperature_K),
                'endpoint_temperature_weights': {k: dict(v) for k, v in self.endpoint_temperature_weights.items()},
                'temperature_domain_K': self.temperature_domain_K}


@dataclass(frozen=True)
class ElectrothermalFrame:
    time_s: float
    temperature_K: tuple
    conductance_S: tuple
    edge_heat_rate_W: tuple
    reservoir_injection_W: tuple
    node_stored_heat_change_J: tuple
    interval_stored_heat_J: tuple
    cumulative_source_heat_J: tuple
    cumulative_reservoir_heat_J: tuple
    free_node_residual_J: tuple
    node_acceptance_bound_J: tuple
    component_energy_residual_J: tuple
    component_acceptance_bound_J: tuple
    electrical_json: str
    heat_load_json: str
    accepted_iterations: int
    interval_diagnostics_available: bool


@dataclass(frozen=True)
class ElectrothermalResult:
    node_ids: tuple
    edge_ids: tuple
    endpoints_by_id: tuple
    thermal_component_labels: tuple
    capacity_J_per_K: tuple
    conductance_W_per_K: tuple
    fixed_node_ids: tuple
    fixed_temperature_K: tuple
    frames: tuple
    iteration_history: tuple
    completed: bool
    input_sha256: str
    provenance_json: str


class ElectrothermalSolveError(RuntimeError):
    """Numerical failure with accepted-only frames; trials are never accepted."""
    def __init__(self, reason, partial_result, attempted_step, attempted_time_s):
        super().__init__(reason)
        self.partial_result = partial_result
        self.attempted_step = attempted_step
        self.attempted_time_s = attempted_time_s


def solve_electrothermal(graph, law, prescribed_voltage_V, capacity_J_per_K,
                         conductance_W_per_K, initial_temperature_K, times_s, *,
                         endpoint_heat_weights, length_unit, external_source_W=None,
                         fixed_temperature_K=(), max_iterations=25, model_metadata=None):
    """Bounded Picard backward Euler, with recomputed same-state acceptance.

    max_iterations is an integer in [1,25]. Input validation raises ValueError;
    failure after an accepted initial state raises ElectrothermalSolveError.
    Existing electrical and Joule provenance/roundoff validation remain active.
    """
    if not isinstance(law, TemperatureConductanceLaw):
        raise TypeError('law must be TemperatureConductanceLaw')
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or not 1 <= max_iterations <= _MAX_ITERATIONS:
        raise ValueError('max_iterations must be an integer in [1,25]')
    ids = tuple(sorted(_id(k, 'node ID') for k in graph.nodes))
    eids = tuple(_id(k, 'edge ID') for k in graph.edges)
    if not ids or len(ids) > _MAX_NODES or len(eids) > _MAX_EDGES:
        raise ValueError('graph exceeds private prototype resource bounds')
    rows = {k: i for i, k in enumerate(ids)}
    ends = tuple((graph.edges[k].node_i, graph.edges[k].node_j) for k in eids)
    if any(a == b or a not in rows or b not in rows for a, b in ends):
        raise ValueError('edges need distinct existing endpoints')
    conn = np.array([(rows[a], rows[b]) for a, b in ends], dtype=int).reshape(-1, 2)
    C = _mapping(capacity_J_per_K, ids, 'capacity', True)
    g = _mapping(conductance_W_per_K, eids, 'thermal conductance')
    initial = _mapping(initial_temperature_K, ids, 'initial temperature')
    # Validate temporal shape before allocating output histories.
    raw_times = list(times_s)
    if any(isinstance(v, (bool, np.bool_)) for v in raw_times):
        raise ValueError('times must be real, not bool')
    times = np.array([_number(v, 'time', nonnegative=True) for v in raw_times])
    if len(times) < 2 or len(times)-1 > _MAX_INTERVALS or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError('bounded increasing time grid starting at zero required')
    dts = np.diff(times)
    if np.any(abs(dts-dts[0]) > 32*np.finfo(float).eps*max(times[-1], dts[0])):
        raise ValueError('uniform time grid required')
    if len(times)*(80*len(ids)+80*len(eids)) > _MAX_RETAINED_SCALARS:
        raise ValueError('requested retained history exceeds scalar budget')
    fixed = {}
    pairs = fixed_temperature_K.items() if isinstance(fixed_temperature_K, Mapping) else fixed_temperature_K
    for key, value in pairs:
        key, value = _id(key, 'fixed node'), _number(value, 'fixed temperature', nonnegative=True)
        if key not in rows or (key in fixed and fixed[key] != value):
            raise ValueError('unknown or conflicting thermal fixed node')
        fixed[key] = value
    fix = np.array(sorted(rows[k] for k in fixed), dtype=int)
    free = np.array([i for i, k in enumerate(ids) if k not in fixed], dtype=int)
    if any(initial[rows[k]] != value for k, value in fixed.items()):
        raise ValueError('initial fixed temperature must match exactly')
    if model_metadata is not None and not isinstance(model_metadata, dict):
        raise ValueError('metadata must be a finite JSON dictionary')
    metadata_text = json.dumps(model_metadata or {}, allow_nan=False)
    if len(metadata_text.encode()) > 16384:
        raise ValueError('metadata exceeds 16 KiB')
    metadata = json.loads(metadata_text)
    # Canonicalize boundary pairs once, retaining duplicate conflicts for the existing solver.
    voltage = list(prescribed_voltage_V.items() if isinstance(prescribed_voltage_V, Mapping) else prescribed_voltage_V)
    law.binding(graph)
    temperature_map = lambda t: dict(zip(ids, map(float, t)))
    initial_G = law.evaluate(graph, temperature_map(initial))
    electrical = solve_linear_transport(graph, initial_G, voltage, length_unit=length_unit)
    load = assemble_joule_heating(graph, electrical, endpoint_weights=endpoint_heat_weights,
                                 external_source_W=external_source_W, length_unit=length_unit)
    # Thermal components are intentionally NOT the electrical components.
    active = conn[g > 0]
    adj = coo_matrix((np.ones(len(active)), (active[:, 0], active[:, 1])), shape=(len(ids), len(ids)))
    count, labels = connected_components(adj, directed=False)
    members = [np.flatnonzero(labels == i) for i in range(count)]
    incident = [[] for _ in ids]
    component_edges = [[] for _ in range(count)]
    for j, (a, b) in enumerate(conn):
        incident[a].append((j, a, b)); incident[b].append((j, a, b))
        component_edges[int(labels[a])].append((j, a, b))
    Q = np.array([load.total_source_W[k] for k in ids])
    q, outward, reservoir = thermal_fields(initial, conn=conn, g=g, incident=incident, fix=fix, Q=Q)
    payload = {'electrical_input_sha256': electrical.provenance['input_sha256'],
               'law': law.to_dict(), 'voltage': voltage,
               'capacity': list(C), 'thermal_conductance': list(g), 'initial': list(initial),
               'times': list(times), 'heat_weights': endpoint_heat_weights, 'external': load.external_source_W,
               'fixed': fixed, 'length_unit': length_unit, 'max_iterations': max_iterations, 'metadata': metadata}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    provenance = json.dumps({'model': 'fixed_geometry_temperature_dependent_resistive_lumped_thermal',
        'method': 'backward_euler_bounded_picard', 'validation_contract': 'transport-roundoff-2',
        'time_accuracy_verified': False, 'geometry_inference': False, 'length_unit': length_unit,
        'units': {'temperature': 'K', 'electrical_conductance': 'S', 'thermal_conductance': 'W/K',
                  'capacity': 'J/K', 'power': 'W', 'energy': 'J'},
        'nonlinear_increment_test': 'each node: abs(candidate-trial) <= 8*ulp(candidate)+1e-12*abs(candidate-previous)',
        'metadata': metadata, 'law': law.to_dict(),
        'law_sha256': hashlib.sha256(json.dumps(law.to_dict(), sort_keys=True, allow_nan=False).encode()).hexdigest(),
        'electrical_components': 'per-frame heat_load_json component_ledger; independent of thermal_component_labels',
        'cumulative_heat_quadrature': 'sum accepted dt times each final right-endpoint source/reservoir; not final power times elapsed time'
        }, sort_keys=True, allow_nan=False)
    frames, history = [], []
    retained_scalars = 0
    retained_bytes = 0

    def scalar_count(value):
        if isinstance(value, dict):
            return sum(scalar_count(v) + 1 for v in value.values())
        if isinstance(value, (list, tuple)):
            return sum(scalar_count(v) for v in value)
        return 1

    def append_frame(value):
        nonlocal retained_scalars, retained_bytes
        # JSON snapshots carry nested allocation/electrical fields, not just T.
        scalar_increment = sum(len(getattr(value, key)) for key in (
            'temperature_K', 'conductance_S', 'edge_heat_rate_W', 'reservoir_injection_W',
            'node_stored_heat_change_J', 'interval_stored_heat_J', 'cumulative_source_heat_J',
            'cumulative_reservoir_heat_J', 'free_node_residual_J', 'node_acceptance_bound_J',
            'component_energy_residual_J', 'component_acceptance_bound_J')) + 3
        scalar_increment += scalar_count(json.loads(value.electrical_json))
        scalar_increment += scalar_count(json.loads(value.heat_load_json))
        byte_increment = len(value.electrical_json.encode()) + len(value.heat_load_json.encode())
        if retained_scalars + scalar_increment + len(history)*6 > _MAX_RETAINED_SCALARS:
            raise ArithmeticError('retained scalar history cap exceeded')
        if retained_bytes + byte_increment > 32*1024*1024:
            raise ArithmeticError('retained serialized snapshots exceed 32 MiB')
        retained_scalars += scalar_increment
        retained_bytes += byte_increment
        frames.append(value)

    def frame(temp, conductance, electric, heat, fields, time, iterations, diagnostic):
        def values(key):
            return tuple(float(x) for x in fields[key])
        dt = float(time)-frames[-1].time_s if frames else 0.
        previous_source = frames[-1].cumulative_source_heat_J if frames else (0.,)*len(ids)
        previous_reservoir = frames[-1].cumulative_reservoir_heat_J if frames else (0.,)*len(ids)
        source_heat = tuple(math.fsum([previous_source[i], _product(dt, heat.total_source_W[n], 'interval source heat')])
                            for i, n in enumerate(ids))
        reservoir_heat = tuple(math.fsum([previous_reservoir[i], _product(dt, float(fields['reservoir'][i]), 'interval reservoir heat')])
                               for i in range(len(ids)))
        if not all(math.isfinite(v) for v in source_heat + reservoir_heat):
            raise ArithmeticError('cumulative heat ledger overflow')
        return ElectrothermalFrame(float(time), tuple(map(float, temp)), tuple(conductance[k] for k in eids),
            values('q'), values('reservoir'), values('stored'), values('interval_stored'), source_heat, reservoir_heat,
            values('residual'), values('nbound'),
            values('energy'), values('cbound'), json.dumps(electric.to_dict(), sort_keys=True, allow_nan=False),
            json.dumps(heat.to_dict(), sort_keys=True, allow_nan=False), iterations, diagnostic)

    def result(completed):
        return ElectrothermalResult(ids, eids, ends, tuple(map(int, labels)), tuple(map(float, C)),
                                    tuple(map(float, g)), tuple(ids[i] for i in fix),
                                    tuple(float(initial[i]) for i in fix), tuple(frames),
                                    tuple(history), completed, digest, provenance)

    empty = {'q': q, 'reservoir': reservoir, 'stored': np.zeros(len(ids)),
             'interval_stored': np.zeros(len(ids)), 'residual': (), 'nbound': (), 'energy': (), 'cbound': ()}
    append_frame(frame(initial, initial_G, electrical, load, empty, 0, 0, False))
    previous = initial.copy()
    for step, dt in enumerate(dts, 1):
        trial = previous.copy()
        try:
            for iteration in range(1, max_iterations+1):
                G = law.evaluate(graph, temperature_map(trial))
                electric = solve_linear_transport(graph, G, voltage, length_unit=length_unit)
                heat = assemble_joule_heating(graph, electric, endpoint_weights=endpoint_heat_weights,
                    external_source_W=external_source_W, length_unit=length_unit)
                subsolve, inc = _solve_thermal_interval(graph, capacity_J_per_K, conductance_W_per_K,
                    temperature_map(previous), float(dt), source_W=heat.total_source_W,
                    fixed_temperature_K=fixed, length_unit=length_unit)
                candidate = subsolve.temperature_K[-1].copy()
                final_G = law.evaluate(graph, temperature_map(candidate))
                final_electric = solve_linear_transport(graph, final_G, voltage, length_unit=length_unit)
                final_heat = assemble_joule_heating(graph, final_electric, endpoint_weights=endpoint_heat_weights,
                    external_source_W=external_source_W, length_unit=length_unit)
                final_Q = np.array([final_heat.total_source_W[k] for k in ids])
                difference = abs(candidate-trial)
                tolerance = np.array([8*math.ulp(float(t)) for t in candidate]) + 1e-12*abs(candidate-previous)
                small = bool(np.all(difference <= tolerance))
                diagnostics = None
                residual_reason = None
                try:
                    diagnostics = evaluate_thermal_step(T=previous, new=candidate, inc=inc, initial=initial,
                        dt=float(dt), Q=final_Q, C=C, g=g, conn=conn, fix=fix, free=free,
                        incident=incident, members=members, component_edges=component_edges)
                except ThermalResidualRejection as exc:
                    # Precision/nonfinite failures propagate; only this typed finite residual retries.
                    residual_reason = str(exc)
                if retained_scalars + (len(history)+1)*6 > _MAX_RETAINED_SCALARS:
                    raise ArithmeticError('iteration scalar history cap exceeded')
                history.append((step, iteration, float(max(difference, default=0.)),
                                small, diagnostics is not None, residual_reason))
                if small and diagnostics is not None:
                    append_frame(frame(candidate, final_G, final_electric, final_heat, diagnostics,
                                        times[step], iteration, True))
                    previous = candidate
                    break
                trial = candidate
            else:
                raise ArithmeticError('coupled Picard iteration limit reached')
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            raise ElectrothermalSolveError(str(exc), result(False), step, float(times[step])) from exc
    return result(True)
