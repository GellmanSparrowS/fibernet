"""Steady scalar electrical transport with explicit edge conductances.

The model is an ideal resistor graph: equipotential junctions, constant
nonnegative conductances, and prescribed node voltages. Geometry and material
properties do not determine conductance. No transient or mechanical coupling.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import numbers
import warnings

import numpy as np
import scipy
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from ._roundoff import upper_sum, upper_product, evaluation_ulp


class FloatingComponentError(ValueError):
    """At least one conducting component has no prescribed voltage."""


class TransportSolveError(RuntimeError):
    """Sparse solve or its independent conservation checks failed."""


def _identifier(value, label):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or
            value < 0 or value > np.iinfo(np.int64).max):
        raise ValueError(f'{label} must be a nonnegative integer ID fitting int64')
    return int(value)


def _number(value, label, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f'{label} must be a real number, not bool')
    value = float(value)
    if not np.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f'{label} must be finite' + (' and nonnegative' if nonnegative else ''))
    return value


def _metadata(value):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError('model_metadata must be a JSON dictionary')
    def keys(item):
        if isinstance(item, dict):
            if any(not isinstance(k, str) for k in item):
                raise ValueError('metadata keys must be strings')
            for v in item.values(): keys(v)
        elif isinstance(item, (list, tuple)):
            for v in item: keys(v)
    keys(value)
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (ValueError, TypeError) as exc:
        raise ValueError('model_metadata must contain finite JSON data') from exc


def _readonly(value, dtype=None):
    a = np.array(value, dtype=dtype, copy=True)
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class LinearTransportResult:
    """Detached electrical fields with explicit stable-ID ordering.

    Current is positive node_i→node_j. Injection is positive supplied into the
    network. Boundary injections are reactions, not free-node residuals.
    """
    node_ids: np.ndarray
    edge_ids: np.ndarray
    endpoints_by_id: np.ndarray
    conductance_S: np.ndarray
    voltage_V: np.ndarray
    current_A: np.ndarray
    node_injection_A: np.ndarray
    edge_power_W: np.ndarray
    prescribed_voltage_V: dict
    free_node_residual_A: float
    global_current_balance_A: float
    dissipated_power_W: float
    boundary_power_W: float
    provenance: dict
    status: str = 'converged'

    def to_dict(self):
        """Return detached JSON-ready fields; no mechanical quantities implied."""
        return {
            'schema': 'fibernet-linear-conductance-1', 'status': self.status,
            'node_ids': self.node_ids.tolist(), 'edge_ids': self.edge_ids.tolist(),
            'endpoints_by_id': self.endpoints_by_id.tolist(),
            'conductance_S': self.conductance_S.tolist(),
            'voltage_V': self.voltage_V.tolist(), 'current_A': self.current_A.tolist(),
            'node_injection_A': self.node_injection_A.tolist(),
            'edge_power_W': self.edge_power_W.tolist(),
            'prescribed_voltage_V': {str(k): v for k, v in self.prescribed_voltage_V.items()},
            'free_node_residual_A': self.free_node_residual_A,
            'global_current_balance_A': self.global_current_balance_A,
            'dissipated_power_W': self.dissipated_power_W,
            'boundary_power_W': self.boundary_power_W,
            'provenance': _metadata(self.provenance),
        }


def solve_linear_transport(graph, conductance_S, dirichlet_voltage_V, *,
                           length_unit, model_metadata=None, rtol=1e-10,
                           current_atol_A=0.0, power_atol_W=0.0):
    """Solve a supplied resistor network using SciPy's sparse direct solver.

    ``conductance_S`` maps every stable edge ID to a finite nonnegative value.
    Zero means inactive, and still has an output row with exactly zero current.
    ``dirichlet_voltage_V`` is an iterable of ``(stable node ID, voltage)``
    pairs (or a mapping). Conflicting duplicate pairs raise; identical pairs
    coalesce. Every positive-conductance component, including singleton nodes,
    must have a prescribed voltage. Floating components raise explicitly.

    Node rows are sorted by ID; edge rows follow graph insertion order. Values
    and geometry are copied, never rescaled. ``length_unit`` declares geometric
    provenance only; conductance remains caller-provided in S. Parallel members
    remain separate rows and contribute additively to the nodal equations.
    """
    if not isinstance(length_unit, str) or length_unit not in {'m', 'cm', 'mm', 'um', 'nm'}:
        raise ValueError('declare geometry length_unit: m, cm, mm, um or nm')
    rtol = _number(rtol, 'rtol', True)
    current_atol_A = _number(current_atol_A, 'current_atol_A', True)
    power_atol_W = _number(power_atol_W, 'power_atol_W', True)
    meta = _metadata(model_metadata)
    node_ids = sorted(_identifier(i, 'node ID') for i in graph.nodes)
    edge_ids = [_identifier(i, 'edge ID') for i in graph.edges]
    if not node_ids:
        raise ValueError('transport requires a nonempty node set')
    rows = {nid: i for i, nid in enumerate(node_ids)}
    positions = np.asarray([graph.nodes[i].position for i in node_ids], dtype=np.float64)
    if positions.shape != (len(node_ids), 3) or not np.isfinite(positions).all():
        raise ValueError('geometry positions must be finite (N,3) values')
    if not isinstance(conductance_S, Mapping):
        raise ValueError('conductance_S must map every stable edge ID to its conductance')
    supplied = {_identifier(k, 'conductance edge ID'): _number(v, 'conductance', True)
                for k, v in conductance_S.items()}
    if set(supplied) != set(edge_ids):
        raise ValueError('conductance keys must match graph edge IDs exactly (no missing or unknown IDs)')
    endpoints = []
    for eid in edge_ids:
        edge = graph.edges[eid]
        a = _identifier(edge.node_i, 'endpoint ID')
        b = _identifier(edge.node_j, 'endpoint ID')
        if a not in rows or b not in rows or a == b:
            raise ValueError('each edge needs two distinct existing endpoint IDs')
        endpoints.append((a, b))
    g = np.asarray([supplied[i] for i in edge_ids], dtype=np.float64)
    conn = np.array([(rows[a], rows[b]) for a, b in endpoints], dtype=np.int64).reshape(-1, 2)
    prescribed = {}
    pairs = dirichlet_voltage_V.items() if isinstance(dirichlet_voltage_V, Mapping) else dirichlet_voltage_V
    try:
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError('Dirichlet entries must be (node ID, voltage) pairs')
            nid, value = pair
            nid = _identifier(nid, 'Dirichlet node ID')
            value = _number(value, 'prescribed voltage')
            if nid not in rows:
                raise ValueError('Dirichlet node ID does not exist')
            if nid in prescribed and prescribed[nid] != value:
                raise ValueError(f'conflicting prescribed voltages for node {nid}')
            prescribed[nid] = value
    except TypeError as exc:
        raise ValueError('Dirichlet input must contain (node ID, voltage) pairs') from exc
    n, e = len(node_ids), len(edge_ids)
    active = conn[g > 0]
    adjacency = coo_matrix((np.ones(len(active)),
                           (active[:, 0], active[:, 1])), shape=(n, n)).tocsr()
    _, labels = connected_components(adjacency, directed=False)
    anchored = {int(labels[rows[nid]]) for nid in prescribed}
    floating = [nid for nid in node_ids if int(labels[rows[nid]]) not in anchored]
    if floating:
        raise FloatingComponentError(f'positive-conductance components lack a voltage anchor; node IDs: {floating}')
    B = coo_matrix((np.tile([1., -1.], e),
                    (np.repeat(np.arange(e), 2), conn.ravel())), shape=(e, n)).tocsr()
    with np.errstate(over='ignore', invalid='ignore'):
        L = (B.T @ diags(g, dtype=np.float64) @ B).tocsc()
    if not np.isfinite(L.data).all():
        raise TransportSolveError('conductance assembly overflowed')
    fixed = np.array(sorted(rows[nid] for nid in prescribed), dtype=np.int64)
    free = np.array([i for i in range(n) if node_ids[i] not in prescribed], dtype=np.int64)
    v = np.zeros(n, dtype=np.float64)
    for nid, value in prescribed.items(): v[rows[nid]] = value
    # Independent components are solved independently. A component with one
    # prescribed potential value has the exact constant solution, avoiding
    # roundoff-only currents in heterogeneous equipotential networks.
    component_rows = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    for members in component_rows:
        component_fixed = np.intersect1d(members, fixed)
        component_free = np.intersect1d(members, free)
        if not len(component_free):
            continue
        values = v[component_fixed]
        if np.all(values == values[0]):
            v[component_free] = values[0]
            continue
        rhs = -L[component_free, :][:, component_fixed] @ values
        if not np.isfinite(rhs).all():
            raise TransportSolveError('Dirichlet right-hand side overflowed')
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', MatrixRankWarning)
                v[component_free] = spsolve(
                    L[component_free, :][:, component_free], rhs, use_umfpack=False)
        except (MatrixRankWarning, RuntimeError, ValueError) as exc:
            raise TransportSolveError('sparse transport solve failed') from exc
    with np.errstate(over='ignore', invalid='ignore'):
        dv = B @ v
        q = np.zeros(e, dtype=np.float64)
        q[g > 0] = g[g > 0] * dv[g > 0]
        edge_power = np.zeros(e, dtype=np.float64)
        edge_power[g > 0] = q[g > 0] * dv[g > 0]
        injection = np.asarray(B.T @ q)
        power = float(np.sum(edge_power))
        boundary_power = 0.0  # populated from stable component-local powers below
    if not all(np.isfinite(x).all() for x in (v, q, injection, edge_power, power, boundary_power)):
        raise TransportSolveError('nonfinite transport output')
    # Nonzero driven currents and dissipations must be normal float64 values.
    # Reject underflow (including subnormals with unreliable relative precision)
    # rather than certifying invented zero fields.
    driven = (g > 0) & (dv != 0)
    tiny = np.finfo(np.float64).tiny
    if np.any(np.abs(q[driven]) < tiny):
        raise TransportSolveError('current underflow: nonzero driven current is not normally representable')
    if np.any(edge_power[driven] < tiny):
        raise TransportSolveError('power underflow: nonzero dissipation is not normally representable')
    residual = float(np.max(np.abs(injection[free]))) if len(free) else 0.
    balance = float(np.sum(injection))
    incident_current = np.asarray(abs(B).T @ np.abs(q))
    # The sparse solve uses ABSOLUTE voltages. The existing 32*eps factor is
    # an engineering backward-residual margin, not a sparse-LU theorem or a
    # forward-accuracy certificate at arbitrarily large voltage offsets.
    # Separately account for evaluation on the actual returned float fields.
    try:
        edge_evaluation = [upper_sum([upper_product(float(g[j]), evaluation_ulp(dv[j])),
                                     evaluation_ulp(q[j])]) if g[j] else 0.0 for j in range(e)]
        node_limits = np.zeros(n)
        solve_allowance = np.zeros(n)
        evaluation_allowance = np.zeros(n)
        u = np.finfo(float).eps / 2
        incident_rows = [[] for _ in range(n)]
        for j, (a, b) in enumerate(conn):
            incident_rows[a].append(j); incident_rows[b].append(j)
        for i, incident in enumerate(incident_rows):
            local_scale = upper_sum(upper_product(float(g[j]), upper_sum([abs(float(v[a])), abs(float(v[b]))]))
                                    for j in incident for a, b in [conn[j]])
            solve_allowance[i] = upper_product(32*np.finfo(float).eps, local_scale)
            ku = upper_product(len(incident), u)
            if ku >= 1:
                raise ArithmeticError('nodal accumulation budget is not representable')
            gamma = math.nextafter(ku/math.nextafter(1-ku, -math.inf), math.inf) if ku else 0.0
            current_scale = upper_sum(abs(float(q[j])) for j in incident)
            evaluation_allowance[i] = upper_sum([upper_sum(edge_evaluation[j] for j in incident),
                upper_product(gamma, current_scale)])
            node_limits[i] = upper_sum([current_atol_A, upper_product(rtol, current_scale),
                                       solve_allowance[i], evaluation_allowance[i]])
    except ArithmeticError as exc:
        raise TransportSolveError('local conservation scale overflowed') from exc
    if np.any(np.abs(injection[free]) > node_limits[free]):
        raise TransportSolveError('node-local current conservation tolerance failed')
    component_checks = []
    for members in component_rows:
        component_fixed = np.intersect1d(members, fixed)
        edge_mask = (g > 0) & np.isin(conn[:, 0], members)
        component_power = float(np.sum(edge_power[edge_mask]))
        # A reference shift avoids cancellation from arbitrary voltage offsets.
        reference = v[component_fixed[0]]
        supplied_power = float(np.dot(v[component_fixed]-reference, injection[component_fixed]))
        component_balance = float(np.sum(injection[members]))
        current_limit = current_atol_A + rtol * float(np.sum(np.abs(injection[component_fixed])))
        power_limit = power_atol_W + rtol * max(abs(component_power), abs(supplied_power))
        if not all(np.isfinite(x) for x in (component_power, supplied_power, component_balance, current_limit, power_limit)):
            raise TransportSolveError('component conservation scale overflowed')
        if abs(component_balance) > current_limit or abs(component_power-supplied_power) > power_limit:
            raise TransportSolveError('component-local current conservation or power balance tolerance failed')
        component_checks.append(dict(node_ids=[node_ids[i] for i in members],
                                     current_balance_A=component_balance,
                                     current_limit_A=current_limit,
                                     dissipated_power_W=component_power,
                                     reference_shifted_boundary_power_W=supplied_power,
                                     power_limit_W=power_limit))
    boundary_power = math.fsum(item['reference_shifted_boundary_power_W']
                               for item in component_checks)
    if not np.isfinite(boundary_power):
        raise TransportSolveError('summed boundary power overflowed')
    binding = dict(node_ids=node_ids, edge_ids=edge_ids, endpoints=endpoints,
                   conductance_S=g.tolist(), prescribed_voltage_V=sorted(prescribed.items()),
                   coordinates=positions.tolist(), length_unit=length_unit)
    fingerprint = hashlib.sha256(json.dumps(binding, sort_keys=True, allow_nan=False).encode()).hexdigest()
    provenance = dict(model_kind='steady-linear-electrical-conductance',
                      validation_contract='transport-roundoff-2',
                      assumptions=['ideal equipotential junctions', 'supplied constant edge conductance',
                                   'no distributed sources or transient storage'],
                      conductance_source='caller supplied; no geometry or material inference',
                      length_unit=length_unit, geometry_conversion='none',
                      units={'conductance': 'S', 'voltage': 'V', 'current': 'A', 'power': 'W'},
                      boundary_power_convention='sum of component reference-shifted boundary supply',
                      solver='scipy.sparse.linalg.spsolve; use_umfpack=False',
                      scipy_version=scipy.__version__, input_sha256=fingerprint,
                      tolerances=dict(rtol=rtol, current_atol_A=current_atol_A,
                                      power_atol_W=power_atol_W),
                      achieved_limits=dict(node_current_A=node_limits.tolist(), components=component_checks,
                                           roundoff_factor=32*np.finfo(float).eps,
                                           solve_roundoff_A=solve_allowance.tolist(),
                                           field_evaluation_roundoff_A=evaluation_allowance.tolist(),
                                           roundoff_coordinates='absolute solved voltage',
                                           roundoff_scope='engineering backward-residual margin; not forward accuracy'),
                      model_metadata=meta)
    return LinearTransportResult(
        _readonly(node_ids, np.int64), _readonly(edge_ids, np.int64),
        _readonly(endpoints, np.int64).reshape(-1, 2), _readonly(g), _readonly(v),
        _readonly(q), _readonly(injection), _readonly(edge_power), dict(prescribed),
        residual, balance, power, boundary_power, provenance)
