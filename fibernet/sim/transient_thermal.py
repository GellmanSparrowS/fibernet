"""Explicit lumped thermal network; constant inputs and backward Euler only.

No geometry/material inference. Array order is explicit; no root API export.
Numerical equation acceptance does not certify time-discretization accuracy.
"""
from dataclasses import dataclass, fields
from collections.abc import Mapping
import hashlib
import json
import math
import numbers
import warnings

import numpy as np
import scipy
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve, MatrixRankWarning

from ._roundoff import upper_sum, upper_product, evaluation_ulp
from ._roundoff import _require_gradual_for_subnormal, _SubnormalRoundoffError


class ThermalSolveError(RuntimeError):
    """Numerical failure; partial_result contains accepted frames only.

    None means even the initial flux was not representable. attempted_step/time
    identify the rejected state; no trial arrays enter the accepted prefix.
    """
    def __init__(self, reason, partial_result, attempted_step, attempted_time_s):
        super().__init__(reason)
        self.partial_result = partial_result
        self.attempted_step = attempted_step
        self.attempted_time_s = attempted_time_s


def _ro(x, dtype=float):
    a = np.array(x, dtype=dtype, copy=True)
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class TransientThermalResult:
    node_ids: np.ndarray
    edge_ids: np.ndarray
    endpoints_by_id: np.ndarray
    positions: np.ndarray
    component_labels: np.ndarray
    capacity_J_per_K: np.ndarray
    conductance_W_per_K: np.ndarray
    source_W: np.ndarray
    fixed_node_ids: np.ndarray
    fixed_temperature_K: np.ndarray
    times_s: np.ndarray
    temperature_K: np.ndarray
    edge_heat_rate_W: np.ndarray
    reservoir_injection_W: np.ndarray
    node_stored_heat_change_J: np.ndarray
    free_node_residual_J: np.ndarray
    component_energy_residual_J: np.ndarray
    node_acceptance_bound_J: np.ndarray
    component_acceptance_bound_J: np.ndarray
    interval_diagnostics_available: np.ndarray
    completed: bool
    input_sha256: str
    provenance_json: str

    def to_dict(self) -> dict:
        """Return detached strict-JSON-ready fields without changing arrays.

        Null means unavailable, never zero: interval diagnostics at t=0, and
        free-node residuals/bounds at fixed-temperature nodes. Availability is
        determined by the existing mask and fixed IDs, not by finding NaNs.
        Other nonfinite values, or inconsistent diagnostic sentinels, raise
        ValueError. ``completed=False`` still describes only accepted frames.
        """
        nt, nn = len(self.times_s), len(self.node_ids)
        mask = np.asarray(self.interval_diagnostics_available)
        if (nt == 0 or mask.shape != (nt,) or mask.dtype.kind != 'b' or
                mask[0] or not mask[1:].all()):
            raise ValueError('interval_diagnostics_available must mark only t=0 unavailable')
        if (self.times_s.shape != (nt,) or self.times_s[0] != 0 or
                np.any(np.diff(self.times_s) <= 0)):
            raise ValueError('times_s must start at zero and strictly increase')
        if (self.node_ids.shape != (nn,) or self.fixed_node_ids.ndim != 1 or
                not np.isin(self.fixed_node_ids, self.node_ids).all() or
                self.component_labels.shape != (nn,)):
            raise ValueError('diagnostic node, fixed-node and component rows must agree')
        nc = len(np.unique(self.component_labels))
        node_available = mask[:, None] & ~np.isin(self.node_ids, self.fixed_node_ids)[None, :]
        component_available = np.broadcast_to(mask[:, None], (nt, nc))
        availability = {
            'free_node_residual_J': node_available,
            'node_acceptance_bound_J': node_available,
            'component_energy_residual_J': component_available,
            'component_acceptance_bound_J': component_available,
        }
        value = {}
        for field in fields(self):
            item = getattr(self, field.name)
            if field.name in availability:
                available = availability[field.name]
                if item.shape != available.shape:
                    raise ValueError(field.name + ' diagnostic shape mismatch')
                if not np.isfinite(item[available]).all():
                    raise ValueError(field.name + ' has nonfinite available diagnostics')
                if not np.isnan(item[~available]).all():
                    raise ValueError(field.name + ' unavailable diagnostics must use NaN sentinels')
                detached = item.astype(object)
                detached[~available] = None
                value[field.name] = detached.tolist()
            elif isinstance(item, np.ndarray):
                if not np.isfinite(item).all():
                    raise ValueError(field.name + ' must contain only finite values')
                value[field.name] = item.tolist()
            else:
                value[field.name] = item
        # Match the existing Joule ledger convention; keep its recorded units,
        # input/source metadata and validation limits rather than infer new ones.
        value['provenance'] = json.loads(value.pop('provenance_json'))
        if not isinstance(value['provenance'], dict):
            raise ValueError('provenance_json must encode a JSON object')
        return json.loads(json.dumps(value, allow_nan=False))


def _id(v):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not 0 <= v <= np.iinfo(np.int64).max:
        raise ValueError('IDs must be nonnegative int64 integers, not bool')
    return int(v)


def _num(v):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real) or not math.isfinite(v):
        raise ValueError('physical inputs must be finite nonnegative real numbers')
    value = float(v)
    try:
        _require_gradual_for_subnormal(value)
    except _SubnormalRoundoffError as exc:
        raise ValueError('nonzero subnormal physical input requires gradual underflow') from exc
    if value < 0:
        raise ValueError('physical inputs must be finite nonnegative real numbers')
    return value


def _mapping(values, ids, name, positive=False):
    if not isinstance(values, Mapping):
        raise ValueError(name+' must be an ID mapping')
    d = {_id(k): _num(v) for k, v in values.items()}
    if set(d) != set(ids):
        raise ValueError(name+' requires exact ID coverage')
    a = np.array([d[i] for i in ids])
    if positive and np.any(a <= 0):
        raise ValueError(name+' must be strictly positive')
    return a


from ._thermal_step import _product, thermal_fields, evaluate_thermal_step


def _solve_transient_thermal(graph, capacity_J_per_K, conductance_W_per_K,
                            initial_temperature_K, times_s, *, source_W=None,
                            fixed_temperature_K=(), length_unit, model_metadata=None, accepted_step_observer=None):
    """Solve only the declared constant lumped model with explicit SI inputs.

    q is positive node_i -> node_j; reservoir injection is positive into graph.
    Initial temperatures equal fixed values exactly. All C>0, g>=0, Q>=0.
    No anchor is required. t=0 interval diagnostics are unavailable (NaN/mask).
    Residual tolerance is 1e-10 local scale PLUS a temperature representation
    allowance, itself capped at 1e-6 local scale. No dimensional absolute floor.
    """
    if length_unit not in ('m','cm','mm','um','nm'):
        raise ValueError('declare geometry length_unit; no conversion inferred')
    ids = sorted(_id(k) for k in graph.nodes)
    eids = [_id(k) for k in graph.edges]
    if not ids:
        raise ValueError('at least one node required')
    rows = {k:i for i,k in enumerate(ids)}
    xyz = np.array([graph.nodes[k].position for k in ids], dtype=float)
    if xyz.shape != (len(ids),3) or not np.isfinite(xyz).all():
        raise ValueError('finite node coordinates required')
    ends = [(_id(graph.edges[k].node_i),_id(graph.edges[k].node_j)) for k in eids]
    if any(a not in rows or b not in rows or a==b for a,b in ends):
        raise ValueError('edges need distinct existing endpoints')
    conn = np.array([(rows[a],rows[b]) for a,b in ends], dtype=int).reshape(-1,2)
    C = _mapping(capacity_J_per_K,ids,'capacity',True)
    g = _mapping(conductance_W_per_K,eids,'conductance')
    T = _mapping(initial_temperature_K,ids,'initial temperature')
    Q = np.zeros(len(ids)) if source_W is None else _mapping(source_W,ids,'source')
    fixed = {}
    pairs = fixed_temperature_K.items() if isinstance(fixed_temperature_K,Mapping) else fixed_temperature_K
    try:
        for k,v in pairs:
            k,v = _id(k),_num(v)
            if k not in rows or (k in fixed and fixed[k]!=v):
                raise ValueError('unknown or conflicting fixed node')
            fixed[k]=v
    except (TypeError,ValueError) as exc:
        raise ValueError('fixed temperatures require valid (node ID, K) pairs') from exc
    fix = np.array(sorted(rows[k] for k in fixed),dtype=int)
    free = np.array([i for i,k in enumerate(ids) if k not in fixed],dtype=int)
    if any(T[rows[k]]!=v for k,v in fixed.items()):
        raise ValueError('fixed initial temperature must match exactly')
    try:
        raw = list(times_s)
        if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,numbers.Real) for v in raw):
            raise ValueError('invalid time entries')
        times=np.array(raw,dtype=float)
    except (TypeError,ValueError) as exc:
        raise ValueError('time grid must be real one-dimensional values') from exc
    eps=np.finfo(float).eps
    if times.ndim!=1 or len(times)<2 or not np.isfinite(times).all() or times[0]!=0 or np.any(np.diff(times)<=0):
        raise ValueError('time grid starts at zero and strictly increases')
    dts=np.diff(times)
    if np.any(np.abs(dts-dts[0])>32*eps*max(abs(times[-1]),dts[0])):
        raise ValueError('uniform time grid required')
    if model_metadata is not None and not isinstance(model_metadata,dict):
        raise ValueError('metadata must be a dictionary')
    try:
        metadata=json.loads(json.dumps({} if model_metadata is None else model_metadata,allow_nan=False))
    except (TypeError,ValueError) as exc:
        raise ValueError('metadata must be finite JSON') from exc
    if not isinstance(metadata,dict):
        raise ValueError('metadata must be a dictionary')
    payload={'node_ids':ids,'edge_ids':eids,'endpoints':ends,'positions':xyz.tolist(),
             'C':C.tolist(),'g':g.tolist(),'T0':T.tolist(),'Q':Q.tolist(),
             'fixed':sorted(fixed.items()),'times_s':times.tolist(),'length_unit':length_unit,'metadata':metadata}
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()
    active=conn[g>0]
    adj=coo_matrix((np.ones(len(active)),(active[:,0],active[:,1])),shape=(len(ids),len(ids)))
    nc,labels=connected_components(adj,directed=False)
    members=[np.flatnonzero(labels==i) for i in range(nc)]
    n,e=len(ids),len(eids)
    # Preserve edge insertion order exactly while avoiding repeated full-edge
    # scans for each node. Parallel/inactive edges retain their own rows.
    incident=[[] for _ in range(n)]
    component_edges=[[] for _ in range(nc)]
    for j,(a,b) in enumerate(conn):
        incident[a].append((j,a,b));incident[b].append((j,a,b))
        component_edges[int(labels[a])].append((j,a,b))
    histories={k:[] for k in ('T','q','reservoir','stored','residual','energy','nbound','cbound')}
    provenance=json.dumps({'validation_contract':'transport-roundoff-2','thermal_capacity_reconstruction_profile':'binary64-rn-capacity-rho-1','model':'constant_lumped_thermal','method':'backward_euler','quadrature':'right_endpoint',
         'units':{'T':'K','C':'J/K','g':'W/K','q':'W','Q':'W','stored':'J','residual':'J'},
         'length_unit':length_unit,'geometry_inference':False,'scipy_version':scipy.__version__,
         'time_accuracy_verified':False,'relative_algebraic_term':1e-10,'relative_representation_cap':1e-6,
         'representation_budget':'capacity plus endpoint reconstruction conduction; component cut edges only; explicit field evaluation',
         'metadata':metadata},sort_keys=True)
    def result(completed=False):
        if not histories['T']:return None
        h=histories;nt=len(h['T'])
        return TransientThermalResult(_ro(ids,np.int64),_ro(eids,np.int64),_ro(ends,np.int64).reshape(-1,2),_ro(xyz),
            _ro(labels,np.int64),_ro(C),_ro(g),_ro(Q),_ro([ids[i] for i in fix],np.int64),_ro(T[fix]),
            _ro(times[:nt]),_ro(h['T']),_ro(h['q']),_ro(h['reservoir']),_ro(h['stored']),_ro(h['residual']),
            _ro(h['energy']),_ro(h['nbound']),_ro(h['cbound']),_ro([False]+[True]*(nt-1),bool),completed,digest,provenance)
    def fields(temp):
        return thermal_fields(temp, conn=conn, g=g, incident=incident, fix=fix, Q=Q)
    is_free=np.ones(n,dtype=bool);is_free[fix]=False
    attempted=0
    try:
        q,outward,reservoir=fields(T)
        for k,v in {'T':T.copy(),'q':q,'reservoir':reservoir,'stored':np.zeros(n),'residual':np.full(n,np.nan),
                    'energy':np.full(nc,np.nan),'nbound':np.full(n,np.nan),'cbound':np.full(nc,np.nan)}.items():histories[k].append(v)
        initial=T.copy()
        for attempted,dt in enumerate(dts,start=1):
            dtg=_product(dt,g,'dt times conductance');dtQ=_product(dt,Q,'source step heat')
            _product(dt,q,'edge step heat')
            rhs=_product(dt,Q-outward,'driving step heat')
            B=coo_matrix((np.tile([1.,-1.],e),(np.repeat(np.arange(e),2),conn.ravel())),shape=(e,n)).tocsr()
            A=(diags(C)+(B.T@diags(dtg)@B)).tocsc()
            if not np.isfinite(A.data).all():raise ArithmeticError('matrix assembly overflow')
            inc=np.zeros(n)
            if len(free) and np.any(rhs[free]!=0):
                with warnings.catch_warnings():
                    warnings.simplefilter('error',MatrixRankWarning)
                    inc[free]=spsolve(A[free,:][:,free],rhs[free])
            if not np.isfinite(inc).all():raise ArithmeticError('nonfinite solution')
            new=T+inc
            if not np.isfinite(new).all() or np.any(new<0):raise ArithmeticError('invalid reconstructed temperature')
            evaluated = evaluate_thermal_step(T=T, new=new, inc=inc, initial=initial,
                dt=dt, Q=Q, C=C, g=g, conn=conn, fix=fix, free=free,
                incident=incident, members=members, component_edges=component_edges)
            if accepted_step_observer is not None:
                accepted_step_observer(_ro(inc))
            for key in histories:
                histories[key].append(evaluated[key])
            T,q,outward=new,evaluated['q'],evaluated['outward']
        return result(True)
    except (ArithmeticError,RuntimeError,MatrixRankWarning,ValueError) as exc:
        raise ThermalSolveError(str(exc),result(False),attempted,float(times[attempted])) from exc


def solve_transient_thermal(graph, capacity_J_per_K, conductance_W_per_K,
                            initial_temperature_K, times_s, *, source_W=None,
                            fixed_temperature_K=(), length_unit, model_metadata=None):
    """Solve only the declared constant lumped model with explicit SI inputs.

    q is positive node_i -> node_j; reservoir injection is positive into graph.
    Initial temperatures equal fixed values exactly. All C>0, g>=0, Q>=0.
    No anchor is required. t=0 interval diagnostics are unavailable (NaN/mask).
    Residual tolerance is 1e-10 local scale PLUS a temperature representation
    allowance, itself capped at 1e-6 local scale. No dimensional absolute floor.
    """
    return _solve_transient_thermal(graph, capacity_J_per_K, conductance_W_per_K,
        initial_temperature_K, times_s, source_W=source_W,
        fixed_temperature_K=fixed_temperature_K, length_unit=length_unit,
        model_metadata=model_metadata)


def _solve_thermal_interval(graph, capacity_J_per_K, conductance_W_per_K,
                            initial_temperature_K, dt, *, source_W,
                            fixed_temperature_K, length_unit):
    """Internal one-interval adapter retaining the original solved increment."""
    increments = []
    result = _solve_transient_thermal(graph, capacity_J_per_K, conductance_W_per_K,
        initial_temperature_K, [0., dt], source_W=source_W,
        fixed_temperature_K=fixed_temperature_K, length_unit=length_unit,
        accepted_step_observer=increments.append)
    return result, increments[0]
