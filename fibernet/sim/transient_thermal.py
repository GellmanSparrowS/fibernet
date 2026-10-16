"""Explicit lumped thermal network; constant inputs and backward Euler only.

No geometry/material inference. Array order is explicit; no root API export.
Numerical equation acceptance does not certify time-discretization accuracy.
"""
from dataclasses import dataclass
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


def _id(v):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not 0 <= v <= np.iinfo(np.int64).max:
        raise ValueError('IDs must be nonnegative int64 integers, not bool')
    return int(v)


def _num(v):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real) or not math.isfinite(v) or v < 0:
        raise ValueError('physical inputs must be finite nonnegative real numbers')
    return float(v)


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


def _product(a, b, label):
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        x = np.asarray(a)*np.asarray(b)
    if not np.isfinite(x).all() or np.any((np.asarray(a) != 0) & (np.asarray(b) != 0) & (x == 0)):
        raise ArithmeticError(label+' is not representable')
    return x


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
    histories={k:[] for k in ('T','q','reservoir','stored','residual','energy','nbound','cbound')}
    provenance=json.dumps({'model':'constant_lumped_thermal','method':'backward_euler','quadrature':'right_endpoint',
         'units':{'T':'K','C':'J/K','g':'W/K','q':'W','Q':'W','stored':'J','residual':'J'},
         'length_unit':length_unit,'geometry_inference':False,'scipy_version':scipy.__version__,
         'time_accuracy_verified':False,'relative_algebraic_term':1e-10,'relative_representation_cap':1e-6,'metadata':metadata},sort_keys=True)
    def result(completed=False):
        if not histories['T']:return None
        h=histories;nt=len(h['T'])
        return TransientThermalResult(_ro(ids,np.int64),_ro(eids,np.int64),_ro(ends,np.int64).reshape(-1,2),_ro(xyz),
            _ro(labels,np.int64),_ro(C),_ro(g),_ro(Q),_ro([ids[i] for i in fix],np.int64),_ro(T[fix]),
            _ro(times[:nt]),_ro(h['T']),_ro(h['q']),_ro(h['reservoir']),_ro(h['stored']),_ro(h['residual']),
            _ro(h['energy']),_ro(h['nbound']),_ro(h['cbound']),_ro([False]+[True]*(nt-1),bool),completed,digest,provenance)
    def fields(temp):
        q=_product(g,temp[conn[:,0]]-temp[conn[:,1]],'edge heat rate')
        outward=np.array([math.fsum((q[j] if a==i else -q[j]) for j,(a,b) in enumerate(conn) if a==i or b==i) for i in range(n)])
        if not np.isfinite(outward).all():raise ArithmeticError('nodal sum overflow')
        reservoir=np.zeros(n)
        with np.errstate(over='ignore', invalid='ignore'):
            reservoir[fix]=outward[fix]-Q[fix]
        # Validate every derived instantaneous field BEFORE accepting t=0 or
        # any later frame. Diagnostic NaNs are deliberately added separately.
        if not all(np.isfinite(a).all() for a in (q,outward,reservoir)):
            raise ArithmeticError('initial/trial heat field not representable: reservoir overflow')
        return q,outward,reservoir
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
            delta=new-T
            if np.any((inc!=0)&(delta==0)):raise ArithmeticError('increment lost to absolute temperature precision')
            qnew,outnew,resnew=fields(new)
            stepq=_product(dt,qnew,'new edge step heat')
            energy=_product(C,delta,'stored step heat')
            stored=_product(C,new-initial,'cumulative stored heat')
            res=np.full(n,np.nan);nb=np.full(n,np.nan);rounding=np.zeros(n)
            for i in free:
                signed=[float(stepq[j] if a==i else -stepq[j]) for j,(a,b) in enumerate(conn) if a==i or b==i]
                scale=math.fsum([abs(energy[i]),abs(dtQ[i])]+[abs(x) for x in signed])
                r=math.fsum([energy[i],-dtQ[i]]+signed)
                b=8*eps*C[i]*(abs(T[i])+abs(new[i])) if scale!=0 else 0.
                if not all(math.isfinite(x) for x in (scale,r,b)):raise ArithmeticError('residual scale overflow')
                if b>1e-6*scale:raise ArithmeticError('temperature precision budget exceeds local heat scale')
                bound=1e-10*scale+b
                if abs(r)>bound or (scale==0 and inc[i]!=0):raise ArithmeticError('free-node discrete energy residual failed')
                res[i],nb[i],rounding[i]=r,bound,b
            cr=[];cb=[]
            reservoir_step=_product(dt,resnew,'reservoir step heat')
            for ix in members:
                r=math.fsum([float(energy[i]) for i in ix]+[-float(dtQ[i]) for i in ix]+[-float(reservoir_step[i]) for i in ix])
                scale=math.fsum([abs(float(energy[i]))+abs(float(dtQ[i]))+abs(float(reservoir_step[i])) for i in ix])
                b=math.fsum(float(rounding[i]) for i in ix);bound=1e-10*scale+b
                if not all(math.isfinite(v) for v in (r,scale,b,bound)) or b>1e-6*scale or abs(r)>bound:raise ArithmeticError('component energy balance failed')
                cr.append(r);cb.append(bound)
            for k,v in {'T':new.copy(),'q':qnew,'reservoir':resnew,'stored':stored,'residual':res,'energy':cr,'nbound':nb,'cbound':cb}.items():histories[k].append(v)
            T,q,outward=new,qnew,outnew
        return result(True)
    except (ArithmeticError,RuntimeError,MatrixRankWarning,ValueError) as exc:
        raise ThermalSolveError(str(exc),result(False),attempted,float(times[attempted])) from exc
