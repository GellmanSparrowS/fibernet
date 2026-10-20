"""Private pure fields/diagnostics extracted from constant thermal solver.

No solver or model inference. Used by one-way and coupled acceptance alike.
"""
import math
import numpy as np
from ._roundoff import upper_sum, upper_product, evaluation_ulp

class ThermalResidualRejection(ArithmeticError):
    """Finite equation residual only; eligible for a nonlinear Picard retry."""


def _product(a, b, label):
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        x = np.asarray(a)*np.asarray(b)
    if not np.isfinite(x).all() or np.any((np.asarray(a) != 0) & (np.asarray(b) != 0) & (x == 0)):
        raise ArithmeticError(label+' is not representable')
    return x


def thermal_fields(temp, *, conn, g, incident, fix, Q):
    n = len(temp)
    difference=temp[conn[:,0]]-temp[conn[:,1]]
    endpoint_bits=temp[conn].view(np.uint64)
    distinct=((endpoint_bits[:,0]!=endpoint_bits[:,1]) &
              (((endpoint_bits[:,0]|endpoint_bits[:,1])&0x7fffffffffffffff)!=0))
    if np.any((g>0)&(difference==0)&distinct):
        raise ArithmeticError('nonzero temperature difference is not representable')
    q=_product(g,difference,'edge heat rate')
    outward=np.array([math.fsum((q[j] if a==i else -q[j]) for j,a,b in incident[i]) for i in range(n)])
    if not np.isfinite(outward).all():raise ArithmeticError('nodal sum overflow')
    reservoir=np.zeros(n)
    with np.errstate(over='ignore', invalid='ignore'):
        reservoir[fix]=outward[fix]-Q[fix]
    # Validate every derived instantaneous field BEFORE accepting t=0 or
    # any later frame. Diagnostic NaNs are deliberately added separately.
    if not all(np.isfinite(a).all() for a in (q,outward,reservoir)):
        raise ArithmeticError('initial/trial heat field not representable: reservoir overflow')
    return q,outward,reservoir

def evaluate_thermal_step(*, T, new, inc, initial, dt, Q, C, g, conn,
                          fix, free, incident, members, component_edges):
    n, e, nc = len(T), len(g), len(members)
    eps = np.finfo(float).eps
    is_free = np.ones(n, dtype=bool)
    is_free[fix] = False
    dtQ = _product(dt, Q, 'source step heat')
    delta=new-T
    if np.any((inc!=0)&(delta==0)):raise ArithmeticError('increment lost to absolute temperature precision')
    qnew,outnew,resnew=thermal_fields(new, conn=conn, g=g, incident=incident, fix=fix, Q=Q)
    stepq=_product(dt,qnew,'new edge step heat')
    energy=_product(C,delta,'stored step heat')
    stored=_product(C,new-initial,'cumulative stored heat')
    # Previous accepted T is the physical step input. Only this step's
    # new=T+inc reconstruction contributes endpoint uncertainty.
    rho=np.array([upper_product(.5, math.ulp(float(new[i]))) if inc[i]!=0 else 0.0 for i in range(n)])
    capacity_budget=np.array([upper_product(8*eps,float(C[i]),upper_sum([abs(float(T[i])),abs(float(new[i]))]))
                              if is_free[i] else 0.0 for i in range(n)])
    difference=new[conn[:,0]]-new[conn[:,1]]
    flux_eval=[upper_sum([upper_product(float(g[j]),evaluation_ulp(difference[j])),evaluation_ulp(qnew[j])])
               if g[j] else 0.0 for j in range(e)]
    step_eval=[upper_sum([upper_product(float(dt),flux_eval[j]),evaluation_ulp(stepq[j])]) for j in range(e)]
    energy_eval=[upper_sum([upper_product(float(C[i]),evaluation_ulp(delta[i])),evaluation_ulp(energy[i])]) for i in range(n)]
    source_eval=[evaluation_ulp(x) for x in dtQ]
    res=np.full(n,np.nan);nb=np.full(n,np.nan)
    for i in free:
        signed=[float(stepq[j] if a==i else -stepq[j]) for j,a,b in incident[i]]
        scale=math.fsum([abs(energy[i]),abs(dtQ[i])]+[abs(x) for x in signed])
        r=math.fsum([energy[i],-dtQ[i]]+signed)
        conduction=upper_sum(upper_product(float(dt),float(g[j]),upper_sum([rho[a],rho[b]]))
                             for j,a,b in incident[i])
        evaluation=upper_sum([energy_eval[i],source_eval[i],evaluation_ulp(r)]+
                             [step_eval[j] for j,a,b in incident[i]])
        if scale==0: capacity_budget[i]=0.0
        b=upper_sum([capacity_budget[i],conduction,evaluation])
        if not all(math.isfinite(x) for x in (scale,r,b)):raise ArithmeticError('residual scale overflow')
        if b>1e-6*scale:raise ArithmeticError('temperature precision budget exceeds local heat scale')
        bound=upper_sum([upper_product(1e-10,scale),b])
        if abs(r)>bound or (scale==0 and inc[i]!=0):raise ThermalResidualRejection('free-node discrete energy residual failed')
        res[i],nb[i]=r,bound
    cr=[];cb=[]
    reservoir_step=_product(dt,resnew,'reservoir step heat')
    for component,ix in enumerate(members):
        r=math.fsum([float(energy[i]) for i in ix]+[-float(dtQ[i]) for i in ix]+[-float(reservoir_step[i]) for i in ix])
        scale=math.fsum([abs(float(energy[i]))+abs(float(dtQ[i]))+abs(float(reservoir_step[i])) for i in ix])
        # Internal free/free reconstruction flux cancels. Only cut
        # edges to fixed reservoirs enter the component allowance.
        fixed_set=set(map(int,fix))
        cut=upper_sum(upper_product(float(dt),float(g[j]),rho[b if a in fixed_set else a])
                      for j,a,b in component_edges[component]
                      if ((a in fixed_set)!=(b in fixed_set)))
        reservoir_eval=[]
        for i in ix:
            if i in fixed_set:
                outward_error=upper_sum([evaluation_ulp(outnew[i])]+
                    [flux_eval[j] for j,a,b in incident[i]])
                reservoir_eval.append(upper_sum([upper_product(float(dt),upper_sum([outward_error,evaluation_ulp(resnew[i])])),
                                                 evaluation_ulp(reservoir_step[i])]))
        b=upper_sum([upper_sum(capacity_budget[i] for i in ix),cut,
                     upper_sum(energy_eval[i] for i in ix),upper_sum(source_eval[i] for i in ix),
                     upper_sum(reservoir_eval),evaluation_ulp(r)])
        bound=upper_sum([upper_product(1e-10,scale),b])
        if not all(math.isfinite(v) for v in (r,scale,b,bound)) or b>1e-6*scale:raise ArithmeticError('component energy balance failed')
        if abs(r)>bound:raise ThermalResidualRejection('component energy balance failed')
        cr.append(r);cb.append(bound)
    return {'T':new.copy(), 'q':qnew, 'outward':outnew, 'reservoir':resnew,
            'stored':stored, 'interval_stored':energy, 'residual':res, 'energy':cr, 'nbound':nb, 'cbound':cb}
