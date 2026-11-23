"""Experimental bounded source-free scalar P1 Tet4 steady conduction."""
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
import math
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim import tet4 as core

__all__ = ["Tet4ConductionResult", "solve_steady_conduction_tet4"]

_MAX_NODES = 1024
_MAX_CELLS = 8000
_MAX_K = 1e6  # W/(m K); bounded prototype envelope
_MIN_K = 1e-6
_ABS_T_MIN = 1.
_ABS_T_MAX = 2000.
_COORD_MAX_M = 1.
_VOLUME_MIN_M3 = 1e-24
_GRAD_MAX_PER_M = 1e12
_TRANSFER_REL = 2e-10
# Centred temperatures can retain variation lost when adding a Kelvin offset.
# The 2e-10 relative gate explicitly limits that representation loss on the
# complete physical field. This is a consistency gate, not a PDE error bound.


@dataclass(frozen=True)
class Tet4ConductionResult:
    """Owned SI fields in original stable-ID order, with per-node component references."""
    node_ids: np.ndarray
    cell_ids: np.ndarray
    temperature_K: np.ndarray
    temperature_difference_K: np.ndarray
    reference_K_by_node: np.ndarray
    prescribed_mask: np.ndarray
    heat_into_solid_W: np.ndarray
    gradient_K_per_m: np.ndarray
    flux_W_per_m2: np.ndarray
    dirichlet_quadratic_W_K: float
    terminal_pairing_W_K: float
    free_residual_pairing_W_K: float
    quadratic_closure_W_K: float
    max_absolute_transfer_scaled_error: float
    free_residual_max_W: float
    normalized_backward: float
    factor_entries: int


def _lock(a):
    return np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape)


def _mapped(mapping,keys,label,low,high,complete=False):
    if not isinstance(mapping,Mapping) or len(mapping)>len(keys):
        raise ValueError('invalid bounded '+label+' mapping')
    out={}
    for j,(k,v) in enumerate(mapping.items()):
        k=core._integer(k,label+' stable ID')
        v=core._float64(v,label)
        if j>=len(keys) or k in out or k not in keys or not low<=v<=high:
            raise ValueError('invalid '+label+' ID or value')
        out[k]=v
    if complete and set(out)!=keys:
        raise ValueError('missing '+label)
    return out


def solve_steady_conduction_tet4(
    bundle: Tet4Bundle, *, conductivity_W_per_m_K_by_material: Mapping,
    prescribed_K_by_id: Mapping,
) -> Tet4ConductionResult:
    """Solve a steady source-free Tet4 snapshot with prescribed absolute Kelvin.

    Conductivity is scalar W/(m K) keyed by stable material ID. Prescribe
    at least one node in each conductive component. Unspecified exterior faces
    are weakly insulated. Positive discrete cap heat enters the solid.
    The Dirichlet quadratic and terminal pairing are measured in W K,
    not stored energy. This is not a heat source, transient, or reverse
    thermoelastic coupling API.
    """
    if not isinstance(bundle, Tet4Bundle):
        raise TypeError('expected a Tet4Bundle')
    if core._row_count(bundle.points,'points')>_MAX_NODES or core._row_count(bundle.tetrahedra,'tetrahedra')>_MAX_CELLS:
        raise ValueError('scalar conduction limited to 1024 nodes and 8000 Tet4')
    mesh=core._validate_static_mesh(bundle)
    x=mesh.points*core._LENGTH_TO_M[mesh.length_unit]; t=mesh.tetrahedra
    n=len(x)
    if np.max(abs(x),initial=0)>_COORD_MAX_M:
        raise ValueError('conduction coordinate limit')
    key={int(k):i for i,k in enumerate(mesh.node_ids)}
    kk=_mapped(conductivity_W_per_m_K_by_material,set(mesh.materials),'conductivity',_MIN_K,_MAX_K,True)
    temp=_mapped(prescribed_K_by_id,set(key),'temperature',_ABS_T_MIN,_ABS_T_MAX)
    if not temp: raise ValueError('each conductive component requires a prescribed node')
    fixed=np.zeros(n,dtype=np.bool_); known=np.zeros(n)
    for k,vt in temp.items():j=key[k];fixed[j]=True;known[j]=vt
    graph=core._free_node_topology(t,n,np.arange(n))
    _,components=connected_components(graph,directed=False)
    if set(components[fixed])!=set(components):
        raise ValueError('each conductive component requires a prescribed node')
    ref=np.empty(n)
    for lab in np.unique(components):
        ix=components==lab
        ref[ix]=np.min(known[ix&fixed])
    theta=np.zeros(n);theta[fixed]=known[fixed]-ref[fixed]
    ge=[]; V=[]; axes=[]
    for i,c in enumerate(t):
        g, v = core._shape_gradients_and_volume_m(x, c, i)
        if not (_VOLUME_MIN_M3<=v<=1) or not np.isfinite(g).all() or np.max(abs(g))>_GRAD_MAX_PER_M:
            raise ValueError('cell outside conduction volume/gradient envelope')
        ge.append(g); V.append(v)
        axes.append(kk[int(mesh.material_ids[i])])
    ge=np.array(ge);V=np.array(V);k=np.array(axes)
    local=(k*V)[:,None,None]*(ge@ge.transpose(0,2,1))
    rows=np.broadcast_to(t[:,:,None],local.shape); cols=np.broadcast_to(t[:,None,:],local.shape)
    A=sparse.coo_matrix((local.ravel(),(rows.ravel(),cols.ravel())),shape=(n,n)).tocsr()
    A=.5*A+.5*A.T;A.eliminate_zeros()
    if not np.isfinite(A.data).all():raise ValueError('nonfinite conduction matrix')
    ids=np.flatnonzero(~fixed);cap=np.flatnonzero(fixed)
    tol=64*np.finfo(np.float64).eps*max(1,len(ids))
    backward=0.; entries=0
    if len(ids):
        graph=core._free_node_topology(t,n,ids)
        order=core._canonical_node_order(graph,x[ids],mesh.node_ids[ids])
        ranks=np.empty_like(order);ranks[order]=np.arange(len(order))
        i,j=graph.nonzero()
        h=int(np.max(abs(ranks[i]-ranks[j]),initial=0))
        entries=(2*len(ids)-h)*(h+1);pad=len(ids)*(h+1)
        mat=A[ids][:,ids]; bound=float(abs(mat).sum(axis=1).max())
        if not np.isfinite(bound) or bound<=0:raise ValueError('unresolved scalar operator scale')
        rhs=-np.asarray(A[ids][:,cap]@theta[cap])/bound
        m=mat/bound
        plan=core._BandPlan(ids,order,h,entries,pad)
        theta[ids]=core._factor_band(m,tol,plan)(rhs)
        den=float(abs(m).sum(axis=1).max())*float(np.max(abs(theta[ids])))+float(np.max(abs(rhs)))
        defect=float(np.max(abs(m@theta[ids]-rhs)))
        backward=defect/den if den>0 else defect
        if not 0<=backward<=tol:raise ValueError('scalar backward residual rejected')
    T=ref+theta;T[fixed]=known[fixed]  # publish input absolute K exactly
    if not np.all((_ABS_T_MIN<=T)&(T<=_ABS_T_MAX)):
        raise ValueError('absolute conduction result outside [1,2000] K; no clipping')
    recon=T-ref
    max_transfer=0.
    for lab in np.unique(components):
        ix=components==lab
        span=float(np.ptp(theta[ix]))
        err=float(np.max(abs(recon[ix]-theta[ix])))
        if span==0.:
            if err!=0.:raise ValueError('unresolved constant absolute temperature reconstruction')
        elif err>_TRANSFER_REL*span:
            raise ValueError('unresolved absolute-to-centered temperature transfer')
        if span>0.:max_transfer=max(max_transfer,err/span)
    # Rebuild fluxes from differences, independently of A@theta, avoiding constant modes.
    grad=np.einsum('ni,nik->nk',theta[t[:,1:]]-theta[t[:,:1]],ge[:,1:,:])
    flux=-k[:,None]*grad
    el=-V[:,None]*np.einsum('nki,ni->nk',ge,flux)
    power=np.zeros(n)
    np.add.at(power,t,el)
    heat=np.where(fixed,power,0.)
    qcell=(k*V)*np.einsum('ni,ni->n',grad,grad)  # K W, not heat flow/power
    if np.any(~np.isfinite(qcell)) or np.any(qcell<0.):
        raise ValueError('invalid conductive quadratic')
    q=math.fsum(qcell)
    work=math.fsum(heat*theta)
    free_work=math.fsum(power[~fixed]*theta[~fixed])
    audited_A=A@theta
    gross_local=(abs(local)*abs(theta[t])[:,None,:]).sum(axis=(1,2))
    labels=components[t[:,0]]
    for lab in np.unique(components):
        ix=components==lab;cells=labels==lab; free=ix&(~fixed)
        flo=256*np.finfo(np.float64).eps*math.fsum(gross_local[cells])
        eps_W=1e-9*math.fsum(abs(el[cells]).ravel())+flo
        if max(float(abs(power[free]).max(initial=0)),abs(math.fsum(power[ix])),
               float(abs(power[ix]-audited_A[ix]).max(initial=0)))>eps_W:
            raise ValueError('conduction flux or free residual rejected')
        U=math.fsum(qcell[cells]); W=math.fsum(heat[ix]*theta[ix]); F=math.fsum(power[free]*theta[free])
        eps_WK=1e-9*(math.fsum(abs(power[ix]*theta[ix]))+abs(U))+flo*float(np.max(abs(theta[ix])))
        if not np.isfinite([U,W,F]).all() or abs(math.fsum((U,-W,-F)))>eps_WK:
            raise ValueError('conduction Dirichlet quadratic failed')
        # Independently bound the *direct* terminal pairing against the
        # geometric scalar quadratic by measured free-row error and roundoff.
        residual_virtual_bound=math.fsum(abs(power[free]*theta[free]))
        if abs(math.fsum((U,-W)))>residual_virtual_bound+eps_WK:
            raise ValueError('conduction terminal pairing unresolved')
    return Tet4ConductionResult(*map(_lock,(mesh.node_ids,mesh.cell_ids,T,theta,ref)),
        *map(_lock,(fixed,heat,grad,flux)),q,work,free_work,math.fsum((q,-work,-free_work)),max_transfer,float(np.max(abs(power[~fixed]),initial=0)),backward,entries)
