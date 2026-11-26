"""Bounded isotropic DC conduction on an owned Tet4 mesh."""
from collections.abc import Mapping
from dataclasses import dataclass
import math
from inspect import signature

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg
from scipy.sparse import diags

from fibernet.io.tet4_meshio import Tet4Bundle, _validate
from fibernet.sim import tet4 as core

__all__ = ['Tet4DCResult', 'solve_dc_tet4']

_CG_RTOL_NAME = 'rtol' if 'rtol' in signature(cg).parameters else 'tol'
_CG_RELATIVE_TOLERANCE = 1e-11
_CG_MAX_ITERATIONS = 2500


def _locked(array):
    return np.frombuffer(array.tobytes(),dtype=array.dtype).reshape(array.shape)


def _table(data, keys, low, high, name, complete=False):
    if not isinstance(data, Mapping) or len(data)>len(keys):
        raise ValueError(f'expected bounded {name} mapping')
    out={}
    for j,(key,value) in enumerate(data.items()):
        key=core._integer(key,name)
        value=core._float64(value,name)
        if j>=len(keys) or key not in keys or key in out or not low<=value<=high:
            raise ValueError(f'invalid {name} ID or value')
        out[key]=value
    if complete and set(out)!=keys:
        raise ValueError(f'missing {name} IDs')
    return out


@dataclass(frozen=True)
class Tet4DCResult:
    """Detached SI fields; currents are positive into the solid at electrodes."""
    node_ids: np.ndarray
    cell_ids: np.ndarray
    voltage_V: np.ndarray
    centered_voltage_V: np.ndarray
    reference_voltage_V: np.ndarray
    electrode_mask: np.ndarray
    electrode_current_into_solid_A: np.ndarray
    current_density_A_per_m2: np.ndarray
    joule_density_W_per_m3: np.ndarray
    joule_power_W_by_cell: np.ndarray
    terminal_power_W: float
    total_joule_power_W: float
    max_free_residual_A: float
    net_electrode_current_A: float
    power_closure_W: float
    factor_entries: int
    solver: str = "banded"
    iterations: int = 0  # Sum over independently solved active components.


def _check_backward(matrix, rhs, solution, tolerance):
    den = float(abs(matrix).sum(axis=1).max())*float(np.max(np.abs(solution)))
    den += float(np.max(np.abs(rhs)))
    err = float(np.max(np.abs(matrix@solution-rhs)))
    if not np.isfinite(den) or not err <= tolerance*den:
        raise ValueError('DC backward residual')


def _iterative_dc(m, rhs):
    """Diagonal equilibration and bounded CG for the symmetric Dirichlet operator."""
    diagonal = m.diagonal()
    if not np.isfinite(diagonal).all() or np.any(diagonal <= 0.):
        raise ValueError('invalid DC Dirichlet diagonal')
    d = 1. / np.sqrt(diagonal)
    D = diags(d)
    eq = (D @ m @ D).tocsr()
    eq_rhs = d * rhs
    if not np.isfinite(eq.data).all() or not np.isfinite(eq_rhs).all():
        raise ValueError('nonfinite equilibrated DC system')
    it = 0

    def tick(_):
        nonlocal it
        it += 1

    x, status = cg(eq, eq_rhs, atol=0., maxiter=_CG_MAX_ITERATIONS, callback=tick,
                   **{_CG_RTOL_NAME: _CG_RELATIVE_TOLERANCE})
    if status or not np.isfinite(x).all():
        raise ValueError('bounded DC conjugate-gradient solve did not converge')
    return d*x, it


_DC_CELL_BATCH = 8192
_DC_MIN_CELL_VOLUME_M3 = 1e-24
_DC_MAX_CELL_VOLUME_M3 = 1.0
_DC_MAX_GRADIENT_PER_M = 1e12


def _checked_geometry_row(x, cell, row):
    g, volume = core._shape_gradients_and_volume_m(x, cell, row)
    if (not _DC_MIN_CELL_VOLUME_M3 <= volume <= _DC_MAX_CELL_VOLUME_M3
            or not np.isfinite(g).all() or np.max(abs(g)) > _DC_MAX_GRADIENT_PER_M):
        raise ValueError('DC cell outside volume/gradient envelope')
    return g, volume


def _batched_geometry(x, cells):
    """Compute gradients and volumes in blocks; keep scalar cutoffs at the margins."""
    gradients = np.empty((len(cells), 4, 3))
    volume = np.empty(len(cells))
    approach = 1e-10
    for start in range(0, len(cells), _DC_CELL_BATCH):
        sl = slice(start, start + _DC_CELL_BATCH)
        tet = cells[sl]
        d = (x[tet[:, 1:]] - x[tet[:, :1]]).transpose(0, 2, 1)
        scale = np.abs(d).max(axis=(1, 2))
        if not np.all(np.isfinite(scale) & (scale > 0.)):
            row = start + int(np.flatnonzero(~(np.isfinite(scale) & (scale > 0.)))[0])
            raise ValueError(f'Tet4 row {row}: invalid geometry after conversion to m')
        with np.errstate(over='ignore', divide='ignore', invalid='ignore', under='ignore'):
            normalized = d / scale[:, None, None]
            conditions = np.linalg.cond(normalized)
            ordinary = np.isfinite(conditions) & (conditions <= 1e8)
            regular = np.flatnonzero(ordinary)
            grads = gradients[sl]
            vols = volume[sl]
            verts = normalized[regular]
            inverse = np.linalg.inv(verts) / scale[regular, None, None]
            grads[regular, 1:] = inverse
            grads[regular, 0] = -inverse.sum(axis=1)
            vols[regular] = np.abs(np.linalg.det(verts)) * scale[regular]**3 / 6.
            abs_g = np.abs(grads[regular]).max(axis=(1, 2))
            # At or outside a numerical boundary, defer to the original
            # scalar evaluation. That preserves its exact admission rule.
            near_boundary = (
                ~np.isfinite(vols[regular]) | ~np.isfinite(abs_g)
                | (vols[regular] <= _DC_MIN_CELL_VOLUME_M3 * (1. + approach))
                | (vols[regular] >= _DC_MAX_CELL_VOLUME_M3 * (1. - approach))
                | (abs_g >= _DC_MAX_GRADIENT_PER_M * (1. - approach)))
            ordinary[regular[near_boundary]] = False
        for i in np.flatnonzero(~ordinary):
            grads[i], vols[i] = _checked_geometry_row(x, tet[i], start + int(i))
    return gradients, volume


def solve_dc_tet4(bundle, *, conductivity_S_per_m_by_material, electrode_voltage_V_by_id,
                  solver="banded"):
    """Solve fixed-voltage DC conduction with an insulating exterior.

    solver="banded" retains the original direct solver and limits.
    solver="cg" uses bounded, diagonally equilibrated conjugate
    gradients on each active component and does not form a band factor.
    Both paths check recovered current and power before returning SI fields.
    """
    if not isinstance(bundle,Tet4Bundle): raise TypeError("expected a Tet4Bundle")
    if not isinstance(solver,str) or solver not in ('banded','cg'):
        raise ValueError('DC solver must be banded or cg')
    n, e = core._row_count(bundle.points,'points'), core._row_count(bundle.tetrahedra,'tetrahedra')
    if solver == 'banded':
        if n>1024 or e>8000: raise ValueError('DC pilot mesh limits')
        mesh=core._validate_static_mesh(bundle)
    else:
        if n>60_000 or e>200_000: raise ValueError('DC CG mesh limits')
        mesh=_validate(bundle.points, bundle.tetrahedra, bundle.node_ids, bundle.cell_ids,
                       bundle.material_ids, bundle.materials, bundle.length_unit)
        core._validate_packed_topology(mesh.tetrahedra)
    x=mesh.points*core._LENGTH_TO_M[mesh.length_unit]; cells=mesh.tetrahedra
    if np.max(abs(x),initial=0)>1.: raise ValueError('DC coordinate limit')
    s=_table(conductivity_S_per_m_by_material,set(mesh.materials),1e-6,1e6,'DC conductivity',True)
    rows={int(key):i for i,key in enumerate(mesh.node_ids)}
    v=_table(electrode_voltage_V_by_id,set(rows),-100.,100.,'electrode voltage')
    core._check_normal_or_zero(list(v.values()),'electrode V')
    n=len(x); held=np.zeros(n,dtype=bool); absolute=np.zeros(n)
    for key,value in v.items(): held[rows[key]]=True; absolute[rows[key]]=value
    graph=core._free_node_topology(cells,n,np.arange(n))
    component_count,labels=connected_components(graph,directed=False)
    if solver == "cg" and component_count>64:
        raise ValueError("DC CG component limit")
    if set(labels[held])!=set(labels):
        raise ValueError('each DC component needs an electrode node')
    refs=np.empty(n); psi=np.zeros(n); enabled=np.zeros(n,dtype=bool)
    for lab in np.unique(labels):
        ix=labels==lab; ih=ix&held; lo=absolute[ih].min(); hi=absolute[ih].max()
        if hi!=lo and hi-lo<1e-6: raise ValueError('unresolved active electrode span')
        refs[ix]=lo; psi[ih]=absolute[ih]-lo; enabled[ix]=hi!=lo
        # Equal electrodes are allowed; unequal represented levels must stay
        # strictly ordered when transformed to the component gauge.
        levels=np.unique(absolute[ih])
        if (np.any(np.diff(levels-lo)<=0.)
                or np.any((absolute[ih]!=lo)&(psi[ih]==0.))
                or np.any((absolute[ih]!=0.)&((lo+psi[ih])==0.))):
            raise ValueError('electrode contrast lost during centering')
    if solver == 'cg':
        g, vol = _batched_geometry(x, cells)
    else:
        values = [_checked_geometry_row(x, cell, i) for i, cell in enumerate(cells)]
        g, vol = map(np.asarray, zip(*values))
    sig=np.array([s[int(mid)] for mid in mesh.material_ids])
    local=(sig*vol)[:,None,None]*(g@g.transpose(0,2,1))
    rr=np.broadcast_to(cells[:,:,None],local.shape); cc=np.broadcast_to(cells[:,None,:],local.shape)
    A=sparse.coo_matrix((local.ravel(),(rr.ravel(),cc.ravel())),shape=(n,n)).tocsr()
    A=.5*A+.5*A.T; A.eliminate_zeros()
    if not np.isfinite(A.data).all(): raise ValueError('nonfinite DC matrix')
    ids=np.flatnonzero(~held&enabled); cap=np.flatnonzero(held)
    entries = iterations = 0
    if len(ids):
        if solver == 'banded':
            graph=core._free_node_topology(cells,n,ids)
            order=core._canonical_node_order(graph,x[ids],mesh.node_ids[ids])
            ranks=np.empty_like(order); ranks[order]=np.arange(len(order))
            i,j=graph.nonzero()
            h=int(np.max(abs(ranks[i]-ranks[j]),initial=0))
            entries=(2*len(ids)-h)*(h+1)
            plan=core._BandPlan(ids,order,h,entries,len(ids)*(h+1))
            a=A[ids][:,ids]; scale=float(abs(a).sum(axis=1).max())
            if not np.isfinite(scale) or scale<=0: raise ValueError('unresolved DC operator scale')
            m=a/scale; rhs=-np.asarray(A[ids][:,cap]@psi[cap])/scale
            tol=64*np.finfo(np.float64).eps*max(1,len(ids))
            psi[ids]=core._factor_band(m,tol,plan)(rhs)
            _check_backward(m, rhs, psi[ids], tol)
        else:
            for lab in np.unique(labels[ids]):
                block=ids[labels[ids]==lab]
                terminals=cap[labels[cap]==lab]
                a=A[block][:,block]
                scale=float(abs(a).sum(axis=1).max())
                if not np.isfinite(scale) or scale<=0:
                    raise ValueError('unresolved DC operator scale')
                m=a/scale
                rhs=-np.asarray(A[block][:,terminals]@psi[terminals])/scale
                psi[block],count=_iterative_dc(m,rhs)
                iterations += count
                _check_backward(m,rhs,psi[block],_CG_RELATIVE_TOLERANCE)
    V=refs+psi; V[held]=absolute[held]
    if not np.all(np.isfinite(V)&(abs(V)<=100.)): raise ValueError('DC potential outside pilot range')
    grad=np.einsum('ni,nik->nk',psi[cells[:,1:]]-psi[cells[:,:1]],g[:,1:])
    J=-sig[:,None]*grad
    q=sig*np.einsum('ni,ni->n',grad,grad); P=vol*q
    el=-vol[:,None]*np.einsum('nki,ni->nk',g,J)
    nodal=np.zeros(n); np.add.at(nodal,cells,el)
    I=np.where(held,nodal,0.)
    action=A@psi
    if not all(np.isfinite(z).all() for z in (psi,nodal,J,P,q,action)):
        raise ValueError('nonfinite DC recovered fields')
    if np.any((np.any(grad!=0.,axis=1))&((q<=0.)|(P<=0.))):
        raise ValueError('nonzero DC gradient lost dissipated power')
    for z in (J,q,P): core._check_normal_or_zero(z,'recovered DC fields')
    cell_labels=labels[cells[:,0]]
    for lab in np.unique(labels):
        ix=labels==lab; ic=cell_labels==lab; fr=ix&~held
        span=float(np.ptp(psi[ix]))
        if np.max(abs((V[ix]-refs[ix])-psi[ix]),initial=0.)>2e-10*span:
            raise ValueError('unresolved absolute DC voltage reconstruction')
        gross=math.fsum(abs(el[ic]).ravel())
        eps=1e-9*gross
        if max(abs(math.fsum(nodal[ix])),abs(math.fsum(I[ix])),
               float(abs(nodal[fr]).max(initial=0.)),float(abs(nodal[ix]-action[ix]).max(initial=0.)))>eps:
            raise ValueError('DC current consistency')
        energy=math.fsum(P[ic]); term=math.fsum(psi[ix]*I[ix]); virtual=math.fsum(psi[fr]*nodal[fr])
        if abs(math.fsum((energy,-term,-virtual)))>1e-9*energy or abs(energy-term)>abs(virtual)+1e-9*energy:
            raise ValueError('DC terminal power consistency')
    energy=math.fsum(P); term=math.fsum(psi*I); virtual=math.fsum(psi[~held]*nodal[~held])
    return Tet4DCResult(*map(_locked,(mesh.node_ids,mesh.cell_ids,V,psi,refs,held,I,J,q,P)),
                       term,energy,float(np.max(abs(nodal[~held]),initial=0.)),
                       math.fsum(I),math.fsum((energy,-term,-virtual)),entries,solver,iterations)
