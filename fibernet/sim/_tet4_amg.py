"""Internal bounded mechanics iteration and independent cell recovery."""
import math
from dataclasses import dataclass
from inspect import signature
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg
from itertools import islice
from fibernet.sim import tet4 as core

# Bounded conventional engineering range for the vectorized experiment.
# Outside it, the reference path preserves exact safeguards.
def batched_response(mesh,u):
    if np.max(np.abs(u),initial=0.) > .05 or np.any((u!=0.) & (np.abs(u)<1e-18)):
        return core._prescribed_element_response(mesh,u)
    if any(not (1e3<=m['young_modulus_pa']<=1e13 and -.95<=m['poisson_ratio']<=.495)
           for m in mesh.materials.values()):
        return core._prescribed_element_response(mesh,u)
    x=mesh.points*core._LENGTH_TO_M[mesh.length_unit]
    if np.max(np.abs(x),initial=0.)>10.:
        return core._prescribed_element_response(mesh,u)
    force=np.zeros(u.shape)
    energies=[]
    keys,idx=np.unique(mesh.material_ids,return_inverse=True)
    E=np.array([mesh.materials[int(k)]['young_modulus_pa'] for k in keys])
    nu=np.array([mesh.materials[int(k)]['poisson_ratio'] for k in keys])
    mu=E/(2*(1+nu))
    bulk=E/(3*(1-2*nu))
    eye=np.eye(3)
    with np.errstate(divide='raise',over='raise',invalid='raise',under='raise'):
        for start in range(0,len(mesh.tetrahedra),4096):
            sl=slice(start,start+4096); t=mesh.tetrahedra[sl]
            offsets=(x[t[:,1:]]-x[t[:,:1]]).transpose(0,2,1)
            scale=np.max(np.abs(offsets),axis=(1,2))
            if np.any(scale<1e-12):
                return core._prescribed_element_response(mesh,u)
            q=offsets/scale[:,None,None]
            if np.any(np.linalg.cond(q)>core._MAX_ELEMENT_CONDITION):
                raise ValueError('element is too ill-conditioned for this solver')
            vol=np.abs(np.linalg.det(q))*scale**3/6.
            if not np.all(np.isfinite(vol)&(vol>0.)):
                raise ValueError('invalid volume in independent reaction and energy recovery')
            gr=np.linalg.inv(q)/scale[:,None,None]
            grad=np.concatenate([-gr.sum(axis=1,keepdims=True),gr],axis=1)
            # Difference from the anchor node removes exact rigid translation.
            du=u[t[:,1:]]-u[t[:,:1]]
            derivative=np.einsum('eni,enj->eij',du,gr)
            strain=0.5*derivative+0.5*derivative.transpose(0,2,1)
            trace=np.trace(strain,axis1=1,axis2=2)
            dev=strain-trace[:,None,None]*eye/3.
            shear=mu[idx[sl]]
            comp=bulk[idx[sl]]
            stress=2*shear[:,None,None]*dev+(comp*trace)[:,None,None]*eye
            local=vol[:,None,None]*np.einsum('eij,enj->eni',stress,grad)
            cell_e=vol*(shear*np.einsum('eij,eij->e',dev,dev)+0.5*comp*trace*trace)
            if not np.isfinite(local).all() or not np.all(np.isfinite(cell_e)&(cell_e>=0.)):
                raise ValueError('nonfinite or negative independent cell response')
            if np.any(np.any(strain!=0.,axis=(1,2)) & (cell_e<=0.)):
                return core._prescribed_element_response(mesh,u)
            np.add.at(force,t,local)
            energies.append(math.fsum(cell_e))
    core._check_normal_or_zero(force,'independent batched mechanical forces')
    if not np.isfinite(force).all():
        raise ValueError('nonfinite independent mechanical forces')
    return force.ravel(),math.fsum(energies)


@dataclass(frozen=True)
class IterativePlan:
    free: np.ndarray


def chunked_assembly(mesh):
    n = mesh.points.size
    stream = core._element_blocks(mesh)
    stiffness = sparse.csr_matrix((n, n), dtype=np.float64)
    while batch := list(islice(stream, 8192)):
        dofs = np.array([ids for ids, _ in batch], dtype=np.int32)
        coeff = np.array([block for _, block in batch])
        ix, jx = np.broadcast_arrays(dofs[:, :, None], dofs[:, None, :])
        stiffness += sparse.coo_matrix((coeff.ravel(), (ix.ravel(), jx.ravel())),
                                       shape=(n, n)).tocsr()
    stiffness = 0.5*stiffness + 0.5*stiffness.T
    stiffness.eliminate_zeros()
    if not np.isfinite(stiffness.data).all():
        raise ValueError('assembled sparse stiffness is outside supported numerical range')
    return stiffness


def factor_constructor(points, free):
    try:
        import pyamg
    except ImportError as exc:
        raise ImportError('Tet4 solver="amg-cg" requires PyAMG; install FiberNet with its solid-iterative extra') from exc
    coords = points - points[0]
    coords /= np.max(np.abs(coords))
    coords -= coords.mean(axis=0)
    modes = np.zeros((len(points), 3, 6))
    eye = np.eye(3)
    modes[:, :, :3] = eye
    for i in range(3):
        modes[:, :, 3+i] = np.cross(eye[i], coords)
    modes = modes.reshape(-1, 6)[free]
    tolerance_name = 'rtol' if 'rtol' in signature(cg).parameters else 'tol'

    def factor(a, tolerance, plan):
        if not np.array_equal(free, plan.free):
            raise ValueError('inconsistent iterative free-DOF plan')
        diag = a.diagonal()
        if not np.isfinite(diag).all() or np.any(diag <= 0.):
            raise ValueError('invalid iterative mechanical stiffness diagonal')
        scale = 1./np.sqrt(diag)
        S = sparse.diags(scale)
        if np.max(np.diff(a.indptr), initial=0) > 384:
            raise ValueError('iterative Tet4 mechanics matrix connectivity exceeds its row limit')
        matrix = sparse.csr_array(S@a@S)
        if not np.isfinite(matrix.data).all():
            raise ValueError('nonfinite iterative Tet4 stiffness after equilibration')
        B = modes / scale[:, None]
        pre = pyamg.smoothed_aggregation_solver(matrix, B=B, symmetry='symmetric',
                                                max_coarse=96, max_levels=6)
        if not 1. <= pre.operator_complexity() <= 4.:
            raise ValueError('iterative Tet4 mechanics preconditioner resource limit')
        preconditioner = pre.aspreconditioner()

        def solve(rhs):
            x, status = cg(matrix, scale*rhs, M=preconditioner, atol=0.,
                           maxiter=350, **{tolerance_name: min(2e-13, 0.1*tolerance)})
            result = scale*x
            if status or not np.isfinite(result).all():
                raise ValueError('bounded iterative Tet4 mechanics did not converge')
            return result

        return solve
    return factor
