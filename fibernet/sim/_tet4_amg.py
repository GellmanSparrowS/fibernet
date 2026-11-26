"""Internal bounded mechanics iteration and independent cell recovery."""
import math
from dataclasses import dataclass
from inspect import signature
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg
from fibernet.sim import tet4 as core

# Bounded engineering range for batched cell recovery.
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


def topology_parents(cells):
    """Validate face incidence and return the reference union order's forest.

    The caller has already bounded and validated the mesh: all point row
    numbers are in [0, 60000). Four unsigned 16-bit values form an injective
    tetrahedron key; three form a face key. Connectivity order is irrelevant
    to equality, but the stable face sort and replay preserve row diagnostics.
    """
    face_keys = core._packed_topology_face_keys(cells)
    order = np.argsort(face_keys, kind='stable')
    shared = core._packed_topology_shared_faces(face_keys[order], cells)
    first, second = order[:-1][shared], order[1:][shared]
    replay = np.argsort(second)
    parent = list(range(len(cells)))

    def find(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for current, previous in zip((second[replay] // 4).tolist(),
                                 (first[replay] // 4).tolist()):
        parent[find(current)] = find(previous)
    return parent


@dataclass(frozen=True)
class IterativePlan:
    free: np.ndarray


_ASSEMBLY_BATCH = 4096


def _element_stiffness_batch(x, tetrahedra, materials, start, allow_fast):
    """Apply the same engineering strain operator to an owned cell block."""
    count = len(tetrahedra)
    local = np.empty((count, 12, 12), dtype=np.float64)
    regular = np.empty(0, dtype=np.intp)
    if allow_fast:
        d = (x[tetrahedra[:, 1:]] - x[tetrahedra[:, :1]]).transpose(0, 2, 1)
        scale = np.max(np.abs(d), axis=(1, 2))
        regular = np.flatnonzero(np.isfinite(scale) & (scale >= 1e-12))
        normalized = d[regular] / scale[regular, None, None]
        standard = np.linalg.cond(normalized) <= 1e8
        regular = regular[standard]
        normalized = normalized[standard]
        volume = np.abs(np.linalg.det(normalized)) * scale[regular]**3 / 6.
        standard = np.isfinite(volume) & (volume > 0.)
        regular = regular[standard]
        normalized, volume = normalized[standard], volume[standard]
        corner = np.linalg.inv(normalized) / scale[regular, None, None]
        grad = np.concatenate((-corner.sum(axis=1, keepdims=True), corner), axis=1)
        strain = np.zeros((len(regular), 6, 12), dtype=np.float64)
        gx, gy, gz = grad.transpose(2, 0, 1)
        strain[:, 0, 0::3], strain[:, 1, 1::3], strain[:, 2, 2::3] = gx, gy, gz
        strain[:, 3, 0::3], strain[:, 3, 1::3] = gy, gx
        strain[:, 4, 1::3], strain[:, 4, 2::3] = gz, gy
        strain[:, 5, 2::3], strain[:, 5, 0::3] = gx, gz
        local[regular] = volume[:, None, None] * (
            strain.transpose(0, 2, 1) @ materials[regular] @ strain)
    scalar = np.ones(count, dtype=bool)
    scalar[regular] = False
    for j in np.flatnonzero(scalar):
        # Preserve the original scalar validation, including its row errors
        # near shape/scale limits, without making other rows take this path.
        grad, volume = core._shape_gradients_and_volume_m(x, tetrahedra[j], start + int(j))
        strain = core._strain_matrix(grad)
        local[j] = volume * (strain.T @ materials[j] @ strain)
    return 0.5 * local + 0.5 * local.transpose(0, 2, 1)


def chunked_assembly(mesh):
    n = mesh.points.size
    x = mesh.points * core._LENGTH_TO_M[mesh.length_unit]
    if not np.isfinite(x).all():
        raise ValueError('point coordinates in m are outside supported range')
    allow_fast = np.max(np.abs(x), initial=0.) <= 10.
    ids, material = np.unique(mesh.material_ids, return_inverse=True)
    tensors = np.stack([core._material_tensor(mesh.materials[int(k)]) for k in ids])
    stiffness = sparse.csr_matrix((n, n), dtype=np.float64)
    for start in range(0, len(mesh.tetrahedra), _ASSEMBLY_BATCH):
        window = slice(start, start + _ASSEMBLY_BATCH)
        cell = mesh.tetrahedra[window]
        dofs = (3 * cell[:, :, None] + np.arange(3)).reshape(-1, 12).astype(np.int32)
        coeff = _element_stiffness_batch(x, cell, tensors[material[window]], start, allow_fast)
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
