# SPDX-License-Identifier: MIT
"""Bounded steady incompressible Stokes flow in an x-periodic Tet4 pore."""
from __future__ import annotations

from dataclasses import dataclass
from inspect import signature

import numpy as np
from scipy.sparse import bmat, coo_matrix, eye, hstack, kron
from scipy.sparse.linalg import LinearOperator, minres, splu
from scipy.spatial import cKDTree

from ..io.tet4_meshio import Tet4Bundle
from . import tet4 as core

__all__ = [
    "PeriodicStokesResult",
    "MatchedPoreFlow",
    "solve_x_periodic_stokes",
    "solve_matched_pore_geometry",
    "solve_matched_pore_geometry_cases",
]

@dataclass(frozen=True)
class PeriodicStokesResult:
    axial_flow_m3_s: float
    cap_flow_m3_s: tuple[float, float]
    power_W: float
    power_relative_error: float
    original_relative_residual: float
    continuity_relative_error: float
    cap_pair_relative_error: float
    cap_volume_relative_error: float
    periodic_vertex_pairs: int
    velocity_dofs: int
    planned_workspace_bytes: int

def _validate_pair(mesh, length):
    pts=mesh.p.T
    facets=mesh.boundary_facets()
    tri=mesh.facets[:,facets].T
    low=np.all(np.abs(pts[tri,0])<2e-12,axis=1)
    high=np.all(np.abs(pts[tri,0]-length)<2e-12,axis=1)
    left=np.unique(tri[low].ravel())
    right=np.unique(tri[high].ravel())
    if len(left)<3 or len(left)!=len(right):
        raise ValueError('periodic cap vertices are missing or asymmetric')
    err,j=cKDTree(pts[left,1:]).query(pts[right,1:])
    if err.max()>2e-11 or len(np.unique(j))!=len(left):
        raise ValueError('periodic cap vertices do not pair bijectively')
    lut=np.arange(len(pts));lut[right]=left[j]
    def sorted_faces(faces):
        return sorted(map(tuple,np.sort(faces,axis=1).tolist()))
    if sorted_faces(tri[low])!=sorted_faces(lut[tri[high]]):
        raise ValueError('periodic cap triangulations do not match')
    if not np.any(~(low|high)):
        raise ValueError('periodic Stokes requires no-slip walls')
    return facets[low],facets[high],facets[~(low|high)],len(left)

_MIB=1024**2
_MAX_NODES=2300
_MAX_TETS=3400
_MAX_DOF=14000
# Charge the one retained factor, row layout and warm vector separately
# from the normal per-case work estimate, including during a cold retry.
_REFERENCE_RESERVE_BYTES=32*_MIB
_REFERENCE_FACTOR_NNZ=1_000_000
_REFERENCE_MAX_NORMALIZED_MOTION=.01


@dataclass
class _PeriodicReference:
    layout: tuple[bytes, ...] | None = None
    normalized_points: np.ndarray | None = None
    factor: object = None
    seed: np.ndarray | None = None

    def can_reuse(self, layout, points):
        return (self.layout is not None and self.layout == layout
                and np.max(np.abs(points-self.normalized_points), initial=0.)
                <= _REFERENCE_MAX_NORMALIZED_MOTION)



def _stokes_admission(bundle,cap_bytes):
    if not isinstance(bundle,Tet4Bundle):
        raise TypeError('expected a Tet4Bundle for the pore domain')
    if isinstance(cap_bytes,(bool,np.bool_)) or not isinstance(cap_bytes,(int,np.integer)):
        raise TypeError('periodic Stokes workspace limit must be integer bytes')
    if not 64*_MIB<=cap_bytes<=512*_MIB:
        raise ValueError('periodic Stokes workspace limit outside supported range')
    n=core._row_count(bundle.points,'points')
    t=core._row_count(bundle.tetrahedra,'tetrahedra')
    if not 4<=n<=_MAX_NODES or not 1<=t<=_MAX_TETS:
        raise ValueError('periodic Stokes point/cell admission limit')
    # Modeled admission separate from allocator/actual high-water memory.
    plan=192*_MIB+62000*t+8192*n
    if plan>cap_bytes:
        raise ValueError('periodic Stokes planned work buffers exceed requested limit')
    return n,t,plan

def _solve_stokes_block(system, rhs, velocity_dofs, *, initial=None,
                        velocity_factor=None, capture=None):
    """Try MINRES with one direct scalar velocity solve and a pressure diagonal."""
    n = system.shape[0]
    if not 3 <= velocity_dofs < n or velocity_dofs % 3:
        raise ValueError("periodic Stokes block layout is not supported")
    diag = system.diagonal()
    vel_diag = diag[:velocity_dofs]
    if np.any(~np.isfinite(vel_diag)) or np.any(vel_diag <= 0.) or np.any(diag[velocity_dofs:]):
        raise ValueError("periodic Stokes block diagonal is not supported")
    velocity = system[:velocity_dofs, :velocity_dofs]
    velocity.eliminate_zeros()
    scalar = velocity[::3, ::3].tocsc()
    if (velocity.nnz != 3 * scalar.nnz
            or (velocity[1::3, 1::3] - scalar).nnz
            or (velocity[2::3, 2::3] - scalar).nnz):
        raise ValueError("periodic Stokes velocity components are not identical")
    # -B A^{-1} B^T is the pressure Schur block; use a positive diagonal
    # approximation to its magnitude so the MINRES preconditioner is SPD.
    coupling = system[velocity_dofs:, :velocity_dofs].tocsr()
    pressure_diag = coupling.power(2) @ (1. / vel_diag)
    if np.any(~np.isfinite(pressure_diag)) or np.any(pressure_diag <= 0.):
        raise ValueError("periodic Stokes pressure diagonal is not supported")
    scalar_factor = velocity_factor
    if scalar_factor is None:
        scalar_factor = splu(scalar, permc_spec="MMD_AT_PLUS_A", options={"SymmetricMode": True})

    def apply(x):
        v = scalar_factor.solve(x[:velocity_dofs].reshape((-1, 3))).ravel()
        return np.r_[v, x[velocity_dofs:] / pressure_diag]

    preconditioner = LinearOperator(system.shape, matvec=apply, dtype=np.float64)
    # SciPy 1.12 uses tol; newer supported releases use rtol.
    tolerance = "rtol" if "rtol" in signature(minres).parameters else "tol"
    warm = {} if initial is None else {"x0": initial.copy()}
    x, info = minres(system, rhs, M=preconditioner, maxiter=300, check=True,
                     **warm, **{tolerance: 2e-13})
    error = np.linalg.norm(system @ x - rhs, np.inf)
    scale = np.linalg.norm(rhs, np.inf)
    if info != 0 or not np.isfinite(error) or not error <= 1e-9 * scale:
        raise ValueError("periodic Stokes block iteration did not reach the algebraic tolerance")
    if capture is not None and scalar_factor.nnz <= _REFERENCE_FACTOR_NNZ:
        capture.extend((scalar_factor, x.copy()))
    return x


def _x_period_m(x_m):
    width = float(np.ptp(x_m))
    if not 1e-10 <= width <= 1.:
        raise ValueError("invalid x period in metres")
    return width


def _in_gradient_range(gradient):
    return 1e-9 <= abs(gradient) <= 1e12


def solve_x_periodic_stokes(
    bundle: Tet4Bundle,
    *,
    viscosity_Pa_s: float,
    pressure_gradient_Pa_per_m: float,
    workspace_limit_bytes: int = 512 * _MIB,
) -> PeriodicStokesResult:
    """Periodic fluctuation pressure, signed unit-aligned driving, and SI flux.

    A uniform body force represents the prescribed axial pressure gradient.
    The sign is the force exerted on the fluid in +x, i.e. minus dp/dx.
    """
    return _solve_x_periodic_stokes(bundle, viscosity_Pa_s=viscosity_Pa_s,
                                    pressure_gradient_Pa_per_m=pressure_gradient_Pa_per_m,
                                    workspace_limit_bytes=workspace_limit_bytes)


def _solve_x_periodic_stokes(bundle, *, viscosity_Pa_s, pressure_gradient_Pa_per_m,
                             workspace_limit_bytes, reference=None, record_reference=False):
    n,t,plan=_stokes_admission(bundle,workspace_limit_bytes)
    visc=core._float64(viscosity_Pa_s,'dynamic viscosity in Pa s')
    pg=core._float64(pressure_gradient_Pa_per_m,'signed driver in Pa/m')
    if not 1e-9<=visc<=1e5 or not _in_gradient_range(pg):
        raise ValueError('fluid viscosity or axial pressure gradient outside pilot range')
    m=core._validate(bundle.points,bundle.tetrahedra,bundle.node_ids,bundle.cell_ids,
                     bundle.material_ids,bundle.materials,bundle.length_unit)
    core._validate_topology(m.tetrahedra)
    try:
        from skfem import (Basis,BilinearForm,ElementTetP1,ElementTetP2,FacetBasis,
                           LinearForm,MeshTet,asm)
        from skfem.models.poisson import laplace
    except ImportError as e:
        raise ImportError('periodic Stokes requires scikit-fem; install FiberNet with its pore-flow extra') from e
    xyz=m.points*core._LENGTH_TO_M[m.length_unit]
    width=_x_period_m(xyz[:,0])
    if not np.all(np.isfinite(xyz)):
        raise ValueError('invalid x period in metres')
    x=(xyz-xyz.min(axis=0))/width
    if np.max(x)>20.:
        raise ValueError('periodic Stokes cell aspect outside pilot')
    q=x[m.tetrahedra]
    jac=q[:,1:]-q[:,:1]
    if np.any(np.linalg.cond(jac)>1e6):
        raise ValueError('ill-conditioned pore cells are not supported')
    mesh=MeshTet(x.T.copy(),m.tetrahedra.T.copy())
    left,right,walls,npair=_validate_pair(mesh,1.)
    if np.unique(mesh.t).size != n:
        raise ValueError('periodic Stokes requires all nodes to be used')
    # Disconnected pore bodies introduce extra pressure null spaces.
    sides=mesh.f2t[:,np.all(mesh.f2t>=0,axis=0)]
    graph=coo_matrix((np.ones(sides.shape[1]),(sides[0],sides[1])),shape=(t,t)).tocsr()
    from scipy.sparse.csgraph import connected_components
    if connected_components(graph,directed=False,return_labels=False)!=1:
        raise ValueError('periodic Stokes disconnected pore body')
    sv=Basis(mesh,ElementTetP2(),intorder=4)
    bp=Basis(mesh,ElementTetP1(),intorder=4)
    nv=3*sv.N
    if nv+bp.N>_MAX_DOF:
        raise ValueError('periodic Stokes reduced-operator size exceeds pilot')
    A=kron(asm(laplace,sv),eye(3,format='csr'),format='csr')
    def derivative(axis):
        @BilinearForm
        def form(u,q,_w):
            return u.grad[axis]*q
        return asm(form,sv,bp)
    axes=[np.arange(q,nv,3) for q in range(3)]
    index=np.arange(nv).reshape(3,sv.N).T.ravel()
    B=hstack([derivative(i) for i in range(3)],format='csr')[:,index]
    B.sort_indices()
    M=bmat([[A,-B.T],[-B,None]],format='csr')
    @LinearForm
    def body(v,_w):
        return v
    f0=np.zeros(nv)
    f0[::3]=asm(body,sv)
    rhs=np.r_[f0,np.zeros(bp.N)]
    rho=np.arange(len(rhs))
    forward=np.arange(len(rhs))
    for pos,offset,stride,count in ((sv.doflocs,0,3,sv.N),(bp.doflocs,nv,1,bp.N)):
        lo=np.flatnonzero(np.abs(pos[0])<2e-12)
        hi=np.flatnonzero(np.abs(pos[0]-1.)<2e-12)
        if len(lo)!=len(hi):
            raise ValueError('periodic P2/P1 nodes do not pair')
        err,nearest=cKDTree(pos[1:,lo].T).query(pos[1:,hi].T)
        if np.max(err)>2e-11 or len(np.unique(nearest))!=len(lo):
            raise ValueError('periodic P2/P1 node coordinates do not pair')
        for j in range(stride):
            forward[stride*hi+j+offset]=stride*lo[nearest]+j+offset
    _,inverse=np.unique(forward,return_inverse=True)
    P=coo_matrix((np.ones(len(rhs)),(rho,inverse)),shape=(len(rhs),int(inverse.max())+1)).tocsr()
    matrix=(P.T@M@P).tocsr()
    reduced=P.T@rhs
    held=np.unique(np.r_[inverse[(3*sv.get_dofs(facets=walls).all()[:,None]+np.arange(3)).ravel()],inverse[nv]])
    free=np.setdiff1d(np.arange(matrix.shape[0]),held)
    system=matrix[free][:,free].tocsc()
    if not len(free):
        raise ValueError('periodic Stokes has no free degrees of freedom')

    # Mesh rows need not put pressure node zero on the low cap. Find the
    # end of the entire reduced velocity block, not the pressure gauge index.
    velocity_dofs = int(np.searchsorted(free, inverse[:nv].max() + 1))

    layout = None
    if reference is not None:
        plan += _REFERENCE_RESERVE_BYTES
        for_layout = (free, inverse, sv.element_dofs, bp.element_dofs,
                      m.node_ids, m.cell_ids, m.tetrahedra)
        layout = tuple(np.asarray(a).tobytes() for a in for_layout)
    capture = [] if record_reference else None

    def solve_and_check(mode):
        u=np.zeros(len(reduced))
        try:
            if mode == 'warm-block':
                u[free] = _solve_stokes_block(system, reduced[free], velocity_dofs,
                                             initial=reference.seed, velocity_factor=reference.factor)
            elif mode == 'block':
                if capture is not None:
                    u[free] = _solve_stokes_block(system, reduced[free], velocity_dofs, capture=capture)
                else:
                    u[free] = _solve_stokes_block(system, reduced[free], velocity_dofs)
            else:
                kwargs = {'options': {'SymmetricMode': True}} if mode == 'structured-lu' else {}
                u[free] = splu(system, permc_spec='MMD_AT_PLUS_A', **kwargs).solve(reduced[free])
        except RuntimeError as e:
            raise ValueError('periodic Stokes operator cannot be factorized') from e
        reconstructed=P@u
        vel=reconstructed[:nv]
        res=np.max(np.abs((matrix@u-reduced)[free]))/np.max(np.abs(reduced))
        flux=float(f0@vel)
        E=float(vel@(A@vel))
        cont=float(np.max(np.abs((P.T@np.r_[np.zeros(nv),B@vel]))))/np.max(np.abs(reduced))
        caps=np.array([float(asm(body,FacetBasis(mesh,ElementTetP2(),
                                                  facets=cap,intorder=4))@vel[::3]) for cap in (left,right)])
        if not np.isfinite(u).all() or not np.isfinite([res,flux,E,cont,*caps]).all() or flux<=0:
            raise ValueError('periodic Stokes nonfinite or nonpositive normalized flow')
        cost=E/flux-1.
        capbalance=float(abs(caps[1]-caps[0])/flux)
        # In discrete weak incompressibility, x is not a periodic pressure
        # test function. Matching cap fluxes and equality of bulk/cap flux
        # are distinct; the latter is a mesh diagnostic, not a linear residual.
        capdelta=float(np.max(np.abs(caps/flux-1.)))
        # Velocity scale = g L^2 / mu, flux adds L^2; signed g is restored once.
        flux_scale=pg*width**4/visc
        power_scale=pg**2*width**5/visc
        if (res>2e-8 or cont>2e-8 or abs(cost)>2e-8 or capbalance>1e-9 or capdelta>1e-2 or
            np.any(~np.isfinite([flux_scale,power_scale]))):
            raise ValueError(f'periodic Stokes failed residual, flow or work closure: {res=}, {cont=}, {cost=}, {capdelta=}')
        return PeriodicStokesResult(flux*flux_scale,tuple(map(float,caps*flux_scale)),E*power_scale,cost,
                                    float(res),float(cont),capbalance,capdelta,int(npair),int(nv),plan)

    # The saddle operator is symmetric but indefinite. Test each candidate
    # against the same physical gates, and keep both direct LU paths.
    if reference is not None and not record_reference and reference.can_reuse(layout, x):
        try:
            return solve_and_check('warm-block')
        except ValueError:
            # Always try a cold solve on this geometry before the direct fallbacks.
            pass
    for mode in ('block', 'structured-lu'):
        try:
            result = solve_and_check(mode)
            if record_reference and mode == 'block' and capture:
                reference.layout = layout
                reference.normalized_points = x
                reference.factor, reference.seed = capture
            return result
        except ValueError:
            if capture:
                capture.clear()
    return solve_and_check('original-lu')


@dataclass(frozen=True)
class MatchedPoreFlow:
    baseline: PeriodicStokesResult
    deformed: PeriodicStokesResult
    relative_bulk_flow_change: float
    relative_cap_flow_change: tuple[float, float]
    min_cell_volume_ratio: float
    max_cell_volume_ratio: float

def _validated_matched_mesh(bundle, budget):
    _stokes_admission(bundle, budget)
    return core._validate(bundle.points, bundle.tetrahedra, bundle.node_ids,
                          bundle.cell_ids, bundle.material_ids, bundle.materials,
                          bundle.length_unit)


def _matched_volume_range(a, b):
    def aligned_rows(old,new):
        k=np.argsort(new); new_sorted=new[k]
        if not np.array_equal(np.sort(old),new_sorted):
            raise ValueError('matched geometry requires identical stable node and cell IDs')
        return k[np.searchsorted(new_sorted,old)]
    nodes=aligned_rows(a.node_ids,b.node_ids)
    cells=aligned_rows(a.cell_ids,b.cell_ids)
    # Topology comparison via stable IDs rather than incidental row orders.
    if not np.array_equal(a.node_ids[a.tetrahedra],b.node_ids[b.tetrahedra[cells]]):
        raise ValueError('matched geometry requires identical ordered tetrahedra')
    if not np.array_equal(a.material_ids,b.material_ids[cells]) or a.materials!=b.materials:
        raise ValueError('matched geometry uses the same pore labels and fluid side')
    x=a.points*core._LENGTH_TO_M[a.length_unit]
    y=b.points[nodes]*core._LENGTH_TO_M[b.length_unit]
    def determinant(positions):
        P=positions[a.tetrahedra]
        return np.linalg.det(P[:,1:]-P[:,:1])
    raw=determinant(x); fresh=determinant(y)
    if not np.isfinite([raw,fresh]).all() or np.any(raw==0.) or np.any(fresh==0.):
        raise ValueError('matched geometry contains collapsed or nonfinite cells in SI')
    ratio=fresh/raw
    lo=float(ratio.min());hi=float(ratio.max())
    if lo<.75 or hi>1.25:
        raise ValueError('matched geometry reverses a cell or exceeds the pilot volume-ratio envelope')
    return lo, hi


def solve_matched_pore_geometry(reference, reshaped, *, viscosity_Pa_s,
                                pressure_gradient_Pa_per_m=None, pressure_drop_Pa=None,
                                workspace_limit_bytes=512*_MIB):
    """Compare a fixed geometry pair at an equal pressure gradient or drop."""
    return solve_matched_pore_geometry_cases(
        reference, (reshaped,), viscosity_Pa_s=viscosity_Pa_s,
        pressure_gradient_Pa_per_m=pressure_gradient_Pa_per_m,
        pressure_drop_Pa=pressure_drop_Pa,
        workspace_limit_bytes=workspace_limit_bytes,
    )[0]


def solve_matched_pore_geometry_cases(reference, reshaped_cases, *, viscosity_Pa_s,
                                      pressure_gradient_Pa_per_m=None, pressure_drop_Pa=None,
                                      workspace_limit_bytes=512*_MIB):
    """Reuse reference Stokes flow for 1–8 ordered fixed-geometry cases.

    Each case uses the same stable mesh IDs, ordered tetrahedra and pore
    labels and shared dynamic viscosity (Pa s). Specify exactly one signed
    driver: ``pressure_gradient_Pa_per_m`` is shared directly;
    ``pressure_drop_Pa`` is divided by each mesh's x period in metres.
    Positive values drive flow toward +x. The usual mesh, workspace,
    volume-ratio and flow closure checks apply independently to every state.

    Return a tuple in input order, or raise without returning a partial
    result. The immutable reference flow is shared within the tuple and
    is not kept as a cache. This is not transient ALE or two-way coupling.
    """
    if type(reshaped_cases) not in (list, tuple) or not 1 <= len(reshaped_cases) <= 8:
        raise ValueError("reshaped_cases must be an ordered list or tuple of 1 to 8 geometries")
    if (pressure_gradient_Pa_per_m is None) == (pressure_drop_Pa is None):
        raise ValueError("specify exactly one pressure_gradient_Pa_per_m or pressure_drop_Pa")
    a = _validated_matched_mesh(reference, workspace_limit_bytes)
    candidates = []
    for geom in tuple(reshaped_cases):
        b = _validated_matched_mesh(geom, workspace_limit_bytes)
        bounds = _matched_volume_range(a, b)
        candidates.append((b, bounds))

    meshes = (a, *(mesh for mesh, _ in candidates))
    gradients = (pressure_gradient_Pa_per_m,) * len(meshes)
    if pressure_drop_Pa is not None:
        drop = core._float64(pressure_drop_Pa, "signed pressure drop in Pa")
        computed = []
        for mesh in meshes:
            width = _x_period_m(mesh.points[:, 0] * core._LENGTH_TO_M[mesh.length_unit])
            gradient = drop / width
            if not _in_gradient_range(gradient):
                raise ValueError("pressure drop implies an axial pressure gradient outside pilot range")
            computed.append(gradient)
        gradients = tuple(computed)
    settings = dict(viscosity_Pa_s=viscosity_Pa_s,
                    workspace_limit_bytes=workspace_limit_bytes)
    can_reserve = len(candidates) > 1 and all(
        _stokes_admission(mesh, workspace_limit_bytes)[2] + _REFERENCE_RESERVE_BYTES <= workspace_limit_bytes
        for mesh in meshes
    )
    seed = _PeriodicReference() if can_reserve else None
    if seed is None:
        q0 = solve_x_periodic_stokes(a, pressure_gradient_Pa_per_m=gradients[0], **settings)
    else:
        q0 = _solve_x_periodic_stokes(a, pressure_gradient_Pa_per_m=gradients[0],
                                     **settings, reference=seed, record_reference=True)
    output = []
    for (mesh, (lo, hi)), driver in zip(candidates, gradients[1:]):
        if seed is None:
            q1 = solve_x_periodic_stokes(mesh, pressure_gradient_Pa_per_m=driver, **settings)
        else:
            q1 = _solve_x_periodic_stokes(mesh, pressure_gradient_Pa_per_m=driver,
                                         **settings, reference=seed)
        delta = tuple(p/q-1. for p, q in zip(q1.cap_flow_m3_s, q0.cap_flow_m3_s))
        output.append(MatchedPoreFlow(q0, q1, q1.axial_flow_m3_s/q0.axial_flow_m3_s-1., delta, lo, hi))
    return tuple(output)
