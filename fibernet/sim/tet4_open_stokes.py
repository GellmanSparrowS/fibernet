# SPDX-License-Identifier: MIT
"""Experimental bounded Stokes flow under an open x-direction pressure drop."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from itertools import combinations
from math import isclose

import numpy as np

from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim import tet4 as core
from fibernet.sim._tet4_amg import topology_parents

__all__ = [
    "OpenStokesResult", "OpenStokesQuadraticVelocity", "solve_open_pressure_stokes",
    "write_open_stokes_velocity_vtu",
]
_NODES = 8000
_TETS = 22000
_DOFS = 125000

@dataclass(frozen=True)
class OpenStokesQuadraticVelocity:
    """Detached SI P2 samples for affine tetrahedra, suitable for VTK."""
    points_m: np.ndarray
    tetrahedra10: np.ndarray
    velocity_m_s: np.ndarray
    # Every row has either (vertex ID, vertex ID) or the two stable edge endpoint IDs.
    source_node_ids: np.ndarray
    source_cell_ids: np.ndarray


@dataclass(frozen=True)
class OpenStokesResult:
    """Detached SI outputs; nodal arrays are vertex samples of the P2/P1 fields."""
    node_ids: np.ndarray
    velocity_at_vertices_m_s: np.ndarray
    pressure_at_vertices_Pa: np.ndarray
    axial_flow_m3_s: float
    outward_cap_flow_m3_s: tuple
    dissipated_power_W: float
    residual_relative: float
    cap_mass_relative_error: float
    dissipation_relative_error: float
    native_velocity_dofs: int
    native_pressure_dofs: int
    iterations: int
    quadratic_velocity: OpenStokesQuadraticVelocity | None = None

def _amg_solution(n, A, B, U, P, S, z):
    from scipy.sparse import csr_matrix,diags
    from scipy.sparse.linalg import LinearOperator,minres
    val,col,row=A.mat.CSR()
    K=csr_matrix((np.asarray(val),np.asarray(col),np.asarray(row)),shape=(S.ndof,S.ndof))
    # Check the transferred operator, including possible symmetric storage.
    test=z.vec.CreateVector()
    testing=np.linspace(.1,.7,S.ndof)
    test.FV().NumPy()[:]=testing
    out=z.vec.CreateVector();out.data=A.mat*test
    observed=np.asarray(out.FV().NumPy())
    if not np.allclose(K@testing,observed,rtol=1e-10,atol=1e-12):
        K=K+K.T-diags(K.diagonal())
    np.testing.assert_allclose(K@testing,observed,rtol=1e-10,atol=1e-12)
    fixed=np.array(list(S.FreeDofs()),dtype=bool)
    masks=[fixed[list(U.Range(i))] for i in range(3)]
    if not (all(np.array_equal(masks[0],mi) for mi in masks) and all(fixed[list(S.Range(1))])):
        raise ValueError("Unsupported native velocity/pressure DOF layout")
    ui=np.concatenate([np.asarray(list(U.Range(c)))[mask] for c,mask in enumerate(masks)])
    pi=np.asarray(list(S.Range(1)))
    nd=int(masks[0].sum())
    A0=K[ui[:nd],:][:,ui[:nd]].tocsc()
    for i in (1,2):
        Ii=ui[i*nd:(i+1)*nd]
        diff=K[Ii,:][:,Ii]-A0
        if diff.nnz and np.max(abs(diff.data))>=1e-13:
            raise ValueError("Native velocity components do not share one scalar operator")
    try:
        from pyamg import smoothed_aggregation_solver
    except ImportError as exc:
        raise ImportError("The open Stokes AMG solver needs the solid-iterative extra") from exc
    mg=smoothed_aggregation_solver(A0.tocsr(),symmetry="symmetric",max_coarse=40)
    scalar_solver=mg.aspreconditioner(cycle="V")
    p0,q0=P.TnT()
    mass=n.LinearForm(P);mass+=q0*n.dx;mass.Assemble()
    scaling=np.asarray(mass.vec.FV().NumPy())
    if np.min(scaling)<=0: raise ValueError("Open Stokes pressure mass preconditioner is not positive")
    idx=np.r_[ui,pi]
    sysmat=K[idx,:][:,idx].tocsr()
    vel=len(ui)
    def pre(t):
        out=np.empty_like(t)
        for i in range(3):
            sl=slice(i*nd,(i+1)*nd)
            out[sl]=scalar_solver@t[sl]
        out[vel:]=t[vel:]/scaling
        return out
    preconditioner=LinearOperator(sysmat.shape,matvec=pre,dtype=np.float64)
    iterations=0
    def tick(_):
        nonlocal iterations
        iterations+=1
    rhs=np.asarray(B.vec.FV().NumPy())[idx]
    ans = _minres_with_continuity_retry(sysmat, rhs, preconditioner, scaling, vel, tick, minres)
    z.vec.FV().NumPy()[idx]=ans
    return iterations


def _minres_with_continuity_retry(operator, rhs, preconditioner, pressure_mass,
                                   velocity_count, callback, minres):
    from scipy.sparse.linalg import LinearOperator

    tol = "rtol" if "rtol" in signature(minres).parameters else "tol"
    initial, info = minres(operator, rhs, M=preconditioner, check=True,
                           maxiter=600, callback=callback, **{tol: 1e-13})
    if info != 0:
        raise ValueError("Open Stokes block iteration did not converge")
    rhs_scale = np.max(np.abs(rhs))
    defect = operator @ initial - rhs
    if not np.all(np.isfinite(defect)) or np.max(np.abs(defect)) >= 1e-8 * rhs_scale:
        raise ValueError("Open Stokes block residual exceeded tolerance")

    def continuity_error(solution, residual):
        # P1 coefficients for q=1 are all one. Summing the pressure rows
        # gives minus signed outward flow; rhs @ solution is inlet work.
        inlet = float(rhs @ solution)
        return abs(float(residual[velocity_count:].sum())) / inlet if inlet > 0 else np.inf

    original_error = continuity_error(initial, defect)
    if original_error < 7.5e-9:
        return initial

    # Keep the original mixed equations. Boost the constant-pressure
    # direction in the SPD inverse preconditioner only for a bounded retry.
    # The native cap and energy checks remain independent acceptance gates.
    boost = 32.0 / float(pressure_mass.sum())

    def pre_continuity(vector):
        value = preconditioner @ vector
        value[velocity_count:] += boost * float(vector[velocity_count:].sum())
        return value

    retry_pre = LinearOperator(operator.shape, matvec=pre_continuity, dtype=np.float64)
    try:
        repaired, info = minres(operator, rhs, M=retry_pre, check=True,
                                 maxiter=600, callback=callback, **{tol: 1e-15})
    except ValueError:
        # A speculative retry can fail its own MINRES numerical checks.
        # The initial result is still subject to all native acceptance gates.
        return initial
    if info == 0 and np.all(np.isfinite(repaired)):
        new_defect = operator @ repaired - rhs
        if (np.max(np.abs(new_defect)) < 1e-8 * rhs_scale
                and continuity_error(repaired, new_defect) < original_error):
            return repaired
    return initial


def _native_mesh(n, xyz, cells):
    from netgen.meshing import Element2D,Element3D,FaceDescriptor,Mesh,MeshPoint,Pnt
    counts=core._validate_topology(cells)
    m=Mesh(dim=3)
    vertices=[m.Add(MeshPoint(Pnt(*p))) for p in xyz]
    g=xyz[cells]; steps=g[:,1:]-g[:,:1]
    if np.any(np.linalg.cond(steps)>1e6):
        raise ValueError("Ill-conditioned pore cells are not supported")
    determinant=np.linalg.det(steps)
    data=cells.copy()
    flip=determinant>0
    data[flip,:2]=data[flip,:2][:,::-1]       # Netgen volume convention
    opposites={}
    for row in data:
        r=list(map(int,row))
        m.Add(Element3D(1,[vertices[i] for i in r]))
        for face in combinations(sorted(r),3):
            if counts[face]==1:
                opposites[face]=next(i for i in r if i not in face)
    m.SetMaterial(1,"fluid")
    handles=[]
    for index,name in enumerate(("inlet","outlet","walls"),1):
        handles.append(m.Add(FaceDescriptor(surfnr=index,domin=1,domout=0,bc=index)))
        m.SetBCName(index-1,name)
    port_count=np.zeros(3,dtype=int)
    for face,opp in sorted(opposites.items()):
        x=xyz[list(face),0]
        tag=0 if np.all(abs(x)<2e-12) else 1 if np.all(abs(x-1.)<2e-12) else 2
        a,b,c=face
        if np.dot(np.cross(xyz[b]-xyz[a],xyz[c]-xyz[a]),xyz[opp]-xyz[a])>0:
            b,c=c,b
        m.Add(Element2D(handles[tag],[vertices[k] for k in (a,b,c)]))
        port_count[tag]+=1
    if np.min(port_count)<1:
        raise ValueError("Requires planar x-min/x-max cap faces and a no-slip wall")
    mesh=n.Mesh(m)
    if mesh.ne!=len(cells) or mesh.nv!=len(xyz):
        raise ValueError("NGSolve did not preserve the admitted mesh")
    source_rows={tuple(point):i for i,point in enumerate(xyz)}
    if len(source_rows)!=len(xyz):
        raise ValueError("Open Stokes requires unique normalized vertex coordinates")
    native_vertex=np.full(len(xyz),-1,dtype=np.intp)
    for vertex in mesh.vertices:
        row=source_rows.get(tuple(vertex.point))
        if row is None or native_vertex[row]>=0:
            raise ValueError("NGSolve vertices cannot be matched uniquely to input node IDs")
        native_vertex[row]=vertex.nr
    if not np.array_equal(np.sort(native_vertex),np.arange(len(xyz))):
        raise ValueError("NGSolve vertex mapping is not a full bijection")
    return mesh,native_vertex


def _quadratic_velocity_field(n, mesh, w, bundle, native_vertex, xyz, points_m, multiplier):
    # VTK's tetra10 convention uses corners then (01,12,20,03,13,23).
    pairs = np.asarray(((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)))
    reference_corners = np.vstack((np.eye(3), np.zeros(3)))
    r = np.vstack((reference_corners, reference_corners[pairs].mean(axis=1)))
    rule = n.IntegrationRule(r.tolist(), [1.] * len(r))
    integration_points = mesh.MapToAllElements(rule, n.VOL)
    local_coords = np.asarray(n.CoefficientFunction((n.x, n.y, n.z))(integration_points))
    local_coords = local_coords.reshape((-1, 10, 3))
    local_velocity = np.asarray(w(integration_points)).reshape((-1, 10, 3))
    if (local_velocity.shape[0] != len(bundle.tetrahedra)
            or not np.isfinite(local_velocity).all() or not np.isfinite(local_coords).all()):
        raise ValueError("Could not recover finite native P2 velocity per original cell")
    reverse = np.argsort(native_vertex)
    native_corners = np.asarray([[reverse[v.nr] for v in e.vertices] for e in mesh.Elements(n.VOL)])
    distance = np.linalg.norm(local_coords[:, :4, None, :] - xyz[native_corners][:, None, :, :], axis=3)
    order = distance.argmin(axis=2)
    if (np.max(distance.min(axis=2)) > 1e-11
            or np.any(np.sort(order, axis=1) != np.arange(4))):
        raise ValueError("Native P2 reference corners do not match the original coordinates")
    native_corners = np.take_along_axis(native_corners, order, axis=1)
    original = {tuple(c): i for i, c in enumerate(np.sort(bundle.tetrahedra, axis=1))}
    try:
        native_rows = np.asarray([original[tuple(c)] for c in np.sort(native_corners, axis=1)])
    except KeyError as exc:
        raise ValueError("Native P2 tetrahedron cannot be mapped to the source mesh") from exc
    if not np.array_equal(np.sort(native_rows), np.arange(len(bundle.tetrahedra))):
        raise ValueError("Native P2 tetrahedra do not bijectively match source IDs")
    native_edges = np.sort(native_corners[:, pairs], axis=2)
    edges, inverse = np.unique(native_edges.reshape(-1, 2), axis=0, return_inverse=True)
    nv = len(points_m)
    gathered = np.c_[native_corners, inverse.reshape(-1, 6) + nv]
    native_points = np.r_[xyz, xyz[edges].mean(axis=1)]
    if not np.allclose(native_points[gathered], local_coords, rtol=0., atol=1e-11):
        raise ValueError("Native P2 edge midpoint geometry does not match the original mesh")
    values = np.empty_like(native_points)
    values[gathered.ravel()] = local_velocity.reshape(-1, 3)
    tol = 64. * np.finfo(float).eps * np.max(np.abs(local_velocity), initial=0.)
    if not np.allclose(values[gathered], local_velocity, rtol=1e-12, atol=tol):
        raise ValueError("Native P2 velocity is inconsistent on a shared vertex or edge")
    signed = np.linalg.det(xyz[native_corners[:, 1:]] - xyz[native_corners[:, :1]])
    # Swap the first two corners and their corresponding edge positions.
    flip = signed < 0.
    gathered[flip] = gathered[flip][:, [1, 0, 2, 3, 4, 6, 5, 8, 7, 9]]
    cells = np.empty_like(gathered)
    cells[native_rows] = gathered
    points = np.r_[points_m, points_m[edges].mean(axis=1)]
    values *= multiplier
    if not np.isfinite(values).all():
        raise ValueError("Open Stokes quadratic velocity has non-finite physical values")
    # Tet4 validation adaptively uses int64 for IDs below 2**63.  VTU
    # keeps one explicit unsigned type for all validated nonnegative IDs.
    ids = np.asarray(bundle.node_ids, dtype=np.uint64)
    point_source = np.r_[np.repeat(ids[:, None], 2, axis=1), np.sort(ids[edges], axis=1)]
    cid = np.array(bundle.cell_ids, dtype=np.uint64, copy=True)
    for array in (points, cells, values, point_source, cid):
        array.setflags(write=False)
    return OpenStokesQuadraticVelocity(points, cells, values, point_source, cid)


def write_open_stokes_velocity_vtu(path, field):
    """Write a new SI tetra10 VTU without overwriting a prior file.

    The parent directory must exist and support same-filesystem hard links.
    """
    import os
    from pathlib import Path
    from tempfile import TemporaryDirectory

    if not isinstance(field, OpenStokesQuadraticVelocity):
        raise TypeError("Expected an OpenStokesQuadraticVelocity; request it at solve time")
    target = Path(path)
    if target.suffix.lower() != ".vtu":
        raise ValueError("Open Stokes visualization output must use a .vtu path")
    parent = target.parent.resolve(strict=True)
    if not parent.is_dir():
        raise NotADirectoryError(parent)
    target = parent / target.name
    if os.path.lexists(target):
        raise FileExistsError(f"Open Stokes VTU destination already exists: {target}")
    points, cells, velocity, node_ids, cell_ids = (
        np.asarray(a) for a in (field.points_m, field.tetrahedra10, field.velocity_m_s,
                                field.source_node_ids, field.source_cell_ids)
    )
    if (points.ndim != 2 or points.shape[1:] != (3,) or not 10 <= len(points) <= _DOFS // 3
            or velocity.shape != points.shape or points.dtype.kind != "f" or velocity.dtype.kind != "f"
            or not np.isfinite(points).all() or not np.isfinite(velocity).all()
            or cells.ndim != 2 or cells.shape[1:] != (10,) or not 1 <= len(cells) <= _TETS
            or cells.dtype.kind not in "iu" or cells.min() < 0 or cells.max() >= len(points)
            or node_ids.dtype != np.uint64 or node_ids.shape != (len(points), 2)
            or cell_ids.dtype != np.uint64 or cell_ids.shape != (len(cells),)):
        raise ValueError("Open Stokes quadratic field dimensions, identifiers, or values are invalid")
    try:
        import meshio
    except ImportError as exc:
        raise ImportError("The Open Stokes VTU writer needs optional Meshio") from exc
    with TemporaryDirectory(prefix=".open-stokes-vtu-", dir=parent) as scratch:
        tmp = Path(scratch) / "velocity.vtu"
        meshio.write(tmp, meshio.Mesh(points, [("tetra10", cells)],
                     point_data={"velocity_m_s": velocity, "source_node_ids": node_ids},
                     cell_data={"source_cell_id": [cell_ids]}), file_format="vtu")
        if not 0 < tmp.stat().st_size <= 16 * 1024**2:
            raise ValueError("Open Stokes VTU is empty or exceeds the 16 MiB limit")
        try:
            os.link(tmp, target)
        except FileExistsError:
            raise
        except OSError as exc:
            raise OSError("Open Stokes VTU destination must support atomic hard-link publication") from exc


def solve_open_pressure_stokes(bundle, *, viscosity_Pa_s, pressure_drop_Pa, solver="amg",
                               include_quadratic_velocity=False):
    """Solve a fixed-geometry, open pressure-end Tet4 pore.

    The two complete planar x-min/x-max caps are open. All other exterior faces
    have no slip; no other boundary types are supported. The weak gradient
    formulation uses natural pressure loads, not periodic driving, prescribed
    inlet velocity, transient inertia or general Cauchy traction. The applied
    high-x boundary load uses zero pressure as its reference. Positive
    pressure_drop_Pa drives positive x flow.
    Select an installed NGSolve 6.2.2608 CPython 3.12 runtime. This experimental
    API admits at most 8,000 vertices, 22,000 cells, and 125,000 raw P2/P1 DOFs.
    The direct method has a lower, 8,000-cell cap. These do not guarantee peak
    memory use. Set include_quadratic_velocity=True to retain full P2
    corner and edge-midpoint velocity on affine tetrahedra in SI units.
    The default avoids that allocation and does not import Meshio.
    """
    if not isinstance(include_quadratic_velocity, bool):
        raise TypeError("include_quadratic_velocity must be a boolean")
    if not isinstance(bundle,Tet4Bundle):
        raise TypeError("Expected a Tet4Bundle for the pore fluid domain")
    if not isinstance(solver,str) or solver not in ("amg","direct"):
        raise ValueError("Open Stokes solver must be amg or direct")
    nv,ne=(core._row_count(bundle.points,"points"),core._row_count(bundle.tetrahedra,"tetrahedra"))
    if not 4<=nv<=_NODES or not 1<=ne<=_TETS:
        raise ValueError("Open Stokes vertex or tetrahedral admission limit")
    if solver=="direct" and ne>8000:
        raise ValueError("Open Stokes direct solver admits at most 8,000 tetrahedra; use amg")
    mu=core._float64(viscosity_Pa_s,"dynamic viscosity in Pa s")
    dp=core._float64(pressure_drop_Pa,"signed pressure drop in Pa")
    if not 1e-9<=mu<=1e5 or not 1e-9<=abs(dp)<=1e9:
        raise ValueError("Open Stokes viscosity or pressure drop outside the supported range")
    b=core._validate(bundle.points,bundle.tetrahedra,bundle.node_ids,bundle.cell_ids,
                     bundle.material_ids,bundle.materials,bundle.length_unit)
    if len(b.materials)!=1:
        raise ValueError("Open Stokes expects one fluid material group")
    parent=topology_parents(b.tetrahedra)
    if np.unique(b.tetrahedra).size!=nv or sum(i==r for i,r in enumerate(parent))!=1:
        raise ValueError("Open Stokes expects one face-connected fluid body and no unused nodes")
    x_m=b.points*core._LENGTH_TO_M[b.length_unit]
    width=float(np.ptp(x_m[:,0]))
    if not 1e-10<=width<=1.:
        raise ValueError("Open Stokes x span outside the supported range")
    xyz=(x_m-x_m.min(axis=0))/width
    if not np.isfinite(xyz).all() or np.max(xyz)>20.:
        raise ValueError("Open Stokes normalized aspect outside the supported range")
    try:
        import ngsolve as n
    except (ImportError,OSError) as exc:
        raise ImportError("The open Stokes solver needs the NGSolve 6.2.2608 native runtime") from exc
    if n.__version__!="6.2.2608":
        raise ImportError("This open Stokes solver is validated with NGSolve 6.2.2608")
    mesh,native_vertex=_native_mesh(n,xyz,b.tetrahedra)
    U=n.VectorH1(mesh,order=2,dirichlet="walls")
    P=n.H1(mesh,order=1)
    S=n.FESpace([U,P])
    if S.ndof>_DOFS:
        raise ValueError("Open Stokes P2/P1 operator exceeds the supported size")
    u,p=S.TrialFunction()
    v,q=S.TestFunction()
    normal=n.specialcf.normal(3)
    free=np.array(list(S.FreeDofs()),dtype=bool)
    # A geometric cap can have no free P2 velocity trace (e.g. a bare,
    # unrefined triangular cap, with all vertex/edge nodes fixed by walls).
    # Reject both such pressure ports before building a singular mixed solve.
    for name in ("inlet", "outlet"):
        load=n.LinearForm(S)
        load+=-n.InnerProduct(v,normal)*n.ds(name)
        load.Assemble()
        magnitude=float(np.max(np.abs(load.vec.FV().NumPy()[free]),initial=0.))
        area=float(n.Integrate(1,mesh,definedon=mesh.Boundaries(name),order=3))
        if not np.isfinite(magnitude) or not magnitude>64*np.finfo(float).eps*area:
            raise ValueError(f"Open Stokes {name} cap has no free pressure-work velocity trace; refine the cap")
        if name=="inlet":
            rhs,scale=load,magnitude
    A=n.BilinearForm(S,symmetric=True)
    A+=(n.InnerProduct(n.Grad(u),n.Grad(v))-p*n.div(v)-q*n.div(u))*n.dx
    A.Assemble()
    z=n.GridFunction(S)
    if solver=="direct":
        z.vec.data=A.mat.Inverse(S.FreeDofs(),inverse="umfpack")*rhs.vec
        iterations=0
    else:
        iterations=_amg_solution(n,A,rhs,U,P,S,z)
    r=z.vec.CreateVector()
    r.data=A.mat*z.vec-rhs.vec
    algebraic=float(np.max(np.abs(r.FV().NumPy()[free])))/scale
    if not np.isfinite(algebraic) or algebraic>1e-8:
        raise ValueError("Open Stokes assembled residual exceeded tolerance")
    w,pr=z.components
    cap=[]
    for name,sign in (("inlet",-1.),("outlet",1.)):
        bc=mesh.Boundaries(name)
        a=float(n.Integrate(n.InnerProduct(w,normal),mesh,definedon=bc,order=5))
        b_axial=float(n.Integrate(w[0],mesh,definedon=bc,order=5))
        if not isclose(a,sign*b_axial,rel_tol=1e-8):
            raise ValueError("Open Stokes planar cap normal verification failed")
        cap.append(a)
    Q=-cap[0]
    if not (Q>0. and cap[1]>0. and isclose(cap[1],Q,rel_tol=1e-8)):
        raise ValueError("Open Stokes outlet flow and inlet flow do not balance")
    energy=float(n.Integrate(n.InnerProduct(n.Grad(w),n.Grad(w)),mesh,order=5))
    if not isclose(energy,Q,rel_tol=1e-8):
        raise ValueError("Open Stokes pressure work and dissipation do not agree")
    factors=dp*width/mu
    velocity=np.empty_like(xyz)
    for dim,comp in enumerate(w.components):
        indices=[comp.space.GetDofNrs(n.NodeId(n.VERTEX,int(i)))[0] for i in native_vertex]
        velocity[:,dim]=comp.vec.FV().NumPy()[indices]*factors
    indices=[P.GetDofNrs(n.NodeId(n.VERTEX,int(i)))[0] for i in native_vertex]
    pressure=pr.vec.FV().NumPy()[indices]*dp
    if not (np.isfinite(velocity).all() and np.isfinite(pressure).all()):
        raise ValueError("Open Stokes recovered non-finite physical fields")
    quadratic = None
    if include_quadratic_velocity:
        quadratic = _quadratic_velocity_field(n, mesh, w, b, native_vertex, xyz, x_m, factors)
        if not np.allclose(quadratic.velocity_m_s[:nv], velocity, rtol=1e-12, atol=0.):
            raise ValueError("Open Stokes P2 export failed the native vertex consistency check")
    unit_Q=factors*width**2
    return OpenStokesResult(b.node_ids.copy(),velocity,pressure,Q*unit_Q,
                            (cap[0]*unit_Q,cap[1]*unit_Q),dp*unit_Q*energy,algebraic,
                            abs(cap[1]/Q-1.),abs(energy/Q-1.),U.ndof,P.ndof,iterations,quadratic)
