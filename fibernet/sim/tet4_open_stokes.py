# SPDX-License-Identifier: MIT
"""Experimental bounded Stokes flow under an open x-direction pressure drop."""

from dataclasses import dataclass
from inspect import signature
from itertools import combinations
from math import isclose

import numpy as np

from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim import tet4 as core
from fibernet.sim._tet4_amg import topology_parents

__all__ = ["OpenStokesResult", "solve_open_pressure_stokes"]
_NODES = 8000
_TETS = 22000
_DOFS = 125000

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
    tol = "rtol" if "rtol" in signature(minres).parameters else "tol"
    ans,info=minres(sysmat,rhs,M=preconditioner,check=True,maxiter=600,callback=tick,**{tol:1e-13})
    if info != 0: raise ValueError("Open Stokes block iteration did not converge")
    if np.max(abs(sysmat@ans-rhs))>=1e-8*np.max(abs(rhs)):
        raise ValueError("Open Stokes block residual exceeded tolerance")
    z.vec.FV().NumPy()[idx]=ans
    return iterations


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


def solve_open_pressure_stokes(bundle, *, viscosity_Pa_s, pressure_drop_Pa, solver="amg"):
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
    memory use.
    """
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
    unit_Q=factors*width**2
    return OpenStokesResult(b.node_ids.copy(),velocity,pressure,Q*unit_Q,
                            (cap[0]*unit_Q,cap[1]*unit_Q),dp*unit_Q*energy,algebraic,
                            abs(cap[1]/Q-1.),abs(energy/Q-1.),U.ndof,P.ndof,iterations)
