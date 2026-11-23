"""Experimental one-way, small-strain bulk Tet4 with prescribed temperature."""
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Optional
import math
import numpy as np

from fibernet.io.tet4_meshio import Tet4Bundle
from . import tet4 as core

__all__ = ["Tet4ThermoelasticResult", "solve_prescribed_temperature_tet4"]

# Numerical pilot envelope, not universal material or physical validity thresholds.
_FORCE_LIMIT_N=1e8
_MIN_NONZERO_ALPHA=1e-10
_MAX_GRADIENT_PER_M=1e12
_CANCELLATION_FRACTION=1e-10
@dataclass(frozen=True)
class Tet4ThermoelasticResult:
    """Detached SI fields in stable node or cell ID order.

    Reactions are physical equilibrium residuals on held and prescribed
    components, zero on free components. Strains are tensors, stresses
    are tensile-positive cell means, and stored energy includes the exact
    P1 within-cell temperature variance. The terminal mechanical pairing is
    (applied nodal force + support reaction)·u, not cumulative work.
    Arrays are backed by immutable bytes; use array.copy() to edit.
    """
    node_ids: np.ndarray
    cell_ids: np.ndarray
    material_ids: np.ndarray
    displacement_m: np.ndarray
    fixed_mask: np.ndarray
    reaction_N: np.ndarray
    total_strain: np.ndarray
    mean_elastic_strain: np.ndarray
    mean_stress_Pa: np.ndarray
    elastic_energy_J_by_cell: np.ndarray
    elastic_energy_J: float
    terminal_mechanical_pairing_J: float
    stress_eigenstrain_integral_J: float
    physical_free_residual_inf_N: float

def _frozen_array(a):
    return np.frombuffer(a.tobytes(order='C'),dtype=a.dtype).reshape(a.shape)

def _effective_force(ext,thermal):
    if max(np.max(abs(ext)),np.max(abs(thermal)))>_FORCE_LIMIT_N:
        raise ValueError('mechanical or thermal nodal load exceeds prototype force scale')
    effective=ext+thermal
    if np.max(abs(effective))>_FORCE_LIMIT_N:
        raise ValueError('combined nodal load exceeds prototype force scale')
    both=(thermal!=0)&(ext!=0)
    if np.any(both&((effective==ext)|(effective==thermal))):
        raise ValueError('effective RHS addition lost a nonzero contribution')
    opposing=both&(np.signbit(ext)!=np.signbit(thermal))
    # Exact opposite represented binary64 operands are a well-defined zero.
    # Other opposing pairs must resolve their nonzero residual; independent
    # physical residual/reaction checks still apply to exact cancellation.
    exact_opposites=opposing&(ext==-thermal)
    if np.any(opposing&~exact_opposites&(abs(effective)<=_CANCELLATION_FRACTION*np.maximum(abs(ext),abs(thermal)))):
        raise ValueError('mechanical and eigenstrain loads nearly cancel')
    core._check_normal_or_zero(effective,'effective nodal force')
    return effective

def _detach_mapping(value, cap, label, normalize, key=lambda k:k):
    if not isinstance(value, Mapping) or len(value)>cap:
        raise ValueError(f'{label}: expected bounded mapping')
    out={}
    for i,(k,v) in enumerate(value.items()):
        k=key(k)
        if i>=cap or k in out:
            raise ValueError(f'{label}: oversized or duplicate mapping')
        out[k]=normalize(v)
    return out

def _detach_force(v):
    if isinstance(v,np.ndarray):
        if v.shape!=(3,): raise ValueError('nodal force must be a 3-vector')
        return v.copy()
    if isinstance(v,(list,tuple)):
        if len(v)!=3: raise ValueError('nodal force must be a 3-vector')
        return tuple(v)
    return v

def _detach_hold(v):
    if isinstance(v,(list,tuple)):
        if not 1<=len(v)<=3: raise ValueError('invalid held axes')
        return tuple(v)
    return v

def _canonical_controls(node_ids,fixed,prescribed):
    # Owner collapses signed/explicit-zero prescribed values into ordinary
    # fixed axes; preserve exactly that trace, not the caller's syntactic form.
    values=np.zeros(fixed.shape) if prescribed is None else prescribed.reshape(fixed.shape)
    holds={}; moves={}
    for j,key in enumerate(node_ids):
        held=tuple(a for i,a in enumerate('xyz') if fixed[j,i] and values[j,i]==0.)
        moved={a:float(values[j,i]) for i,a in enumerate('xyz') if values[j,i]!=0.}
        if held: holds[int(key)]=held
        if moved: moves[int(key)]=moved
    return holds,moves or None

def solve_prescribed_temperature_tet4(
    bundle: Tet4Bundle,
    *,
    temperature_K_by_id: Mapping,
    stress_free_temperature_K: float,
    alpha_per_K_by_material: Mapping,
    forces_by_id: Mapping,
    fixed_dofs_by_id: Mapping,
    prescribed_displacements_m_by_id: Optional[Mapping] = None,
) -> Tet4ThermoelasticResult:
    """Solve an experimental, one-way thermoelastic snapshot.

    Supply an absolute Kelvin value per stable node, a common stress-free
    Kelvin reference, and an expansion coefficient per declared material
    ID in inverse kelvin. Temperature is affine within each tetrahedron;
    each cell is homogeneous in E, ν, and isotropic α. Nodal forces are in
    N and prescribed displacements in m, independent of the geometry unit.

    In addition to static Tet4 size, topology, and numerical guards, this
    pilot admits only K in [1, 2000], α zero or magnitude in [1e-10, 1e-3]
    per K, coordinates within ±1 m per component, cell volumes in
    [1e-24, 1] m³, E in [1e3, 1e13] Pa, ν in [-0.5, 0.45], thermal
    endpoint strain within ±0.01, shape-gradient components within ±1e12/m,
    mechanical, equivalent thermal and combined forces within ±1e8 N,
    prescribed components within ±0.01 m, and engineering strain
    components within ±0.02. Unresolved load cancellation is rejected.
    These are numerical pilot limits, not universal physical limits.

    Twice stored elastic energy equals the terminal pairing minus
    the stress–eigenstrain integral, up to numerical tolerance. This
    does not solve heat transfer or estimate interface/peak stresses.
    """
    mesh = core._validate_static_mesh(bundle)
    result, _ = _solve_prescribed_temperature_tet4_on_validated_mesh(
        mesh, temperature_K_by_id=temperature_K_by_id,
        stress_free_temperature_K=stress_free_temperature_K,
        alpha_per_K_by_material=alpha_per_K_by_material,
        forces_by_id=forces_by_id, fixed_dofs_by_id=fixed_dofs_by_id,
        prescribed_displacements_m_by_id=prescribed_displacements_m_by_id,
    )
    return result


def _solve_prescribed_temperature_tet4_on_validated_mesh(
    mesh, *, temperature_K_by_id, stress_free_temperature_K,
    alpha_per_K_by_material, forces_by_id, fixed_dofs_by_id,
    prescribed_displacements_m_by_id,
):
    """Return solver fields and the same owned absolute temperatures it used."""
    X_m=mesh.points*core._LENGTH_TO_M[mesh.length_unit]
    if not np.all(np.isfinite(X_m)) or np.max(abs(X_m))>1.:
        raise ValueError('prototype coordinate scale exceeds 1 m')

    ids={int(k):i for i,k in enumerate(mesh.node_ids)}
    mats={int(k):i for i,k in enumerate(mesh.materials)}
    def required(mapping, index, label, lo, hi):
        stable = lambda k: core._integer(k, label)
        owned = _detach_mapping(
            mapping, len(index), label, lambda z: core._float64(z, label), stable
        )
        a=np.full(len(index),np.nan)
        for i,z in core._row_lookup(owned,index,label):
            a[i]=core._float64(z,label)
        if not np.all((lo<=a)&(a<=hi)):
            raise ValueError(label+' missing IDs or out of narrow prototype range')
        return a
    ref=core._float64(stress_free_temperature_K,'stress-free K')
    if not 1<=ref<=2000:raise ValueError('prototype stress-free K outside [1,2000]')
    temperature_K=required(temperature_K_by_id,ids,'node K',1.,2000.)
    dT=temperature_K-ref
    al=required(alpha_per_K_by_material,mats,'material alpha /K',-1e-3,1e-3)
    if np.any((al!=0)&(abs(al)<_MIN_NONZERO_ALPHA)):
        raise ValueError('nonzero alpha below prototype numerical floor')
    stable_key=lambda k:core._integer(k,'mechanical node ID')
    external=_detach_mapping(forces_by_id,len(ids),'forces',_detach_force,stable_key)
    original_holds=_detach_mapping(fixed_dofs_by_id,len(ids),'holds',_detach_hold,stable_key)
    def motion_row(v):
        return _detach_mapping(v,3,'prescribed axes',lambda z:z)
    original_moves=None if prescribed_displacements_m_by_id is None else _detach_mapping(
        prescribed_displacements_m_by_id,len(ids),'prescribed displacement',motion_row,stable_key)
    ext, fixed, prescribed=core._loads_and_constraints(mesh,external,original_holds,original_moves)
    holds,moves=_canonical_controls(mesh.node_ids,fixed,prescribed)
    core._static_topology_plan(mesh,fixed) # preserves existing assembly and structural gate
    # Narrow ordinary SI domain: avoid advertising the extreme range of core.
    if abs(ext).max()>_FORCE_LIMIT_N or (prescribed is not None and np.max(abs(prescribed))>1e-2):
        raise ValueError('prototype external forces or prescribed motion out of range')
    bases=[]; local=[[] for _ in ids]; delta=np.array([1.,1.,1.,0.,0.,0.])
    for j,(tet,B,C,V) in enumerate(core._element_kinematics(mesh)):
        mid=int(mesh.material_ids[j]); props=mesh.materials[mid]
        E,nu=props['young_modulus_pa'],props['poisson_ratio']
        a=al[mats[mid]]
        thermal=a*dT[tet]
        if not (1e-24<=V<=1 and 1e3<=E<=1e13 and -.5<=nu<=.45 and np.max(abs(thermal))<=.01):
            raise ValueError('cell outside narrow thermoelastic pilot range')
        if np.max(abs(B))>_MAX_GRADIENT_PER_M:
            raise ValueError('cell shape gradient outside prototype scale')
        core._check_normal_or_zero(thermal,'endpoint thermal strain')
        if np.any((a!=0)&(dT[tet]!=0)&(thermal==0)):
            raise ValueError('thermal strain underflow')
        mean=math.fsum(thermal)/4
        variance=math.fsum((float(e)-mean)**2 for e in thermal)/20
        kappa=E/(3*(1-2*nu)); L=V*core._field_dot(B.T,core._field_dot(C,mean*delta)).reshape(4,3)
        for i,f in zip(tet,L):local[i].append(f)
        bases.append((tet,B,C,V,mean,variance,kappa))
    def sum_rows(groups):
        return np.array([[math.fsum(v[k] for v in zz) for k in range(3)] for zz in groups])
    th=sum_rows(local)
    effective=_effective_force(ext,th)
    result=core._solve_static_validated(mesh,{int(i):tuple(float(z) for z in row) for i,row in zip(mesh.node_ids,effective)},
        holds,False,moves)
    u=result.displacement_m
    totals=[]; elastics=[]; stresses=[]; energies=[]; eig=[]; local=[[] for _ in ids]
    for tet,B,C,V,mean,var,kappa in bases:
        e=core._engineering_strain(B,u[tet])
        if np.max(abs(e))>.02:raise ValueError('total strain outside narrow pilot small-strain range')
        elastic=e-mean*delta; stress=core._field_dot(C,elastic)
        mu=C[3,3]; tr=math.fsum(elastic[:3]); dev=elastic[:3]-tr/3
        ev=V*(mu*(float(dev@dev)+.5*float(elastic[3:]@elastic[3:]))+.5*kappa*tr**2+4.5*kappa*var)
        coupling=V*(mean*math.fsum(stress[:3])-9*kappa*var)
        F=V*core._field_dot(B.T,stress).reshape(4,3)
        for i,f in zip(tet,F):local[i].append(f)
        totals.append(e); elastics.append(elastic); stresses.append(stress); energies.append(ev); eig.append(coupling)
    physical=sum_rows(local)-ext
    reaction=np.where(fixed,physical,0.)
    limit=1e-9*max(np.max(abs(th)),np.max(abs(ext)),np.max(abs(physical)),1e-20)
    if (np.max(abs(physical[~fixed]),initial=0)>limit or
        np.max(abs(reaction-result.reaction_N))>limit):
        raise ValueError('physical residual differs from guarded reduced solve')
    pairing=math.fsum(((reaction+ext)*u).ravel())
    U=math.fsum(energies); eigen_work=math.fsum(eig)
    # Discrete internal virtual work includes the measured free-DOF solver
    # residual. The public mechanical pairing excludes it. In compatible
    # free expansion, omitting this term compares a first-order residual
    # against elastic energy that is second-order in the strain error.
    residual_pairing=math.fsum(physical[~fixed]*u[~fixed])
    if not np.isfinite([pairing,U,eigen_work,residual_pairing]).all() or U<0:
        raise ValueError('thermal energy identity not resolved')
    work_scale=max(abs(pairing),abs(eigen_work),abs(residual_pairing),2*U,
                   np.max(abs(th))*np.max(abs(u))*1e-12,1e-30)
    balance=math.fsum((2*U,-pairing,eigen_work,-residual_pairing))
    if abs(balance)>1e-8*work_scale:
        raise ValueError('thermal energy identity not resolved')
    def tensor(rows, strain):
        a=np.array(rows)
        if strain:a[:,3:]*=.5
        return core._tensor_rows(a)
    arr=[mesh.node_ids,mesh.cell_ids,mesh.material_ids,u,fixed,reaction,tensor(totals,True),
         tensor(elastics,True),tensor(stresses,False),np.array(energies)]
    result = Tet4ThermoelasticResult(
        *map(_frozen_array, arr), U, pairing, eigen_work,
        float(np.max(abs(physical[~fixed]), initial=0)),
    )
    return result, temperature_K
