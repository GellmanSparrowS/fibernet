"""Same-mesh, one-way solid DC to steady heat conduction."""
from dataclasses import dataclass
import math
from types import MappingProxyType

import numpy as np

from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim import tet4 as core
from fibernet.sim.tet4_dc import Tet4DCResult, solve_dc_tet4
from fibernet.sim.tet4_conduction import (
    Tet4SourceConductionResult, Tet4ConvectionConductionResult,
    solve_steady_conduction_tet4_with_source, solve_steady_conduction_tet4_with_convection,
)


@dataclass(frozen=True)
class Tet4DCThermalResult:
    electrical: Tet4DCResult
    thermal: Tet4SourceConductionResult


@dataclass(frozen=True)
class Tet4DCConvectionResult(Tet4DCThermalResult):
    thermal: Tet4ConvectionConductionResult


def _snapshot(bundle):
    if not isinstance(bundle,Tet4Bundle): raise TypeError('expected a Tet4Bundle')
    if core._row_count(bundle.points,'points')>1024 or core._row_count(bundle.tetrahedra,'tetrahedra')>8000:
        raise ValueError('DC-thermal pilot mesh limits')
    owned=core._validate_static_mesh(bundle)
    def lock(a):
        return np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape)
    mats=MappingProxyType({k:MappingProxyType(dict(v)) for k,v in owned.materials.items()})
    return Tet4Bundle(*map(lock,(owned.points,owned.tetrahedra,owned.node_ids,owned.cell_ids,owned.material_ids)),
                      mats,owned.length_unit)


def _verify_source_transfer(mesh, electrical):
    q=electrical.joule_density_W_per_m3; p=electrical.joule_power_W_by_cell
    if not np.array_equal(electrical.cell_ids,mesh.cell_ids) or not np.array_equal(electrical.node_ids,mesh.node_ids):
        raise ValueError('DC-thermal stable ID mismatch')
    if np.any((q!=0.)&((q<1e-8)|(q>1e8))) or not np.isfinite(q).all():
        raise ValueError('Joule density outside thermal source envelope')
    if not np.array_equal(p>0.,q>0.) or not np.isfinite(p).all():
        raise ValueError('unresolved Joule cell power')
    groups=[[] for _ in mesh.node_ids]
    for cell,power in zip(mesh.tetrahedra,p):
        lump=float(power)*.25
        if power and not lump: raise ValueError('unresolved Joule nodal lump')
        for i in cell:
            if lump: groups[i].append(lump)
    expected=[]
    for g in groups:
        total=math.fsum(g)
        if len(g)>1:
            j=min(range(len(g)),key=g.__getitem__)
            if total==math.fsum(v for i,v in enumerate(g) if i!=j):
                raise ValueError('Joule nodal heat lost a nonzero cell contribution')
        expected.append(total)
    return dict(zip(map(int,mesh.cell_ids),map(float,q))),np.array(expected)


def solve_dc_thermal_tet4(bundle, *, conductivity_S_per_m_by_material, electrode_voltage_V_by_id,
                          conductivity_W_per_m_K_by_material, prescribed_K_by_id):
    """Solve DC and transfer cell Joule power to steady heat conduction."""
    return _solve_dc_thermal(bundle, conductivity_S_per_m_by_material, electrode_voltage_V_by_id,
                             conductivity_W_per_m_K_by_material, prescribed_K_by_id)


def solve_dc_thermal_tet4_with_convection(
    bundle, *, conductivity_S_per_m_by_material, electrode_voltage_V_by_id,
    conductivity_W_per_m_K_by_material, prescribed_K_by_id,
    faces_by_node_id, convection_W_per_m2_K, ambient_K,
) -> Tet4DCConvectionResult:
    """Transfer DC Joule heat and include a prescribed-ambient convection boundary.

    Electrical and thermal conductivities are independent material maps.
    Use distinct stable-ID exterior triangles and one uniform coefficient
    in W/(m² K); ambient and prescribed temperatures are absolute kelvin.
    The steady temperature does not feed back into the DC solve.
    The mesh and its material records are copied once; do not mutate the
    inputs while they are being read. Thermal controls are read by the
    thermal solver after the Joule source has been validated.
    """
    return _solve_dc_thermal(
        bundle,conductivity_S_per_m_by_material,electrode_voltage_V_by_id,
        conductivity_W_per_m_K_by_material,prescribed_K_by_id,
        convection=dict(faces_by_node_id=faces_by_node_id,
                        convection_W_per_m2_K=convection_W_per_m2_K,ambient_K=ambient_K),
    )


def _solve_dc_thermal(bundle, sigma, voltages, conductivity, prescribed, *, convection=None):
    mesh=_snapshot(bundle)  # do not read caller geometry or material records again
    e=solve_dc_tet4(mesh,conductivity_S_per_m_by_material=sigma,
                   electrode_voltage_V_by_id=voltages)
    src,nodal=_verify_source_transfer(mesh,e)
    thermal_args=dict(conductivity_W_per_m_K_by_material=conductivity,
                      prescribed_K_by_id=prescribed,source_W_per_m3_by_cell=src)
    h=(solve_steady_conduction_tet4_with_source(mesh,**thermal_args) if convection is None
       else solve_steady_conduction_tet4_with_convection(mesh,**thermal_args,**convection))
    if (not np.array_equal(h.cell_ids,mesh.cell_ids) or not np.array_equal(h.node_ids,mesh.node_ids)
            or not np.array_equal(h.integrated_source_W_by_cell,e.joule_power_W_by_cell)
            or not np.array_equal(h.source_by_node_W,nodal)
            or h.total_source_W!=e.total_joule_power_W):
        raise ValueError('DC-thermal heat handoff mismatch')
    return Tet4DCThermalResult(e,h) if convection is None else Tet4DCConvectionResult(e,h)
