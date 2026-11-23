"""Private same-call, one-way solid DC to steady volumetric heating."""
from dataclasses import dataclass
import math
from types import MappingProxyType

import numpy as np

from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim import tet4 as core
from fibernet.sim.tet4_dc import Tet4DCResult, solve_dc_tet4
from fibernet.sim.tet4_conduction import Tet4SourceConductionResult, solve_steady_conduction_tet4_with_source


@dataclass(frozen=True)
class Tet4DCThermalResult:
    electrical: Tet4DCResult
    thermal: Tet4SourceConductionResult


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
    """Private same-call handoff; no transfer of external arrays or reverse feedback."""
    mesh=_snapshot(bundle)  # do not read caller geometry or material records again
    e=solve_dc_tet4(mesh,conductivity_S_per_m_by_material=conductivity_S_per_m_by_material,
                   electrode_voltage_V_by_id=electrode_voltage_V_by_id)
    src,nodal=_verify_source_transfer(mesh,e)
    h=solve_steady_conduction_tet4_with_source(mesh,
           conductivity_W_per_m_K_by_material=conductivity_W_per_m_K_by_material,
           prescribed_K_by_id=prescribed_K_by_id, source_W_per_m3_by_cell=src)
    if (not np.array_equal(h.cell_ids,mesh.cell_ids) or not np.array_equal(h.node_ids,mesh.node_ids)
            or not np.array_equal(h.integrated_source_W_by_cell,e.joule_power_W_by_cell)
            or not np.array_equal(h.source_by_node_W,nodal)
            or h.total_source_W!=e.total_joule_power_W):
        raise ValueError('DC-thermal heat handoff mismatch')
    return Tet4DCThermalResult(e,h)
