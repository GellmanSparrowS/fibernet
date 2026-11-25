"""Material contributions from static mechanical cell fields."""
from dataclasses import replace
import math

import numpy as np
import pytest

from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim.tet4 import solve_static_tet4
from fibernet.sim.tet4 import static_tet4_strain_energy_by_material as summarize


def _problem(permuted, zero=False, fields=True):
    points = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[0.,0.,-1.]])
    node_ids = np.array([0,2**53+3,7,9,2**64-1], dtype=np.uint64)
    materials = {0:{"young_modulus_pa":1e9,"poisson_ratio":.25},
                 2**64-1:{"young_modulus_pa":3e9,"poisson_ratio":.25}}
    order = [1,0] if permuted else [0,1]
    cells = np.array([[0,1,2,3],[0,2,1,4]])[order]
    mesh = Tet4Bundle(points, cells, node_ids, np.array([7,9])[order],
        np.array(list(materials), dtype=np.uint64)[order], materials, "mm")
    strain = 1e-4*np.array([[2,1.5,-.5],[1.5,-.5,.25],[-.5,.25,.1]])
    caps = {int(i):dict(zip("xyz", (x@strain)*1e-3)) for i,x in zip(node_ids,points)}
    fixed = dict.fromkeys(map(int,node_ids),("x","y","z")) if zero else {}
    return solve_static_tet4(mesh, forces_by_id={}, fixed_dofs_by_id=fixed,
                             prescribed_displacements_m_by_id=None if zero else caps,
                             return_cell_fields=fields),strain


@pytest.mark.parametrize("permuted",[False,True])
def test_normal_and_shear_energy_preserves_exact_ids(permuted):
    result,e=_problem(permuted)
    trace=float(np.trace(e))
    deviator=e-trace*np.eye(3)/3
    expected=lambda E: (1e-9/6)*(.5*(E/1.5)*trace**2+(E/2.5)*np.sum(deviator**2))
    values=summarize(result)
    assert list(values)==[0,2**64-1]
    assert values==pytest.approx({0:expected(1e9),2**64-1:expected(3e9)},rel=1e-12,abs=0)
    assert math.fsum(values.values())==pytest.approx(result.strain_energy_J,rel=1e-12,abs=0)


def test_zero_and_missing_fields():
    result,_=_problem(False,zero=True)
    assert summarize(result)=={0:0.,2**64-1:0.}
    with pytest.raises(ValueError,match="return_cell_fields"):
        summarize(_problem(False,fields=False)[0])
    with pytest.raises(TypeError,match="static mechanical"):
        summarize(object())


@pytest.mark.parametrize("name,value",[
    ("cell_ids",np.array([7,7])),
    ("material_ids",np.array([0,1],dtype=float)),
    ("volume_m3",np.array([1.,-1.])),
    ("strain",np.zeros((2,2,2))),
    ("stress_Pa",np.full((2,3,3),np.nan)),
])
def test_malformed_fields_fail(name,value):
    result,_=_problem(False)
    with pytest.raises(ValueError):
        summarize(replace(result,cell_fields=replace(result.cell_fields,**{name:value})))


def test_inconsistent_and_nonphysical_fields_fail():
    result,_=_problem(False)
    with pytest.raises(ValueError,match="disagrees"):
        summarize(replace(result,strain_energy_J=1.1*result.strain_energy_J))
    with pytest.raises(ValueError,match="nonphysical"):
        summarize(replace(result,cell_fields=replace(result.cell_fields,
                                                    stress_Pa=-result.cell_fields.stress_Pa)))
