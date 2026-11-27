"""Reproduce a square duct and a generated three-path pore, using the load API."""
from dataclasses import replace
from itertools import product
import json,math
import numpy as np
from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.examples.tet4_periodic_pore import _scaffold_voxels,_voxel_bundle

from fibernet.sim.tet4_open_stokes import solve_open_pressure_stokes as solve

delta_p=.0035
mu=.001
# A bare triangular cap has only wall-fixed vertices and edges. Even
# geometrically connected valid tetrahedra need an admitted pressure port.
prism=Tet4Bundle(
    np.asarray([(0,0,0),(0,1,0),(0,0,1),(1,0,0),(1,1,0),(1,0,1)],dtype=float),
    np.asarray([(0,1,2,5),(0,1,4,5),(0,3,4,5)]),
    np.arange(6,dtype=np.uint64),np.arange(3,dtype=np.uint64),
    np.ones(3,dtype=np.uint64),{1:{"young_modulus_pa":1e9,"poisson_ratio":.3}},"mm",
)
for solver in ("direct", "amg"):
    try:
        solve(prism,viscosity_Pa_s=mu,pressure_drop_Pa=delta_p,solver=solver)
    except ValueError as error:
        assert "cap has no free pressure-work velocity trace" in str(error), error
    else:
        raise AssertionError("Did not reject the geometric cap with a locked P2 trace")
square=_voxel_bundle(product(range(16),range(6),range(6)))
blank=solve(square,viscosity_Pa_s=mu,pressure_drop_Pa=delta_p,solver="direct")
odd=range(1,200,2)
factor=(1.-192./math.pi**5*sum(math.tanh(k*math.pi/2)/k**5 for k in odd))/12.
width_mm=.2*6
length_mm=.2*16
expect=factor*(width_mm*1e-3)**4*(delta_p/(length_mm*1e-3))/mu
assert abs(blank.axial_flow_m3_s/expect-1)<.005
r=2
voxels=set(product(range(8),range(7),range(7)))-_scaffold_voxels()
dense={tuple(r*i+j for i,j in zip(v,d)) for v in voxels for d in product(range(r),repeat=3)}
b=_voxel_bundle(dense); b=replace(b,points=b.points/r)
flow=solve(b,viscosity_Pa_s=mu,pressure_drop_Pa=delta_p)
np.testing.assert_array_equal(flow.node_ids,b.node_ids)
assert len(b.tetrahedra)>12000 and flow.iterations>0
assert 0<flow.axial_flow_m3_s and flow.cap_mass_relative_error<1e-8
assert math.isclose(flow.dissipated_power_W,delta_p*flow.axial_flow_m3_s,rel_tol=1e-8)
# Simple fixed-pressure scaling: u scales by length; cross-sectional area
# by length squared. This is independent of the original scaffold shape.
scale=1.1
scaled=replace(b,points=b.points*scale)
reverse=solve(scaled,viscosity_Pa_s=mu*2,pressure_drop_Pa=-delta_p)
assert math.isclose(reverse.axial_flow_m3_s,-.5*flow.axial_flow_m3_s*scale**3,rel_tol=1e-8)
print(json.dumps(dict(tetrahedra=len(b.tetrahedra),square_analytic_relative_error=abs(blank.axial_flow_m3_s/expect-1),
     flow_m3_s=flow.axial_flow_m3_s,reverse_m3_s=reverse.axial_flow_m3_s,iterations=flow.iterations,
     cap_mass_relative_error=max(flow.cap_mass_relative_error,reverse.cap_mass_relative_error))))
