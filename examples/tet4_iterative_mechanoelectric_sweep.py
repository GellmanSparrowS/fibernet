# SPDX-License-Identifier: MIT
"""Compute the resistance of a fused lattice over a small proportional stroke sweep.

Install the optional solid-iterative extra. Run from the source root:
python -m examples.tet4_iterative_mechanoelectric_sweep
"""
from dataclasses import replace
from math import fsum, isclose
import json
import numpy as np

from fibernet.sim.tet4 import solve_static_tet4
from fibernet.sim.tet4_dc import solve_dc_tet4
from .tet4_mechanoelectric_looped_network import _bundle, _fixture, _lattice


def run(*, refinement=3):
    points,tets=_lattice(refinement=refinement)
    mesh=_bundle(points,tets,'stiff_half')
    high,grip,fixed,electrodes,_=_fixture(mesh)
    stroke_m=1.5e-6
    displacement={nid:{'x':stroke_m} for nid in grip}
    mech=solve_static_tet4(mesh,forces_by_id={},fixed_dofs_by_id=fixed,
                           prescribed_displacements_m_by_id=displacement,
                           solver='amg-cg')
    reaction=fsum(mech.reaction_N[high,0])
    xyz=points[tets]
    initial_vol=np.linalg.det(xyz[:,1:]-xyz[:,:1])
    if reaction <= 0 or not isclose(mech.strain_energy_J,.5*reaction*stroke_m,rel_tol=1e-8):
        raise ValueError("Calculated grip reaction and elastic energy disagree")
    result=[]
    for fraction in (-1.,-.5,0.,.5,1.):
        # Fixed linear elasticity and zero preload: any proportional signed
        # stroke has the same displacement shape. DC is solved at each shape.
        deformed=points + (1000.*fraction)*mech.displacement_m
        tet=deformed[tets]
        volume=np.linalg.det(tet[:,1:]-tet[:,:1])
        if not np.all(np.isfinite(volume) & (volume*initial_vol > 0.)):
            raise ValueError("Stroke would invert or collapse a solid cell")
        dc=solve_dc_tet4(replace(mesh,points=deformed),
                        conductivity_S_per_m_by_material=dict.fromkeys(mesh.materials,1.),
                        electrode_voltage_V_by_id=electrodes,solver='cg')
        current=fsum(dc.electrode_current_into_solid_A[high])
        if not current>0.:
            raise ValueError("Expected positive current at the high electrode")
        result.append(dict(stroke_m=stroke_m*fraction, reaction_N=reaction*fraction,
                    strain_energy_J=mech.strain_energy_J*fraction**2, resistance_ohm=1./current))
    return dict(nodes=len(points),tetrahedra=len(tets),states=result)


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
