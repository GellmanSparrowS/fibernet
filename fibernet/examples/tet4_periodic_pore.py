# SPDX-License-Identifier: MIT
"""Compare a periodic duct with and without a synthetic three-path scaffold.

Run after installing FiberNet with the pore-flow extra on Python 3.10+:
    python -m fibernet.examples.tet4_periodic_pore
"""
from dataclasses import asdict, replace
from itertools import product,permutations
import json
import numpy as np
from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim.tet4_pore import solve_x_periodic_stokes,solve_matched_pore_geometry

def _scaffold_voxels():
    # All axial paths and crosslinks are strictly interior to the test duct.
    I=set(product(range(2,6),[2],[2]))
    I.update(product(range(2,6),[4],[2]))
    I.update(product(range(2,6),[2],[4]))
    I.update(product([3],range(2,5),[2]))
    I.update(product([4],[2],range(2,5)))
    assert len(I)>10
    return I

def bundle(obstacle=True):
    vox=set(product(range(8),range(7),range(7)))
    return _voxel_bundle(vox-_scaffold_voxels() if obstacle else vox)

def _voxel_bundle(voxels, elastic_modulus=1.):
    cells=sorted(voxels)
    pattern=[np.vstack((np.zeros(3,dtype=int),
                         np.cumsum(np.eye(3,dtype=int)[list(s)],axis=0))) for s in permutations(range(3))]
    pts=np.unique(np.concatenate([np.asarray(cells)[:,None,:]+a[None,:,:] for a in pattern],axis=1).reshape(-1,3),axis=0)
    lookup={tuple(v):i for i,v in enumerate(pts)}
    t=np.asarray([[lookup[tuple(v)] for v in a+c] for c in cells for a in pattern])
    assert len(t)==6*len(cells)
    return Tet4Bundle(pts*.2,t,np.arange(len(pts))*11+3,np.arange(len(t))*7+1,np.ones(len(t),int),
                      {1:dict(young_modulus_pa=elastic_modulus,poisson_ratio=.3)},'mm')
def demo():
    b=bundle(True)
    blank=bundle(False)
    kw=dict(viscosity_Pa_s=.001,pressure_gradient_Pa_per_m=500.)
    q0=solve_x_periodic_stokes(blank,**kw)
    q1=solve_x_periodic_stokes(b,**kw)
    assert q0.axial_flow_m3_s>q1.axial_flow_m3_s>0.
    # Uniform scale is a manufactured covariance check, not elasticity.
    factor=1.025
    q2=solve_matched_pore_geometry(b,replace(b,points=b.points*factor),**kw)
    np.testing.assert_allclose(q2.relative_bulk_flow_change,factor**4-1.,rtol=1e-9)
    np.testing.assert_allclose(q2.relative_cap_flow_change,factor**4-1.,rtol=1e-9)
    print(json.dumps(dict(cells=len(b.cell_ids),empty_flow=q0.axial_flow_m3_s,
                network_flow=q1.axial_flow_m3_s,paired=asdict(q2))),flush=True)
if __name__=='__main__':
    demo()
