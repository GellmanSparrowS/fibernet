from dataclasses import replace
from itertools import permutations,product
import math
import numpy as np
from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim.tet4_conduction import solve_steady_conduction_tet4 as solve
def fixture(two_components=False):
    grid=np.array(list(product(range(4),range(2),range(2))))
    ix={tuple(p):i for i,p in enumerate(grid)}
    t=[];mid=[]
    for x in range(3):
        o=np.array([x,0,0])
        for order in permutations(range(3)):
            walk=[o.copy()]
            for ax in order:o=o.copy();o[ax]+=1;walk.append(o)
            t.append([ix[tuple(p)] for p in walk]);mid.append(17 if x==0 else 9)
            o=np.array([x,0,0])
    pts=grid.astype(float); ids=np.arange(len(pts),dtype=np.uint64)+np.uint64(2**53)
    t=np.array(t);mid=np.array(mid); n=len(pts)
    if two_components:
        pts=np.vstack((pts,pts+[0,3,0]));t=np.vstack((t,t+n));mid=np.r_[mid,mid]
        ids=np.arange(len(pts),dtype=np.uint64)+np.uint64(2**53)
    b=Tet4Bundle(pts,t,ids,np.arange(len(t)),mid,
         {17:{'young_modulus_pa':1e8,'poisson_ratio':.2},9:{'young_modulus_pa':4e8,'poisson_ratio':.32}},'mm')
    return b, np.flatnonzero(grid[:,0]==0),np.flatnonzero(grid[:,0]==3)
k={17:1.,9:4.}
def go(b,l,h,a=330.,c=300.):
    bd={int(b.node_ids[j]):a for j in l};bd.update({int(b.node_ids[j]):c for j in h})
    return solve(b,conductivity_W_per_m_K_by_material=k,prescribed_K_by_id=bd)
def error(f,message):
    try:f()
    except ValueError as e:assert message in str(e),str(e)
    else:raise AssertionError(message+' unexpectedly accepted')


def test_layered_series_and_bounded_admission():
    b,l,h=fixture();r=go(b,l,h)
    np.testing.assert_array_equal(r.node_ids,b.node_ids)
    np.testing.assert_allclose(r.temperature_K,np.repeat([330,310,305,300],4),rtol=0,atol=5e-13)
    np.testing.assert_allclose(r.flux_W_per_m2,np.tile([20000.,0.,0.],(len(b.tetrahedra),1)),rtol=0,atol=3e-9)
    np.testing.assert_allclose(r.gradient_K_per_m[:,0],np.repeat([-20000.,-5000.],[6,12]),rtol=0,atol=1e-9)
    np.testing.assert_allclose([math.fsum(r.heat_into_solid_W[l]),math.fsum(r.heat_into_solid_W[h]),
                                r.dirichlet_quadratic_W_K,r.terminal_pairing_W_K],[.02,-.02,.6,.6],rtol=1e-12,atol=0)
    rev=go(b,l,h,300.,330.)
    np.testing.assert_allclose(rev.flux_W_per_m2,-r.flux_W_per_m2,rtol=0,atol=3e-9)
    same=go(b,l,h,320.,320.)
    np.testing.assert_array_equal(same.temperature_K,320.)
    np.testing.assert_array_equal(same.flux_W_per_m2,0.)
    assert same.dirichlet_quadratic_W_K==same.terminal_pairing_W_K==0.
    mt=go(replace(b,points=b.points*1e-3,length_unit='m'),l,h)
    np.testing.assert_array_equal(mt.temperature_K,r.temperature_K)
    b2,l2,h2=fixture(True)
    error(lambda:go(b2,l2,h2),'requires a prescribed node')
    error(lambda:solve(b,conductivity_W_per_m_K_by_material={17:0.,9:4.},prescribed_K_by_id={int(b.node_ids[0]):315.}),'conductivity')
    error(lambda:go(b,l,h,0.,300.),'temperature')
    # A single input ULP near 300 K is insufficient to publish any internal
    # fraction of the span in absolute Kelvin. No rounding/clamping it through.
    one=np.nextafter(300.,np.inf)
    error(lambda:go(b,l,h,one,300.),'unresolved absolute-to-centered')
    # A separated second component has its own anchor and reference, no
    # borrowed 300 K reference for reconstructing its variable temperatures.
    bd={int(b2.node_ids[j]):330. for j in l};bd.update({int(b2.node_ids[j]):300. for j in h})
    bd.update({int(b2.node_ids[16+j]):1450. for j in l});bd.update({int(b2.node_ids[16+j]):1400. for j in h})
    separate=solve(b2,conductivity_W_per_m_K_by_material=k,prescribed_K_by_id=bd)
    np.testing.assert_allclose(separate.temperature_K[:16],r.temperature_K,rtol=0,atol=5e-13)
    np.testing.assert_allclose(separate.temperature_K[16:],np.repeat([1450.,1450.-100/3,1400.+25/3,1400.],4),rtol=0,atol=5e-13)
    np.testing.assert_array_equal(separate.reference_K_by_node,np.repeat([300.,1400.],16))
    # Preserve order of exact large IDs through point/cell permutation.
    o=np.array([5,1,13,8,15,2,0,7,9,6,12,11,3,14,10,4]); io=np.argsort(o)
    cell_order=np.arange(len(b.tetrahedra))[::-1]
    sm=replace(b,points=b.points[o],node_ids=b.node_ids[o],tetrahedra=io[b.tetrahedra[cell_order]],
               cell_ids=b.cell_ids[cell_order],material_ids=b.material_ids[cell_order])
    sr=go(sm,io[l],io[h])
    np.testing.assert_array_equal(sr.node_ids,b.node_ids[o])
    np.testing.assert_array_equal(sr.cell_ids,b.cell_ids[cell_order])
    np.testing.assert_allclose(sr.temperature_K[io],r.temperature_K,rtol=0,atol=5e-13)
    np.testing.assert_allclose(sr.flux_W_per_m2[::-1],r.flux_W_per_m2,rtol=0,atol=3e-9)
    for z in vars(r).values():
     if isinstance(z,np.ndarray):
      assert not z.flags.writeable
      try:z.setflags(write=True)
      except ValueError:pass
      else:raise AssertionError('result is mutable')


def test_obtuse_tetrahedron_out_of_range_is_rejected_without_clipping():
    # Shape gradient rows for local nodes 0 and 2 have positive dot product.
    x=np.array([[0.,0.,0.],[1.,0.,0.],[2.,1.,0.],[0.,0.,1.]])
    g=np.linalg.inv((x[1:]-x[:1]).T);g=np.vstack((-g.sum(axis=0),g))
    A=g@g.T
    assert A[0,2]>0
    ids=np.array([16,4,29,8],dtype=np.uint64)
    m=Tet4Bundle(x,np.array([[0,1,2,3]]),ids,np.array([53]),np.array([57]),
                {57:dict(young_modulus_pa=1e8,poisson_ratio=.3)},'mm')
    fixed=np.array([1.,2000.,1.])
    value=-float(A[0,1:]@fixed)/A[0,0]
    assert value<1.
    try:solve(m,conductivity_W_per_m_K_by_material={57:1.},prescribed_K_by_id=dict(zip(map(int,ids[1:]),fixed)))
    except ValueError as ex:
     assert 'outside [1,2000] K; no clipping' in str(ex)
    else:raise AssertionError('out-of-range P1 result admitted')
