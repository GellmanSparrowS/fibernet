"""Actual CPU kernel checks; internal fields expose the elastic kick, no mock law."""
import json
from pathlib import Path
import numpy as np
import pytest

ti = pytest.importorskip('taichi')
from fibernet.core.structure_graph import StructureGraph
from fibernet.sim.accelerated import TaichiEngine


@pytest.fixture(scope='module')
def engine():
    assert ti.__version__ == (1, 7, 4)
    return TaichiEngine(arch='cpu', num_threads=4)


def fixture(edges=((0,1),), scale=1.):
    g=StructureGraph(dimension=3)
    for p in ([0.,0.,0.],[1.,.3,.2],[.2,1.1,-.1]):
        g.add_node(np.array(p)*scale, merge=False)
    for a,b in edges: g.add_edge(a,b,allow_parallel=True)
    return g


def assembly(engine,g,x,k):
    """Read actual velocity kick/dt with no drag/dashpot/constraints/external load.

    dynamics allocates its actual cached production kernel; resetting fields
    lets that same kernel be tested at a prescribed deformed state.
    """
    engine.dynamics(g,spring_k=k,num_steps=1,save_interval=1,dt=1e-3,dashpot=0.,drag=0.)
    c=TaichiEngine._field_cache[('mass-state-1',3,len(g.nodes),len(g.edges))]
    c['ti_pos'].from_numpy(np.asarray(x,dtype=float));c['ti_vel'].fill(0)
    c['substep']()
    return c['ti_vel'].to_numpy()/1e-3


def arrays(g):
    x=np.array([n.position for n in g.nodes.values()])
    ed=g.edge_array()
    return x,ed,np.linalg.norm(x[ed[:,1]]-x[ed[:,0]],axis=1)


def energy(x,ed,l0,k):
    return np.sum(k/(2*l0)*(np.linalg.norm(x[ed[:,1]]-x[ed[:,0]],axis=1)-l0)**2)


class TestNativeElasticLaw:
    @pytest.mark.parametrize('state', ['rest','deformed','compressed'])
    def test_actual_kick_matches_force_kernel(self,engine,state):
        g=fixture();x,ed,l0=arrays(g);k=np.array([7.])
        if state=='deformed':x[1]+=[.2,-.1,.15]
        if state=='compressed':x[1]*=.6
        actual=assembly(engine,g,x,k);expected=engine.compute_forces(x,l0,k,ed)
        np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=2e-14)
        if state=='rest': np.testing.assert_array_equal(actual,np.zeros_like(x))


    def test_orientation_parallel_and_balance(self,engine):
        g=fixture(((0,1),(1,0),(1,2)));x,ed,l0=arrays(g);x[1]+=[.2,-.1,.15];k=np.array([2.,5.,3.])
        f=assembly(engine,g,x,k)
        np.testing.assert_allclose(f,engine.compute_forces(x,l0,k,ed),rtol=2e-13,atol=2e-14)
        np.testing.assert_allclose(f.sum(axis=0),0,atol=2e-14)
        np.testing.assert_allclose(np.cross(x,f).sum(axis=0),0,atol=2e-14)
        h=fixture(((1,0),(0,1),(2,1)))
        np.testing.assert_allclose(f,assembly(engine,h,x,k),rtol=2e-13,atol=2e-14)
        single=fixture(((0,1),(1,2)))
        np.testing.assert_allclose(f,assembly(engine,single,x,np.array([7.,3.])),rtol=2e-13,atol=2e-14)


    def test_coordinate_rescaling(self,engine):
        base=None
        for scale in (1.,1e-3,1e3):
            g=fixture(scale=scale);x,ed,l0=arrays(g);x[1]+=scale*np.array([.2,-.1,.15]);k=np.array([7.])
            f=assembly(engine,g,x,k)
            np.testing.assert_allclose(f,engine.compute_forces(x,l0,k,ed),rtol=2e-13,atol=2e-14)
            if base is None:base=f
            np.testing.assert_allclose(f,base,rtol=2e-13,atol=2e-14)


    def test_potential_gradient_and_rotation(self,engine):
        g=fixture(((0,1),(1,0),(1,2),(2,0)));x,ed,l0=arrays(g);x[1]+=[.2,-.1,.15];k=np.array([2.,5.,3.,4.])
        f=assembly(engine,g,x,k);grad=np.zeros_like(x);h=2e-6
        for i in range(3):
            for j in range(3):
                xp=x.copy();xm=x.copy();xp[i,j]+=h;xm[i,j]-=h
                grad[i,j]=(energy(xp,ed,l0,k)-energy(xm,ed,l0,k))/(2*h)
        np.testing.assert_allclose(f,-grad,rtol=2e-9,atol=2e-10)
        q=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        rotated=fixture(((0,1),(1,0),(1,2),(2,0)))
        for node in rotated.nodes.values():node.position=q@node.position+np.array([3.,-2.,1.])
        np.testing.assert_allclose(assembly(engine,rotated,x@q.T+[3.,-2.,1.],k),f@q.T,rtol=2e-13,atol=2e-14)


    @pytest.mark.parametrize('distance',[0.,.5e-12,1e-12,2e-12])
    def test_explicit_small_distance_branch(self,engine,distance):
        g=fixture();x,ed,l0=arrays(g);x[1]=[distance,0,0];k=np.array([7.])
        actual=assembly(engine,g,x,k)
        np.testing.assert_allclose(actual,engine.compute_forces(x,l0,k,ed),rtol=2e-13,atol=2e-14)
        # Preserve the existing compiled kernel's threshold semantics.
        # At nominal 1e-12, literal precision can select the nonzero branch.
        if distance<1e-12:np.testing.assert_array_equal(actual,0)


    def test_tiny_network_followup_and_unit_mass_kick_drift(self,engine):
        g=fixture(((0,1),(1,2),(2,0)));x,ed,l0=arrays(g);ext=np.array([[0.,0.,0.],[.2,-.1,.05],[0.,0.,0.]])
        dt=1e-3;k=np.array([7.,3.,4.]);steps=12
        res=engine.dynamics(g,spring_k=k,external_force=ext,num_steps=steps,save_interval=1,dt=dt,dashpot=0.,drag=0.)
        # Independent stepping uses the public native force kernel, not a duplicate
        # spring law; checks actual dynamics' unit-mass kick-then-drift convention.
        p=x.copy();v=np.zeros_like(p)
        for i in range(steps):
            v+=dt*(engine.compute_forces(p,l0,k,ed)+ext);p+=dt*v
            np.testing.assert_allclose(res.positions_trajectory[i+1],p,rtol=1e-13,atol=2e-14)
        assert np.isfinite(res.energy) and res.energy>=0
        assert res.metadata['node_ids']==list(g.nodes)
        assert res.metadata['edge_ids']==list(g.edges)
        np.testing.assert_allclose(res.energy,energy(p,ed,l0,k),rtol=1e-11,atol=1e-20)


    def test_closed_mixed_coordinate_work(self,engine):
        g=fixture();x,ed,l0=arrays(g);k=np.array([7.])
        corners=[x.copy() for _ in range(5)]
        offsets=[(0,0),(.12,0),(.12,.09),(0,.09),(0,0)]
        # Gauss-Legendre work integral around a deformed, non-colliding loop.
        for p,(a,b) in zip(corners,offsets):p[1]+=[.2+a,-.1+b,.15]
        nodes,weights=np.polynomial.legendre.leggauss(8);work=0.
        for a,b in zip(corners,corners[1:]):
            d=b-a
            for t,w in zip((nodes+1)/2,weights/2):
                work+=w*np.sum(assembly(engine,g,a+t*d,k)*d)
        assert abs(work)<2e-13
