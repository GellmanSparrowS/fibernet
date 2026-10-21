"""Actual CPU snapshot damping; no physical solver mocks."""
import os
import numpy as np
import pytest
ti=pytest.importorskip('taichi')
from fibernet import StructureGraph
from fibernet.sim.accelerated import TaichiEngine

@pytest.fixture(scope='module')
def engine():
    threads=int(os.environ.get('FIBERNET_TEST_THREADS','1'))
    engine=TaichiEngine(arch='cpu',num_threads=threads)
    assert ti.cfg.cpu_max_num_threads == threads
    return engine

def graph(edges):
    g=StructureGraph(dimension=3)
    for p in ([0.,0.,0.],[1.,0.,0.],[2.,.2,0.],[1.,1.,.1]):g.add_node(p,merge=False)
    for a,b in edges:g.add_edge(a,b,allow_parallel=True)
    return g

def oracle(g,force,dash,drag=0.,fixed=(),scheduled=False):
    x0=np.array([n.position for n in g.nodes.values()]);x=x0.copy();v=np.zeros_like(x)
    e=g.edge_array();L=np.linalg.norm(x[e[:,0]]-x[e[:,1]],axis=1);out=[x.copy()]
    for step in range(20):
        old=v.copy(); f=np.zeros_like(x)
        for (a,b),l in zip(e,L):
            d=x[a]-x[b];r=np.linalg.norm(d);direction=d/(r+1e-6)
            elastic=-(r-l)/l*d/r if r>1e-12 else np.zeros(3)
            damp=-dash*np.dot(old[a]-old[b],direction)*direction*l
            f[a]+=elastic+damp;f[b]-=elastic+damp
        v+=.01*f;v+=.01*force;v*=np.exp(-drag*.01);x+=.01*v
        x[list(fixed)]=x0[list(fixed)];v[list(fixed)]=0
        if scheduled:x[0]=x0[0]+[.001*(step+1),0,0];v[0]=0
        out.append(x.copy())
    return np.array(out)

def run(engine,g,dash=None,force=None,drag=0.,fixed=(),scheduled=False):
    if force is None:force=np.array([[0.,0.,0.],[0.,0.,0.],[1.,.1,0.],[0.,0.,0.]])
    kwargs={} if dash is None else {'dashpot':dash}
    r=engine.dynamics(g,external_force=force,stiffness=1.,drag=drag,dt=.01,num_steps=20,save_interval=1,fixed_nodes=list(fixed),displacement_schedule={0:[(0,np.zeros(3)),(20,np.array([.02,0,0]))]} if scheduled else None,**kwargs)
    return np.array(r.positions_trajectory),force

class TestDashpotStaging:
    @pytest.mark.parametrize('edges',[[(0,1),(1,2)],[(0,1),(1,2),(1,3)],[(0,1),(1,0),(1,2)]])
    @pytest.mark.parametrize('dash',[None,0.])
    def test_old_velocity_recurrence_and_order(self,engine,edges,dash):
        g=graph(edges);actual,f=run(engine,g,dash)
        np.testing.assert_allclose(actual,oracle(g,f,10. if dash is None else dash),rtol=2e-14,atol=2e-14)
        reverse=graph([(b,a) for a,b in reversed(edges)]);other,_=run(engine,reverse,dash)
        np.testing.assert_allclose(actual,other,rtol=2e-14,atol=2e-14)

    def test_same_shape_reset(self,engine):
        g=graph([(0,1),(1,2)]);a,_=run(engine,g)
        run(engine,g,force=np.full((4,3),3.))
        b,_=run(engine,g);np.testing.assert_array_equal(a,b)
        zero,_=run(engine,g,force=np.zeros((4,3)))
        np.testing.assert_array_equal(zero,np.broadcast_to(np.array([n.position for n in g.nodes.values()]),zero.shape))

    def test_drag_fixed_schedule_stages(self,engine):
        g=graph([(0,1),(1,2),(1,3)]);a,f=run(engine,g,drag=.7,fixed=(0,),scheduled=True)
        np.testing.assert_allclose(a,oracle(g,f,10.,drag=.7,fixed=(0,),scheduled=True),rtol=2e-14,atol=2e-14)

    def test_missing_snapshot_layout_reallocates(self,engine):
        g=graph([(0,1),(1,2)]);a,_=run(engine,g)
        cache=TaichiEngine._field_cache[('mass-state-1',3,4,2)]
        del cache['ti_vel_old']
        b,_=run(engine,g)
        assert 'ti_vel_old' in TaichiEngine._field_cache[('mass-state-1',3,4,2)]
        assert TaichiEngine._field_cache[('mass-state-1',3,4,2)] is not cache
        np.testing.assert_array_equal(a,b)

    def test_runtime_reset_reallocates(self,engine):
        g=graph([(0,1),(1,2)]);a,_=run(engine,g)
        cache=TaichiEngine._field_cache[('mass-state-1',3,4,2)]
        ti.reset()
        b,_=run(engine,g)
        assert TaichiEngine._field_cache[('mass-state-1',3,4,2)] is not cache
        assert ti.cfg.cpu_max_num_threads == int(os.environ.get('FIBERNET_TEST_THREADS','1'))
        np.testing.assert_array_equal(a,b)
