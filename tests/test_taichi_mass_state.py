"""Native CPU mass/state contract; no inferred units, contact or work ledger."""
import copy
import json
import os
from pathlib import Path
import numpy as np
import pytest

from fibernet import StructureGraph
from fibernet.sim.accelerated import (SimResult, TaichiEngine, _resolve_nodal_masses,
    _validate_dynamics_state, _checked_kinetic_energy, _validated_fixed_rows)


def graph(n=2):
    g=StructureGraph(dimension=3)
    for i in range(n):g.add_node([float(i), .03*(i%2), 0.],merge=False)
    for i in range(n-1):g.add_edge(i,i+1)
    return g


@pytest.fixture(scope='module')
def engine():
    ti=pytest.importorskip('taichi')
    e=TaichiEngine(arch='cpu',num_threads=1)
    assert ti.cfg.cpu_max_num_threads==1
    return e


def run(e,g,**kw):
    args=dict(stiffness=2.,dashpot=0.,drag=0.,dt=.01,num_steps=5,save_interval=2)
    args.update(kw)
    return e.dynamics(g,**args)


@pytest.mark.parametrize('mass',[0.,[1.],[1.,0.],[1.,-1.],[1.,np.nan],[1.,np.inf],
    [[1.],[1.]],[True,True],['1','2'],[1+0j,2+0j],np.ma.array([1.,2.],mask=[0,1])])
def test_invalid_mass(mass):
    with pytest.raises(ValueError):_resolve_nodal_masses([0,1],mass,None)


def test_map_and_detachment_validation():
    for mapping in ({0:1},{0:1,1:2,2:3},{False:1,1:2},{0.:1,1:2},{'0':1,'1':2}):
        with pytest.raises(ValueError):_resolve_nodal_masses([0,1],None,mapping)
    with pytest.raises(ValueError):_resolve_nodal_masses([0,1],[1,2],{0:1,1:2})
    x=np.array([2.,3.]);m=_resolve_nodal_masses([0,1],x,None);x[:]=9
    np.testing.assert_array_equal(m,[2,3])
    tiny=np.nextafter(np.longdouble(0),np.longdouble(1))
    if tiny>0 and float(tiny)==0:
        with pytest.raises(ValueError):_resolve_nodal_masses([0,1],np.array([tiny,1],dtype=np.longdouble),None)


class TestNativeMassState:
    def test_force_mass_and_center_of_mass(self,engine):
        g=graph();f=np.array([[1.,0,0],[-1.,0,0]])
        r=run(engine,g,nodal_masses=[2.,4.],external_force=f,num_steps=1,save_interval=3,record_state=True)
        s=r.dynamics_state;v=np.asarray(s['velocities']);x=np.asarray(s['positions'])
        np.testing.assert_allclose(v[-1],.01*f/np.array([2,4])[:,None],rtol=1e-14,atol=0)
        np.testing.assert_allclose((v[-1]*np.array([2,4])[:,None]).sum(0),0,atol=1e-16)
        np.testing.assert_allclose((x[-1]*np.array([2,4])[:,None]).sum(0),(x[0]*np.array([2,4])[:,None]).sum(0),atol=1e-14)
        assert s['steps']==[0,1] and s['work_ledger']['value'] is None

    def test_default_ones_cache_A_B_None(self,engine):
        g=graph();f=np.array([[0.,0,0],[1.,0,0]])
        old=run(engine,g,external_force=f)
        a=run(engine,g,external_force=f,nodal_masses=[2.,3.],record_state=True)
        b=run(engine,g,external_force=f,nodal_masses=[3.,7.],record_state=True)
        assert a.dynamics_state['masses']!=b.dynamics_state['masses']
        again=run(engine,g,external_force=f)
        ones=run(engine,g,external_force=f,nodal_masses=[1.,1.])
        for r in (again,ones):
            np.testing.assert_array_equal(r.deformed_positions,old.deformed_positions)
            np.testing.assert_array_equal(r.positions_trajectory,old.positions_trajectory)
        assert old.dynamics_state is None and len(old.positions_trajectory)==3
        assert [h['step'] for h in old.history]==[2,4]
        assert 'native_axial_inputs' not in old.metadata
        assert ones.metadata['native_axial_inputs']['masses']==[1.,1.]

    @pytest.mark.parametrize('steps,interval,expected',[(1,1,[0,1]),(4,2,[0,2,4]),(5,2,[0,2,4,5]),(2,9,[0,2])])
    def test_sampling(self,engine,steps,interval,expected):
        r=run(engine,graph(),record_state=True,num_steps=steps,save_interval=interval)
        s=r.dynamics_state
        assert s['steps']==expected and s['times']==[i*.01 for i in expected]
        assert [row['step'] for row in r.history]==s['steps']
        np.testing.assert_array_equal(r.positions_trajectory,s['positions'])
        assert s['final']['positions']==s['positions'][-1]
        assert s['final']['velocities']==s['velocities'][-1]
        assert len(s['kinetic_energy'])==len(expected)

    def test_sparse_ids_parallel_schedule(self,engine):
        g=graph();g._nodes={30:g._nodes[0],-7:g._nodes[1]}
        for node_id,node in g.nodes.items():node.node_id=node_id
        assert all(node_id==node.node_id for node_id,node in g.nodes.items())
        g.edges[0].node_i=30;g.edges[0].node_j=-7
        g.add_edge(-7,30,allow_parallel=True)
        masses={-7:4.,30:2.}
        r=run(engine,g,nodal_masses_by_id=masses,record_state=True,fixed_nodes=[0],
              displacement_schedule={0:[(0,np.zeros(3)),(5,np.array([.02,0,0]))]})
        assert r.dynamics_state['node_ids']==[30,-7]
        assert r.dynamics_state['masses']==[2.,4.]
        assert r.dynamics_state['final']['velocities'][0]==[0.,0.,0.]
        assert r.dynamics_state['final']['positions'][0][0]==.02
        masses[-7]=99
        assert r.metadata['native_axial_inputs']['masses']==[2.,4.]

    def test_json_old_summary_details_and_rejections(self,engine,tmp_path):
        g=graph();units=dict(length='m',time='s',mass='kg',frame='caller-world')
        r=run(engine,g,nodal_masses=[2.,3.],record_state=True,state_units=units)
        r.metadata['dynamics_state']={'bad':'shadow'}
        for detail in (False,True):
            p=tmp_path/('detail.json' if detail else 'summary.json');r.save(p,detailed=detail)
            loaded=SimResult.load(p);s=loaded.dynamics_state
            assert s['representation']==('sampled' if detail else 'summary')
            assert ('positions' in s)==detail
            assert loaded.to_dict(detailed=detail)['dynamics_state']==r.to_dict(detailed=detail)['dynamics_state']
        units['mass']='changed-after-run'
        assert r.dynamics_state['units']['mass']=='kg'
        old=run(engine,g);old.save(tmp_path/'old.json',detailed=True)
        assert SimResult.load(tmp_path/'old.json').dynamics_state is None
        assert 'dynamics_state' not in old.to_dict()
        valid=r.to_dict(detailed=True)
        mutations=[lambda s:s.update(schema='native_axial_state/99'),lambda s:s.update(representation='unknown'),
          lambda s:s.update(availability='unavailable'),lambda s:s.update(node_ids=[1,0]),
          lambda s:s.update(masses=[1,0]),lambda s:s.update(times=[0]),
          lambda s:s.update(steps=[0,2,2,5]),lambda s:s['positions'][0][0].__setitem__(0,float('nan')),
          lambda s:s['kinetic_energy'].__setitem__(0,99),lambda s:s['final'].__setitem__('time',99),
          lambda s:s['work_ledger'].__setitem__('value',0),lambda s:s.update(extra=1)]
        for mutate in mutations:
            obj=copy.deepcopy(valid);mutate(obj['dynamics_state']);p=tmp_path/'bad.json';p.write_text(json.dumps(obj))
            with pytest.raises(ValueError):SimResult.load(p)
        mismatch=copy.deepcopy(valid);mismatch['deformed_positions'][0][0]+=1
        p=tmp_path/'bad-final.json';p.write_text(json.dumps(mismatch))
        with pytest.raises(ValueError):SimResult.load(p)
        mismatch=copy.deepcopy(valid);mismatch['native_axial_inputs']['masses'][0]=99
        p.write_text(json.dumps(mismatch))
        with pytest.raises(ValueError):SimResult.load(p)
        obj=r.to_dict();obj['dynamics_state']['final']['total_energy']=None
        p=tmp_path/'bad-summary.json';p.write_text(json.dumps(obj))
        with pytest.raises(ValueError):SimResult.load(p)

    def test_explicit_validation_before_fields(self,engine):
        g=graph();before=set(TaichiEngine._field_cache)
        for kw in [dict(dt=0),dict(dt=np.inf),dict(dt=True),dict(dashpot=-1),dict(drag=np.nan),
                   dict(external_force=np.ones((2,2))),dict(external_force=np.full((2,3),np.nan))]:
            with pytest.raises(ValueError):run(engine,g,nodal_masses=[1,2],**kw)
        assert set(TaichiEngine._field_cache)==before
        # Native initialization may enable subnormal flushing. Preserve the
        # stored binary64 witness independently of nextafter/arithmetic.
        normal=run(engine,g,nodal_masses=[1.,1.],num_steps=1,record_state=True)
        assert np.isfinite(normal.dynamics_state['positions']).all()
        masses=np.array([1,0x3ff0000000000000],dtype=np.uint64).view(np.float64)
        assert int(masses.view(np.uint64)[0])==1
        before=set(TaichiEngine._field_cache)
        if not bool((masses>0).all()):
            with pytest.raises(ValueError,match='^nodal_masses must be strictly positive$'):
                run(engine,g,nodal_masses=masses,num_steps=1,record_state=True)
            assert set(TaichiEngine._field_cache)==before
        else:
            # Supported gradual arithmetic retains the original success case.
            r=run(engine,g,nodal_masses=masses,num_steps=1,record_state=True)
            assert np.isfinite(r.dynamics_state['positions']).all()

    def test_numpy_200_node_recurrence(self,engine):
        g=graph(200);x=np.array([n.position for n in g.nodes.values()]);v=np.zeros_like(x)
        masses=np.linspace(1,3,200);e=g.edge_array();l=np.linalg.norm(x[e[:,1]]-x[e[:,0]],axis=1)
        ext=np.zeros_like(x);ext[-1,0]=1.;dt=.001
        r=run(engine,g,nodal_masses=masses,external_force=ext,record_state=True,dt=dt,num_steps=4,save_interval=1,dashpot=10.,drag=.2)
        for step in range(1,5):
            force=np.zeros_like(x)
            for (a,b),length in zip(e,l):
                d=x[a]-x[b];dist=np.linalg.norm(d);direction=d/(dist+1e-6)
                spring=-2*(dist-length)/length*d/dist
                damp=-10*np.dot(v[a]-v[b],direction)*direction*length
                force[a]+=spring+damp;force[b]-=spring+damp
            v+=dt*(force+ext)/masses[:,None];v*=np.exp(-.2*dt);x+=dt*v
            np.testing.assert_allclose(r.dynamics_state['positions'][step],x,rtol=2e-14,atol=2e-14)
            np.testing.assert_allclose(r.dynamics_state['velocities'][step],v,rtol=2e-12,atol=1e-14)

    def test_dop853_refinement(self,engine):
        from scipy.integrate import solve_ivp
        g=graph();x0=np.array([n.position for n in g.nodes.values()]);l=np.linalg.norm(x0[1]-x0[0]);mass=np.array([2.,3.]);ext=np.array([[1.,0,0],[-1.,0,0]])
        def rhs(t,y):
            x=y[:6].reshape(2,3);v=y[6:].reshape(2,3);d=x[1]-x[0];length=np.linalg.norm(d);f=2*(length-l)/l*d/length
            return np.r_[v.ravel(),((np.array([f,-f])+ext)/mass[:,None]).ravel()]
        ref=solve_ivp(rhs,(0,.1),np.r_[x0.ravel(),np.zeros(6)],method='DOP853',rtol=1e-11,atol=1e-13).y[:6,-1].reshape(2,3)
        errors=[]
        for dt in (.01,.005):
            r=run(engine,g,nodal_masses=mass,external_force=ext,dt=dt,num_steps=round(.1/dt),record_state=True)
            errors.append(np.linalg.norm(r.deformed_positions-ref))
        assert errors[1]<.6*errors[0]

    def test_clamped_constant_load_one_step(self,engine):
        f=np.array([[0.,0,0],[2.,0,0]])
        r=run(engine,graph(),nodal_masses=[3.,4.],external_force=f,fixed_nodes=[0],record_state=True,num_steps=1)
        np.testing.assert_array_equal(r.dynamics_state['final']['velocities'][0],0)
        np.testing.assert_allclose(r.dynamics_state['final']['velocities'][1],[.005,0,0],atol=1e-16)

    def test_recorded_nonfinite_failure_is_not_hidden(self,engine):
        with pytest.raises(ValueError):
            run(engine,graph(),nodal_masses=[1.,1.],external_force=np.full((2,3),1e308),dt=1e308,num_steps=1,record_state=True)


@pytest.mark.parametrize('mass,velocity,expected',[(1e-200,1e200,5e199),(1e200,1e-200,5e-201),(2.,3.,9.)])
def test_stable_kinetic_products(mass,velocity,expected):
    assert _checked_kinetic_energy([mass],[[velocity,0.,0.]])==pytest.approx(expected,rel=2e-15,abs=0)

@pytest.mark.parametrize('mass,velocity',[(1e-300,1e-300),(1e300,1e300)])
def test_unrepresentable_kinetic_rejected(mass,velocity):
    with pytest.raises(ValueError):_checked_kinetic_energy([mass],[[velocity,0.,0.]])


def test_negative_stable_ids_preserved():
    np.testing.assert_array_equal(_resolve_nodal_masses([-1,2],None,{-1:3,2:7}),[3.,7.])


@pytest.mark.parametrize('mass,velocity,kinetic',[(1e-200,1e200,5e199),(1e200,1e-200,5e-201)])
def test_compensated_kinetic_json_record(mass,velocity,kinetic):
    positions=[[[0.,0.,0.],[1.,0.,0.]]]
    velocities=[[[velocity,0.,0.],[0.,0.,0.]]]
    final=dict(step=0,time=0.,positions=positions[0],velocities=velocities[0],
               kinetic_energy=kinetic,potential_energy=0.,total_energy=kinetic,near_cutoff=False)
    state=dict(schema='native_axial_state/1',representation='sampled',availability='available',
        units=None,dt=.01,node_ids=[0,1],masses=[mass,1.],steps=[0],times=[0.],
        positions=positions,velocities=velocities,kinetic_energy=[kinetic],potential_energy=[0.],
        total_energy=[kinetic],near_cutoff=[False],final=final,cutoff_check='sampled_only',
        work_ledger={'status':'unavailable','value':None,'reason':'not_collected'})
    assert _validate_dynamics_state(json.loads(json.dumps(state)))['kinetic_energy']==[kinetic]


@pytest.mark.parametrize('rows', [[-1],[True],[.5],[2],[[0]],np.ma.array([0],mask=[True]),'0',0])
def test_invalid_opt_in_fixed_rows(rows):
    with pytest.raises(ValueError):_validated_fixed_rows(rows,2)


def test_valid_fixed_rows_are_not_stable_ids():
    assert _validated_fixed_rows([0,1],2)==[0,1]
    assert _validated_fixed_rows(np.array([1],dtype=np.int64),2)==[1]
    with pytest.raises(ValueError):_validated_fixed_rows([-7],2)
