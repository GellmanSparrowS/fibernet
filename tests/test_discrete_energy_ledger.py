"""Exact arithmetic and native-stage ledger acceptance, separately gated."""
import copy
import json
import math
from decimal import Decimal, localcontext
from fractions import Fraction as Q
import numpy as np
import pytest
from fibernet.sim._energy_accounting import (
    CHANNELS, rational, nearest, upward, quantum, closure, validate_closure,
    ExactLedger, validate_ledger,
)


class TestFrozenArithmetic:
    @pytest.mark.parametrize('x', [0., -0., 1., -1., 2.**-1022, 5e-324, 1e300])
    def test_exact_ratios(self, x):
        assert rational(x) == Q(*x.as_integer_ratio())
        assert nearest(rational(x)) == x
        if x == 0: assert math.copysign(1, nearest(rational(x))) == 1

    def test_half_ulp_and_upward_subnormal(self):
        assert quantum(0.) == quantum(5e-324) == Q(1, 2**1075)
        assert quantum(1.) == Q(1, 2**53)
        assert quantum(2.) == Q(1, 2**52)
        assert upward(Q(1, 2**1075)) == 5e-324
        assert rational(upward(Q(1, 3))) >= Q(1, 3)
        assert nearest(Q(1) + Q(1, 2**53)) == 1.

    @pytest.mark.parametrize('q', [Q(1, 2**1076), Q(-1, 2**1076), Q(2)**1024])
    def test_output_range_rejections(self, q):
        with pytest.raises(ValueError): nearest(q)

    @pytest.mark.parametrize('x', [float('nan'), float('inf'), -float('inf'), True, '1'])
    def test_nonfinite_or_nonreal_rejections(self, x):
        with pytest.raises(ValueError): rational(x)

    @pytest.mark.parametrize('values,h0,h1', [([0.]*6,0.,0.),
        ([1e300,-1e300,1.,-1.,0.,0.],0.,0.),
        ([5e-324,0.,0.,0.,0.,0.],0.,5e-324),
        ([-3.,2.,-1.,4.,0.,0.],1.,3.)])
    def test_closure_and_single_channel_tamper(self, values, h0, h1):
        channels=dict(zip(CHANNELS,values)); rec=closure(channels,h0,h1)
        validate_closure(rec,channels,h0,h1)
        exact_r=abs(sum(map(rational,values),Q())-(rational(h1)-rational(h0)))
        exact_b=sum(map(quantum,values),Q())+quantum(h0)+quantum(h1)
        assert exact_r <= exact_b
        assert rec['residual_abs']==upward(exact_r)
        assert rec['bound']==upward(exact_b)
        bad=dict(channels)
        # Choose a representable perturbation beyond the independently derived bound.
        key=CHANNELS[-1]; bad[key]=nearest(rational(bad[key])+4*exact_b)
        with pytest.raises(ValueError): validate_closure(rec,bad,h0,h1)

    def test_zero_bound_display_is_not_used_to_accept(self):
        values=dict.fromkeys(CHANNELS,0.)
        rec=closure(values,0.,0.)
        assert rec['bound'] > 0 and rec['residual_abs']==0
        rec['residual_abs']=5e-324
        with pytest.raises(ValueError):validate_closure(rec,values,0.,0.)

    def test_exact_channel_telescope_and_geometric_subtraction(self):
        endpoint=lambda k,u:dict(kinetic_energy=k,potential_energy=u,total_energy=k+u)
        initial=endpoint(1e200,0.)
        l=ExactLedger(initial,np.array([[3.,0.,0.]]))
        a=np.array([[1e16,0.,0.]]); e=np.array([[1.,0.,0.]])
        l.advance(1,1.,(1e200,3.,4.,2.,5.,6.),(0.,7.,8.,9.),a,e,False)
        row=l.sample(1,1.,endpoint(6.,9.))
        assert sum(l.totals.values(),Q()) == Q(15)-rational(1e200)
        expected=3*Q(float(e[0,0]-a[0,0]))
        assert row['geometric_external_work']['value']==float(expected)
        error=abs(expected-3*(Q(1)-Q(10**16)))
        assert rational(row['geometric_external_work']['subtraction_bound'])>=error
        validate_closure(row['closure'],row['channels'],initial['total_energy'],15.)

    def test_zero_force_skips_unrepresentable_geometric_subtraction(self):
        endpoint=dict(kinetic_energy=0.,potential_energy=0.,total_energy=0.)
        l=ExactLedger(endpoint,np.zeros((1,3)))
        l.advance(1,1.,(0.,)*6,(0.,)*4,np.full((1,3),-1e308),np.full((1,3),1e308),False)
        assert l.sample(1,1.,endpoint)['geometric_external_work']['value']==0.

from fibernet import StructureGraph
from fibernet.sim.accelerated import SimResult, TaichiEngine, _checked_kinetic_energy


def graph(n=3):
    g=StructureGraph(dimension=3)
    for i in range(n):g.add_node([float(i),0.,0.],merge=False)
    for i in range(n-1):g.add_edge(i,i+1)
    return g


@pytest.fixture(scope='module')
def engine():
    ti=pytest.importorskip('taichi')
    e=TaichiEngine(arch='cpu',num_threads=1)
    assert ti.cfg.cpu_max_num_threads==1
    return e


def run(engine,g=None,**kwargs):
    if g is None:g=graph()
    args=dict(stiffness=2.,dashpot=10.,drag=.3,dt=.01,num_steps=5,save_interval=2,
              record_state=True,record_energy_ledger=True)
    args.update(kwargs)
    return engine.dynamics(g,**args)


def reference_stages(x0,masses,force,steps,dt,dash,drag,fixed,schedule,force_scale=1.):
    x=x0.copy();v=np.zeros_like(x);out=[]
    lengths=np.linalg.norm(x0[1:]-x0[:-1],axis=1)
    for step in range(steps):
        a=x.copy();old=v.copy()
        for i,length in enumerate(lengths):
            d=x[i]-x[i+1];dist=np.linalg.norm(d);direction=d/(dist+1e-6)
            elastic=-2.*(dist-length)/length*d/dist if dist>1e-12 else np.zeros(3)
            damp=-dash*np.dot(old[i]-old[i+1],direction)*direction*length
            impulse=(elastic+damp)*dt*force_scale
            v[i]+=impulse/masses[i];v[i+1]-=impulse/masses[i+1]
        edge=v.copy();v+=force*dt/masses[:,None];external=v.copy()
        v*=np.exp(-drag*dt);dragged=v.copy();x+=v*dt;drift=x.copy()
        for i in fixed:x[i]=x0[i];v[i]=0
        xf=x.copy();vf=v.copy()
        if schedule:x[0]=x0[0]+np.array([.002*(step+1),0.,0.]);v[0]=0
        out.append(dict(a=a,old=old,edge=edge,external=external,drag=dragged,drift=drift,
                        fixed_x=xf,fixed_v=vf,final_x=x.copy(),final_v=v.copy()))
    return out


class TestNativeLedger:
    @pytest.mark.parametrize('dash,drag,fixed,scheduled',[(0.,0.,[],False),(10.,.3,[],False),
        (10.,.3,[0],False),(10.,.3,[0],True)])
    def test_actual_stages_independent_recurrence(self,engine,monkeypatch,dash,drag,fixed,scheduled):
        g=graph();m=np.array([2.,3.,5.]);force=np.array([[.1,0,0],[0,0,0],[1.,0,0]])
        x0=np.array([n.position for n in g.nodes.values()]);observed=[]
        original=ExactLedger.advance
        def capture(self,*args,**kw):
            cache=TaichiEngine._field_cache[('energy-ledger-1',3,3,2)]
            observed.append({key:cache[field].to_numpy() for key,field in dict(old='ti_vel_old',
                edge='ledger_v_edge',external='ledger_v_external',drag='ledger_v_drag',drift='ledger_x_drift',
                fixed_x='ledger_x_fixed',fixed_v='ledger_v_fixed',final_x='ti_pos',final_v='ti_vel').items()})
            return original(self,*args,**kw)
        monkeypatch.setattr(ExactLedger,'advance',capture)
        schedule={0:[(0,np.zeros(3)),(5,np.array([.01,0,0]))]} if scheduled else None
        result=run(engine,g,nodal_masses=m,external_force=force,dashpot=dash,drag=drag,
                   fixed_nodes=fixed,displacement_schedule=schedule,save_interval=1)
        reference=reference_stages(x0,m,force,5,.01,dash,drag,fixed,scheduled)
        for actual,expected in zip(observed,reference):
            for key in actual:
                np.testing.assert_allclose(actual[key],expected[key],rtol=2e-12,atol=2e-15)
        # A deliberately wrong recurrence is caught independently of scalar closure.
        bad=reference_stages(x0,m,force,5,.01,dash,drag,fixed,scheduled,force_scale=1.01)
        assert np.max(np.abs(bad[-1]['final_v']-observed[-1]['final_v']))>1e-10
        ledger=result.algorithmic_energy_ledger
        assert ledger['substeps']==5 and ledger['steps']==[0,1,2,3,4,5]
        totals=dict.fromkeys(CHANNELS,Q()); prev_k=prev_u=0.
        for index,actual in enumerate(observed,1):
            k=lambda key:_checked_kinetic_energy(m,actual[key])
            u=lambda key:float(.5*np.sum(2.*(np.linalg.norm(actual[key][1:]-actual[key][:-1],axis=1)-1.)**2))
            kb,kc,kd,kf,kg=map(k,('edge','external','drag','fixed_v','final_v'))
            ue,uf,ug=map(u,('drift','fixed_x','final_x'))
            a,b,c,d,f,g=map(Q,(prev_k,kb,kc,kd,kf,kg));aa,e,ff,gg=map(Q,(prev_u,ue,uf,ug))
            inc=(b-a,c-b,d-c,e-aa,f-d+ff-e,g-f+gg-ff)
            for key,delta in zip(CHANNELS,inc):totals[key]+=delta
            row=ledger['samples'][index]
            assert row['channels']=={key:float(value) for key,value in totals.items()}
            assert row['elastic_integration_defect']['status']==('available' if dash==0 else 'unavailable')
            prev_k,prev_u=kg,ug
        assert ledger['final']['channels']['drag_delta_K']<=0
        if scheduled:
            assert ledger['final']['channels']['schedule_projection_delta_H']!=0
            assert ledger['final']['channels']['fixed_projection_delta_H']!=0

    def test_off_on_off_exact_and_cache_reset(self,engine):
        force=np.array([[0.,0,0],[0.,0,0],[1.,0,0]])
        a=run(engine,external_force=force,record_energy_ledger=False)
        b=run(engine,external_force=force)
        run(engine,external_force=-force,nodal_masses=[3.,5.,7.])
        c=run(engine,external_force=force,record_energy_ledger=False)
        d=run(engine,external_force=force)
        assert a.dynamics_state==b.dynamics_state==c.dynamics_state==d.dynamics_state
        for r in (a,b,c,d):
            np.testing.assert_array_equal(r.positions_trajectory,a.positions_trajectory)
        assert b.algorithmic_energy_ledger==d.algorithmic_energy_ledger
        assert 'algorithmic_energy_ledger' not in a.to_dict(True)
        a.metadata['algorithmic_energy_ledger']={'old_custom_key':'must_migrate'}
        assert 'algorithmic_energy_ledger' not in a.to_dict(True)

    def test_zero_steps_rest_cutoff_and_validation(self,engine):
        r=run(engine,num_steps=0);l=r.algorithmic_energy_ledger
        assert l['substeps']==0 and l['steps']==[0] and l['worst_step'] is None
        assert all(x==0 for x in l['final']['channels'].values())
        assert l['final']['closure']['residual_abs']==0.
        r.to_dict(True);r.to_dict(False)
        before=set(TaichiEngine._field_cache)
        for kw in (dict(record_state=False),dict(record_energy_ledger=1),dict(num_steps=True),
                   dict(num_steps=-1),dict(spring_k=-1),dict(dashpot=-1),dict(drag=float('inf'))):
            with pytest.raises(ValueError):run(engine,**kw)
        assert set(TaichiEngine._field_cache)==before
        with pytest.raises(ValueError):run(engine,num_steps=0,record_energy_ledger=False)
        tiny=graph(2);tiny.nodes[1].position=np.array([1e-13,0.,0.])
        r=run(engine,tiny,num_steps=0)
        assert r.algorithmic_energy_ledger['near_cutoff_any'] is True
        assert r.algorithmic_energy_ledger['final']['potential_energy']>0

    @pytest.mark.parametrize('explicit',[False,True])
    def test_json_roundtrip_and_tamper(self,engine,tmp_path,explicit):
        kw=dict(nodal_masses=[2.,3.,5.]) if explicit else {}
        r=run(engine,external_force=np.array([[0.,0,0],[0.,0,0],[1.,0,0]]),**kw)
        for detailed in (False,True):
            p=tmp_path/'ledger.json';r.save(p,detailed=detailed)
            restored=SimResult.load(p)
            assert restored.to_dict(detailed)==r.to_dict(detailed)
            good=r.to_dict(detailed)
            for count in (99, True, 3.0, None):
                bad=copy.deepcopy(good);bad['n_nodes']=count;p.write_text(json.dumps(bad))
                with pytest.raises(ValueError):SimResult.load(p)
            mutations=[lambda v:v.update(schema='native_axial_discrete_ledger/99'),
                lambda v:v.update(extra=1), lambda v:v.update(availability='unavailable'),
                lambda v:v['final']['channels'].__setitem__('drag_delta_K',1.),
                lambda v:v['final']['closure'].__setitem__('residual_abs',0.123),
                lambda v:v['final'].__setitem__('total_energy',1.),
                lambda v:v['physical_work']['external_work'].update(status='available',value=0.),
                lambda v:v['masses'].__setitem__(0,99.),
                lambda v:v.update(node_ids=[2,1,0]), lambda v:v.update(units={'mass':'kg'}),
                lambda v:v.update(substeps=99),lambda v:v['final']['geometric_external_work'].__setitem__('minus_external_kick',1.)]
            if detailed:mutations += [lambda v:v['steps'].__setitem__(1,99),lambda v:v['samples'][0]['channels'].__setitem__('drag_delta_K',1.)]
            for mutate in mutations:
                bad=copy.deepcopy(good);mutate(bad['algorithmic_energy_ledger']);p.write_text(json.dumps(bad))
                with pytest.raises(ValueError):SimResult.load(p)
        old=run(engine,record_energy_ledger=False);old.save(tmp_path/'old.json',True)
        assert SimResult.load(tmp_path/'old.json').algorithmic_energy_ledger is None

    def test_high_precision_physical_energies_and_compensated_mass(self,engine):
        result=run(engine,graph(2),nodal_masses=[2.,4.],external_force=np.array([[0.,0,0],[1.,0,0]]),num_steps=2,save_interval=1)
        state=result.dynamics_state
        with localcontext() as ctx:
            ctx.prec=100
            for frame,velocity,k,u in zip(state['positions'],state['velocities'],state['kinetic_energy'],state['potential_energy']):
                d=lambda x:Decimal.from_float(float(x))
                exact_k=sum(d(m)*sum(d(c)**2 for c in row)/2 for m,row in zip([2.,4.],velocity))
                exact_l=sum((d(b)-d(a))**2 for a,b in zip(frame[0],frame[1])).sqrt()
                exact_u=(exact_l-1)**2
                assert abs(d(k)-exact_k)<=d(math.ulp(k))*4
                # U is evaluated using the frozen binary64 norm/strain expression;
                # this independent physical check explicitly includes coordinate cancellation.
                dx=float(frame[1][0]-frame[0][0])
                # The collinear norm reduces to abs(dx). Bound subtraction and
                # sqrt/square roundoff by 4 ULP(dx), then propagate (L-1)^2.
                dl=d(math.ulp(dx))*4
                energy_bound=2*abs(exact_l-1)*dl+dl*dl+d(math.ulp(u))*4
                assert abs(d(u)-exact_u)<=energy_bound
            for mass,v in ((1e-200,1e200),(1e200,1e-200)):
                k=_checked_kinetic_energy([mass],[[v,0.,0.]])
                exact=d(mass)*d(v)**2/2
                assert abs(d(k)-exact)<=d(math.ulp(k))*4

    def test_small_undamped_defect_refinement(self,engine):
        values=[]
        for dt in (.01,.005):
            r=run(engine,graph(2),dashpot=0.,drag=0.,dt=dt,num_steps=round(.1/dt),
                external_force=np.array([[0.,0,0],[1.,0,0]]),save_interval=100)
            values.append(abs(r.algorithmic_energy_ledger['final']['edge_drift_delta']))
        assert values[1]<.65*values[0]


class TestPostNativeArithmetic:
    def test_binary64_subnormal_after_native(self,engine):
        import struct
        from fibernet.sim._energy_accounting import parse_binary64
        # Force a native call before auditing subnormal bit patterns.
        run(engine,num_steps=1)
        bits=lambda x:struct.unpack('>Q',struct.pack('>d',x))[0]
        minimum=struct.unpack('>d',struct.pack('>Q',1))[0]
        assert rational(minimum)==Q(1,2**1074)
        assert bits(nearest(Q(1,2**1074)))==1
        assert bits(upward(Q(1,2**1075)))==1
        assert bits(closure(dict.fromkeys(CHANNELS,0.),0.,0.)['bound'])==4
        assert bits(parse_binary64('5e-324'))==1
        assert quantum(minimum)==Q(1,2**1075)
        for value in [Q(1,2**1075),Q(1,2**1076)]:
            with pytest.raises(ValueError):nearest(value)
        # A finite subnormal is never mistaken for zero during validation.
        ch=dict.fromkeys(CHANNELS,0.);ch[CHANNELS[0]]=minimum
        rec=closure(ch,0.,minimum)
        validate_closure(rec,ch,0.,minimum)
        rec['bound']=0.
        with pytest.raises(ValueError):validate_closure(rec,ch,0.,minimum)

    def test_initial_and_zero_summary_geometric_tamper(self,engine):
        from fibernet.sim._energy_accounting import ledger_summary
        from fibernet.sim.accelerated import _dynamics_state_summary
        r=run(engine,num_steps=0)
        for detail in (False,True):
            state=r.dynamics_state if detail else _dynamics_state_summary(r.dynamics_state)
            original=r.algorithmic_energy_ledger if detail else ledger_summary(r.algorithmic_energy_ledger)
            for key in ('value','subtraction_bound','minus_external_kick'):
                bad=copy.deepcopy(original)
                row=bad['final']; work=row['geometric_external_work']
                if key=='subtraction_bound':work[key]=1.
                else:work['value']=work['minus_external_kick']=1.
                work['bound']=upward(rational(work['subtraction_bound'])+quantum(work['value']))
                work['difference_bound']=upward(rational(work['subtraction_bound'])+quantum(work['minus_external_kick'])+quantum(row['channels']['external_kick_delta_K']))
                if detail:bad['samples'][-1]=copy.deepcopy(row)
                with pytest.raises(ValueError):validate_ledger(bad,state)

    def test_rounding_boundaries_after_native(self,engine):
        import random
        import struct
        run(engine,num_steps=1)
        bits=lambda x:struct.unpack('>Q',struct.pack('>d',x))[0]
        f=lambda x:struct.unpack('>d',struct.pack('>Q',x))[0]
        rng=random.Random(7913)
        for lo in [0,1,2,2**52-1,2**52,0x3ff0000000000000,0x7feffffffffffffe]+[rng.randrange(0x7fefffffffffffff) for _ in range(100)]:
            a,b=rational(f(lo)),rational(f(lo+1));mid=(a+b)/2
            expected=lo if lo%2==0 else lo+1
            if expected:
                assert bits(nearest(mid))==expected
                assert bits(nearest(-mid))==(expected|(1<<63))
            else:
                with pytest.raises(ValueError):nearest(mid)
            assert bits(nearest(mid+(b-a)/4))==lo+1
            if lo:assert bits(nearest(mid-(b-a)/4))==lo
            assert rational(upward(mid))>=mid
        maximum=rational(f(0x7fefffffffffffff));q=quantum(f(0x7fefffffffffffff))
        assert bits(nearest(maximum+q/2))==0x7fefffffffffffff
        with pytest.raises(ValueError):nearest(maximum+q)
        with pytest.raises(ValueError):upward(maximum+q/2)


class TestAdditionalStageSemantics:
    def test_unsampled_preprojection_cutoff(self,engine):
        r=run(engine,graph(2),stiffness=0.,dashpot=0.,drag=0.,dt=1.,num_steps=1,
            fixed_nodes=[1],external_force=np.array([[0.,0,0],[-1.,0,0]]))
        assert r.dynamics_state['near_cutoff']==[False,False]
        assert r.algorithmic_energy_ledger['near_cutoff_any'] is True
        assert r.algorithmic_energy_ledger['final']['channels']['fixed_projection_delta_H']<0

    def test_combined_dashpot_cross_term(self,engine):
        m=np.array([2.,4.]);force=np.array([[0.,0,0],[1.,0,0]])
        r=run(engine,graph(2),nodal_masses=m,external_force=force,drag=0.,num_steps=3,save_interval=1)
        x=np.asarray(r.dynamics_state['positions'][-2]);v=np.asarray(r.dynamics_state['velocities'][-2])
        d=x[0]-x[1];length=np.linalg.norm(d);direction=d/(length+1e-6)
        ps=-2.*(length-1)*d/length*.01
        pd=-10.*np.dot(v[0]-v[1],direction)*direction*.01
        cross=sum(Q(float(a))*Q(float(b))/Q(float(mass)) for mass in m for a,b in zip(ps,pd))
        assert cross!=0
        # The exact kinetic identity contains this cross term. Independent
        # spring-only and dashpot-only changes therefore cannot be simply added.
        energy=lambda vel:sum(Q(float(mass))*sum(c*c for c in row)/2 for mass,row in zip(m,vel))
        old=[[Q(float(c)) for c in row] for row in v]
        spring=[[old[i][j]+(1 if i==0 else -1)*Q(float(ps[j]))/Q(float(m[i])) for j in range(3)] for i in range(2)]
        damp=[[old[i][j]+(1 if i==0 else -1)*Q(float(pd[j]))/Q(float(m[i])) for j in range(3)] for i in range(2)]
        both=[[spring[i][j]+damp[i][j]-old[i][j] for j in range(3)] for i in range(2)]
        assert energy(both)-energy(spring)-energy(damp)+energy(old)==cross
        assert r.algorithmic_energy_ledger['physical_work']['dashpot_work']['reason']=='combined_kick_not_decomposed'


class TestLedgerJsonParser:
    def test_bounded_exact_parser_after_native(self,engine):
        import struct
        from fibernet.sim._energy_accounting import parse_binary64
        run(engine,num_steps=1)
        bits=lambda x:struct.unpack('>Q',struct.pack('>d',x))[0]
        assert bits(parse_binary64('-0.0'))==1<<63
        assert bits(parse_binary64('-0e999999999'))==1<<63
        assert bits(parse_binary64('0e-999999999'))==0
        assert bits(parse_binary64('4.9406564584124654e-324'))==1
        assert bits(parse_binary64('2.4703282292062328e-324'))==1
        for token in ('1e999999999','1e-999999999','1e309','1e-325','2.4703282292062327e-324','NaN','Infinity'):
            with pytest.raises(ValueError):parse_binary64(token)
