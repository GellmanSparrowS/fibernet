"""AST-isolated actual Python orchestration with fake fields; no Taichi/FiberNet import."""
import ast,pathlib,time,json,hashlib,types,sys,dataclasses,copy
import numpy as np
from _source_check_support import run_source_check
from test_taichi_event_compat_source import restore_historical_save

def peak_rss_kib():
    try:
        import resource
    except ModuleNotFoundError as exc:
        if exc.name != "resource":
            raise
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1024 if sys.platform == "darwin" else rss


def run_checks():
    D=pathlib.Path(__file__).resolve().parent;start=time.process_time();checks=[]
    BASE=(D/'fixtures/observer_prior71_accelerated.txt').read_text(encoding='utf-8');NEW=(D.parent/'fibernet/sim/accelerated.py').read_text(encoding='utf-8');assert hashlib.sha256(BASE.encode()).hexdigest()=='fe69dd2e2624e061597fab00ee33e254a109e76d50baae936773d43349f3f74d'
    def check(name,condition):
     assert condition,name
     checks.append(name)
    def parse(txt):return ast.parse(txt)
    def kernels(tree):return [ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and any(ast.unparse(d)=='ti.kernel' for d in n.decorator_list)]
    check('all native kernel ASTs unchanged',kernels(parse(BASE))==kernels(parse(NEW)))
    oldnodes={n.name:n for n in parse(BASE).body if hasattr(n,'name')};newnodes={n.name:n for n in parse(NEW).body if hasattr(n,'name')}
    check('SimResult unchanged outside sealed ledger save repair',ast.dump(oldnodes['SimResult'])==ast.dump(restore_historical_save(newnodes['SimResult'])))
    oldmethod=next(n for n in oldnodes['TaichiEngine'].body if isinstance(n,ast.FunctionDef) and n.name=='dynamics');newmethod=next(n for n in newnodes['TaichiEngine'].body if isinstance(n,ast.FunctionDef) and n.name=='dynamics')
    check('all old positional arguments unchanged',ast.dump(oldmethod.args.args[0])==ast.dump(newmethod.args.args[0]) and [ast.dump(x) for x in oldmethod.args.args]==[ast.dump(x) for x in newmethod.args.args] and [ast.dump(x) for x in oldmethod.args.defaults]==[ast.dump(x) for x in newmethod.args.defaults])
    check('keyword-only addition preserves prior ordering/defaults',[x.arg for x in newmethod.args.kwonlyargs]==[x.arg for x in oldmethod.args.kwonlyargs]+['state_observer'] and [ast.dump(x) for x in newmethod.args.kw_defaults[:-1]]==[ast.dump(x) for x in oldmethod.args.kw_defaults])
    class Result:
     def __init__(self,**kwargs):self.__dict__.update(kwargs)
     def compute_detailed(self,*a,**k):pass
    class Field:
     def __init__(self):self.a=None;self.exports=0
     def from_numpy(self,a):self.a=np.array(a,copy=True)
     def to_numpy(self):self.exports+=1;return self.a.copy()
     def __setitem__(self,k,v):self.a=v
     def __getitem__(self,k):return self.a
    
    def load(txt,label):
     tree=parse(txt);wanted={'_native_axial_event_dataclass','_finite_real_array','_strict_state_ids','_state_units','_validated_fixed_rows','_resolve_nodal_masses','_normalize_spring_stiffness','NativeAxialObserverEvent','_emit_axial_observation'}
     nodes=[n for n in tree.body if getattr(n,'name',None) in wanted];method=next(n for n in next(n for n in tree.body if getattr(n,'name',None)=='TaichiEngine').body if isinstance(n,ast.FunctionDef) and n.name=='dynamics');nodes.append(method)
     mod=types.ModuleType(label);sys.modules[label]=mod;ns=mod.__dict__;ns.update(sys=sys,np=np,time=time,os=__import__('os'),dataclass=dataclasses.dataclass,field=dataclasses.field,Optional=__import__('typing').Optional,SimResult=Result,TaichiEngine=types.SimpleNamespace(_field_cache={}),_check_taichi_owner=lambda:None,_ensure_taichi=lambda*a:None,_prepare_displacement_schedule=lambda*a:{},_validate_dynamics_state=lambda s,*a:s,_checked_kinetic_energy=lambda m,v:float(.5*np.sum(m[:,None]*v*v)))
     compiled=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[]);exec(compile(ast.fix_missing_locations(compiled),'<AST-isolated actual orchestration>','exec'),ns);return ns
    
    BN=load(BASE,'baseline_isolated');CN=load(NEW,'candidate_isolated')
    
    def setup(ns,ids=(17,83)):
     x=np.array([[0.,0.,0.],[1.,0.,0.]]);edges=np.array([[0,1]])
     ns['_graph_to_arrays']=lambda g:(x.copy(),edges.copy(),list(ids),None)
     ns['_element_data']=lambda p,e:(np.array([1.]),None)
     fields={k:Field() for k in ['ti_pos','ti_vel','ti_vel_old','ti_mass','ti_mass_mode','ti_edges','ti_L0','ti_k','ti_fixed','ti_ext','ti_pos0','ti_sched','ti_sched_mask','ti_dt','ti_dashpot','ti_drag','ti_ramp']}
     count={'steps':0}
     def prescribed_mock_step():
      # Prescribed mock sentinel only, no integration or physical force calculation.
      count['steps']+=1;fields['ti_pos'].a=x.copy();fields['ti_pos'].a[1,2]=count['steps']*.01
      fields['ti_vel'].a=np.zeros_like(x);fields['ti_vel'].a[1,2]=count['steps']*.1
     fields['substep']=prescribed_mock_step
     ns['TaichiEngine']._field_cache={('mass-state-1',3,2,1):fields}
     return types.SimpleNamespace(arch='cpu',num_threads=1),types.SimpleNamespace(num_nodes=2,edges={91:None}),fields,count
    
    def run(ns,obs='absent',record=False,ids=(17,83),**extra):
     self,g,f,c=setup(ns,ids);kw=dict(num_steps=4,save_interval=2,dt=.125,dashpot=0.,drag=0.,damping=0.,record_state=record,state_units={'length':'m','time':'s','mass':'kg','frame':'world_xyz'})
     if obs!='absent':kw['state_observer']=obs
     kw.update(extra)
     try:r=ns['dynamics'](self,g,**kw)
     except Exception as e:return None,f,c,e
     return r,f,c,None
    
    def normalized(r):
     def norm(v):
      if isinstance(v,np.ndarray):return v.tolist()
      if isinstance(v,dict):return {k:norm(vv) for k,vv in v.items() if k!='time_seconds'}
      if isinstance(v,(tuple,list)):return [norm(a) for a in v]
      return v
     return norm(vars(r))
    for record in [False,True]:
     baseline,bf,bc,e=run(BN,record=record);assert e is None
     default,df,dc,e=run(CN,record=record);assert e is None
     explicit,ef,ec,e=run(CN,None,record);assert e is None
     check(f'default result unchanged record={record}',normalized(default)==normalized(baseline)==normalized(explicit))
     check(f'default no additional exports record={record}',{k:f.exports for k,f in bf.items() if isinstance(f,Field)}=={k:f.exports for k,f in df.items() if isinstance(f,Field)}=={k:f.exports for k,f in ef.items() if isinstance(f,Field)})
     events=[];observed,of,oc,e=run(CN,events.append,record);assert e is None
     check(f'observer numeric mock result parity record={record}',normalized(observed)==normalized(baseline))
     check(f'N+1 events including t0/final record={record}',[e.step for e in events]==list(range(5)) and [e.time for e in events]==[i*.125 for i in range(5)])
     check(f'noncontiguous stable IDs record={record}',all(e.node_ids==(17,83) for e in events))
     check(f'observer adds exactly N+1 x/v downloads record={record}',of['ti_pos'].exports-df['ti_pos'].exports==5 and of['ti_vel'].exports-df['ti_vel'].exports==5)
     check(f'event schema/units detached record={record}',all(e.schema=='native_axial_observer/1' and e.units==(('frame','world_xyz'),('length','m'),('mass','kg'),('time','s')) for e in events))
     for i,event in enumerate(events):
      for arr in [event.positions,event.velocities]:
       try:arr.setflags(write=True)
       except ValueError:pass
       else:raise AssertionError('Immutable byte buffer allowed writable flag')
       try:arr[0,0]=99
       except ValueError:pass
       else:raise AssertionError('Mutation accepted')
      try:event.step=90
      except dataclasses.FrozenInstanceError:pass
      else:raise AssertionError('Frozen event mutated')
     check(f'owned immutable snapshots record={record}',all(a.dtype==np.dtype('float64') and a.shape==(2,3) for event in events for a in (event.positions,event.velocities)) and all(event.positions.tobytes()==np.array([[0.,0.,0.],[1.,0.,event.step*.01]],dtype=np.float64).tobytes() for event in events) and not np.shares_memory(events[0].positions,events[1].positions) and not np.shares_memory(events[-1].positions,observed.deformed_positions) and events[0].positions[1,2]==0.)
     if record:
      check('observer/native mock samples agree',all(np.array_equal(events[step].positions,observed.dynamics_state['positions'][i]) and np.array_equal(events[step].velocities,observed.dynamics_state['velocities'][i]) for i,step in enumerate(observed.dynamics_state['steps'])))
    for failure_step in [0,3]:
     called=[];sentinel=RuntimeError('exact callback exception')
     def fail(event):
      called.append(event.step)
      if event.step==failure_step:raise sentinel
     r,f,c,e=run(CN,fail)
     check(f'original callback exception and stop step{failure_step}',e is sentinel and c['steps']==failure_step and called==list(range(failure_step+1)) and r is None)
    for kw in [dict(state_observer=7),dict(state_observer=lambda e:None,record_energy_ledger=True),dict(state_observer=lambda e:None,displacement_schedule={0:[(0,[0,0,0])]})]:
     obs=kw.pop('state_observer');r,f,c,e=run(CN,obs,**kw)
     check('reject unsupported controls before mock step '+str(len(checks)),isinstance(e,(TypeError,ValueError)) and c['steps']==0 and sum(v.exports for v in f.values() if isinstance(v,Field))==0)
    r,f,c,e=run(CN,lambda e:None,ids=(17,17));check('enabled duplicate IDs rejected',isinstance(e,ValueError) and c['steps']==0)
    r,f,c,e=run(CN,lambda e:None,state_units={'length':'m'});check('enabled invalid units rejected',isinstance(e,ValueError) and c['steps']==0)
    check('no solver/native imports','taichi' not in sys.modules and not any(k=='fibernet' or k.startswith('fibernet.') for k in sys.modules))
    # In-file helper is already a bound source dependency; mutation changes that identity.
    contract=(D.parent/'fibernet/ml/_generate_contract.py').read_text(encoding='utf-8');check('accelerated source already generation-bound',"'sim/accelerated.py'" in contract)
    mutated=NEW.replace("schema='native_axial_observer/1'","schema='native_axial_observer/2'",1);check('helper-only mutation changes generation dependency bytes',hashlib.sha256(NEW.encode()).hexdigest()!=hashlib.sha256(mutated.encode()).hexdigest())
    report=dict(status='PASS_SOURCE_AND_ORCHESTRATION_MOCKS_ONLY',checks=checks,count=len(checks),CPU_seconds=time.process_time()-start,RSS_KiB=peak_rss_kib(),native_imports=False,native_calls=0,physics_trajectories=0,limitations=['AST-isolated actual method with fake fields/prescribed mock sentinels, not native parity','SimResult mocked; unchanged actual SimResult AST checked','No full package imports or production serialization runtime','Generation hash binding checked, actual checkpoint resume rejection not executed'])
    assert len(checks)==31
    return report

def test_observer_mock_contract():
    assert json.loads(run_source_check(__file__))['count']==31

if __name__=="__main__":
    print(json.dumps(run_checks()))
