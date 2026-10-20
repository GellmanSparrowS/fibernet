"""Source/mocked sampling contracts; no native or physics qualification."""
import copy,json,math,tempfile,unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from fibernet import simulate
from fibernet.integrations import CGRequest,prepare_cg,lammps_backend as b
from fibernet.integrations import _cg_network_profile as net
from fibernet.sim import _lammps_result as adapter
from fibernet.sim.accelerated import SimResult
from fibernet.core.structure_graph import StructureGraph
from test_lammps_cg import fixture,RUNTIME

def network(steps=1600000,every=12500,thermo=2500):
 g,q=fixture(False)
 q=replace(q,run=b.RunSpec(1.25e-7,steps,every, q.run.box_m),profile=net.NAME,thermo_every=thermo,pair_encoding='static-groups-zero-cutoff/1')
 return g,q,prepare_cg(g,q)
def raw(p):
 c=json.loads(p.contract_json);points=tuple(tuple(n['position']) for n in json.loads(p.graph_json)['nodes']);z=((0.,0.,0.),)*len(p.node_ids)
 return b.LAMMPSRunResult(p.identity,tuple(b.TrajectoryFrame(i,i*p.timestep_s,p.node_ids,points,z,z) for i in range(0,p.steps+1,p.sample_every)),tuple(b.ThermoFrame(i,0.,0.,0.,(0.,)*6) for i in range(0,p.steps+1,c['thermo_every'])),'')
def texts(p):
 r=raw(p);c=json.loads(p.contract_json);dump=['ITEM: UNITS','si']
 for f in r.trajectory:
  dump+=['ITEM: TIME',str(f.time_s),'ITEM: TIMESTEP',str(f.step),'ITEM: NUMBER OF ATOMS',str(len(p.node_ids)),'ITEM: BOX BOUNDS ff ff ff']+[' '.join(map(str,row)) for row in c['box_m']]+['ITEM: ATOMS id type x y z vx vy vz fx fy fz']
  dump += [f'{i} {i} '+' '.join(map(str,(*pos,0.,0.,0.,0.,0.,0.))) for i,pos in enumerate(f.positions_m,1)]
 log=['Step Atoms PotEng KinEng TotEng Pxx Pyy Pzz Pxy Pxz Pyz']+[f'{f.step} {len(p.node_ids)} 0 0 0 0 0 0 0 0 0' for f in r.thermo]+['Loop time',p.completion_marker]
 return '\n'.join(dump)+'\n','\n'.join(log)+'\n'
class NetworkTests(unittest.TestCase):
 def test_separate_long_schedules_and_regeneration(self):
  g,q,p=network();self.assertEqual(p,b.validate_prepared(p));c=json.loads(p.contract_json)
  self.assertEqual((c['integration_steps'],c['output_frames'],c['thermo_frames']),(1600000,129,641))
  self.assertIn('thermo 2500\n',p.input_text);self.assertIn('custom 12500 trajectory.dump',p.input_text)
  with self.assertRaises(b.LAMMPSContractError):q.run.checked()
 def test_parser_observation_schedule(self):
  _,_,p=network();dump,log=texts(p);r=b.parse_outputs(p,dump,log,max_output_bytes=net.PROFILE['raw_bytes']);self.assertEqual((len(r.trajectory),len(r.thermo)),(129,641))
  for d,l in [(dump.replace('ITEM: TIMESTEP\n12500','ITEM: TIMESTEP\n12501'),log),(dump,log.replace('2500 4 ','2501 4 ',1))]:
   with self.assertRaises(b.LAMMPSContractError):b.parse_outputs(p,d,l)
 def test_public_projection_serialization(self):
  g,q,p=network();r=adapter.to_sim_result(p,raw(p),RUNTIME,0,True)
  self.assertEqual(len(r.dynamics_state['steps']),129);self.assertEqual(r.dynamics_state['steps'][-1],1600000)
  self.assertEqual(r.dynamics_state['observation_scope'],net.SCOPE)
  with tempfile.TemporaryDirectory() as d:
   for detail in (True,False):
    f=Path(d)/'result.json';r.save(f,detailed=detail);self.assertEqual(SimResult.load(f,profile=net.NAME).to_dict(detail),r.to_dict(detail))
    with self.assertRaises(ValueError):SimResult.load(f)
  class Task:
   def __init__(self,p,*a):self.result=raw(p);self.error=None
   def start(self):pass
   def wait(self):return b.TaskState.SUCCEEDED
   def cancel(self):pass
  with patch.object(b,'LAMMPSTask',Task):
   result=simulate(g,backend='lammps',mode='dynamics',cg_request=q,runtime=RUNTIME,workdir='unused',limits=b.ResourceLimits(),record_state=True)
  self.assertEqual(result.dynamics_state['thermo_frames'],641)
 def test_refusal_contracts(self):
  g,q,p=network()
  for change in ({'profile':'bad'},{'pair_encoding':None},{'thermo_every':True},{'thermo_every':0},{'thermo_every':3000},{'run':replace(q.run,steps=1600001)},{'run':replace(q.run,sample_every=1)},{'thermo_every':1}):
   with self.assertRaises((ValueError,b.LAMMPSContractError)):prepare_cg(g,replace(q,**change))
  for key,value in [('output_frames',128),('integration_steps',True),('observation_scope','all-step safe')]:
   c=json.loads(p.contract_json);c[key]=value
   with self.assertRaises(b.LAMMPSContractError):b.validate_prepared(replace(p,contract_json=b._json(c)))
  rr=adapter.to_sim_result(p,raw(p),RUNTIME,0,True);rr.metadata['model_contract']['profile']['max_nodes']=9999;rr.metadata['model_contract_sha256']=b._digest(b._json(rr.metadata['model_contract']))
  with self.assertRaises(ValueError):rr.to_dict(True)
 def test_network_count_caps_and_runzero(self):
  g=StructureGraph(dimension=3)
  for i in range(898):g.add_node([i/1000,0,0])
  q=CGRequest('m',2,{i:1+i*1e-8 for i in g.nodes},{},{},897,{},b.RunSpec(1e-7,0,1,((-2,2),)*3),pair_encoding='static-groups-zero-cutoff/1',profile=net.NAME,thermo_every=1)
  p=prepare_cg(g,q);self.assertEqual(len(p.node_ids),898);b.validate_prepared(p)
  with self.assertRaises(b.LAMMPSContractError):prepare_cg(g,replace(q,profile=None,thermo_every=None))
  for i in range(898,1025):g.add_node([i/1000,0,0])
  with self.assertRaises(b.LAMMPSContractError):prepare_cg(g,replace(q,masses_kg={i:1 for i in g.nodes}))
 def test_task_forecast_limits_no_runtime_launch(self):
  _,_,p=network();limits=b.ResourceLimits(60,120,2*1024**3,64*1024**2,64*1024**2)
  with patch.object(b.RuntimeSpec,'checked',lambda s:s):
   task=b.LAMMPSTask(p,RUNTIME,'unused',limits)
   self.assertEqual(task.output_forecast['output_frames'],129)
   with self.assertRaises(ValueError):b.LAMMPSTask(p,RUNTIME,'unused',replace(limits,output_bytes=1024))
  for bad in (replace(limits,cpu_seconds=61),replace(limits,wall_seconds=121),replace(limits,output_bytes=64*1024**2+1)):
   with self.assertRaises(b.LAMMPSContractError):bad.checked(profile=net.NAME)
if __name__=='__main__':unittest.main()
