"""Opt-in cached NSQ contracts; mocks only, no native qualification."""
import copy,json,math,tempfile,unittest
from pathlib import Path
from dataclasses import replace
from fibernet.integrations import prepare_cg,lammps_backend as b
from fibernet.integrations._lammps_cg import _neighbor_policy,NEIGHBOR_POLICY,GROUP_ENCODING
from fibernet.sim import _lammps_result as adapter
from fibernet.sim.accelerated import SimResult
from test_lammps_cg import fixture,raw,RUNTIME
from test_lammps_cg_mixed_roles import mixed_case,mocked_raw

POLICY={'name':NEIGHBOR_POLICY,'skin_m':.2}
class NeighborTests(unittest.TestCase):
 def test_optin_only_and_regeneration(self):
  g,q=fixture();old=prepare_cg(g,q);p=prepare_cg(g,replace(q,neighbor_policy=POLICY));c=json.loads(p.contract_json)
  self.assertNotIn('neighbor_policy',json.loads(old.contract_json));self.assertEqual(old,prepare_cg(g,replace(q,neighbor_policy=None)))
  self.assertEqual(p,b.validate_prepared(p));self.assertEqual(old.graph_json,p.graph_json);self.assertEqual(old.data_text,p.data_text)
  self.assertIn('neighbor 0.20000000000000001 nsq\n',p.input_text);self.assertIn('neigh_modify every 1 delay 0 check yes once no\n',p.input_text)
  self.assertEqual(c['neighbor_policy']['displacement_group'],'all');self.assertNotIn('include ',p.input_text)
 def test_expanded_communication(self):
  g,q=fixture();q=replace(q,maximum_bond_length_m=1.5,neighbor_policy=POLICY)
  p=prepare_cg(g,q);self.assertIn('comm_modify cutoff 3.2000000000000002\n',p.input_text)
 def test_refuse_bad_input_and_lost_skin(self):
  g,q=fixture()
  bad=[{},[],True,{'name':'bin','skin_m':.2},{**POLICY,'every':1},{'name':NEIGHBOR_POLICY},*[dict(POLICY,skin_m=x) for x in (0,-1,True,float('nan'),float('inf'),1e-300,1e-30,6,1e308)]]
  for v in bad:
   with self.subTest(v=v),self.assertRaises(ValueError):prepare_cg(g,replace(q,neighbor_policy=v))
  with self.assertRaises(ValueError):_neighbor_policy(POLICY,((-5,5),)*3,[1e308])
 def test_strict_serialized_tamper(self):
  g,q=fixture();p=prepare_cg(g,replace(q,neighbor_policy=POLICY));c=json.loads(p.contract_json)
  for key,value in [('name','bin'),('check',False),('every',2),('every',1.),('once',True),('displacement_group','mobile'),('boundary','periodic'),('qualification','validated'),('skin_m',0)]:
   v=copy.deepcopy(c);v['neighbor_policy'][key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):b.validate_prepared(replace(p,contract_json=b._json(v)))
  for v in (None,{},dict(c['neighbor_policy'],extra=1)):
   cc=copy.deepcopy(c);cc['neighbor_policy']=v
   with self.assertRaises(ValueError):b.validate_prepared(replace(p,contract_json=b._json(cc)))
 def test_result_roundtrip_and_rehashed_forgery(self):
  g,q=fixture();p=prepare_cg(g,replace(q,neighbor_policy=POLICY));s=adapter.to_sim_result(p,raw(p),RUNTIME,0,True)
  with tempfile.TemporaryDirectory() as td:
   path=Path(td)/'result.json';s.save(path,detailed=True);self.assertEqual(SimResult.load(path).to_dict(detailed=True),s.to_dict(detailed=True))
   d=json.loads(path.read_text());d['model_contract']['neighbor_policy']['check']=False;d['model_contract_sha256']=b._digest(b._json(d['model_contract']));path.write_text(json.dumps(d))
   with self.assertRaises(ValueError):SimResult.load(path)
 def test_grouped_mixed_network_roundtrip(self):
  g,q,_=mixed_case(GROUP_ENCODING,'cg_network_mixed_sampled_v1');p=prepare_cg(g,replace(q,neighbor_policy=POLICY));self.assertEqual(p,b.validate_prepared(p));s=adapter.to_sim_result(p,mocked_raw(p),RUNTIME,0,True)
  with tempfile.TemporaryDirectory() as td:
   path=Path(td)/'result.json';s.save(path,detailed=True);self.assertEqual(SimResult.load(path,profile=q.profile).to_dict(detailed=True),s.to_dict(detailed=True))
 def test_unified_public_api_keeps_policy(self):
  from fibernet import simulate
  from unittest.mock import patch
  g,q,_=mixed_case(GROUP_ENCODING,'cg_network_mixed_sampled_v1');q=replace(q,neighbor_policy=POLICY)
  class Task:
   def __init__(self,p,*args):self.result=mocked_raw(p);self.error=None
   def start(self):pass
   def wait(self):return b.TaskState.SUCCEEDED
   def cancel(self):pass
  with patch.object(b,'LAMMPSTask',Task):
   s=simulate(g,backend='lammps',mode='dynamics',cg_request=q,runtime=RUNTIME,workdir='unused',limits=b.ResourceLimits(),record_state=True)
  self.assertEqual(s.metadata['model_contract']['neighbor_policy']['name'],NEIGHBOR_POLICY)
if __name__=='__main__':unittest.main()
