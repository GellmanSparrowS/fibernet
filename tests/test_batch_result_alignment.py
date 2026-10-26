"""Batch identity alignment and nested result-save regressions."""
import unittest,tempfile,contextlib,io
from pathlib import Path
from unittest.mock import patch
from fibernet.utils.batch import parameter_study
from fibernet.sim._viscoelastic_fem import ViscoelasticResult,_UNITS,_hash

def fixture():
 state=dict(schema='fibernet.viscoelastic_state/1',node_ids=[0,1],edge_ids=[0],units=_UNITS,fingerprint='synthetic-container-only',displacement=[0.]*6,section_strain=[[0.,0.,0.]],time=0.,stored_energy=0.,cumulative={k:0. for k in ('physical_dissipation','numerical_dissipation','work')});state['checksum']=_hash(state)
 return ViscoelasticResult(dict(schema='fibernet.viscoelastic_result/1',node_ids=[0,1],edge_ids=[0],times=[0.],displacements=[[[0.,0.,0.],[0.,0.,0.]]],reactions=[[[0.,0.,0.],[0.,0.,0.]]],section_strain=[[[0.,0.,0.]]],axial_stress=[[0.]],axial_force=[[0.]],bending_moment=[[[0.,0.]]],free_residual=[0.],energies={k:[0.] for k in ('stored','physical_dissipation','numerical_dissipation','work','balance_residual')},metadata=dict(units=_UNITS,fingerprint='synthetic-container-only'),final_state=state))
class BatchTests(unittest.TestCase):
 def study(self,parallel=False):
  def gen(x):
   if x==1:raise ValueError('generation rejected1')
   return x
  def sim(x):
   if x==2:raise RuntimeError('simulation rejected2')
   return x*10
  with contextlib.redirect_stdout(io.StringIO()):return parameter_study({'x':[0,1,2,3]},gen,sim,parallel=parallel,max_workers=2)
 def test_middle_failures_alignment_and_distinct_index_axes(self):
  s=self.study();self.assertEqual(s['params'],[{'x':0},{'x':2},{'x':3}]);self.assertEqual(s['results'],[0,30]);self.assertEqual(s['errors'][0]['index'],1)
  self.assertEqual(s['success_indices'],[0,2]);self.assertEqual(s['successful_params'],[{'x':0},{'x':3}]);self.assertEqual(s['generation_errors'],[dict(index=1,params={'x':1},error_type='ValueError',error='generation rejected1')])
  self.assertEqual([s['params'][i] for i in s['success_indices']],s['successful_params'])
 def test_parallel_same_success_mapping(self):
  a,b=self.study(),self.study(True)
  for k in ('params','results','success_indices','successful_params','generation_errors'):self.assertEqual(a[k],b[k])
  self.assertEqual(a['errors'][0]['index'],b['errors'][0]['index'])
 def test_all_generations_fail(self):
  def gen(**kw):raise ValueError('no graph')
  with contextlib.redirect_stdout(io.StringIO()):s=parameter_study({'x':[1,2]},gen,lambda x:x)
  self.assertEqual(s['params'],[]);self.assertEqual(s['results'],[]);self.assertEqual(s['success_indices'],[]);self.assertEqual(len(s['generation_errors']),2)
 def test_all_simulations_fail(self):
  def sim(x):raise ValueError('no result')
  with contextlib.redirect_stdout(io.StringIO()):s=parameter_study({'x':[1,2]},lambda x:x,sim)
  self.assertEqual(s['params'],[{'x':1},{'x':2}]);self.assertEqual(s['results'],[]);self.assertEqual(s['successful_params'],[]);self.assertEqual(s['success_indices'],[]);self.assertEqual(len(s['errors']),2)
 def test_all_success_legacy_values_unchanged(self):
  with contextlib.redirect_stdout(io.StringIO()):s=parameter_study({'x':[1,2]},lambda x:x,lambda x:x*10)
  self.assertEqual(s['params'],[{'x':1},{'x':2}]);self.assertEqual(s['results'],[10,20]);self.assertEqual(s['errors'],[]);self.assertEqual(s['generation_errors'],[]);self.assertEqual(s['success_indices'],[0,1])
class SaveTests(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.p=Path(self.tmp.name)
 def test_nested_save_load_roundtrip(self):
  r=fixture();p=self.p/'new/deep/result.json';r.save(p);self.assertEqual(ViscoelasticResult.load(p).to_dict(),r.to_dict())
 def test_existing_parent_and_replacement(self):
  p=self.p/'result.json';p.write_bytes(b'old');r=fixture();r.save(p);self.assertEqual(ViscoelasticResult.load(p).to_dict(),r.to_dict());self.assertEqual(list(self.p.iterdir()),[p])
 def test_replace_failure_preserves_existing_bytes(self):
  p=self.p/'result.json';p.write_bytes(b'old')
  with patch('fibernet.sim._viscoelastic_fem.os.replace',side_effect=OSError('blocked replace')):
   with self.assertRaisesRegex(OSError,'blocked replace'):fixture().save(p)
  self.assertEqual(p.read_bytes(),b'old');self.assertEqual(list(self.p.iterdir()),[p])
 def test_parent_file_error_before_target_creation(self):
  p=self.p/'file';p.write_bytes(b'parentfile')
  with self.assertRaises((FileExistsError,NotADirectoryError)):fixture().save(p/'result.json')
  self.assertEqual(p.read_bytes(),b'parentfile')
if __name__=='__main__':unittest.main()
