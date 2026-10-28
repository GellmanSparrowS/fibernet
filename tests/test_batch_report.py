"""Opt-in batch diagnostics preserve CSV behavior and failure provenance."""
import csv,json,os,tempfile,unittest,errno
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import fibernet.easy as easy
ns=vars(easy)
class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.csv=self.root/'results.csv';self.report=self.root/'report.json'
  self.generated=[]
  def gen(**kw):
   self.generated.append(kw)
   if kw['unit']=='badgen':raise ValueError('unsupported geometry')
   return SimpleNamespace(num_nodes=3,num_edges=2,unit=kw['unit'])
  def sim(g,**kw):
   if g.unit=='badsim':raise RuntimeError('simulation did not converge')
   return SimpleNamespace(max_force=1.,max_displacement=2.,energy=3.)
  bindings=patch.dict(ns,pattern_2d=gen,simulate=sim);bindings.start();self.addCleanup(bindings.stop)
 def runbatch(self,cfg,**kw):return ns['batch_simulate'](cfg,str(self.csv),**kw)
 def test_default_unchanged(self):
  out=self.runbatch([{'unit':'good'},{'unit':'badsim'}]);self.assertEqual(out,str(self.csv));self.assertFalse(self.report.exists());self.assertEqual(list(csv.DictReader(self.csv.read_text().splitlines()))[0]['id'],'0')
 def test_mixed_order_and_links(self):
  cfg=[{'unit':'badgen'},{'unit':'good','grid':(2,3)},{'unit':'badsim'},{'unit':'good'}]
  self.runbatch(cfg,report_path=self.report);r=json.loads(self.report.read_text());self.assertEqual(r['success_indices'],[1,3]);self.assertEqual([c['index'] for c in r['cases']],[0,1,2,3]);self.assertEqual([c['csv_row'] for c in r['cases']],[None,0,None,1]);self.assertEqual(r['cases'][0]['stage'],'generation');self.assertEqual(r['cases'][2]['error_type'],'RuntimeError');self.assertNotIn('traceback',self.report.read_text());self.assertEqual(r['cases'][1]['config']['grid'],[2,3])
 def test_all_success(self):
  self.runbatch([{}],report_path=self.report);r=json.loads(self.report.read_text());self.assertTrue(r['csv_written']);self.assertEqual(r['failure_count'],0)
 def test_all_fail_preserves_old_csv(self):
  self.csv.write_bytes(b'old');
  with self.assertRaisesRegex(RuntimeError,'All batch'):self.runbatch([{'unit':'badgen'}],report_path=self.report)
  r=json.loads(self.report.read_text());self.assertFalse(r['csv_written']);self.assertIsNone(r['csv_sha256']);self.assertEqual(self.csv.read_bytes(),b'old')
 def test_empty(self):
  with self.assertRaisesRegex(RuntimeError,'All batch'):self.runbatch([],report_path=self.report)
  self.assertEqual(json.loads(self.report.read_text())['cases'],[]);self.assertFalse(self.csv.exists())
 def test_bad_config_preflight(self):
  for cfg in ([{'opaque':object()}],[{'x':float('nan')}],[{1:'key'}]):
   with self.assertRaisesRegex(ValueError,'JSON-safe'):self.runbatch(cfg,report_path=self.report)
  self.assertEqual(self.generated,[]);self.assertFalse(self.csv.exists())
 def test_same_path(self):
  with self.assertRaisesRegex(ValueError,'differ'):self.runbatch([{}],report_path=self.csv)
  self.assertEqual(self.generated,[])
 def test_symlink_alias(self):
  self.csv.write_text('old');alias=self.root/'alias'
  try:alias.symlink_to(self.csv)
  except OSError as exc:
   if exc.errno in (errno.EPERM,errno.EACCES,errno.ENOTSUP) or getattr(exc,'winerror',None)==1314:
    self.skipTest('filesystem or user cannot create symlinks: '+str(exc))
   raise
  with self.assertRaisesRegex(ValueError,'differ'):self.runbatch([{}],report_path=alias)
  self.assertEqual(self.csv.read_text(),'old')
 def test_hardlink_alias(self):
  self.csv.write_text('old');alias=self.root/'alias';os.link(self.csv,alias)
  with self.assertRaisesRegex(ValueError,'differ'):self.runbatch([{}],report_path=alias)
  self.assertEqual(self.csv.read_text(),'old')
 def test_replace_failure(self):
  self.report.write_text('old')
  with patch('os.replace',side_effect=OSError('replace unavailable')):
   with self.assertRaisesRegex(RuntimeError,'report could not be written'):self.runbatch([{}],report_path=self.report)
  self.assertEqual(self.report.read_text(),'old');self.assertEqual(list(self.root.glob('.batch-report-*')),[]);self.assertTrue(self.csv.exists())
 def test_progress_failure_has_one_case_with_actual_csv_link(self):
  def output(message):
   if 'max_force=' in message:raise OSError('progress unavailable')
  with patch('builtins.print',side_effect=output):self.runbatch([{}],report_path=self.report)
  r=json.loads(self.report.read_text());self.assertEqual(len(r['cases']),1);self.assertEqual(r['cases'][0]['stage'],'progress');self.assertEqual(r['cases'][0]['csv_row'],0);self.assertTrue(r['csv_written']);self.assertEqual(r['success_indices'],[]);self.assertEqual(r['csv_indices'],[0]);self.assertEqual(r['success_count'],0)
 def test_missing_parent_preflight(self):
  with self.assertRaisesRegex(ValueError,'parent'):self.runbatch([{}],report_path=self.root/'missing/report.json')
  self.assertEqual(self.generated,[])
 def test_serialization_preserves_old(self):
  self.report.write_text('old')
  with self.assertRaises(TypeError):ns['_write_batch_report'](self.report,{'x':object()})
  self.assertEqual(self.report.read_text(),'old');self.assertEqual(list(self.root.glob('.batch-report-*')),[])
if __name__=='__main__':unittest.main()
