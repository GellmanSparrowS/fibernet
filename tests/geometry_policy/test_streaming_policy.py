"""Production batch control flow and CSV checkpoint I/O, inert graph/engine.

No real JSON graph construction, physical solve or result serializer is used
here; production result serialization has a separate isolated harness.
"""
from __future__ import annotations
import ast
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
import weakref
from .support import ROOT,H,NP,load_functions

class BatchPolicy(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.inputs=self.base/'in';self.inputs.mkdir();self.out=self.base/'out'
        self.calls=[];self.refs=[];owner=self
        class Graph: pass
        def parse(raw):
            # This is checked at parse entry, before constructing the next graph.
            self.assertEqual(sum(r() is not None for r in self.refs),0)
            item=json.loads(raw);g=Graph();g.num_nodes=3;g.num_edges=2
            g.edges={9:types.SimpleNamespace(internal_points=[1] if item['internal'] else []),2:types.SimpleNamespace(internal_points=None)}
            self.refs.append(weakref.ref(g));self.calls.append(('parse',item['internal']));return g
        class Engine:
            def __init__(self): owner.calls.append(('engine',))
            def stretch_test(self,g,**kw):
                owner.calls.append(('solve',kw.copy()))
                class Result:
                    max_force=max_stretch=mean_stretch=std_stretch=time_seconds=1.
                    def save(self,path,detailed=False):
                        Path(path).write_text(json.dumps({'stubbed_result':True}));owner.calls.append(('save',path))
                return Result()
        self.ns={**H,'np':NP,'json':json,'Path':Path,'__file__':str(ROOT/'fibernet/easy.py'),'TaichiEngine':Engine}
        tree=ast.parse((ROOT/'fibernet/easy.py').read_text(encoding='utf-8'))
        for n in tree.body:
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id.startswith('_BATCH_') for t in n.targets):
                self.ns[n.targets[0].id]=ast.literal_eval(n.value)
        names=[n.name for n in tree.body if isinstance(n,ast.FunctionDef) and (n.name.startswith('_batch_') or n.name=='batch_simulate_from_json')]
        load_functions('fibernet/easy.py',names,self.ns)
        self.ns['_batch_graph_from_bytes']=parse
        self.run_batch=self.ns['batch_simulate_from_json']
    def put(self,name,internal): (self.inputs/name).write_text(json.dumps({'internal':internal}))
    def rows(self):
        with (self.out/'simulation_results.csv').open() as f:return list(csv.DictReader(f))

    def test_stream_failure_continuation_release_and_same_resume(self):
        self.put('a.json',True);self.put('b.json',False)
        self.run_batch(str(self.inputs),str(self.out),num_steps=1)
        rows=self.rows();self.assertEqual([r['success'] for r in rows],['False','True'])
        self.assertIn('internal_points',rows[0]['error'])
        self.assertFalse((self.out/'a_result.json').exists());self.assertTrue((self.out/'b_result.json').exists())
        self.assertEqual([c[0] for c in self.calls].count('solve'),1)
        self.assertEqual(next(c[1] for c in self.calls if c[0]=='solve')['geometry_policy'],'reject_internal_points')
        binding=json.loads(rows[0]['run_binding'])
        self.assertEqual(binding['parameters']['geometry_policy'],'reject_internal_points')
        helper='fibernet/sim/_taichi_geometry.py'
        self.assertEqual(binding['implementation']['sources'][helper],hashlib.sha256((ROOT/helper).read_bytes()).hexdigest())
        before=(self.out/'simulation_results.csv').read_bytes();self.calls.clear()
        self.run_batch(str(self.inputs),str(self.out),num_steps=1)
        self.assertEqual(self.calls,[]);self.assertEqual((self.out/'simulation_results.csv').read_bytes(),before)

    def test_explicit_chord_forwards_for_every_input(self):
        self.put('a.json',True);self.put('b.json',False)
        self.run_batch(str(self.inputs),str(self.out),geometry_policy='endpoint_chord')
        self.assertEqual([r['success'] for r in self.rows()],['True','True'])
        self.assertTrue(all(c[1]['geometry_policy']=='endpoint_chord' for c in self.calls if c[0]=='solve'))

    def test_invalid_policy_fails_before_engine_or_output(self):
        self.put('a.json',False)
        with self.assertRaises(H['InvalidGeometryPolicyError']):
            self.run_batch(str(self.inputs),str(self.out),geometry_policy=None)
        self.assertEqual(self.calls,[]);self.assertFalse(self.out.exists())

    def test_policy_helper_and_old_binding_mismatches_fail_closed(self):
        self.put('a.json',False);self.run_batch(str(self.inputs),str(self.out))
        before=(self.out/'simulation_results.csv').read_bytes();self.calls.clear()
        with self.assertRaisesRegex(ValueError,'parameters mismatch'):
            self.run_batch(str(self.inputs),str(self.out),geometry_policy='endpoint_chord')
        recipe=self.ns['_batch_implementation_binding'];changed=recipe()
        changed['sources']['fibernet/sim/_taichi_geometry.py']='0'*64
        self.ns['_batch_implementation_binding']=lambda:changed
        with self.assertRaisesRegex(ValueError,'implementation mismatch'):self.run_batch(str(self.inputs),str(self.out))
        self.assertEqual(self.calls,[]);self.assertEqual((self.out/'simulation_results.csv').read_bytes(),before)
        self.ns['_batch_implementation_binding']=recipe
        # Old rows remain unmigrated; test only an isolated copied CSV.
        row=self.rows()[0];b=json.loads(row['run_binding']);del b['parameters']['geometry_policy']
        b['parameters_sha256']=self.ns['_batch_value_sha256'](b['parameters']);row['run_binding']=json.dumps(b)
        with (self.out/'simulation_results.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
        old=(self.out/'simulation_results.csv').read_bytes()
        with self.assertRaisesRegex(ValueError,'parameters mismatch'):self.run_batch(str(self.inputs),str(self.out))
        self.assertEqual((self.out/'simulation_results.csv').read_bytes(),old)

    def test_captured_byte_change_guard_stays_outside_failure_rows(self):
        self.put('a.json',False);owner=self
        class MutatingConstructor:
            def __init__(self):owner.put('a.json',True)
        self.ns['TaichiEngine']=MutatingConstructor
        with self.assertRaisesRegex(ValueError,'changed after batch preflight'):
            self.run_batch(str(self.inputs),str(self.out))
        self.assertFalse((self.out/'simulation_results.csv').exists());self.assertEqual(self.refs,[])
