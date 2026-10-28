"""Production generate/plan/binding/checkpoint flow, inert factory/provider/features.

Production checkpoint JSON is saved/read. Dataset NPY save and numeric feature
extraction are deliberately inert; no claim of real dataset/provider execution.
"""
from __future__ import annotations
import ast
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import platform
from numbers import Real,Integral
import stat
import tempfile
import time
import types
import unittest
from .support import ROOT,H,NP,graph,load_functions

class DatasetPolicy(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.out=self.base/'dataset';self.calls=[];owner=self
        self.contract_ns={**H,'np':NP,'hashlib':hashlib,'importlib':importlib,'json':json,'math':math,
                          'platform':platform,'Path':Path,'Real':Real,'Integral':Integral,
                          '__file__':str(ROOT/'fibernet/ml/_generate_contract.py')}
        tree=ast.parse((ROOT/'fibernet/ml/_generate_contract.py').read_text(encoding='utf-8'))
        for n in tree.body:
            if isinstance(n,ast.Assign):self.contract_ns[n.targets[0].id]=ast.literal_eval(n.value)
        load_functions('fibernet/ml/_generate_contract.py',[n.name for n in tree.body if isinstance(n,ast.FunctionDef)],self.contract_ns)
        self.contract=types.SimpleNamespace(**{k:v for k,v in self.contract_ns.items() if not k.startswith('__')})
        def builtin(): pass
        def factory(**kw): owner.calls.append(('graph',kw));return graph(kw['n_internal']>0)
        self.pattern=types.SimpleNamespace(_UNIT_FACTORIES={'square':builtin},_unit_square=builtin,pattern_2d=factory)
        class Engine:
            def __init__(self):owner.calls.append(('engine',))
            def stretch_test(self,g,**kw):
                owner.calls.append(('solve',kw));return types.SimpleNamespace(max_force=1.,max_stretch=1.,mean_stretch=1.,std_stretch=0.,time_seconds=.1)
        class Sample(types.SimpleNamespace):pass
        class Dataset:
            def __init__(self,samples,feature_names=None,target_names=None):
                self.samples=samples;self.n_samples=len(samples);self.n_features=len(feature_names or []);self.target_names=target_names or []
            def save(self,out):
                owner.calls.append(('dataset_save',));Path(out).mkdir(exist_ok=True);(Path(out)/'metadata.json').write_text('{"inert_dataset_save":true}')
        self.Dataset=Dataset
        self.ns={**H,'pattern':self.pattern,'contract':self.contract,'TaichiEngine':Engine,'Path':Path,
            'DatasetSample':Sample,'_extract_graph_features':lambda g:{'n_nodes':g.num_nodes},'_json_safe':lambda v:v,
            'json':json,'time':time,'stat':stat,'os':os,'tempfile':tempfile,'logging':logging}
        load_functions('fibernet/ml/dataset.py',['_dataset_file_mode','_save_checkpoint'],self.ns)
        load_functions('fibernet/ml/dataset.py',['generate'],self.ns,class_name='FiberNetDataset')
        self.generate=lambda **kw:self.ns['generate'](Dataset,**dict(units=['square'],grid_range=[(1,1)],radius_range=[.1],num_steps=1,verbose=False,**kw))
    def checkpoint(self):return json.loads((self.out/'checkpoint.json').read_text())

    def test_strict_mixed_builtin_plan_rejected_before_graph_engine_writes(self):
        with self.assertRaisesRegex(H['GeometryPolicyError'],'built-in plan'):
            self.generate(n_internal_range=[0,2],save_dir=str(self.out))
        self.assertEqual(self.calls,[]);self.assertFalse(self.out.exists())
        # The default remains [0,2,4]; no convenience fallback to [0].
        with self.assertRaises(H['GeometryPolicyError']):self.generate()
        self.assertEqual(self.calls,[])

    def test_chord_full_plan_binding_metadata_and_complete_resume(self):
        ds=self.generate(n_internal_range=[0,2],geometry_policy='endpoint_chord',save_dir=str(self.out))
        self.assertEqual(ds.n_samples,2)
        solves=[c[1] for c in self.calls if c[0]=='solve']
        self.assertEqual([s['geometry_policy'] for s in solves],['endpoint_chord']*2)
        self.assertEqual([s.metadata['taichi_geometry']['ignored_internal_point_edge_ids'] for s in ds.samples],[[],[9]])
        ck=self.checkpoint();self.assertEqual(ck['run_binding']['solver']['geometry_policy'],'endpoint_chord')
        h='sim/_taichi_geometry.py'
        self.assertEqual(ck['run_binding']['implementation']['source_sha256'][h],hashlib.sha256((ROOT/'fibernet'/h).read_bytes()).hexdigest())
        before=(self.out/'checkpoint.json').read_bytes();self.calls.clear()
        again=self.generate(n_internal_range=[0,2],geometry_policy='endpoint_chord',save_dir=str(self.out))
        self.assertEqual(again.n_samples,2);self.assertEqual(self.calls,[])
        self.assertEqual((self.out/'checkpoint.json').read_bytes(),before)

    def test_straight_strict_plan_accepted_and_explicit_policy_bound(self):
        ds=self.generate(n_internal_range=[0],save_dir=str(self.out))
        self.assertEqual(ds.n_samples,1)
        self.assertEqual(ds.samples[0].metadata['taichi_geometry']['policy'],'reject_internal_points')

    def test_strict_plan_uses_selected_jobs_after_existing_shuffle_and_cap(self):
        # The deterministic shuffle stand-in establishes ordering only. It does
        # not claim NumPy RNG equivalence or change production sampling.
        shuffle_calls = []
        reverse = [False]

        class RandomState:
            def __init__(self, seed):
                shuffle_calls.append(('seed', seed))

            def shuffle(self, plan):
                shuffle_calls.append(('shuffle', [p['n_internal'] for p in plan]))
                if reverse[0]:
                    plan.reverse()

        self.contract_ns['np'] = types.SimpleNamespace(
            **vars(NP), random=types.SimpleNamespace(RandomState=RandomState))
        ds = self.generate(n_internal_range=[0, 2], max_samples=1)
        self.assertEqual([s.params['n_internal'] for s in ds.samples], [0])
        self.assertEqual(shuffle_calls[1], ('shuffle', [0, 2]))
        self.calls.clear()
        reverse[0] = True
        with self.assertRaisesRegex(H['GeometryPolicyError'], 'built-in plan'):
            self.generate(n_internal_range=[0, 2], max_samples=1)
        self.assertEqual(self.calls, [])

    def test_empty_plan_stays_zero_work(self):
        ds = self.generate(n_internal_range=[])
        self.assertEqual(ds.n_samples, 0)
        self.assertEqual(self.calls, [])

    def test_custom_registry_identity_uses_actual_graph_before_engine(self):
        # A registered replacement with a familiar name is NOT trusted as builtin.
        self.pattern._UNIT_FACTORIES['square']=lambda:None
        self.pattern.pattern_2d=lambda **kw:graph(True)
        with self.assertRaises(H['UnsupportedInternalPointsError']):self.generate(n_internal_range=[0])
        self.assertEqual(self.calls,[])
        self.pattern.pattern_2d=lambda **kw:graph(False)
        ds=self.generate(n_internal_range=[2])
        self.assertEqual(ds.n_samples,1)
        # n_internal is a factory request, not proof of actual stored geometry.
        self.assertEqual(ds.samples[0].metadata['taichi_geometry']['ignored_internal_point_edge_ids'],[])

    def test_custom_chord_optin_and_geometry_failure_not_silent_partial(self):
        self.pattern._UNIT_FACTORIES['square']=lambda:None
        ds=self.generate(n_internal_range=[0,2],geometry_policy='endpoint_chord')
        self.assertEqual(ds.n_samples,2)
        self.calls.clear()
        with self.assertRaises(H['UnsupportedInternalPointsError']):self.generate(n_internal_range=[0,2])
        self.assertEqual(len([c for c in self.calls if c[0]=='solve']),1)
        self.assertFalse(self.out.exists())

    def test_invalid_policy_precedes_graph_provider_and_writes(self):
        with self.assertRaises(H['InvalidGeometryPolicyError']):self.generate(n_internal_range=[0],geometry_policy=False,save_dir=str(self.out))
        self.assertEqual(self.calls,[]);self.assertFalse(self.out.exists())

    def test_policy_helper_old_binding_and_unbound_checkpoint_fail_closed(self):
        self.generate(n_internal_range=[0],geometry_policy='endpoint_chord',save_dir=str(self.out))
        before=(self.out/'checkpoint.json').read_bytes();self.calls.clear()
        with self.assertRaisesRegex(ValueError,'binding mismatch'):self.generate(n_internal_range=[0],save_dir=str(self.out))
        recipe=self.contract.implementation_recipe
        def changed(*args):
            r=recipe(*args);r['source_sha256']['sim/_taichi_geometry.py']='0'*64;return r
        self.contract.implementation_recipe=changed
        with self.assertRaisesRegex(ValueError,'binding mismatch'):
            self.generate(n_internal_range=[0],geometry_policy='endpoint_chord',save_dir=str(self.out))
        self.contract.implementation_recipe=recipe
        self.assertEqual(self.calls,[]);self.assertEqual((self.out/'checkpoint.json').read_bytes(),before)
        ck=self.checkpoint();del ck['run_binding']['solver']['geometry_policy'];ck['run_binding_sha256']=self.contract.digest(ck['run_binding'])
        (self.out/'checkpoint.json').write_text(json.dumps(ck))
        with self.assertRaisesRegex(ValueError,'binding mismatch'):
            self.generate(n_internal_range=[0],geometry_policy='endpoint_chord',save_dir=str(self.out))
        (self.out/'checkpoint.json').write_text('{"samples":[]}')
        with self.assertRaisesRegex(ValueError,'legacy or unknown'):
            self.generate(n_internal_range=[0],geometry_policy='endpoint_chord',save_dir=str(self.out))
        self.assertEqual(self.calls,[])
