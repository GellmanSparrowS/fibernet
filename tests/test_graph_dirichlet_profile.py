"""Pure synthetic fixtures for the explicit six-column condition profile."""
import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from fibernet.ml import graph_training_contract as C
from test_graph_training_contract import source, spec, bind, legacy

DECLARATION=b'Synthetic fixture: no prescribed nodal current injections; Dirichlet electrodes only.\n'


def encoded_spec():
    result=spec()
    result.update(schema='fibernet.graph_feature_spec.v2',condition_profile='dirichlet_voltage_mask.v1')
    return result


def encoded_source(gid, edges=2):
    item=source(gid,edges)
    item['conditions'].update(schema='fibernet.electrical_conditions.v2',conditioning='dirichlet_voltage_mask.v1',
        current_declaration={'kind':'no_nodal_current_injections.v1','source_ref':'synthetic:fixture-declaration',
                             'source_sha256':hashlib.sha256(DECLARATION).hexdigest()})
    return item


def encoded_bind(sources, **kwargs):
    return bind(sources,feature_spec=encoded_spec(),**kwargs)


def sample(inputs, row=0):
    return inputs.as_dict()['samples'][row]


class TestDirichletProfile(unittest.TestCase):
    def test_01_six_columns_zero_fixed_and_free(self):
        sources={'x':encoded_source('x')};inputs=encoded_bind(sources)
        arrays=C.project_graph_inputs(inputs,sources=sources)['x']
        self.assertEqual(inputs.as_dict()['schema'],'fibernet.graph_inputs.v3')
        self.assertEqual(arrays['node_features'].dtype,np.dtype('float32'))
        np.testing.assert_array_equal(arrays['node_features'],[[0.,0.,0.,1.,0.,1.],[.5,0.,0.,1.,1.,1.],[0.,.5,0.,1.,0.,0.]])
        self.assertNotEqual(arrays['node_features'][0,-1],arrays['node_features'][2,-1])
        np.testing.assert_array_equal(arrays['edge_index'],sources['x']['batch'].message_index)
        self.assertEqual(arrays['edge_features'].shape,(4,4))
        batch=sources['x']['batch']
        self.assertEqual(batch.edge_index.shape,(2,2))
        self.assertEqual(arrays['edge_index'].shape,(2,4))
        np.testing.assert_array_equal(batch.node_ids,[2,7,11])
        np.testing.assert_array_equal(batch.edge_ids[batch.message_parent_rows],batch.message_parent_ids)
        np.testing.assert_array_equal(arrays['edge_features'][::2],arrays['edge_features'][1::2])
        np.testing.assert_array_equal(np.bincount(arrays['edge_index'][1],minlength=3),[2,2,0])
        for key in ('node_ids','edge_ids','message_parent_ids','message_orientation'):
            self.assertIsNotNone(getattr(sources['x']['batch'],key))

    def test_02_variable_cardinality_explicit_only(self):
        sources={g:encoded_source(g) for g in ('a','b','c')}
        sources['a']['conditions']['boundary_voltages'].pop()
        sources['c']['conditions']['boundary_voltages'].append({'node_id':11,'value':2.,'unit':'V'})
        bound=encoded_bind(sources)
        views=C.project_graph_inputs(bound,sources=sources)
        self.assertEqual([int(views[g]['node_features'][:,-1].sum()) for g in sources],[1,2,3])
        for v in sources.values():
            v['conditions'].update(schema='fibernet.electrical_conditions.v1',conditioning='metadata_only_fixed_rule')
            del v['conditions']['current_declaration']
        with self.assertRaisesRegex(ValueError,'mixed/unencoded numerical condition rules unsupported by this profile'):bind(sources)

    def test_03_electrode_values_positions_and_hashes(self):
        sources={'x':encoded_source('x')};before=encoded_bind(sources);base=sample(before)
        for mutation in ('swap','move','voltage'):
            changed=copy.deepcopy(sources);rows=changed['x']['conditions']['boundary_voltages']
            if mutation=='swap':rows[0]['value'],rows[1]['value']=rows[1]['value'],rows[0]['value']
            elif mutation=='move':rows[0]['node_id']=11
            else:rows[1]['value']=-2.5
            after=encoded_bind(changed);record=sample(after)
            for key in ('source_hash','projection_hash','input_hash'):
                self.assertNotEqual(base[key],record[key],(mutation,key))
            self.assertEqual(base['topology_hash'],record['topology_hash'])
            with self.assertRaisesRegex(ValueError,'bound raw authority'):C.project_graph_inputs(before,sources=changed)

    def test_04_zero_and_raw_precision_policy(self):
        sources={'x':encoded_source('x')};before=encoded_bind(sources)
        sources['x']['conditions']['boundary_voltages'][0]['value']=-0.0
        after=encoded_bind(sources);view=C.project_graph_inputs(after,sources=sources)['x']['node_features']
        self.assertFalse(np.signbit(view[0,4]));self.assertFalse(np.signbit(view[2,4]))
        self.assertNotEqual(sample(before)['source_hash'],sample(after)['source_hash'])
        self.assertNotEqual(sample(before)['input_hash'],sample(after)['input_hash'])
        self.assertEqual(sample(before)['projection_hash'],sample(after)['projection_hash'])
        sources['x']['conditions']['boundary_voltages'][1]['value']=1.+2**-30
        fine=encoded_bind(sources)
        self.assertEqual(sample(after)['projection_hash'],sample(fine)['projection_hash'])
        self.assertNotEqual(sample(after)['source_hash'],sample(fine)['source_hash'])

    def test_05_missing_and_unsupported_current_authority(self):
        for mutation in ('missing','empty','unknown','injection','zero_injection','bad_hash','bad_reference','extra','none'):
            item=encoded_source('x');cond=item['conditions']
            if mutation=='missing':del cond['current_declaration']
            elif mutation=='empty':cond['current_declaration']={}
            elif mutation=='unknown':cond['current_declaration']['kind']='unknown'
            elif mutation in ('injection','zero_injection'):cond['nodal_currents']=[{'node_id':11,'value':1. if mutation=='injection' else 0.,'unit':'A'}]
            elif mutation=='bad_hash':cond['current_declaration']['source_sha256']='not-a-hash'
            elif mutation=='bad_reference':cond['current_declaration']['source_ref']=''
            elif mutation=='extra':cond['current_declaration']['assumed']=True
            else:cond['nodal_currents']=None
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):encoded_bind({'x':item})

    def test_06_profile_version_selection_no_migration(self):
        item=encoded_source('x')
        with self.assertRaises(ValueError):bind({'x':item})
        with self.assertRaises(ValueError):encoded_bind({'x':source('x')})
        for bad in ('v3','unknown',None,{},False):
            fs=encoded_spec();fs['condition_profile']=bad
            with self.subTest(profile=bad),self.assertRaises(ValueError):bind({'x':item},feature_spec=fs)
        for key,value in [('schema','fibernet.electrical_conditions.v3'),('conditioning','metadata_only_fixed_rule')]:
            changed=copy.deepcopy(item);changed['conditions'][key]=value
            with self.assertRaises(ValueError):encoded_bind({'x':changed})
        fs=encoded_spec();fs['node_scale']+=[1.,1.]
        with self.assertRaises(ValueError):bind({'x':item},feature_spec=fs)
        for schema in ('fibernet.graph_inputs.v1','fibernet.graph_inputs.v2','fibernet.graph_inputs.v4'):
            forged=encoded_bind({'x':item}).as_dict();forged['schema']=schema
            with self.assertRaises(ValueError):C._inputs(C._record('inputs',forged))
        mixed={'x':item,'y':source('y')}
        with self.assertRaises(ValueError):encoded_bind(mixed)

    def test_07_node_ids_units_types_and_bounds(self):
        for mutation in ('duplicate','unknown','bool_id','float_id','unit','bool','nonfinite','overflow','underflow','integer_precision','empty','extra'):
            item=encoded_source('x');rows=item['conditions']['boundary_voltages']
            if mutation=='duplicate':rows.append(dict(rows[0]))
            elif mutation=='unknown':rows[0]['node_id']=99
            elif mutation=='bool_id':rows[0]['node_id']=True
            elif mutation=='float_id':rows[0]['node_id']=2.
            elif mutation=='unit':rows[0]['unit']='mV'
            elif mutation=='bool':rows[0]['value']=False
            elif mutation=='nonfinite':rows[0]['value']=float('nan')
            elif mutation=='overflow':rows[0]['value']=1e100
            elif mutation=='underflow':rows[0]['value']=1e-100
            elif mutation=='integer_precision':rows[0]['value']=2**53+1
            elif mutation=='empty':rows.clear()
            else:rows[0]['solved_voltage']=0.
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):encoded_bind({'x':item})

    def test_08_derived_tampering_and_ownership(self):
        sources={'x':encoded_source('x')};inputs=encoded_bind(sources)
        views=C.project_graph_inputs(inputs,sources=sources)
        self.assertEqual(inputs,encoded_bind(sources,derived=views))
        original=views['x']['node_features'].copy()
        for col in (0,4,5):
            forged=copy.deepcopy(views);forged['x']['node_features'][0,col]+=1.
            with self.assertRaisesRegex(ValueError,'authoritative recomputation'):encoded_bind(sources,derived=forged)
        views['x']['node_features'][:]=123.;views['x']['edge_index'][:]=99
        np.testing.assert_array_equal(C.project_graph_inputs(inputs,sources=sources)['x']['node_features'],original)
        self.assertEqual(inputs,encoded_bind(sources))
        for array in C.project_graph_inputs(inputs,sources=sources)['x'].values():
            self.assertTrue(array.flags.owndata)
        with self.assertRaises(ValueError):C.project_graph_inputs(inputs,sources={})
        sources['x']['source_refs']['conditions']='different-label'
        with self.assertRaisesRegex(ValueError,'bound raw authority'):C.project_graph_inputs(inputs,sources=sources)

    def test_09_condition_provenance_not_predictor(self):
        sources={'x':encoded_source('x')};before=encoded_bind(sources)
        for field,value in [('source_sha256','a'*64),('source_ref','synthetic:another-authority')]:
            changed=copy.deepcopy(sources);changed['x']['conditions']['current_declaration'][field]=value
            after=encoded_bind(changed)
            self.assertNotEqual(sample(before)['source_hash'],sample(after)['source_hash'])
            self.assertNotEqual(sample(before)['input_hash'],sample(after)['input_hash'])
            self.assertEqual(sample(before)['projection_hash'],sample(after)['projection_hash'])
            with self.assertRaises(ValueError):C.project_graph_inputs(before,sources=changed)
        changed=copy.deepcopy(sources);changed['x']['conditions']['boundary_voltages'].reverse()
        after=encoded_bind(changed)
        self.assertEqual(sample(before)['projection_hash'],sample(after)['projection_hash'])
        self.assertNotEqual(sample(before)['source_hash'],sample(after)['source_hash'])

    def test_10_train_only_stats_and_source_tamper(self):
        sources={g:encoded_source(g) for g in ('train','validation','final')}
        sources['validation']['conditions']['boundary_voltages'].pop()
        sources['final']['conditions']['boundary_voltages'][1]['value']=2.
        inputs=encoded_bind(sources)
        split=C.bind_explicit_group_split(inputs,train_ids=['train'],validation_ids=['validation'],final_ids=['final'],split_seed=42)
        stats=C.training_degree_histogram(inputs,split,training_sources={'train':sources['train']})
        self.assertEqual(stats.as_dict()['summary']['histogram'],[1,0,2])
        with self.assertRaisesRegex(ValueError,'exact training sources only'):C.training_degree_histogram(inputs,split,training_sources=sources)
        sources['validation']['conditions']['boundary_voltages'][0]['value']=99.
        sources['final']['external']['rows'][30]['conductivity_S_per_m']['value']=-1.
        self.assertEqual(stats,C.training_degree_histogram(inputs,split,training_sources={'train':sources['train']}))
        sources['train']['conditions']['current_declaration']['source_sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'mutated or coalesced'):C.training_degree_histogram(inputs,split,training_sources={'train':sources['train']})

    def test_11_stable_id_alignment_topology_not_predictor(self):
        sources={'x':encoded_source('x')};before=encoded_bind(sources)
        original=C.project_graph_inputs(before,sources=sources)['x']
        b=sources['x']['batch'];sources['x']['batch']=replace(b,node_ids=np.array([102,107,111],dtype='i8'))
        for row in sources['x']['conditions']['boundary_voltages']:row['node_id']+=100
        after=encoded_bind(sources);views=C.project_graph_inputs(after,sources=sources)['x']
        for key in original:np.testing.assert_array_equal(original[key],views[key])
        self.assertNotEqual(sample(before)['topology_hash'],sample(after)['topology_hash'])
        self.assertNotEqual(sample(before)['source_hash'],sample(after)['source_hash'])
        self.assertEqual(sample(before)['projection_hash'],sample(after)['projection_hash'])
        np.testing.assert_array_equal(b.message_parent_ids,[30,30,31,31])
        np.testing.assert_array_equal(b.message_orientation,[1,-1,1,-1])

    def test_12_forged_rule_and_schema_reject(self):
        inputs=encoded_bind({'x':encoded_source('x')})
        for key,value in [('voltage_scale',2.),('voltage_scale',1),('zero_policy','unknown'),('current_policy','inferred'),('conditioning','metadata_only_fixed_rule')]:
            forged=inputs.as_dict();forged['samples'][0]['condition_record'][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):C._inputs(C._record('inputs',forged))
        forged=inputs.as_dict();forged['samples'][0]['condition_record']['extra']=0
        with self.assertRaises(ValueError):C._inputs(C._record('inputs',forged))
        forged=inputs.as_dict();forged['samples'][0]['condition_rule']='other'
        with self.assertRaises(ValueError):C._inputs(C._record('inputs',forged))

    def test_13_legacy_projection_and_lineage_unchanged(self):
        old_sources={g:source(g) for g in ('a','b','c')};old=bind(old_sources)
        self.assertEqual(old.as_dict()['schema'],'fibernet.graph_inputs.v2')
        self.assertEqual(C.project_graph_inputs(old,sources=old_sources)['a']['node_features'].shape,(3,4))
        sources={g:encoded_source(g) for g in ('a','b','c')};inputs=encoded_bind(sources)
        raw=legacy(inputs)
        record=C.make_lineage_sidecar(raw,result_bytes=b'{"fixture":true}',reconstructed_inputs=inputs,raw_sources=sources)
        self.assertEqual(record.as_dict()['weight_compatibility'],'not_implemented')
        self.assertEqual(record.as_dict()['status'],'new_contract_lineage_only')

    def test_14_no_optional_backends_loaded(self):
        # Collection and earlier tests may legitimately load optional backends.
        # Exercise this profile in a fresh interpreter with the existing guard.
        root = Path(__file__).resolve().parents[1]
        script = """
import json
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(root), str(root / 'tests')]
from test_graph_training_import_boundary import RejectOptionalImports, is_optional
assert not any(name == 'fibernet' or name.startswith('fibernet.') or is_optional(name)
               for name in sys.modules)
guard = RejectOptionalImports()
sys.meta_path.insert(0, guard)
try:
    from test_graph_dirichlet_profile import C, encoded_bind, encoded_source
    assert Path(C.__file__).resolve() == root / 'fibernet/ml/graph_training_contract.py'
    sources = {'x': encoded_source('x')}
    inputs = encoded_bind(sources)
    arrays = C.project_graph_inputs(inputs, sources=sources)['x']
    assert inputs.as_dict()['schema'] == 'fibernet.graph_inputs.v3'
    assert arrays['node_features'].shape == (3, 6)
    loaded = sorted(name for name in sys.modules if is_optional(name))
    assert not guard.attempts, guard.attempts
    assert not loaded, loaded
    print(json.dumps({'optional_import_attempts': guard.attempts,
                      'loaded_optional_modules': loaded, 'node_columns': 6}))
finally:
    sys.meta_path.remove(guard)
"""
        env = os.environ.copy()
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            env[name] = '1'
        result = subprocess.run(
            [sys.executable, '-I', '-B', '-c', script, str(root)],
            check=True, capture_output=True, text=True, timeout=60, env=env,
        )
        self.assertEqual(json.loads(result.stdout), {
            'optional_import_attempts': [], 'loaded_optional_modules': [],
            'node_columns': 6,
        })


if __name__=='__main__':unittest.main()
