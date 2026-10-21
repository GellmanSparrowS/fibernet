"""Portable graph-input contract tests using the normal FiberNet package."""
import copy
from dataclasses import replace, FrozenInstanceError
import hashlib
import json
import struct
from types import MappingProxyType
import unittest

import numpy as np

from fibernet.ml import graph_training_contract as C
from fibernet.ml import graph_data as GD


def batch(gid, edges=2):
    a=lambda v,d: np.array(v,dtype=d)
    physical=np.array([[0]*edges,[1]*edges],dtype=np.int64).reshape(2,edges)
    messages=np.empty((2,2*edges),dtype=np.int64)
    messages[:,::2]=physical;messages[:,1::2]=physical[::-1]
    return GD.GraphBatch('fibernet.graph_batch.v1',(gid,),a([2],'i8'),'m','specimen_xy',
        a([0,3],'i8'),a([0,edges],'i8'),a([2,7,11],'i8'),a(list(range(30,30+edges)),'i8'),
        a([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],'f8'),physical,
        np.empty((0,3),dtype='f8'),np.zeros(edges+1,dtype='i8'),np.zeros(edges,dtype=bool),
        messages,np.repeat(np.arange(edges,dtype='i8'),2),np.repeat(a(list(range(30,30+edges)),'i8'),2),
        np.tile(a([1,-1],'i1'),edges),('radius','chord_length','polyline_length'),
        np.tile(a([[.1,1.,1.]],'f8'),(edges,1)),np.ones((edges,3),dtype=bool),
        ({},),({},{},{}),tuple({} for _ in range(edges)))


def spec():
    return {'schema':'fibernet.graph_feature_spec.v1','length_unit':'m','coordinate_frame':'specimen_xy',
        'dtype':'float32','node_offset':[0.,0.,0.,0.],'node_scale':[2.,2.,2.,2.],
        'features':[{'name':'radius','source':{'snapshot':'radius'},'unit':'m','offset':0.,'scale':.1,'required':True,'parity':'even'},
                    {'name':'conductivity','source':{'external':'conductivity_S_per_m'},'unit':'S/m','offset':0.,'scale':10000.,'required':True,'parity':'even'}]}


def source(gid, edges=2):
    return {'batch':batch(gid,edges),
        'external':{'units':{'conductivity_S_per_m':'S/m'},'rows':{i:{'conductivity_S_per_m':{'value':10000.,'available':True}} for i in range(30,30+edges)}},
        'conditions':{'schema':'fibernet.electrical_conditions.v1','task_rule_id':'left_right_fixed_v1',
            'conditioning':'metadata_only_fixed_rule','boundary_voltages':[{'node_id':2,'value':0.,'unit':'V'},{'node_id':7,'value':1.,'unit':'V'}],'nodal_currents':[]},
        'source_refs':{'graph':'raw/'+gid,'external':'raw/'+gid+'/external','conditions':'raw/'+gid+'/bc'}}


def bind(sources, feature_spec=None, groups=None, derived=None):
    return C.bind_graph_inputs([v['batch'] for v in sources.values()],source_groups=groups or {k:'mother-'+k for k in sources},
        feature_spec=spec() if feature_spec is None else feature_spec,
        external={k:v['external'] for k,v in sources.items()},conditions={k:v['conditions'] for k,v in sources.items()},
        source_refs={k:v['source_refs'] for k,v in sources.items()},derived_views=derived)


def setup(edges=2):
    sources={g:source(g,edges) for g in ('train','validation','final')}
    inputs=bind(sources)
    split=C.bind_explicit_group_split(inputs,train_ids=['train'],validation_ids=['validation'],final_ids=['final'],split_seed=42)
    return sources,inputs,split


def legacy(inputs):
    # Synthetic v1 metadata has a two-way development split, not a final role.
    rows=inputs.as_dict()['samples'][:2];ids=[r['graph_id'] for r in rows]
    return json.dumps({'schema':'fibernet.gine_artifact.v1','kind':'inference_best_validation',
        'model_config':{'version':'fibernet.edge_gine.v1','hidden':16},'target':{'name':'conductance','unit':'mS'},
        'training_manifest':{'graph_ids':ids,'source_groups':{r['graph_id']:r['source_group'] for r in rows},
            'input_digests':{g:'a'*64 for g in ids},'split':{'graph_ids':ids,'train_indices':[0],'validation_indices':[1],
                'config_sha256':'b'*64,'seed':42}},'state':{'fixture':{'dtype':'float32','shape':[1],'values':[1.]}},
        'runtime':{'torch':'synthetic-fixture','sources':{'edge_gine':'c'*64}}},indent=2).encode()


class TestGraphTrainingContract(unittest.TestCase):
    def test_01_parallel_edges_and_isolate(self):
        sources,inputs,split=setup()
        stats=C.training_degree_histogram(inputs,split,training_sources={'train':sources['train']}).as_dict()
        self.assertEqual(stats['summary']['histogram'],[1,0,2]);self.assertEqual(stats['summary']['node_population'],3)
        self.assertEqual(stats['summary']['incoming_messages'],4);self.assertEqual(stats['physical_edges'],2)
        self.assertEqual(np.bincount(sources['train']['batch'].message_index[1],minlength=3).tolist(),[2,2,0])
        self.assertEqual(C.validate_pna_degree_admission(C._record('degree',stats),scalers=['identity','amplification','attenuation']).as_dict()['status'],'denominator_checks_only_not_backend_validation')

    def test_02_isolated_vs_pna_and_empty_population(self):
        sources,inputs,split=setup(0)
        stats=C.training_degree_histogram(inputs,split,training_sources={'train':sources['train']})
        self.assertEqual(stats.as_dict()['summary']['histogram'],[3])
        C.validate_pna_degree_admission(stats,scalers=['identity'])
        C.validate_pna_degree_admission(stats,scalers=['attenuation','inverse_linear'])
        for scaler in ('linear','amplification'):
            with self.subTest(scaler=scaler),self.assertRaisesRegex(ValueError,'zero denominator'):
                C.validate_pna_degree_admission(stats,scalers=[scaler])
        empty=C._histogram_summary([0]);self.assertEqual(empty['status'],'empty_population')
        self.assertIsNone(empty['mean_degree']);self.assertIsNone(empty['mean_log_degree'])
        bad=stats.as_dict();bad['summary']=empty
        with self.assertRaises(ValueError): C.validate_pna_degree_admission(C._record('degree',bad),scalers=['identity'])
        with self.assertRaises(ValueError): C.bind_explicit_group_split(inputs,train_ids=[],validation_ids=['validation'],final_ids=['train','final'],split_seed=42)

    def test_03_direct_graph_adversaries(self):
        valid=source('x')
        cases={'node_ptr':np.array([0,2],dtype='i8'),'edge_ptr':np.array([0,1],dtype='i8'),
               'internal_ptr':np.array([0,1,0],dtype='i8'),'edge_index':np.array([[0,0],[3,1]],dtype='i8'),
               'node_ids':np.array([2,7],dtype='i8'),'dimensions':np.array([4],dtype='i8')}
        for name,value in cases.items():
            item=copy.deepcopy(valid);item['batch']=replace(item['batch'],**{name:value})
            with self.subTest(field=name),self.assertRaises(ValueError): bind({'x':item})
        loop=copy.deepcopy(valid);loop['batch']=replace(loop['batch'],edge_index=np.array([[0,0],[0,1]],dtype='i8'))
        with self.assertRaises(ValueError): bind({'x':loop})
        with self.assertRaises(ValueError): C._snapshot(object())
        with self.assertRaises(ValueError): C._snapshot(replace(valid['batch'],graph_ids=('x','y')))

    def test_04_message_maps_and_bound_coalescence(self):
        for field in ('message_parent_rows','message_parent_ids','message_orientation','message_index'):
            item=source('x');arr=getattr(item['batch'],field).copy();arr.flat[0]=99
            item['batch']=replace(item['batch'],**{field:arr})
            with self.subTest(field=field),self.assertRaises(ValueError): bind({'x':item})
        item=source('x');item['batch']=replace(item['batch'],message_index=item['batch'].message_index[:,:-1])
        with self.assertRaises(ValueError): bind({'x':item})
        sources,inputs,split=setup();different=source('train',1)
        # A consistent different graph is valid on its own, not proven illicit.
        bind({'train':different})
        with self.assertRaisesRegex(ValueError,'mutated or coalesced'):
            C.training_degree_histogram(inputs,split,training_sources={'train':different})

    def test_05_authority_projection_and_material_provenance(self):
        item=source('x');snap=C._snapshot(item['batch']);raw,lookup=C._external(item['external'],snap,spec())
        view=C._project(snap,spec(),lookup,None)
        bind({'x':item},derived={'x':view})
        for field in view:
            bad={k:v.copy() for k,v in view.items()};bad[field].flat[0]+=1
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'derived view'):
                bind({'x':item},derived={'x':bad})
        reverse={k:v.copy() for k,v in view.items()};reverse['edge_features'][1,0]+=1
        with self.assertRaises(ValueError): bind({'x':item},derived={'x':reverse})
        for change in ('geometry','mask','placeholder','missing_external'):
            bad=source('x')
            if change=='geometry': bad['batch'].edge_features[0,1]=2.
            elif change in ('mask','placeholder'):
                bad['batch'].edge_feature_available[0,0]=False
                if change=='mask':bad['batch'].edge_features[0,0]=0.
            else:bad['external']['rows'][30]['conductivity_S_per_m']={'value':None,'available':False}
            with self.subTest(change=change),self.assertRaises(ValueError): bind({'x':bad})
        with self.assertRaises(ValueError): C.bind_graph_inputs([view],source_groups={},feature_spec=spec(),external={},conditions={},source_refs={})
        p=GD.GraphBatchProvenance('fibernet.graph_batch.provenance.v1',np.zeros((3,6),bool),np.zeros(3,bool),('missing',)*3,
            (None,None),np.zeros(2,bool),('missing',)*2,np.zeros(2,dtype='i8'),np.zeros(2,bool),('missing',)*2)
        v2=source('x');v2['batch']=replace(v2['batch'],schema_version='fibernet.graph_batch.v2',provenance=p)
        bind({'x':v2})
        p.edge_material_available[0]=True
        with self.assertRaises(ValueError): bind({'x':v2})

    def test_06_identity_coverage_and_not_features(self):
        for ids in ([2,2,11],[-1,7,11],[11,7,2]):
            item=source('x');item['batch']=replace(item['batch'],node_ids=np.array(ids,dtype='i8'))
            with self.subTest(ids=ids),self.assertRaises(ValueError): bind({'x':item})
        item=source('x');item['batch']=replace(item['batch'],node_ids=np.array([True,False,True],dtype=bool))
        with self.assertRaises(ValueError): bind({'x':item})
        item=source('x');item['batch']=replace(item['batch'],edge_ids=np.array([2**63,2**63+1],dtype='u8'))
        with self.assertRaises(ValueError): bind({'x':item})
        item=source('x');item['external']['rows'][True]=item['external']['rows'].pop(30)
        with self.assertRaises(ValueError): bind({'x':item})
        with self.assertRaises(ValueError): bind({'x':source('x')},groups={'wrong':'mother'})
        original=source('x');relabeled=source('x');relabeled['batch'].node_ids[:]=[20,70,110]
        relabeled['conditions']['boundary_voltages'][0]['node_id']=20;relabeled['conditions']['boundary_voltages'][1]['node_id']=70
        a=bind({'x':original}).as_dict()['samples'][0];b=bind({'x':relabeled}).as_dict()['samples'][0]
        self.assertNotEqual(a['input_hash'],b['input_hash']);self.assertEqual(a['projection_hash'],b['projection_hash'])

    def test_07_ownership_and_mutation(self):
        sources,inputs,split=setup();before=inputs.payload_json
        detached=inputs.as_dict();detached['samples'][0]['source_refs']['graph']='changed'
        self.assertEqual(inputs.payload_json,before)
        with self.assertRaises(FrozenInstanceError): inputs.digest='wrong'
        sources['train']['batch'].positions[0,2]=1.
        with self.assertRaises(ValueError): C.training_degree_histogram(inputs,split,training_sources={'train':sources['train']})
        self.assertEqual(inputs.payload_json,before)
        sources,inputs,split=setup();sources['train']['external']['rows'][30]['conductivity_S_per_m']['value']=9000.
        with self.assertRaises(ValueError): C.training_degree_histogram(inputs,split,training_sources={'train':sources['train']})

    def test_08_numeric_serialization_and_golden_encoding(self):
        for bad in (np.ma.array([1.]),np.array([object()],dtype=object),np.array([1],dtype='u8')):
            with self.subTest(dtype=str(type(bad))),self.assertRaises(ValueError): C.canonical_bytes(bad)
        for raw in ('{"a":1,"a":2}','{"x":NaN}','{"x":Infinity}','['*33+'0'+']'*33):
            with self.subTest(raw=raw[:20]),self.assertRaises(ValueError): C.parse_json(raw)
        cycle=[];cycle.append(cycle)
        with self.assertRaises(ValueError): C.canonical_bytes(cycle)
        for v in (float('nan'),float('inf'),2**63,'x'*(C.MAX_TEXT+1)):
            with self.subTest(value_type=type(v).__name__),self.assertRaises(ValueError): C.canonical_bytes(v)
        with self.assertRaises(ValueError): C.parse_json(' '*(C.MAX_BYTES+1))
        with self.assertRaises(ValueError): C.canonical_bytes([0]*C.MAX_ITEMS)
        with self.assertRaises(ValueError): C._histogram_summary([True])
        with self.assertRaises(ValueError): C._histogram_summary([1,0])
        self.assertEqual(C.canonical_bytes(np.array([1.,-0.],dtype='>f8')),C.canonical_bytes(np.array([1.,-0.],dtype='<f8')))
        self.assertNotEqual(C.canonical_bytes(0.),C.canonical_bytes(-0.))
        self.assertNotEqual(C.canonical_bytes(1),C.canonical_bytes(1.))
        self.assertEqual(C.canonical_bytes(7),b'fibernet.graph_contract.v1\x00i\x00\x00\x00\x00\x00\x00\x00\x017')
        self.assertEqual(C.canonical_bytes(-0.),b'fibernet.graph_contract.v1\x00f\x00\x00\x00\x00\x00\x00\x00\x08'+b'\x80'+b'\x00'*7)
        self.assertEqual(C.canonical_bytes({'b':2,'a':1}),C.canonical_bytes({'a':1,'b':2}))
        for value in (1e300,1e-300,2**53+1):
            item=source('x');item['external']['rows'][30]['conductivity_S_per_m']['value']=value
            with self.subTest(value=value),self.assertRaises(ValueError): bind({'x':item})
        item=source('x');item['batch']=replace(item['batch'],positions=item['batch'].positions.astype('f4'))
        with self.assertRaises(ValueError): bind({'x':item})
        bad=spec();bad['unknown']=1
        with self.assertRaises(ValueError): bind({'x':source('x')},feature_spec=bad)

    def test_09_split_adversaries(self):
        sources,inputs,split=setup()
        for train,validation,final in [([],['validation'],['train','final']),(['train','train'],['validation'],['final']),(['missing'],['validation'],['final']),(['train'],['validation'],[]),(['train'],['train'],['final'])]:
            with self.subTest(roles=(train,validation,final)),self.assertRaises(ValueError):
                C.bind_explicit_group_split(inputs,train_ids=train,validation_ids=validation,final_ids=final,split_seed=42)
        shared=bind(sources,groups={'train':'mother','validation':'mother','final':'other'})
        with self.assertRaisesRegex(ValueError,'mother groups overlap'):
            C.bind_explicit_group_split(shared,train_ids=['train'],validation_ids=['validation'],final_ids=['final'],split_seed=42)
        with self.assertRaises(ValueError): C.bind_explicit_group_split(inputs,train_ids=['train'],validation_ids=['validation'],final_ids=['final'],split_seed=True)
        with self.assertRaises(ValueError): C.bind_explicit_group_split(inputs,train_ids=['train'],validation_ids=['validation'],final_ids=['final'],split_seed=42,algorithm='new_rng')
        with self.assertRaises(ValueError): C._split(inputs,replace(split,digest='0'*64))

    def test_10_training_only_statistics(self):
        sources,inputs,split=setup();first=C.training_degree_histogram(inputs,split,training_sources={'train':sources['train']})
        class NeverArray:
            def __array__(self,*args,**kwargs): raise AssertionError('held-out array accessed')
        for role in ('validation','final'):
            sources[role]['batch']=replace(sources[role]['batch'],message_index=NeverArray(),positions=NeverArray())
        again=C.training_degree_histogram(inputs,split,training_sources={'train':sources['train']})
        self.assertEqual(first,again)
        other={g:source(g,20 if g!='train' else 2) for g in ('train','validation','final')}
        changed=bind(other);changed_split=C.bind_explicit_group_split(changed,train_ids=['train'],validation_ids=['validation'],final_ids=['final'],split_seed=42)
        new=C.training_degree_histogram(changed,changed_split,training_sources={'train':other['train']})
        self.assertEqual(first.as_dict()['summary'],new.as_dict()['summary']);self.assertNotEqual(first.digest,new.digest)
        self.assertNotEqual(split.digest,changed_split.digest)
        with self.assertRaises(ValueError): C.training_degree_histogram(inputs,split,training_sources=sources)
        forged=first.as_dict();forged['summary']['node_population']=999
        with self.assertRaises(ValueError): C.validate_pna_degree_admission(C._record('degree',forged),scalers=['identity'])
        forged=first.as_dict();forged['physical_edges']=3
        with self.assertRaises(ValueError): C.validate_pna_degree_admission(C._record('degree',forged),scalers=['identity'])

    def test_11_hash_independence_and_raw_precision(self):
        sources,inputs,split=setup();baseline=[r['input_hash'] for r in inputs.as_dict()['samples']]
        for item in sources.values():
            item['batch'].graph_metadata[0].update(target=123.,model={'name':'PNA'},runtime={'torch':'changed'})
            item['batch'].node_metadata[0].update(label=99.)
            item['batch'].edge_metadata[0].update(solved_voltage=8.)
            item['source_refs']['graph']='mixed-file-reference-with-different-target-metadata'
        changed=bind(sources);s2=C.bind_explicit_group_split(changed,train_ids=['train'],validation_ids=['validation'],final_ids=['final'],split_seed=42)
        self.assertEqual(baseline,[r['input_hash'] for r in changed.as_dict()['samples']]);self.assertEqual(split,s2)
        self.assertNotEqual(inputs.digest,changed.digest)  # Annotation envelope differs, data identity does not.
        self.assertNotEqual(C._hash('target/v1',{'unit':'mS'}),C._hash('target/v1',{'unit':'S'}))
        a=source('x');b=source('x');b['external']['rows'][30]['conductivity_S_per_m']['value']=10000.000001
        one=bind({'x':a}).as_dict()['samples'][0];two=bind({'x':b}).as_dict()['samples'][0]
        self.assertEqual(one['projection_hash'],two['projection_hash']);self.assertNotEqual(one['source_hash'],two['source_hash']);self.assertNotEqual(one['input_hash'],two['input_hash'])
        changed_spec=spec();changed_spec['node_scale'][0]=4.
        self.assertNotEqual(one['input_hash'],bind({'x':a},feature_spec=changed_spec).as_dict()['samples'][0]['input_hash'])
        # Domain separation prevents reusing a payload digest in a different role.
        self.assertNotEqual(C._hash('source/v1',{}),C._hash('projection/v1',{}))

    def test_12_conditions_and_units(self):
        item=source('x');base=bind({'x':item}).as_dict()['samples'][0]['source_hash']
        item['conditions']['boundary_voltages'][1]['value']=2.
        self.assertNotEqual(base,bind({'x':item}).as_dict()['samples'][0]['source_hash'])
        for bad in ('unit','node','conditioning','schema','missing'):
            item=source('x')
            if bad=='unit':item['conditions']['boundary_voltages'][0]['unit']='mV'
            elif bad=='node':item['conditions']['boundary_voltages'][0]['node_id']=999
            elif bad=='conditioning':item['conditions']['conditioning']='variable_unencoded'
            elif bad=='schema':item['conditions']['schema']='mechanical_constraints_inferred_from_faces'
            else:del item['conditions']['nodal_currents']
            with self.subTest(case=bad),self.assertRaises(ValueError):bind({'x':item})
        item=source('x');item['external']['units']['conductivity_S_per_m']='S'
        with self.assertRaises(ValueError):bind({'x':item})
        item=source('x');item['batch']=replace(item['batch'],length_unit='mm')
        with self.assertRaises(ValueError):bind({'x':item})
        item=source('x');item['conditions']['task_rule_id']='different'
        with self.assertRaises(ValueError):bind({'x':item,'y':source('y')})
        self.assertEqual(spec()['features'][1]['unit'],'S/m')
        self.assertEqual(json.loads(legacy(bind({'x':source('x')})))['target']['unit'],'mS')

    def test_13_legacy_metadata_only_lineage(self):
        sources,inputs,split=setup();raw=legacy(inputs);results=b'{"synthetic_fixture": true}'
        before=(hashlib.sha256(raw).hexdigest(),hashlib.sha256(results).hexdigest())
        blocked=C.make_lineage_sidecar(raw,result_bytes=results,reconstructed_inputs=inputs).as_dict()
        self.assertEqual(blocked['status'],'blocked_missing_raw_authority');self.assertIsNone(blocked['new_contract'])
        self.assertEqual(blocked['legacy_identity']['split'],json.loads(raw)['training_manifest']['split'])
        self.assertEqual(blocked['legacy_identity']['model_config'],json.loads(raw)['model_config'])
        accepted=C.make_lineage_sidecar(raw,result_bytes=results,reconstructed_inputs=inputs,raw_sources=sources).as_dict()
        self.assertEqual(accepted['status'],'new_contract_lineage_only');self.assertEqual(accepted['weight_compatibility'],'not_implemented')
        self.assertEqual(before,(hashlib.sha256(raw).hexdigest(),hashlib.sha256(results).hexdigest()))
        self.assertEqual(accepted['legacy_identity']['artifact_bytes_sha256'],before[0])
        invalid=json.loads(raw);invalid['schema']='unknown'
        with self.assertRaises(ValueError):C.make_lineage_sidecar(json.dumps(invalid).encode(),result_bytes=results)
        sources['train']['external']['rows'][30]['conductivity_S_per_m']['value']=8000.
        with self.assertRaises(ValueError):C.make_lineage_sidecar(raw,result_bytes=results,reconstructed_inputs=inputs,raw_sources=sources)

    def test_14_synthetic_group_split_coverage(self):
        ids = [f"sample-{i:02d}" for i in range(48)]
        sources = {gid: source(gid) for gid in ids}
        groups = {gid: f"mother-{i // 4:02d}" for i, gid in enumerate(ids)}
        inputs = bind(sources, groups=groups)
        split = C.bind_explicit_group_split(
            inputs, train_ids=ids[:24], validation_ids=ids[24:32],
            final_ids=ids[32:], split_seed=42,
        )
        roles = split.as_dict()['roles']
        self.assertEqual([len(roles[k]) for k in ('train', 'validation', 'final')], [24, 8, 16])
        grouped = {role: {groups[gid] for gid in members} for role, members in roles.items()}
        self.assertEqual([len(grouped[k]) for k in ('train', 'validation', 'final')], [6, 2, 4])
        self.assertFalse(grouped['train'] & grouped['validation'])
        self.assertFalse(grouped['train'] & grouped['final'])
        self.assertFalse(grouped['validation'] & grouped['final'])
        stats = C.training_degree_histogram(
            inputs, split, training_sources={gid: sources[gid] for gid in ids[:24]},
        ).as_dict()
        self.assertEqual(stats['summary']['histogram'], [24, 0, 48])
        self.assertEqual(stats['summary']['incoming_messages'], 96)
        self.assertEqual(stats['summary']['node_population'], 72)

    def test_15_frozen_record_transport_and_integrity(self):
        sources, inputs, split = setup()
        degree = C.training_degree_histogram(inputs, split, training_sources={'train': sources['train']})
        envelope = {'kind': degree.kind, 'payload_json': degree.payload_json, 'digest': degree.digest}
        restored = C.FrozenRecord(**json.loads(json.dumps(envelope)))
        self.assertEqual(restored.as_dict(), degree.as_dict())
        self.assertEqual(C.validate_pna_degree_admission(restored, scalers=['identity']),
                         C.validate_pna_degree_admission(degree, scalers=['identity']))
        for changes in ({'digest': '0' * 64}, {'payload_json': '{}'}, {'kind': 'inputs'}):
            changed = dict(envelope, **changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                C.FrozenRecord(**changed).as_dict()

    def test_16_complete_structure_graph_example(self):
        from examples.graph_training_contract import build_records, make_source
        self.assertIs(type(make_source('specimen', 1.0e-3)['batch']), GD.GraphBatch)
        inputs, split, degree, admission = build_records()
        self.assertEqual(len(inputs.as_dict()['samples']), 3)
        self.assertEqual(split.as_dict()['roles'], {
            'train': ['train-a'], 'validation': ['validation-b'], 'final': ['final-c'],
        })
        self.assertEqual(degree.as_dict()['summary']['histogram'], [1, 0, 2])
        self.assertEqual(degree.as_dict()['summary']['node_population'], 3)
        self.assertEqual(degree.as_dict()['summary']['incoming_messages'], 4)
        self.assertEqual(admission.as_dict()['statistics_hash'], degree.digest)


    def test_17_histogram_summary_exact_numeric_types(self):
        sources,inputs,split=setup(0)
        baseline=C.training_degree_histogram(inputs,split,training_sources={'train':sources['train']}).as_dict()
        C.validate_pna_degree_admission(C._record('degree',baseline),scalers=['identity'])
        cases=[('incoming_messages',False),('incoming_messages',0.0),('incoming_messages','0'),
               ('node_population',True),('node_population',3.0),('node_population',-1),
               ('mean_degree',False),('mean_degree',0),('mean_degree',-0.0),
               ('mean_log_degree',False),('mean_log_degree',0),('status',True),
               ('histogram',[True]),('histogram',[3.0]),('histogram',[3,0])]
        for field,value in cases:
            bad=copy.deepcopy(baseline);bad['summary'][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                C.validate_pna_degree_admission(C._record('degree',bad),scalers=['identity'])
        for field,value in [('physical_edges',False),('physical_edges',0.0)]:
            bad=copy.deepcopy(baseline);bad[field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                C.validate_pna_degree_admission(C._record('degree',bad),scalers=['identity'])
        for field,value in [('node_count',3.0),('physical_edge_count',False)]:
            bad=copy.deepcopy(baseline);bad['training'][0][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                C.validate_pna_degree_admission(C._record('degree',bad),scalers=['identity'])

    def test_18_fixed_numerical_condition_roles(self):
        sources,inputs,split=setup()
        for mutation in ('value','swapped_roles','fewer_slots','more_slots','current','signed_zero','int_float','bool'):
            bad=copy.deepcopy(sources);condition=bad['validation']['conditions']
            if mutation=='value':condition['boundary_voltages'][1]['value']=2.
            elif mutation=='swapped_roles':condition['boundary_voltages'].reverse()
            elif mutation=='fewer_slots':condition['boundary_voltages'].pop()
            elif mutation=='more_slots':condition['boundary_voltages'].append({'node_id':11,'value':0.,'unit':'V'})
            elif mutation=='current':condition['nodal_currents']=[{'node_id':11,'value':1.,'unit':'A'}]
            elif mutation=='signed_zero':condition['boundary_voltages'][0]['value']=-0.0
            elif mutation=='int_float':condition['boundary_voltages'][1]['value']=1
            else:condition['boundary_voltages'][0]['value']=False
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):bind(bad)
        # Graph-local node identities may differ, preserving explicit role-slot order.
        changed=copy.deepcopy(sources)
        changed['validation']['batch']=replace(changed['validation']['batch'],node_ids=np.array([102,107,111],dtype='i8'))
        for row in changed['validation']['conditions']['boundary_voltages']:row['node_id']+=100
        rebound=bind(changed).as_dict()
        self.assertEqual(rebound['schema'],'fibernet.graph_inputs.v2')
        self.assertEqual(rebound['samples'][0]['condition_record'],rebound['samples'][1]['condition_record'])
        self.assertNotEqual(inputs.as_dict()['samples'][1]['source_hash'],rebound['samples'][1]['source_hash'])
        # Same explicit nonzero current roles are allowed, but not differing values.
        for gid,item in changed.items():
            item['conditions']['nodal_currents']=[{'node_id':111 if gid=='validation' else 11,'value':.25,'unit':'A'}]
        bind(changed)
        changed['validation']['conditions']['nodal_currents'][0]['value']=.5
        with self.assertRaises(ValueError):bind(changed)
        # Reusing forged metadata must also fail, not only fresh raw binding.
        forged=inputs.as_dict();forged['samples'][1]['condition_record']['boundary_voltages'][1]['value']=2.
        with self.assertRaises(ValueError):C.bind_explicit_group_split(C._record('inputs',forged),train_ids=['train'],validation_ids=['validation'],final_ids=['final'],split_seed=42)
        forged=inputs.as_dict()
        for row in forged['samples']:row['condition_record']['boundary_voltages'][1]['value']=2.
        forged_inputs=C._record('inputs',forged)
        forged_split=C.bind_explicit_group_split(forged_inputs,train_ids=['train'],validation_ids=['validation'],final_ids=['final'],split_seed=42)
        with self.assertRaises(ValueError):C.training_degree_histogram(forged_inputs,forged_split,training_sources={'train':sources['train']})
        old=inputs.as_dict();old['schema']='fibernet.graph_inputs.v1'
        with self.assertRaises(ValueError):C._inputs(C._record('inputs',old))

    def test_19_legacy_overlap_and_index_types(self):
        sources,inputs,split=setup();baseline=json.loads(legacy(inputs));results=b'{}'
        self.assertEqual(baseline['training_manifest']['graph_ids'],['train','validation'])
        for train,validation in [([0],[0]),([0,0],[1]),([0],[1,1]),([-1],[1]),([0],[2]),
                                 ([False],[1]),([0],[True]),([0.0],[1]),(['0'],[1]),([], [0,1]),([0,1],[])]:
            bad=copy.deepcopy(baseline);bad['training_manifest']['split'].update(train_indices=train,validation_indices=validation)
            raw=json.dumps(bad).encode()
            for authority in (False,True):
                kwargs={'reconstructed_inputs':inputs,'raw_sources':sources} if authority else {}
                with self.subTest(train=train,validation=validation,authority=authority),self.assertRaises(ValueError):
                    C.make_lineage_sidecar(raw,result_bytes=results,**kwargs)
        accepted=C.make_lineage_sidecar(json.dumps(baseline).encode(),result_bytes=results,reconstructed_inputs=inputs,raw_sources=sources).as_dict()
        self.assertEqual(accepted['legacy_identity']['split'],baseline['training_manifest']['split'])
        self.assertEqual(accepted['legacy_identity']['graph_ids'],['train','validation'])
        self.assertEqual(len(accepted['new_contract']['samples']),3)

    def test_20_legacy_exact_original_development_coverage(self):
        sources,inputs,split=setup();baseline=json.loads(legacy(inputs));results=b'{"unchanged":true}'
        # An uncovered development ID must not be silently accepted.
        manifest=baseline['training_manifest'];manifest['graph_ids'].append('final')
        manifest['source_groups']['final']='mother-final';manifest['input_digests']['final']='a'*64
        manifest['split']['graph_ids'].append('final')
        raw=json.dumps(baseline).encode()
        for authority in (False,True):
            kwargs={'reconstructed_inputs':inputs,'raw_sources':sources} if authority else {}
            with self.subTest(authority=authority),self.assertRaises(ValueError):C.make_lineage_sidecar(raw,result_bytes=results,**kwargs)
        # Complete original development coverage is accepted and preserved verbatim.
        manifest['split']['train_indices']=[0,2]
        good_raw=json.dumps(baseline,indent=2).encode();before=hashlib.sha256(good_raw).hexdigest()
        good=C.make_lineage_sidecar(good_raw,result_bytes=results,reconstructed_inputs=inputs,raw_sources=sources).as_dict()
        self.assertEqual(good['legacy_identity']['split'],manifest['split'])
        self.assertEqual(good['legacy_identity']['artifact_bytes_sha256'],before)
        self.assertEqual(hashlib.sha256(good_raw).hexdigest(),before)
        for mutation in ('duplicate_id','reordered_split_ids','missing_group','extra_digest','group_overlap','split_groups','split_digests'):
            bad=copy.deepcopy(baseline);m=bad['training_manifest']
            if mutation=='duplicate_id':m['graph_ids'][2]='train'
            elif mutation=='reordered_split_ids':m['split']['graph_ids'].reverse()
            elif mutation=='missing_group':del m['source_groups']['final']
            elif mutation=='extra_digest':m['input_digests']['unknown']='d'*64
            elif mutation=='group_overlap':m['source_groups']['validation']=m['source_groups']['train']
            elif mutation=='split_groups':m['split']['groups']={}
            else:m['split']['input_digests']={}
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                C.make_lineage_sidecar(json.dumps(bad).encode(),result_bytes=results)
        # Synthetic larger development metadata exercises exact coverage without weights.
        large = json.loads(legacy(inputs))
        ids = [f"development-{i:02d}" for i in range(32)]
        large['training_manifest'] = {
            'graph_ids': ids,
            'source_groups': {gid: f"mother-{i // 4:02d}" for i, gid in enumerate(ids)},
            'input_digests': {gid: 'a' * 64 for gid in ids},
            'split': {'graph_ids': ids, 'train_indices': list(range(24)),
                      'validation_indices': list(range(24, 32)),
                      'config_sha256': 'b' * 64, 'seed': 42},
        }
        large_raw = json.dumps(large).encode()
        large_results = b'{"synthetic_fixture":true}'
        sidecar = C.make_lineage_sidecar(large_raw, result_bytes=large_results).as_dict()
        self.assertEqual(sidecar['status'], 'blocked_missing_raw_authority')
        self.assertEqual(len(sidecar['legacy_identity']['graph_ids']), 32)
        self.assertEqual(sidecar['legacy_identity']['artifact_bytes_sha256'], hashlib.sha256(large_raw).hexdigest())
        self.assertEqual(sidecar['legacy_identity']['results_bytes_sha256'], hashlib.sha256(large_results).hexdigest())


if __name__ == '__main__':
    unittest.main()
