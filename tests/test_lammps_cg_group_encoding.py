"""Opt-in encoding/source tests, never native execution or equivalence proof."""
import copy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fibernet import simulate
from fibernet.core.structure_graph import StructureGraph
from fibernet.integrations import CGRequest,CGAngle,MorsePair,prepare_cg
from fibernet.integrations import lammps_backend as b
from fibernet.integrations._lammps_cg import GROUP_ENCODING
from fibernet.sim import _lammps_result as adapter
from fibernet.sim.accelerated import SimResult
from test_lammps_cg import fixture,raw,RUNTIME


def compile_case(selected=(0,),edges=False):
    g,q=fixture(edges)
    q=replace(q,projectile_pairs={n:MorsePair(1.,2.,1.,3.) for n in selected},pair_encoding=GROUP_ENCODING)
    return g,q,prepare_cg(g,q)


def groups_and_pairs(p):
    groups={};selected=set();exclusions=[]
    for line in p.input_text.splitlines():
        w=line.split()
        if w[:1]==['group'] and len(w)>3 and w[2]=='id':groups[w[1]]=set(map(int,w[3:]))
        if w[:2]==['pair_coeff','*']:continue
        if w[:1]==['pair_coeff']:selected.add(tuple(map(int,w[1:3])))
        if w[:3]==['neigh_modify','exclude','group']:exclusions.append((w[3],w[4]))
    def excluded(i,j):return any((i in groups[a] and j in groups[z]) or (j in groups[a] and i in groups[z]) for a,z in exclusions)
    return groups,selected,excluded


class GroupEncodingTests(unittest.TestCase):
    def test_default_bytes_match_frozen_baseline(self):
        baseline=json.loads((Path(__file__).parent/'fixtures/lammps_cg_default_encoding_v1.json').read_text())
        for name,edges,steps,pairs in [('base',True,0,None),('no_pairs',True,2,{}),('all_pairs',False,0,{n:MorsePair(1.,2.,1.,3.) for n in range(3)})]:
            g,q=fixture(edges,steps)
            if pairs is not None:q=replace(q,projectile_pairs=pairs)
            p=prepare_cg(g,q)
            actual={'identity':p.identity,**{k:hashlib.sha256(getattr(p,k).encode()).hexdigest() for k in ('graph_json','contract_json','data_text','input_text')}}
            self.assertEqual(actual,baseline['cases'][name])
            self.assertNotIn('pair_encoding',json.loads(p.contract_json))

    def test_all_partial_and_empty_selection_truth_table(self):
        for chosen in ((),(0,),(0,2),(0,1,2)):
            g,q,p=compile_case(chosen);groups,selected,excluded=groups_and_pairs(p)
            expected={tuple(sorted((n+1,4))) for n in chosen}
            self.assertEqual(selected,expected)
            for i in range(1,5):
                for j in range(i+1,5):self.assertEqual(not excluded(i,j),(i,j) in expected)
            self.assertEqual(groups['cg_strand'],{1,2,3})
            self.assertEqual(groups['cg_disabled'],{4}|{n+1 for n in range(3) if n not in chosen})
            self.assertEqual(p,b.validate_prepared(p))
            self.assertEqual(p,prepare_cg(g,q))

    def test_no_nonselected_group_can_be_empty(self):
        _,_,p=compile_case((0,1,2))
        self.assertIn('group cg_disabled id 4\n',p.input_text)
        self.assertNotIn('group cg_disabled empty',p.input_text)

    def test_safe_sentinel_and_immutable_order(self):
        _,_,p=compile_case();lines=p.input_text.splitlines()
        self.assertEqual(lines.count('pair_coeff * * 0 1 0 0'),1)
        self.assertLess(lines.index('pair_coeff * * 0 1 0 0'),next(i for i,x in enumerate(lines) if x.startswith('pair_coeff 1 4 ')))
        self.assertLess(lines.index('neigh_modify exclude group cg_strand cg_strand'),lines.index('run 0'))
        self.assertIn('special_bonds lj/coul 1 1 1',lines)
        self.assertNotIn('hybrid',p.input_text)
        self.assertNotIn('neigh_modify exclude type',p.input_text)
        self.assertNotIn('pair_modify',p.input_text)
        self.assertEqual(sum(x.startswith('pair_style ') for x in lines),1)

    def test_physical_data_masses_angles_and_ids_unchanged(self):
        g,q=fixture();q=replace(q,masses_kg={**q.masses_kg,1:math.nextafter(1.,2.)})
        old=prepare_cg(g,q);new=prepare_cg(g,replace(q,pair_encoding=GROUP_ENCODING))
        self.assertEqual(old.graph_json,new.graph_json);self.assertEqual(old.data_text,new.data_text)
        a,z=json.loads(old.contract_json),json.loads(new.contract_json)
        for key in set(a)-{'pair_policy','required_styles'}:self.assertEqual(a[key],z[key])
        self.assertEqual(z['type_to_mass_kg'][0][1],1.)
        self.assertEqual(z['type_to_mass_kg'][1][1],math.nextafter(1.,2.))
        self.assertEqual(new.manifest()['angle_to_stable_id'],{'1':9})

    def test_sparse_ids_and_projectile_not_last(self):
        g,q=fixture(False);d=g.to_dict();mapping={0:10,1:9999,2:800,3:40}
        for n in d['nodes']:n['id']=mapping[n['id']]
        d['next_node_id']=10000;g=StructureGraph.from_dict(d)
        q=replace(q,masses_kg={mapping[n]:v for n,v in q.masses_kg.items()},projectile_node_id=40,projectile_pairs={10:MorsePair(1,2,1,3)},pair_encoding=GROUP_ENCODING)
        p=prepare_cg(g,q);groups,selected,excluded=groups_and_pairs(p)
        self.assertEqual(p.node_ids,(10,40,800,9999));self.assertEqual(selected,{(1,2)})
        self.assertEqual(groups['cg_strand'],{1,3,4});self.assertEqual(groups['cg_disabled'],{2,3,4})
        self.assertFalse(excluded(1,2));self.assertTrue(excluded(2,3));b.validate_prepared(p)

    def test_coincident_excluded_sites_supported_selected_refused(self):
        g,q=fixture(False);d=g.to_dict();d['nodes'][1]['position']=d['nodes'][0]['position'];d['nodes'][2]['position']=d['nodes'][3]['position'];g=StructureGraph.from_dict(d)
        q=replace(q,pair_encoding=GROUP_ENCODING);p=prepare_cg(g,q);_,_,excluded=groups_and_pairs(p)
        self.assertTrue(excluded(1,2));self.assertTrue(excluded(3,4))
        r=adapter.to_sim_result(p,raw(p),RUNTIME,0,True);self.assertEqual(r.n_nodes,4)
        with self.assertRaises(b.LAMMPSContractError):prepare_cg(g,replace(q,projectile_pairs={2:MorsePair(1,2,1,3)}))

    def test_refuses_unknown_policy_and_custom_diagnostics(self):
        g,q=fixture()
        for bad in (False,True,0,[],{},'unknown','static-groups-zero-cutoff/2'):
            with self.assertRaises(b.LAMMPSContractError):prepare_cg(g,replace(q,pair_encoding=bad))
        with self.assertRaises(TypeError):CGRequest(**vars(q),diagnostics=['single'])
        _,_,p=compile_case()
        for tail in ('compute illegal all pair/local eng\n','pair_write 1 2 2 r 0 1 out table\n','neigh_modify exclude none\n'):
            with self.assertRaises(b.LAMMPSContractError):b.validate_prepared(replace(p,input_text=p.input_text+tail))

    def test_forged_contract_and_commands_refused(self):
        _,_,p=compile_case()
        for kind in ('name','claim','extra','sentinel','group','styles','remove'):
            c=json.loads(p.contract_json)
            if kind=='name':c['pair_encoding']['name']='unknown'
            if kind=='claim':c['pair_encoding']['native_equivalence']='VALIDATED'
            if kind=='extra':c['pair_encoding']['extra']=1
            if kind=='sentinel':c['pair_encoding']['disabled_coefficients'][0]=False
            if kind=='group':c['pair_encoding']['exclusions']=[]
            if kind=='styles':c['required_styles']['pair']=['zero']
            if kind=='remove':del c['pair_encoding']
            with self.assertRaises(b.LAMMPSContractError):b.validate_prepared(replace(p,contract_json=b._json(c)))
        with self.assertRaises(b.LAMMPSContractError):b.validate_prepared(replace(p,input_text=p.input_text.replace('group cg_disabled id 2 3 4','group cg_disabled id 1 2 3 4')))

    def test_saved_serialization_and_forged_rehashed_metadata(self):
        _,_,p=compile_case();r=adapter.to_sim_result(p,raw(p),RUNTIME,0,True)
        self.assertEqual(r.metadata['model_contract']['pair_encoding']['native_equivalence'],'NOT_YET_VALIDATED')
        with tempfile.TemporaryDirectory() as d:
            for detail in (False,True):
                out=Path(d)/'r.json';r.save(out,detailed=detail)
                self.assertEqual(SimResult.load(out).to_dict(detail),r.to_dict(detail))
        for kind in ('name','claim','sentinel','unknown'):
            rr=copy.deepcopy(r);c=rr.metadata['model_contract']
            if kind=='name':c['pair_encoding']['name']='unknown'
            if kind=='claim':c['pair_encoding']['native_equivalence']='VALIDATED'
            if kind=='sentinel':c['pair_encoding']['disabled_coefficients'][0]=True
            if kind=='unknown':c['other_encoding']='pretend'
            rr.metadata['model_contract_sha256']=b._digest(b._json(c))
            with self.assertRaises(ValueError):rr.to_dict(True)

    def test_unified_public_api_mock(self):
        g,q,_=compile_case()
        class Task:
            def __init__(self,p,*args):self.result=raw(p);self.error=None
            def start(self):pass
            def wait(self):return b.TaskState.SUCCEEDED
            def cancel(self):pass
        with patch.object(b,'LAMMPSTask',Task):
            r=simulate(g,backend='lammps',mode='dynamics',cg_request=q,runtime=RUNTIME,workdir='unused',limits=b.ResourceLimits(),record_state=True)
        self.assertEqual(r.metadata['model_contract']['pair_encoding']['name'],GROUP_ENCODING)

    def test_group_only_native_squared_domain_refusals(self):
        g,q=fixture(False)
        for coeff in (MorsePair(1.,1e200,1e-200,3e-200),MorsePair(1.,1e-200,1e200,3e200)):
            request=replace(q,projectile_pairs={0:coeff})
            prepare_cg(g,request)  # Existing default acceptance is unchanged.
            with self.assertRaises(b.LAMMPSContractError):prepare_cg(g,replace(request,pair_encoding=GROUP_ENCODING))
        d=g.to_dict();d['nodes'][3]['position']=[0.,0.,0.];d['nodes'][0]['position']=[1e-200,0.,0.]
        for position in ([1e-200,0.,0.], [1e-162,1e-162,1e-162]):
            d['nodes'][0]['position']=position
            tiny=StructureGraph.from_dict(d)
            prepare_cg(tiny,q)
            with self.assertRaises(b.LAMMPSContractError):prepare_cg(tiny,replace(q,pair_encoding=GROUP_ENCODING))

    def test_caps_remain_unchanged(self):
        g=StructureGraph(dimension=3)
        for i in range(65):g.add_node([i,0,0])
        q=CGRequest('m',2,{i:1. for i in range(65)},{},{},64,{},b.RunSpec(.001,0,1,((-100,100),)*3),pair_encoding=GROUP_ENCODING)
        with self.assertRaises(b.LAMMPSContractError):prepare_cg(g,q)

if __name__=='__main__':unittest.main()
