"""Synthetic-only result/consumer boundary tests. No stored research states."""
from dataclasses import FrozenInstanceError
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import unittest
from unittest.mock import patch
# Python 3.9-3.11 may query the Windows version through a stdlib subprocess.
# Cache that platform metadata before auditing dependency and projection work.
# No NumPy, SciPy, FiberNet or provider import precedes the audit boundary.
platform.uname()
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
PROHIBITED=[]
def audit(event,args):
    if (event in ('subprocess.Popen','os.system','os.fork','os.forkpty','os.posix_spawn','os.exec',
                  'socket.__new__','socket.connect','socket.bind','socket.getaddrinfo')
            or (event=='import' and (args[0]=='run_case' or args[0].startswith('lightweight_cv05')))):
        PROHIBITED.append(event);raise RuntimeError('No provider, process or network in projection tests')
sys.addaudithook(audit)
class BlockTaichi(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='taichi' or fullname.startswith('taichi.'):
            raise ModuleNotFoundError("No module named 'taichi' (synthetic-test guard)",name='taichi')
sys.meta_path.insert(0,BlockTaichi())
import numpy as np
# Older NumPy testing imports probe optional CPU features with lscpu. Keep the
# audit active and use NumPy's missing-command fallback during that import only.
with patch('subprocess.run',side_effect=FileNotFoundError('CPU discovery disabled for isolated synthetic tests')):
    import numpy.testing
from fibernet.core.structure_graph import StructureGraph
from fibernet.core.material import Material
from fibernet.sim.l3 import bind_l3
from fibernet.sim.accelerated import SimResult,TaichiEngine
from fibernet.sim import l3_result as p
# Isolated plotting uses Matplotlib's bundled/system-path fallback fonts.
# Fontconfig subprocess discovery is unavailable, without relaxing the audit.
MPL_CACHE=tempfile.TemporaryDirectory(prefix='l3-projection-mpl-')
os.environ['MPLCONFIGDIR']=MPL_CACHE.name
with patch('subprocess.check_output',side_effect=FileNotFoundError('Fontconfig disabled for isolated synthetic tests')):
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
MODEL={'joint':'welded','section':'circular','constitutive':'isotropic_elastic_brittle','loading':'clamped_positive_y','contact':False}


def synthetic_fixture(unit='mm',radius_unit='um'):
    scale={'m':1.,'mm':.001,'um':.000001}[unit];radius_scale={'m':1.,'mm':.001,'um':.000001}[radius_unit]
    graph=StructureGraph(dimension=2)
    material=Material(name='synthetic material',youngs_modulus=1000.,poissons_ratio=.25)
    for xy in ((0,0),(.01,0),(.01,.01)):graph.add_node([xy[0]/scale,xy[1]/scale,0])
    graph.add_edge(0,1,radius=.0001/radius_scale,material=material,n_internal=0,segments=1)
    graph.add_edge(1,2,radius=.0001/radius_scale,material=material,n_internal=0,segments=1)
    graph.add_edge(0,1,radius=.0001/radius_scale,material=material,n_internal=0,segments=1,allow_parallel=True)
    d=graph.to_dict();ids={0:7,1:42,2:99};edgeids={0:22,1:9,2:33}
    for n in d['nodes']:n['id']=ids[n['id']]
    d['nodes'].reverse()
    for e in d['edges']:e['id']=edgeids[e['id']];e['node_i']=ids[e['node_i']];e['node_j']=ids[e['node_j']]
    d.pop('next_node_id');d.pop('next_edge_id');graph=StructureGraph.from_dict(d)
    graph.metadata['private_marker']='SYNTHETIC_METADATA_DO_NOT_EXPORT'
    binding=bind_l3(graph,coordinate_unit=unit,radius_unit=radius_unit,
        edge_data={eid:dict(failure_strain=.01,physical_span_id=eid,source_edge_id=eid,initially_active=True,reference_length_m=.01) for eid in graph.edges},
        fixed_node_ids=[7],loaded_node_ids=[99],model=MODEL)
    b=json.loads(binding.canonical);nodes=[42,7,99];edges=[33,9,22]
    reference={n['id']:n['position'] for n in b['record']['nodes']};topology={e['id']:[e['node_i'],e['node_j']] for e in b['record']['segments']}
    displacement=[[.001,.0002],[0.,0.],[.001,.001]]
    values={'displacements_m':displacement,'positions_m':[[reference[nid][j]+u[j] for j in range(2)] for nid,u in zip(nodes,displacement)],
        'rotations_rad':[.2,0.,.7],'active':[True,False,True],
        'axial_strain':[.03,.2,-.04],'axial_force_N':[-3.,99.,2.],
        'end_moments_Nm':[[.1,-.1],[.2,-.2],[.3,-.3]],'max_tensile_strain':[.04,.2,.05],'element_energy_J':[1.,.5,2.]}
    view={'schema':'fibernet.l3-frame-view/1','frame':4,'node_ids':nodes,'edge_ids':edges,
        'reference_positions_m':[reference[i] for i in nodes],'edge_node_ids':[topology[i] for i in edges],
        'fields':values,'field_units':dict(p.UNITS),'field_rows':{k:'node_ids' if k in p.NODE_FIELDS else 'edge_ids' for k in p.UNITS},
        'field_components':{'displacements_m':['x','y'],'positions_m':['x','y'],'rotations_rad':['theta'],'end_moments_Nm':['node_i','node_j']},
        'active_edge_ids':[33,22],'failed_edge_ids':[9],'initial_inactive_ids':[],'newly_failed_ids':[9],
        'observation':{'step':2,'phase':'post_damage','u_m':.001,'reaction_N':2.,'internal_energy_J':3.5,'external_work_J':4.,'equilibrium_jump_energy_J':.5,'energy_balance_residual_J':0.},
        'original_units':{k:b[k] for k in ('coordinate_unit','radius_unit','coordinate_scale_to_m','radius_scale_to_m')},
        'new_execution':False,'provider_license_verified':False,'historical_native_state':'succeeded','historical_outer_status':'FAILED_WORKER',
        'provenance':{'binding_sha256':binding.sha256,'private_marker':'SYNTHETIC_PATH_DO_NOT_EXPORT'},
        'unknown_description':'SYNTHETIC_PROVIDER_DESCRIPTION_DO_NOT_EXPORT'}
    return graph,binding,view


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.graph,self.binding,self.view=synthetic_fixture()
        self.before=p._raw(self.graph.to_dict());self.temp=tempfile.TemporaryDirectory(prefix='projection-test-');self.addCleanup(self.temp.cleanup)
        for name in ('dynamics','compute_forces','stretch_test'):
            patcher=patch.object(TaichiEngine,name,side_effect=AssertionError('No engine permitted'));patcher.start();self.addCleanup(patcher.stop)
        patcher=patch.object(SimResult,'compute_detailed',side_effect=AssertionError('No spring postprocessing permitted'));patcher.start();self.addCleanup(patcher.stop)
    def tearDown(self):
        self.assertEqual(PROHIBITED,[]);self.assertEqual(p._raw(self.graph.to_dict()),self.before)
        self.assertNotIn('run_case',sys.modules);self.assertNotIn('taichi',sys.modules)
    def project(self,view=None,**kwargs):
        raw=p._raw(self.view if view is None else view)
        return p.project_l3_result(self.graph,self.binding,frame_view_bytes=raw,expected_view_sha256=kwargs.get('expected',p._sha(raw)))
    def test_public_error_family_and_exports(self):
        from fibernet.sim.l3 import L3InputError
        self.assertTrue(issubclass(p.L3ProjectionError,L3InputError))
        self.assertEqual(set(p.__all__),{'L3ProjectionError','L3ResultProjection','L3RenderInputs','project_l3_result','load_l3_projection','l3_render_inputs'})
    def test_legacy_view_profile_is_explicit_same_field_contract(self):
        public=self.project().to_dict()['frame_data']
        self.view['schema']='private.l3.saved-frame-view/1'
        self.assertEqual(self.project().to_dict()['frame_data'],public)
    def test_unknown_frame_schema_rejected(self):
        self.view['schema']='unknown-frame-format'
        with self.assertRaises(p.L3ProjectionError):self.project()
    def test_bad_bound_graph_uses_public_projection_error(self):
        g=self.graph.copy();g.nodes[42].position[2]=1.
        raw=p._raw(self.view)
        with self.assertRaises(p.L3ProjectionError):p.project_l3_result(g,self.binding,frame_view_bytes=raw,expected_view_sha256=p._sha(raw))
    def test_imports_exclude_private_execution_modules(self):
        self.assertFalse(any(name.endswith(('l3_backend','_l3_worker','_execution_owner','_execution_boundary')) for name in sys.modules))
    def test_preserves_all_native_channels_without_simresult(self):
        result=self.project();record=result.to_dict();data=record['frame_data']
        self.assertNotIsInstance(result,SimResult);self.assertEqual(data['fields'],self.view['fields'])
        self.assertEqual(data['node_ids'],[42,7,99]);self.assertEqual(data['edge_ids'],[33,9,22])
        self.assertEqual(data['fields']['rotations_rad'],[.2,0.,.7]);self.assertEqual(data['field_units']['rotations_rad'],'rad')
        self.assertNotIn('edge_stretches',data['fields']);self.assertEqual(record['omitted_native_fields'],['gradient'])
    def test_plain_simresult_loader_rejects_new_result_family(self):
        file=Path(self.temp.name)/'projection.json';file.write_bytes(self.project().canonical)
        with self.assertRaisesRegex(ValueError,'Unsupported result schema'):SimResult.load(file)
    def test_own_export_roundtrip_preserves_byte_identity(self):
        result=self.project();loaded=p.load_l3_projection(self.graph,self.binding,projection_bytes=result.canonical,expected_projection_sha256=result.sha256)
        self.assertEqual(result,loaded);self.assertEqual(result.sha256,hashlib.sha256(result.canonical).hexdigest())
    def test_no_private_metadata_or_provider_identity_in_export(self):
        raw=self.project().canonical
        for marker in (b'SYNTHETIC_METADATA',b'SYNTHETIC_PATH',b'SYNTHETIC_PROVIDER',b'private_marker',b'unknown_description'):self.assertNotIn(marker,raw)
    def test_snapshot_copies_and_recompute_refusal(self):
        result=self.project();copy=result.to_dict();copy['frame_data']['fields']['rotations_rad'][0]=99
        self.assertEqual(result.to_dict()['frame_data']['fields']['rotations_rad'][0],.2)
        with self.assertRaises(FrozenInstanceError):result.canonical=b'{}'
        with self.assertRaisesRegex(p.L3ProjectionError,'spring'):result.compute_detailed(self.graph,stiffness=2.)
    def test_explicit_renderer_inputs_reorder_by_id_and_hide_failed_edges(self):
        result=self.project();inputs=p.l3_render_inputs(self.graph,self.binding,result)
        self.assertEqual(inputs.node_ids,(42,7,99));self.assertEqual(inputs.active_edge_ids,(22,33))
        self.assertEqual(set(inputs.original.edges),{9,22,33});self.assertEqual(set(inputs.deformed.edges),{22,33})
        np.testing.assert_array_equal(inputs.axial_force_N,[2.,-3.]);np.testing.assert_array_equal(inputs.displacement_data,[[1.,.2],[0.,0.],[1.,1.]])
        np.testing.assert_array_equal(inputs.deformed.nodes[42].position,[11.,.2,0.]);self.assertEqual(inputs.coordinate_unit,'mm')
        self.assertFalse(inputs.displacement_data.flags.writeable);self.assertFalse(inputs.axial_force_N.flags.writeable)
        with self.assertRaises(ValueError):inputs.displacement_data.setflags(write=True)
        inputs.deformed.nodes[42].position[:]=99
    def test_actual_explicit_2d_consumers_render_synthetic_only(self):
        import matplotlib;matplotlib.use('Agg',force=True)
        import matplotlib.pyplot as plt
        from fibernet.viz.render import render_deformation,render_graph
        inputs=p.l3_render_inputs(self.graph,self.binding,self.project())
        a=render_deformation(**inputs.deformation_kwargs());b=render_graph(**inputs.force_kwargs())
        self.assertEqual(len(a.axes[1].collections[0].get_segments()),2)
        np.testing.assert_array_equal(a.axes[1].collections[0].get_segments()[0],[[0.,0.],[11.,.2]])
        self.assertEqual(len(b.axes[0].collections[0].get_segments()),2)
        plt.close(a);plt.close(b)
    def test_all_inactive_display_does_not_recreate_edges(self):
        self.view['fields']['active']=[False]*3;self.view['active_edge_ids']=[];self.view['failed_edge_ids']=[33,9,22]
        inputs=p.l3_render_inputs(self.graph,self.binding,self.project());self.assertEqual(inputs.deformed.num_edges,0)
        self.assertIsNone(inputs.deformation_kwargs()['displacement_data']);self.assertEqual(inputs.force_kwargs()['color_by'],'uniform')
        import matplotlib.pyplot as plt
        from fibernet.viz.render import render_deformation,render_graph
        a=render_deformation(**inputs.deformation_kwargs());b=render_graph(**inputs.force_kwargs())
        self.assertEqual(len(a.axes[1].collections),0);self.assertEqual(len(b.axes[0].collections),0);plt.close(a);plt.close(b)
    def test_display_copy_spatial_index_tracks_saved_positions(self):
        inputs=p.l3_render_inputs(self.graph,self.binding,self.project())
        self.assertEqual(inputs.deformed.add_node([11.,.2,0.],merge=True),42)
        self.assertEqual(inputs.deformed.num_nodes,3)
        self.assertNotEqual(inputs.deformed.add_node([10.,0.,0.],merge=True),42)
    def test_escaped_surrogate_is_projection_error(self):
        raw=b'{"x":"\\ud800"}'
        with self.assertRaises(p.L3ProjectionError):p.project_l3_result(self.graph,self.binding,frame_view_bytes=raw,expected_view_sha256=p._sha(raw))
    def test_oversized_graph_box_rejected_before_public_rebind(self):
        g=self.graph.copy();g.box_size=np.ones(100)
        raw=p._raw(self.view)
        with patch.object(p,'bind_l3',side_effect=AssertionError('Must reject before graph serialization')):
            with self.assertRaisesRegex(p.L3ProjectionError,'graph box'):p.project_l3_result(g,self.binding,frame_view_bytes=raw,expected_view_sha256=p._sha(raw))
    def test_graph_counter_bound_precedes_public_rebind(self):
        g=self.graph.copy();g._next_node_id=2**64
        raw=p._raw(self.view)
        with patch.object(p,'bind_l3',side_effect=AssertionError('Must reject before graph serialization')):
            with self.assertRaisesRegex(p.L3ProjectionError,'int64'):p.project_l3_result(g,self.binding,frame_view_bytes=raw,expected_view_sha256=p._sha(raw))
    def test_material_change_refused(self):
        g=self.graph.copy();next(iter(g.edges.values())).material.youngs_modulus*=2
        raw=p._raw(self.view)
        with self.assertRaises(ValueError):p.project_l3_result(g,self.binding,frame_view_bytes=raw,expected_view_sha256=p._sha(raw))
    def test_graph_mutated_after_projection_refused(self):
        result=self.project();g=self.graph.copy();g.nodes[42].position[0]+=1
        with self.assertRaises(ValueError):p.l3_render_inputs(g,self.binding,result)
    def test_all_supported_unit_pairs_keep_si_record_and_display_units(self):
        for unit in ('m','mm','um'):
            for radius in ('m','mm','um'):
                g,b,v=synthetic_fixture(unit,radius);raw=p._raw(v);r=p.project_l3_result(g,b,frame_view_bytes=raw,expected_view_sha256=p._sha(raw));i=p.l3_render_inputs(g,b,r)
                self.assertEqual(r.to_dict()['frame_data']['fields']['axial_force_N'],[-3.,99.,2.]);self.assertEqual(i.coordinate_unit,unit)
                np.testing.assert_allclose(i.deformed.nodes[42].position[:2]*{'m':1.,'mm':.001,'um':.000001}[unit],v['fields']['positions_m'][0],rtol=0,atol=2e-18)
    def test_wrong_hash_rejected(self):
        with self.assertRaisesRegex(p.L3ProjectionError,'identity'):self.project(expected='0'*64)
    def test_duplicate_json_key_rejected(self):
        raw=p._raw(self.view).replace(b'"frame":4',b'"frame":4,"frame":4')
        with self.assertRaisesRegex(p.L3ProjectionError,'Malformed'):p.project_l3_result(self.graph,self.binding,frame_view_bytes=raw,expected_view_sha256=p._sha(raw))
    def test_oversized_bytes_rejected(self):
        raw=b'0'*(p.MAX_BYTES+1)
        with self.assertRaisesRegex(p.L3ProjectionError,'Bounded bytes'):p.project_l3_result(self.graph,self.binding,frame_view_bytes=raw,expected_view_sha256=p._sha(raw))
    def test_invalid_root_container_rejected(self):
        raw=b'[]'
        with self.assertRaises(p.L3ProjectionError):p.project_l3_result(self.graph,self.binding,frame_view_bytes=raw,expected_view_sha256=p._sha(raw))
    def test_injected_export_metadata_refused(self):
        result=self.project();value=result.to_dict();value['private']='SYNTHETIC_DO_NOT_EXPORT';raw=p._raw(value)
        with self.assertRaises(p.L3ProjectionError):p.load_l3_projection(self.graph,self.binding,projection_bytes=raw,expected_projection_sha256=p._sha(raw))
    def test_injected_export_policy_refused(self):
        value=self.project().to_dict();value['force_policy']='reinterpret as spring';raw=p._raw(value)
        with self.assertRaises(p.L3ProjectionError):p.load_l3_projection(self.graph,self.binding,projection_bytes=raw,expected_projection_sha256=p._sha(raw))

MUTATIONS=[
 ('boolean_id',lambda v:v['node_ids'].__setitem__(0,True)),
 ('duplicate_ids',lambda v:v['node_ids'].__setitem__(0,7)),
 ('foreign_edge_id',lambda v:v['edge_ids'].__setitem__(0,123)),
 ('wrong_units',lambda v:v['field_units'].__setitem__('positions_m','mm')),
 ('wrong_row_axis',lambda v:v['field_rows'].__setitem__('rotations_rad','edge_ids')),
 ('reversed_moment_ends',lambda v:v['field_components'].__setitem__('end_moments_Nm',['node_j','node_i'])),
 ('wrong_reference',lambda v:v['reference_positions_m'][0].__setitem__(0,99.)),
 ('reversed_endpoints',lambda v:v['edge_node_ids'][0].reverse()),
 ('wrong_position',lambda v:v['fields']['positions_m'][0].__setitem__(0,99.)),
 ('boolean_numeric',lambda v:v['fields']['axial_force_N'].__setitem__(0,True)),
 ('integer_mask',lambda v:v['fields']['active'].__setitem__(0,1)),
 ('extra_translation_column',lambda v:v['fields']['displacements_m'][0].append(.2)),
 ('missing_rotation',lambda v:v['fields'].pop('rotations_rad')),
 ('extra_stretch_alias',lambda v:v['fields'].__setitem__('edge_stretches',[1.,1.,1.])),
 ('wrong_original_units',lambda v:v['original_units'].__setitem__('coordinate_unit','m')),
 ('wrong_active_ids',lambda v:v.__setitem__('active_edge_ids',[22,9])),
 ('wrong_failed_ids',lambda v:v.__setitem__('failed_edge_ids',[])),
 ('unknown_failure',lambda v:v.__setitem__('newly_failed_ids',[123])),
 ('boolean_frame',lambda v:v.__setitem__('frame',False)),
 ('false_new_execution',lambda v:v.__setitem__('new_execution',True)),
 ('unearned_license',lambda v:v.__setitem__('provider_license_verified',True)),
 ('bad_binding',lambda v:v['provenance'].__setitem__('binding_sha256','0'*64)),
 ('private_phase',lambda v:v['observation'].__setitem__('phase','/private/source')),
 ('bool_energy',lambda v:v['observation'].__setitem__('internal_energy_J',True)),
]
for name,mutate in MUTATIONS:
    def test(self,mutate=mutate):
        mutate(self.view)
        with self.assertRaises((p.L3ProjectionError,ValueError)):self.project()
    setattr(ProjectionTests,'test_resealed_'+name,test)

if __name__=='__main__':unittest.main(verbosity=2)
