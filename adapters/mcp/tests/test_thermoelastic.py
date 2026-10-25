"""Wire/layout/native-spy and atomic publication tests; no physical solve."""
import copy
import ast
from contextlib import contextmanager
import importlib.abc
import json
import runpy
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'fibernet','numpy','scipy','networkx','torch','taichi','lammps','exudyn'}:
            raise AssertionError('No scientific runtime in contract test: '+fullname)
sys.meta_path.insert(0,NoNative())

import anyio
import jsonschema
from fibernet_mcp import server
from fibernet_mcp import thermoelastic as th
from fibernet_mcp.safety import _CoreVerification, canonical, sha, read_at

GID='1'*32

def record():
    return {'version':'3.1','dimension':2,'tolerance':1e-8,
            'nodes':[{'id':29,'position':[1.,0.,0.]},{'id':7,'position':[0.,0.,0.]}],
            'edges':[{'id':91,'node_i':7,'node_j':29,'radius':.001,'material':{},'internal_points':None}]}

def args():
    return {'graph_id':GID,'length_unit':'m','E_Pa':1e9,'stress_free_temperature_K':300.,
            'temperature_K':[{'node_id':29,'value_K':300.},{'node_id':7,'value_K':300.}],
            'expansion_alpha_per_K':[{'edge_id':91,'value_per_K':0.}],
            'constraints':[{'node_id':7,'ux_m':0.,'uy_m':0.,'theta_rad':0.}],
            'loads':[{'node_id':29,'Fx_N':1e-6,'Fy_N':0.,'Mz_Nm':0.}]}

def info():
    return {'node_count':2,'edge_count':1,'length_unit':'m','frame_id':'test_xy',
            'unit_provenance':'operator_asserted','frame_provenance':'operator_asserted'}

def request(): return th.snapshot_request(info(),args(),record(),len(canonical(record())))

def snapshot(req=None):
    """Synthetic contract layout, deliberately not a mechanical reference."""
    req=request() if req is None else req
    temp,alpha,con,loads=th._maps(req,record())
    ids=[7,29];prescribed=[con.get(i,[None]*3) for i in ids]
    mask=[[v is not None for v in row] for row in prescribed]
    value={'profile':th.PROFILE,'node_ids':ids,'edge_ids':[91],
        'original_geometry':{'dimension':2,'positions_xyz':[[0.,0.,0.],[1.,0.,0.]],'radii':[.001],'internal_points':[None]},
        'endpoints_by_id':[[7,29]],'positions_m':[[0.,0.],[1.,0.]],'radii_m':[.001],
        'length_unit':'m','E_Pa':req['E_Pa'],'stress_free_temperature_K':req['stress_free_temperature_K'],
        'temperature_K':[temp[i] for i in ids],'expansion_alpha_per_K':[alpha[91]],
        'constraint_mask':mask,'prescribed_values':[[0. if v is None else v for v in row] for row in prescribed],
        'loads':[loads.get(i,[0.,0.,0.]) for i in ids]}
    value.update(input_sha256=sha(canonical(value)),units=th.UNITS,
        interpolation='linear_endpoint_temperature_mean_axial/1',numerical_regularization=False,
        total_axial_strain=[0.],thermal_axial_strain=[0.],elastic_axial_strain=[0.],
        displacement=[[0.]*3]*2,sigma_axial=[0.],sigma_bending=[0.],sigma_total=[0.],
        moments=[[0.,0.]],edge_forces=[[0.]*3],reactions=[[0.]*3]*2,
        free_dof_indices=[i*3+k for i,row in enumerate(mask) for k,v in enumerate(row) if not v],
        component_balance=[0.]*3,component_acceptance_bound=[1e-10]*3,balance_origin_node_id=7,
        applicability={'model':'linear_planar_welded_Euler_Bernoulli','stability_qualified':False,
            'fiber_signs':[-1,1],'endpoint_order':['node_i','node_j'],
            'ceilings':{'thermal_strain':1e-5,'mean_total_axial_strain':1e-4,'rotation_and_slope_rad':1e-3,'extreme_fiber_strain':1e-4},
            'edge_rows':[{'transverse_polynomial_normalized_x_m':[0.]*4,'slope_sample_x_over_L':[0.,1.],
                'slope_rad':[0.,0.],'transverse_chord_slope_rad':0.,'endpoint_curvature_per_m':[0.,0.],
                'endpoint_thermal_strain':[0.,0.],'endpoint_total_fiber_strain':[[0.,0.]]*2,
                'endpoint_elastic_fiber_strain':[[0.,0.]]*2}]})
    value['free_residual']=[0.]*len(value['free_dof_indices']);value['free_acceptance_bound']=[1e-10]*len(value['free_dof_indices'])
    return value

@contextmanager
def native_spy(req):
    graph=types.SimpleNamespace(nodes={r['id']:r for r in record()['nodes']},edges={91:record()['edges'][0]},
        to_dict=lambda:dict(record(),nodes=sorted(record()['nodes'],key=lambda r:r['id'])))
    decode=Mock(return_value=graph);owner=Mock(return_value=types.SimpleNamespace(to_dict=lambda:snapshot(req)))
    modules={name:types.ModuleType(name) for name in ['fibernet','fibernet.core','fibernet.core.structure_graph','fibernet.sim','fibernet.sim.thermoelastic']}
    modules['fibernet.core.structure_graph'].StructureGraph=types.SimpleNamespace(from_dict=decode)
    modules['fibernet.sim.thermoelastic'].solve_thermoelastic_network_snapshot=owner
    with patch.dict(sys.modules,modules): yield owner,decode

@contextmanager
def store_context():
    with tempfile.TemporaryDirectory() as folder:
        folder=Path(folder);out=folder/'out';out.mkdir();source=folder/'graph.json';source.write_bytes(canonical(record()))
        manifest=folder/'manifest.json';manifest.write_bytes(canonical({'source_tree':server.TREE,'graphs':[{'path':str(source),'sha256':sha(source.read_bytes()),'length_unit':'m','frame_id':'test_xy'}]}))
        with patch.object(server,'_verify_core_result'):
            store=server.ManagedStore(str(manifest),str(out),_CoreVerification('/not-exposed','a'*64,246))
        try: yield store,next(iter(store.graphs)),out
        finally: store.close()

class SnapshotContract(unittest.TestCase):
    def test_exact_eighth_schema(self):
        tools=json.loads(Path(server.__file__).with_name('tool_schemas.json').read_text())['tools']
        self.assertEqual(len(tools),8);self.assertEqual(tools[-1]['name'],'fibernet_solve_thermoelastic_network_snapshot')
        for tool in tools:
            for direction in ('inputSchema','outputSchema'):jsonschema.Draft202012Validator.check_schema(tool[direction])
        jsonschema.Draft202012Validator(tools[-1]['inputSchema']).validate(args())
        for field in args():
            bad=args();del bad[field]
            with self.subTest(field=field),self.assertRaises(jsonschema.ValidationError):
                jsonschema.Draft202012Validator(tools[-1]['inputSchema']).validate(bad)

    def test_valid_wire_and_detached_native_once(self):
        req=request();original=copy.deepcopy(req)
        with native_spy(req) as (owner,decode):result=th.solve_snapshot(record(),req,'b'*64)
        owner.assert_called_once();decode.assert_called_once()
        self.assertEqual(result['snapshot'],snapshot(req));self.assertEqual(req,original)
        self.assertNotIn('size_limits',owner.call_args.kwargs)
        self.assertEqual(owner.call_args.kwargs['temperature_K'],{29:300.,7:300.})
        self.assertEqual(owner.call_args.kwargs['constraints'],{7:[0.,0.,0.]})
        self.assertEqual(result['request_sha256'],sha(canonical(req)))

    def test_nonzero_temperature_forwarding_is_only_mock(self):
        a=args();a['temperature_K'][0]['value_K']=301.;a['expansion_alpha_per_K'][0]['value_per_K']=-1e-6
        req=th.snapshot_request(info(),a,record(),100)
        with native_spy(req) as (owner,_):result=th.solve_snapshot(record(),req,'b'*64)
        self.assertIn('snapshot',result);owner.assert_called_once()
        self.assertEqual(owner.call_args.kwargs['temperature_K'][29],301.)
        self.assertEqual(owner.call_args.kwargs['expansion_alpha_per_K'][91],-1e-6)

    def test_empty_loads_nullable_constraints_and_mm_are_not_double_scaled(self):
        a=args();a['loads']=[];a['constraints'][0]['theta_rad']=None;a['length_unit']='mm'
        req=th.snapshot_request(dict(info(),length_unit='mm'),a,record(),100)
        value=snapshot(req);value['length_unit']='mm';value['positions_m']=[[0.,0.],[.001,0.]];value['radii_m']=[.000001]
        value['input_sha256']=sha(canonical({k:value[k] for k in th.IDENTITY_FIELDS}))
        th.validate_snapshot(value,record(),req)
        self.assertEqual(value['prescribed_values'],[[0.,0.,0.],[0.,0.,0.]])
        self.assertEqual(value['loads'],[[0.,0.,0.],[0.,0.,0.]])

    def test_worker_raw_source_and_request_binding_before_native(self):
        with patch('resource.setrlimit'):
            worker=runpy.run_module('fibernet_mcp.worker',run_name='snapshot_worker_contract_test')
        main=worker['main'];main.__globals__['_verify_core_result']=lambda:types.SimpleNamespace(provenance=lambda:{'test':'verified'})
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);source=folder/'source.json';wire=folder/'request.json';output=folder/'output.json'
            source.write_bytes(canonical(record()));wire.write_bytes(canonical(request()))
            for fault in ('source_hash','request_hash','source_size','request_size','request_symlink'):
                with self.subTest(fault=fault):
                    source.write_bytes(canonical(record()));wire.write_bytes(canonical(request()));supplied=wire
                    if fault=='source_size':source.write_bytes(b' '*(th.MAX_SNAPSHOT_SOURCE_BYTES+1))
                    if fault=='request_size':wire.write_bytes(b' '*(th.MAX_SNAPSHOT_REQUEST_BYTES+1))
                    if fault=='request_symlink':
                        supplied=folder/'link.json';supplied.symlink_to(wire)
                    source_hash='0'*64 if fault=='source_hash' else sha(source.read_bytes())
                    request_hash='0'*64 if fault=='request_hash' else sha(wire.read_bytes())
                    argv=['worker','thermoelastic_snapshot',str(source),str(output),source_hash,str(supplied),request_hash]
                    try:
                        with patch.object(sys,'argv',argv),patch.object(th,'solve_snapshot') as owner:
                            with self.assertRaises((ValueError,OSError)):main()
                        owner.assert_not_called();self.assertFalse(output.exists())
                    finally:
                        if supplied!=wire:supplied.unlink()

    def test_bad_ids_coverage_numbers_and_units(self):
        for field in ('temperature_K','expansion_alpha_per_K','constraints','loads'):
            idkey='edge_id' if field=='expansion_alpha_per_K' else 'node_id'
            for mode in ('duplicate','unknown','bool','string','float'):
                bad=args()
                if mode=='duplicate':bad[field].append(copy.deepcopy(bad[field][0]))
                else:bad[field][0][idkey]={'unknown':999,'bool':True,'string':'7','float':7.0}[mode]
                with self.subTest(field=field,mode=mode),self.assertRaises(th.SnapshotContractError):th.snapshot_request(info(),bad,record(),100)
        for field in ('temperature_K','expansion_alpha_per_K'):
            bad=args();bad[field].pop()
            with self.assertRaises(th.SnapshotContractError):th.snapshot_request(info(),bad,record(),100)
        for field,value in [('E_Pa',True),('E_Pa',0),('E_Pa',float('inf')),('stress_free_temperature_K',0),('length_unit','unspecified'),('length_unit','mm')]:
            bad=args();bad[field]=value
            with self.subTest(field=field,value=value),self.assertRaises(th.SnapshotContractError):th.snapshot_request(info(),bad,record(),100)
        for field in ('constraints','loads'):
            bad=args();key='ux_m' if field=='constraints' else 'Fx_N';bad[field][0][key]=True
            with self.assertRaises(th.SnapshotContractError):th.snapshot_request(info(),bad,record(),100)

    def test_caps_and_no_hidden_arguments(self):
        for update in ({'node_count':193},{'edge_count':257}):
            with self.assertRaises(th.SnapshotContractError):th.snapshot_request(dict(info(),**update),args(),record(),100)
        with self.assertRaises(th.SnapshotContractError):th.snapshot_request(info(),args(),record(),th.MAX_SNAPSHOT_SOURCE_BYTES+1)
        with patch.object(th,'MAX_SNAPSHOT_REQUEST_BYTES',1),self.assertRaises(th.SnapshotContractError):th.snapshot_request(info(),args(),record(),100)
        for field in ('size_limits','solver','output_path','temperature_default'):
            with self.subTest(field=field),self.assertRaises(th.SnapshotContractError):th.snapshot_request(info(),dict(args(),**{field:1}),record(),100)

    def test_every_native_field_is_mandatory(self):
        value=snapshot();self.assertEqual(len(value),36);th.validate_snapshot(value,record(),request())
        for field in value:
            bad=copy.deepcopy(value);del bad[field]
            with self.subTest(field=field),self.assertRaises(th.SnapshotContractError):th.validate_snapshot(bad,record(),request())
        with self.assertRaises(th.SnapshotContractError):th.validate_snapshot(dict(value,extra=1),record(),request())

    def test_native_identity_and_derived_shape_tampering(self):
        mutations={'profile':'other','input_sha256':'0'*64,'node_ids':[True,29],'edge_ids':[92],
          'endpoints_by_id':[[7,30]],'length_unit':'mm','E_Pa':2e9,'temperature_K':[300.,301.],
          'loads':[[1.,0.,0.],[0.,0.,0.]],'constraint_mask':[[1,True,True],[False]*3],
          'displacement':[[0.,0.]]*2,'sigma_total':[float('nan')],'free_dof_indices':[3,4],
          'free_acceptance_bound':[-1.,1.,1.],'component_acceptance_bound':[0.,1.,1.],
          'units':dict(th.UNITS,force='kN'),'balance_origin_node_id':29,'numerical_regularization':True}
        for field,value in mutations.items():
            bad=snapshot();bad[field]=value
            with self.subTest(field=field),self.assertRaises(th.SnapshotContractError):th.validate_snapshot(bad,record(),request())
        bad=snapshot();bad['applicability']['edge_rows'][0]['slope_rad']=[0.]
        with self.assertRaises(th.SnapshotContractError):th.validate_snapshot(bad,record(),request())
        with patch.object(th,'MAX_SNAPSHOT_RESULT_BYTES',1),self.assertRaises(th.SnapshotContractError):th.validate_snapshot(snapshot(),record(),request())

    def test_worker_binding_and_sanitized_errors(self):
        result={'snapshot':snapshot(),'source_sha256':'b'*64,'request_sha256':sha(canonical(request())),'_core_provenance':{}}
        th.validate_worker_snapshot(result,record(),request(),'b'*64,{})
        for field in ('source_sha256','request_sha256','_core_provenance'):
            with self.assertRaises(th.SnapshotContractError):th.validate_worker_snapshot(dict(result,**{field:'c'*64}),record(),request(),'b'*64,{})
        for exception,code in [(ValueError('/secret/path'), 'SNAPSHOT_DOMAIN_REJECTED'),(FloatingPointError('/secret/path'),'SNAPSHOT_DOMAIN_REJECTED'),(ImportError('/secret/path'),'CORE_DEPENDENCY_UNAVAILABLE')]:
            with native_spy(request()) as (owner,_):
                owner.side_effect=exception;result=th.solve_snapshot(record(),request(),'b'*64)
            self.assertEqual(result['error_code'],code);self.assertNotIn('/secret',json.dumps(result))

    def test_lossy_source_refuses_before_native(self):
        with native_spy(request()) as (owner,decode):
            decode.return_value.to_dict=lambda:dict(record(),metadata={'lost':True})
            result=th.solve_snapshot(record(),request(),'b'*64)
        owner.assert_not_called();self.assertEqual(result['error_code'],'SNAPSHOT_SOURCE_ROUNDTRIP')

    def test_callback_caps_and_integrity_before_read_or_worker(self):
        class Model:
            def __init__(self,**kw):self.__dict__.update(kw)
            def model_dump_json(self,**kw):
                return json.dumps(self.__dict__,default=lambda x:x.__dict__)
        tree=ast.parse(Path(server.__file__).read_text())
        serve=next(n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='serve')
        call=copy.deepcopy(next(n for n in serve.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='call'))
        call.decorator_list=[]
        with store_context() as (store,gid,_):
            ns=dict(vars(server));ns.update(store=store,jsonschema=jsonschema,
                mt=types.SimpleNamespace(CallToolResult=Model,TextContent=Model))
            tools=json.loads(Path(server.__file__).with_name('tool_schemas.json').read_text())['tools']
            ns['indexed']={d['name']:d for d in tools}
            sup=types.SimpleNamespace(run=Mock(side_effect=AssertionError('must not spawn')));ns['sup']=sup
            exec(compile(ast.Module(body=[call],type_ignores=[]),'<callback-only>','exec'),ns)
            arguments=dict(args(),graph_id=gid)
            for fault in ('count','size','digest'):
                with self.subTest(fault=fault):
                    old=store.graphs[gid]['node_count']
                    source=Path(store.path)/gid;before=source.read_bytes()
                    try:
                        if fault=='count':store.graphs[gid]['node_count']=193
                        if fault=='digest':source.write_bytes(before+b' ')
                        reader=Mock(side_effect=AssertionError('must not read')) if fault!='digest' else server.read_at
                        ns['read_at']=reader
                        if fault=='size':
                            with patch.object(server.os,'stat',return_value=types.SimpleNamespace(st_size=th.MAX_SNAPSHOT_SOURCE_BYTES+1)):
                                reply=anyio.run(ns['call'],'fibernet_solve_thermoelastic_network_snapshot',arguments)
                        else:reply=anyio.run(ns['call'],'fibernet_solve_thermoelastic_network_snapshot',arguments)
                        self.assertTrue(reply.isError);self.assertFalse(store.artifacts);sup.run.assert_not_called()
                        if fault!='digest':reader.assert_not_called()
                    finally:store.graphs[gid]['node_count']=old;source.write_bytes(before)

    def test_new_worker_timeout_and_cancel_reap_without_publication(self):
        from test_shared_lifecycle import ControlledProcess
        async def exercise(store,gid,cancel,hard_kill):
            process=ControlledProcess(hard_kill=hard_kill)
            supervisor=server.Supervisor(store,timeout=.001)
            source=(Path(store.path)/gid).read_bytes()
            result=None
            with anyio.CancelScope() as scope:
                def launch(*arguments,**kwargs):
                    if cancel:scope.cancel()
                    return process
                with patch.object(server.subprocess,'Popen',side_effect=launch),patch.object(server.os,'killpg',side_effect=process.signal):
                    result=await supervisor.run('thermoelastic_snapshot',gid,request())
            if cancel:self.assertTrue(scope.cancelled_caught)
            else:self.assertEqual(result['error']['code'],'WORKER_TIMEOUT')
            self.assertEqual(process.wait_calls,1);self.assertTrue(all(process.shielded_observations))
            self.assertIsNone(supervisor.active);self.assertFalse(store.artifacts)
            self.assertEqual((Path(store.path)/gid).read_bytes(),source)
            self.assertFalse(list(Path(store.path).glob('request-*')))
            self.assertFalse(list(Path(store.path).glob('worker-*')))
        for cancel in (False,True):
            for hard_kill in (False,True):
                with self.subTest(cancel=cancel,hard_kill=hard_kill),store_context() as (store,gid,_):
                    anyio.run(exercise,store,gid,cancel,hard_kill)

    def test_publish_success_manifest_bytes_and_dedup(self):
        async def exercise(store,gid):
            reply=await server.publish_snapshot(store,gid,snapshot(),store.core_provenance,request(),lambda r:None)
            a=reply['artifact'];self.assertEqual(a['kind'],'thermoelastic_network_snapshot')
            self.assertEqual(read_at(store.fd,a['artifact_id']),canonical(snapshot()))
            again=await server.publish_snapshot(store,gid,snapshot(),store.core_provenance,request(),lambda r:None)
            self.assertEqual(again,reply);self.assertEqual(len(store.artifacts),1)
        with store_context() as (store,gid,_):anyio.run(exercise,store,gid)

    def test_reply_failure_cancel_and_io_failure_publish_nothing(self):
        async def exercise(store,gid):
            before=set(Path(store.path).iterdir());total=store.total
            with self.assertRaises(ValueError):
                await server.publish_snapshot(store,gid,snapshot(),store.core_provenance,request(),lambda r:(_ for _ in ()).throw(ValueError('schema')))
            self.assertFalse(store.artifacts)
            with anyio.CancelScope() as scope:
                scope.cancel()
                await server.publish_snapshot(store,gid,snapshot(),store.core_provenance,request(),lambda r:None)
                self.fail('Cancellation must win')
            self.assertTrue(scope.cancelled_caught);self.assertFalse(store.artifacts)
            with patch('fibernet_mcp.safety.os.fsync',side_effect=OSError('save failure')):
                with self.assertRaises(OSError):await server.publish_snapshot(store,gid,snapshot(),store.core_provenance,request(),lambda r:None)
            self.assertFalse(store.artifacts);self.assertEqual(store.total,total);self.assertEqual(set(Path(store.path).iterdir()),before)
        with store_context() as (store,gid,_):anyio.run(exercise,store,gid)

if __name__=='__main__':unittest.main()
