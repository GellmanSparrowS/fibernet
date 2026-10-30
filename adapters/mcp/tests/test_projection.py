"""Report-only MCP/native parity; no generation, mesh, physics or training run."""
import ast
import asyncio
import base64
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from unittest.mock import patch

import anyio
import jsonschema
import psutil
import pytest

from projection_fixtures import fixture_bytes
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import mcp.types as mt

from fibernet.core.structure_graph import StructureGraph
from fibernet.core.fiber_network_profile import FiberNetworkProfile, FiberNetworkProjectionLimits, FiberNetworkProjectionError
from fibernet_mcp import projection, server
from fibernet_mcp.projection import projection_request, project_report, ProjectionContractError
from fibernet_mcp.safety import canonical, sha, ContractError, graph_preflight, strict_json

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT.parents[1]
TOOL = 'fibernet_project_fiber_network_report'
INFO = {'node_count': 3, 'edge_count': 3, 'length_unit': 'm', 'frame_id': 'lab_xyz'}


def small():
    # Deliberately non-ID order, directed parallel physical edges, and a null-shear
    # material reconstructed by the existing native loader, not a new wire model.
    return {'version':'3.1','dimension':3,'tolerance':1e-6,
        'nodes':[{'id':90,'position':[1.,0.,0.]},{'id':10,'position':[0.,0.,0.]},
                 {'id':2**53-1,'position':[0.,0.,0.]}],
        'edges':[{'id':900,'node_i':90,'node_j':10,'radius':.1,'material':{}},
                 {'id':7,'node_i':10,'node_j':90,'radius':.1,'material':{'shear_modulus':None},'internal_points':[]},
                 {'id':2**53-1,'node_i':10,'node_j':90,'radius':.1,'material':{}}]}


def native_report(data, request):
    return StructureGraph.from_dict(data).to_fiber_network(profile=FiberNetworkProfile(
        **{**request, 'limits': FiberNetworkProjectionLimits(**request['limits'])}))[1]


def configure(tmp_path, entries, timeout=10, guard=False):
    graphs=[]
    for i,(data,unit,frame) in enumerate(entries):
        path=tmp_path/('graph-'+str(i)+'.json')
        raw=data if isinstance(data,bytes) else canonical(data)
        path.write_bytes(raw)
        graphs.append({'path':str(path),'sha256':sha(raw),'length_unit':unit,'frame_id':frame})
    manifest=tmp_path/'manifest.json';manifest.write_bytes(canonical({'source_tree':server.TREE,'graphs':graphs}))
    output=tmp_path/'output';output.mkdir()
    paths=[str(ROOT/'src'),str(CORE),os.environ.get('PYTHONPATH','')]
    guards=tmp_path/'provider-guards'
    if guard:
        guards.mkdir()
        # A test-only import observer is inherited by real server and worker
        # interpreters. It neither changes native sources nor delays execution.
        (guards/'sitecustomize.py').write_text('''import atexit, importlib.abc, json, os, pathlib, sys
blocked=set("taichi lammps torch torch_geometric sklearn pandas gym gymnasium stable_baselines3 skopt pyvista vtk matplotlib mpl_toolkits plotly".split())
attempts=[]
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in blocked:
   attempts.append(fullname)
   raise ImportError('projection test blocks external scientific providers')
sys.meta_path.insert(0,Guard())
def save():
 pathlib.Path('''+repr(str(guards))+''',str(os.getpid())+'.json').write_text(json.dumps({'attempts':attempts,'loaded':[name for name in sys.modules if name.split('.')[0] in blocked]}))
atexit.register(save)
''')
        paths.insert(0,str(guards))
    env={**os.environ,'PYTHONPATH':os.pathsep.join(paths),'PYTHONDONTWRITEBYTECODE':'1',
         'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
    params=StdioServerParameters(command=sys.executable,args=['-B','-m','fibernet_mcp.server',
        '--manifest',str(manifest),'--output-root',str(output),'--timeout',str(timeout),'--audit','audit.json'],env=env)
    return params,output,graphs,guards


async def content(session, artifact):
    assert artifact['kind']=='fiber_network_projection_report'
    assert json.loads((await session.read_resource(artifact['uri'])).contents[0].text)==artifact
    assert (await session.call_tool('fibernet_artifact_info',{'artifact_id':artifact['artifact_id']})).structuredContent['artifact']==artifact
    chunks=[];offset=0
    while True:
        response=await session.call_tool('fibernet_artifact_read',{'artifact_id':artifact['artifact_id'],'offset':offset,'length':32768})
        assert not response.isError
        assert len(response.model_dump_json().encode())<65536
        item=response.structuredContent;chunk=base64.b64decode(item['data'],validate=True)
        assert len(chunk)<=32768
        chunks.append(chunk);offset=item['next_offset']
        if item['eof']:break
    raw=b''.join(chunks)
    assert len(raw)==artifact['byte_size'] and sha(raw)==artifact['sha256']
    return raw,len(chunks)


def test_one_profiled_dispatch_and_unchanged_native_owner():
    data=small();request=projection_request(INFO,{})
    expected=native_report(data,request)
    original=StructureGraph.to_fiber_network;calls=[]
    def observed(self,*,profile=None):
        calls.append(profile)
        return original(self,profile=profile)
    with patch.object(StructureGraph,'to_fiber_network',observed):
        assert project_report(data,request)=={'projection_report':expected}
    assert len(calls)==2 and isinstance(calls[0],FiberNetworkProfile) and calls[1] is None
    graph=StructureGraph.from_dict(data)
    assert graph.edges[7].material.shear_modulus is not None
    assert expected['missing_material_edge_ids']==[]
    assert [row['edge_id'] for row in expected['edges']]==[900,7,2**53-1]


@pytest.mark.parametrize('limit',list(projection.PROJECTION_LIMITS))
def test_all_native_admission_limits_and_no_partial_result(limit):
    data=small()
    with patch.dict(projection.PROJECTION_LIMITS,{limit:1}):
        try: request=projection_request(INFO,{})
        except ProjectionContractError as exc:
            assert exc.code=='PROJECTION_GRAPH_LIMIT';return
        result=project_report(data,request)
    assert result['error_code']=='PROJECTION_LIMIT' and 'projection_report' not in result


def test_bounded_diagnostics_group_order_and_giant_id_refusal():
    data=small()
    data['edges'][0]['internal_points']=[[.5,.2,0.]]
    data['edges'][1]['node_j']=2**53-1
    request=projection_request(INFO,{})
    actual=project_report(data,request)
    assert [v['kind'] for v in actual['diagnostics']['violations']]==['internal_points','short_edges']
    assert [v['edge_ids'] for v in actual['diagnostics']['violations']]==[[900],[7]]
    with patch.object(StructureGraph,'to_fiber_network',side_effect=FiberNetworkProjectionError({'internal_points':tuple(range(140))})):
        result=project_report(small(),request)
    diagnostic=result['diagnostics']['violations'][0]
    assert diagnostic=={'kind':'internal_points','edge_ids':list(range(128)),'total_count':140,'truncated':True}
    for violations in ({'unexpected':(1,)},{'internal_points':(True,)},{'internal_points':(2**10000,)}):
        with patch.object(StructureGraph,'to_fiber_network',side_effect=FiberNetworkProjectionError(violations)):
            assert project_report(small(),request)['error_code']=='PROJECTION_OUTPUT_CONTRACT_FAILURE'


def test_real_stdio_saved_complex_report_reference(tmp_path):
    reference=json.loads((ROOT/'fixtures/projection_reference.json').read_text())
    source=fixture_bytes('projection_native31.json');expected=fixture_bytes('projection_report.json')
    assert sha(source)==reference['source_sha256'] and sha(expected)==reference['native_report_sha256']
    assert len(expected)==458163
    params,output,graphs,guards=configure(tmp_path,[(source,'m','lab_xyz')],guard=True)
    async def run():
        with anyio.fail_after(30):
            async with stdio_client(params) as (reader,writer):
                async with ClientSession(reader,writer) as session:
                    assert (await session.initialize()).protocolVersion=='2025-11-25'
                    listing=await session.list_tools()
                    assert len(listing.tools)==9 and len(listing.model_dump_json().encode())<65536
                    definition=next(t for t in listing.tools if t.name==TOOL)
                    assert definition.execution.taskSupport=='forbidden'
                    provenance=json.loads((await session.read_resource('fibernet://provenance/1')).contents[0].text)
                    manifest=json.loads((ROOT/'src/fibernet_mcp/core_manifest.json').read_text())
                    assert provenance['canonical_core_manifest_sha256']==sha(canonical(manifest))
                    assert provenance['source_file_count']==250
                    catalog=(await session.call_tool('fibernet_catalog',{})).structuredContent
                    assert 'project_fiber_network_report' in catalog['capabilities']
                    graph=catalog['graphs'][0];gid=graph['graph_id']
                    strict=await session.call_tool(TOOL,{'graph_id':gid})
                    assert strict.isError and strict.structuredContent['error']['code']=='PROJECTION_POLICY_REJECTED'
                    ids=json.loads(expected)['nonempty_internal_point_edge_ids']
                    assert strict.structuredContent['diagnostics']=={'violations':[{'kind':'internal_points','edge_ids':ids,'total_count':112,'truncated':False}]}
                    assert not list(next(output.glob('fibernet-mcp-*')).glob('worker-*'))
                    call=await session.call_tool(TOOL,{'graph_id':gid,'geometry_policy':'endpoint_chord'})
                    assert not call.isError,call
                    value=call.structuredContent
                    assert value['source_sha256']==sha(source) and value['status']=='completed'
                    assert value['scope']=='checked_projection_report_only'
                    assert value['request_sha256']==sha(canonical(value['effective_profile']))
                    assert value['counts']==json.loads(expected)['counts']
                    raw,pages=await content(session,value['artifact'])
                    assert raw==expected and pages==14
                    again=await session.call_tool(TOOL,{'graph_id':gid,'geometry_policy':'endpoint_chord','short_edge_policy':'reject'})
                    assert again.structuredContent['artifact']==value['artifact']
                    for args in ({'graph_id':gid,'geometry_policy':'repair'},{'graph_id':gid,'limits':{}},{'graph_id':'0'*32}):
                        failed=await session.call_tool(TOOL,args)
                        assert failed.isError and 'artifact' not in failed.structuredContent
                    (tmp_path/'parity.json').write_bytes(canonical({'source_sha256':sha(source),'report_sha256':sha(raw),'bytes':len(raw),'pages':pages,'counts':value['counts'],'provenance':provenance,'reference':'frozen native report; exact native3.1 reload identity established by original acceptance','scope':'interface and report parity only; no physics, mesh or network transfer'}))
    anyio.run(run)
    audit=json.loads((output/'audit.json').read_text())
    assert len(audit['workers'])==3 and audit['artifacts']==1 and all(w['reaped'] for w in audit['workers'])
    assert not list(output.glob('fibernet-mcp-*'))
    reports=list(guards.glob('*.json'));assert len(reports)>=4
    assert all(json.loads(p.read_text())=={'attempts':[],'loaded':[]} for p in reports)
    assert fixture_bytes('projection_native31.json')==source


def test_real_stdio_saved_variants_and_errors(tmp_path):
    data=small();curved=copy.deepcopy(data);curved['edges'][0]['internal_points']=[[.5,.2,0.]]
    curved['edges'][1]['node_j']=2**53-1
    conflict=copy.deepcopy(data);conflict['metadata']={'length_unit':'mm'}
    big=copy.deepcopy(data);big['edges'][0]['segments']=8193
    overflow=copy.deepcopy(data);overflow['edges'][0]['radius']=1e308
    entries=[(data,'m','lab_xyz'),(curved,'m','lab_xyz'),(conflict,'m','lab_xyz'),(big,'m','lab_xyz'),(data,'unspecified','lab_xyz'),(data,'m','unknown'),(overflow,'m','lab_xyz')]
    params,output,_,_=configure(tmp_path,entries)
    expected=native_report(strict_json((tmp_path/'graph-0.json').read_bytes()),projection_request(INFO,{}))
    lossy_request=projection_request(INFO,{'geometry_policy':'endpoint_chord','short_edge_policy':'omit_legacy'})
    lossy=native_report(strict_json((tmp_path/'graph-1.json').read_bytes()),lossy_request)
    async def run():
        async with stdio_client(params) as (reader,writer):
            async with ClientSession(reader,writer) as session:
                await session.initialize()
                graphs=(await session.call_tool('fibernet_catalog',{})).structuredContent['graphs']
                for i,args,wanted in [(0,{},expected),(1,{'geometry_policy':'endpoint_chord','short_edge_policy':'omit_legacy'},lossy)]:
                    result=await session.call_tool(TOOL,{'graph_id':graphs[i]['graph_id'],**args})
                    assert not result.isError,result
                    raw,_=await content(session,result.structuredContent['artifact'])
                    assert raw==canonical(wanted)
                    if i==0:
                        repeated=await session.call_tool(TOOL,{'graph_id':graphs[i]['graph_id'],'geometry_policy':'reject_internal_points','short_edge_policy':'reject'})
                        assert repeated.structuredContent['artifact']==result.structuredContent['artifact']
                for i,code in [(1,'PROJECTION_POLICY_REJECTED'),(2,'INVALID_PROJECTION_INPUT'),(3,'PROJECTION_LIMIT'),(4,'DECLARED_LENGTH_UNIT_REQUIRED'),(5,'DECLARED_FRAME_REQUIRED'),(6,'PROJECTION_NUMERIC_RANGE')]:
                    result=await session.call_tool(TOOL,{'graph_id':graphs[i]['graph_id']})
                    assert result.isError and result.structuredContent['error']['code']==code
                    assert 'artifact' not in result.structuredContent
                assert not (await session.call_tool('fibernet_catalog',{})).isError
    anyio.run(run)
    assert json.loads((output/'audit.json').read_text())['artifacts']==2


def test_wire_material_and_integer_admission_unchanged():
    graph_preflight(small())
    for value in (None,):
        data=small();data['edges'][0]['material']=value
        with pytest.raises(ContractError):graph_preflight(data)
    data=small();del data['edges'][0]['material']
    with pytest.raises(ContractError):graph_preflight(data)
    for value in (True,3.0,2**53):
        data=small();data['edges'][0]['id']=value
        with pytest.raises(ContractError):graph_preflight(data)


def test_no_publication_for_output_contract_failures_or_provenance(tmp_path):
    params,output,_,_=configure(tmp_path,[(small(),'m','lab_xyz')])
    manifest=tmp_path/'manifest.json';store=server.ManagedStore(str(manifest),str(output))
    request=projection_request(INFO,{})
    report=native_report(small(),request)
    # Source-extracted callback control is explicitly not the real-client test.
    tree=ast.parse((ROOT/'src/fibernet_mcp/server.py').read_text())
    serve=next(n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='_serve_admitted')
    call=copy.deepcopy(next(n for n in serve.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='call'));call.decorator_list=[]
    ns={**vars(server),'store':store,'jsonschema':jsonschema,'mt':mt,
        'indexed':{d['name']:d for d in json.loads((ROOT/'src/fibernet_mcp/tool_schemas.json').read_text())['tools']}}
    exec(compile(ast.Module(body=[call],type_ignores=[]),'<projection-publication-control>','exec'),ns)
    gid=next(iter(store.graphs))
    class Result:
        async def run(self,*args):return copy.deepcopy(self.value)
    supervisor=Result();ns['sup']=supervisor
    try:
        supervisor.value={'projection_report':report,'_core_provenance':store.core_provenance}
        good=asyncio.run(ns['call'](TOOL,{'graph_id':gid}));assert not good.isError
        artifact=good.structuredContent['artifact']
        assert artifact['source_sha256']==store.graphs[gid]['source_sha256']
        assert artifact['sha256']==sha(canonical(report)) and artifact['byte_size']==len(canonical(report))
        assert artifact['uri']=='fibernet://artifacts/'+artifact['artifact_id']
        before=dict(store.artifacts);total=store.total
        for mode in ('counts_missing','counts_invalid','report_field_missing','provenance',
                     'missing_report','missing_error_category'):
            value=copy.deepcopy(report);identity=store.core_provenance
            if mode=='counts_missing':del value['counts']
            elif mode=='counts_invalid':value['counts']['fibers']='not a count'
            elif mode=='report_field_missing':del value['edges']
            elif mode=='provenance':identity={**identity,'canonical_core_manifest_sha256':'0'*64}
            supervisor.value={'projection_report':value,'_core_provenance':identity}
            if mode=='missing_report':supervisor.value={'_core_provenance':identity}
            if mode=='missing_error_category':supervisor.value={'error_code':'PROJECTION_LIMIT','_core_provenance':identity}
            failed=asyncio.run(ns['call'](TOOL,{'graph_id':gid}))
            assert failed.isError and 'artifact' not in failed.structuredContent
            if mode!='provenance':
                assert failed.structuredContent['error']['code']=='PROJECTION_OUTPUT_CONTRACT_FAILURE'
                assert failed.structuredContent['error']['category']=='internal'
            assert store.artifacts==before and store.total==total
        supervisor.value={'projection_report':report,'_core_provenance':store.core_provenance}
        assert asyncio.run(ns['call'](TOOL,{'graph_id':gid})).structuredContent['artifact']==good.structuredContent['artifact']
        files=set(store.files)
        for error in (ContractError('controlled storage refusal'),OSError('controlled storage failure'),server.ArtifactLimitError('controlled quota')):
            with patch.object(store,'stage_artifact',side_effect=error):
                failed=asyncio.run(ns['call'](TOOL,{'graph_id':gid}))
            assert failed.isError and 'artifact' not in failed.structuredContent
            if isinstance(error,server.ArtifactLimitError):
                assert failed.structuredContent['error']['code']=='PROJECTION_RESOURCE_LIMIT'
                assert failed.structuredContent['error']['category']=='resource'
            assert store.artifacts==before and store.total==total and store.files==files
    finally:store.close()


@pytest.mark.parametrize('mode',['timeout','cancel','kill'])
def test_real_stdio_inflight_cancellation_and_reaping(tmp_path,mode):
    params,output,_,_=configure(tmp_path,[(fixture_bytes('projection_native31.json'),'m','lab_xyz')],timeout=.001 if mode=='timeout' else 10)
    async def run():
        with anyio.fail_after(15):
            async with stdio_client(params) as (reader,writer):
                async with ClientSession(reader,writer) as session:
                    await session.initialize()
                    gid=(await session.call_tool('fibernet_catalog',{})).structuredContent['graphs'][0]['graph_id']
                    args={'graph_id':gid,'geometry_policy':'endpoint_chord'}
                    if mode=='timeout':
                        failed=await session.call_tool(TOOL,args)
                        assert failed.isError and failed.structuredContent['error']['code']=='WORKER_TIMEOUT'
                    else:
                        async with anyio.create_task_group() as group:
                            async def invoke():await session.call_tool(TOOL,args)
                            group.start_soon(invoke)
                            with anyio.fail_after(5):
                                while True:
                                    workers=[p for p in psutil.Process().children(recursive=True) if 'fibernet_mcp.worker' in p.cmdline()]
                                    if workers:break
                                    await anyio.sleep(.002)
                            if mode=='kill':os.kill(workers[0].pid,signal.SIGSTOP)
                            group.cancel_scope.cancel()
                        await anyio.sleep(.5)
                    assert not (await session.call_tool('fibernet_catalog',{})).isError
    anyio.run(run)
    audit=json.loads((output/'audit.json').read_text())
    assert audit['artifacts']==0 and audit['workers'] and all(w['reaped'] for w in audit['workers'])
    assert audit['workers'][0]['reason']==('timed_out' if mode=='timeout' else 'cancelled')
    if mode=='kill':assert audit['workers'][0]['returncode']==-9
    assert not list(output.glob('fibernet-mcp-*'))
    assert all(not Path('/proc',str(w['pid'])).exists() for w in audit['workers'])
    (tmp_path/'cancellation_scope.json').write_bytes(canonical({'operation':'actual report worker','mode':mode,'controlled_hold':mode=='kill','hold_method':'SIGSTOP of actual worker for hard-kill escalation only' if mode=='kill' else None,'scope':'MCP request cancellation and supervisor ownership; not numerical/physical cancellation'}))


def test_new_operation_request_budget_and_failed_launch_cleanup(tmp_path):
    _,output,_,_=configure(tmp_path,[(small(),'m','lab_xyz')])
    store=server.ManagedStore(str(tmp_path/'manifest.json'),str(output));supervisor=server.Supervisor(store)
    gid=next(iter(store.graphs));request=projection_request(INFO,{})
    try:
        with patch('fibernet_mcp.server.subprocess.Popen',side_effect=AssertionError('must reject before launch')):
            result=asyncio.run(supervisor.run('fiber_projection_report',gid,{'padding':'x'*4096}))
        assert result['error']['code']=='PROJECTION_REQUEST_LIMIT' and supervisor.active is None
        with patch('fibernet_mcp.server.subprocess.Popen',side_effect=OSError('controlled launch failure')):
            with pytest.raises(OSError):asyncio.run(supervisor.run('fiber_projection_report',gid,request))
        assert not list(Path(store.path).glob('request-*')) and not store.artifacts
        with patch('fibernet_mcp.projection.MAX_PROJECTION_REQUEST_BYTES',1):
            with pytest.raises(ProjectionContractError) as caught:projection_request(INFO,{})
        assert caught.value.code=='PROJECTION_REQUEST_LIMIT'
    finally:store.close()


def test_report_byte_budget_after_native_admission():
    request=projection_request(INFO,{})
    with patch('fibernet_mcp.projection.MAX_PROJECTION_REPORT_BYTES',1):
        result=project_report(small(),request)
    assert result=={'error_code':'PROJECTION_RESULT_LIMIT','error_category':'resource'}


def test_projection_artifact_session_quotas_and_existing_reuse(tmp_path):
    _,output,_,_=configure(tmp_path,[(small(),'m','lab_xyz')])
    store=server.ManagedStore(str(tmp_path/'manifest.json'),str(output))
    gid=next(iter(store.graphs));identity=store.core_provenance
    kind='fiber_network_projection_report';value={'synthetic_storage_control':0}
    assert server.MAX_SESSION==48*1024*1024
    try:
        first=store.artifact(gid,value,identity,kind=kind)
        before=dict(store.artifacts);files=set(store.files);total=store.total
        # Lower only the test quota to the bytes already stored. No48MiB
        # allocation is needed to exercise the identical production boundary.
        with patch('fibernet_mcp.server.MAX_SESSION',total):
            assert store.artifact(gid,value,identity,kind=kind)==first
            with pytest.raises(ContractError,match='session storage limit'):
                store.artifact(gid,{'synthetic_storage_control':1},identity,kind=kind)
        assert store.artifacts==before and store.files==files and store.total==total
        for i in range(1,64):store.artifact(gid,{'synthetic_storage_control':i},identity,kind=kind)
        assert len(store.artifacts)==64 and store.artifact(gid,value,identity,kind=kind)==first
        before=dict(store.artifacts);files=set(store.files);total=store.total
        with pytest.raises(ContractError,match='artifact limit'):
            store.artifact(gid,{'synthetic_storage_control':64},identity,kind=kind)
        assert store.artifacts==before and store.files==files and store.total==total
        assert store.info(first['artifact_id'])==first
    finally:store.close()


@pytest.mark.parametrize('mode',['oversize','malformed','provenance'])
def test_new_operation_completed_output_boundary_reaps_without_publication(tmp_path,mode):
    _,output,_,_=configure(tmp_path,[(small(),'m','lab_xyz')])
    store=server.ManagedStore(str(tmp_path/'manifest.json'),str(output));supervisor=server.Supervisor(store)
    gid=next(iter(store.graphs));request=projection_request(INFO,{});reaped=[]
    source=Path(store.path)/gid;source_bytes=source.read_bytes();source_hash=sha(source_bytes)
    class Completed:
        pid=999999;returncode=0
        def poll(self):return 0
        def wait(self):reaped.append(True);return 0
    def launch(command,**kwargs):
        if mode=='oversize':raw=b' '* (projection.MAX_PROJECTION_WORKER_BYTES+1)
        elif mode=='malformed':raw=b'{bad json'
        else:raw=canonical({'core_provenance':{},'result':{}})
        assert len(command)==10
        assert command[:5]==[sys.executable,'-B','-m','fibernet_mcp.worker','fiber_projection_report']
        assert Path(command[5])==source and command[7]==source_hash
        assert source.read_bytes()==source_bytes
        destination=Path(command[6])
        assert destination.parent==Path(store.path) and destination.name.startswith('worker-')
        assert destination not in (source,Path(command[8]))
        destination.write_bytes(raw)
        assert source.read_bytes()==source_bytes and sha(source.read_bytes())==source_hash
        return Completed()
    try:
        with patch('fibernet_mcp.server.subprocess.Popen',side_effect=launch):
            result=asyncio.run(supervisor.run('fiber_projection_report',gid,request))
        assert source.read_bytes()==source_bytes and sha(source.read_bytes())==source_hash
        assert result['error']['code']==('WORKER_PROVENANCE_MISMATCH' if mode=='provenance' else 'PROJECTION_WORKER_OUTPUT_REJECTED')
        assert result['error']['category']=='internal' and reaped==[True]
        assert supervisor.audit[0]['reason']=='failed' and supervisor.audit[0]['reaped']
        assert supervisor.active is None and not store.artifacts
        assert not list(Path(store.path).glob('worker-*')) and not list(Path(store.path).glob('request-*'))
    finally:store.close()


@pytest.mark.parametrize('mode',['source_digest','request_digest','request_symlink','request_oversize'])
def test_new_worker_rejects_tampered_or_oversized_input(tmp_path,mode):
    params,output,graphs,_=configure(tmp_path,[(small(),'m','lab_xyz')])
    source=tmp_path/'graph-0.json';request=tmp_path/'request.json'
    raw=canonical(projection_request(INFO,{}));request.write_bytes(raw);expected_request=sha(raw)
    expected_source=graphs[0]['sha256']
    if mode=='source_digest':source.write_bytes(source.read_bytes()+b' ')
    elif mode=='request_digest':request.write_bytes(raw+b' ')
    elif mode=='request_symlink':
        backing=tmp_path/'backing.json';request.rename(backing);request.symlink_to(backing)
    else:
        raw=b' '*4097;request.write_bytes(raw);expected_request=sha(raw)
    target=tmp_path/'result.json'
    result=subprocess.run([sys.executable,'-B','-m','fibernet_mcp.worker','fiber_projection_report',
        str(source),str(target),expected_source,str(request),expected_request],env=params.env,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=10)
    assert result.returncode!=0 and not target.exists()
    (tmp_path/'worker-refusal.json').write_bytes(canonical({'case':mode,'returncode':result.returncode,'output_published':False}))


@pytest.mark.parametrize('missing',[False,True])
def test_material_wire_refused_at_real_server_startup(tmp_path,missing):
    data=small()
    if missing:del data['edges'][0]['material']
    else:data['edges'][0]['material']=None
    params,output,_,_=configure(tmp_path,[(data,'m','lab_xyz')])
    result=subprocess.run([params.command,*params.args],env=params.env,input=b'',stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=10)
    assert result.returncode==2 and result.stdout==b'' and not list(output.glob('fibernet-mcp-*'))
