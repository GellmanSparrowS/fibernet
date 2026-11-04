"""Focused adapter checks and genuine SDK/native parity, not new physics validation."""
import asyncio
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from unittest.mock import patch

import anyio
import psutil
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from fibernet.core.structure_graph import StructureGraph
from fibernet.sim.linear_transport import solve_linear_transport, TransportSolveError
from fibernet_mcp.electrical import (
    TransportContractError, transport_request, solve_transport,
    MAX_TRANSPORT_RESULT_BYTES)
from fibernet_mcp.safety import canonical, sha, ContractError, strict_json
from fibernet_mcp.server import ManagedStore, Supervisor, TREE

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT.parents[1]


def network(rows=16, columns=16, parallel=94):
    """256 stable-ID nodes and 1,024 distinct physical edges, including parallels."""
    nodes = [{'id': 10000+13*i, 'position': [float(i % columns), float(i // columns), 0.]}
             for i in range(rows*columns)]
    pairs = []
    for r in range(rows):
        for c in range(columns):
            i = r*columns+c
            if c+1 < columns: pairs.append((i, i+1))
            if r+1 < rows: pairs.append((i, i+columns))
            if c+1 < columns and r+1 < rows:
                pairs.extend([(i, i+columns+1), (i+1, i+columns)])
    pairs += pairs[:parallel]
    edges = [{'id': 80000+17*i, 'node_i': nodes[a]['id'], 'node_j': nodes[b]['id']}
             for i, (a, b) in enumerate(pairs)]
    # Retain stable physical IDs through non-ID insertion order and orientations.
    edges = edges[::2][::-1]+edges[1::2]
    for i, edge in enumerate(edges):
        if i % 3 == 0: edge['node_i'], edge['node_j'] = edge['node_j'], edge['node_i']
    graph = StructureGraph.from_dict({'dimension': 3, 'nodes': nodes[::-1], 'edges': edges})
    data = graph.to_dict()
    # to_dict sorts IDs. Deliberately reorder the actual wire lists afterwards.
    data['nodes'] = data['nodes'][::-1]
    data['edges'] = data['edges'][::2][::-1]+data['edges'][1::2]
    assert [n['id'] for n in data['nodes']] != sorted(n['id'] for n in data['nodes'])
    assert [e['id'] for e in data['edges']] != sorted(e['id'] for e in data['edges'])
    request = {
        'conductance_S': [{'edge_id': edge['id'], 'value_S': 0.0 if i % 53 == 0 else .5+(i % 17)/13}
                         for i, edge in enumerate(data['edges'])],
        'dirichlet_voltage_V': [{'node_id': nodes[i]['id'], 'value_V': 1.5 if i < columns else -.25}
                               for i in list(range(columns))+list(range((rows-1)*columns, rows*columns))]}
    return data, request


def configuration(tmp_path, entries):
    graphs=[]
    for i,(data,unit) in enumerate(entries):
        path=tmp_path/('graph-'+str(i)+'.json'); path.write_bytes(canonical(data))
        graphs.append({'path':str(path),'sha256':sha(path.read_bytes()),'length_unit':unit,'frame_id':'test_frame'})
    manifest=tmp_path/'manifest.json'; manifest.write_bytes(canonical({'source_tree':TREE,'graphs':graphs}))
    output=tmp_path/'output';output.mkdir()
    return manifest,output


def environment():
    return {**os.environ, 'PYTHONPATH':os.pathsep.join([str(ROOT/'src'),str(CORE)]),
            'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
            'PYTHONDONTWRITEBYTECODE':'1'}


def parameters(manifest,output,timeout=10,audit='audit.json'):
    return StdioServerParameters(command=sys.executable,args=['-m','fibernet_mcp.server',
        '--manifest',str(manifest),'--output-root',str(output),'--timeout',str(timeout),
        '--audit',audit],env=environment())


@pytest.mark.parametrize('change,code', [
    (lambda r:r['conductance_S'].append(dict(r['conductance_S'][0])), 'DUPLICATE_CONDUCTANCE_ID'),
    (lambda r:r['conductance_S'][0].update(value_S=-1), 'INVALID_TRANSPORT_INPUT'),
    (lambda r:r.update(dirichlet_voltage_V=[]), 'INVALID_TRANSPORT_INPUT'),
])
def test_request_rejects_ambiguous_or_invalid_values(change,code):
    data,args=network(3,3,2);change(args)
    with pytest.raises(TransportContractError) as error:
        transport_request({'node_count':9,'edge_count':22,'length_unit':'m'},args)
    assert error.value.code==code


@pytest.mark.parametrize('field,value', [('edge_id',True),('edge_id',3.0),('edge_id',2**53),
                                         ('value_S',True),('value_S',float('nan')),('value_S',float('inf'))])
def test_numeric_contract(field,value):
    _,args=network(3,3,2);args['conductance_S'][0][field]=value
    with pytest.raises((ContractError,ValueError)):
        transport_request({'node_count':9,'edge_count':22,'length_unit':'m'},args)


@pytest.mark.parametrize('info,code', [
    ({'node_count':257,'edge_count':1,'length_unit':'m'},'TRANSPORT_GRAPH_LIMIT'),
    ({'node_count':2,'edge_count':1025,'length_unit':'m'},'TRANSPORT_GRAPH_LIMIT'),
    ({'node_count':2,'edge_count':1,'length_unit':'unspecified'},'DECLARED_LENGTH_UNIT_REQUIRED'),
])
def test_request_bounds(info,code):
    _,args=network(3,3,2)
    with pytest.raises(TransportContractError) as error:transport_request(info,args)
    assert error.value.code==code


def test_uses_native_api_and_result_without_physics_copy():
    data,args=network(3,3,2)
    request=transport_request({'node_count':9,'edge_count':22,'length_unit':'mm'},args)
    expected=solve_linear_transport(StructureGraph.from_dict(data),
        {r['edge_id']:r['value_S'] for r in args['conductance_S']},
        [(r['node_id'],r['value_V']) for r in args['dirichlet_voltage_V']],length_unit='mm').to_dict()
    with patch('fibernet.sim.linear_transport.solve_linear_transport',wraps=solve_linear_transport) as native:
        actual=solve_transport(data,request)
    assert native.call_count==1 and actual=={'electrical':expected}
    assert expected['provenance']['units']=={'conductance':'S','voltage':'V','current':'A','power':'W'}
    assert expected['provenance']['geometry_conversion']=='none'


def test_bounded_output_does_not_publish_partial_result():
    data,args=network(3,3,2);args['length_unit']='m'
    with patch('fibernet_mcp.electrical.MAX_TRANSPORT_RESULT_BYTES',1):
        assert solve_transport(data,args)=={'error_code':'TRANSPORT_RESULT_LIMIT','error_category':'resource'}


def test_request_file_cleaned_on_failed_launch(tmp_path):
    data,args=network(3,3,2);manifest,output=configuration(tmp_path,[(data,'m')])
    store=ManagedStore(str(manifest),str(output));supervisor=Supervisor(store)
    try:
        with patch('fibernet_mcp.server.subprocess.Popen',side_effect=OSError('injected launch failure')):
            with pytest.raises(OSError):asyncio.run(supervisor.run('linear_transport',next(iter(store.graphs)),args))
        assert not list(Path(store.path).glob('request-*')) and supervisor.active is None
    finally:store.close()
    assert not list(output.iterdir())


def test_request_byte_budget():
    _,args=network(3,3,2)
    with patch('fibernet_mcp.electrical.MAX_TRANSPORT_REQUEST_BYTES',1):
        with pytest.raises(TransportContractError) as error:
            transport_request({'node_count':9,'edge_count':22,'length_unit':'m'},args)
    assert error.value.code=='TRANSPORT_REQUEST_LIMIT'


@pytest.mark.parametrize('tamper',['digest','symlink'])
def test_worker_refuses_tampered_request_without_output(tmp_path,tamper):
    data,args=network(3,3,2);args['length_unit']='m'
    source=tmp_path/'graph.json';source.write_bytes(canonical(data))
    request=tmp_path/'request.json';raw=canonical(args);request.write_bytes(raw)
    output=tmp_path/'worker-result.json'
    digest=sha(raw)
    if tamper=='digest':request.write_bytes(raw+b' ')
    else:
        backing=tmp_path/'backing.json';request.rename(backing);request.symlink_to(backing)
    result=subprocess.run([sys.executable,'-m','fibernet_mcp.worker','linear_transport',
        str(source),str(output),sha(source.read_bytes()),str(request),digest],
        env=environment(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=10)
    (tmp_path/'worker_stderr.log').write_bytes(result.stderr)
    assert result.returncode!=0 and not output.exists()
    if tamper=='digest':assert b'request integrity mismatch' in result.stderr


async def read_artifact(session,artifact):
    assert artifact['kind']=='linear_transport_result'
    manifest=await session.read_resource(artifact['uri'])
    assert json.loads(manifest.contents[0].text)==artifact
    info=await session.call_tool('fibernet_artifact_info',{'artifact_id':artifact['artifact_id']})
    assert not info.isError and info.structuredContent['artifact']==artifact
    chunks=[];offset=0
    while True:
        part=await session.call_tool('fibernet_artifact_read',{'artifact_id':artifact['artifact_id'],'offset':offset,'length':32768})
        assert not part.isError
        value=part.structuredContent;chunk=base64.b64decode(value['data']);chunks.append(chunk)
        assert len(chunk)<=32768
        offset=value['next_offset']
        if value['eof']:break
    raw=b''.join(chunks)
    assert sha(raw)==artifact['sha256'] and len(raw)==artifact['byte_size']
    assert len(raw)<=MAX_TRANSPORT_RESULT_BYTES
    return json.loads(raw),len(chunks)


def test_real_sdk_near_cap_native_parity_and_invalid_inputs(tmp_path):
    data,args=network();assert (len(data['nodes']),len(data['edges']))==(256,1024)
    assert [n['id'] for n in data['nodes']] != sorted(n['id'] for n in data['nodes'])
    assert [e['id'] for e in data['edges']] != sorted(e['id'] for e in data['edges'])
    assert len({tuple(sorted((e['node_i'],e['node_j']))) for e in data['edges']})<1024
    oversized=copy.deepcopy(data);oversized['nodes'].append({'id':999999,'position':[99.,99.,0.]})
    oversized['next_node_id']=1000000
    tiny,tiny_args=network(2,2,0)
    manifest,output=configuration(tmp_path,[(data,'mm'),(data,'unspecified'),(oversized,'mm'),(tiny,'m')])
    before=canonical(data)
    native=solve_linear_transport(StructureGraph.from_dict(data),
        {v['edge_id']:v['value_S'] for v in args['conductance_S']},
        [(v['node_id'],v['value_V']) for v in args['dirichlet_voltage_V']],length_unit='mm').to_dict()
    report={'node_count':256,'edge_count':1024,'wire_node_ids_unsorted':True,'wire_edge_ids_unsorted':True,'physics_scope':'native-result parity only; no new physical qualification'}
    async def run():
        with anyio.fail_after(60):
            async with stdio_client(parameters(manifest,output)) as (reader,writer):
                async with ClientSession(reader,writer) as session:
                    assert (await session.initialize()).protocolVersion=='2025-11-25'
                    tools=await session.list_tools();assert len(tools.tools)==9
                    definition=next(t for t in tools.tools if t.name=='fibernet_solve_linear_transport')
                    assert definition.execution.taskSupport=='forbidden'
                    cap=await session.read_resource('fibernet://capabilities')
                    assert json.loads(cap.contents[0].text)['physics']==['steady_linear_electrical_conductance','prescribed_temperature_linear_planar_welded_snapshot']
                    catalog=(await session.call_tool('fibernet_catalog',{})).structuredContent
                    graphs=catalog['graphs'];gid=graphs[0]['graph_id']
                    call_args={'graph_id':gid,**args}
                    response=await session.call_tool('fibernet_solve_linear_transport',call_args)
                    assert not response.isError,response
                    result,pages=await read_artifact(session,response.structuredContent['artifact'])
                    assert result==native
                    assert result['node_ids']==sorted(n['id'] for n in data['nodes'])
                    assert result['edge_ids']==[e['id'] for e in data['edges']]
                    assert result['endpoints_by_id']==[[e['node_i'],e['node_j']] for e in data['edges']]
                    assert pages>1
                    (tmp_path/'native_result.json').write_bytes(canonical(native))
                    (tmp_path/'mcp_result.json').write_bytes(canonical(result))
                    again=await session.call_tool('fibernet_solve_linear_transport',call_args)
                    assert not again.isError and again.structuredContent['artifact']==response.structuredContent['artifact']
                    same_boundary=copy.deepcopy(call_args)
                    same_boundary['dirichlet_voltage_V'].append(dict(same_boundary['dirichlet_voltage_V'][0]))
                    repeated=await session.call_tool('fibernet_solve_linear_transport',same_boundary)
                    assert not repeated.isError and repeated.structuredContent['artifact']==response.structuredContent['artifact']
                    changed=copy.deepcopy(call_args);changed['dirichlet_voltage_V'][0]['value_V']=2.5
                    altered=await session.call_tool('fibernet_solve_linear_transport',changed)
                    assert not altered.isError and altered.structuredContent['artifact']!=response.structuredContent['artifact']
                    altered_result,_=await read_artifact(session,altered.structuredContent['artifact'])
                    altered_native=solve_linear_transport(StructureGraph.from_dict(data),
                        {v['edge_id']:v['value_S'] for v in changed['conductance_S']},
                        [(v['node_id'],v['value_V']) for v in changed['dirichlet_voltage_V']],length_unit='mm').to_dict()
                    assert altered_result==altered_native
                    report.update(native_all_fields_exact=True,changed_input_all_fields_exact=True,
                                  identical_boundary_duplicates_coalesce=True,paged_reads=pages,artifact=response.structuredContent['artifact'])
                    cases=[]
                    def case(label,arguments,code):cases.append((label,arguments,code))
                    duplicate=copy.deepcopy(call_args);duplicate['conductance_S'][1]=dict(duplicate['conductance_S'][0])
                    case('duplicate edge',duplicate,'DUPLICATE_CONDUCTANCE_ID')
                    missing=copy.deepcopy(call_args);missing['conductance_S'].pop()
                    case('missing edge',missing,'INVALID_TRANSPORT_INPUT')
                    unknown=copy.deepcopy(call_args);unknown['conductance_S'][0]['edge_id']=999999
                    case('unknown edge',unknown,'INVALID_TRANSPORT_INPUT')
                    conflicting=copy.deepcopy(call_args);conflicting['dirichlet_voltage_V'].append({'node_id':args['dirichlet_voltage_V'][0]['node_id'],'value_V':77.})
                    case('conflicting voltage',conflicting,'INVALID_TRANSPORT_INPUT')
                    unknown_node=copy.deepcopy(call_args);unknown_node['dirichlet_voltage_V'][0]['node_id']=999999
                    case('unknown node',unknown_node,'INVALID_TRANSPORT_INPUT')
                    floating=copy.deepcopy(call_args)
                    for value in floating['conductance_S']:value['value_S']=0.
                    case('unanchored zero-conductance components',floating,'FLOATING_COMPONENT')
                    case('unit unspecified',{**call_args,'graph_id':graphs[1]['graph_id']},'DECLARED_LENGTH_UNIT_REQUIRED')
                    case('node limit',{**call_args,'graph_id':graphs[2]['graph_id']},'TRANSPORT_GRAPH_LIMIT')
                    case('unknown handle',{**call_args,'graph_id':'0'*32},'INVALID_ARGUMENT_OR_HANDLE')
                    case('extra option',{**call_args,'rtol':1},'INVALID_ARGUMENT_OR_HANDLE')
                    boolean=copy.deepcopy(call_args);boolean['conductance_S'][0]['value_S']=True
                    case('bool conductance',boolean,'INVALID_ARGUMENT_OR_HANDLE')
                    negative=copy.deepcopy(call_args);negative['conductance_S'][0]['value_S']=-1
                    case('negative conductance',negative,'INVALID_ARGUMENT_OR_HANDLE')
                    overflow={**copy.deepcopy(tiny_args),'graph_id':graphs[3]['graph_id']}
                    for item in overflow['conductance_S']:item['value_S']=1e308
                    for item in overflow['dirichlet_voltage_V']:item['value_V']*=1e308
                    with pytest.raises(TransportSolveError) as native_failure:
                        solve_linear_transport(StructureGraph.from_dict(tiny),
                            {v['edge_id']:v['value_S'] for v in overflow['conductance_S']},
                            [(v['node_id'],v['value_V']) for v in overflow['dirichlet_voltage_V']],length_unit='m')
                    report['native_failure']={'type':type(native_failure.value).__name__,'message':str(native_failure.value)}
                    case('native numerical failure',overflow,'TRANSPORT_SOLVE_FAILED')
                    failures=[]
                    for label,arguments,code in cases:
                        failed=await session.call_tool('fibernet_solve_linear_transport',arguments)
                        assert failed.isError and failed.structuredContent['error']['code']==code,(label,failed)
                        assert 'artifact' not in failed.structuredContent
                        assert str(tmp_path) not in json.dumps(failed.structuredContent)
                        failures.append({'case':label,'response':failed.structuredContent})
                    (tmp_path/'failed_outcomes.json').write_bytes(canonical(failures))
                    report['invalid_cases']=len(failures)
                    assert not (await session.call_tool('fibernet_catalog',{})).isError
    anyio.run(run)
    assert canonical(data)==before
    audit=json.loads((output/'audit.json').read_text())
    assert all(w['reaped'] for w in audit['workers']) and audit['artifacts']==2
    assert not list(output.glob('fibernet-mcp-*'))
    report['audit']=audit;(tmp_path/'report.json').write_text(json.dumps(report,indent=2))


@pytest.mark.parametrize('mode',['timeout','cancel','kill'])
def test_real_sdk_transport_timeout_cancellation_and_reaping(tmp_path,mode):
    data,args=network();manifest,output=configuration(tmp_path,[(data,'m')])
    async def run():
        with anyio.fail_after(15):
            async with stdio_client(parameters(manifest,output,.001 if mode=='timeout' else 10)) as (reader,writer):
                async with ClientSession(reader,writer) as session:
                    await session.initialize()
                    gid=(await session.call_tool('fibernet_catalog',{})).structuredContent['graphs'][0]['graph_id']
                    if mode=='timeout':
                        result=await session.call_tool('fibernet_solve_linear_transport',{'graph_id':gid,**args})
                        assert result.isError and result.structuredContent['error']['code']=='WORKER_TIMEOUT'
                    else:
                        async with anyio.create_task_group() as group:
                            async def invoke():await session.call_tool('fibernet_solve_linear_transport',{'graph_id':gid,**args})
                            group.start_soon(invoke)
                            with anyio.fail_after(5):
                                while True:
                                    workers=[p for p in psutil.Process().children(recursive=True) if 'fibernet_mcp.worker' in p.cmdline()]
                                    if workers:break
                                    await anyio.sleep(.005)
                            if mode=='kill':os.kill(workers[0].pid,signal.SIGSTOP)
                            group.cancel_scope.cancel()
                        await anyio.sleep(.5)
                    assert not (await session.call_tool('fibernet_catalog',{})).isError
    anyio.run(run)
    audit=json.loads((output/'audit.json').read_text());assert audit['artifacts']==0
    assert audit['workers'] and all(w['reaped'] for w in audit['workers'])
    assert audit['workers'][0]['reason']==('timed_out' if mode=='timeout' else 'cancelled')
    if mode=='kill':assert audit['workers'][0]['returncode']==-9
    assert not list(output.glob('fibernet-mcp-*'))
    assert all(not Path('/proc',str(w['pid'])).exists() for w in audit['workers'])


def test_physical_number_normalization_is_field_scoped():
    message={'jsonrpc':'2.0','id':1,'method':'tools/call','params':{
        'name':'fibernet_solve_linear_transport','arguments':{
            'conductance_S':[{'edge_id':2**53-1,'value_S':10**18}],
            'dirichlet_voltage_V':[{'node_id':2**53-1,'value_V':10**18}]}}}
    raw=canonical(message)
    with pytest.raises(ContractError):strict_json(raw)
    decoded=strict_json(raw,electrical_rpc=True)
    arguments=decoded['params']['arguments']
    assert type(arguments['conductance_S'][0]['value_S']) is float
    assert arguments['conductance_S'][0]['edge_id']==2**53-1
    for field in ('id','extra'):
        bad=copy.deepcopy(message);bad[field]=2**53
        with pytest.raises(ContractError):strict_json(canonical(bad),electrical_rpc=True)
    for field,identifier in [('conductance_S','edge_id'),('dirichlet_voltage_V','node_id')]:
        bad=copy.deepcopy(message);bad['params']['arguments'][field][0][identifier]=2**53
        with pytest.raises(ContractError):strict_json(canonical(bad),electrical_rpc=True)
    bad=copy.deepcopy(message);bad['params']['name']='fibernet_validate'
    with pytest.raises(ContractError):strict_json(canonical(bad),electrical_rpc=True)
    bad=copy.deepcopy(message);bad['params']['arguments']['conductance_S'][0]['extra']=2**53
    with pytest.raises(ContractError):strict_json(canonical(bad),electrical_rpc=True)
    with pytest.raises(ContractError):strict_json(raw.replace(b'1000000000000000000',b'NaN'),electrical_rpc=True)
    bad=copy.deepcopy(message);bad['params']['arguments']['frame']='x'*4097
    with pytest.raises(ContractError):strict_json(canonical(bad),electrical_rpc=True)


def test_real_sdk_integer_and_exponent_physical_values_share_session(tmp_path):
    data=StructureGraph.from_dict({'dimension':3,'nodes':[{'id':10,'position':[0.,0.,0.]},
        {'id':90,'position':[1.,0.,0.]}],'edges':[{'id':205,'node_i':10,'node_j':90}]}).to_dict()
    manifest,output=configuration(tmp_path,[(data,'m')])
    evidence=[]
    async def run():
        async with stdio_client(parameters(manifest,output)) as (reader,writer):
            async with ClientSession(reader,writer) as session:
                await session.initialize()
                gid=(await session.call_tool('fibernet_catalog',{})).structuredContent['graphs'][0]['graph_id']
                first_artifact=None
                for conductance,voltage in [(1e18,1.),(10**18,1),(1.,1e18),(1,10**18)]:
                    request={'graph_id':gid,'conductance_S':[{'edge_id':205,'value_S':conductance}],
                        'dirichlet_voltage_V':[{'node_id':10,'value_V':voltage},{'node_id':90,'value_V':0}]}
                    response=await session.call_tool('fibernet_solve_linear_transport',request)
                    assert not response.isError,response
                    actual,_=await read_artifact(session,response.structuredContent['artifact'])
                    expected=solve_linear_transport(StructureGraph.from_dict(data),{205:conductance},
                        [(10,voltage),(90,0)],length_unit='m').to_dict()
                    assert actual==expected
                    artifact=response.structuredContent['artifact']
                    if len(evidence)%2:assert artifact==evidence[-1]['artifact']
                    evidence.append({'conductance':conductance,'voltage':voltage,'artifact':artifact})
                    first_artifact=first_artifact or artifact
                for field,scalar in [('conductance_S','value_S'),('dirichlet_voltage_V','value_V')]:
                    request={'graph_id':gid,'conductance_S':[{'edge_id':205,'value_S':1}],
                        'dirichlet_voltage_V':[{'node_id':10,'value_V':1},{'node_id':90,'value_V':0}]}
                    request[field][0][scalar]=10**400
                    failed=await session.call_tool('fibernet_solve_linear_transport',request)
                    assert failed.isError and failed.structuredContent['error']['code']=='INVALID_ARGUMENT_OR_HANDLE'
                    evidence.append({'unrepresentable_field':field,'response':failed.structuredContent})
                    assert not (await session.call_tool('fibernet_catalog',{})).isError
                    await read_artifact(session,first_artifact)
    anyio.run(run)
    (tmp_path/'numeric_wire_evidence.json').write_bytes(canonical(evidence))
    assert not list(output.glob('fibernet-mcp-*'))
