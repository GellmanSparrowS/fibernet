"""Real MCP SDK client acceptance, no direct server handler calls."""
import asyncio,base64,hashlib,json,os,sys,time,uuid,signal
import psutil
from pathlib import Path
import anyio
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parent
CORE=os.environ.get('FIBERNET_CORE_PATH')
(ROOT/'evidence').mkdir(exist_ok=True)
from fibernet_mcp.server import TREE
configuration={'source_tree':TREE,'graphs':[{'path':str(ROOT/'fixtures/voronoi_4340.json'),'sha256':hashlib.sha256((ROOT/'fixtures/voronoi_4340.json').read_bytes()).hexdigest(),'length_unit':'unspecified','frame_id':'unknown'}]}
(ROOT/'evidence/manifest.json').write_text(json.dumps(configuration,indent=2))
RUN=ROOT/'evidence'/('run-'+uuid.uuid4().hex[:8]);RUN.mkdir()
ENV={**os.environ,'PYTHONPATH':os.pathsep.join([str(ROOT/'src')]+([str(Path(CORE).resolve())] if CORE else [])+
                                       ([os.environ['PYTHONPATH']] if os.environ.get('PYTHONPATH') else [])),
 'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MPLCONFIGDIR':str(ROOT/'evidence/mpl'),
 'XDG_CACHE_HOME':str(ROOT/'evidence/cache'),'PYTHONDONTWRITEBYTECODE':'1'}



def expected_core_provenance(manifest):
    raw = json.dumps(manifest, sort_keys=True, separators=(',', ':'),
                     ensure_ascii=True).encode('utf-8')
    return {'schema_version': 'fibernet.core-provenance/1',
            'canonical_core_manifest_sha256': hashlib.sha256(raw).hexdigest(),
            'source_file_count': len(manifest), 'scope': 'fibernet_python_sources'}

async def run(timeout=10,audit='acceptance.json',cancel=False,force_kill=False):
    p=StdioServerParameters(command=sys.executable,args=['-m','fibernet_mcp.server','--manifest',str(ROOT/'evidence/manifest.json'),'--output-root',str(RUN),'--timeout',str(timeout),'--audit',audit],env=ENV)
    records=[]
    async with stdio_client(p) as (r,w):
        async with ClientSession(r,w) as session:
            init=await session.initialize();assert init.protocolVersion=='2025-11-25'
            listing=await session.list_tools();assert len(listing.tools)==8
            assert len((await session.list_resource_templates()).resourceTemplates)==1
            await session.read_resource('fibernet://capabilities')
            expected = expected_core_provenance(json.loads((ROOT/'src/fibernet_mcp/core_manifest.json').read_text()))
            resources = await session.list_resources()
            assert {str(item.uri) for item in resources.resources} == {'fibernet://capabilities', 'fibernet://provenance/1'}
            provenance = await session.read_resource('fibernet://provenance/1')
            assert len(provenance.contents) == 1
            assert json.loads(provenance.contents[0].text) == expected
            cat=(await session.call_tool('fibernet_catalog',{})).structuredContent
            g=cat['graphs'][0];assert g['node_count']==4340 and g['edge_count']==5073
            if cancel:
                # Cancel a real requested topology process, not a fake sleep tool.
                async with anyio.create_task_group() as group:
                    async def invoke():
                        await session.call_tool('fibernet_analyze_topology',{'graph_id':g['graph_id']})
                    group.start_soon(invoke)
                    deadline=time.monotonic()+3
                    while True:
                        workers=[p for p in psutil.Process().children(recursive=True) if 'fibernet_mcp.worker' in p.cmdline()]
                        if workers: break
                        assert time.monotonic()<deadline
                        await anyio.sleep(.01)
                    if force_kill: os.kill(workers[0].pid,signal.SIGSTOP)
                    await anyio.sleep(.1)
                    group.cancel_scope.cancel()
                await anyio.sleep(.5)
                assert not (await session.call_tool('fibernet_catalog',{})).isError
                return {'cancel_client_remained_responsive':True}
            valid=await session.call_tool('fibernet_validate',{'graph_id':g['graph_id']})
            if timeout<1:
                assert valid.isError and valid.structuredContent['error']['code']=='WORKER_TIMEOUT'
                return {'timeout_verified':True}
            assert not valid.isError,valid
            assert valid.structuredContent['validation_status']=='valid'
            a=await session.call_tool('fibernet_analyze_topology',{'graph_id':g['graph_id']});assert not a.isError,a
            data=a.structuredContent;assert data['metrics']['cycle_rank_physical']==734
            aid=data['artifact']['artifact_id']
            assert (await session.call_tool('fibernet_artifact_info',{'artifact_id':aid})).structuredContent['artifact']==data['artifact']
            resource=await session.read_resource(data['artifact']['uri']);assert json.loads(resource.contents[0].text)==data['artifact']
            raw=b'';offset=0
            while True:
                r=await session.call_tool('fibernet_artifact_read',{'artifact_id':aid,'offset':offset,'length':32768});assert not r.isError,r
                d=r.structuredContent;raw+=base64.b64decode(d['data']);offset=d['next_offset']
                if d['eof']:break
            assert hashlib.sha256(raw).hexdigest()==data['artifact']['sha256']
            # Independent native reference, every scalar and full ID-aligned vector.
            from fibernet.core.structure_graph import StructureGraph
            from fibernet.analysis.topology_metrics import analyze_topology
            from dataclasses import asdict
            expected=asdict(analyze_topology(StructureGraph.load_json(str(ROOT/'fixtures/voronoi_4340.json'))))
            assert json.loads(raw)==json.loads(json.dumps(expected))
            again=(await session.call_tool('fibernet_analyze_topology',{'graph_id':g['graph_id']})).structuredContent
            assert again['artifact']==data['artifact']
            for name,args in [('fibernet_validate',{'graph_id':'../outside'}),('fibernet_validate',{'graph_id':True}),('fibernet_validate',{'graph_id':'0'*32}),('fibernet_validate',{'graph_id':g['graph_id'],'extra':1}),('fibernet_artifact_read',{'artifact_id':aid,'offset':-1}),('fibernet_artifact_read',{'artifact_id':aid,'offset':0,'length':32769}),('fibernet_artifact_read',{'artifact_id':aid,'offset':len(raw)+1})]:
                assert (await session.call_tool(name,args)).isError
            for uri in ['file:///etc/passwd','https://example.com','fibernet://artifacts/../../etc/passwd','fibernet://artifacts/'+'0'*32]:
                try:await session.read_resource(uri)
                except Exception:pass
                else:raise AssertionError('resource escape accepted')
            # Filesystem adversary cannot turn managed handles into external reads.
            session_dir=next(RUN.glob('fibernet-mcp-*'))
            artifact_path=session_dir/aid
            artifact_path.unlink();artifact_path.symlink_to(ROOT/'fixtures/voronoi_4340.json')
            assert (await session.call_tool('fibernet_artifact_read',{'artifact_id':aid,'offset':0})).isError
            artifact_path.unlink();artifact_path.write_bytes(raw)
            source_path=session_dir/g['graph_id'];source_bytes=source_path.read_bytes()
            source_path.write_bytes(source_bytes+b' ')
            assert (await session.call_tool('fibernet_validate',{'graph_id':g['graph_id']})).isError
            source_path.write_bytes(source_bytes)
            return {'node_count':4340,'edge_count':5073,'metrics':data['metrics'],'artifact_bytes':len(raw),'artifact_sha256':data['artifact']['sha256'],'source_sha256':g['source_sha256'],'negative_tool_cases':7,'negative_resource_cases':4,'all_native_fields_equal':True}

async def main():
    report={}
    for label,kwargs in [('roundtrip',{}),('timeout',{'timeout':.001,'audit':'timeout.json'}),('cancel',{'cancel':True,'audit':'cancel.json'}),('kill',{'cancel':True,'force_kill':True,'audit':'kill.json'})]:
        report[label]=await run(**kwargs)
    # Every launched worker must have been reaped; no late publication on cancellation.
    for file in ['acceptance.json','timeout.json','cancel.json','kill.json']:
        evidence=json.loads((RUN/file).read_text())
        assert evidence['workers'] and all(w['reaped'] for w in evidence['workers'])
        for worker in evidence['workers']:
            assert not Path('/proc',str(worker['pid'])).exists()
        report[file]=evidence
    assert report['kill.json']['workers'][0]['returncode']==-9
    assert report['cancel.json']['workers'][0]['reason']=='cancelled'
    assert report['cancel.json']['artifacts']==0 and report['timeout.json']['artifacts']==0
    report['run_directory']=str(RUN)
    (ROOT/'evidence'/('acceptance-report-'+RUN.name+'.json')).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':anyio.run(main)
