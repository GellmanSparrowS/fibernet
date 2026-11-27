"""Real SDK test for tilted parallel members and two domain failures."""
import asyncio,base64,json,math,os,sys,tempfile
from pathlib import Path
from fibernet.core.structure_graph import StructureGraph
from fibernet_mcp.safety import canonical,sha
from fibernet_mcp.server import TREE
from fibernet_mcp.translated_transport import TOOL, candidate, parent_request, parent_result
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client

def graph(extra=False):
    g=StructureGraph(dimension=3,box_size=(2,2,2))
    g.metadata['coordinate_frame']='lab_xyz'
    a,b,c=[g.add_node(q) for q in ((0,0,0),(1,0,0),(1,1,0))]
    for n,m in [(a,b),(a,b),(b,c),(a,c)]:g.add_edge(n,m,allow_parallel=True)
    if extra:g.add_node((0,0,1.5))
    return g

async def run(root):
    graphs=[graph(),graph(True),graph()]
    graphs[2].metadata.clear()
    graphs[2].metadata.update(geometry={'coordinate_frame':'other_frame'})
    entries=[]
    for i,g in enumerate(graphs):
        f=root/f'graph-{i}.json'; data=canonical(g.to_dict());f.write_bytes(data)
        entries.append(dict(path=str(f),sha256=sha(data),length_unit='m',frame_id='lab_xyz'))
    m=root/'manifest.json';m.write_bytes(canonical(dict(graphs=entries,source_tree=TREE)))
    p=StdioServerParameters(command=sys.executable,args=['-B','-m','fibernet_mcp.server','--manifest',str(m),
        '--output-root',str(root),'--audit','audit.json','--enable-translated-transport-3d'],env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
    async with stdio_client(p) as (r,w), ClientSession(r,w) as s:
        await s.initialize()
        cat=(await s.call_tool('fibernet_catalog',{})).structuredContent
        gold={}
        for i,(g, meta) in enumerate(zip(graphs,cat['graphs'])):
            ids=list(g.nodes); edges=list(g.edges)
            args=dict(graph_id=meta['graph_id'],frame_id='lab_xyz',length_unit='m', translations_m=[
            dict(node_id=j,value_m=[0,0,.5] if index==1 else [0,0,0]) for index,j in enumerate(ids)],
            reference_conductance_S=[dict(edge_id=j,value_S=k) for j,k in zip(edges,[1,3,5,2])],
            dirichlet_voltage_V=[dict(node_id=ids[0],value_V=0),dict(node_id=ids[2],value_V=1)])
            ok=await s.call_tool(TOOL,args)
            if i==2:
                assert ok.isError,ok
                e=ok.structuredContent['error']
                assert e['code']=='TRANSLATED_TRANSPORT_FRAME_MISMATCH' and e['category']=='contract',e
                gold['nested_frame_mismatch']=e['code']
                continue
            if i==1:
                assert ok.isError,ok
                e=ok.structuredContent['error'];assert e['code']=='TRANSLATED_TRANSPORT_FLOATING_COMPONENT' and e['category']=='domain',e
                gold['floating']=e['code'];continue
            assert not ok.isError,ok
            a=ok.structuredContent['artifact'];f=await s.call_tool('fibernet_artifact_read',{'artifact_id':a['artifact_id'],'offset':0})
            b=base64.b64decode(f.structuredContent['data'],validate=True);assert f.structuredContent['eof'] and sha(b)==a['sha256']
            tr=json.loads(b)['transport']
            assert set(tr['edge_ids'])==set(edges) and len(tr['edge_ids'])==4
            got=tr['dissipated_power_W'];expected=2+20/(9*math.sqrt(1.25))
            assert abs(got-expected)<1e-12,(got,expected)
            gold.update(analytic_effective_S=expected,native_effective_S=got)
            collapsed=args|{'translations_m':[dict(node_id=j,value_m=[-1,0,0] if index==1 else [0,0,0]) for index,j in enumerate(ids)]}
            fail=await s.call_tool(TOOL,collapsed);assert fail.isError
            e=fail.structuredContent['error'];assert e['code']=='TRANSLATED_TRANSPORT_INVALID_GEOMETRY' and e['category']=='domain',e
            gold['collapse']=e['code']
    # Bind the verifier to the staged source/controls, not to equal but empty
    # worker-originated arrays. Test valid request first to rule out a vacuous check.
    g=graphs[0]; source=canonical(g.to_dict()); meta=cat['graphs'][0]
    gid=meta['graph_id']; ids=list(g.nodes); edges=list(g.edges)
    args=dict(graph_id=gid,frame_id='lab_xyz',length_unit='m',
        translations_m=[dict(node_id=n,value_m=[0,0,0]) for n in ids],
        reference_conductance_S=[dict(edge_id=e,value_S=1) for e in edges],
        dirichlet_voltage_V=[dict(node_id=ids[0],value_V=0), dict(node_id=ids[-1],value_V=1)])
    request=parent_request(gid,meta,source,args)
    controls=canonical(request['controls'])
    artifact=json.loads(candidate(source,controls,source_digest=sha(source),request_digest=sha(controls),
                                  manifest_unit='m',manifest_frame='lab_xyz'))
    core={'mock_core_binding_for_verifier':'not a protocol fixture'}
    envelope=dict(source_sha256=sha(source),worker_request_sha256=sha(canonical(request)),
                  _core_provenance=core,translated_transport=artifact)
    assert parent_result(envelope,meta,source,request,core)==artifact
    from copy import deepcopy
    from fibernet_mcp.safety import ContractError
    mutations=0
    for field in ('node_ids','edge_ids','endpoints_by_id','conductance_S','translations_m','reference_conductance_S','input_sha256'):
        bad=deepcopy(envelope)
        result=bad['translated_transport']
        result['preparation'][field]=[] if field!='input_sha256' else '0'*64
        if field in result['transport']: result['transport'][field]=[]
        try: parent_result(bad,meta,source,request,core)
        except ContractError: mutations+=1
        else: raise AssertionError(field)
    gold['corrupted_bindings_refused']=mutations
    audit=json.loads((root/'audit.json').read_text())
    assert audit['artifacts']==1 and len(audit['workers'])==3 and all(x['reaped'] for x in audit['workers'])
    return gold|{'artifact_count':audit['artifacts']}
async def main():
    with tempfile.TemporaryDirectory() as name: print(json.dumps(await run(Path(name)),indent=2))
if __name__ == "__main__":
    asyncio.run(main())
