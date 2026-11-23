"""Real SDK local mixed registry parity for two bundled Tet4 fixtures."""
import base64
from collections import Counter
from hashlib import sha256
from itertools import combinations
import json,os,sys,tempfile
from pathlib import Path
import anyio,numpy as np
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client
from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim.tet4_dc_thermal import solve_dc_thermal_tet4_with_convection
from fibernet_mcp.server import TREE
from fibernet_mcp.tet4_contract import canonical,load_fixture,parse_req,TET4_TOOL
ADAPTER=Path(__file__).resolve().parents[1]

def cfg(enabled):
    p=ADAPTER/'fixtures/voronoi_4340.json'
    manifest=dict(source_tree=TREE,graphs=[dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest(),length_unit='unspecified',frame_id='unknown')])
    if enabled: manifest['tet4_fixtures']=['simple','welded']
    return manifest

def fixture_case(name,mid,sigma=.5,mixed=False):
    raw,digest,mesh=load_fixture(name)
    p,t,n,c,tag,mat,u=mesh
    vol=Tet4Bundle(np.array(p,float),np.array(t),np.array(n,np.uint64),np.array(c,np.uint64),np.array(tag,np.uint64),mat,u)
    tags=set(map(str,mat))
    if name=='welded':
        cs=Counter(tuple(sorted(f)) for cell in t for f in combinations(cell,3))
        l=[i for i,q in enumerate(p) if q[0]==-1.5]; r=[i for i,q in enumerate(p) if q[0]==1.5]
        side=[f for f,k in sorted(cs.items()) if k==1 and not abs(sum(p[i][0] for i in f))==4.5]
        assert len(side)==912
        ambient=300.0528 if mixed else 299.95
    else:
        l,r,side=[0],[1],[(0,2,3)];ambient=299.95
    controls=dict(mesh_id=mid,conductivity_S_per_m_by_material={k:(sigma if k=='17' else 2.) for k in tags},
                  conductivity_W_per_m_K_by_material=dict.fromkeys(tags,.25),
                  electrode_voltage_V_by_id={**dict.fromkeys([str(n[i]) for i in l],0.),**dict.fromkeys([str(n[i]) for i in r],.3)},
                  prescribed_K_by_id={**dict.fromkeys([str(n[i]) for i in l],299.9 if mixed else 300.),
                                      **dict.fromkeys([str(n[i]) for i in r],300.1 if mixed else 300.)},
                  faces_by_node_id=[[str(n[i]) for i in f] for f in side],convection_W_per_m2_K=100.,ambient_K=ambient)
    return digest,mesh,vol,controls

def proc(root,conf):
    f=root/'manifest.json'; f.write_bytes(canonical(conf))
    return StdioServerParameters(command=sys.executable,args=['-B','-m','fibernet_mcp.server','--manifest',str(f),'--output-root',str(root),'--audit','audit.json'],env=dict(os.environ))

async def run(root,enabled):
    p=proc(root,cfg(enabled))
    result={'enabled':enabled}
    async with stdio_client(p) as (r,w), ClientSession(r,w) as s:
        await s.initialize()
        tools=await s.list_tools(); ns={t.name for t in tools.tools}
        assert len(ns)==(10 if enabled else 9) and (TET4_TOOL in ns)==enabled
        assert len(tools.model_dump_json().encode())<60000
        cat=(await s.call_tool('fibernet_catalog',{})).structuredContent
        g,=cat['graphs']; assert (g['node_count'],g['edge_count'])==(4340,5073)
        val=await s.call_tool('fibernet_validate',{'graph_id':g['graph_id']}); assert not val.isError and val.structuredContent['validation_status']=='valid'
        result['tools']=len(ns); result['tools_list_payload_bytes']=len(tools.model_dump_json().encode())
        if not enabled:
            assert 'tet4_meshes' not in cat
            return result
        meshes={q['fixture']:q for q in cat['tet4_meshes']}
        found=[]
        for name,sigma,mixed in (('simple',.5,False),('welded',2.,False),('welded',.5,True)):
            mid=meshes[name]['mesh_id']; digest,mesh,vol,a=fixture_case(name,mid,sigma,mixed)
            r=await s.call_tool(TET4_TOOL,a)
            assert not r.isError,r
            ok=r.structuredContent
            assert ok['source_sha256']==digest and ok['controls_sha256']==sha256(canonical(a)).hexdigest()
            art=ok['artifact'];id_=art['artifact_id']; assert art['kind']=='tet4_dc_convection_result' and art['producer']['fibernet_version']=='4.21.0'
            inf=await s.call_tool('fibernet_artifact_info',{'artifact_id':id_});assert inf.structuredContent['artifact']==art
            rr=await s.read_resource(art['uri']);assert json.loads(rr.contents[0].text)==art
            b=b'';offset=0
            while True:
                rr=await s.call_tool('fibernet_artifact_read',{'artifact_id':id_,'offset':offset,'length':32768})
                assert not rr.isError
                q=rr.structuredContent;d=base64.b64decode(q['data'],validate=True)
                b+=d;assert q['next_offset']==len(b) and len(d)<=32768
                offset=q['next_offset']
                if q['eof']:break
            assert sha256(b).hexdigest()==art['sha256'] and len(b)==art['byte_size']
            row=json.loads(b)
            direct=solve_dc_thermal_tet4_with_convection(vol,**parse_req(canonical(a),mesh,mid))
            dc,h=direct.electrical,direct.thermal
            expected=dict(voltage_V=dc.voltage_V,temperature_K=h.temperature_K,joule_power_W_by_cell=dc.joule_power_W_by_cell,
                          convective_heat_into_solid_W_by_node=h.convective_heat_into_solid_W_by_node,prescribed_heat_into_solid_W_by_node=h.heat_into_solid_W)
            np.testing.assert_array_equal(row['node_ids'],list(map(str,dc.node_ids)))
            np.testing.assert_array_equal(row['cell_ids'],list(map(str,dc.cell_ids)))
            for k,vals in expected.items():np.testing.assert_array_equal(row['fields'][k],vals)
            assert row['summary']==ok['summary'] and abs(h.balance_W)<1e-16
            conv=h.convective_heat_into_solid_W_by_node
            if mixed: assert float(np.min(conv))<0.<float(np.max(conv)) and abs(h.total_convective_heat_into_solid_W)<np.sum(abs(conv))*.01
            found.append(dict(fixture=name,nodes=len(mesh[0]),cells=len(mesh[1]),sigma=sigma,mixed_sign=mixed,artifact=art['sha256'],
                              inflow_W=float(np.sum(conv[conv>0])),outflow_W=float(np.sum(conv[conv<0])),net_W=h.total_convective_heat_into_solid_W,balance_W=h.balance_W))
        # Cross-kind isolation, disallowed manual source and hostile numerals
        # are checked at the named tool boundary, then a real graph call works.
        for tool,bad in ((TET4_TOOL,a|{'mesh_id':g['graph_id']}),(TET4_TOOL,a|{'ambient_K':10**80}),
                        (TET4_TOOL,a|{'source_W_per_m3_by_cell':{'123':1.}}),('fibernet_validate',{'graph_id':mid})):
            e=await s.call_tool(tool,bad)
            assert e.isError and e.structuredContent['error']['code']==('SOLID_INVALID_ARGUMENT_OR_HANDLE' if tool==TET4_TOOL else 'INVALID_ARGUMENT_OR_HANDLE'),e
        excessive=a|{'electrode_voltage_V_by_id':{k:v*100. for k,v in a['electrode_voltage_V_by_id'].items()}}
        error=await s.call_tool(TET4_TOOL,excessive)
        assert error.isError and error.structuredContent['error']['code']=='SOLID_SOURCE_ENVELOPE',error
        check=await s.call_tool('fibernet_validate',{'graph_id':g['graph_id']});assert not check.isError
        result.update(cases=found,live_invalid_requests_rejected=5)
    assert not list(root.glob('fibernet-mcp-*'))
    audit=json.loads((root/'audit.json').read_bytes())
    assert audit['artifacts']==3 and len(audit['workers'])==6 and all(x['reaped'] for x in audit['workers'])
    result['audit']=audit
    return result

async def main():
    with tempfile.TemporaryDirectory(prefix='fiber-mcp-mixed-') as t:
        p=Path(t);(p/'legacy').mkdir();(p/'mixed').mkdir()
        results=[await run(p/'legacy',False),await run(p/'mixed',True)]
        print(json.dumps(results,indent=2))
if __name__=='__main__':anyio.run(main)
