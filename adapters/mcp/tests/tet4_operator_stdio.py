"""Actual operator snapshot registration and comparative fitted-mesh solves."""
import base64
import json
import os
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import subprocess,sys,tempfile

import anyio
import numpy as np
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client
from fibernet.io.tet4_meshio import read_tet4_vtu_bundle
from fibernet.sim.tet4_dc_thermal import solve_dc_thermal_tet4_with_convection
from fibernet_mcp.safety import canonical,open_directory
from fibernet_mcp.server import TREE
from fibernet_mcp.tet4_contract import TET4_TOOL,parse_mesh,parse_req
from fibernet_mcp.tet4_operator import load_operator_sources
recipe_path=Path(__file__).resolve().parents[1]/'examples'/'prepare_fitted_tet4.py'
spec=spec_from_file_location('_prepare_fitted_tet4',recipe_path)
recipe=module_from_spec(spec)
spec.loader.exec_module(recipe)
NAME,prepare=recipe.NAME,recipe.prepare

def cfg(root,entry):
    graph=root/'graph.json'
    graph.write_bytes(canonical(dict(version='3.1',dimension=3,tolerance=1e-6,
                            nodes=[dict(id=0,position=[0,0,0])],edges=[])))
    return dict(source_tree=TREE,graphs=[dict(path=str(graph),sha256=sha256(graph.read_bytes()).hexdigest(),
        length_unit='unspecified',frame_id='unknown')],tet4_fixtures=['simple'],tet4_sources=[entry])

def command(manifest,out):
    return [sys.executable,'-B','-m','fibernet_mcp.server','--manifest',str(manifest),'--output-root',str(out),'--audit','audit.json']

async def run(root,via_vtu):
    root=root/'operator'; entry,cases,original=prepare(root,via_vtu=via_vtu); out=root/'output';out.mkdir()
    target=root/'tet4-sources'/f'{NAME}.tet4.json'
    assert target.stat().st_size==111499
    untouched=target.read_bytes()
    mesh=parse_mesh(untouched,entry['sha256'])
    native=read_tet4_vtu_bundle(root/'fitted-axial-junction.zip') if via_vtu else original
    for field,expected in zip((native.points,native.tetrahedra,native.node_ids,native.cell_ids,native.material_ids),mesh[:5]):
        np.testing.assert_array_equal(field,expected)
    assert native.materials==mesh[5] and native.length_unit==mesh[6]=='mm'
    assert len(mesh[5])==76
    manifest=root/'manifest.json'; conf=cfg(root,entry)
    manifest.write_bytes(canonical(conf))
    cmd=command(manifest,out)
    settings=StdioServerParameters(command=cmd[0],args=cmd[1:],env=dict(os.environ))
    by_case={}
    async with stdio_client(settings) as (reader,writer), ClientSession(reader,writer) as session:
        assert (await session.initialize()).protocolVersion=='2025-11-25'
        tools=await session.list_tools()
        assert len(tools.tools)==10 and len(tools.model_dump_json().encode())<60000
        r=await session.call_tool('fibernet_catalog',{}); assert not r.isError,r
        catalog=r.structuredContent
        bundled,operator=catalog['tet4_meshes']
        assert bundled['fixture']=='simple' and 'source_kind' not in bundled
        mid=operator['mesh_id']
        assert operator==dict(mesh_id=mid,source_sha256=entry['sha256'],source_kind='operator',
                                    name=NAME,unit='mm',nodes=694,cells=2241)
        graph=catalog['graphs'][0]['graph_id']
        assert not (await session.call_tool('fibernet_validate',{'graph_id':graph})).isError
        # Only the verified managed bytes survive mutation of the original.
        target.write_bytes(b'mutated-local-fixture')
        frozen,=out.glob(f'fibernet-mcp-*/{mid}')
        assert frozen.read_bytes()==untouched and frozen.stat().st_mode&0o777==0o600
        for name,a in cases.items():
            a=a|dict(mesh_id=mid)
            r=await session.call_tool(TET4_TOOL,a);assert not r.isError,r
            reply=r.structuredContent
            assert reply['mesh_id']==mid and reply['source_sha256']==entry['sha256']
            desc=reply['artifact']; off=0; result=b''
            while True:
                msg=await session.call_tool('fibernet_artifact_read',{'artifact_id':desc['artifact_id'],'offset':off})
                assert not msg.isError,msg
                q=msg.structuredContent;block=base64.b64decode(q['data'],validate=True)
                assert len(block)<=32768
                result+=block;off=q['next_offset']
                if q['eof']:break
                assert block
            assert len(result)==desc['byte_size'] and sha256(result).hexdigest()==desc['sha256']
            v=json.loads(result)
            assert v['controls_sha256']==reply['controls_sha256']==sha256(canonical(a)).hexdigest()
            direct=solve_dc_thermal_tet4_with_convection(native,**parse_req(canonical(a),mesh,mid))
            dc,heat=direct.electrical,direct.thermal
            assert v['node_ids']==list(map(str,dc.node_ids)) and v['cell_ids']==list(map(str,dc.cell_ids))
            for f,truth in dict(temperature_K=heat.temperature_K,voltage_V=dc.voltage_V,
                joule_power_W_by_cell=dc.joule_power_W_by_cell,
                convective_heat_into_solid_W_by_node=heat.convective_heat_into_solid_W_by_node,
                prescribed_heat_into_solid_W_by_node=heat.heat_into_solid_W).items():
                np.testing.assert_array_equal(v['fields'][f],truth)
            assert abs(heat.balance_W)<1e-16 and v['summary']==reply['summary']
            by_case[name]=dict(artifact_sha256=desc['sha256'],controls_sha256=v['controls_sha256'],summary=v['summary'])
        assert len({d['controls_sha256'] for d in by_case.values()})==2
        assert by_case['lower_junction_conductivity']['summary']['total_joule_power_W'] < by_case['uniform_electrical']['summary']['total_joule_power_W']
        assert not (await session.call_tool('fibernet_validate',{'graph_id':graph})).isError
    assert not list(out.glob('fibernet-mcp-*'))
    lifecycle=json.loads((out/'audit.json').read_bytes())
    assert lifecycle['artifacts']==2 and len(lifecycle['workers'])==4 and all(w['reaped'] for w in lifecycle['workers'])
    # A new session must re-verify the pinned digest, not reuse a cached copy.
    doomed=subprocess.run(cmd,input=b'',capture_output=True,timeout=8)
    assert doomed.returncode==2 and doomed.stdout==b'' and not list(out.glob('fibernet-mcp-*'))
    return dict(sdk='1.29.0',source=entry,source_mutation_followups=True,refuses_next_start=True,offline_native_zip=via_vtu,
        nodes=694,tet4=2241,material_count=76,selected_faces=len(a['faces_by_node_id']),cases=by_case)

async def main(via_vtu):
    with tempfile.TemporaryDirectory(prefix='fiber-operator-tet4-fit-') as t:
        print(json.dumps(await run(Path(t),via_vtu),indent=2))
if __name__=='__main__':
    assert sys.argv[1:] in ([],['--offline-vtu'])
    anyio.run(main,bool(sys.argv[1:]))
