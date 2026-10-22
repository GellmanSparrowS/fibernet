"""Inspect built notices/bindings and exercise an installed adapter over stdio."""
import argparse
import base64
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile


parser = argparse.ArgumentParser()
parser.add_argument('--wheels', type=Path, required=True)
parser.add_argument('--installed-root', type=Path,
                    help='Explicit pip --target directory, otherwise use the active environment')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
source = root.parent.parent
manifest = json.loads((root / 'src/fibernet_mcp/core_manifest.json').read_text())
license_bytes = (source / 'LICENSE').read_bytes()
assert (root / 'LICENSE').read_bytes() == license_bytes
adapter_wheels = list(args.wheels.glob('fibernet_mcp-*.whl'))
core_wheels = list(args.wheels.glob('fibernet-*.whl'))
assert len(adapter_wheels) == len(core_wheels) == 1
with zipfile.ZipFile(adapter_wheels[0]) as wheel:
    notices = [name for name in wheel.namelist()
               if '.dist-info/' in name and name.endswith('/LICENSE')]
    assert len(notices) == 1 and wheel.read(notices[0]) == license_bytes
    assert json.loads(wheel.read('fibernet_mcp/core_manifest.json')) == manifest
    metadata = next(name for name in wheel.namelist() if name.endswith('/METADATA'))
    assert 'License-File: LICENSE' in wheel.read(metadata).decode()
with zipfile.ZipFile(core_wheels[0]) as wheel:
    for relative, expected in manifest.items():
        assert hashlib.sha256(wheel.read('fibernet/' + relative)).hexdigest() == expected

environment = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}
if args.installed_root:
    installed = args.installed_root.resolve()
    sys.path.insert(0, str(installed))
    environment['PYTHONPATH'] = str(installed)
import anyio
import fibernet_mcp
from fibernet_mcp.safety import verify_core
from fibernet_mcp.server import TREE
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

core_root = Path(verify_core()).resolve()
adapter_root = Path(fibernet_mcp.__file__).resolve().parent
assert not core_root.is_relative_to(source) and not adapter_root.is_relative_to(source)
if args.installed_root:
    assert core_root.is_relative_to(installed) and adapter_root.is_relative_to(installed)




def expected_core_provenance(manifest):
    raw = json.dumps(manifest, sort_keys=True, separators=(',', ':'),
                     ensure_ascii=True).encode('utf-8')
    return {'schema_version': 'fibernet.core-provenance/1',
            'canonical_core_manifest_sha256': hashlib.sha256(raw).hexdigest(),
            'source_file_count': len(manifest), 'scope': 'fibernet_python_sources'}

def electrical_fixture():
    from fibernet.core.structure_graph import StructureGraph
    nodes=[{'id':100+11*i,'position':[float(i%8),float(i//8),0.]} for i in range(64)]
    pairs=[]
    for row in range(8):
        for column in range(8):
            i=row*8+column
            if column<7:pairs.append((i,i+1))
            if row<7:pairs.append((i,i+8))
            if row<7 and column<7:pairs.append((i,i+9))
    pairs+=pairs[:10]
    edges=[{'id':500+7*i,'node_i':nodes[a]['id'],'node_j':nodes[b]['id']}
           for i,(a,b) in enumerate(pairs)]
    for i,edge in enumerate(edges):
        if i%2:edge['node_i'],edge['node_j']=edge['node_j'],edge['node_i']
    data=StructureGraph.from_dict({'dimension':3,'nodes':nodes,'edges':edges}).to_dict()
    data['nodes'].reverse();data['edges'].reverse()
    assert [e['id'] for e in data['edges']]!=sorted(e['id'] for e in data['edges'])
    request={'conductance_S':[{'edge_id':e['id'],'value_S':0 if i%53==0 else 1+(i%7)/8}
        for i,e in enumerate(data['edges'])],
        'dirichlet_voltage_V':[{'node_id':nodes[i]['id'],'value_V':1 if i<8 else 0}
            for i in list(range(8))+list(range(56,64))]}
    return data,request


async def read_content(session,artifact):
    raw=b'';offset=0
    while True:
        chunk=await session.call_tool('fibernet_artifact_read',{
            'artifact_id':artifact['artifact_id'],'offset':offset,'length':32768})
        assert not chunk.isError
        value=chunk.structuredContent;raw+=base64.b64decode(value['data'],validate=True)
        assert value['next_offset']>offset or value['eof']
        offset=value['next_offset']
        if value['eof']:break
    assert hashlib.sha256(raw).hexdigest()==artifact['sha256']
    return json.loads(raw)


async def smoke(area):
    fixture = (root / 'fixtures/voronoi_4340.json').resolve()
    configuration = {'source_tree': TREE, 'graphs': [{
        'path': str(fixture), 'sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(),
        'length_unit': 'unspecified', 'frame_id': 'unknown'}]}
    electrical,request=electrical_fixture()
    electrical_path=area/'electrical.json'
    electrical_path.write_text(json.dumps(electrical,allow_nan=False))
    configuration['graphs'].append({'path':str(electrical_path),
        'sha256':hashlib.sha256(electrical_path.read_bytes()).hexdigest(),
        'length_unit':'mm','frame_id':'installed_test'})
    operator_manifest = area / 'manifest.json' 
    operator_manifest.write_text(json.dumps(configuration))
    output = area / 'output'
    output.mkdir()
    environment.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(area / 'mpl'),
                       XDG_CACHE_HOME=str(area / 'cache'))
    parameters = StdioServerParameters(
        command=sys.executable, args=['-m', 'fibernet_mcp.server',
                                     '--manifest', str(operator_manifest),
                                     '--output-root', str(output)],
        env=environment, cwd=str(area),
    )
    with anyio.fail_after(20):
        async with stdio_client(parameters) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                assert (await session.initialize()).protocolVersion == '2025-11-25'
                assert len((await session.list_tools()).tools) == 6
                resources = await session.list_resources()
                assert {str(item.uri) for item in resources.resources} == {'fibernet://capabilities', 'fibernet://provenance/1'}
                provenance = await session.read_resource('fibernet://provenance/1')
                assert len(provenance.contents) == 1
                assert json.loads(provenance.contents[0].text) == expected_core_provenance(manifest)
                result = await session.call_tool('fibernet_catalog', {})
                assert not result.isError
                graph = result.structuredContent['graphs'][0]
                assert (graph['node_count'], graph['edge_count']) == (4340, 5073)
                analysis = await session.call_tool('fibernet_analyze_topology',
                                                   {'graph_id': graph['graph_id']})
                assert not analysis.isError
                artifact = analysis.structuredContent['artifact']
                raw = b''
                offset = 0
                while True:
                    chunk = await session.call_tool('fibernet_artifact_read', {
                        'artifact_id': artifact['artifact_id'], 'offset': offset,
                        'length': 32768})
                    assert not chunk.isError
                    value = chunk.structuredContent
                    raw += base64.b64decode(value['data'], validate=True)
                    assert value['next_offset'] > offset or value['eof']
                    offset = value['next_offset']
                    if value['eof']:
                        break
                assert hashlib.sha256(raw).hexdigest() == artifact['sha256']
                from fibernet.core.structure_graph import StructureGraph
                from fibernet.analysis.topology_metrics import analyze_topology
                reference = asdict(analyze_topology(StructureGraph.load_json(str(fixture))))
                assert json.loads(raw) == json.loads(json.dumps(reference))
                graph=result.structuredContent['graphs'][1]
                response=await session.call_tool('fibernet_solve_linear_transport',
                    {'graph_id':graph['graph_id'],**request})
                assert not response.isError,response
                artifact=response.structuredContent['artifact']
                assert artifact['kind']=='linear_transport_result'
                actual=await read_content(session,artifact)
                from fibernet.sim.linear_transport import solve_linear_transport
                import fibernet.sim.linear_transport as native_module
                assert Path(native_module.__file__).resolve().is_relative_to(core_root)
                expected=solve_linear_transport(StructureGraph.from_dict(electrical),
                    {v['edge_id']:v['value_S'] for v in request['conductance_S']},
                    [(v['node_id'],v['value_V']) for v in request['dirichlet_voltage_V']],
                    length_unit='mm').to_dict()
                assert actual==expected
                assert actual['edge_ids']==[e['id'] for e in electrical['edges']]
                assert actual['endpoints_by_id']==[[e['node_i'],e['node_j']] for e in electrical['edges']]
                # Both protocol and native computation above use installed packages.
                return {'nodes':len(electrical['nodes']),'edges':len(electrical['edges']),
                        'parallel_edges':10,'wire_ids_unsorted':True,
                        'native_all_fields_exact':True,'artifact_sha256':artifact['sha256']}
    assert not list(output.iterdir())


with tempfile.TemporaryDirectory(prefix='fibernet-mcp-installed-') as temporary:
    electrical_report=anyio.run(smoke, Path(temporary))
    assert not list((Path(temporary)/'output').iterdir())
print(json.dumps({'wheel_license_bytes_match': True, 'wheel_core_hashes': len(manifest),
                  'installed_core_hashes': len(manifest), 'real_sdk_initialize': True,
                  'listed_tools': 6, 'catalog_nodes': 4340, 'catalog_edges': 5073,
                  'installed_worker_full_native_parity': True,
                  'installed_electrical': electrical_report,
                  'source_tree': TREE}, indent=2))
