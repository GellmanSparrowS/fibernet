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


async def smoke(area):
    fixture = (root / 'fixtures/voronoi_4340.json').resolve()
    configuration = {'source_tree': TREE, 'graphs': [{
        'path': str(fixture), 'sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(),
        'length_unit': 'unspecified', 'frame_id': 'unknown'}]}
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
                assert len((await session.list_tools()).tools) == 5
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
    assert not list(output.iterdir())


with tempfile.TemporaryDirectory(prefix='fibernet-mcp-installed-') as temporary:
    anyio.run(smoke, Path(temporary))
print(json.dumps({'wheel_license_bytes_match': True, 'wheel_core_hashes': len(manifest),
                  'installed_core_hashes': len(manifest), 'real_sdk_initialize': True,
                  'listed_tools': 5, 'catalog_nodes': 4340, 'catalog_edges': 5073,
                  'installed_worker_full_native_parity': True,
                  'source_tree': TREE}, indent=2))
