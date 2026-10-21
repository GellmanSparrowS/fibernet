"""Real-server refusals paired with positive startup and exact boundary checks."""
import copy
import errno
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from acceptance import ROOT, ENV
from fibernet_mcp.safety import ContractError, graph_preflight, read_file, strict_json
from fibernet_mcp.server import BoundedInput, ManagedStore, TREE

P = Path(tempfile.mkdtemp(prefix='negative-', dir=ROOT / 'evidence'))
OUT = P / 'output'
OUT.mkdir(mode=0o700)
FIXTURE = ROOT / 'fixtures/voronoi_4340.json'
base = json.loads(FIXTURE.read_text())
results = []
GENERIC_STDERR = b'FiberNet MCP startup or transport rejected\n'


def config(source, name, expected=None):
    raw = source.read_bytes()
    manifest = {
        'source_tree': TREE,
        'graphs': [{
            'path': str(source),
            'sha256': expected or hashlib.sha256(raw).hexdigest(),
            'length_unit': 'unspecified',
            'frame_id': 'unknown',
        }],
    }
    path = P / (name + '.manifest.json')
    path.write_text(json.dumps(manifest))
    return path


POSITIVE_MANIFEST = config(FIXTURE, 'positive-control')


def command(manifest):
    return [sys.executable, '-m', 'fibernet_mcp.server', '--manifest',
            str(manifest), '--output-root', str(OUT)]


async def positive_control():
    """Use the identical interpreter, environment and owned output root."""
    parameters = StdioServerParameters(
        command=sys.executable, args=command(POSITIVE_MANIFEST)[1:], env=ENV)
    with anyio.fail_after(10):
        async with stdio_client(parameters) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                initialized = await session.initialize()
                assert initialized.protocolVersion == '2025-11-25'
                response = await session.call_tool('fibernet_catalog', {})
                assert not response.isError
                graphs = response.structuredContent['graphs']
                assert len(graphs) == 1
                assert graphs[0]['node_count'] == 4340
                assert graphs[0]['edge_count'] == 5073
                assert graphs[0]['source_sha256'] == hashlib.sha256(
                    FIXTURE.read_bytes()).hexdigest()
                resource = await session.read_resource('fibernet://capabilities')
                assert json.loads(resource.contents[0].text)['source_tree'] == TREE
    assert not list(OUT.iterdir()), 'positive-control session was not cleaned'


def expect_contract(operation, reason):
    try:
        operation()
    except ContractError as exc:
        assert str(exc) == reason, (reason, str(exc))
    else:
        raise AssertionError('expected ContractError: ' + reason)


def source_boundary(path, manifest, boundary, reason):
    if boundary == 'strict_json':
        expect_contract(lambda: strict_json(path.read_bytes()), reason)
    elif boundary == 'graph_preflight':
        # Parsing must succeed so a parser failure cannot satisfy this test.
        data = strict_json(path.read_bytes())
        expect_contract(lambda: graph_preflight(data), reason)
    elif boundary == 'source_digest':
        def open_store():
            store = ManagedStore(str(manifest), str(OUT))
            store.close()
        expect_contract(open_store, reason)
        assert not list(OUT.iterdir()), 'failed store construction leaked files'
    elif boundary in ('symlink_leaf', 'symlink_parent'):
        # The no-follow OS boundary raises OSError rather than ContractError.
        expected_errnos = ({errno.ELOOP} if boundary == 'symlink_leaf'
                           else {errno.ENOTDIR, errno.ELOOP})
        try:
            read_file(str(path))
        except OSError as exc:
            assert exc.errno in expected_errnos, (boundary, exc.errno)
        else:
            raise AssertionError('symlink was accepted at the no-follow boundary')
    else:
        raise AssertionError('unknown test boundary: ' + boundary)


def server_refusal(name, manifest, payload, boundary, reason):
    # Pair every negative with a real SDK control; startup/configuration failure
    # must never stand in for the intended malformed-input rejection.
    anyio.run(positive_control)
    process = subprocess.run(
        command(manifest), input=payload, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=ENV, timeout=10)
    assert process.returncode == 2 and not process.stdout, (
        name, process.returncode, process.stdout, process.stderr)
    assert process.stderr == GENERIC_STDERR, (name, process.stderr)
    assert not list(OUT.iterdir()), 'negative session was not cleaned'
    results.append({
        'case': name, 'boundary': boundary, 'expected_reason': reason,
        'boundary_verified': True, 'same_root_sdk_control': True,
        'generic_stderr': True, 'rejected': True,
    })


def rejected(name, boundary, reason, raw=None, path=None, expected=None):
    if path is None:
        path = P / (name + '.json')
        path.write_bytes(raw)
    manifest = config(path, name, expected)
    source_boundary(path, manifest, boundary, reason)
    server_refusal(name, manifest, b'', boundary, reason)


for name, edit, boundary, reason in [
    ('bool_id', lambda d: d['nodes'][0].update(id=True),
     'graph_preflight', 'invalid integer'),
    ('fraction_id', lambda d: d['nodes'][0].update(id=1.5),
     'graph_preflight', 'invalid integer'),
    ('duplicate_id', lambda d: d['nodes'][1].update(id=d['nodes'][0]['id']),
     'graph_preflight', 'duplicate node ID'),
    ('version', lambda d: d.update(version='999'),
     'graph_preflight', 'unsupported graph version or dimension'),
    ('unknown_field', lambda d: d.update(code='print(1)'),
     'graph_preflight', 'invalid record fields'),
    ('nonfinite', lambda d: d['nodes'][0].update(position=[float('nan'), 0, 0]),
     'strict_json', 'invalid strict JSON'),
    ('string_position', lambda d: d['nodes'][0].update(position=['1', 0, 0]),
     'graph_preflight', 'invalid finite number'),
    ('oversized_nodes', lambda d: d.update(nodes=d['nodes'] * 3),
     'graph_preflight', 'entity budget exceeded'),
    ('long_metadata', lambda d: d.update(metadata={'text': 'x' * 4097}),
     'strict_json', 'string budget exceeded'),
]:
    data = copy.deepcopy(base)
    edit(data)
    rejected(name, boundary, reason, raw=json.dumps(data).encode())

rejected('duplicate_key', 'strict_json', 'invalid strict JSON',
         raw=b'{"version":"3.1","version":"3.1"}')
data = {'x': 1}
for _ in range(20):
    data = {'x': data}
rejected('deep_metadata', 'strict_json', 'JSON depth budget exceeded',
         raw=json.dumps(data).encode())
rejected('source_hash', 'source_digest', 'source digest mismatch',
         raw=json.dumps(base).encode(), expected='0' * 64)
link = P / 'source-link.json'
link.symlink_to(FIXTURE)
rejected('symlink_source', 'symlink_leaf', 'ELOOP', path=link)
parent = P / 'linked-parent'
parent.symlink_to(ROOT / 'fixtures', target_is_directory=True)
rejected('symlink_parent', 'symlink_parent', 'ENOTDIR or ELOOP',
         path=parent / 'voronoi_4340.json')


async def transport_boundary(payload, reason):
    # Exercise the real pre-SDK input boundary without changing production code.
    with patch.object(sys, 'stdin', SimpleNamespace(buffer=io.BytesIO(payload))):
        iterator = BoundedInput().lines()
        try:
            await iterator.__anext__()
        except ContractError as exc:
            assert str(exc) == reason, (reason, str(exc))
        else:
            raise AssertionError('expected transport ContractError: ' + reason)
        finally:
            await iterator.aclose()


for name, payload, reason in [
    ('oversize_transport', b'x' * (20 * 1024 * 1024 + 1) + b'\n',
     'transport line budget'),
    ('duplicate_transport_key',
     b'{"jsonrpc":"2.0","jsonrpc":"2.0","id":1,"method":"initialize"}\n',
     'invalid strict JSON'),
]:
    anyio.run(transport_boundary, payload, reason)
    server_refusal(name, POSITIVE_MANIFEST, payload, 'BoundedInput', reason)

(ROOT / 'evidence' / ('adversarial-report-' + P.name + '.json')).write_text(
    json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
shutil.rmtree(P)
