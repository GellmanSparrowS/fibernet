"""Exact native-core binding refusals, paired with a real SDK positive control."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from acceptance import ROOT, ENV
from fibernet_mcp.safety import ContractError, verify_core


async def positive(environment, output):
    params = StdioServerParameters(
        command=sys.executable,
        args=['-m', 'fibernet_mcp.server', '--manifest',
              str(ROOT / 'evidence/manifest.json'), '--output-root', str(output)],
        env=environment, cwd=str(output.parent),
    )
    with anyio.fail_after(10):
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                assert (await session.initialize()).protocolVersion == '2025-11-25'
                assert len((await session.list_tools()).tools) == 5
                result = await session.call_tool('fibernet_catalog', {})
                assert not result.isError
                graph = result.structuredContent['graphs'][0]
                assert (graph['node_count'], graph['edge_count']) == (4340, 5073)


original = Path(verify_core())
manifest = json.loads((ROOT / 'src/fibernet_mcp/core_manifest.json').read_text())
assert len(manifest) == 216
records = []
with tempfile.TemporaryDirectory(prefix='binding-', dir=ROOT / 'evidence') as temporary:
    area = Path(temporary)
    core = area / 'fibernet'
    output = area / 'output'
    output.mkdir()
    for relative, digest in manifest.items():
        source = original / relative
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
        destination = core / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    environment = dict(ENV)
    environment['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'src'), str(area)])
    with patch.object(importlib.util, 'find_spec', return_value=SimpleNamespace(origin=str(core / '__init__.py'))):
        assert Path(verify_core()) == core
    anyio.run(positive, environment, output)
    assert not list(output.iterdir())
    for relative in ('sim/_roundoff.py', 'sim/linear_transport.py', 'sim/transient_thermal.py',
                     'integrations/lammps_backend.py', 'integrations/_lammps_worker.py',
                     '__init__.py', 'api.py', 'easy.py', 'viz/__init__.py',
                     'sim/accelerated.py', 'sim/_energy_accounting.py'):
        target = core / relative
        saved = target.read_bytes()
        try:
            target.write_bytes(saved + b'\n# Deliberate test-only core drift.\n')
            with patch.object(importlib.util, 'find_spec', return_value=SimpleNamespace(origin=str(core / '__init__.py'))):
                try:
                    verify_core()
                except ContractError as error:
                    assert str(error) == 'native core differs from frozen source'
                else:
                    raise AssertionError('changed core accepted at hash boundary')
            result = subprocess.run(
                [sys.executable, '-m', 'fibernet_mcp.server', '--manifest',
                 str(ROOT / 'evidence/manifest.json'), '--output-root', str(output)],
                input=b'', stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=environment, cwd=str(area), timeout=10,
            )
            assert result.returncode == 2 and result.stdout == b''
            assert result.stderr == b'FiberNet MCP startup or transport rejected\n'
            assert not list(output.iterdir())
            records.append({'module': relative, 'exact_hash_boundary_refusal': True,
                            'real_startup_refusal': True})
        finally:
            target.write_bytes(saved)
    anyio.run(positive, environment, output)
    assert not list(output.iterdir())

report = {'core_hashes_verified': len(manifest), 'sdk_positive_controls': 2,
          'same_output_root_and_core': True, 'refusals': records}
print(json.dumps(report, indent=2))
