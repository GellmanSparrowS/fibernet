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
assert len(manifest) == 217
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
    cases = [(relative, None) for relative in ('sim/_roundoff.py', 'sim/linear_transport.py', 'sim/transient_thermal.py',
                     'integrations/lammps_backend.py', 'integrations/_lammps_worker.py',
                     '__init__.py', 'api.py', 'easy.py', 'viz/__init__.py',
                     'sim/accelerated.py', 'sim/_energy_accounting.py', 'ml/graph_data.py', 'ml/dataset.py', 'ml/gnn.py', 'ml/__init__.py')]
    cases.append(('ml/graph_data.py', b"'youngs_modulus': 'Pa'"))
    cases.append(('ml/edge_gine.py', b"_MATERIAL_VERSION = 'fibernet.edge_gine.material_view.v1'"))
    cases.append(('io/mesh_export.py', b'vertex_offset = 0  # Convert to one-based OBJ indices only when writing.'))
    cases.append(('gen/solid_export.py', b'zero_area = _first_zero_area_face(vertices, faces)'))
    cases.append(('gen/tube_solid.py', b'mesh=result.to_mesh64()'))
    cases.append(('gen/tube_solid.py', b'first_index=int(indices.min())'))
    cases.append(('gen/tube_solid.py', b'result=m.Manifold(mesh)'))
    cases.append(('gen/solid_export.py', b"packed = _binary_stl_vertices(solid) if path.suffix.lower() == '.stl' else None"))
    for relative, material_unit in cases:
        target = core / relative
        saved = target.read_bytes()
        try:
            if material_unit is None:
                changed = saved + b'\n# Deliberate test-only core drift.\n'
            else:
                assert saved.count(material_unit) == (2 if material_unit == b'mesh=result.to_mesh64()' else 1)
                if relative == 'gen/tube_solid.py':
                    replacement = (b'result=result' if material_unit == b'result=m.Manifold(mesh)'
                                   else b'first_index=0'
                                   if material_unit == b'first_index=int(indices.min())'
                                   else b'mesh=result.to_mesh()')
                elif relative == 'gen/solid_export.py':
                    replacement = (b"packed = solid.vertices.astype('<f4') if path.suffix.lower() == '.stl' else None"
                                   if material_unit.startswith(b'packed = ') else b'zero_area = None')
                elif relative == 'io/mesh_export.py':
                    replacement = material_unit.replace(b'vertex_offset = 0', b'vertex_offset = 1')
                else:
                    replacement = (b"_MATERIAL_VERSION = 'fibernet.edge_gine.material_view.v0'"
                                   if relative == 'ml/edge_gine.py' else b"'youngs_modulus': 'MPa'")
                changed = saved.replace(material_unit, replacement, 1)
            target.write_bytes(changed)
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
            records.append({'module': relative,
                            'case': ('tube_native_cleanup_drift' if material_unit == b'result=m.Manifold(mesh)'
                                     else 'tube_index_range_guard_drift' if material_unit == b'first_index=int(indices.min())'
                                     else 'tube_mesh64_output_drift' if relative == 'gen/tube_solid.py'
                                     else 'binary_stl_serialization_guard_drift' if material_unit and material_unit.startswith(b'packed = ')
                                     else 'solid_zero_area_guard_drift' if relative == 'gen/solid_export.py'
                                     else 'obj_index_origin_drift' if relative == 'io/mesh_export.py'
                                     else 'gine_material_profile_drift' if relative == 'ml/edge_gine.py'
                                     else 'material_feature_unit_drift' if material_unit else 'source_drift'),
                            'exact_hash_boundary_refusal': True,
                            'real_startup_refusal': True})
        finally:
            target.write_bytes(saved)
    anyio.run(positive, environment, output)
    assert not list(output.iterdir())

report = {'core_hashes_verified': len(manifest), 'sdk_positive_controls': 2,
          'same_output_root_and_core': True, 'refusals': records}
print(json.dumps(report, indent=2))
