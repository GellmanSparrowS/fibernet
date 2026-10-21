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
assert len(manifest) == 235
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
    cases = [(relative, None) for relative in ('rl/algorithms.py', 'sim/electrothermal.py', 'sim/_thermal_step.py', 'sim/_taichi_viscoelastic.py', 'integrations/_lammps_cg.py', 'integrations/_harmonic_profile.py', 'integrations/_cg_network_profile.py', 'sim/_lammps_result.py', 'core/viscoelastic.py', 'sim/_viscoelastic_fem.py', 'ml/beam_frame_fem.py', 'sim/joule_heating.py', 'sim/_roundoff.py', 'sim/linear_transport.py', 'sim/transient_thermal.py',
                     'integrations/lammps_backend.py', 'integrations/_lammps_worker.py',
                     '__init__.py', 'gen/__init__.py', 'gen/manufacturing_bridge.py',
                     'gen/manufacturing_trails.py', 'api.py', 'easy.py', 'viz/__init__.py',
                     'sim/accelerated.py', 'sim/_energy_accounting.py', 'ml/graph_data.py', 'ml/dataset.py', 'ml/gnn.py', 'ml/__init__.py')]
    cases.append(('ml/graph_data.py', b"'youngs_modulus': 'Pa'"))
    cases.append(('ml/edge_gine.py', b"_MATERIAL_VERSION = 'fibernet.edge_gine.material_view.v1'"))
    cases.append(('io/mesh_export.py', b'vertex_offset = 0  # Convert to one-based OBJ indices only when writing.'))
    cases.append(('gen/solid_export.py', b'zero_area = _first_zero_area_face(vertices, faces)'))
    cases.append(('gen/tube_solid.py', b'mesh=result.to_mesh64()'))
    cases.append(('gen/tube_solid.py', b'first_index=int(indices.min())'))
    cases.append(('gen/tube_solid.py', b'result=m.Manifold(mesh)'))
    cases.append(('integrations/lammps_integration.py', b'raise self._lammps_import_error'))
    cases.append(('integrations/lammps_backend.py', b'if not required <= set(pins):'))
    cases.append(('integrations/_lammps_worker.py', b'if h.hexdigest() != expected:'))
    cases.append(('integrations/lammps_backend.py', b'except BaseException as exc:'))
    cases.append(('integrations/lammps_backend.py', b'if self.state == TaskState.CREATED and self._process is None:'))
    cases.append(('gen/solid_export.py', b'<vertex x="%.17g" y="%.17g" z="%.17g"/>'))
    cases.append(('gen/solid_export.py', b"packed = _binary_stl_vertices(solid) if path.suffix.lower() == '.stl' else None"))
    high_level_mutations = {
        ('sim/_lammps_result.py', None): (None, 'high_level_adapter_source_drift'),
        ('sim/_lammps_result.py', b"SCHEMA = 'lammps_harmonic_state/1'"): (b"SCHEMA = 'native_axial_state/1'", 'high_level_backend_state_schema_drift'),
        ('sim/_lammps_result.py', b'displacements=displacement/scale, deformed_positions=positions/scale'): (b'displacements=displacement, deformed_positions=positions', 'high_level_graph_units_drift'),
        ('sim/_lammps_result.py', b'ids, edges = list(prepared.node_ids), list(prepared.edge_ids)'): (b'ids, edges = list(reversed(prepared.node_ids)), list(reversed(prepared.edge_ids))', 'high_level_stable_ids_drift'),
        ('sim/_lammps_result.py', b'coefficients[:,0]*(lengths-coefficients[:,1])'): (b'coefficients[:,0]*lengths', 'high_level_equilibrium_force_law_drift'),
        ('sim/_lammps_result.py', b'if set(kwargs)-allowed:'): (b'if False:', 'high_level_strict_options_drift'),
        ('sim/accelerated.py', b"if self.metadata.get('backend') == 'lammps':\n            raise ValueError("): (b'if False:\n            raise ValueError(', 'high_level_recompute_guard_drift'),
    }
    cases.extend(high_level_mutations)
    for relative, material_unit in cases:
        target = core / relative
        saved = target.read_bytes()
        try:
            if material_unit is None:
                changed = saved + b'\n# Deliberate test-only core drift.\n'
            else:
                assert saved.count(material_unit) == (2 if material_unit == b'mesh=result.to_mesh64()' else 1)
                if (relative, material_unit) in high_level_mutations:
                    replacement = high_level_mutations[(relative, material_unit)][0]
                elif relative == 'gen/tube_solid.py':
                    replacement = (b'result=result' if material_unit == b'result=m.Manifold(mesh)'
                                   else b'first_index=0'
                                   if material_unit == b'first_index=int(indices.min())'
                                   else b'mesh=result.to_mesh()')
                elif relative == 'gen/solid_export.py':
                    replacement = (b"packed = solid.vertices.astype('<f4') if path.suffix.lower() == '.stl' else None"
                                   if material_unit.startswith(b'packed = ')
                                   else material_unit.replace(b'%.17g', b'%.9g')
                                   if material_unit.startswith(b'<vertex x=') else b'zero_area = None')
                elif relative == 'integrations/lammps_backend.py':
                    replacement = (b'except Exception as exc:' if material_unit == b'except BaseException as exc:'
                                   else b'if self.state == TaskState.CREATED:' if material_unit.startswith(b'if self.state == TaskState.CREATED')
                                   else b'if False:')
                elif relative == 'integrations/_lammps_worker.py':
                    replacement = b'if False:'
                elif relative == 'integrations/lammps_integration.py':
                    replacement = b'return LAMMPSResult()'
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
                            'case': (high_level_mutations.get((relative, material_unit), (None, None))[1] or ('tube_native_cleanup_drift' if material_unit == b'result=m.Manifold(mesh)'
                                     else 'tube_index_range_guard_drift' if material_unit == b'first_index=int(indices.min())'
                                     else 'tube_mesh64_output_drift' if relative == 'gen/tube_solid.py'
                                     else 'binary_stl_serialization_guard_drift' if material_unit and material_unit.startswith(b'packed = ')
                                     else 'three_mf_coordinate_precision_drift' if material_unit and material_unit.startswith(b'<vertex x=')
                                     else 'solid_zero_area_guard_drift' if relative == 'gen/solid_export.py'
                                     else 'legacy_lammps_failure_semantics_drift' if relative == 'integrations/lammps_integration.py'
                                     else 'lammps_start_baseexception_ownership_drift' if material_unit == b'except BaseException as exc:'
                                     else 'lammps_created_owned_cancel_drift' if material_unit and material_unit.startswith(b'if self.state == TaskState.CREATED')
                                     else 'serial_runtime_required_pins_drift' if material_unit and relative == 'integrations/lammps_backend.py'
                                     else 'serial_worker_rehash_drift' if material_unit and relative == 'integrations/_lammps_worker.py'
                                     else 'obj_index_origin_drift' if relative == 'io/mesh_export.py'
                                     else 'gine_material_profile_drift' if relative == 'ml/edge_gine.py'
                                     else 'material_feature_unit_drift' if material_unit else 'source_drift')),
                            'exact_hash_boundary_refusal': True,
                            'real_startup_refusal': True})
        finally:
            target.write_bytes(saved)
    for case in ('added_python_file', 'missing_python_file', 'symlink_directory'):
        target = core / ('sim/electrothermal.py' if case == 'missing_python_file'
                         else 'sim/_unexpected.py' if case == 'added_python_file'
                         else '_unexpected_directory')
        saved = target.read_bytes() if case == 'missing_python_file' else None
        try:
            if case == 'missing_python_file':
                target.unlink()
            elif case == 'added_python_file':
                target.write_bytes(b'# Unlisted executable source.\n')
            else:
                target.symlink_to(core / 'sim', target_is_directory=True)
            expected = ('native core inventory unavailable' if case == 'symlink_directory'
                        else 'native core file inventory differs from frozen source')
            with patch.object(importlib.util, 'find_spec', return_value=SimpleNamespace(origin=str(core / '__init__.py'))):
                try:
                    verify_core()
                except ContractError as error:
                    assert str(error) == expected
                else:
                    raise AssertionError('changed inventory accepted')
            result = subprocess.run(
                [sys.executable, '-m', 'fibernet_mcp.server', '--manifest',
                 str(ROOT / 'evidence/manifest.json'), '--output-root', str(output)],
                input=b'', stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=environment, cwd=str(area), timeout=10,
            )
            assert result.returncode == 2 and result.stdout == b''
            assert result.stderr == b'FiberNet MCP startup or transport rejected\n'
            assert not list(output.iterdir())
            records.append({'module': str(target.relative_to(core)), 'case': case,
                            'exact_inventory_boundary_refusal': True,
                            'real_startup_refusal': True})
        finally:
            if saved is not None:
                target.write_bytes(saved)
            else:
                target.unlink()
    # Traversal errors must not masquerade as a complete inventory.
    def failed_walk(*args, **kwargs):
        kwargs['onerror'](PermissionError('test-only directory traversal failure'))
        return iter(())
    with patch('fibernet_mcp.safety.os.walk', side_effect=failed_walk):
        try:
            verify_core()
        except ContractError as error:
            assert str(error) == 'native core inventory unavailable'
        else:
            raise AssertionError('unreadable inventory accepted')
    anyio.run(positive, environment, output)
    assert not list(output.iterdir())

report = {'core_hashes_verified': len(manifest), 'sdk_positive_controls': 2,
          'same_output_root_and_core': True, 'inventory_traversal_error_refused': True, 'refusals': records}
print(json.dumps(report, indent=2))
