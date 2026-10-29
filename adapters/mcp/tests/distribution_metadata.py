"""Installed-package binding smoke, without tool calls or native computation.

Run with Python -B, --wheels DIR and --installed-root DIR. Build/install are
separate steps. The wheel directory must contain one matching native wheel and
one adapter wheel. This check does not replace distribution.py's full parity
checks. Only disposable copies are changed for the one stale-manifest refusal.
Use --work-root DIR to retain receipts, refusal output and failure diagnostics.
"""
import argparse
import builtins
import contextlib
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback
import zipfile


TOOLS = (
    'fibernet_catalog', 'fibernet_validate', 'fibernet_analyze_topology',
    'fibernet_artifact_info', 'fibernet_artifact_read',
    'fibernet_solve_linear_transport', 'fibernet_prepare_deformed_transport',
    'fibernet_solve_thermoelastic_network_snapshot',
    'fibernet_project_fiber_network_report',
)
NATIVE_SOURCE_COUNT = 248
FORBIDDEN = frozenset((
    'fibernet', 'numpy', 'scipy', 'networkx', 'matplotlib', 'torch',
    'torch_geometric', 'taichi', 'lammps', 'exudyn', 'mujoco', 'pybullet',
))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('utf-8')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def snapshot(root):
    """Read all files so an installed target cannot be silently changed."""
    result = {}
    for path in sorted(root.rglob('*')):
        assert not path.is_symlink(), 'symlink in inspected package tree'
        if path.is_file():
            result[path.relative_to(root).as_posix()] = digest(path.read_bytes())
    return result


def package_payload(root):
    return {name: value for name, value in snapshot(root).items()
            if name.endswith(('.py', '.json'))}


@contextlib.contextmanager
def work_area(requested):
    if requested is None:
        with tempfile.TemporaryDirectory(prefix='fibernet-package-check-') as temporary:
            yield Path(temporary)
    else:
        assert not requested.is_symlink(), 'work root must not be a symlink'
        area = requested.resolve()
        if area.exists():
            assert area.is_dir() and not any(area.iterdir()), 'work root must be empty'
        else:
            area.mkdir(parents=True)
        # Retain success and failure evidence, including the stale adapter copy.
        yield area


def record_exception(area, label, error):
    # Capture the initiating error inside the SDK context before exit/cleanup
    # can replace it with a transport or task-group exception.
    record = {'type': type(error).__name__, 'message': str(error),
              'traceback': ''.join(traceback.format_exception(
                  type(error), error, error.__traceback__))}
    (area / (label + '.json')).write_bytes(canonical(record) + b'\n')


def inspect_wheels(directory, adapter):
    source = adapter / 'src/fibernet_mcp'
    expected_adapter = package_payload(source)
    manifest = json.loads((source / 'core_manifest.json').read_bytes())
    assert len(manifest) == NATIVE_SOURCE_COUNT
    assert 'examples/linear_static_network.py' in manifest
    assert 'sim/deformed_transport.py' in manifest
    license_bytes = (adapter / 'LICENSE').read_bytes()
    adapter_wheels = list(directory.glob('fibernet_mcp-*.whl'))
    core_wheels = list(directory.glob('fibernet-*.whl'))
    assert len(adapter_wheels) == len(core_wheels) == 1
    assert set(directory.glob('*.whl')) == set(adapter_wheels + core_wheels)
    for path, package in ((adapter_wheels[0], 'fibernet_mcp'),
                          (core_wheels[0], 'fibernet')):
        with zipfile.ZipFile(path) as wheel:
            names = wheel.namelist()
            assert len(names) == len(set(names)) and wheel.testzip() is None
            notices = [name for name in names
                       if '.dist-info/' in name and name.endswith('/LICENSE')]
            assert len(notices) == 1
            assert wheel.read(notices[0]) == license_bytes
            metadata = [name for name in names if name.endswith('.dist-info/METADATA')]
            assert len(metadata) == 1
            assert 'License-File: LICENSE' in wheel.read(metadata[0]).decode('utf-8')
            prefix = package + '/'
            members = {name[len(prefix):]: digest(wheel.read(name)) for name in names
                       if name.startswith(prefix) and not name.endswith('/')}
            if package == 'fibernet_mcp':
                # No fixture, private harness, extra module, or alternate schema
                # is permitted in the runtime package.
                assert members == expected_adapter
                assert json.loads(wheel.read(prefix + 'core_manifest.json')) == manifest
            else:
                actual = {name: value for name, value in members.items()
                          if name.endswith('.py')}
                assert actual == manifest
    return manifest, expected_adapter


# The same installed entry point is exercised in the positive and refusal
# cases. Guards prohibit native computation, network access, and child workers.
# Looking up fibernet's spec remains allowed because verify_core only reads it.
CHILD = r'''
import builtins
import importlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys

mode, target, adapter, operator, output, receipt = sys.argv[1:]
target, adapter = Path(target).resolve(), Path(adapter).resolve()
for name, expected in [('fibernet', target/'fibernet/__init__.py'),
                       ('fibernet_mcp', adapter/'fibernet_mcp/__init__.py')]:
    spec = importlib.util.find_spec(name)
    assert spec is not None and Path(spec.origin).resolve() == expected
forbidden = set('fibernet numpy scipy networkx matplotlib torch torch_geometric taichi lammps exudyn mujoco pybullet'.split())
attempts = {'imports': [], 'processes': [], 'network': []}
original_import = builtins.__import__
original_import_module = importlib.import_module
def check_import(name):
    if name.split('.')[0] in forbidden:
        attempts['imports'].append(name)
        raise AssertionError('native/provider import forbidden')
def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    absolute = importlib.util.resolve_name('.' * level + name,
        (globals or {}).get('__package__')) if level else name
    check_import(absolute)
    return original_import(name, globals, locals, fromlist, level)
def guarded_import_module(name, package=None):
    check_import(importlib.util.resolve_name(name, package) if name.startswith('.') else name)
    return original_import_module(name, package)
def audit(event, args):
    if event in ('subprocess.Popen', 'os.system', 'os.fork', 'os.forkpty', 'os.posix_spawn', 'os.exec'):
        attempts['processes'].append(event)
        raise AssertionError('worker/process launch forbidden')
    if event in ('socket.connect', 'socket.connect_ex', 'socket.getaddrinfo'):
        attempts['network'].append(event)
        raise AssertionError('network access forbidden')
builtins.__import__ = guarded_import
importlib.import_module = guarded_import_module
sys.addaudithook(audit)
try:
    if mode == 'verify-refusal':
        from fibernet_mcp.safety import ContractError, verify_core
        try:
            verify_core()
        except ContractError as error:
            assert str(error) == 'native core differs from frozen source'
            print('expected stale preparation hash refused')
        else:
            raise AssertionError('stale manifest accepted')
    else:
        assert mode == 'server'
        sys.argv = ['fibernet_mcp.server', '--manifest', operator, '--output-root', output]
        runpy.run_module('fibernet_mcp.server', run_name='__main__')
finally:
    loaded = sorted(name for name in sys.modules if name.split('.')[0] in forbidden)
    workers = sorted(name for name in sys.modules if name == 'fibernet_mcp.worker')
    foreign = []
    for name, module in tuple(sys.modules.items()):
        if name == 'fibernet_mcp' or name.startswith('fibernet_mcp.'):
            path = getattr(module, '__file__', None)
            if path and not Path(path).resolve().is_relative_to(adapter/'fibernet_mcp'):
                foreign.append(name)
    Path(receipt).write_text(json.dumps({'attempts': attempts,
        'native_modules_loaded': loaded, 'worker_modules_loaded': workers,
        'foreign_adapter_modules': foreign}, sort_keys=True) + '\n')
'''


def assert_receipt(path):
    value = json.loads(path.read_bytes())
    assert value == {'attempts': {'imports': [], 'processes': [], 'network': []},
                     'native_modules_loaded': [], 'worker_modules_loaded': [],
                     'foreign_adapter_modules': []}, value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheels', required=True, type=Path)
    parser.add_argument('--installed-root', required=True, type=Path)
    parser.add_argument('--work-root', type=Path,
                        help='fresh or empty directory in which evidence is retained')
    args = parser.parse_args()
    assert sys.platform == 'linux', 'the adapter supports Linux confinement'
    assert sys.dont_write_bytecode, 'run this check with Python -B'
    adapter = Path(__file__).resolve().parents[1]
    target = args.installed_root.resolve(strict=True)
    assert not target.is_relative_to(adapter), 'use a separate installed target'
    manifest, expected_adapter = inspect_wheels(args.wheels.resolve(strict=True), adapter)
    assert snapshot(target / 'fibernet_mcp') == expected_adapter
    installed_core = {name: value for name, value in snapshot(target / 'fibernet').items()
                      if name.endswith('.py')}
    assert installed_core == manifest
    installed_before = snapshot(target)
    assert not any(name.split('.')[0] in FORBIDDEN for name in sys.modules)
    assert not any(name == 'fibernet_mcp' or name.startswith('fibernet_mcp.')
                   for name in sys.modules)
    sys.path.insert(0, str(target))
    for name in ('fibernet', 'fibernet_mcp'):
        spec = importlib.util.find_spec(name)
        assert spec is not None
        assert Path(spec.origin).resolve() == target / name / '__init__.py'

    attempts = []
    original_import = builtins.__import__
    original_import_module = importlib.import_module

    def check_import(name):
        if name.split('.')[0] in FORBIDDEN:
            attempts.append(name)
            raise AssertionError('native/provider import forbidden')

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        absolute = importlib.util.resolve_name('.' * level + name,
            (globals or {}).get('__package__')) if level else name
        check_import(absolute)
        return original_import(name, globals, locals, fromlist, level)

    def guarded_import_module(name, package=None):
        check_import(importlib.util.resolve_name(name, package)
                     if name.startswith('.') else name)
        return original_import_module(name, package)

    builtins.__import__ = guarded_import
    importlib.import_module = guarded_import_module
    try:
        import anyio
        import fibernet_mcp
        from fibernet_mcp.preparation import PREPARATION_LIMITS
        from fibernet_mcp.thermoelastic import SNAPSHOT_LIMITS
        from fibernet_mcp.safety import verify_core
        from fibernet_mcp.server import TREE
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        assert Path(fibernet_mcp.__file__).resolve() == target / 'fibernet_mcp/__init__.py'
        assert Path(verify_core()).resolve() == target / 'fibernet'
        expected = {'schema_version': 'fibernet.core-provenance/1',
                    'canonical_core_manifest_sha256': digest(canonical(manifest)),
                    'source_file_count': NATIVE_SOURCE_COUNT,
                    'scope': 'fibernet_python_sources'}
        environment = {key: value for key, value in os.environ.items()
                       if key not in ('PYTHONPATH', 'FIBERNET_CORE_PATH')}
        environment.update(PYTHONPATH=str(target), PYTHONDONTWRITEBYTECODE='1')

        async def no_call_tool(*args, **kwargs):
            raise AssertionError('tool calls are outside this package smoke')

        with work_area(args.work_root) as area:
            output = area / 'output'
            output.mkdir()
            # A serialized fixture, not a StructureGraph constructor or generator.
            graph = {'version': '3.1', 'dimension': 3, 'tolerance': 1e-6,
                     'nodes': [{'id': 0, 'position': [0, 0, 0]}], 'edges': []}
            graph_path = area / 'graph.json'
            graph_path.write_bytes(canonical(graph))
            operator = area / 'operator.json'
            operator.write_bytes(canonical({'source_tree': TREE, 'graphs': [{
                'path': str(graph_path), 'sha256': digest(graph_path.read_bytes()),
                'length_unit': 'unspecified', 'frame_id': 'unknown'}]}))

            def command(mode, package_root, receipt):
                return [sys.executable, '-B', '-c', CHILD, mode, str(target),
                        str(package_root), str(operator), str(output), str(receipt)]

            async def pristine():
                receipt = area / 'pristine.json'
                invocation = command('server', target, receipt)
                parameters = StdioServerParameters(command=invocation[0],
                    args=invocation[1:], env=environment, cwd=str(area))
                try:
                    with anyio.fail_after(20):
                        async with stdio_client(parameters) as (reader, writer):
                            try:
                                async with ClientSession(reader, writer) as session:
                                    try:
                                        session.call_tool = no_call_tool
                                        initialized = await session.initialize()
                                        assert initialized.protocolVersion == '2025-11-25'
                                        listing = await session.list_tools()
                                        assert tuple(tool.name for tool in listing.tools) == TOOLS
                                        provenance = await session.read_resource('fibernet://provenance/1')
                                        assert len(provenance.contents) == 1
                                        assert json.loads(provenance.contents[0].text) == expected
                                        capabilities = await session.read_resource('fibernet://capabilities')
                                        assert len(capabilities.contents) == 1
                                        actual = json.loads(capabilities.contents[0].text)
                                        assert actual == {'schema_version': 'fibernet.mcp/1',
                                            'protocol': '2025-11-25', 'source_tree': TREE,
                                            'physics': ['steady_linear_electrical_conductance','prescribed_temperature_linear_planar_welded_snapshot'],
                                            'preparation': ['caller_prescribed_deformed_transport'],
                                            'preparation_limits': PREPARATION_LIMITS,
                                            'thermoelastic_limits': SNAPSHOT_LIMITS,
                                            'transport': 'stdio', 'platform': 'linux'}
                                    except BaseException as error:
                                        record_exception(area, 'sdk-request-error', error)
                                        raise
                            except BaseException as error:
                                record_exception(area, 'sdk-session-error', error)
                                raise
                except BaseException as error:
                    record_exception(area, 'sdk-transport-error', error)
                    raise
                assert_receipt(receipt)

            anyio.run(pristine)
            assert not list(output.iterdir())
            stale = area / 'stale'
            stale.mkdir()
            shutil.copytree(target / 'fibernet_mcp', stale / 'fibernet_mcp')
            changed = dict(manifest)
            key = 'sim/deformed_transport.py'
            changed[key] = '0' * 64 if manifest[key] != '0' * 64 else '1' * 64
            assert [name for name in manifest if changed[name] != manifest[name]] == [key]
            (stale / 'fibernet_mcp/core_manifest.json').write_bytes(canonical(changed))
            stale_environment = dict(environment)
            stale_environment['PYTHONPATH'] = os.pathsep.join((str(stale), str(target)))
            for mode, label in (('verify-refusal', 'direct'), ('server', 'startup')):
                receipt = area / (label + '.json')
                try:
                    result = subprocess.run(command(mode, stale, receipt), input=b'',
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        env=stale_environment, cwd=area, timeout=15)
                except subprocess.TimeoutExpired as error:
                    (area / (label + '.stdout')).write_bytes(error.stdout or b'')
                    (area / (label + '.stderr')).write_bytes(error.stderr or b'')
                    record_exception(area, label + '-error', error)
                    raise
                (area / (label + '.stdout')).write_bytes(result.stdout)
                (area / (label + '.stderr')).write_bytes(result.stderr)
                (area / (label + '-exit.json')).write_bytes(canonical(
                    {'returncode': result.returncode}) + b'\n')
                if mode == 'verify-refusal':
                    assert result.returncode == 0 and result.stderr == b''
                    assert result.stdout == b'expected stale preparation hash refused\n'
                else:
                    assert result.returncode == 2 and result.stdout == b''
                    assert result.stderr == b'FiberNet MCP startup or transport rejected\n'
                assert_receipt(receipt)
                assert not list(output.iterdir())
        assert not attempts
        assert not any(name.split('.')[0] in FORBIDDEN for name in sys.modules)
        assert all(not getattr(module, '__file__', None)
                   or Path(module.__file__).resolve().is_relative_to(target / 'fibernet_mcp')
                   for name, module in tuple(sys.modules.items())
                   if name == 'fibernet_mcp' or name.startswith('fibernet_mcp.'))
        assert snapshot(target) == installed_before
        assert Path(verify_core()).resolve() == target / 'fibernet'
    finally:
        builtins.__import__ = original_import
        importlib.import_module = original_import_module
    print(json.dumps({'wheel_and_installed_source_binding': True,
        'native_source_count': NATIVE_SOURCE_COUNT, 'license_bytes_match': True,
        'listed_tools': len(TOOLS), 'tool_names': list(TOOLS), 'core_provenance': expected,
        'sdk_sessions': 1, 'call_tool_requests': 0, 'scientific_workers': 0,
        'provider_import_attempts': 0, 'stale_manifest_key': key,
        'direct_and_startup_refusals': True, 'pristine_installation_unchanged': True,
        'output_empty_after_close': True}, indent=2))


if __name__ == '__main__':
    main()
