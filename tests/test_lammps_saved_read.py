"""Synthetic saved bundles only: these tests are not native physics evidence."""
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
from unittest.mock import patch

import pytest

from fibernet.integrations import lammps_saved as reader
from fibernet.integrations import lammps_backend as backend
from fibernet.integrations import prepare_cg
from examples.cg_bent_branched_network import build_case
from test_lammps_backend_contract import prepared, fixture
from test_lammps_cg_fiber_only import synthetic_texts


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def tmp_path(tmp_path):
    # macOS may expose the system temp directory through /var -> /private/var.
    # Positive fixtures use a canonical caller path; explicit symlink tests do not.
    return tmp_path.resolve()


@pytest.fixture(autouse=True)
def forbid_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('native/runtime/network execution forbidden')
    monkeypatch.setattr(backend.RuntimeSpec, 'checked', forbidden)
    monkeypatch.setattr(backend.LAMMPSTask, '__init__', forbidden)
    monkeypatch.setattr(backend.LAMMPSTask, 'start', forbidden)
    monkeypatch.setattr('subprocess.Popen', forbidden)
    monkeypatch.setattr(os, 'system', forbidden)
    monkeypatch.setattr(os, 'execve', forbidden)
    monkeypatch.setattr(socket, 'socket', forbidden)


def bundle(path, p=None, cg=False):
    path.mkdir()
    if cg:
        graph, request = build_case(steps=2)
        p = prepare_cg(graph, request)
        dump, log = synthetic_texts(p)
    else:
        p = p or prepared()
        dump, log = fixture(p)
    runtime = backend.RuntimeSpec('cli_serial_static', '/never/read/python',
                                 '/never/read/native', expected_version=20250722,
                                 expected_banner='LAMMPS (synthetic)')
    manifest = dict(p.manifest(), runtime=asdict(runtime),
                    limits=asdict(backend.ResourceLimits()))
    values = {'graph.json': p.graph_json, 'manifest.json': canonical(manifest),
              'model.data': p.data_text, 'in.model': p.input_text,
              'trajectory.dump': dump, 'log.lammps': 'LAMMPS (synthetic)\n'+log}
    for name, text in values.items():
        (path/name).write_text(text, encoding='utf-8')
    return p


def read(path, **kwargs):
    if 'expected_manifest_sha256' not in kwargs:
        kwargs['expected_manifest_sha256'] = digest(path/'manifest.json')
    return reader.read_saved_lammps_run(path, **kwargs)


def change_manifest(path, callback):
    value = json.loads((path/'manifest.json').read_text())
    callback(value)
    (path/'manifest.json').write_text(canonical(value))


@pytest.mark.parametrize('cg', [False, True])
def test_exact_offline_roundtrip_and_immutable_receipt(tmp_path, cg):
    root = tmp_path/'saved'
    p = bundle(root, cg=cg)
    before = {n: (root/n).read_bytes() for n in reader._FILES}
    opened = read(root, expected_banner='LAMMPS (synthetic)')
    assert opened.prepared == p
    assert opened.result == backend.parse_outputs(p, (root/'trajectory.dump').read_text(),
                                                 (root/'log.lammps').read_text(),
                                                 max_output_bytes=16*reader._MIB)
    receipt = opened.receipt
    assert receipt['new_execution'] is False
    assert receipt['runtime_checked'] is False
    assert receipt['native_execution_verified'] is False
    assert receipt['physics_verified'] is False
    assert receipt['caller_anchor_authenticated'] is False
    assert receipt['compiler_revision_in_saved_manifest'] == 'not_recorded'
    assert receipt['node_ids'] == list(p.node_ids)
    assert receipt['edge_ids'] == list(p.edge_ids)
    receipt['node_ids'].append(999)
    assert opened.receipt['node_ids'] == list(p.node_ids)
    assert opened.result.workdir == ''
    assert opened.result.trajectory[0].node_ids == p.node_ids
    if not cg:
        assert receipt['graph_length_unit'] == 'mm'
        assert opened.result.trajectory[0].positions_m[1] == (.001, 0., 0.)
    assert {n: (root/n).read_bytes() for n in reader._FILES} == before
    assert not any(name == 'lammps' or name.startswith('lammps.') for name in sys.modules)


def test_full_history_profile_restores_name_from_frozen_record(tmp_path):
    p = prepared(profile='harmonic_full_history_v1')
    root = tmp_path/'saved'
    bundle(root, p)
    assert read(root).prepared == p


def test_relocation_and_untrusted_metadata_never_open_paths(tmp_path):
    root = tmp_path/'old'
    bundle(root)
    change_manifest(root, lambda m: m['runtime'].update(
        python_executable='/private/denied/path', file_sha256=[['/outside/secret', '0'*64]],
        instruction='DO NOT EXECUTE THIS TEXT', environment=[['EVIL', 'value']]))
    (root/'scratch').symlink_to('/outside/unused')
    (root/'unused').write_text('Not an instruction source')
    pinned = digest(root/'manifest.json')
    relocated = tmp_path/'new'
    root.rename(relocated)
    opened = read(relocated, expected_manifest_sha256=pinned)
    assert opened.receipt['extra_files'] == 'ignored; not validated'
    assert json.loads(opened.unverified_metadata_json)['runtime']['python_executable'] == '/private/denied/path'


@pytest.mark.parametrize('name', reader._FILES)
def test_missing_files_reject(tmp_path, name):
    root = tmp_path/'saved'
    bundle(root)
    pin = digest(root/'manifest.json')
    (root/name).unlink()
    with pytest.raises(reader.LAMMPSSavedReadError):
        read(root, expected_manifest_sha256=pin)


@pytest.mark.parametrize('kind', ['symlink', 'directory', 'fifo'])
@pytest.mark.parametrize('name', ['graph.json', 'trajectory.dump'])
def test_nonregular_and_symlink_leaves_reject(tmp_path, kind, name):
    root = tmp_path/'saved'
    bundle(root)
    saved = root/name
    saved.unlink()
    if kind == 'directory':
        saved.mkdir()
    elif kind == 'fifo':
        os.mkfifo(saved)
    else:
        target = tmp_path/'outside'
        target.write_text('{}')
        saved.symlink_to(target)
    with pytest.raises(reader.LAMMPSSavedReadError):
        read(root)


def test_root_symlink_reject(tmp_path):
    root = tmp_path/'saved'
    bundle(root)
    alias = tmp_path/'alias'
    alias.symlink_to(root)
    with pytest.raises(reader.LAMMPSSavedReadError, match='root symlink'):
        read(alias)


@pytest.mark.parametrize('key,value', [
    ('total_bytes', True), ('total_bytes', 0), ('total_bytes', 21*reader._MIB),
    ('output_bytes', -1), ('json_depth', 33), ('json_items', 100001),
    ('deadline_seconds', True), ('deadline_seconds', float('nan')),
    ('deadline_seconds', 0), ('deadline_seconds', 61),
])
def test_invalid_limits_before_any_open(tmp_path, key, value):
    limits = replace(reader.LAMMPSSavedReadLimits(), **{key: value})
    with patch.object(reader.os, 'open', side_effect=AssertionError('must not open')):
        with pytest.raises(reader.LAMMPSSavedReadError):
            reader.read_saved_lammps_run(tmp_path, expected_manifest_sha256='0'*64, limits=limits)


@pytest.mark.parametrize('key,value', [
    ('total_bytes', 100), ('prepared_bytes', 100), ('json_bytes', 100),
    ('output_bytes', 100), ('max_nodes', 1), ('max_edges', 1), ('max_steps', 1),
])
def test_positive_small_limits_fail_closed(tmp_path, key, value):
    root = tmp_path/'saved'
    bundle(root, cg=key == 'max_edges')
    with pytest.raises(reader.LAMMPSSavedReadError):
        read(root, limits=replace(reader.LAMMPSSavedReadLimits(), **{key: value}))


@pytest.mark.parametrize('value', ['', 'A'*64, '0'*63, True, None])
def test_explicit_caller_digest_required(tmp_path, value):
    with pytest.raises(reader.LAMMPSSavedReadError, match='caller must supply'):
        reader.read_saved_lammps_run(tmp_path, expected_manifest_sha256=value)


def test_wrong_caller_digest(tmp_path):
    root = tmp_path/'saved'
    bundle(root)
    with pytest.raises(reader.LAMMPSSavedReadError, match='caller manifest digest mismatch'):
        read(root, expected_manifest_sha256='0'*64)


@pytest.mark.parametrize('name', reader._FILES)
def test_invalid_utf8(tmp_path, name):
    root = tmp_path/'saved'
    bundle(root)
    (root/name).write_bytes(b'\xff')
    with pytest.raises(reader.LAMMPSSavedReadError, match='UTF-8'):
        read(root)


@pytest.mark.parametrize('text,match', [
    ('{"a":1,"a":2}', 'duplicate JSON key'),
    ('{"a":NaN}', 'nonfinite'), ('{"a":Infinity}', 'nonfinite'),
    ('{"a":1e1000}', 'nonfinite'), ('{"a":123456789012345678901}', 'integer too large'),
    ('{"a":'+'['*40+'0'+']'*40+'}', 'depth'), ('[]', 'object'),
    ('{"a": [}', 'invalid JSON'),
])
@pytest.mark.parametrize('name', ['manifest.json', 'graph.json'])
def test_strict_json_rejection(tmp_path, text, match, name):
    root = tmp_path/'saved'
    bundle(root)
    (root/name).write_text(text)
    with pytest.raises(reader.LAMMPSSavedReadError, match=match):
        read(root)


def test_json_item_budget_and_quoted_braces(tmp_path):
    root = tmp_path/'saved'
    bundle(root)
    change_manifest(root, lambda m: m['runtime'].update(comment='['*100+'\\"'+']'*100))
    read(root)
    with pytest.raises(reader.LAMMPSSavedReadError, match='item budget'):
        read(root, limits=replace(reader.LAMMPSSavedReadLimits(), json_items=10))


@pytest.mark.parametrize('field', ['identity', 'graph_sha256', 'schema'])
def test_manifest_identity_schema_corruption(tmp_path, field):
    root = tmp_path/'saved'
    bundle(root)
    change_manifest(root, lambda m: m.__setitem__(field, 'wrong'))
    with pytest.raises(reader.LAMMPSSavedReadError, match='manifest schema'):
        read(root)


@pytest.mark.parametrize('map_name', ['atom_to_node_id', 'bond_to_edge_id',
                                    'atom_to_type_id', 'type_to_mass_kg'])
@pytest.mark.parametrize('value', [True, 1.0, '1', 999])
def test_type_sensitive_redundant_maps(tmp_path, map_name, value):
    root = tmp_path/'saved'
    bundle(root)
    def mutate(m):
        key = next(iter(m[map_name]))
        m[map_name][key] = value
    change_manifest(root, mutate)
    with pytest.raises(reader.LAMMPSSavedReadError, match='manifest schema'):
        read(root)


def test_cg_angle_map_mismatch(tmp_path):
    root = tmp_path/'saved'
    bundle(root, cg=True)
    change_manifest(root, lambda m: m['angle_to_stable_id'].__setitem__('1', True))
    with pytest.raises(reader.LAMMPSSavedReadError, match='manifest schema'):
        read(root)


@pytest.mark.parametrize('name', ['graph.json', 'model.data', 'in.model'])
def test_bytes_changed_without_updated_manifest(tmp_path, name):
    root = tmp_path/'saved'
    bundle(root)
    (root/name).write_text((root/name).read_text()+' ')
    with pytest.raises(reader.LAMMPSSavedReadError, match='manifest schema'):
        read(root)


def reseal(root, p):
    old = json.loads((root/'manifest.json').read_text())
    updated = dict(p.manifest(), runtime=old['runtime'], limits=old['limits'])
    (root/'manifest.json').write_text(canonical(updated))
    (root/'graph.json').write_text(p.graph_json)
    (root/'model.data').write_text(p.data_text)
    (root/'in.model').write_text(p.input_text)


@pytest.mark.parametrize('change', ['script', 'units', 'scale', 'steps_boolean',
                                   'duplicate_node', 'boolean_node', 'profile', 'float_id'])
def test_resealed_contradictions_and_compiler_drift_fail_closed(tmp_path, change):
    root = tmp_path/'saved'
    p = bundle(root)
    graph, contract = json.loads(p.graph_json), json.loads(p.contract_json)
    if change == 'script':
        p = replace(p, input_text=p.input_text+'# different historical compiler revision\n')
    elif change == 'units':
        contract['units'] = 'real'
    elif change == 'scale':
        contract['length_to_m'] = 1
    elif change == 'steps_boolean':
        contract['steps'] = True
    elif change == 'duplicate_node':
        graph['nodes'][1]['id'] = graph['nodes'][0]['id']
    elif change == 'boolean_node':
        graph['nodes'][0]['id'] = True
    elif change == 'float_id':
        graph['nodes'][0]['id'] = 10.0
    elif change == 'profile':
        contract['profile'] = {'name': 'not_supported'}
        p = replace(p, profile='not_supported')
    p = replace(p, graph_json=canonical(graph), contract_json=canonical(contract))
    reseal(root, p)
    with pytest.raises(reader.LAMMPSSavedReadError) as error:
        read(root)
    if change in ('script', 'units', 'scale', 'profile'):
        assert isinstance(error.value, reader.LAMMPSCompilerCompatibilityError)
        assert 'no migration attempted' in str(error.value)
    assert (root/'in.model').read_text() == p.input_text


@pytest.mark.parametrize('change', ['truncated', 'marker', 'wrong_unit', 'wrong_id', 'nonfinite'])
def test_output_protocol_reuses_parser(tmp_path, change):
    root = tmp_path/'saved'
    p = bundle(root)
    dump, log = (root/'trajectory.dump').read_text(), (root/'log.lammps').read_text()
    if change == 'truncated':
        dump = dump[:len(dump)//2]
    elif change == 'marker':
        log = log.replace(p.completion_marker, 'wrong_marker')
    elif change == 'wrong_unit':
        dump = dump.replace('ITEM: UNITS\nsi', 'ITEM: UNITS\nreal')
    elif change == 'wrong_id':
        dump = dump.replace('2 2 0.001', '99 2 0.001')
    elif change == 'nonfinite':
        dump = dump.replace('2 2 0.001', '2 2 nan')
    (root/'trajectory.dump').write_text(dump)
    (root/'log.lammps').write_text(log)
    with pytest.raises(backend.LAMMPSContractError):
        read(root)


@pytest.mark.parametrize('banner', ['', True, 'bad\nline', 'different'])
def test_bad_banner(tmp_path, banner):
    root = tmp_path/'saved'
    bundle(root)
    with pytest.raises(reader.LAMMPSSavedReadError, match='banner'):
        read(root, expected_banner=banner)


def test_reader_deadline(tmp_path):
    root = tmp_path/'saved'
    bundle(root)
    with patch.object(reader.time, 'monotonic', side_effect=[0., 11.]):
        with pytest.raises(reader.LAMMPSSavedReadError, match='deadline'):
            read(root)


def test_growth_is_bounded_even_when_initial_stat_is_stale(tmp_path):
    root = tmp_path/'saved'
    bundle(root)
    real_read = reader.os.read
    def growing(fd, size):
        real_read(fd, size)
        return b'x'*size
    with patch.object(reader.os, 'read', side_effect=growing):
        with pytest.raises(reader.LAMMPSSavedReadError, match='byte budget'):
            read(root, expected_manifest_sha256='0'*64)


def test_mutation_after_snapshot_rejected(tmp_path):
    root = tmp_path/'saved'
    bundle(root)
    parser = reader.parse_outputs
    def changed(*args, **kwargs):
        result = parser(*args, **kwargs)
        (root/'graph.json').write_text('{}')
        return result
    with patch.object(reader, 'parse_outputs', side_effect=changed):
        with pytest.raises(reader.LAMMPSSavedReadError, match='changed during validation'):
            read(root)


def test_no_automatic_simresult_or_runtime_evidence(tmp_path):
    root = tmp_path/'saved'
    bundle(root)
    with patch('fibernet.sim._lammps_result.to_sim_result', side_effect=AssertionError('no invented wall time')):
        value = read(root)
    assert not hasattr(value, 'task')
    assert not hasattr(value, 'sim_result')
    assert value.receipt['historical_task_status'] == 'not_recorded_in_input_manifest'


pytestmark = pytest.mark.skipif(os.name != 'posix' or not hasattr(os, 'O_NOFOLLOW'),
                                reason='saved file reader requires POSIX no-follow descriptors')
