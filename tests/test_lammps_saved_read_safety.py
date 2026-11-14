"""Independent synthetic safety probes. No native engine execution."""
import builtins
import json
import os
from dataclasses import replace
from unittest.mock import patch
import pytest
from fibernet.integrations import lammps_saved as reader
from test_lammps_saved_read import bundle, read, digest, reseal, change_manifest, forbid_execution, tmp_path


def test_exact_opens_and_no_python_file_provider(tmp_path):
    root = tmp_path / 'saved'
    bundle(root)
    change_manifest(root, lambda m: m['runtime'].update(python_executable='/not/a/provider', environment={'PYTHONPATH': '/not/a/provider'}, instruction='exec this'))
    pin = digest(root/'manifest.json')
    real_open = os.open
    opened = []
    def tracked(path, flags, *args, **kwargs):
        opened.append((str(path), kwargs.get('dir_fd')))
        return real_open(path, flags, *args, **kwargs)
    with patch.object(reader.os, 'open', side_effect=tracked), patch.object(builtins, 'open', side_effect=AssertionError('unexpected Python file open')):
        value = reader.read_saved_lammps_run(root, expected_manifest_sha256=pin)
    assert opened[0][0] == str(root)
    assert [x[0] for x in opened[1:]] == list(reader._FILES)
    assert all(x[1] is not None for x in opened[1:])
    assert value.receipt['runtime_checked'] is False


def test_same_size_inode_replacement_rejected(tmp_path):
    root = tmp_path/'saved'
    bundle(root)
    parser = reader.parse_outputs
    def replace_after_parse(*args, **kwargs):
        result = parser(*args, **kwargs)
        replacement = root/'replacement'
        replacement.write_bytes((root/'graph.json').read_bytes())
        replacement.replace(root/'graph.json')
        return result
    with patch.object(reader, 'parse_outputs', side_effect=replace_after_parse):
        with pytest.raises(reader.LAMMPSSavedReadError, match='changed during validation'):
            read(root)


def test_same_length_inplace_mutation_rejected(tmp_path):
    root = tmp_path/'saved'
    bundle(root)
    parser = reader.parse_outputs
    def mutate_after_parse(*args, **kwargs):
        result = parser(*args, **kwargs)
        p = root/'log.lammps'
        p.write_text(p.read_text().replace('synthetic', 'SYNTHETIC'))
        return result
    with patch.object(reader, 'parse_outputs', side_effect=mutate_after_parse):
        with pytest.raises(reader.LAMMPSSavedReadError, match='changed during validation'):
            read(root)


def test_executable_looking_script_resealed_still_rejected(tmp_path):
    root = tmp_path/'saved'
    p = bundle(root)
    reseal(root, replace(p, input_text=p.input_text+'shell touch SHOULD_NEVER_EXIST\n'))
    with pytest.raises(reader.LAMMPSCompilerCompatibilityError):
        read(root)
    assert not (root/'SHOULD_NEVER_EXIST').exists()


def test_output_edit_can_pass_same_manifest_anchor_and_is_not_authenticated(tmp_path):
    root = tmp_path/'saved'
    bundle(root)
    pin = digest(root/'manifest.json')
    first = read(root, expected_manifest_sha256=pin)
    lines = (root/'trajectory.dump').read_text().splitlines()
    changed = 0
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) == 11 and parts[0].isdigit():
            parts[-1] = '123.25'
            lines[i] = ' '.join(parts)
            changed += 1
    assert changed > 0
    (root/'trajectory.dump').write_text('\n'.join(lines)+'\n')
    second = read(root, expected_manifest_sha256=pin)
    assert first.receipt['file_sha256']['trajectory.dump'] != second.receipt['file_sha256']['trajectory.dump']
    assert second.receipt['native_execution_verified'] is False
    assert second.receipt['physics_verified'] is False
    assert second.result.trajectory[0].forces_N[0][-1] == 123.25


@pytest.mark.parametrize('value', [True, 1.0, '1', None, [], {}])
def test_sample_every_malformed_types_reject(tmp_path, value):
    root = tmp_path/'saved'
    bundle(root)
    change_manifest(root, lambda m: m['contract'].__setitem__('sample_every', value))
    with pytest.raises(reader.LAMMPSSavedReadError):
        read(root)


@pytest.mark.parametrize('value', [True, 0, -1, '1', None, [], {}])
def test_timestep_malformed_types_reject(tmp_path, value):
    root = tmp_path/'saved'
    bundle(root)
    change_manifest(root, lambda m: m['contract'].__setitem__('timestep_s', value))
    with pytest.raises(reader.LAMMPSSavedReadError):
        read(root)


def test_symlink_ancestor_rejected(tmp_path):
    real = tmp_path/'real'
    real.mkdir()
    root = real/'saved'
    bundle(root)
    alias = tmp_path/'alias'
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(reader.LAMMPSSavedReadError, match='root symlink'):
        read(alias/'saved')


def test_multibyte_raw_json_budget(tmp_path):
    root = tmp_path/'saved'
    bundle(root)
    m = json.loads((root/'manifest.json').read_text())
    m['runtime']['comment'] = '\u4e2d'*1000
    text = json.dumps(m, ensure_ascii=False)
    (root/'manifest.json').write_text(text)
    limits = replace(reader.LAMMPSSavedReadLimits(), json_bytes=len(text))
    assert len(text.encode()) > limits.json_bytes
    with pytest.raises(reader.LAMMPSSavedReadError, match='byte budget'):
        read(root, limits=limits)


pytestmark = pytest.mark.skipif(os.name != 'posix' or not hasattr(os, 'O_NOFOLLOW'),
                                reason='saved file reader requires POSIX no-follow descriptors')
