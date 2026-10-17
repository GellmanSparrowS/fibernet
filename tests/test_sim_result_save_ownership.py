"""Synthetic persistence tests: no solvers or Taichi runtime initialization."""
import hashlib
import json
import os
from pathlib import Path
import warnings

import numpy as np
import pytest
import fibernet.sim.accelerated as m


def seed(path, existing=True):
    if existing:
        path.write_bytes(b'old destination')
        path.chmod(0o640)
    sentinel = path.with_name(path.name + '.tmp')
    sentinel.write_bytes(b'unrelated sentinel')
    return sentinel


@pytest.mark.parametrize('existing', [False, True])
@pytest.mark.parametrize('detailed', [False, True])
def test_success_owns_private_unique_temporary(tmp_path, monkeypatch, existing, detailed):
    path = tmp_path / 'result.json'
    sentinel = seed(path, existing)
    original = Path.replace
    seen = []
    # Windows exposes read/write flags; chmod does not establish a private ACL.
    expected_mode = (0o640 if existing else 0o600) if os.name == "posix" else 0o666
    def replace(source, target):
        assert source.parent == path.parent and source not in (path, sentinel)
        assert source.stat().st_mode & 0o7777 == expected_mode
        seen.append(source)
        return original(source, target)
    monkeypatch.setattr(Path, 'replace', replace)
    result = m.SimResult(mode='synthetic', energy=1.5)
    result.save(path, detailed=detailed)
    assert sentinel.read_bytes() == b'unrelated sentinel'
    assert path.stat().st_mode & 0o7777 == expected_mode
    assert len(seen) == 1 and not seen[0].exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ['result.json', 'result.json.tmp']
    if not detailed:
        # Keep the frozen LF reference while checking native text-mode bytes.
        reference = json.dumps(result.to_dict(), indent=2).encode('utf-8')
        assert hashlib.sha256(reference).hexdigest() == '6fe01d8892260c9a12c428973f8361bb556b2d09109dd85088830f8aadc2e8c1'
        assert path.read_bytes() == reference.replace(b'\n', os.linesep.encode('ascii'))
    assert path.read_text() == json.dumps(result.to_dict(detailed=detailed), indent=2)


@pytest.mark.parametrize('existing', [False, True])
@pytest.mark.parametrize('stage', ['serialize', 'write', 'chmod', 'replace', 'create', 'fdopen'])
def test_failures_preserve_originals(tmp_path, monkeypatch, existing, stage):
    path = tmp_path / 'result.json'
    sentinel = seed(path, existing)
    result = m.SimResult(metadata={'bad': object()} if stage == 'serialize' else {})
    def fail(*args, **kwargs):
        raise OSError(stage + ' injection')
    if stage == 'write':
        def partial(data, stream, **kwargs):
            stream.write('partial')
            fail()
        monkeypatch.setattr(m.json, 'dump', partial)
    elif stage == 'chmod':
        monkeypatch.setattr(m.os, 'chmod', fail)
    elif stage == 'replace':
        monkeypatch.setattr(Path, 'replace', fail)
    elif stage == 'create':
        monkeypatch.setattr(m.tempfile, 'mkstemp', fail)
    elif stage == 'fdopen':
        monkeypatch.setattr(m.os, 'fdopen', fail)
    with pytest.raises(TypeError if stage == 'serialize' else OSError):
        result.save(path)
    assert sentinel.read_bytes() == b'unrelated sentinel'
    assert path.read_bytes() == b'old destination' if existing else not path.exists()
    assert not list(tmp_path.glob('.*.result-*'))


@pytest.mark.parametrize('kind', ['symlink', 'dangling', 'directory',
    pytest.param('fifo', marks=pytest.mark.skipif(not hasattr(os, 'mkfifo'), reason='FIFO creation unavailable')),
    pytest.param('special', marks=pytest.mark.skipif(os.name != 'posix', reason='POSIX special permission bits'))])
def test_nonordinary_targets_rejected(tmp_path, kind):
    path = tmp_path / 'result.json'
    sentinel = seed(path, False)
    real = tmp_path / 'referent'
    if kind in ('symlink', 'dangling'):
        if kind == 'symlink':
            real.write_bytes(b'referent')
        path.symlink_to(real)
    elif kind == 'directory':
        path.mkdir()
    elif kind == 'fifo':
        os.mkfifo(path)
    else:
        path.write_bytes(b'original')
        path.chmod(0o1640)
    mode = path.lstat().st_mode
    with pytest.raises(ValueError):
        m.SimResult().save(path)
    assert path.lstat().st_mode == mode
    assert sentinel.read_bytes() == b'unrelated sentinel'
    if kind == 'symlink':
        assert real.read_bytes() == b'referent'
    assert not list(tmp_path.glob('.*.result-*'))


@pytest.mark.parametrize('commit', [False, True])
@pytest.mark.parametrize('warning_error', [False, True])
def test_cleanup_never_masks_outcome(tmp_path, monkeypatch, commit, warning_error):
    path = tmp_path / 'result.json'
    sentinel = seed(path)
    original = Path.unlink
    def unlink(source, *args, **kwargs):
        if '.result-' in source.name:
            raise OSError('cleanup injection')
        return original(source, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', unlink)
    result = m.SimResult(metadata={} if commit else {'bad': object()})
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('error' if warning_error else 'always')
        if commit:
            result.save(path)
            assert path.read_bytes() != b'old destination'
        else:
            with pytest.raises(TypeError, match='Cannot serialize object'):
                result.save(path)
            assert path.read_bytes() == b'old destination'
        if not warning_error:
            assert any('cleanup injection' in str(w.message) for w in captured)
    assert sentinel.read_bytes() == b'unrelated sentinel'


def test_nested_parent_and_numpy_serialization(tmp_path):
    path = tmp_path / 'nested' / 'result.json'
    result = m.SimResult(displacements=np.array([[1., 2.]]),
                         metadata={'scalar': np.int64(7), 'array': np.array([3, 4])})
    result.save(path, detailed=True)
    assert json.loads(path.read_text())['scalar'] == 7
    np.testing.assert_equal(m.SimResult.load(path).displacements, result.displacements)
