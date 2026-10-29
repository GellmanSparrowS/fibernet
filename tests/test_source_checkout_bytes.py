"""Byte-sealed sources keep their Git bytes even in CRLF-oriented checkouts."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROTECTED = (
    'tests/fixtures/legacy_analysis_v1/api_delta.json',
    'tests/fixtures/deformed_transport_preparation_v1/arithmetic_reference.py.txt',
    'tests/fixtures/deformed_transport_preparation_v1/planar_golden.json',
    'tests/fixtures/saved_electrothermal_v1/axial_chain.json.gz.b64',
    'tests/fixtures/saved_electrothermal_v1/planar_network.json.gz.b64',
    'fibernet/core/structure_graph.py',
    'fibernet/__init__.py',
    'fibernet/sim/__init__.py',
    'fibernet/sim/saved_temperature.py',
    'fibernet/sim/thermoelastic.py',
    'fibernet/sim/_thermoelastic.py',
    'tests/fixtures/optional_facades_v2/sim_thermoelastic.py.txt',
    'tests/fixtures/optional_facades_v2/sim_lammps_capabilities_delta.json',
    'tests/fixtures/optional_facades_v2/root_fiber_network_profile_delta.json',
    'tests/fixtures/optional_facades_v2/root.py.txt',
    'tests/fixtures/optional_facades_v2/sim.py.txt',
    'tests/fixtures/optional_facades_v2/sim_nodal_controls.py.txt',
    'tests/fixtures/optional_facades_v2/sim_prior.py.txt',
    'adapters/mcp/tests/fixtures/provenance_v1_outputs.json',
    'fibernet/easy.py',
    'tests/fixtures/easy_rl_failure_v1/easy.py.txt',
    'tests/fixtures/taichi_geometry_policy_v1/easy_delta.json',
    'tests/fixtures/taichi_geometry_policy_v1/accelerated_delta.json',
    'fibernet/api.py',
    'tests/fixtures/public_api_contract_v2/fibernet/__init__.py.txt',
    'tests/fixtures/public_api_contract_v2/fibernet/easy.py.txt',
    'tests/fixtures/public_api_contract_v2/fibernet/api.py.txt',
    'tests/fixtures/public_api_contract_v2/fibernet/sim/accelerated.py.txt',
    'fibernet/sim/accelerated.py',
    'tests/fixtures/result_processing_v1/accelerated_delta.json',
    'tests/fixtures/compute_detailed_validation_v1/accelerated_delta.json',
    'fibernet/integrations/_lammps_cg.py',
    'fibernet/sim/_energy_accounting.py',
    'fibernet/viz/__init__.py',
    'tests/fixtures/observer_prior71_accelerated.txt',
)
CORE_MANIFEST = json.loads(
    (ROOT / 'adapters/mcp/src/fibernet_mcp/core_manifest.json').read_text(
        encoding='utf-8'
    )
)
PROTECTED = tuple(dict.fromkeys(
    (*PROTECTED, *('fibernet/' + name for name in CORE_MANIFEST))
))

# Only bounded, explicit bulk staging has a larger I/O allowance. These are
# harness limits, not process-tree or whole-test wall-clock guarantees.
GIT_TIMEOUT_SECONDS = 15
BULK_ADD_TIMEOUT_SECONDS = 60
BULK_ADD_MAX_FILES = 300
BULK_ADD_MAX_BYTES = 5 * 1024 * 1024
DIAGNOSTIC_TAIL_CHARS = 4096


def _staging_size(repo, names):
    if not names or len(names) > BULK_ADD_MAX_FILES:
        raise ValueError('bulk staging requires 1 to 300 explicit files')
    seen = set()
    total = 0
    for name in names:
        # Guard this fixed test-owned inventory: relative literal names, no
        # traversal/pathspec syntax or NUL/CR/LF. This is not a complete
        # Windows filename or filesystem-alias validator.
        if (not isinstance(name, str) or not name or
                any(char in name for char in '\\:*?[]\x00\r\n') or
                any(part in ('', '.', '..') or part.endswith(('.', ' '))
                    for part in name.split('/'))):
            raise ValueError('bulk staging requires literal relative file paths')
        key = name.casefold()
        if key in seen:
            raise ValueError('duplicate bulk staging path')
        seen.add(key)
        target = repo
        for part in name.split('/'):
            target = target / part
            if target.is_symlink():
                raise ValueError('bulk staging does not accept symlinks')
        if not target.is_file():
            raise ValueError('bulk staging requires existing regular files')
        try:
            target.resolve().relative_to(repo.resolve())
        except ValueError:
            raise ValueError('bulk staging path escapes temporary repository')
        total += target.stat().st_size
        if total > BULK_ADD_MAX_BYTES:
            raise ValueError('bulk staging exceeds 5 MiB')
    return total


def _diagnostic_tail(value):
    # TimeoutExpired output may be bytes even with text=True. Escape controls
    # and bound the rendered tail, rather than dumping file contents or argv.
    return ascii((value or '')[-DIAGNOSTIC_TAIL_CHARS:])[-DIAGNOSTIC_TAIL_CHARS:]


def _git(repo, env, autocrlf, *args):
    timeout = GIT_TIMEOUT_SECONDS
    staged_bytes = None
    if args and args[0] == 'add':
        if len(args) < 3 or args[1] != '--':
            raise ValueError('bulk staging requires add -- with explicit files')
        staged_bytes = _staging_size(repo, args[2:])
        timeout = BULK_ADD_TIMEOUT_SECONDS
    started = time.monotonic()
    try:
        return subprocess.run(
            ['git', '-c', 'core.autocrlf=' + autocrlf,
             '-c', 'core.safecrlf=false', '-c', 'core.attributesFile=' + os.devnull,
             *args], cwd=repo, env=env,
            check=True, capture_output=True, text=True, encoding='utf-8',
            timeout=timeout,
        ).stdout
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
        # subprocess.run handles the direct child before raising. Do not retry,
        # remove index.lock, or continue checkout on a partially staged index.
        # Failure to report diagnostics must not replace the original error.
        try:
            print('git failure: phase={} autocrlf={} files={} bytes={} '
                  'budget={}s elapsed={:.3f}s repo={} index_lock={}'.format(
                      args[0] if args else '<none>', autocrlf,
                      len(args) - 2 if staged_bytes is not None else None,
                      staged_bytes, timeout, time.monotonic() - started,
                      ascii(str(repo))[-1024:], (repo / '.git/index.lock').exists()))
            print('stdout_tail=' + _diagnostic_tail(error.stdout))
            print('stderr_tail=' + _diagnostic_tail(error.stderr))
        except Exception:
            pass
        raise


def test_checkout_inventory_covers_the_native_core_manifest():
    core = ROOT / 'fibernet'
    assert set(CORE_MANIFEST) == {
        path.relative_to(core).as_posix() for path in core.rglob('*.py')
    }
    for name, expected in CORE_MANIFEST.items():
        assert hashlib.sha256((core / name).read_bytes()).hexdigest() == expected


@pytest.mark.parametrize('autocrlf', ['false', 'true'])
def test_byte_sealed_checkout_preserves_raw_hashes(tmp_path, autocrlf):
    repo = tmp_path / 'repo'
    repo.mkdir()
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith('GIT_')}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull,
               GIT_CONFIG_NOSYSTEM='1', GIT_ATTR_NOSYSTEM='1')
    def git(*args):
        return _git(repo, env, autocrlf, *args)
    git('init', '--template=', '-q')
    (repo / '.gitattributes').write_bytes((ROOT / '.gitattributes').read_bytes())
    expected = {}
    for name in PROTECTED:
        raw = (ROOT / name).read_bytes()
        expected[name] = hashlib.sha256(raw).hexdigest()
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    control = b'first line\nsecond line\n'
    (repo / 'unprotected.txt').write_bytes(control)
    # New nested package sources must retain either original newline style too.
    for name, raw in (
        ('fibernet/checkout_probe/nested/raw_lf.py', control),
        ('fibernet/checkout_probe/nested/raw_crlf.py', control.replace(b'\n', b'\r\n')),
    ):
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        expected[name] = hashlib.sha256(raw).hexdigest()
    git('add', '--', '.gitattributes', 'unprotected.txt', *expected)
    # Remove only these disposable working copies, then materialize from index.
    for name in (*expected, 'unprotected.txt'):
        (repo / name).unlink()
    git('checkout-index', '--all', '--force')
    for name in expected:
        raw = (repo / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected[name]
        # Byte checks are intentionally not whitespace/line-ending normalizers.
        assert hashlib.sha256(raw + b'\n').hexdigest() != expected[name]
        if b'\n' in raw:
            assert hashlib.sha256(raw.replace(b'\n', b'\r\n')).hexdigest() != expected[name]
    materialized = (repo / 'unprotected.txt').read_bytes()
    assert materialized == (control.replace(b'\n', b'\r\n') if autocrlf == 'true' else control)


@pytest.mark.parametrize('count, accepted', [(300, True), (301, False)])
def test_bulk_add_file_count_boundary(tmp_path, monkeypatch, count, accepted):
    names = tuple('file_{}.txt'.format(index) for index in range(count))
    for name in names:
        (tmp_path / name).write_bytes(b'')
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout='staged')
    monkeypatch.setattr(subprocess, 'run', run)
    if accepted:
        assert _git(tmp_path, {}, 'false', 'add', '--', *names) == 'staged'
        assert len(calls) == 1
        assert calls[0][0][-count:] == list(names)
        assert calls[0][1]['timeout'] == 60
    else:
        with pytest.raises(ValueError, match='300'):
            _git(tmp_path, {}, 'false', 'add', '--', *names)
        assert calls == []


@pytest.mark.parametrize('extra, accepted', [(0, True), (1, False)])
def test_bulk_add_byte_boundary(tmp_path, monkeypatch, extra, accepted):
    # Aggregate limit, not a per-file limit; include every named staging input.
    (tmp_path / '.gitattributes').write_bytes(b'a')
    (tmp_path / 'payload').write_bytes(b'x' * (5 * 1024 * 1024 - 1 + extra))
    calls = []
    def run(argv, **kwargs):
        calls.append(kwargs)
        return subprocess.CompletedProcess(argv, 0, stdout='')
    monkeypatch.setattr(subprocess, 'run', run)
    if accepted:
        _git(tmp_path, {}, 'true', 'add', '--', '.gitattributes', 'payload')
        assert len(calls) == 1
        assert calls[0]['timeout'] == 60
    else:
        with pytest.raises(ValueError, match='5 MiB'):
            _git(tmp_path, {}, 'true', 'add', '--', '.gitattributes', 'payload')
        assert calls == []


@pytest.mark.parametrize('names', [
    (), ('missing',), ('.',), ('..',), ('/absolute',), ('../escape',),
    ('a/../b',), ('a//b',), ('a/./b',), ('a\\b',), ('C:/file',),
    (':(glob)*',), ('*.py',), ('a?',), ('[ab]',), ('nul\x00byte',),
    ('line\nfeed',), ('line\rreturn',), ('a.',), ('a ',), (None,),
    ('file', 'file'), ('file', 'FILE'), ('directory',),
])
def test_bulk_add_rejects_invalid_inputs_before_git(tmp_path, monkeypatch, names):
    (tmp_path / 'file').write_bytes(b'x')
    (tmp_path / 'directory').mkdir()
    def forbidden(*args, **kwargs):
        pytest.fail('invalid staging input launched Git')
    monkeypatch.setattr(subprocess, 'run', forbidden)
    with pytest.raises(ValueError):
        _git(tmp_path, {}, 'false', 'add', '--', *names)


@pytest.mark.parametrize('args', [('add',), ('add', '.'), ('add', '-A')])
def test_bulk_add_rejects_implicit_pathspecs(tmp_path, monkeypatch, args):
    def forbidden(*args, **kwargs):
        pytest.fail('implicit staging launched Git')
    monkeypatch.setattr(subprocess, 'run', forbidden)
    with pytest.raises(ValueError):
        _git(tmp_path, {}, 'false', *args)


def test_bulk_add_rejects_symlink_component_without_launch(tmp_path, monkeypatch):
    (tmp_path / 'parent').mkdir()
    (tmp_path / 'parent/file').write_bytes(b'x')
    original = Path.is_symlink
    monkeypatch.setattr(Path, 'is_symlink',
                        lambda path: path == tmp_path / 'parent' or original(path))
    def forbidden(*args, **kwargs):
        pytest.fail('symlink staging launched Git')
    monkeypatch.setattr(subprocess, 'run', forbidden)
    with pytest.raises(ValueError, match='symlinks'):
        _git(tmp_path, {}, 'false', 'add', '--', 'parent/file')


@pytest.mark.parametrize('output', [None, '', b'', '\n\x00' * 5000, b'\n\x00' * 5000])
def test_diagnostic_tail_handles_empty_text_and_bytes(output):
    rendered = _diagnostic_tail(output)
    assert len(rendered) <= 4096
    assert '\n' not in rendered
    assert '\x00' not in rendered


@pytest.mark.parametrize('autocrlf', ['false', 'true'])
@pytest.mark.parametrize('args, budget', [
    (('init', '--template=', '-q'), 15),
    (('checkout-index', '--all', '--force'), 15),
    (('add', '--', 'file'), 60),
])
def test_git_budget_and_options(tmp_path, monkeypatch, autocrlf, args, budget):
    (tmp_path / 'file').write_bytes(b'x')
    env = {'SENTINEL': 'unchanged'}
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout='result')
    monkeypatch.setattr(subprocess, 'run', run)
    assert _git(tmp_path, env, autocrlf, *args) == 'result'
    assert calls == [([
        'git', '-c', 'core.autocrlf=' + autocrlf,
        '-c', 'core.safecrlf=false', '-c', 'core.attributesFile=' + os.devnull,
        *args,
    ], dict(cwd=tmp_path, env=env, check=True, capture_output=True,
            text=True, encoding='utf-8', timeout=budget))]


@pytest.mark.parametrize('failure', ['timeout', 'nonzero'])
def test_bulk_add_failure_diagnostics_and_no_retry(tmp_path, monkeypatch, capsys, failure):
    (tmp_path / 'file').write_bytes(b'abc')
    (tmp_path / '.git').mkdir()
    lock = tmp_path / '.git/index.lock'
    lock.write_bytes(b'leave failed index alone')
    output = b'prefix' + b'x' * 10000 + b'\nTAIL'
    if failure == 'timeout':
        error = subprocess.TimeoutExpired(['git'], 60, output=output, stderr=output)
    else:
        error = subprocess.CalledProcessError(1, ['git'], output=output, stderr=output)
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        raise error
    monkeypatch.setattr(subprocess, 'run', run)
    ticks = iter([10.0, 70.25])
    monkeypatch.setattr(time, 'monotonic', lambda: next(ticks))
    with pytest.raises(type(error)) as caught:
        _git(tmp_path, {}, 'true', 'add', '--', 'file')
    assert caught.value is error
    assert len(calls) == 1
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3
    assert 'phase=add autocrlf=true files=1 bytes=3 budget=60s elapsed=60.250s' in lines[0]
    assert 'index_lock=True' in lines[0]
    for line, prefix in zip(lines[1:], ('stdout_tail=', 'stderr_tail=')):
        assert line.startswith(prefix)
        assert len(line) <= len(prefix) + 4096
        assert 'prefix' not in line
        assert 'TAIL' in line
    assert lock.read_bytes() == b'leave failed index alone'
    assert (tmp_path / 'file').read_bytes() == b'abc'


def test_diagnostic_failure_preserves_original_error(tmp_path, monkeypatch):
    (tmp_path / 'file').write_bytes(b'x')
    error = subprocess.TimeoutExpired(['git'], 60)
    def run(*args, **kwargs):
        raise error
    def broken_print(*args, **kwargs):
        raise OSError('diagnostic output unavailable')
    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr('builtins.print', broken_print)
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        _git(tmp_path, {}, 'false', 'add', '--', 'file')
    assert caught.value is error


@pytest.mark.parametrize('autocrlf', ['false', 'true'])
def test_roundtrip_stops_before_checkout_when_staging_fails(tmp_path, monkeypatch, autocrlf):
    error = subprocess.TimeoutExpired(['git'], 60)
    phases = []
    def run(argv, **kwargs):
        phases.append(argv[7])
        if argv[7] == 'add':
            raise error
        return subprocess.CompletedProcess(argv, 0, stdout='')
    monkeypatch.setattr(subprocess, 'run', run)
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        test_byte_sealed_checkout_preserves_raw_hashes(tmp_path, autocrlf)
    assert caught.value is error
    assert phases == ['init', 'add']
    assert (tmp_path / 'repo/unprotected.txt').read_bytes() == b'first line\nsecond line\n'
