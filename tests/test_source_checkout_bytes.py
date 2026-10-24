"""Byte-sealed sources keep their Git bytes even in CRLF-oriented checkouts."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
ROUNDTRIP_PROTECTED = (
    'fibernet/core/structure_graph.py',
    'fibernet/__init__.py',
    'fibernet/sim/__init__.py',
    'fibernet/sim/saved_temperature.py',
    'fibernet/sim/thermoelastic.py',
    'fibernet/sim/_thermoelastic.py',
    'tests/fixtures/optional_facades_v2/sim_thermoelastic.py.txt',
    'tests/fixtures/optional_facades_v2/sim_lammps_capabilities_delta.json',
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
    (*ROUNDTRIP_PROTECTED, *('fibernet/' + name for name in CORE_MANIFEST))
))
CHECKOUT_ATTRIBUTES = {
    'text': 'unset',
    'filter': 'unspecified',
    'working-tree-encoding': 'unspecified',
}


def test_checkout_inventory_covers_the_native_core_manifest():
    core = ROOT / 'fibernet'
    assert set(CORE_MANIFEST) == {
        path.relative_to(core).as_posix() for path in core.rglob('*.py')
    }
    for name, expected in CORE_MANIFEST.items():
        assert hashlib.sha256((core / name).read_bytes()).hexdigest() == expected


def _checkout_git(repo, autocrlf):
    repo.mkdir()
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith('GIT_')}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull,
               GIT_CONFIG_NOSYSTEM='1', GIT_ATTR_NOSYSTEM='1')
    def git(*args, input=None):
        return subprocess.run(
            ['git', '-c', 'core.autocrlf=' + autocrlf,
             '-c', 'core.safecrlf=false', '-c', 'core.attributesFile=' + os.devnull,
             *args], cwd=repo, env=env,
            input=input, check=True, capture_output=True, text=True,
            encoding='utf-8', timeout=15,
        ).stdout
    git('init', '--template=', '-q')
    (repo / '.gitattributes').write_bytes((ROOT / '.gitattributes').read_bytes())
    return git


def _assert_protected_checkout_attributes(git):
    # Git resolves the full inventory without copying or staging every source.
    # NUL-delimited input/output avoids quoting and command-line size limits.
    output = git('check-attr', '-z', '--stdin', *CHECKOUT_ATTRIBUTES,
                 input='\0'.join(PROTECTED) + '\0')
    assert output.endswith('\0'), 'Git attribute report is not NUL-terminated'
    fields = output[:-1].split('\0')
    assert len(fields) % 3 == 0, 'Git attribute report has an incomplete record'
    observed = {(path, attribute): value for path, attribute, value in
                zip(fields[::3], fields[1::3], fields[2::3])}
    expected_keys = {(path, attribute) for path in PROTECTED
                     for attribute in CHECKOUT_ATTRIBUTES}
    assert len(fields) == 3 * len(observed), 'Git attribute report repeats a record'
    assert set(observed) == expected_keys, 'Git attribute report omits or adds a path/attribute'
    for path in PROTECTED:
        for attribute, expected in CHECKOUT_ATTRIBUTES.items():
            actual = observed[path, attribute]
            assert actual == expected, (
                f'{path}: {attribute}={actual}; expected {expected}'
            )


@pytest.mark.parametrize('autocrlf', ['false', 'true'])
def test_byte_sealed_checkout_preserves_raw_hashes(tmp_path, autocrlf):
    repo = tmp_path / 'repo'
    git = _checkout_git(repo, autocrlf)
    _assert_protected_checkout_attributes(git)
    # Exhaustive attribute checks plus a bounded real-byte round trip replace
    # copying the growing native-core inventory into two disposable repositories.
    # The separate inventory test still verifies every native-core source hash.
    expected = {}
    for name in ROUNDTRIP_PROTECTED:
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


@pytest.mark.parametrize('autocrlf', ['false', 'true'])
@pytest.mark.parametrize('attribute_newline', [b'\n', b'\r\n'], ids=['lf-attributes', 'crlf-attributes'])
@pytest.mark.parametrize('mutation, attribute, value', [
    ('remove-recursive-rule', 'text', 'unspecified'),
    ('override-text', 'text', 'set'),
    ('override-filter', 'filter', 'checkout_probe'),
    ('override-encoding', 'working-tree-encoding', 'UTF-16LE'),
])
def test_checkout_attribute_guard_rejects_manifest_only_mutations(
        tmp_path, autocrlf, attribute_newline, mutation, attribute, value):
    repo = tmp_path / 'repo'
    git = _checkout_git(repo, autocrlf)
    # This path is deliberately absent from the bounded byte round trip.
    target = next(path for path in PROTECTED if path not in ROUNDTRIP_PROTECTED)
    assert target.removeprefix('fibernet/') in CORE_MANIFEST
    attributes = repo / '.gitattributes'
    original = attributes.read_bytes().replace(b'\r\n', b'\n')
    original = original.replace(b'\n', attribute_newline)
    if mutation == 'remove-recursive-rule':
        lines = original.splitlines(keepends=True)
        rule = b'fibernet/**/*.py -text'
        assert sum(line.rstrip(b'\r\n') == rule for line in lines) == 1
        changed = b''.join(line for line in lines if line.rstrip(b'\r\n') != rule)
    else:
        setting = attribute if value == 'set' else f'{attribute}={value}'
        changed = original + f'\n{target} {setting}\n'.encode('utf-8')
    attributes.write_bytes(changed)
    with pytest.raises(AssertionError) as error:
        _assert_protected_checkout_attributes(git)
    assert str(error.value).splitlines()[0] == (
        f'{target}: {attribute}={value}; expected {CHECKOUT_ATTRIBUTES[attribute]}'
    )
