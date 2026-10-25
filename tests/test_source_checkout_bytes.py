"""Byte-sealed sources keep their Git bytes even in CRLF-oriented checkouts."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROTECTED = (
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
        return subprocess.run(
            ['git', '-c', 'core.autocrlf=' + autocrlf,
             '-c', 'core.safecrlf=false', '-c', 'core.attributesFile=' + os.devnull,
             *args], cwd=repo, env=env,
            check=True, capture_output=True, text=True, encoding='utf-8', timeout=15,
        ).stdout
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
