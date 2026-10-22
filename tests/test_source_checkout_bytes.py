"""Byte-sealed sources keep their Git bytes even in CRLF-oriented checkouts."""
import hashlib
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROTECTED = (
    'fibernet/__init__.py',
    'fibernet/sim/__init__.py',
    'fibernet/sim/saved_temperature.py',
    'tests/fixtures/optional_facades_v2/root.py.txt',
    'tests/fixtures/optional_facades_v2/sim.py.txt',
    'tests/fixtures/optional_facades_v2/sim_nodal_controls.py.txt',
    'tests/fixtures/optional_facades_v2/sim_prior.py.txt',
    'adapters/mcp/tests/fixtures/provenance_v1_outputs.json',
    'fibernet/easy.py',
    'tests/fixtures/easy_rl_failure_v1/easy.py.txt',
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


@pytest.mark.parametrize('autocrlf', ['false', 'true'])
def test_byte_sealed_checkout_preserves_raw_hashes(tmp_path, autocrlf):
    repo = tmp_path / 'repo'
    repo.mkdir()
    def git(*args):
        return subprocess.run(
            ['git', '-c', 'core.autocrlf=' + autocrlf,
             '-c', 'core.safecrlf=false', *args], cwd=repo,
            check=True, capture_output=True, text=True, encoding='utf-8', timeout=15,
        ).stdout
    git('init', '-q')
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
    git('add', '--', '.gitattributes', 'unprotected.txt', *PROTECTED)
    # Remove only these disposable working copies, then materialize from index.
    for name in (*PROTECTED, 'unprotected.txt'):
        (repo / name).unlink()
    git('checkout-index', '--all', '--force')
    for name in PROTECTED:
        raw = (repo / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected[name]
        # Byte checks are intentionally not whitespace/line-ending normalizers.
        assert hashlib.sha256(raw + b'\n').hexdigest() != expected[name]
        assert hashlib.sha256(raw.replace(b'\n', b'\r\n')).hexdigest() != expected[name]
    materialized = (repo / 'unprotected.txt').read_bytes()
    assert materialized == (control.replace(b'\n', b'\r\n') if autocrlf == 'true' else control)
