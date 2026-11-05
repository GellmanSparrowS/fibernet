"""Synthetic L3 result contracts run without contaminating other test imports."""
import ast
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/l3_result_v1/contract.py'


def test_l3_result_synthetic_contract_in_isolated_child():
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    completed = subprocess.run(
        [sys.executable, '-I', '-B', str(FIXTURE)], cwd=ROOT,
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert 'Ran 50 tests' in completed.stderr
    assert 'OK' in completed.stderr


def test_l3_result_python39_grammar_and_provider_free_imports():
    source = ROOT / 'fibernet/sim/l3_result.py'
    tree = ast.parse(source.read_text(encoding='utf-8'), feature_version=(3, 9))
    ast.parse(FIXTURE.read_text(encoding='utf-8'), feature_version=(3, 9))
    forbidden = ('l3_backend', '_l3_worker', '_execution_boundary', '_execution_owner',
                 'lightweight_cv05', 'run_case', 'subprocess', 'socket', 'pathlib')
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [value.name for value in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or '']
        else:
            continue
        assert not any(name.split('.')[-1] in forbidden for name in names)
