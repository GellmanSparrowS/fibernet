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


def test_l3_result_platform_bootstrap_keeps_audit_boundary():
    # Model the older Windows stdlib's one-time version query without spawning
    # a process. Exercise it again on NumPy import, after the audit is installed.
    code = r'''
import builtins
import platform
import runpy
import socket
import subprocess
import sys

real_uname = platform.uname
real_import = builtins.__import__
real_add_hook = sys.addaudithook
real_run = subprocess.run
cached = None
queries = 0
cpu_queries = 0
audit_installed = False
first_fibernet = False

def windows_uname():
    global cached, queries
    if cached is None:
        sys.audit('subprocess.Popen', 'synthetic-platform-query', [], None, None)
        cached = real_uname()
        queries += 1
    return cached

def add_hook(hook):
    global audit_installed
    real_add_hook(hook)
    audit_installed = True

def checked_import(name, *args, **kwargs):
    global first_fibernet, cpu_queries
    if name == 'numpy':
        assert audit_installed, 'NumPy must stay inside the audit boundary'
        platform.machine()
    if name == 'numpy.testing' and cpu_queries == 0:
        assert audit_installed, 'NumPy testing must stay inside the audit boundary'
        try:
            subprocess.run('synthetic-cpu-query', capture_output=True, text=True)
        except FileNotFoundError:
            cpu_queries += 1
        else:
            raise AssertionError('CPU discovery must use the missing-command fallback')
    if name == 'fibernet' or name.startswith('fibernet.'):
        assert audit_installed, 'FiberNet must stay inside the audit boundary'
        assert subprocess.run is real_run, 'CPU-discovery patch must end before FiberNet import'
        first_fibernet = True
    return real_import(name, *args, **kwargs)

platform.uname = windows_uname
sys.addaudithook = add_hook
builtins.__import__ = checked_import
namespace = runpy.run_path(sys.argv[1])
assert queries == 1 and first_fibernet and audit_installed
assert cpu_queries == 1 and subprocess.run is real_run
assert namespace['PROHIBITED'] == []

def refused(event, call):
    before = list(namespace['PROHIBITED'])
    try:
        call()
    except RuntimeError as exc:
        assert str(exc) == 'No provider, process or network in projection tests'
    else:
        raise AssertionError('Expected audit refusal: ' + event)
    assert namespace['PROHIBITED'] == before + [event]

refused('subprocess.Popen', lambda: subprocess.run([sys.executable, '-c', 'pass'], check=True))
refused('socket.__new__', socket.socket)
refused('import', lambda: __import__('run_case'))
print('platform bootstrap and audit boundary passed')
'''
    completed = subprocess.run(
        [sys.executable, '-I', '-B', '-c', code, str(FIXTURE)], cwd=ROOT,
        env=dict(os.environ, OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1'),
        capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert 'platform bootstrap and audit boundary passed' in completed.stdout
