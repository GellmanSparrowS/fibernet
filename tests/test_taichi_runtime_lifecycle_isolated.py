"""Opt-in subprocess acceptance tests for the runtime ownership contract.

Run using the real optional CPU interpreter:
FIBERNET_RUN_LIFECYCLE_ACCEPTANCE=1 python -m pytest -q \
    tests/test_taichi_runtime_lifecycle_isolated.py
Every native-runtime case executes in its own hard-bounded subprocess.
"""
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get('FIBERNET_RUN_LIFECYCLE_ACCEPTANCE') != '1',
    reason='Native-runtime acceptance cases require explicit opt in',
)

CASE = r'''
import threading
import sys
import numpy as np
import taichi as ti
from taichi.lang import impl
from fibernet.core.structure_graph import StructureGraph
from fibernet.sim.accelerated import TaichiEngine

def graph(n):
    g = StructureGraph(dimension=3)
    ids = [g.add_node(np.array([float(i),0.,0.])) for i in range(n)]
    for x,y in zip(ids,ids[1:]): g.add_edge(x,y)
    return g

def dynamics(e,n=3):
    f = np.zeros((n,3)); f[-1,0] = 1.
    return e.dynamics(graph(n), fixed_nodes=[0], external_force=f,
        stiffness=10., dt=.001, num_steps=3, save_interval=1,
        dashpot=0., drag=0.).deformed_positions.copy()

def forces(e):
    return e.compute_forces(np.array([[0.,0.,0.],[1.2,0.,0.],[2.4,0.,0.]]),
        np.ones(2), np.full(2,10.), np.array([[0,1],[1,2]])).copy()

def same(a,b): np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-14)
def reject(call):
    try: call()
    except RuntimeError as exc:
        assert str(exc), 'Failure must explain the policy violation'
    else: raise AssertionError('Expected explicit RuntimeError')

scenario = sys.argv[1]
if scenario == 'version_guard':
    import fibernet.sim.accelerated as accelerated
    ti.__version__=(1,6,0)
    assert accelerated.SimResult(energy=2.).energy == 2.
    try: TaichiEngine(arch='cpu',num_threads=2)
    except RuntimeError as exc: assert 'taichi==1.7.4' in str(exc)
    else: raise AssertionError('Unsupported version must fail before native init')
    assert impl.get_runtime().prog is None
    print('PASS '+scenario,flush=True);sys.exit(0)
if scenario == 'unknown_external':
    ti.init(arch=ti.cpu,cpu_max_num_threads=2)
    runtime=impl.get_runtime()
    reject(lambda: TaichiEngine(arch='cpu',num_threads=2))
    assert impl.get_runtime() is runtime
    ti.reset()
    a=TaichiEngine(arch='cpu',num_threads=2);dynamics(a);forces(a)
    print('PASS '+scenario,flush=True);sys.exit(0)
if scenario == 'worker_owned':
    errors=[]
    owned_engines=[]
    def owned():
        try:
            e=TaichiEngine(arch='cpu',num_threads=2)
            owned_engines.append(e)
            x=dynamics(e); f=forces(e)
            same(x,dynamics(e)); same(f,forces(e))
            b=TaichiEngine(arch='cpu',num_threads=2)
            dynamics(b,4); same(x,dynamics(e))
        except BaseException as exc: errors.append(exc)
    t=threading.Thread(target=owned);t.start();t.join(20)
    assert not t.is_alive() and not errors, errors
    # Ownership survives thread exit; a new GUI worker is not an automatic transfer.
    reject(lambda: dynamics(owned_engines[0]))
    def replacement():
        try: reject(lambda: TaichiEngine(arch='cpu',num_threads=2))
        except BaseException as exc: errors.append(exc)
    t2=threading.Thread(target=replacement);t2.start();t2.join(10)
    assert not t2.is_alive() and not errors, errors
    print('PASS '+scenario,flush=True)
    sys.exit(0)
a = TaichiEngine(arch='cpu',num_threads=2)
x = dynamics(a); f = forces(a)
runtime = impl.get_runtime()
if scenario == 'same_size':
    b = TaichiEngine(arch='cpu',num_threads=2)
    assert impl.get_runtime() is runtime, 'Same backend must not reinitialize'
    same(x,dynamics(b)); same(f,forces(b)); same(x,dynamics(a)); same(f,forces(a))
elif scenario == 'different_size':
    b = TaichiEngine(arch='cpu',num_threads=2)
    assert impl.get_runtime() is runtime
    y = dynamics(b,4); same(x,dynamics(a)); same(y,dynamics(b,4)); same(f,forces(a))
elif scenario in ('reset','external_init'):
    old_dynamics = dict(TaichiEngine._field_cache)
    old_forces = dict(TaichiEngine._compute_forces_cache)
    if scenario == 'reset': ti.reset()
    else: ti.init(arch=ti.cpu,cpu_max_num_threads=2)
    assert impl.get_runtime() is not runtime
    if scenario == 'external_init':
        reject(lambda: dynamics(a))
        reject(lambda: forces(a))
        assert not TaichiEngine._field_cache and not TaichiEngine._compute_forces_cache
        ti.reset()
    same(x,dynamics(a)); same(f,forces(a))
    assert all(v is not old_dynamics.get(k) for k,v in TaichiEngine._field_cache.items())
    assert all(v is not old_forces.get(k) for k,v in TaichiEngine._compute_forces_cache.items())
elif scenario == 'architecture_mismatch':
    reject(lambda: TaichiEngine(arch='gpu',num_threads=2))
    assert impl.get_runtime() is runtime, 'Reject before runtime reset'
    same(x,dynamics(a)); same(f,forces(a))
elif scenario == 'thread_configuration':
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        b=TaichiEngine(arch='cpu',num_threads=3)
    assert any('thread count' in str(w.message) for w in caught)
    assert impl.get_runtime() is runtime and ti.cfg.cpu_max_num_threads == 2
    same(x,dynamics(b))
elif scenario == 'thread_rejection':
    errors=[]
    def worker():
        try:
            reject(lambda: TaichiEngine(arch='cpu',num_threads=2))
            reject(lambda: dynamics(a))
            reject(lambda: forces(a))
            reject(lambda: a.stretch_test(graph(3),num_steps=3))
            reject(TaichiEngine.clear_field_cache)
        except BaseException as exc: errors.append(exc)
    t=threading.Thread(target=worker); t.start(); t.join(10)
    assert not t.is_alive(), 'Worker did not fail promptly'
    assert not errors, errors
    assert impl.get_runtime() is runtime
    same(x,dynamics(a)); same(f,forces(a))
else: raise AssertionError(scenario)
print('PASS '+scenario,flush=True)
'''


@pytest.mark.parametrize('scenario', [
    'same_size', 'different_size', 'reset', 'external_init',
    'architecture_mismatch', 'thread_rejection', 'worker_owned',
    'unknown_external', 'thread_configuration', 'version_guard',
])
def test_lifecycle_policy_in_isolated_process(scenario, tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, 'TI_DISABLE_VERSION_CHECK': '1',
           'TI_SKIP_VERSION_CHECK': 'ON', 'OMP_NUM_THREADS': '2',
           'PYTHONFAULTHANDLER': '1',
           'TI_OFFLINE_CACHE_FILE_PATH': str(tmp_path / 'taichi'),
           'MPLCONFIGDIR': str(tmp_path / 'matplotlib'),
           'XDG_CACHE_HOME': str(tmp_path / 'xdg'),
           'PYTHONPATH': str(root) + os.pathsep + os.environ.get('PYTHONPATH', '')}
    for key in ('TI_ARCH',):
        env.pop(key, None)
    result = subprocess.run([sys.executable, '-u', '-c', CASE, scenario],
                            cwd=root, env=env, capture_output=True, text=True,
                            timeout=60)
    assert result.returncode == 0, (
        f'{scenario}: return code {result.returncode}\n'
        f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}'
    )
    assert 'PASS ' + scenario in result.stdout
