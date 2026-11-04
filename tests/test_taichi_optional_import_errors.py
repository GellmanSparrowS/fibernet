"""Provider-free checks: distinguish absent Taichi from broken installations."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = r'''
import importlib.abc
import sys

kind, entry = sys.argv[1:]
sentinel = {
    'absent': ModuleNotFoundError('absent sentinel', name='taichi'),
    'broken_import': ImportError('ABI sentinel', name='taichi'),
    'broken_nested': ModuleNotFoundError('nested sentinel', name='taichi._lib.core'),
    'broken_runtime': RuntimeError('runtime sentinel'),
}[kind]
attempts = []

class BlockTaichi(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'taichi':
            attempts.append(fullname)
            raise sentinel

assert not any(n.split('.')[0] == 'taichi' for n in sys.modules)
sys.meta_path.insert(0, BlockTaichi())
import fibernet as fn
graph = fn.StructureGraph(dimension=2)
a = graph.add_node([0., 0., 0.], merge=False)
b = graph.add_node([1., 0., 0.], merge=False)
graph.add_edge(a, b, radius=.05, segments=1)
simulate = fn.simulate
assert attempts == []

def invoke():
    if entry == 'alias':
        return fn.TaichiEngine
    return simulate(graph, backend='spring', num_steps=1)

if kind == 'absent':
    if entry == 'alias':
        engine = invoke()
        from fibernet.sim.accelerated import TaichiEngine, SimResult
        assert engine is TaichiEngine
        assert fn.SimResult is SimResult
        call = engine
    else:
        call = invoke
    try:
        call()
    except ImportError as error:
        assert str(error) == 'Taichi required: pip install taichi==1.7.4'
    else:
        raise AssertionError('absent provider did not raise installation error')
    assert attempts == ['taichi']
else:
    for _ in range(2):
        try:
            invoke()
        except BaseException as error:
            assert error is sentinel, (type(error), str(error))
        else:
            raise AssertionError('broken provider was suppressed')
        assert 'TaichiEngine' not in vars(fn)
        assert 'fibernet.sim.accelerated' not in sys.modules
    assert attempts == ['taichi', 'taichi']
assert not any(n.split('.')[0] == 'taichi' for n in sys.modules)
'''


@pytest.mark.parametrize('kind', [
    'absent', 'broken_import', 'broken_nested', 'broken_runtime',
])
@pytest.mark.parametrize('entry', ['alias', 'simulate'])
def test_taichi_optional_import_boundary(kind, entry, tmp_path):
    spec = importlib.util.find_spec('fibernet')
    assert spec is not None and spec.origin is not None
    package_parent = str(Path(spec.origin).resolve().parent.parent)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH'] = os.pathsep.join([package_parent, env.get('PYTHONPATH', '')])
    result = subprocess.run(
        [sys.executable, '-B', '-c', SCRIPT, kind, entry],
        cwd=tmp_path, env=env, capture_output=True, text=True,
        encoding='utf-8', timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
