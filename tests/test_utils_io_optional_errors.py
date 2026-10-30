"""Exercise the exact export function bodies with inert PyVista substitutes."""
import ast
import builtins
from pathlib import Path
from types import SimpleNamespace
import pytest
SOURCE = Path(__file__).resolve().parents[1] / 'fibernet/utils/io.py'

def exporter(name, provider=None, error=None):
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = {'FiberNetwork': object}
    def imported(module, *args, **kwargs):
        assert module == 'pyvista'
        if error is not None:
            raise error
        return provider
    scope['__builtins__'] = dict(vars(builtins), __import__=imported)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), scope)
    return scope[name]

@pytest.mark.parametrize('name', ['export_vtk', 'export_stl'])
@pytest.mark.parametrize('kind', ['absent', 'broken', 'nested', 'downstream', 'success'])
def test_optional_export_boundaries(name, kind, tmp_path):
    target = tmp_path / 'output.vtk'
    network = SimpleNamespace(fibers=[SimpleNamespace(centerline=[[0,0,0],[1,0,0]], radius=0.2)])
    trace=[]
    class Mesh:
        def tube(self, **kwargs):
            trace.append(('tube', kwargs)); return self
        def save(self, path):
            trace.append(('save', path))
    failure = ImportError('original downstream sentinel')
    def spline(points, **kwargs):
        trace.append(('spline', points, kwargs))
        if kind == 'downstream': raise failure
        return Mesh()
    provider = SimpleNamespace(Spline=spline)
    error = {'absent': ModuleNotFoundError('missing pyvista', name='pyvista'),
             'broken': ImportError('broken provider sentinel'),
             'nested': ModuleNotFoundError('missing vtk dependency', name='vtk')}.get(kind)
    fn=exporter(name, provider, error)
    if kind == 'success':
        assert fn(network, str(target)) is None
        assert trace == [('spline', network.fibers[0].centerline, {'n_points': 10}),
                         ('tube', {'radius':0.2, 'capping':True}), ('save',str(target))]
    else:
        with pytest.raises(ImportError) as caught: fn(network,str(target))
        if kind == 'absent':
            assert "fibernet[viz]" in str(caught.value)
            assert caught.value.__cause__ is error
        else:
            assert caught.value is (failure if kind == 'downstream' else error)
        assert not target.exists()
        if kind != 'downstream': assert trace == []
