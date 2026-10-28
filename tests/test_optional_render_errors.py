"""Provider-free regressions for explicit rendering failures and import hygiene."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]


class OptionalUXTests(unittest.TestCase):
    def run_fresh(self, code):
        with tempfile.TemporaryDirectory() as folder:
            env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1')
            result = subprocess.run([sys.executable, '-B', '-c', textwrap.dedent(code)],
                                    cwd=folder, env=env, capture_output=True,
                                    text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(result.stdout, '')
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_missing_pyvista_is_actionable_without_output_or_provider(self):
        self.run_fresh('''
            import importlib.abc, sys
            attempts = []
            class Block(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == 'pyvista':
                        attempts.append(fullname)
                        raise ModuleNotFoundError('blocked optional dependency', name=fullname)
                    if fullname.split('.')[0] in {'taichi', 'exudyn', 'vtk', 'vtkmodules'}:
                        raise AssertionError('unexpected native import: ' + fullname)
            sys.meta_path.insert(0, Block())
            import fibernet as fn
            fn.pattern_2d(unit='square', box=(1, 1), grid=(1, 1))
            from fibernet.viz import render3d
            assert not attempts, 'core or renderer import tried PyVista'
            for call in (lambda: render3d.render_network_3d(None),
                         lambda: render3d.render_deformation(None, None),
                         lambda: render3d.render_structure_graph_3d(None)):
                try:
                    call()
                except ImportError as exc:
                    assert "python -m pip install 'fibernet[viz]'" in str(exc)
                    assert isinstance(exc.__cause__, ModuleNotFoundError)
                    assert exc.__cause__.name == 'pyvista'
                else:
                    raise AssertionError('missing renderer returned success/None')
            assert len(attempts) == 3
        ''')

    def test_broken_optional_installation_error_is_not_relabelled(self):
        self.run_fresh('''
            import builtins
            from fibernet.viz import render3d
            original = builtins.__import__
            for error in (ModuleNotFoundError('missing internal VTK', name='vtkmodules'),
                          ImportError('native binary ABI mismatch')):
                def broken(name, *args, **kwargs):
                    if name == 'pyvista':
                        raise error
                    return original(name, *args, **kwargs)
                builtins.__import__ = broken
                try:
                    render3d.render_structure_graph_3d(None)
                except ImportError as exc:
                    assert exc is error
                    assert 'fibernet[viz]' not in str(exc)
                else:
                    raise AssertionError('broken installation was masked')
                finally:
                    builtins.__import__ = original
        ''')

    def test_root_import_does_not_install_global_futurewarning_ignore(self):
        self.run_fresh('''
            import warnings
            import numpy, scipy
            warnings.filterwarnings('error', category=FutureWarning,
                                    message='fibernet-user-policy-witness')
            import fibernet
            try:
                warnings.warn('fibernet-user-policy-witness', FutureWarning)
            except FutureWarning:
                pass
            else:
                raise AssertionError('import disabled the caller FutureWarning policy')
        ''')

    def test_ml_utils_reload_preserves_warning_filters(self):
        self.run_fresh('''
            import warnings, importlib
            import fibernet.ml.utils as utils
            warnings.filterwarnings('error', category=FutureWarning,
                                    message='fibernet-user-policy-witness')
            before = list(warnings.filters)
            importlib.reload(utils)
            assert warnings.filters == before
        ''')


if __name__ == '__main__':
    unittest.main()
