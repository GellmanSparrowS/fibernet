"""The desktop adapter and public package must coexist in one Python process."""
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / 'FiberScope'


def _check(order):
    source = (
        'import sys; sys.path.insert(0, %r); '
        'from fslab import StructureFactory; import fibernet as fn; '
        'from fibernet.version import __version__; '
        'assert fn.__version__ == __version__; '
        'assert callable(fn.pattern_2d); '
        'assert StructureFactory(unit="square", grid_x=1, grid_y=1).build().num_edges > 0'
    ) % str(APP)
    if order == 'library_first':
        source = 'import fibernet; ' + source
    result = subprocess.run([sys.executable, '-c', source], cwd=ROOT,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_app_first():
    _check('app_first')


def test_library_first():
    _check('library_first')
