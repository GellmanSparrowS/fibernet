"""Fresh-child controls for source-only checks; no production imports."""
import importlib.abc
import os
from pathlib import Path
import runpy
import subprocess
import sys


def run_source_check(path, *controls):
    result = subprocess.run(
        [sys.executable, '-B', __file__, str(path), *controls],
        cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
        capture_output=True, text=True, encoding='utf-8', timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def install_import_process_audit():
    """Permit only two known dependency CPU queries, with caller provenance.

    This is an import regression assertion, not a sandbox. NumPy 2.0's
    check_support_sve runs lscpu; CPython platform._Processor runs uname -p.
    Import blockers must be installed before this hook or any dependency.
    """
    probes = []
    def audit(event, args):
        if event not in {'subprocess.Popen', 'os.system', 'os.posix_spawn'}:
            return
        frame = sys._getframe(1)
        while frame and frame.f_globals.get('__name__') == 'subprocess':
            frame = frame.f_back
        caller = (frame.f_globals.get('__name__'), frame.f_code.co_name) if frame else None
        allowed = {
            ('uname', '-p'): ('platform', 'from_subprocess'),
            ('lscpu',): ('numpy.testing._private.utils', 'check_support_sve'),
        }
        if event == 'subprocess.Popen':
            executable, argv, cwd, env = args
            command = tuple(argv) if isinstance(argv, (list, tuple)) else (argv,)
            if (command in allowed and caller == allowed[command]
                    and executable == command[0] and cwd is None and env is None
                    and command not in probes):
                probes.append(command)
                return
        raise AssertionError('unexpected import process: ' + event)
    sys.addaudithook(audit)
    return probes


def _main():
    path = Path(sys.argv[1]).resolve()
    controls = set(sys.argv[2:])
    class BlockNative(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in {'fibernet', 'taichi'}:
                raise AssertionError('forbidden source-check import: ' + fullname)
            if '--no-resource' in controls and fullname == 'resource':
                raise ModuleNotFoundError('simulated unavailable resource', name='resource')
    sys.meta_path.insert(0, BlockNative())
    assert not any(n.split('.')[0] in {'fibernet', 'taichi'} for n in sys.modules)
    if '--cp1252-default' in controls:
        original = Path.read_text
        def read_text(self, encoding=None, errors=None):
            return original(self, encoding=encoding or 'cp1252', errors=errors)
        Path.read_text = read_text
    sys._fibernet_source_check_child = True
    sys.argv = [str(path)]
    runpy.run_path(str(path), run_name='__main__')


if __name__ == '__main__':
    _main()
