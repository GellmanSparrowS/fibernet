"""Fresh-child controls for source-only checks; no production imports."""
import importlib.abc
import os
import ntpath
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
    """Permit four exact dependency/platform queries, once per caller.

    This is an import regression assertion, not a sandbox. NumPy 2.0 probes
    SVE with lscpu. CPython platform uses uname -p, file -b on the resolved
    interpreter, and (on Windows) the system command interpreter's ver builtin.
    Import blockers must be installed before this hook or any dependency.
    """
    probes = []
    # Match platform._follow_symlinks: resolve the file's links, not parent dirs.
    interpreter = os.path.abspath(sys.executable)
    while os.path.islink(interpreter):
        interpreter = os.path.normpath(os.path.join(
            os.path.dirname(interpreter), os.readlink(interpreter)))
    file_command = ('file', '-b', interpreter)
    file_environment = dict(os.environ, LC_ALL='C')
    windows_command = None
    windows_executable = None
    if sys.platform == 'win32':
        root = os.environ.get('SystemRoot', '')
        comspec = os.environ.get('ComSpec', '')
        expected = ntpath.join(root, 'System32', 'cmd.exe')
        # Do not allow an arbitrary COMSPEC override, PATH search or shell text.
        if (ntpath.isabs(root) and ntpath.isabs(comspec)
                and ntpath.normcase(ntpath.normpath(comspec))
                == ntpath.normcase(ntpath.normpath(expected))):
            windows_executable = comspec
            windows_command = comspec + ' /c "ver"'

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
        probe = None
        if event == 'subprocess.Popen':
            executable, argv, cwd, env = args
            command = tuple(argv) if isinstance(argv, (list, tuple)) else (argv,)
            if (command in allowed and caller == allowed[command]
                    and executable == command[0] and cwd is None and env is None):
                probe = command
            elif (sys.platform != 'win32' and command == file_command
                    and caller == ('platform', '_syscmd_file')
                    and executable == 'file' and cwd is None
                    and env == file_environment):
                probe = file_command
            elif (windows_command is not None and argv == windows_command
                    and caller == ('platform', '_syscmd_ver')
                    and executable in (None, windows_executable)
                    and cwd is None and env is None):
                # Older CPython leaves executable=None; newer versions supply
                # the same absolute COMSPEC. Both represent this one query.
                probe = ('windows-ver',)
            if probe is not None and probe not in probes:
                probes.append(probe)
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
