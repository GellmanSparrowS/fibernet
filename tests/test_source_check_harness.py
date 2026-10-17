"""Pure controls for child isolation, locale reads and process assertions."""
import json
from pathlib import Path
import sys
import types

import pytest

from _source_check_support import run_source_check
from test_lazy_plot_imports import run_python


HERE = Path(__file__).resolve().parent


def test_source_checks_ignore_contaminated_parent(monkeypatch):
    for name in ('fibernet', 'fibernet.sim', 'taichi'):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    report = json.loads(run_source_check(HERE / 'test_taichi_observer_source.py'))
    assert report['count'] == 31 and report['native_imports'] is False
    run_source_check(HERE / 'test_taichi_event_compat_source.py')


def test_source_checks_without_resource_and_non_utf8_default():
    report = json.loads(run_source_check(
        HERE / 'test_taichi_observer_source.py', '--no-resource', '--cp1252-default'))
    assert report['count'] == 31 and report['RSS_KiB'] is None
    run_source_check(HERE / 'test_taichi_event_compat_source.py', '--cp1252-default')
    run_source_check(HERE / 'test_taichi_api_dispatch_source.py', '--cp1252-default')


@pytest.mark.parametrize('module', ['fibernet', 'taichi'])
def test_child_refuses_native_import_attempt(tmp_path, module):
    script = tmp_path / 'attempt.py'
    script.write_text('import ' + module, encoding='utf-8')
    with pytest.raises(AssertionError, match='forbidden source-check import: ' + module):
        run_source_check(script)


def test_import_process_audit_is_narrow():
    run_python('''
        import sys
        import subprocess
        probes = install_import_process_audit()
        # Audit-event fixtures exercise both approved caller/argv pairs without
        # requiring either platform command to be installed on this host.
        for module, function, argv in [
            ('platform', 'from_subprocess', ['uname', '-p']),
            ('numpy.testing._private.utils', 'check_support_sve', ['lscpu']),
        ]:
            namespace = {'__name__': module, 'sys': sys}
            exec('def ' + function + '(args):\\n    sys.audit("subprocess.Popen", *args)', namespace)
            call = namespace[function]
            for args in [
                (argv[0], argv, '.', None),
                (argv[0], argv, None, {}),
                ('/different/' + argv[0], argv, None, None),
                (argv[0], argv + ['--extra'], None, None),
            ]:
                prior = list(probes)
                try:
                    call(args)
                except AssertionError as exc:
                    assert str(exc) == 'unexpected import process: subprocess.Popen'
                else:
                    raise AssertionError('altered dependency probe was permitted')
                assert probes == prior
            call((argv[0], argv, None, None))
            prior = list(probes)
            try:
                call((argv[0], argv, None, None))
            except AssertionError as exc:
                assert str(exc) == 'unexpected import process: subprocess.Popen'
            else:
                raise AssertionError('repeated dependency probe was permitted')
            assert probes == prior
        assert probes == [('uname', '-p'), ('lscpu',)]
        # Same argv from a direct caller, altered argv, shells and spawn remain
        # rejected. These synthetic events never execute an external command.
        for event, args in [
            ('subprocess.Popen', ('uname', ['uname', '-p'], None, None)),
            ('subprocess.Popen', ('uname', ['uname', '-a'], None, None)),
            ('os.system', ('echo unexpected',)),
            ('os.posix_spawn', ('uname', ['uname', '-p'], {})),
        ]:
            try:
                sys.audit(event, *args)
            except AssertionError as exc:
                assert str(exc) == 'unexpected import process: ' + event
            else:
                raise AssertionError('unexpected process was permitted')
        try:
            subprocess.run([sys.executable, '-c', 'raise SystemExit(91)'], check=True)
        except AssertionError as exc:
            assert str(exc) == 'unexpected import process: subprocess.Popen'
        else:
            raise AssertionError('unapproved real launch was permitted')
    ''')


def test_file_probe_requires_exact_target_environment_and_caller():
    run_python('''
        import os
        import sys
        # Synthetic events also exercise the Unix branch on Windows hosts.
        sys.platform = 'darwin'
        probes = install_import_process_audit()
        import platform
        argv = ['file', '-b', platform._follow_symlinks(sys.executable)]
        env = dict(os.environ, LC_ALL='C')
        namespace = {'__name__': 'platform', 'sys': sys}
        exec('def _syscmd_file(args):\\n    sys.audit("subprocess.Popen", *args)', namespace)
        call = namespace['_syscmd_file']
        def refuse(args, caller=call):
            prior = list(probes)
            try:
                caller(args)
            except AssertionError as exc:
                assert str(exc) == 'unexpected import process: subprocess.Popen'
            else:
                raise AssertionError('altered file probe was permitted')
            assert probes == prior
        for args in [
            ('file', argv, '.', env),
            ('file', argv, None, None),
            ('file', argv, None, dict(env, LC_ALL='en_US.UTF-8')),
            ('file', argv, None, dict(env, EXTRA_PROBE_VALUE='1')),
            ('/different/file', argv, None, env),
            ('file', ['file', '-b', '/another/interpreter'], None, env),
            ('file', argv + ['extra'], None, env),
        ]:
            refuse(args)
        valid = ('file', argv, None, env)
        refuse(valid, lambda args: sys.audit('subprocess.Popen', *args))
        call(valid)
        assert probes == [tuple(argv)]
        refuse(valid)
    ''')


@pytest.mark.parametrize('old_executable', [False, True])
def test_windows_ver_probe_is_exact_once_only(old_executable):
    run_python('''
        import os
        import sys
        sys.platform = 'win32'
        os.environ['SystemRoot'] = 'C:/Windows'
        os.environ['ComSpec'] = 'C:/Windows/System32/cmd.exe'
        probes = install_import_process_audit()
        shell = os.environ['ComSpec']
        argv = shell + ' /c "ver"'
        namespace = {'__name__': 'platform', 'sys': sys}
        exec('def _syscmd_ver(args):\\n    sys.audit("subprocess.Popen", *args)', namespace)
        call = namespace['_syscmd_ver']
        def refuse(args, caller=call):
            prior = list(probes)
            try:
                caller(args)
            except AssertionError as exc:
                assert str(exc) == 'unexpected import process: subprocess.Popen'
            else:
                raise AssertionError('altered shell probe was permitted')
            assert probes == prior
        for args in [
            (shell, argv, '.', None),
            (shell, argv, None, {}),
            ('C:/other/cmd.exe', argv, None, None),
            (shell, shell + ' /c "ver & echo changed"', None, None),
            (shell, shell + ' /c "command /c ver"', None, None),
            (None, 'cmd.exe /c "ver"', None, None),
            (shell, [argv], None, None),
        ]:
            refuse(args)
        valid = (None if OLD_EXECUTABLE else shell, argv, None, None)
        refuse(valid, lambda args: sys.audit('subprocess.Popen', *args))
        call(valid)
        assert probes == [('windows-ver',)]
        refuse(valid)
        refuse((shell if OLD_EXECUTABLE else None, argv, None, None))
    '''.replace('OLD_EXECUTABLE', repr(old_executable)))


def test_windows_ver_probe_refuses_non_system_comspec():
    run_python('''
        import os
        import sys
        sys.platform = 'win32'
        os.environ['SystemRoot'] = 'C:/Windows'
        os.environ['ComSpec'] = 'C:/other/cmd.exe'
        probes = install_import_process_audit()
        namespace = {'__name__': 'platform', 'sys': sys}
        exec('def _syscmd_ver(args):\\n    sys.audit("subprocess.Popen", *args)', namespace)
        try:
            namespace['_syscmd_ver'](('C:/other/cmd.exe', 'C:/other/cmd.exe /c "ver"', None, None))
        except AssertionError as exc:
            assert str(exc) == 'unexpected import process: subprocess.Popen'
        else:
            raise AssertionError('non-system COMSPEC was permitted')
        assert probes == []
    ''')
