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
