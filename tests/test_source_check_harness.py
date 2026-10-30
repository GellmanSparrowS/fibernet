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
    assert report['count'] == 37 and report['native_imports'] is False
    run_source_check(HERE / 'test_taichi_event_compat_source.py')


def test_source_checks_without_resource_and_non_utf8_default():
    report = json.loads(run_source_check(
        HERE / 'test_taichi_observer_source.py', '--no-resource', '--cp1252-default'))
    assert report['count'] == 37 and report['RSS_KiB'] is None
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


@pytest.mark.parametrize('separator', ['/', chr(92)])
@pytest.mark.parametrize('version,old_executable', [('3.9.13', True), ('3.10.11', False), ('3.11.9', False)])
def test_windows_ver_probe_is_exact_once_per_ancestry(version, old_executable, separator):
    run_python('''
        import os
        import sys
        sys.platform = 'win32'
        os.environ['SystemRoot'] = 'C:' + SEPARATOR + 'Windows'
        os.environ['ComSpec'] = os.environ['SystemRoot'] + SEPARATOR + 'System32' + SEPARATOR + 'cmd.exe'
        probes = install_import_process_audit()
        shell = os.environ['ComSpec']
        argv = shell + ' /c "ver"'
        namespace = {'__name__': 'platform', 'sys': sys}
        exec('def _syscmd_ver(args):\\n    sys.audit("subprocess.Popen", *args)\\ndef win32_ver(args):\\n    return _syscmd_ver(args)\\ndef uname(args):\\n    return win32_ver(args)\\ndef platform(args):\\n    return win32_ver(args)', namespace)
        valid = (None if OLD_EXECUTABLE else shell, argv, None, None)
        def refuse(args, call):
            prior = list(probes)
            try:
                call(args)
            except AssertionError as exc:
                assert str(exc) == 'unexpected import process: subprocess.Popen'
            else:
                raise AssertionError('altered shell probe was permitted')
            assert probes == prior
        for wrong in ('_syscmd_ver', 'win32_ver'):
            refuse(valid, namespace[wrong])
        refuse(valid, lambda args: sys.audit('subprocess.Popen', *args))
        for entry in ('uname', 'platform'):
            call = namespace[entry]
            for args in [
                (shell, argv, '.', None),
                (shell, argv, None, {}),
                ('C:/other/cmd.exe', argv, None, None),
                (shell, shell + ' /c "ver & echo changed"', None, None),
                (shell, shell + ' /c "command /c ver"', None, None),
                (None, 'cmd.exe /c "ver"', None, None),
                (shell, [argv], None, None),
            ]:
                refuse(args, call)
            call(valid)
            refuse(valid, call)
            refuse((shell if OLD_EXECUTABLE else None, argv, None, None), call)
        assert probes == [('windows-ver', 'uname'), ('windows-ver', 'platform')]
    '''.replace('OLD_EXECUTABLE', repr(old_executable)).replace('SEPARATOR', repr(separator)))


@pytest.mark.parametrize('version', ['3.9.13', '3.10.11', '3.11.9'])
def test_windows_lscpu_audit_string_with_no_executable(version):
    run_python('''
        import sys
        sys.platform = 'win32'
        probes = install_import_process_audit()
        namespace = {'__name__': 'numpy.testing._private.utils', 'sys': sys}
        exec('def check_support_sve(args):\\n    sys.audit("subprocess.Popen", *args)', namespace)
        call = namespace['check_support_sve']
        valid = (None, 'lscpu', None, None)
        def refuse(args, caller=call):
            prior = list(probes)
            try:
                caller(args)
            except AssertionError as exc:
                assert str(exc) == 'unexpected import process: subprocess.Popen'
            else:
                raise AssertionError('altered Windows lscpu probe was permitted')
            assert probes == prior
        for args in [(None, 'lscpu --extra', None, None), (None, ['lscpu'], None, None),
                     (None, 'lscpu', '.', None), (None, 'lscpu', None, {}),
                     ('other.exe', 'lscpu', None, None)]:
            refuse(args)
        refuse(valid, lambda args: sys.audit('subprocess.Popen', *args))
        call(valid)
        assert probes == [('lscpu',)]
        refuse(valid)
        refuse(('lscpu', 'lscpu', None, None))
    ''')


def test_windows_ver_probe_refuses_non_system_comspec():
    run_python('''
        import os
        import sys
        sys.platform = 'win32'
        os.environ['SystemRoot'] = 'C:/Windows'
        os.environ['ComSpec'] = 'C:/other/cmd.exe'
        probes = install_import_process_audit()
        namespace = {'__name__': 'platform', 'sys': sys}
        exec('def _syscmd_ver(args):\\n    sys.audit("subprocess.Popen", *args)\\ndef win32_ver(args):\\n    return _syscmd_ver(args)\\ndef uname(args):\\n    return win32_ver(args)', namespace)
        try:
            namespace['uname'](('C:/other/cmd.exe', 'C:/other/cmd.exe /c "ver"', None, None))
        except AssertionError as exc:
            assert str(exc) == 'unexpected import process: subprocess.Popen'
        else:
            raise AssertionError('non-system COMSPEC was permitted')
        assert probes == []
    ''')


def test_import_process_diagnostics_do_not_disclose_arguments_or_environment():
    run_python('''
        import contextlib
        import io
        import json
        import sys
        probes = install_import_process_audit()
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            try:
                sys.audit('subprocess.Popen', '/private/executable',
                          ['private-argument'], '/private/cwd', {'PRIVATE_SECRET': 'secret-value'})
            except AssertionError as exc:
                assert str(exc) == 'unexpected import process: subprocess.Popen'
            else:
                raise AssertionError('unexpected process accepted')
        diagnostic = output.getvalue()
        assert not any(value in diagnostic for value in
                       ('/private/', 'private-argument', 'PRIVATE_SECRET', 'secret-value'))
        record = json.loads(diagnostic.removeprefix('import process audit: '))
        assert record['reason'] == 'unmatched'
        assert record['event'] == 'subprocess.Popen'
        assert record['argv_form'] == 'list'
        assert record['default_cwd'] is False and record['default_env'] is False
        assert probes == []
    ''')


def test_result_detail_span_cache_preserves_validation(tmp_path):
    script = tmp_path / 'span_cache_controls.py'
    script.write_text('''
import gc
from pathlib import Path
import weakref
from unittest.mock import patch

import test_taichi_event_compat_source as contract

span = contract._result_detail_span
raw = b'class SimResult:\\n    def compute_detailed(self):\\n        pass\\n'
span.cache_clear()
try:
    # Equal content in a new bytes object reuses parsing; changed content does not.
    with patch.object(contract, 'result_detail_method',
                      wraps=contract.result_detail_method) as extract:
        first = span(raw)
        assert first == (2, 3)
        assert span(bytes(bytearray(raw))) == first
        assert extract.call_count == 1
        shifted = b'\\n' + raw
        assert span(shifted) == (3, 4)
        assert extract.call_count == 2
        assert span.cache_info().maxsize == 1
        assert span.cache_info().currsize == 1
        assert span(raw) == first
        assert extract.call_count == 3

    # Uniqueness and parser failures propagate even after a valid cache warmup.
    for changed, message in (
        (b'class Other:\\n    pass\\n', 'expected one SimResult class'),
        (raw + raw, 'expected one SimResult class'),
        (b'class SimResult:\\n    pass\\n', 'expected one compute_detailed method'),
        (raw + b'    def compute_detailed(self):\\n        pass\\n',
         'expected one compute_detailed method'),
    ):
        with patch.object(contract, 'result_detail_method',
                          wraps=contract.result_detail_method) as extract:
            for _ in range(2):
                try:
                    span(changed)
                except AssertionError as error:
                    assert str(error) == message
                else:
                    raise AssertionError('invalid AST cardinality was cached or accepted')
            assert extract.call_count == 2
    try:
        span(b'class SimResult(:\\n')
    except SyntaxError:
        pass
    else:
        raise AssertionError('invalid syntax was accepted')

    # The cached value contains integers only and must not retain the AST node.
    original_extract = contract.result_detail_method
    references = []
    def track_node(value):
        node = original_extract(value)
        references.append(weakref.ref(node))
        return node
    span.cache_clear()
    with patch.object(contract, 'result_detail_method', side_effect=track_node):
        bounds = span(raw)
    gc.collect()
    assert type(bounds) is tuple and all(type(value) is int for value in bounds)
    assert len(references) == 1 and references[0]() is None

    # Compare reviewed output with the unchanged uncached extractor in-process.
    def uncached_span(value):
        method = original_extract(value)
        return method.lineno, method.end_lineno
    with patch.object(contract, '_result_detail_span', side_effect=uncached_span):
        expected = contract.reviewed_result_identity_source()
        expected_current = contract.reviewed_taichi_import_source()
    span.cache_clear()
    assert contract.reviewed_result_identity_source() == expected
    assert contract.reviewed_result_identity_source() == expected
    assert contract.reviewed_taichi_import_source() == expected_current
    assert expected_current == contract.SOURCE.read_bytes()

    # Warm-cache reconstruction still rereads both fixtures and checks lineage.
    baseline = contract.public_baseline('fibernet/sim/accelerated.py')
    method_fixture = (contract.ROOT / 'tests/fixtures/result_identity_contract_v1'
                      / 'compute_detailed.py.txt')
    original_read = Path.read_bytes
    for fixture, suffix, message, build in (
        (baseline, b'\\nclass SimResult:\\n    pass\\n',
         'expected one SimResult class', contract.reviewed_result_identity_source),
        (method_fixture, b'\\n# changed fixture\\n',
         'reviewed result identity method fixture changed',
         contract.reviewed_result_identity_source),
        (baseline, b'\\n# changed fixture\\n',
         'spring source delta base changed', contract.reviewed_spring_provenance_source),
    ):
        assert contract.reviewed_result_identity_source() == expected
        def changed_read(path):
            value = original_read(path)
            return value + suffix if path == fixture else value
        with patch.object(Path, 'read_bytes', changed_read):
            try:
                build()
            except AssertionError as error:
                assert str(error) == message
            else:
                raise AssertionError('changed fixture was accepted after cache warmup')
        assert contract.reviewed_result_identity_source() == expected
    try:
        contract.require_public_source('fibernet/sim/accelerated.py',
                                       expected_current + b'\\n# unreviewed\\n')
    except AssertionError as error:
        assert str(error) == 'public source bytes changed: fibernet/sim/accelerated.py'
    else:
        raise AssertionError('unreviewed source was accepted after cache warmup')
finally:
    span.cache_clear()
''', encoding='utf-8')
    run_source_check(script, '--cp1252-default')
