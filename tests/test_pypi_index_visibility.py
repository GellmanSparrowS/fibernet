"""Repository-only tests for the exact post-upload workflow code; no network/pip."""
import ast
import copy
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import textwrap
import time
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / '.github/workflows/publish-pypi.yml'
VERSION = '4.4.4'
FILES = {'fibernet-4.4.4-py3-none-any.whl': 'a'*64, 'fibernet-4.4.4.tar.gz': 'b'*64}
MANIFEST = {'version': VERSION, 'files': FILES}
PUBLISHED = {'info': {'version': VERSION}, 'urls': [
    {'filename': name, 'digests': {'sha256': digest}} for name, digest in FILES.items()]}


def workflow_verifier():
    text = WORKFLOW.read_text(encoding='utf-8')
    jobs = re.findall(r'(?ms)^  verify-published:\n(.*?)(?=^  [\w-]+:\n|\Z)', text)
    assert len(jobs) == 1, 'expected exactly one verify-published job'
    return jobs[0]


def workflow_code():
    blocks = re.findall(r"(?ms)^          python - <<'PY'\n(.*?)^          PY[ \t]*$",
                        workflow_verifier())
    assert len(blocks) == 1, 'expected exactly one Python verifier entry point'
    return textwrap.dedent(blocks[0])


def functions():
    tree = ast.parse(workflow_code())
    tree.body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    namespace = {}
    exec(compile(tree, str(WORKFLOW), 'exec'), namespace)
    return namespace


def snapshot(names=None):
    return {'status': 200, 'url': 'https://pypi.org/simple/fibernet/',
            'data': {'name': 'fibernet', 'files': [
                {'filename': name, 'hashes': {'sha256': FILES[name]},
                 'url': 'https://files.pythonhosted.org/packages/test/'+name,
                 'yanked': False} for name in (FILES if names is None else names)]}}


def pip_report():
    wheel = f'fibernet-{VERSION}-py3-none-any.whl'
    return {'install': [{
        'metadata': {'name': 'fibernet', 'version': VERSION},
        'requested': True, 'is_direct': False, 'is_yanked': False,
        'download_info': {'url': 'https://files.pythonhosted.org/packages/test/' + wheel,
                          'archive_info': {'hashes': {'sha256': FILES[wheel]}}}}]}


def stale_pip_output(versions='4.4.2, 4.4.3'):
    return (f'ERROR: Could not find a version that satisfies the requirement '
            f'fibernet=={VERSION} (from versions: {versions})\n'
            f'ERROR: No matching distribution found for fibernet=={VERSION}\n')


class Clock:
    def __init__(self): self.now = 0.; self.delays = []
    def monotonic(self): return self.now
    def sleep(self, seconds):
        assert 0 < seconds <= 15
        self.delays.append(seconds); self.now += seconds


class TestPyPIIndexVisibility(unittest.TestCase):
    def setUp(self): self.ns = functions(); self.clock = Clock()

    def wait(self, responses, latency=0.):
        calls = []
        def run(argv, **kwargs):
            self.assertEqual(argv[1:4], ['-I','-S','-c'])
            self.assertGreater(kwargs['timeout'], 0)
            self.assertLessEqual(kwargs['timeout'], 20)
            self.assertTrue(kwargs['check'])
            self.assertNotIn('shell', kwargs)
            calls.append(kwargs['timeout'])
            if latency > kwargs['timeout']:
                self.clock.now += kwargs['timeout']
                raise subprocess.TimeoutExpired(argv, kwargs['timeout'])
            self.clock.now += latency
            data = responses[min(len(calls)-1, len(responses)-1)]
            return subprocess.CompletedProcess(argv, 0, json.dumps(data), '')
        with patch('time.monotonic',self.clock.monotonic), patch('time.sleep',self.clock.sleep), patch('subprocess.run',run):
            self.ns['wait_for_simple_index'](MANIFEST, 180.)
        return calls

    def test_immediate_success(self):
        self.assertEqual(self.wait([snapshot()]), [20]); self.assertEqual(self.clock.now,0)

    def test_late_index_then_success(self):
        self.assertEqual(len(self.wait([snapshot([]),snapshot([next(iter(FILES))]),snapshot()])),3)
        self.assertEqual(self.clock.now,20); self.assertEqual(self.clock.delays,[10,10])

    def test_404_then_success(self):
        late={'status':404,'url':'https://pypi.org/simple/fibernet/','data':None}
        self.assertEqual(len(self.wait([late,snapshot()])),2)

    def test_persistent_missing_stops_at_180_seconds(self):
        with self.assertRaisesRegex(RuntimeError,'180 seconds'):self.wait([snapshot([])])
        self.assertEqual(self.clock.now,180); self.assertEqual(len(self.clock.delays),18)

    def test_requests_consume_the_same_deadline(self):
        with self.assertRaises(subprocess.TimeoutExpired):self.wait([snapshot([])],latency=12)
        self.assertEqual(self.clock.now,180)

    def test_simple_wait_exposes_blocking_request_timeout_without_visibility_retry(self):
        with self.assertRaises(subprocess.TimeoutExpired):self.wait([snapshot()],latency=21)
        self.assertEqual(self.clock.now,20); self.assertEqual(self.clock.delays,[])

    def test_hash_mismatch_fails_without_retry_even_if_other_file_missing(self):
        data=snapshot([next(iter(FILES))]);data['data']['files'][0]['hashes']['sha256']='c'*64
        with self.assertRaisesRegex(RuntimeError,'hash mismatch'):self.wait([data])
        self.assertEqual(self.clock.now,0); self.assertEqual(self.clock.delays,[])

    def test_duplicate_file_fails_without_retry(self):
        data=snapshot();data['data']['files'].append(copy.deepcopy(data['data']['files'][0]))
        with self.assertRaisesRegex(RuntimeError,'identity or hash mismatch'):self.wait([data])

    def test_unrelated_release_does_not_satisfy_target(self):
        data=snapshot()
        for entry in data['data']['files']:entry['filename']=entry['filename'].replace('4.4.4','4.4.3')
        with self.assertRaisesRegex(RuntimeError,'180 seconds'):self.wait([data])
        self.assertEqual(self.clock.now,180)

    def test_wrong_manifest_version_cannot_match_file_names(self):
        bad=copy.deepcopy(MANIFEST);bad['version']='4.4.5'
        with self.assertRaisesRegex(RuntimeError,'release identity'):self.ns['simple_index_ready'](bad,snapshot())

    def test_wrong_project_and_sources_are_not_visibility_retries(self):
        for mutation in ('project','index','file','yanked','yanked-empty','yanked-null','yanked-zero','status'):
            with self.subTest(mutation=mutation):
                data=snapshot()
                if mutation=='project':data['data']['name']='other'
                elif mutation=='index':data['url']='https://other.example/simple/fibernet/'
                elif mutation=='file':data['data']['files'][0]['url']='https://other.example/packages/file'
                elif mutation=='yanked':data['data']['files'][0]['yanked']='withdrawn'
                elif mutation=='yanked-empty':data['data']['files'][0]['yanked']=''
                elif mutation=='yanked-null':data['data']['files'][0]['yanked']=None
                elif mutation=='yanked-zero':data['data']['files'][0]['yanked']=0
                else:data['status']=503
                with self.assertRaises(RuntimeError):self.wait([data])
                self.assertEqual(self.clock.delays,[])

    def test_transport_failure_is_terminal(self):
        with patch('subprocess.run',side_effect=subprocess.CalledProcessError(1,['python'])):
            with self.assertRaisesRegex(RuntimeError,'not a missing-release retry'):self.ns['fetch_simple_snapshot'](20)

    def test_actual_stalled_child_is_terminated_by_parent_timeout(self):
        real_run = subprocess.run
        def stalled(argv, **kwargs):
            argv = list(argv); argv[4] = 'import time; time.sleep(30)'
            return real_run(argv, **kwargs)
        started = time.perf_counter()
        with patch('subprocess.run', stalled):
            with self.assertRaises(subprocess.TimeoutExpired):
                self.ns['fetch_simple_snapshot'](.1)
        # Accommodate process startup on busy CI; the 30-second child must not finish.
        self.assertLess(time.perf_counter()-started, 10)

    def test_malformed_response_is_terminal(self):
        with patch('subprocess.run',return_value=subprocess.CompletedProcess([],0,'not-json','')):
            with self.assertRaises(json.JSONDecodeError):self.ns['fetch_simple_snapshot'](20)

    def test_fetch_uses_official_json_index_no_cache_and_bounded_body(self):
        def run(argv,**kwargs):
            ast.parse(argv[4])
            self.assertIn("'https://pypi.org/simple/fibernet/'",argv[4])
            self.assertIn("'Accept': 'application/vnd.pypi.simple.v1+json, application/vnd.pypi.simple.v1+html; q=0.1, text/html; q=0.01'",argv[4])
            self.assertIn("'Cache-Control': 'no-cache'",argv[4])
            self.assertIn('response.read(2*1024*1024+1)',argv[4])
            self.assertEqual(argv[-1],'3.5');self.assertEqual(kwargs['timeout'],3.5)
            return subprocess.CompletedProcess(argv,0,json.dumps(snapshot()),'')
        with patch('subprocess.run',run):self.assertEqual(self.ns['fetch_simple_snapshot'](3.5),snapshot())

    def test_wrong_published_version_or_hash_never_reaches_simple_wait(self):
        for field in ('version','hash','file'):
            with self.subTest(field=field):
                data=copy.deepcopy(PUBLISHED)
                if field=='version':data['info']['version']='4.4.3'
                elif field=='hash':data['urls'][0]['digests']['sha256']='c'*64
                else:data['urls'][0]['filename']='unexpected.whl'
                clock=Clock()
                with patch.dict(os.environ,MANIFEST=json.dumps(MANIFEST)), patch('time.monotonic',clock.monotonic), patch('time.sleep',clock.sleep), patch('urllib.request.urlopen',side_effect=lambda *args,**kwargs:io.BytesIO(json.dumps(data).encode())), patch('subprocess.run') as child:
                    with self.assertRaisesRegex(RuntimeError,'Published files were not visible'):
                        exec(compile(workflow_code(),str(WORKFLOW),'exec'),{})
                    child.assert_not_called();self.assertEqual(clock.now,300)

    def assert_pip_command(self, argv, python, report):
        self.assertEqual(argv, [
            str(python), '-m', 'pip', '--isolated', '--disable-pip-version-check',
            '--no-cache-dir', '--retries', '0', '--no-color', 'install', '--verbose',
            '--index-url', 'https://pypi.org/simple', '--only-binary=:all:',
            '--report', str(report), f'fibernet=={VERSION}',
        ])

    def test_correct_official_hashes_precede_simple_check_and_pip(self):
        calls = []

        def release(request, **kwargs):
            self.assertEqual(request.full_url, f'https://pypi.org/pypi/fibernet/{VERSION}/json')
            calls.append('version-json')
            return io.BytesIO(json.dumps(PUBLISHED).encode())

        def child(argv, **kwargs):
            if argv[1:3] == ['-m', 'venv']:
                self.assertEqual(Path(argv[3]), work / 'published-core')
                self.assertTrue(kwargs['check'])
                calls.append('venv')
                return subprocess.CompletedProcess(argv, 0)
            if argv[1:4] == ['-I', '-S', '-c']:
                calls.append('simple-json')
                return subprocess.CompletedProcess(argv, 0, json.dumps(snapshot()), '')
            report = Path(argv[argv.index('--report') + 1])
            self.assert_pip_command(argv, work / 'published-core/bin/python', report)
            self.assertEqual(report.parent.parent, work)
            report.write_text(json.dumps(pip_report()), encoding='utf-8')
            calls.append('pip')
            return subprocess.CompletedProcess(argv, 0, '', '')

        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            with patch.dict(os.environ, MANIFEST=json.dumps(MANIFEST), RUNNER_TEMP=temporary), \
                    patch('urllib.request.urlopen', side_effect=release), patch('subprocess.run', side_effect=child):
                exec(compile(workflow_code(), str(WORKFLOW), 'exec'), {})
        self.assertEqual(calls, ['version-json', 'venv', 'simple-json', 'pip'])

    def run_pip(self, attempts, *, snapshots=None, deadline=180.):
        calls = []
        self.pip_calls = calls
        snapshots = iter(snapshots) if snapshots is not None else None
        attempts = iter(attempts)

        def child(argv, **kwargs):
            if argv[1:4] == ['-I', '-S', '-c']:
                calls.append('simple-json')
                data = next(snapshots) if snapshots is not None else snapshot()
                if isinstance(data, subprocess.TimeoutExpired):
                    self.clock.now += kwargs['timeout']
                if isinstance(data, Exception):
                    raise data
                return subprocess.CompletedProcess(argv, 0, json.dumps(data), '')
            self.assert_pip_command(argv, python, report)
            self.assertEqual(kwargs['stdout'], subprocess.PIPE)
            self.assertEqual(kwargs['stderr'], subprocess.STDOUT)
            self.assertFalse(kwargs['check'])
            self.assertEqual(kwargs['timeout'], deadline - self.clock.now)
            self.assertNotIn('shell', kwargs)
            calls.append('pip')
            event = next(attempts)
            if isinstance(event, Exception):
                raise event
            if isinstance(event, dict):
                report.write_text(json.dumps(event), encoding='utf-8')
                return subprocess.CompletedProcess(argv, 0, '', '')
            return subprocess.CompletedProcess(argv, 1, event, '')

        with tempfile.TemporaryDirectory() as temporary:
            python = Path(temporary) / 'published-core/bin/python'
            report = Path(temporary) / 'install.json'
            with patch('time.monotonic', self.clock.monotonic), \
                    patch('time.sleep', self.clock.sleep), \
                    patch('subprocess.run', side_effect=child), redirect_stdout(io.StringIO()):
                self.ns['install_from_index'](python, MANIFEST, deadline, report)

    def test_silent_simple_timeout_retries_then_installs_accepted_wheel(self):
        timeout = subprocess.TimeoutExpired(['python'], 20)
        self.run_pip([pip_report()], snapshots=[timeout, snapshot()])
        self.assertEqual(self.pip_calls, ['simple-json', 'simple-json', 'pip'])
        self.assertEqual(self.clock.now, 20)
        self.assertEqual(self.clock.delays, [])

    def test_second_timeout_is_terminal_even_after_stale_pip(self):
        timeout = subprocess.TimeoutExpired(['python'], 20)
        with self.assertRaisesRegex(RuntimeError, 'timed out; not a missing-release retry'):
            self.run_pip([stale_pip_output()], snapshots=[timeout, snapshot(), timeout])
        self.assertEqual(self.pip_calls, ['simple-json', 'simple-json', 'pip', 'simple-json'])
        self.assertEqual(self.clock.now, 50)
        self.assertEqual(self.clock.delays, [10])

    def test_simple_timeout_at_deadline_does_not_retry(self):
        timeout = subprocess.TimeoutExpired(['python'], 5)
        with self.assertRaisesRegex(RuntimeError, 'exceeded 180 seconds'):
            self.run_pip([], snapshots=[timeout], deadline=5.)
        self.assertEqual(self.pip_calls, ['simple-json'])
        self.assertEqual(self.clock.now, 5)

    def test_simple_timeout_with_diagnostic_does_not_retry(self):
        timeout = subprocess.TimeoutExpired(['python'], 20, stderr=b'TLS failed')
        with self.assertRaisesRegex(RuntimeError, 'timed out; not a missing-release retry'):
            self.run_pip([], snapshots=[timeout])
        self.assertEqual(self.pip_calls, ['simple-json'])

    def test_wrong_hash_after_timeout_still_blocks_before_pip(self):
        timeout = subprocess.TimeoutExpired(['python'], 20)
        bad = snapshot()
        bad['data']['files'][0]['hashes']['sha256'] = 'c'*64
        with self.assertRaisesRegex(RuntimeError, 'hash mismatch'):
            self.run_pip([], snapshots=[timeout, bad])
        self.assertEqual(self.pip_calls, ['simple-json', 'simple-json'])

    def test_stale_pip_rechecks_simple_then_installs_accepted_wheel(self):
        self.run_pip([stale_pip_output(), pip_report()])
        self.assertEqual(self.pip_calls, ['simple-json', 'pip', 'simple-json', 'pip'])
        self.assertEqual(self.clock.delays, [10])

    def test_stale_pip_rechecks_hash_before_retry(self):
        bad = snapshot()
        bad['data']['files'][0]['hashes']['sha256'] = 'c'*64
        with self.assertRaisesRegex(RuntimeError, 'hash mismatch'):
            self.run_pip([stale_pip_output()], snapshots=[snapshot(), bad])
        self.assertEqual(self.pip_calls, ['simple-json', 'pip', 'simple-json'])

    def test_stale_pip_shares_deadline_with_simple_checks(self):
        with self.assertRaisesRegex(RuntimeError, '180 seconds'):
            self.run_pip([stale_pip_output()] * 18)
        self.assertEqual(self.pip_calls, ['simple-json', 'pip'] * 18)
        self.assertEqual(self.clock.now, 180)

    def test_pip_hard_failures_do_not_retry(self):
        stale = stale_pip_output()
        for diagnostic in (
            'ERROR: No matching distribution found for required-dependency>=5',
            'ERROR: These packages do not match the hashes',
            'Could not fetch URL https://pypi.org/simple/fibernet/: TLS failure\n' + stale,
            'WARNING: connection failed\n' + stale,
            'ERROR: Ignored versions requiring a different Python version\n' + stale,
            stale_pip_output('none'), stale_pip_output(VERSION), stale_pip_output('4.4.5'),
        ):
            with self.subTest(diagnostic=diagnostic):
                with self.assertRaisesRegex(RuntimeError, 'not a stale-index retry'):
                    self.run_pip([diagnostic])
                self.assertEqual(self.pip_calls, ['simple-json', 'pip'])
                self.assertEqual(self.clock.delays, [])

    def test_pip_timeout_does_not_retry(self):
        timeout = subprocess.TimeoutExpired(['pip'], 180, output=b'timed-out child diagnostic')
        with self.assertRaisesRegex(RuntimeError, 'pip timed out'):
            self.run_pip([timeout])
        self.assertEqual(self.pip_calls, ['simple-json', 'pip'])
        self.assertEqual(self.clock.delays, [])

    def test_successful_pip_exit_rejects_wrong_report_without_retry(self):
        for mutation in ('origin', 'hash', 'version', 'direct', 'yanked', 'not-requested', 'duplicate'):
            with self.subTest(mutation=mutation):
                value = pip_report()
                row = value['install'][0]
                if mutation == 'origin':
                    row['download_info']['url'] = row['download_info']['url'].replace(
                        'files.pythonhosted.org', 'other.example')
                elif mutation == 'hash':
                    row['download_info']['archive_info']['hashes']['sha256'] = 'c'*64
                elif mutation == 'version':
                    row['metadata']['version'] = '4.4.3'
                elif mutation == 'direct':
                    row['is_direct'] = True
                elif mutation == 'yanked':
                    row['is_yanked'] = True
                elif mutation == 'not-requested':
                    row['requested'] = False
                else:
                    value['install'].append(copy.deepcopy(row))
                with self.assertRaisesRegex(RuntimeError, 'pip-selected|exactly one'):
                    self.run_pip([value])
                self.assertEqual(self.pip_calls, ['simple-json', 'pip'])
                self.assertEqual(self.clock.delays, [])

    def test_workflow_keeps_installation_and_permissions_contract(self):
        text = workflow_verifier()
        self.assertIn('    needs: [build, publish]\n', text)
        self.assertIn('    timeout-minutes: 10\n    permissions: {}\n', text)
        self.assertIn('"$RUNNER_TEMP/published-core/bin/python" -m pip check', text)
        self.assertIn("assert fibernet.__version__ == importlib.metadata.version('fibernet') == os.environ['VERSION']", text)
        self.assertIn('exclude tests/test_pypi_index_visibility.py', (ROOT/'MANIFEST.in').read_text())


if __name__=='__main__':unittest.main()
