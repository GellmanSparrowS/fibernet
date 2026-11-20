"""Repository-only tests for the exact post-upload workflow code; no network/pip."""
import ast
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import textwrap
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / '.github/workflows/publish-pypi.yml'
VERSION = '4.4.4'
FILES = {'fibernet-4.4.4-py3-none-any.whl': 'a'*64, 'fibernet-4.4.4.tar.gz': 'b'*64}
MANIFEST = {'version': VERSION, 'files': FILES}
PUBLISHED = {'info': {'version': VERSION}, 'urls': [
    {'filename': name, 'digests': {'sha256': digest}} for name, digest in FILES.items()]}


def workflow_code():
    text = WORKFLOW.read_text(encoding='utf-8')
    section = text.split('      - name: Verify PyPI exposes exactly the uploaded hashes\n', 1)[1]
    block = section.split('      - name: Install the exact published wheel and smoke-test it\n', 1)[0]
    return textwrap.dedent(block.split("          python - <<'PY'\n", 1)[1].rsplit('          PY\n', 1)[0])


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
            self.ns['wait_for_simple_index'](MANIFEST)
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
        with self.assertRaisesRegex(RuntimeError,'after 180 seconds'):self.wait([snapshot([])])
        self.assertEqual(self.clock.now,180); self.assertEqual(len(self.clock.delays),18)

    def test_requests_consume_the_same_deadline(self):
        with self.assertRaisesRegex(RuntimeError,'request failed'):self.wait([snapshot([])],latency=12)
        self.assertEqual(self.clock.now,180)

    def test_single_blocking_request_is_not_retried_as_missing(self):
        with self.assertRaisesRegex(RuntimeError,'not a missing-release retry'):self.wait([snapshot()],latency=21)
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
        with self.assertRaisesRegex(RuntimeError,'after 180 seconds'):self.wait([data])
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
            with self.assertRaisesRegex(RuntimeError, 'not a missing-release retry'):
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

    def test_correct_official_hashes_precede_simple_check(self):
        with patch.dict(os.environ,MANIFEST=json.dumps(MANIFEST)), patch('urllib.request.urlopen',return_value=io.BytesIO(json.dumps(PUBLISHED).encode())), patch('subprocess.run',return_value=subprocess.CompletedProcess([],0,json.dumps(snapshot()),'')) as child:
            exec(compile(workflow_code(),str(WORKFLOW),'exec'),{})
        self.assertEqual(child.call_count,1)

    def test_workflow_keeps_installation_and_permissions_contract(self):
        text=WORKFLOW.read_text(encoding='utf-8')
        self.assertIn('    timeout-minutes: 10\n    permissions: {}\n',text)
        self.assertIn('"$RUNNER_TEMP/published-core/bin/python" -m pip --isolated install --index-url https://pypi.org/simple --only-binary=:all: "fibernet==$VERSION"',text)
        self.assertIn('"$RUNNER_TEMP/published-core/bin/python" -m pip check',text)
        self.assertIn("assert fibernet.__version__ == importlib.metadata.version('fibernet') == os.environ['VERSION']",text)
        self.assertIn('exclude tests/test_pypi_index_visibility.py',(ROOT/'MANIFEST.in').read_text())


if __name__=='__main__':unittest.main()
