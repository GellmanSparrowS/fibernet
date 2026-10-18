"""Exercise the CLI reporter under a legacy Windows encoding without GUI dependencies."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


REPORTER = Path(__file__).resolve().parents[1] / 'FiberScope' / 'scripts' / 'run_tests.py'
HARNESS = r'''
import runpy
import sys
from subprocess import CompletedProcess
from unittest.mock import patch

script, mode = sys.argv[1:]
assert sys.stdout.encoding == 'cp1252', sys.stdout.encoding
assert sys.stdout.errors == 'strict', sys.stdout.errors

def fake_run(argv, **kwargs):
    assert kwargs['env']['PYTHONUTF8'] == '1'
    assert kwargs['encoding'] == 'utf-8'
    assert kwargs['capture_output'] and kwargs['text']
    if argv[-1].endswith('fake_fail.py'):
        return CompletedProcess(argv, 1, 'synthetic suite started\n',
            'AssertionError: synthetic failure: 制造测试\n')
    return CompletedProcess(argv, 0, 'synthetic next suite: 完成\n', '')

sys.argv = [script, '--only', 'fake_fail,fake_pass' if mode == 'failure' else 'fake_pass']
with patch('subprocess.run', side_effect=fake_run):
    runpy.run_path(script, run_name='__main__')
'''


class TestFiberScopeReporting(unittest.TestCase):
    def run_reporter(self, mode):
        env = dict(os.environ, PYTHONIOENCODING='cp1252:strict', PYTHONUTF8='0',
                   PYTHONDONTWRITEBYTECODE='1')
        return subprocess.run([sys.executable, '-c', HARNESS, str(REPORTER), mode],
                              env=env, capture_output=True, timeout=15)

    def test_unicode_failure_is_reported_and_next_suite_runs(self):
        result = self.run_reporter('failure')
        self.assertEqual(result.returncode, 1)
        output = result.stdout.decode('utf-8')
        self.assertIn('AssertionError: synthetic failure: 制造测试', output)
        self.assertIn('synthetic next suite: 完成', output)
        self.assertIn('SUMMARY: 1/2 passed', output)
        self.assertEqual(result.stderr, b'')

    def test_unicode_success_preserves_zero_exit(self):
        result = self.run_reporter('success')
        self.assertEqual(result.returncode, 0)
        output = result.stdout.decode('utf-8')
        self.assertIn('synthetic next suite: 完成', output)
        self.assertIn('SUMMARY: 1/1 passed', output)
        self.assertEqual(result.stderr, b'')


if __name__ == '__main__':
    unittest.main()
