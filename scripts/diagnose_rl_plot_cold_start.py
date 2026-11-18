"""Collect bounded, cold-cache RL plotting evidence without altering the tests."""
import argparse
import faulthandler
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

SOURCES = {
    'tests/test_rl_plot_state.py': '1c2d254583f64b5d9209d59037094e3c0c20a80561ad37a727093dac32360e12',
    'fibernet/rl/utils.py': '71630e24cc406a299504f30db2a0b682ead4fc0d1ddb0d9724efe655b144894a',
}


def _text(value):
    return value.decode('utf-8', errors='replace') if isinstance(value, bytes) else value or ''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    actual = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in SOURCES}
    if actual != SOURCES:
        raise ValueError('diagnostic source identity differs from the inspected test/plot code')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    test = root/'tests/test_rl_plot_state.py'
    results = []
    for label, scenario, bundled in (
            ('exports_reference', 'public_exports', False),
            ('plot_values_reference', 'plot_values', False),
            ('plot_values_bundled_fonts', 'plot_values', True)):
        with tempfile.TemporaryDirectory(prefix='fibernet-rl-plot-') as cache:
            env = dict(os.environ, MPLBACKEND='svg', MPLCONFIGDIR=cache,
                       XDG_CACHE_HOME=cache, OPENBLAS_NUM_THREADS='1',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
            # Import timing itself adds stderr traffic; keep the reference unprofiled.
            env.pop('PYTHONPROFILEIMPORTTIME', None)
            env.pop('MPL_IGNORE_SYSTEM_FONTS', None)
            if bundled:
                env['MPL_IGNORE_SYSTEM_FONTS'] = '1'
            bootstrap = '\n'.join([
                'import faulthandler,runpy,sys',
                'faulthandler.enable()',
                'faulthandler.dump_traceback_later(20, repeat=True)',
                'sys.argv = ' + repr([str(test), '--scenario', scenario]),
                'runpy.run_path(' + repr(str(test)) + ", run_name='__main__')",
                'faulthandler.cancel_dump_traceback_later()',
            ])
            started = time.perf_counter()
            try:
                child = subprocess.run([sys.executable, '-u', '-B', '-c', bootstrap],
                                       cwd=root, env=env, capture_output=True,
                                       text=True, timeout=45)
                stdout, stderr = child.stdout, child.stderr
                passed = child.returncode == 0 and ('RL_PLOT_SCENARIO_PASSED='+scenario) in stdout
                status, returncode = ('passed' if passed else 'failed'), child.returncode
            except subprocess.TimeoutExpired as exc:
                stdout, stderr = _text(exc.stdout), _text(exc.stderr)
                status, returncode = 'timeout', None
            elapsed = time.perf_counter() - started
        (args.output_dir/(label+'.stdout.txt')).write_text(stdout, encoding='utf-8')
        (args.output_dir/(label+'.stderr.txt')).write_text(stderr, encoding='utf-8')
        result = {'case': label, 'scenario': scenario, 'status': status,
                  'elapsed_s': elapsed, 'deadline_s': 45, 'returncode': returncode,
                  'font_control': env.get('MPL_IGNORE_SYSTEM_FONTS'),
                  'watchdog_snapshot': 'most recent call first' in stderr}
        results.append(result)
        print(json.dumps(result), flush=True)
        if stderr:
            print('--- '+label+' stderr ---\n'+stderr, flush=True)
    report = {'source_sha256': actual, 'python': sys.version,
              'platform': sys.platform, 'architecture': platform.machine(),
              'image_os': os.environ.get('ImageOS'),
              'image_version': os.environ.get('ImageVersion'),
              'packages': {name: importlib.metadata.version(name)
                           for name in ('numpy', 'scipy', 'matplotlib')},
              'cases': results,
              'method': 'original assertions;isolated cold caches;20s/40s watchdog;45s deadline;separate font-discovery control'}
    (args.output_dir/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n',
                                             encoding='utf-8')
    if any(row['status'] != 'passed' for row in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
