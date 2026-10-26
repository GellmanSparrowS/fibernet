"""Stdlib-only subprocess boundary for optional plotting imports and state."""
import os
from pathlib import Path
import subprocess
import sys


def run_worker(filename):
    tests = Path(__file__).resolve().parent
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH'] = os.pathsep.join([str(tests.parent), str(tests), env.get('PYTHONPATH', '')])
    result = subprocess.run(
        [sys.executable, '-m', 'unittest', 'discover', '-s', str(tests), '-p', filename, '-v'],
        cwd=tests.parent, env=env, capture_output=True, text=True, timeout=90,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    return output
