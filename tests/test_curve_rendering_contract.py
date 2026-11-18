"""Curve-rendering contracts run in fresh processes, never the pytest backend."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('profile', ['protocol', 'actual'])
def test_curve_rendering_contract(profile, tmp_path):
    """No solver/generator; optional actual-class profile never imports Taichi."""
    worker = Path(__file__).with_name('_curve_rendering_worker.py')
    output = tmp_path / profile
    env = os.environ.copy()
    env.update(MPLBACKEND='Agg', MPLCONFIGDIR=str(tmp_path / 'mplconfig'),
               XDG_CACHE_HOME=str(tmp_path / 'cache'), PYTHONDONTWRITEBYTECODE='1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    # Source checkout imports must not depend on the caller's working directory.
    previous = env.get('PYTHONPATH')
    env['PYTHONPATH'] = str(worker.parents[1]) + (os.pathsep + previous if previous else '')
    completed = subprocess.run(
        [sys.executable, '-B', str(worker), '--mode', profile, '--output', str(output)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    transcript = completed.stdout + completed.stderr
    (tmp_path / (profile + '.log')).write_text(transcript)
    if completed.returncode == 77:
        pytest.skip(transcript.strip())
    assert completed.returncode == 0, transcript
    result = json.loads((output / 'result.json').read_text())
    assert result['success'] and result['tests'] == 8, result
    assert result['errors'] == result['failures'] == 0, result
    assert not result['taichi_imported'], result
