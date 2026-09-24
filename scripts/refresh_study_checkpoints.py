"""Recompute resumable study checkpoints and replace only identical case data.

Run: python scripts/refresh_study_checkpoints.py
Each study writes a separate .rebuild.json first. A mismatch preserves both
files for inspection and exits without replacing the established result.
"""
import json
import os
from pathlib import Path
import subprocess
import sys


class CheckpointRefresh:
    STUDIES = ('percolation_pruning_pilot',
               'percolation_pruning_matched',
               'percolation_cycle_intervention')

    def __init__(self, root=None):
        self.root = Path(root or Path(__file__).resolve().parents[1])

    def run(self):
        for name in self.STUDIES:
            script = self.root / 'benchmarks' / (name + '.py')
            original = self.root / 'benchmarks' / 'results' / (name + '.json')
            rebuild = original.with_suffix('.rebuild.json')
            subprocess.run([sys.executable, str(script), '--output', str(rebuild)],
                           cwd=self.root, check=True)
            previous = json.loads(original.read_text(encoding='utf-8'))
            current = json.loads(rebuild.read_text(encoding='utf-8'))
            if (previous['config'] != current['config']
                    or previous['cases'] != current['cases']):
                raise AssertionError(name + ' changed numerically; inspect rebuild')
            os.replace(rebuild, original)
            print('[checkpoint_refresh] %s: %d cases identical' %
                  (name, len(current['cases'])))


if __name__ == '__main__':
    CheckpointRefresh().run()
