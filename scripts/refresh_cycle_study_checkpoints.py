"""Recompute cycle studies and atomically accept only identical case results.

Run after changing shared solver or APP structure source:
    python scripts/refresh_cycle_study_checkpoints.py

The .rebuild.json files are resumable. A mismatch leaves both files in place.
"""
import json
import os
from pathlib import Path
import subprocess
import sys


class CycleCheckpointRefresh:
    STUDIES = ("constrained_cycle_intervention", "ai_cycle_selector",
               "independent_fem_cycle_intervention")

    def __init__(self, root=None):
        self.root = Path(root or Path(__file__).resolve().parents[1])

    def run(self):
        for name in self.STUDIES:
            original = self.root / "benchmarks" / "results" / (name + ".json")
            rebuild = original.with_suffix(".rebuild.json")
            subprocess.run([sys.executable, str(self.root / "benchmarks" /
                                             (name + ".py")), "--output", str(rebuild)],
                           cwd=self.root, check=True)
            old = json.loads(original.read_text(encoding="utf-8"))
            new = json.loads(rebuild.read_text(encoding="utf-8"))
            if old["config"] != new["config"] or old["cases"] != new["cases"]:
                raise AssertionError(name + " changed numerically; inspect rebuild")
            if any(old.get(key) != new.get(key) for key in
                   set(old).intersection(new) - {"cases", "config", "engine_hash",
                                                     "reference_sha256"}):
                raise AssertionError(name + " summary changed; inspect rebuild")
            os.replace(rebuild, original)
            print("[cycle_refresh] %s: %d cases identical" %
                  (name, len(new["cases"])), flush=True)


if __name__ == "__main__":
    CycleCheckpointRefresh().run()
