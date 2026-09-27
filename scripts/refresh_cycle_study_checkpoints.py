"""Recompute cycle studies and atomically accept only identical case results.

Run after changing shared solver or APP structure source:
    python scripts/refresh_cycle_study_checkpoints.py
For the one-time platform-independent geometry-byte migration only:
    python scripts/refresh_cycle_study_checkpoints.py --allow-geometry-hash-upgrade

The .rebuild.json files are resumable. A mismatch leaves both files in place.
"""
import argparse
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

    @staticmethod
    def _without_geometry_hash(value):
        if isinstance(value, dict):
            return {key: CycleCheckpointRefresh._without_geometry_hash(item)
                    for key, item in value.items() if key != "geometry_hash"}
        if isinstance(value, list):
            return [CycleCheckpointRefresh._without_geometry_hash(item)
                    for item in value]
        return value

    def run(self, allow_geometry_hash_upgrade=False):
        for name in self.STUDIES:
            original = self.root / "benchmarks" / "results" / (name + ".json")
            rebuild = original.with_suffix(".rebuild.json")
            subprocess.run([sys.executable, str(self.root / "benchmarks" /
                                             (name + ".py")), "--output", str(rebuild)],
                           cwd=self.root, check=True)
            old = json.loads(original.read_text(encoding="utf-8"))
            new = json.loads(rebuild.read_text(encoding="utf-8"))
            previous = old["cases"]
            current = new["cases"]
            if allow_geometry_hash_upgrade:
                previous = self._without_geometry_hash(previous)
                current = self._without_geometry_hash(current)
            if old["config"] != new["config"] or previous != current:
                raise AssertionError(name + " changed numerically; inspect rebuild")
            if any(old.get(key) != new.get(key) for key in
                   set(old).intersection(new) - {"cases", "config", "engine_hash",
                                                     "reference_sha256",
                                                     "analysis_hash", "source_hash"}):
                raise AssertionError(name + " summary changed; inspect rebuild")
            os.replace(rebuild, original)
            print("[cycle_refresh] %s: %d cases identical" %
                  (name, len(new["cases"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-geometry-hash-upgrade", action="store_true")
    args = parser.parse_args()
    CycleCheckpointRefresh().run(args.allow_geometry_hash_upgrade)
