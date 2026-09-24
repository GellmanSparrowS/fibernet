"""Refresh the corrected independent FEM screen only after exact case comparison.

Run: python scripts/refresh_fem_checkpoint.py
The historical pre-correction JSON is deliberately left untouched.
"""
import json
import os
from pathlib import Path
import subprocess
import sys


class FemCheckpointRefresh:
    def __init__(self, root=None):
        self.root = Path(root or Path(__file__).resolve().parents[1])

    def run(self):
        original = (self.root / "benchmarks" / "results" /
                    "recruitment_fem_validation_parallel.json")
        rebuild = original.with_suffix(".rebuild.json")
        subprocess.run([sys.executable, str(self.root / "benchmarks" /
                                        "recruitment_fem_validation.py"),
                        "--output", str(rebuild)], cwd=self.root, check=True)
        old = json.loads(original.read_text(encoding="utf-8"))
        new = json.loads(rebuild.read_text(encoding="utf-8"))
        if old["config"] != new["config"] or old["cases"] != new["cases"]:
            raise AssertionError("FEM cases changed; inspect rebuild")
        os.replace(rebuild, original)
        print("[fem_refresh] %d corrected cases identical" % len(new["cases"]))


if __name__ == "__main__":
    FemCheckpointRefresh().run()
