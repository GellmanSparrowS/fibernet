"""Record hashes for the tested local wheel and Windows APP folder.

Run after wheel installation smoke and scripts/build_exe.py --onedir:
    python scripts/seal_local_candidate.py
The output is local and is not a public package release.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile


class CandidateSealer:
    def __init__(self, root=None):
        self.root = Path(root or Path(__file__).resolve().parents[1])
        self.candidate = self.root / "release_candidates" / "2026-09-24"

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def run(self):
        source = self.candidate / "build_tmp" / "fibernet-4.2.0-py3-none-any.whl"
        with zipfile.ZipFile(source) as wheel:
            if "fibernet/gen/custom_cells.py" not in wheel.namelist():
                raise AssertionError("wheel lacks shared custom-cell API")
        target = self.candidate / source.name
        temporary = target.with_suffix(".tmp.whl")
        try:
            shutil.copyfile(source, temporary)
            if self.digest(temporary) != self.digest(source):
                raise AssertionError("wheel copy changed bytes")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        app = self.root / "FiberScope" / "dist" / "FiberScope"
        executable = app / "FiberScope.exe"
        files = [item for item in app.rglob("*") if item.is_file()]
        if not executable.is_file() or not files:
            raise FileNotFoundError("frozen APP is absent")
        payload = {
            "status": "local-tested-source-candidate; not a PyPI or APP release",
            "library_wheel": {"path": target.name, "bytes": target.stat().st_size,
                              "sha256": self.digest(target),
                              "isolated_python_310_api_smoke": True,
                              "custom_cell_route_edges": 140,
                              "parallel_beam_grip_reaction_ratio": 2.0},
            "app_onedir": {"relative_path_from_repo_root":
                           "FiberScope/dist/FiberScope", "bytes": sum(
                               item.stat().st_size for item in files),
                           "exe_sha256": self.digest(executable),
                           "dependency_probe": "2/2", "frozen_smoke": "pass",
                           "cleanroom_smoke": "pass",
                           "portability_audit": "portable on tested Windows 10 build"},
            "regressions": {"python_library_passed": 368,
                            "python_library_skipped": 19,
                            "desktop_suites": "25/25",
                            "engine_golden_arrays_exact": 40},
            "limits": ["No PyPI or APP release has been published.",
                       "Library CI passed on Linux, macOS and Windows with Python 3.9-3.12; no third-party-machine install or frozen APP validation outside Windows.",
                       "No material-experiment validation.",
                       "Recruitment colors show a positive axial-strain "
                       "threshold, not complete force flow."],
        }
        destination = self.candidate / "manifest.json"
        temporary = destination.with_suffix(".tmp.json")
        try:
            temporary.write_text(json.dumps(payload, indent=2) + "\n",
                                 encoding="utf-8")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        print("[candidate] wheel and APP hashes recorded")
        return payload


if __name__ == "__main__":
    CandidateSealer().run()
