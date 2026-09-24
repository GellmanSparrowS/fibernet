"""Make one bounded, reproducible Work-mode handoff ZIP from committed source.

Run: python scripts/build_work_bundle.py
The validated wheel is included for convenience; the frozen APP binary is not.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

from preflight_publish import PublishPreflight


class WorkBundleBuilder:
    def __init__(self, root=None):
        self.root = Path(root or Path(__file__).resolve().parents[1])
        self.output = self.root / "handoff" / "FiberNet_Work_Mode_2026-09-24.zip"

    @staticmethod
    def digest(path):
        value = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                value.update(block)
        return value.hexdigest()

    def run(self):
        PublishPreflight(self.root).run()
        raw = subprocess.check_output(["git", "ls-files", "-z"], cwd=self.root)
        files = sorted({Path(item.decode("utf-8")) for item in raw.split(b"\0")
                        if item and (self.root / item.decode("utf-8")).is_file()})
        candidate = self.root / "release_candidates" / "2026-09-24"
        extras = {
            "validation/local_candidate_manifest.json": candidate / "manifest.json",
            "validation/fibernet-4.2.0-py3-none-any.whl":
                candidate / "fibernet-4.2.0-py3-none-any.whl",
        }
        for path in extras.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.root, text=True).strip()
        manifest = {"git_commit": commit, "files": {}}
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output.with_suffix(".zip.tmp")
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=6, allowZip64=True) as archive:
                for relative in files:
                    source = self.root / relative
                    name = "FiberNet/" + relative.as_posix()
                    archive.write(source, name)
                    manifest["files"][name] = self.digest(source)
                for name, source in extras.items():
                    archive.write(source, "FiberNet/" + name)
                    manifest["files"]["FiberNet/" + name] = self.digest(source)
                archive.writestr("FiberNet/WORK_MODE_FILES_SHA256.json",
                                 json.dumps(manifest, ensure_ascii=False,
                                            indent=2) + "\n")
            with zipfile.ZipFile(temporary) as archive:
                if archive.testzip() is not None:
                    raise AssertionError("handoff ZIP failed CRC verification")
                if "FiberNet/handoff/WORK_MODE_BRIEF_ZH.md" not in archive.namelist():
                    raise AssertionError("handoff brief is absent")
            os.replace(temporary, self.output)
        finally:
            temporary.unlink(missing_ok=True)
        print("[work_bundle] %d files, %.1f MB, sha256 %s" %
              (len(manifest["files"]), self.output.stat().st_size / 1e6,
               self.digest(self.output)))
        return self.output


if __name__ == "__main__":
    WorkBundleBuilder().run()
