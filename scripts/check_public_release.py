"""Download and verify the public FiberScope release archive.

Run: python scripts/check_public_release.py
Interrupted downloads retain a .part file and resume when the server supports Range.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request
import zipfile


class PublicReleaseCheck:
    def __init__(self, output_dir=None, max_bytes=250_000_000):
        self.output_dir = Path(output_dir or Path(os.getcwd()) / "release_candidates" / "2026-09-27")
        self.max_bytes = int(max_bytes)
        self.api_url = "https://api.github.com/repos/GellmanSparrowS/fibernet/releases/tags/v4.2.0"
        self.archive_name = "FiberScope-3.1.0-Windows-x64.zip"

    @staticmethod
    def digest(path):
        value = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                value.update(block)
        return value.hexdigest()

    @staticmethod
    def request(url, start=0):
        headers = {"User-Agent": "FiberNet-release-check"}
        if start:
            headers["Range"] = "bytes=%d-" % start
        return urllib.request.Request(url, headers=headers)

    def download(self, url, target, expected_size, expected_sha):
        if target.is_file() and target.stat().st_size == expected_size:
            if self.digest(target) == expected_sha:
                return
        partial = target.with_suffix(target.suffix + ".part")
        start = partial.stat().st_size if partial.is_file() else 0
        if start >= expected_size:
            start = 0
        with urllib.request.urlopen(self.request(url, start), timeout=90) as response:
            append = start > 0 and response.status == 206
            total = start if append else 0
            with partial.open("ab" if append else "wb") as handle:
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    total += len(block)
                    if total > self.max_bytes or total > expected_size:
                        raise ValueError("release archive exceeds expected size")
                    handle.write(block)
                handle.flush()
                os.fsync(handle.fileno())
        if partial.stat().st_size != expected_size or self.digest(partial) != expected_sha:
            raise AssertionError("release archive size or SHA256 mismatch")
        os.replace(partial, target)

    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(self.request(self.api_url), timeout=30) as response:
            release = json.load(response)
        if release["tag_name"] != "v4.2.0" or release["draft"] or release["prerelease"]:
            raise AssertionError("unexpected release identity")
        assets = {item["name"]: item for item in release["assets"]}
        archive = assets[self.archive_name]
        sums = assets["SHA256SUMS.txt"]
        with urllib.request.urlopen(self.request(sums["browser_download_url"]), timeout=30) as response:
            lines = response.read().decode("utf-8-sig").strip().splitlines()
        expected = dict(line.split(maxsplit=1)[::-1] for line in lines)
        sha = expected[self.archive_name]
        if archive.get("digest") != "sha256:" + sha:
            raise AssertionError("release asset digest differs from SHA256SUMS")
        target = self.output_dir / (self.archive_name[:-4] + "-github.zip")
        self.download(archive["browser_download_url"], target, archive["size"], sha)
        with zipfile.ZipFile(target) as bundle:
            if bundle.testzip() is not None:
                raise AssertionError("release archive CRC failed")
            names = bundle.namelist()
            if not any(name.endswith("/FiberScope.exe") for name in names):
                raise AssertionError("FiberScope.exe missing from archive")
        print("[public_release] %d bytes, %d files, SHA256 %s" %
              (target.stat().st_size, len(names), sha))
        return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None)
    arguments = parser.parse_args()
    PublicReleaseCheck(output_dir=arguments.output_dir).run()
