"""Render the GitHub homepage with versioned absolute links for PyPI.

Run: python scripts/build_pypi_readme.py
The resulting Markdown is the package long description in pyproject.toml.
"""

import re
from pathlib import Path


class PyPIReadmeBuilder:
    def __init__(self, root=None, version="v4.2.0"):
        self.root = Path(root or Path(__file__).resolve().parents[1])
        self.version = version
        self.repository = "https://github.com/GellmanSparrowS/fibernet"
        self.raw = "https://raw.githubusercontent.com/GellmanSparrowS/fibernet"

    def _replace(self, match):
        prefix, target, suffix = match.groups()
        if target.startswith(("#", "https://", "http://", "mailto:")):
            return match.group(0)
        path, sep, anchor = target.partition("#")
        if not (self.root / path).is_file():
            raise FileNotFoundError("PyPI README target is missing: " + path)
        if prefix.startswith("!["):
            destination = "%s/%s/%s" % (self.raw, self.version, path)
        else:
            destination = "%s/blob/%s/%s" % (self.repository, self.version, path)
        return prefix + destination + (sep + anchor if sep else "") + suffix

    def run(self):
        content = (self.root / "README.md").read_text(encoding="utf-8")
        content = re.sub(r"(!?\[[^\]]*\]\()([^\)]+)(\))", self._replace,
                         content)
        if "docs/media/three_dimensional_topologies.gif" not in content:
            raise AssertionError("3D gallery missing from PyPI description")
        (self.root / "PYPI_README.md").write_bytes(content.encode("utf-8"))
        print("[pypi_readme] versioned links prepared")


if __name__ == "__main__":
    PyPIReadmeBuilder().run()
