"""Render the tested docs preview as the repository homepage.

Run: python scripts/promote_homepage.py
Only local relative Markdown targets are rebased from docs/ to repo root.
"""
import posixpath
import re
from pathlib import Path


class HomepagePromoter:
    def __init__(self, root=None):
        self.root = Path(root or Path(__file__).resolve().parents[1])

    @staticmethod
    def _rebase(match):
        prefix, target, suffix = match.groups()
        if target.startswith(("#", "https://", "http://", "mailto:")):
            return match.group(0)
        path, sep, anchor = target.partition("#")
        rebased = posixpath.normpath(posixpath.join("docs", path))
        return prefix + rebased + (sep + anchor if sep else "") + suffix

    def run(self):
        source = self.root / "docs" / "README_VNEXT.md"
        content = source.read_text(encoding="utf-8")
        content = content.replace(
            "This page is a **local development preview**, not a description of a released unified version.",
            "This page documents the **4.2.0 development branch**; a new PyPI or desktop release has not yet been published.")
        content = content.replace(
            "The page has not replaced the public GitHub README.",
            "The current homepage is source-backed and distinguishes tested code from published releases.")
        content = content.replace("The local unified branch adds an English default interface",
                                  "This unified branch adds an English default interface")
        content = content.replace("it remains local to this branch",
                                  "it remains a working research artifact in this branch")
        content = re.sub(r"(!?\[[^\]]*\]\()([^\)]+)(\))", self._rebase,
                         content)
        missing = []
        for target in re.findall(r"!?\[[^\]]*\]\(([^\)]+)\)", content):
            if target.startswith(("#", "https://", "http://", "mailto:")):
                continue
            path = target.split("#", 1)[0]
            if not (self.root / path).is_file():
                missing.append(target)
        if missing:
            raise FileNotFoundError("homepage links missing: " + ", ".join(missing))
        destination = self.root / "README.md"
        destination.write_text(content, encoding="utf-8")
        print("[homepage] %d local links verified" %
              len(re.findall(r"!?\[[^\]]*\]\(([^\)]+)\)", content)))


if __name__ == "__main__":
    HomepagePromoter().run()
