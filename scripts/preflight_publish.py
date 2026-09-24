"""Check source file boundaries, common credential formats and homepage links.

Run: python scripts/preflight_publish.py
No matching secret content is printed to stdout.
"""
import re
from pathlib import Path
import subprocess


class PublishPreflight:
    SECRET_MARKERS = {
        "GitHub token": re.compile(rb"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}"),
        "GitHub fine-grained token": re.compile(rb"github_pat_[A-Za-z0-9_]{30,}"),
        "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "AWS access key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    }

    def __init__(self, root=None):
        self.root = Path(root or Path(__file__).resolve().parents[1])

    def source_files(self):
        raw = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=self.root)
        paths = sorted({Path(item.decode("utf-8")) for item in raw.split(b"\0")
                        if item})
        return [path for path in paths if (self.root / path).is_file()]

    def run(self):
        files = self.source_files()
        problems = []
        for relative in files:
            path = self.root / relative
            if path.stat().st_size > 20_000_000:
                problems.append(str(relative) + ": above 20 MB")
                continue
            content = path.read_bytes()
            if b"\0" in content[:1024]:
                continue
            for label, marker in self.SECRET_MARKERS.items():
                if marker.search(content):
                    problems.append(str(relative) + ": possible " + label)
        for relative in ("README.md", "docs/README_VNEXT.md"):
            source = (self.root / relative).read_text(encoding="utf-8")
            parent = (self.root / relative).parent
            for target in re.findall(r"!?\[[^\]]*\]\(([^\)]+)\)", source):
                if target.startswith(("#", "http://", "https://", "mailto:")):
                    continue
                if not (parent / target.split("#", 1)[0]).is_file():
                    problems.append(relative + ": broken link " + target)
        if problems:
            raise ValueError("\n".join(problems))
        print("[publish_preflight] %d source files and both homepages PASS" %
              len(files))
        return files


if __name__ == "__main__":
    PublishPreflight().run()
