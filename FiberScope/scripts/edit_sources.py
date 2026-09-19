"""Apply exact, bounded source replacements while preserving newline style.

Usage: python scripts/edit_sources.py edits.json
JSON entries: {"path": "relative/file.py", "old": "...", "new": "..."}.
"""
import json
from pathlib import Path
import sys


class SourceEdits:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def apply(self, entries):
        prepared = {}
        for entry in entries:
            path = (self.root / entry['path']).resolve()
            if self.root not in path.parents:
                raise ValueError('source path escapes project')
            if path not in prepared:
                raw = path.read_bytes()
                prepared[path] = [raw.decode('utf-8-sig').replace('\r\n', '\n'),
                                  '\r\n' if b'\r\n' in raw else '\n']
            text, newline = prepared[path]
            count = text.count(entry['old'])
            if count != entry.get('count', 1):
                raise ValueError('%s: expected %s matches, got %d' %
                                 (path, entry.get('count', 1), count))
            prepared[path][0] = text.replace(entry['old'], entry['new'])
        for path, (text, newline) in prepared.items():
            path.write_bytes(text.replace('\n', newline).encode('utf-8'))


if __name__ == '__main__':
    SourceEdits(Path(__file__).resolve().parents[1]).apply(
        json.loads(Path(sys.argv[1]).read_text(encoding='utf-8-sig')))
