"""Refresh code inventory without discarding curated notes.

Run: python scripts/update_code_registry.py
"""
import ast
import hashlib
import json
from pathlib import Path


class CodeRegistry:
    def __init__(self, root):
        self.root = Path(root)

    def update(self):
        path = self.root / 'files_registry.json'
        entries = json.loads(path.read_text(encoding='utf-8'))
        files = [self.root / 'run.py']
        for folder in ('fslab', 'studio', 'scripts', 'tests'):
            files.extend(sorted((self.root / folder).glob('*.py')))
        for file in files:
            raw = file.read_bytes()
            tree = ast.parse(raw.decode('utf-8-sig'))
            key = file.relative_to(self.root).as_posix()
            entry = entries.setdefault(key, {})
            doc = ast.get_docstring(tree) or ''
            entry.setdefault('purpose', doc.split('\n')[0])
            entry.setdefault('status', 'indexed')
            entry['symbols'] = [node.name for node in tree.body
                                if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
            entry['sha256'] = hashlib.sha256(raw).hexdigest()
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(entries, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
        tmp.replace(path)
        print('[registry] indexed %d code files' % len(files))


if __name__ == '__main__':
    CodeRegistry(Path(__file__).resolve().parents[1]).update()
