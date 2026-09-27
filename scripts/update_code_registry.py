"""Update the library code inventory while preserving curated notes.

Run: python scripts/update_code_registry.py
"""
import ast
import hashlib
import json
from pathlib import Path


class CodeRegistry:
    FOLDERS = ('fibernet', 'tests', 'scripts', 'examples', 'benchmarks',
               'manuscript')

    def __init__(self, root=None):
        self.root = Path(root or Path(__file__).resolve().parents[1])

    def update(self):
        path = self.root / 'code_registry.json'
        entries = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        found = set()
        for folder in self.FOLDERS:
            for file in sorted((self.root / folder).rglob('*.py')):
                if '__pycache__' in file.parts:
                    continue
                key = file.relative_to(self.root).as_posix()
                found.add(key)
                raw = file.read_bytes()
                tree = ast.parse(raw.decode('utf-8-sig'))
                entry = entries.setdefault(key, {})
                doc = ast.get_docstring(tree) or ''
                entry.setdefault('purpose', doc.splitlines()[0] if doc else file.stem)
                entry['status'] = 'archived' if '_archived' in file.parts else 'active'
                entry['symbols'] = [node.name for node in tree.body
                                    if isinstance(node, (ast.ClassDef, ast.FunctionDef,
                                                         ast.AsyncFunctionDef))]
                entry['sha256'] = hashlib.sha256(raw).hexdigest()
        for key in entries.keys() - found:
            entries[key]['status'] = 'removed'
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(entries, ensure_ascii=False, indent=1) + '\n',
                       encoding='utf-8')
        tmp.replace(path)
        print('[code_registry] indexed %d files' % len(found))


if __name__ == '__main__':
    CodeRegistry().update()
