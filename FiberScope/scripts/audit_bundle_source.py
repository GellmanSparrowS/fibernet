"""Verify bundled application bytecode matches the current source tree.

Run: python scripts/audit_bundle_source.py [dist/FiberScope.exe]
"""
import argparse
import hashlib
import marshal
from pathlib import Path
import types
from PyInstaller.archive.readers import CArchiveReader


def signature(value):
    if isinstance(value, types.CodeType):
        return (value.co_code.hex(), value.co_names, value.co_varnames,
                value.co_freevars, value.co_cellvars, value.co_flags,
                value.co_argcount, value.co_posonlyargcount, value.co_kwonlyargcount,
                tuple(signature(v) for v in value.co_consts))
    if isinstance(value, tuple):
        return tuple(signature(v) for v in value)
    if isinstance(value, frozenset):
        return ('frozenset', tuple(sorted(repr(signature(v)) for v in value)))
    return value


class BundleAudit:
    def run(self, exe):
        root = Path(__file__).resolve().parents[1]
        archive = CArchiveReader(str(exe))
        name = next(k for k in archive.toc if k.endswith('.pyz'))
        pyz = archive.open_embedded_archive(name)
        mismatches = []
        if not any(name.replace('\\', '/').startswith('fast_simplification.libs/') and name.lower().endswith('.dll') for name in archive.toc):
            mismatches.append('fast_simplification private DLL directory missing')
        checked = 0
        for folder in ('fslab', 'studio'):
            for path in sorted((root / folder).glob('*.py')):
                module = folder if path.stem == '__init__' else folder + '.' + path.stem
                bundled = pyz.extract(module)
                source = compile(path.read_bytes(), str(path), 'exec', optimize=0)
                if signature(bundled) != signature(source):
                    mismatches.append(module)
                checked += 1
        bundled = marshal.loads(archive.extract('run'))
        if signature(bundled) != signature(compile((root/'run.py').read_bytes(), 'run.py', 'exec')):
            mismatches.append('run')
        print('[bundle-source] checked=%d mismatches=%s sha256=%s' % (
            checked+1, mismatches, hashlib.sha256(Path(exe).read_bytes()).hexdigest()))
        return int(bool(mismatches))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exe', nargs='?', default='dist/FiberScope.exe')
    raise SystemExit(BundleAudit().run(parser.parse_args().exe))
