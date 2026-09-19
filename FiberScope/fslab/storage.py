"""Validated data directories and strict atomic checkpoint replacement."""
import os
from pathlib import Path
import sys
import tempfile
import uuid
from functools import lru_cache


def atomic_replace(source, destination):
    """Never copy bytes over a committed checkpoint when rename fails."""
    os.replace(source, destination)


@lru_cache(maxsize=8)
def writable_data_dir(kind):
    """Prefer user storage; probe both initial and replacing atomic writes."""
    if kind not in ('datasets', 'training', 'manufacturing', 'workflows'):
        raise ValueError('unknown storage category')
    root = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parents[1]
    candidates = [Path(os.environ.get('LOCALAPPDATA', Path.home()))/'FiberScope'/kind,
                  Path.home()/'.fiberscope'/kind,
                  root/'_cache'/kind,
                  Path(tempfile.gettempdir())/'FiberScope'/kind]
    for folder in candidates:
        token = '.probe_' + uuid.uuid4().hex
        source, target = folder/(token+'.tmp'), folder/(token+'.json')
        try:
            folder.mkdir(parents=True, exist_ok=True)
            source.write_text('first', encoding='utf-8')
            atomic_replace(source, target)
            source.write_text('second', encoding='utf-8')
            atomic_replace(source, target)
            if target.read_text(encoding='utf-8') != 'second':
                continue
            return str(folder)
        except OSError:
            continue
        finally:
            for path in (source, target):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
    raise OSError('No data directory supports atomic checkpoints; choose a writable local folder')
