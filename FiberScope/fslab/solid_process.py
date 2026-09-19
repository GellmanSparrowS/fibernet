"""Isolated solid construction with atomic output and interruptible parent polling."""
from dataclasses import asdict
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import time
import numpy as np


def build_isolated(network, settings, progress=None, stop_cb=None):
    from .print_export import FiberSolid
    with tempfile.TemporaryDirectory(prefix='fiberscope-solid-') as folder:
        root = Path(folder)
        np.savez(root/'input.npz', positions=network.positions, edges=network.edges)
        (root/'settings.json').write_text(json.dumps(asdict(settings)), encoding='utf-8')
        command = ([sys.executable] if getattr(sys, 'frozen', False) else
                   [sys.executable, str(Path(__file__).resolve().parents[1]/'run.py')])
        with (root/'process.log').open('w') as log:
            process = subprocess.Popen(command+['--solid-worker', folder], stdout=log, stderr=log,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            started = time.monotonic()
            try:
                while process.poll() is None:
                    if stop_cb and stop_cb():
                        raise InterruptedError('Solid generation cancelled')
                    if time.monotonic()-started > 600:
                        raise TimeoutError('Solid generation timed out')
                    try:
                        value = int((root/'progress').read_text())
                        if progress: progress(value)
                    except (OSError, ValueError):
                        pass
                    time.sleep(.08)
                if process.returncode:
                    error = root/'error.txt'
                    raise RuntimeError(error.read_text(encoding='utf-8') if error.exists() else 'Solid worker failed')
                with np.load(root/'result.npz') as data:
                    return FiberSolid(data['vertices'], data['faces'], data['centers'],
                                      float(data['resolution']), settings)
            finally:
                if process.poll() is None:
                    process.terminate()
                process.wait()


def worker(folder):
    from types import SimpleNamespace
    from .print_export import PrintSettings, build_solid
    root = Path(folder)
    try:
        with np.load(root/'input.npz') as data:
            network = SimpleNamespace(positions=data['positions'], edges=data['edges'])
        settings = PrintSettings(**json.loads((root/'settings.json').read_text(encoding='utf-8')))
        solid = build_solid(network, settings, lambda p:(root/'progress').write_text(str(p)))
        np.savez(root/'result.tmp.npz', vertices=solid.vertices, faces=solid.faces,
                 centers=solid.centers, resolution=solid.resolution)
        os.replace(root/'result.tmp.npz', root/'result.npz')
        return 0
    except Exception as exc:
        (root/'error.txt').write_text(str(exc), encoding='utf-8')
        return 1
