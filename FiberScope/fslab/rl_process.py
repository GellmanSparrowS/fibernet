"""Launch a CPU-bounded optional training runtime without blocking Qt."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from dataclasses import asdict
from .structure import CUSTOM_CELLS
from .inverse import InverseRecord
from .simcache import _load_cache
from .search_algorithms import atomic_json
from .storage import writable_data_dir


def training_python():
    configured = os.environ.get('FIBERSCOPE_TRAIN_PYTHON')
    if configured:
        path = Path(configured)
        if not path.is_file():
            raise FileNotFoundError('FIBERSCOPE_TRAIN_PYTHON does not point to Python')
        return str(path)
    if not getattr(sys, 'frozen', False):
        return sys.executable
    for root in ('anaconda3', 'miniconda3'):
        path = Path.home() / root / 'envs' / 'ml310' / 'python.exe'
        if path.is_file():
            return str(path)
    raise RuntimeError('Set FIBERSCOPE_TRAIN_PYTHON to a Python environment with stable-baselines3 and gymnasium')


def run_external(factory, target, algorithm, parameters, budget, seed, callback=None, stop_cb=None,
                 amplitude=.6, resume=False, model=None):
    root = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[1]))
    runtime = root / 'training_runtime' if getattr(sys, 'frozen', False) else root
    worker = runtime / 'scripts' / 'rl_worker.py'
    request = dict(factory=asdict(factory), target=target, algorithm=algorithm,
                   parameters=parameters, seed=seed, custom_cells=CUSTOM_CELLS,
                   amplitude=amplitude, protocol=23)
    import pickle
    model_data = pickle.dumps(model, protocol=4) if model is not None else None
    request['model_hash'] = hashlib.sha256(model_data).hexdigest() if model_data else None
    signature = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:16]
    training_root = os.environ.get('FIBERSCOPE_TRAINING_DIR')
    base = (Path(training_root) if training_root else Path(writable_data_dir('training'))) / signature
    base.mkdir(parents=True, exist_ok=True)
    pointer = base / 'latest.json'
    if resume and pointer.exists():
        name = json.loads(pointer.read_text(encoding='utf-8'))['folder']
        if Path(name).name != name:
            raise ValueError('invalid checkpoint folder')
        folder = base / name
    else:
        import uuid
        folder = base / uuid.uuid4().hex[:12]
        atomic_json(pointer, {'folder': folder.name})
    folder.mkdir(parents=True, exist_ok=True)
    if model_data is not None:
        from .storage import atomic_replace
        temporary = folder / 'model.tmp'
        temporary.write_bytes(model_data)
        atomic_replace(temporary, folder / 'model.pkl')
    request['budget'] = int(budget)
    atomic_json(folder / 'request.json', request)
    (folder / 'STOP').unlink(missing_ok=True)
    env = dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if getattr(sys, 'frozen', False):
        env['FIBERNET_ROOT'] = str(runtime)
    with (folder / 'stderr.log').open('w', encoding='utf-8') as error:
        process = subprocess.Popen([training_python(), '-u', str(worker), str(folder / 'request.json')],
            cwd=str(runtime), env=env, stdout=subprocess.PIPE, stderr=error, text=True, encoding='utf-8',
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        finished = threading.Event()

        def monitor():
            while not finished.wait(.2):
                if stop_cb and stop_cb():
                    (folder / 'STOP').touch()
                    return
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        try:
            for line in process.stdout:
                try:
                    event = json.loads(line)
                    record = InverseRecord(**event['record'])
                except (ValueError, TypeError, KeyError):
                    continue
                snapshot = folder / ('candidate_%06d.npz' % record.eval_id)
                if model_data is not None:
                    from .model_inverse import load_prediction
                    run = load_prediction(snapshot)
                else:
                    run = _load_cache(str(snapshot))
                if callback:
                    callback(record, run)
                snapshot.unlink(missing_ok=True)
            code = process.wait()
        finally:
            finished.set()
            thread.join(timeout=1.)
            process.stdout.close()
    if code:
        message = (folder / 'stderr.log').read_text(encoding='utf-8')[-2500:]
        raise RuntimeError(message or 'training worker failed')
    result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
    result['records'] = [InverseRecord(**r) for r in result['records']]
    if model_data is not None:
        from .model_inverse import load_prediction
        snapshot = folder / 'best_prediction.npz'
        preview = load_prediction(snapshot) if snapshot.exists() else None
        result.update(best_run=None, best_preview=preview, evaluation_mode='surrogate',
                      best_prediction=preview.prediction.tolist() if preview is not None else None)
    else:
        result['best_run'] = _load_cache(str(folder / 'best_run.npz'))
    result['checkpoint'] = str(folder)
    return result
