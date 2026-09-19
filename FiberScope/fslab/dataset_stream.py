"""Atomic bounded datasets with explicit label provenance and resumable seeds."""
import hashlib
import json
import os
import tempfile
from dataclasses import asdict
import numpy as np
from .learning import FEATURE_SCHEMA, select_candidate
from .structure import StructureFactory, CUSTOM_CELLS
from .storage import atomic_replace


def dataset_config(unit, amp, pert, seed0, mode='physics', acquisition='random'):
    from .mlmodel import SIM_KW
    from .simcache import ENGINE_SRC_HASH
    config = dict(schema=FEATURE_SCHEMA, topology='topnet26', unit=unit, amp=float(amp), pert=float(pert),
                seed0=int(seed0), mode=mode, acquisition=acquisition, simulation=SIM_KW,
                custom_cell=CUSTOM_CELLS.get(unit), engine_source=ENGINE_SRC_HASH)
    return json.loads(json.dumps(config))


def config_id(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


class DatasetStream:
    def __init__(self, config, path=None, model=None, pool_size=24):
        self.config, self.path, self.model = config, path, model
        self.pool_size = max(1, min(int(pool_size), 128))
        self.X, self.Y, self.specs = [], [], []
        self.next_seed = int(config['seed0'])
        self.failures = []
        if config['mode'] not in ('generate', 'physics', 'surrogate'):
            raise ValueError('unknown generation mode')
        if config['mode'] == 'surrogate' and model is None:
            raise ValueError('train a model before requesting predictions')
        # Predictions depend on model state: do not silently resume another model.
        if config['mode'] == 'surrogate' and path and os.path.exists(path):
            raise ValueError('use a new prediction dataset path for each model')
        if path and os.path.exists(path):
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z['meta']))
                if meta.get('config') != config:
                    raise ValueError('dataset configuration/schema mismatch; use a new cache path')
                self.X, self.Y = list(z['X']), list(z['Y'])
                self.specs = json.loads(str(z['specs']))
                self.next_seed = int(meta['next_seed'])
                self.failures = meta.get('failures', [])
                if len(self.X) != len(self.Y) or len(self.X) != len(self.specs):
                    raise ValueError('incomplete dataset checkpoint')

    def _save(self):
        if not self.path:
            return
        folder = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(folder, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix='dataset_', suffix='.npz', dir=folder)
        os.close(fd)
        try:
            np.savez_compressed(tmp, X=self.arrays()[0], Y=self.arrays()[1],
                                specs=json.dumps(self.specs),
                                meta=json.dumps(dict(config=self.config, next_seed=self.next_seed,
                                                     failures=self.failures[-100:])))
            atomic_replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def arrays(self):
        return np.asarray(self.X, float).reshape(-1, 14), np.asarray(self.Y, float).reshape(-1, 3)

    def generate(self, n, progress_cb=None, stop_cb=None):
        from .mlmodel import light_features, make_sample
        n = int(n)
        if not 1 <= n <= 2000:
            raise ValueError('dataset size must be 1..2000')
        cfg = self.config
        attempts = 0
        while len(self.X) < n:
            if stop_cb and stop_cb():
                break
            candidates, features = [], []
            count = self.pool_size if cfg['acquisition'] != 'random' else 1
            for _ in range(count):
                seed = self.next_seed
                self.next_seed += 1
                rng = np.random.default_rng(seed)
                factory = StructureFactory(unit=cfg['unit'], seed=seed,
                    line_displacements=rng.uniform(-cfg['amp'], cfg['amp'], (5, 2)).tolist(),
                    perturbation=cfg['pert'] * rng.uniform())
                try:
                    graph = factory.build()
                    x = light_features(graph.node_positions(), graph.edge_array())
                    if not np.isfinite(x).all():
                        raise ValueError('non-finite features')
                    candidates.append(factory)
                    features.append(x)
                except (ValueError, MemoryError) as exc:
                    self.failures.append(dict(seed=seed, error=str(exc)))
                if stop_cb and stop_cb():
                    break
            if not candidates:
                self._save()
                raise ValueError('no valid candidate structures')
            if stop_cb and stop_cb():
                self._save()
                break
            index = select_candidate(features, *self.arrays(), cfg['acquisition'], np.random.default_rng(self.next_seed))
            factory, x = candidates[index], features[index]
            try:
                if cfg['mode'] == 'physics':
                    x, y = make_sample(factory)
                elif cfg['mode'] == 'surrogate':
                    y = self.model.predict(x)
                else:
                    y = np.full(3, np.nan)
                self.X.append(x)
                self.Y.append(y)
                self.specs.append(asdict(factory))
            except (ValueError, FloatingPointError) as exc:
                self.failures.append(dict(seed=factory.seed, error=str(exc)))
            self._save()
            attempts += 1
            if attempts > n * 3:
                raise ValueError('too many rejected structures; checkpoint retained')
            if progress_cb:
                progress_cb(len(self.X), n)
        return self.arrays()


def generate_dataset(unit, n=60, amp=.2, pert=.1, seed0=0, path=None,
                     progress_cb=None, stop_cb=None, mode='physics', acquisition='random', model=None):
    config = dataset_config(unit, amp, pert, seed0, mode, acquisition)
    return DatasetStream(config, path, model).generate(n, progress_cb, stop_cb)
