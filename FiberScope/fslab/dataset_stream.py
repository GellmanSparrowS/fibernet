"""FiberScope configuration adapter for the shared resumable dataset core."""
import hashlib
import json
from dataclasses import asdict

from fibernet.ml.physical_dataset import DatasetStream as SharedDatasetStream
from fibernet.ml.physical_surrogate import light_features
from .learning import FEATURE_SCHEMA
from .structure import StructureFactory, CUSTOM_CELLS


def dataset_config(unit, amp, pert, seed0, mode='physics', acquisition='random'):
    from .mlmodel import SIM_KW
    from .simcache import ENGINE_SRC_HASH
    config = dict(schema=FEATURE_SCHEMA, topology='topnet26', unit=unit,
                  amp=float(amp), pert=float(pert), seed0=int(seed0),
                  mode=mode, acquisition=acquisition, simulation=SIM_KW,
                  custom_cell=CUSTOM_CELLS.get(unit), engine_source=ENGINE_SRC_HASH)
    return json.loads(json.dumps(config))


def config_id(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


class DatasetStream(SharedDatasetStream):
    def __init__(self, config, path=None, model=None, pool_size=24):
        super().__init__(config, path, model, pool_size,
                         factory_builder=self._factory,
                         feature_fn=light_features,
                         sample_fn=self._sample,
                         spec_fn=asdict)

    @staticmethod
    def _factory(seed, cfg, rng):
        return StructureFactory(
            unit=cfg['unit'], seed=seed,
            line_displacements=rng.uniform(-cfg['amp'], cfg['amp'], (5, 2)).tolist(),
            perturbation=cfg['pert'] * rng.uniform())

    @staticmethod
    def _sample(factory):
        from .mlmodel import make_sample
        return make_sample(factory)


def generate_dataset(unit, n=60, amp=.2, pert=.1, seed0=0, path=None,
                     progress_cb=None, stop_cb=None, mode='physics',
                     acquisition='random', model=None):
    config = dataset_config(unit, amp, pert, seed0, mode, acquisition)
    return DatasetStream(config, path, model).generate(n, progress_cb, stop_cb)
