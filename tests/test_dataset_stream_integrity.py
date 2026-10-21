"""Dataset rows are validated and committed together before checkpointing."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from fibernet.ml.physical_dataset import DatasetStream


def make_stream(path=None, mode="physics", sample_fn=None, spec_fn=None,
                feature_fn=None, pool_size=2, acquisition="random"):
    graph = SimpleNamespace(node_positions=lambda: np.zeros((2, 2)),
                            edge_array=lambda: np.array([[0, 1]]))
    return DatasetStream(
        dict(seed0=0, mode=mode, acquisition=acquisition), path=path,
        model=SimpleNamespace(predict=lambda x: np.ones(3)),
        pool_size=pool_size,
        factory_builder=lambda seed, cfg, rng: SimpleNamespace(
            seed=seed, build=lambda: graph),
        feature_fn=feature_fn or (lambda nodes, edges: np.ones(14)),
        sample_fn=sample_fn or (lambda factory: (np.ones(14), np.ones(3))),
        spec_fn=spec_fn or (lambda factory: {"seed": factory.seed}),
    )


def test_rejected_spec_does_not_commit_partial_row(tmp_path):
    def reject(factory):
        raise ValueError("invalid spec")
    path = tmp_path / "data.npz"
    stream = make_stream(path, spec_fn=reject)
    with pytest.raises(ValueError, match="too many rejected"):
        stream.generate(1)
    assert len(stream.X) == len(stream.Y) == len(stream.specs) == 0
    resumed = make_stream(path)
    assert resumed.arrays()[0].shape == (0, 14)
    assert resumed.next_seed == stream.next_seed


@pytest.mark.parametrize("x,y", [
    (np.full(14, np.nan), np.ones(3)),
    (np.ones(14), np.full(3, np.nan)),
    (np.ones(14), np.full(3, np.inf)),
    (np.ones(28), np.ones(3)),
    (np.ones(14), np.ones(6)),
    (np.ones((1, 14)), np.ones(3)),
])
def test_invalid_physical_sample_never_enters_checkpoint(tmp_path, x, y):
    path = tmp_path / "data.npz"
    stream = make_stream(path, sample_fn=lambda factory: (x, y))
    with pytest.raises(ValueError, match="too many rejected"):
        stream.generate(1)
    assert len(stream.X) == len(stream.Y) == len(stream.specs) == 0
    assert make_stream(path).arrays()[1].shape == (0, 3)


def test_callback_arrays_are_owned_by_stream():
    x, y = np.ones(14), np.ones(3)
    stream = make_stream(sample_fn=lambda factory: (x, y))
    stream.generate(1)
    x[:] = 7
    y[:] = 8
    np.testing.assert_array_equal(stream.arrays()[0], np.ones((1, 14)))
    np.testing.assert_array_equal(stream.arrays()[1], np.ones((1, 3)))


def test_generate_mode_retains_explicit_nan_labels(tmp_path):
    path = tmp_path / "data.npz"
    x, y = make_stream(path, mode="generate").generate(2)
    assert x.shape == (2, 14) and y.shape == (2, 3)
    assert np.isnan(y).all()
    np.testing.assert_array_equal(make_stream(path, mode="generate").arrays()[0], x)


@pytest.mark.parametrize("x,y", [
    (np.ones((1, 28)), np.ones((1, 3))),
    (np.ones((1, 14)), np.ones((1, 6))),
    (np.ones((1, 14)), np.full((1, 3), np.nan)),
    (np.full((1, 14), np.inf), np.ones((1, 3))),
])
def test_corrupt_checkpoint_rejected_on_load(tmp_path, x, y):
    path = tmp_path / "data.npz"
    config = dict(seed0=0, mode="physics", acquisition="random")
    np.savez(path, X=x, Y=y, specs=json.dumps([{"seed": 0}]),
             meta=json.dumps(dict(config=config, next_seed=1)))
    with pytest.raises(ValueError, match="checkpoint"):
        make_stream(path)


@pytest.mark.parametrize('spec', [{"bad": object()}, {"bad": float('nan')}])
def test_unserializable_spec_does_not_commit_row(tmp_path, spec):
    stream = make_stream(tmp_path / 'data.npz', spec_fn=lambda factory: spec)
    with pytest.raises(ValueError, match='too many rejected'):
        stream.generate(1)
    assert len(stream.X) == len(stream.Y) == len(stream.specs) == 0
    assert make_stream(stream.path).arrays()[0].shape == (0, 14)


def test_invalid_surrogate_labels_are_rejected(tmp_path):
    stream = make_stream(tmp_path / 'predictions.npz', mode='surrogate')
    stream.model.predict = lambda x: np.full(3, np.nan)
    with pytest.raises(ValueError, match='too many rejected'):
        stream.generate(1)
    assert len(stream.X) == len(stream.Y) == len(stream.specs) == 0


@pytest.mark.parametrize('features', [np.ones(28), np.full(14, np.nan)])
def test_invalid_candidate_features_rejected_before_selection(tmp_path, features):
    path = tmp_path / 'data.npz'
    stream = make_stream(path, feature_fn=lambda nodes, edges: features)
    with pytest.raises(ValueError, match='no valid candidate'):
        stream.generate(1)
    assert make_stream(path).arrays()[0].shape == (0, 14)


def test_active_pool_size_is_bound_to_checkpoint(tmp_path):
    path = tmp_path / 'active.npz'
    stream = make_stream(path, acquisition='diversity', pool_size=2)
    stream.generate(1)
    resumed = make_stream(path, acquisition='diversity', pool_size=2)
    assert resumed.next_seed == 2
    with pytest.raises(ValueError, match='pool size mismatch'):
        make_stream(path, acquisition='diversity', pool_size=3)


def test_random_sampling_ignores_unused_pool_size(tmp_path):
    path = tmp_path / 'random.npz'
    make_stream(path, pool_size=2).generate(1)
    resumed = make_stream(path, pool_size=3)
    resumed.generate(2)
    assert resumed.next_seed == 2


def test_legacy_active_pool_resume_warns_without_discarding_data(tmp_path):
    path = tmp_path / 'legacy.npz'
    original = make_stream(path, acquisition='diversity', pool_size=2)
    original.generate(1)
    with np.load(path, allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    meta = json.loads(str(data['meta']))
    meta.pop('acquisition_pool_size')
    data['meta'] = json.dumps(meta)
    np.savez(path, **data)
    with pytest.warns(RuntimeWarning, match='sampling sequence compatibility'):
        resumed = make_stream(path, acquisition='diversity', pool_size=2)
    np.testing.assert_array_equal(resumed.arrays()[0], original.arrays()[0])
    resumed.generate(2)
    with np.load(path, allow_pickle=False) as saved:
        assert json.loads(str(saved['meta']))['acquisition_pool_size'] == 2
