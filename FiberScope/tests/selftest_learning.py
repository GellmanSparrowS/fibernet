"""Run all registered models, acquisition strategies and checkpoint contracts."""
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fslab.learning import MODEL_SPECS, ACQUISITIONS, Regressor, select_candidate
from fslab.dataset_stream import DatasetStream, dataset_config


def main():
    rng = np.random.default_rng(14)
    X = rng.normal(size=(80, 14))
    Y = X[:, :3] * [1., 2., 3.] + .05 * rng.normal(size=(80, 3))
    for key in MODEL_SPECS:
        model = Regressor(key)
        hist = model.train(X, Y, epochs=12)
        assert np.isfinite(model.predict(X)).all()
        assert not set(model.train_indices) & set(model.val_indices)
        assert np.allclose(model.x_mean, X[model.train_indices].mean(0))
        assert hist['epochs_run'] >= 1
        print('[learning] %s validated: %d held-out samples' % (key, len(model.val_indices)))
    for method in ACQUISITIONS:
        index = select_candidate(X[20:], X[:20], Y[:20], method, np.random.default_rng(2))
        assert 0 <= index < 60
    with tempfile.TemporaryDirectory() as tmp:
        cfg = dataset_config('square', .2, .1, 0, 'generate')
        path = os.path.join(tmp, 'data.npz')
        stream = DatasetStream(cfg, path)
        a, _ = stream.generate(3)
        stream = DatasetStream(cfg, path)
        b, labels = stream.generate(5)
        assert np.array_equal(a, b[:3]) and len(b) == 5 and np.isnan(labels).all()
        assert len({s['seed'] for s in stream.specs}) == 5
        assert len(DatasetStream(cfg, path).generate(5)[0]) == 5
        try:
            DatasetStream(dict(cfg, amp=.3), path)
            raise AssertionError('mismatched cache accepted')
        except ValueError:
            pass
    print('[learning] acquisitions, unlabelled provenance and atomic resume PASS')


if __name__ == '__main__':
    main()
