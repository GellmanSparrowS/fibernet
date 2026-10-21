"""Explicit point-metric contract; real uncertainty loss and model included."""
import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from fibernet.ml.models import FiberUncertaintyNet, gaussian_nll_loss
from fibernet.ml.training import (
    TrainingHistory, _regression_metrics, cross_validate_torch,
    load_model_bundle, save_model_bundle, train_model,
)


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return x + self.bias


def fit(model, x, y, **kwargs):
    return train_model(model, x, y, X_val=x, y_val=y, epochs=2,
                       batch_size=4, scheduler_type="step", lr=0,
                       early_stopping_patience=0, device="cpu",
                       verbose=False, **kwargs)


def means(k, seen=None):
    def adapter(raw):
        assert raw.ndim == 2 and raw.shape[1] == 2 * k
        assert not raw.requires_grad and not torch.is_grad_enabled()
        if seen is not None:
            seen.append((raw.device.type, tuple(raw.shape)))
        return raw[:, :k]
    return adapter


@pytest.mark.parametrize("k", [1, 2])
def test_builtin_uncertainty_loss_only_and_explicit_adapter(k):
    torch.manual_seed(1729)
    x = torch.randn(8, 3)
    y = torch.randn(8, k)
    model = FiberUncertaintyNet(3, k, hidden=[4], dropout=0)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    torch.manual_seed(101)
    unavailable = fit(model, x, y, loss_fn=gaussian_nll_loss)
    assert unavailable.val_metrics == {"r2": [None, None], "rmse": [None, None]}
    assert all(s["reason"] == "prediction_shape_mismatch"
               for s in unavailable.val_metric_status)
    assert unavailable.val_metric_status[0]["prediction_shape"] == [4, 2*k]
    model.load_state_dict(state)
    seen = []
    predictions = []
    extract = means(k, seen)
    def adapter(raw):
        point = extract(raw)
        predictions.append(point.clone())
        return point
    torch.manual_seed(101)
    declared = fit(model, x, y, loss_fn=gaussian_nll_loss, metric_prediction=adapter)
    for epoch in range(2):
        oracle = _regression_metrics(
            y.numpy(), torch.cat(predictions[epoch*2:epoch*2+2]).numpy())
        for key in oracle:
            assert declared.val_metrics[key][epoch] == pytest.approx(oracle[key])
    assert unavailable.train_losses == declared.train_losses
    assert unavailable.val_losses == declared.val_losses
    assert unavailable.lr_history == declared.lr_history
    assert seen and all(d == "cpu" for d, _ in seen)
    assert declared.metric_prediction == "explicit_callable"


@pytest.mark.parametrize("k", [1, 2])
def test_uncertainty_cv_adapter_and_strict_default(k):
    torch.manual_seed(1729)
    x, y = torch.randn(8, 3), torch.randn(8, k)
    factory = lambda: FiberUncertaintyNet(3, k, hidden=[4], dropout=0)
    kw = dict(n_folds=2, epochs=1, batch_size=4, device="cpu",
              scheduler_type="step", loss_fn=gaussian_nll_loss)
    with pytest.raises(ValueError, match="prediction shape"):
        cross_validate_torch(factory, x, y, **kw)
    seen = []
    scores = cross_validate_torch(factory, x, y, metric_prediction=means(k, seen), **kw)
    assert len(scores["fold_r2s"]) == 2
    assert np.isfinite(scores["fold_rmses"]).all()
    assert len(seen) == 4  # each fold's validation plus restored-model score


@pytest.mark.parametrize("x,y,r2,rmse", [
    ([[1, 100], [2, 110], [3, 120], [4, 130]],
     [[0, 100], [1, 110], [2, 120], [3, 130]], .6, math.sqrt(.5)),
    ([[0], [0]], [[1], [1]], 0, 1),
    ([[1], [1]], [[1], [1]], 1, 0),
    ([[1]], [[1]], None, 0),
    ([[1], [2]], [1, 2], 1, 0),
    ([1, 2], [[1], [2]], 1, 0),
])
def test_point_oracle_scalar_constant_singleton(x, y, r2, rmse):
    hist = fit(Identity(), np.asarray(x, dtype="f"), np.asarray(y, dtype="f"))
    assert hist.val_metrics["r2"] == ([None, None] if r2 is None else pytest.approx([r2]*2))
    assert hist.val_metrics["rmse"] == pytest.approx([rmse]*2)
    if r2 is None:
        assert hist.val_metric_status[0] == {
            "status": "partial", "r2": "insufficient_samples", "rmse": "available"}


@pytest.mark.parametrize("adapter,exception,message", [
    (lambda x: x.numpy(), TypeError, "torch.Tensor"),
    (lambda x: x.bool(), ValueError, "real numeric"),
    (lambda x: x.to(torch.complex64), ValueError, "real numeric"),
    (lambda x: x[:1], ValueError, "metric_prediction shape"),
    (lambda x: x[:, None], ValueError, "metric_prediction shape"),
    (lambda x: x[:, :0], ValueError, "metric_prediction shape"),
    (lambda x: x * float("nan"), ValueError, "finite"),
    (lambda x: (_ for _ in ()).throw(RuntimeError("broken")), ValueError, "adapter failed"),
    (42, TypeError, "callable"),
])
def test_bad_adapter_errors(adapter, exception, message):
    with pytest.raises(exception, match=message):
        fit(Identity(), np.ones((4, 1)), np.ones((4, 1)), metric_prediction=adapter)


def test_raw_shape_adapter_and_inplace_isolation():
    observed = []
    def adapter(raw):
        observed.append(tuple(raw.shape))
        raw.add_(5)
        return raw
    losses = []
    def loss(raw, target):
        losses.append(tuple(raw.shape))
        return (raw - target).square().mean()
    hist = fit(Identity(), np.ones((4, 1)), np.ones(4),
               metric_prediction=adapter, loss_fn=loss)
    assert observed == [(4, 1), (4, 1)]
    assert losses == [(4,)] * 4
    assert hist.val_losses == [0, 0]
    assert hist.val_metrics["rmse"] == [5, 5]


@pytest.mark.parametrize("shape", [(4, 2), (4, 1, 1)])
def test_unknown_shape_is_not_guessed(shape):
    def loss(raw, y):
        return raw.square().mean()
    hist = fit(Identity(), np.ones(shape), np.ones((4, 1)), loss_fn=loss)
    assert hist.val_metrics["r2"] == [None, None]
    assert hist.val_metrics["rmse"] == [None, None]


def test_no_validation_and_nonfinite_are_truthful():
    hist = train_model(Identity(), np.ones((4, 1)), np.ones((4, 1)),
                       epochs=2, scheduler_type="step", verbose=False)
    assert hist.val_losses == []
    assert hist.val_metrics == {"r2": [None, None], "rmse": [None, None]}
    assert all(s["reason"] == "validation_unavailable" for s in hist.val_metric_status)
    hist = fit(Identity(), np.full((4, 1), np.nan), np.ones((4, 1)))
    assert hist.val_metrics["r2"] == [None, None]
    assert hist.val_metric_status[0]["reason"] == "nonfinite_prediction_or_target"


def test_history_serialization_console_bundle_plot(tmp_path, capsys):
    model = Identity()
    hist = train_model(model, [[1.]], [[1.]], X_val=[[1.]], y_val=[[1.]],
                       epochs=2, lr=0, scheduler_type="step", device="cpu")
    assert "R²=unavailable (insufficient_samples)" in capsys.readouterr().out
    assert hist.history_schema_version == 2
    assert hist.metric_definition == "point_regression_uniform_output_v1"
    path = tmp_path / "new.json"
    hist.save(path)
    assert TrainingHistory.load(path).to_dict() == hist.to_dict()
    json.dumps(hist.to_dict(), allow_nan=False)  # finite loss, null singleton R²
    bundle = tmp_path / "model.pt"
    save_model_bundle(model, bundle, history=hist)
    _, info = load_model_bundle(bundle, model=Identity())
    assert info["history"] == hist.to_dict()
    pytest.importorskip("matplotlib")
    fig = hist.plot(str(tmp_path / "history.png"))
    assert [len(ax.lines) for ax in fig.axes] == [3, 1]
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_legacy_and_generic_not_relabelled(tmp_path):
    generic = TrainingHistory()
    generic.update(0, 1, train_metrics={"reconstruction": 1, "kl": .3})
    assert "metric_definition" not in generic.to_dict()
    assert "history_schema_version" not in generic.to_dict()
    old = generic.to_dict()
    old["val_metrics"] = {"r2": [-4e12, float("nan")], "rmse": [1., 0.]}
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(old))
    loaded = TrainingHistory.load(path).to_dict()
    assert json.dumps(loaded) == json.dumps(old)


def test_finite_inputs_overflow_scores_are_unavailable():
    def bounded_loss(raw, target):
        return raw.sum() * 0
    with np.errstate(over="ignore", invalid="ignore"):
        hist = fit(Identity(), np.full((4, 1), 1e20), np.ones((4, 1)),
                   loss_fn=bounded_loss)
    assert hist.val_metrics["rmse"] == [None, None]
    assert hist.val_metric_status[0]["rmse"] == "nonfinite_metric"


def test_adapter_wrong_device_fails_clearly():
    with pytest.raises(ValueError, match="preserve the input device"):
        fit(Identity(), np.ones((4, 1)), np.ones((4, 1)),
            metric_prediction=lambda raw: torch.empty(raw.shape, device="meta"))


def test_cv_nonfinite_target_reports_unavailable(monkeypatch):
    from fibernet.ml import training
    monkeypatch.setattr(training, "train_model", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="CV point metrics unavailable"):
        cross_validate_torch(Identity, np.ones((4, 1)), np.full((4, 1), np.nan),
                             n_folds=2, metric_prediction=lambda raw: raw)
