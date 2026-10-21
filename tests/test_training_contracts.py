"""Point-regression CV and per-invocation training lifecycle contracts."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from fibernet.ml import training


class ExactModel(torch.nn.Module):
    def __init__(self, squeeze=False, duplicate=False):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.squeeze = squeeze
        self.duplicate = duplicate

    def forward(self, x):
        result = x[:, :1] * self.scale
        if self.duplicate:
            return torch.cat([result, result], dim=1)
        return result.squeeze(-1) if self.squeeze else result


@pytest.mark.parametrize("target_vector,squeeze", [(True, False), (False, True), (False, False)])
def test_cv_normalizes_only_scalar_output_axes(monkeypatch, target_vector, squeeze):
    pytest.importorskip("sklearn")
    monkeypatch.setattr(training, "train_model", lambda *args, **kwargs: None)
    x = np.arange(12, dtype=np.float32).reshape(-1, 1)
    y = x[:, 0] if target_vector else x.copy()
    result = training.cross_validate_torch(lambda: ExactModel(squeeze=squeeze), x, y, n_folds=3)
    assert result["fold_rmses"] == [0.0] * 3
    assert result["fold_r2s"] == [1.0] * 3


def test_cv_rejects_different_output_count(monkeypatch):
    pytest.importorskip("sklearn")
    monkeypatch.setattr(training, "train_model", lambda *args, **kwargs: None)
    x = np.arange(12, dtype=np.float32).reshape(-1, 1)
    with pytest.raises(ValueError, match="shape"):
        training.cross_validate_torch(lambda: ExactModel(duplicate=True), x, x, n_folds=3)


@pytest.mark.parametrize("targets,preds,r2,rmse", [
    ([[0, 100], [2, 102]], [[1, 101], [1, 101]], 0.0, 1.0),
    ([1, 3, 5], [1, 3, 5], 1.0, 0.0),
    ([2, 2], [2, 2], 1.0, 0.0),
    ([2, 2], [1, 1], 0.0, 1.0),
    ([[0, 2], [2, 2]], [[1, 2], [1, 2]], 0.5, np.sqrt(0.5)),
])
def test_regression_metrics_outputwise_contract(targets, preds, r2, rmse):
    result = training._regression_metrics(targets, preds)
    assert result["r2"] == pytest.approx(r2)
    assert result["rmse"] == pytest.approx(rmse)


def test_regression_metrics_single_row_r2_is_nan():
    result = training._regression_metrics([1.0], [[1.0]])
    assert np.isnan(result["r2"])
    assert result["rmse"] == 0.0


@pytest.mark.parametrize("targets,preds", [
    ([], []), (np.empty((2, 0)), np.empty((2, 0))),
    (np.ones((2, 1, 1)), np.ones((2, 1, 1))),
    (np.ones((2, 1)), np.ones((2, 2))),
    (np.ones((2, 1)), np.ones((1, 2))),
    (1.0, 1.0),
])
def test_regression_metrics_rejects_invalid_axes(targets, preds):
    with pytest.raises(ValueError):
        training._regression_metrics(targets, preds)


def train(model, **kwargs):
    x = torch.arange(1, 9, dtype=torch.float32).reshape(-1, 1)
    options = dict(epochs=5, batch_size=8, scheduler_type="step", verbose=False,
                   device="cpu", early_stopping_patience=2,
                   optimizer=torch.optim.SGD(model.parameters(), lr=0))
    options.update(kwargs)
    return training.train_model(model, x, x[:, 0], **options)


def stale_checkpoint(tmp_path):
    path = tmp_path / "best.pt"
    stale = ExactModel()
    with torch.no_grad():
        stale.scale.fill_(9)
    torch.save({"model_state_dict": stale.state_dict()}, path)
    return path


@pytest.mark.parametrize("with_checkpoint", [False, True])
def test_no_validation_runs_all_epochs_and_ignores_stale_checkpoint(tmp_path, with_checkpoint):
    model = ExactModel()
    path = stale_checkpoint(tmp_path) if with_checkpoint else None
    original = path.read_bytes() if path else None
    history = train(model, checkpoint_path=path)
    assert len(history.train_losses) == 5
    assert model.scale.item() == 1
    assert history.val_losses == []
    if path:
        assert path.read_bytes() == original


@pytest.mark.parametrize("validation", [{"X_val": np.ones((2, 1))}, {"y_val": np.ones(2)},
                                       {"X_val": np.empty((0, 1)), "y_val": np.empty(0)}])
def test_validation_requires_complete_nonempty_pair(validation):
    with pytest.raises(ValueError, match="[Vv]alidation|X_val|y_val"):
        train(ExactModel(), **validation)


class StatefulModel(ExactModel):
    _version = 7

    def __init__(self):
        super().__init__()
        self.extra = {"epochs": []}
        self.loaded_version = None

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.scale.add_(1)
            self.extra["epochs"].append(self.scale.item())
        return super().forward(x)

    def get_extra_state(self):
        return self.extra

    def set_extra_state(self, state):
        self.extra = state

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, *args, **kwargs):
        self.loaded_version = local_metadata.get("version")
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, *args, **kwargs)


@pytest.mark.parametrize("with_checkpoint", [False, True])
def test_best_state_is_deep_copied_with_extra_state_and_metadata(tmp_path, with_checkpoint):
    model = StatefulModel()
    losses = iter([1.0, 2.0, 3.0])

    def loss(prediction, target):
        value = 0.0 if model.training else next(losses)
        return prediction.sum() * 0 + value

    path = tmp_path / "nested" / "best.pt" if with_checkpoint else None
    history = train(model, loss_fn=loss, X_val=np.ones((2, 1)), y_val=np.ones(2),
                    checkpoint_path=path)
    assert len(history.train_losses) == 3
    assert model.scale.item() == 2
    assert model.extra == {"epochs": [2.0]}
    assert model.loaded_version == 7
    assert history.best_epoch == 0
    if path:
        saved = torch.load(path, weights_only=False)
        assert saved["epoch"] == 0
        assert saved["val_loss"] == 1
        assert saved["model_state_dict"]["_extra_state"] == {"epochs": [2.0]}


@pytest.mark.parametrize("epochs", [0, 3])
def test_no_current_best_never_reads_stale_checkpoint(tmp_path, epochs, monkeypatch):
    path = stale_checkpoint(tmp_path)
    original = path.read_bytes()
    model = ExactModel()

    def loss(prediction, target):
        return prediction.sum() * 0 + (0.0 if model.training else float("nan"))

    def forbidden_load(*args, **kwargs):
        raise AssertionError("ordinary training must not read a previous checkpoint")

    monkeypatch.setattr(torch, "load", forbidden_load)
    train(model, epochs=epochs, loss_fn=loss, X_val=np.ones((2, 1)), y_val=np.ones(2),
          checkpoint_path=path)
    assert model.scale.item() == 1
    assert path.read_bytes() == original


def test_custom_uncertainty_loss_retains_unequal_output_shape_support():
    model = ExactModel(duplicate=True)

    def uncertainty_loss(prediction, target):
        assert prediction.shape == (8, 2)
        assert target.shape == (8,)
        return ((prediction[:, 0] - target) ** 2).mean()

    assert len(train(model, epochs=1, loss_fn=uncertainty_loss).train_losses) == 1


def test_cv_uses_uniform_average_multioutput_r2(monkeypatch):
    pytest.importorskip("sklearn")
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold

    class OffsetModel(ExactModel):
        def forward(self, x):
            return x * self.scale + 1

    monkeypatch.setattr(training, "train_model", lambda *args, **kwargs: None)
    base = np.arange(12, dtype=np.float32)
    x = np.column_stack([base, base * 2 + 100])
    expected = [r2_score(x[index], x[index] + 1, multioutput="uniform_average")
                for _, index in KFold(n_splits=3, shuffle=True, random_state=42).split(x)]
    result = training.cross_validate_torch(OffsetModel, x, x, n_folds=3)
    assert result["fold_r2s"] == pytest.approx(expected)
    assert result["fold_rmses"] == [1.0] * 3
