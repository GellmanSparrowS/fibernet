"""Implicit MSE refuses broadcasting; custom losses keep their own contract."""
import copy
from types import MethodType

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from fibernet.ml import training


class PointModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * self.scale


def fit(model, x, y, **kwargs):
    options = dict(epochs=1, batch_size=8, scheduler_type="step",
                   verbose=False, device="cpu", num_workers=0,
                   optimizer=torch.optim.SGD(model.parameters(), lr=0))
    options.update(kwargs)
    return training.train_model(model, x, y, **options)


@pytest.mark.parametrize("shape,target_shape", [
    ((2, 2), (2, 1)), ((1, 2), (1, 1)),
    ((2, 1, 1), (2, 1)), ((2, 1), (2, 1, 1)),
])
@pytest.mark.parametrize("schedule", ["step", "warmup_cosine"])
def test_first_bad_batch_does_not_step_or_mutate_parameters(
        shape, target_shape, schedule, monkeypatch):
    model = PointModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
    # Seed nonempty accumulated optimizer state, without stepping.
    optimizer.state[model.scale]["momentum_buffer"] = torch.tensor(3.)
    before = copy.deepcopy(optimizer.state_dict())
    parameter = model.scale.detach().clone()
    calls = []
    monkeypatch.setattr(optimizer, "step", MethodType(
        lambda self, *a, **k: calls.append("step"), optimizer))
    with pytest.raises(ValueError, match="Default MSE training prediction shape"):
        fit(model, torch.ones(shape), torch.ones(target_shape),
            optimizer=optimizer, scheduler_type=schedule)
    assert calls == []
    assert torch.equal(model.scale, parameter)
    after = optimizer.state_dict()
    # Ordinary scheduler setup may alter LR metadata; no update step is allowed.
    assert after["state"].keys() == before["state"].keys()
    assert torch.equal(after["state"][0]["momentum_buffer"],
                       before["state"][0]["momentum_buffer"])


def test_validation_mismatch_precedes_checkpoint_selection(tmp_path, monkeypatch):
    model = PointModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    path = tmp_path / "best.pt"
    path.write_bytes(b"previous checkpoint remains untouched")
    original = path.read_bytes()
    saves = []
    updates = []
    monkeypatch.setattr(torch, "save", lambda *a, **k: saves.append(a))
    monkeypatch.setattr(training.TrainingHistory, "update", lambda *a, **k: updates.append(a))
    with pytest.raises(ValueError, match="Default MSE validation prediction shape"):
        fit(model, torch.ones((2, 1)), torch.zeros((2, 1)), optimizer=optimizer,
            X_val=torch.ones((2, 2)), y_val=torch.ones((2, 1)), checkpoint_path=path)
    assert saves == [] and updates == []
    assert path.read_bytes() == original
    # Valid training already ran: this guard does not promise transactional rollback.
    assert model.scale.item() < 1


@pytest.mark.parametrize("prediction,target,expected", [
    ([[1, 2], [3, 4]], [[0, 0], [0, 0]], 7.5),
    ([[1], [3]], [0, 1], 2.5),
    ([1, 3], [[0], [1]], 2.5),
    ([[1]], [0], 1.),
    ([[[1]], [[3]]], [[[0]], [[1]]], 2.5),
])
def test_default_mse_matches_hand_computed_equal_shape_or_scalar_loss(
        prediction, target, expected):
    model = PointModel()
    x, y = torch.tensor(prediction, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
    history = fit(model, x, y, X_val=x, y_val=y)
    assert history.train_losses == pytest.approx([expected])
    assert history.val_losses == pytest.approx([expected])


def test_explicit_mse_retains_broadcast_semantics():
    model = PointModel()
    with pytest.warns(UserWarning, match="target size"):
        history = fit(model, [[1, 2], [3, 4]], [[1], [3]], loss_fn=torch.nn.MSELoss())
    assert history.train_losses == [.5]


def test_metric_adapter_cannot_repair_default_loss_shape():
    calls = []
    def adapter(raw):
        calls.append(True)
        return raw[:, :1]
    with pytest.raises(ValueError, match="Default MSE training prediction shape"):
        fit(PointModel(), torch.ones((2, 2)), torch.ones((2, 1)), metric_prediction=adapter)
    assert calls == []


def test_custom_loss_preserves_non_scalar_singleton_normalization():
    shapes = []
    def loss(prediction, target):
        shapes.append((tuple(prediction.shape), tuple(target.shape)))
        return (prediction - target).square().mean()
    history = fit(PointModel(), torch.ones((2, 1, 1)), torch.zeros((2, 1)), loss_fn=loss)
    assert shapes == [((2, 1), (2, 1))]
    assert history.train_losses == [1.]


def test_graph_derived_tabular_baseline_has_explicit_groups_features_labels():
    """Ordered/perturbed curved-parallel graphs feed a tiny tabular contract only."""
    from fibernet.core.structure_graph import StructureGraph
    from fibernet.ml.graph_data import graph_batch

    rows = []
    source_groups = ["mother-a", "mother-a", "mother-b", "mother-b",
                     "mother-c", "mother-c"]
    feature_names = ("physical_edges_over_42", "total_chord_length_over_40m",
                     "total_polyline_length_over_40m")
    label_names = ("synthetic_2chord_plus_polyline",)
    rng = np.random.default_rng(1729)
    for specimen in range(6):
        graph = StructureGraph(dimension=2)
        positions = np.array([[i, j, 0.] for i in range(5) for j in range(5)], dtype=float)
        positions *= 1 + (specimen // 2) * .1
        if specimen % 2:
            positions[:, :2] += rng.uniform(-.05, .05, (25, 2))
        nodes = [graph.add_node(position) for position in positions]
        for i in range(5):
            for j in range(5):
                row = 5 * i + j
                for neighbor in ([row + 5] if i < 4 else []) + ([row + 1] if j < 4 else []):
                    graph.add_edge(nodes[row], nodes[neighbor], radius=.1)
        for a, b in ((0, 1), (23, 24)):
            midpoint = (positions[a] + positions[b]) / 2 + np.array([.2, 0., 0.])
            graph.add_edge(nodes[a], nodes[b], radius=.1, allow_parallel=True,
                           internal_points=np.array([midpoint]))
        batch = graph_batch([graph], graph_ids=[f"synthetic-{specimen}"], length_unit="m",
                            coordinate_frame="specimen_xy",
                            edge_features=("chord_length", "polyline_length"))
        assert len(batch.node_ids) == 25 and len(batch.edge_ids) == 42
        assert batch.message_index.shape[1] == 84
        assert batch.edge_feature_available.all()
        chord, polyline = batch.edge_features.sum(axis=0)
        assert polyline > chord
        rows.append([len(batch.edge_ids) / 42, chord / 40, polyline / 40])
    x = np.asarray(rows, dtype=np.float32)
    # Explicit analytic fixture labels, not mechanical solver observations.
    y = (2 * x[:, 1:2] + x[:, 2:3]).copy()
    assert set(source_groups[:4]).isdisjoint(source_groups[4:])
    assert x.shape == (6, len(feature_names)) and y.shape == (6, len(label_names))

    class LengthBaseline(PointModel):
        def forward(self, features):
            return (features[:, 1:2] + features[:, 2:3]) * self.scale

    history = fit(LengthBaseline(), x[:4], y[:4], X_val=x[4:], y_val=y[4:])
    # Frozen model: checks the loss contract, not optimization quality/generalization.
    assert history.train_losses == pytest.approx([float(np.mean(x[:4, 1] ** 2))])
    assert history.val_losses == pytest.approx([float(np.mean(x[4:, 1] ** 2))])
