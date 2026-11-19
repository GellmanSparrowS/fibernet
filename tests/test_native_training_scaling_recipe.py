"""Execute the documented train-only preprocessing and named prediction fences."""

from dataclasses import replace
from hashlib import sha256
import json
import pickle
from pathlib import Path
import re

import numpy as np
import pytest

from fibernet.ml.dataset import DatasetSample, FiberNetDataset


pytest.importorskip("sklearn")
DOC = Path(__file__).resolve().parents[1] / "docs/architecture/native-dataset-training-usage.md"


@pytest.fixture
def recipe():
    contents = DOC.read_text(encoding="utf-8")
    section = contents.split("## Standardize on training groups\n", 1)[1].split("\n## ", 1)[0]
    fences = re.findall(r"^```python\n(.*?)^```$", section, flags=re.MULTILINE | re.DOTALL)
    assert len(fences) == 2
    train_scope, predict_scope = {}, {}
    exec(compile(fences[0], str(DOC), "exec"), train_scope)
    exec(compile(fences[1], str(DOC), "exec"), predict_scope)
    return train_scope["fit_scaled_groups"], predict_scope["predict_scaled_groups"]


@pytest.fixture
def source():
    groups = [f"origin-{name}" for name in "abcd" for _ in range(3)]
    rows = [
        DatasetSample(
            f"row-{i}", "fixture", (1, 1), {"variant": i % 3},
            {"length_m": float(i if group != "origin-b" else i + 1e9),
             "radius_m": 7.0 if group != "origin-b" else 1000.0,
             "unused_missing": None},
            {"force_N": float(i * i + 1), "other_target": None},
            {"parent": group},
        )
        for i, group in enumerate(groups)
    ]
    return FiberNetDataset(rows, ["length_m", "radius_m", "unused_missing"],
                           ["force_N", "other_target"]), groups


def test_documented_scaler_and_model_share_the_exact_source_split(recipe, source, tmp_path):
    fit, predict = recipe
    dataset, groups = source
    names = ["length_m", "radius_m"]
    before = [replace(row, features=dict(row.features), labels=dict(row.labels),
                      metadata=dict(row.metadata)) for row in dataset.samples]
    directory = tmp_path / "artifacts"
    metrics = fit(dataset, names, "force_N", groups, directory)
    state = json.loads((directory / "preprocessing.json").read_text())
    split = metrics["training_data"]["split"]
    assert split["test_rows"] == [3, 4, 5]
    train = [i for i in range(12) if i not in (3, 4, 5)]
    assert state["fit_rows"] == split["train_rows"] == train
    assert state["fit_groups"] == [groups[i] for i in train]
    assert "origin-b" not in state["fit_groups"]
    assert state["fit_sample_ids"] == [dataset.samples[i].sample_id for i in train]
    assert state["raw_names"] == names and state["z_names"] == ["z_length_m", "z_radius_m"]
    raw = np.array([[row.features[name] for name in names] for row in dataset.samples])
    np.testing.assert_allclose(state["mean"], raw[train].mean(axis=0))
    np.testing.assert_allclose(state["scale"], [raw[train, 0].std(), 1.0])
    assert state["mean"][0] < 20.0  # The held-out sentinel did not affect the fit.
    assert dataset.samples == before

    model_bytes = (directory / "model.pkl").read_bytes()
    assert sha256(model_bytes).hexdigest() == state["model_sha256"]
    artifact = pickle.loads(model_bytes)  # Created locally by this test.
    assert artifact["features"] == state["z_names"]
    raw_rows = [{name: row.features[name] for name in reversed(names)} for row in dataset.samples]
    reconstructed = (raw - state["mean"]) / state["scale"]
    np.testing.assert_allclose(predict(directory, raw_rows)["predictions"],
                               artifact["model"].predict(reconstructed))

    (directory / "model.pkl").write_bytes(b"different model bytes")
    with pytest.raises(ValueError, match="does not match the saved scaler"):
        predict(directory, raw_rows)

    (directory / "model.pkl").write_bytes(model_bytes)
    state["z_names"].reverse()
    (directory / "preprocessing.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="scaler output columns"):
        predict(directory, raw_rows)


@pytest.mark.parametrize(("name", "declared", "message"), [
    ("force_N", True, "exclude every declared target"),
    ("other_target", True, "exclude every declared target"),
    ("undeclared_value", False, "Select declared raw Dataset feature names"),
])
def test_raw_input_admission_precedes_scaling(recipe, source, name, declared, message,
                                              tmp_path, monkeypatch):
    from sklearn.preprocessing import StandardScaler

    fit, _ = recipe
    dataset, groups = source
    if declared:
        dataset.feature_names.append(name)
    for sample in dataset.samples:
        sample.features[name] = 12.0

    def unexpected_fit(*args, **kwargs):
        pytest.fail("The recipe attempted to fit a scaler before checking target names")

    monkeypatch.setattr(StandardScaler, "fit", unexpected_fit)
    directory = tmp_path / "uncreated"
    with pytest.raises(ValueError, match=message):
        fit(dataset, ["length_m", name], "force_N", groups, directory)
    assert not directory.exists()


@pytest.mark.parametrize(("dtype", "base"), [(np.float16, 10000), (np.float32, 1000000)])
def test_adjacent_low_precision_inputs_match_float64_training(recipe, tmp_path, dtype, base):
    fit, predict = recipe
    low = dtype(base)
    high = np.nextafter(low, dtype(np.inf))
    raw_values = [low, high]
    groups = [f"parent-{i}" for i in range(4) for _ in raw_values]
    samples = [
        DatasetSample(f"row-{i}", "fixture", (1, 1), {}, {"reading": raw_values[i % 2]},
                      {"response": float(i % 2)})
        for i in range(len(groups))
    ]
    dataset = FiberNetDataset(samples, ["reading"], ["response"])
    directory = tmp_path / "artifacts"
    fit(dataset, ["reading"], "response", groups, directory)
    python_rows = [{"reading": float(value)} for value in raw_values]
    numeric_rows = [{"reading": value} for value in raw_values]
    expected = predict(directory, python_rows)["predictions"]
    np.testing.assert_allclose(expected, [0., 1.], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(predict(directory, numeric_rows)["predictions"], expected)


def test_raw_scalar_admission_survives_standardization(recipe, source, tmp_path):
    fit, predict = recipe
    dataset, groups = source
    directory = tmp_path / "artifacts"
    fit(dataset, ["length_m"], "force_N", groups, directory)
    invalid = ["1", np.ma.masked, np.complex64(1 + 0j), np.inf, 10**400]
    for value in invalid:
        with pytest.raises(ValueError, match="finite real numbers"):
            predict(directory, [{"length_m": value}])
    expected = predict(directory, [{"length_m": 1.0}])
    assert predict(directory, [{"length_m": np.bool_(True)}]) == expected
