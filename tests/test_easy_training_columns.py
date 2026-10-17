"""CSV selection boundary; tiny fixtures are API checks, not model validation."""
import csv
import pickle

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from fibernet import train_model
from fibernet.ml.dataset import DatasetSample, FiberNetDataset


@pytest.fixture
def dataset_csv(tmp_path):
    rng = np.random.default_rng(20261016)
    samples = []
    for i in range(64):
        x, noise, independent = rng.normal(size=3)
        y = 3 * x + noise
        samples.append(DatasetSample(f"square-{i}", "square", (1, 1), {},
                                     {"x": x, "independent": independent},
                                     {"response": y, "copy": y}))
    dataset = FiberNetDataset(samples, feature_names=["independent", "x"],
                             target_names=["response", "copy"])
    path = tmp_path / "dataset.csv"
    dataset.to_csv(str(path))
    return path


@pytest.fixture
def before_training(monkeypatch):
    """Every invalid request must fail before either splitting or fitting."""
    def forbidden(*args, **kwargs):
        pytest.fail("invalid input reached split or estimator fit")
    monkeypatch.setattr("sklearn.model_selection.train_test_split", forbidden)
    for cls in (LinearRegression, RandomForestRegressor, GradientBoostingRegressor):
        monkeypatch.setattr(cls, "fit", forbidden)


@pytest.mark.parametrize("numeric_ids", [False, True])
def test_dataset_auto_excludes_all_labels_and_metadata(dataset_csv, tmp_path, numeric_ids):
    if numeric_ids:
        frame = pd.read_csv(dataset_csv)
        frame["sample_id"] = np.arange(len(frame))
        frame.to_csv(dataset_csv, index=False)
    artifact_path = tmp_path / "auto.pkl"
    auto = train_model(dataset_csv, target="label_response", model_type="lr",
                       save_path=artifact_path)
    explicit = train_model(dataset_csv, target="label_response", model_type="lr",
                           features=["feat_independent", "feat_x"])
    assert auto == explicit
    with artifact_path.open("rb") as handle:
        artifact = pickle.load(handle)
    assert artifact["features"] == ["feat_independent", "feat_x"]
    assert artifact["model"].n_features_in_ == 2


@pytest.mark.parametrize("features,match", [
    ([], "nonempty"), ("feat_x", "list"), (["feat_x", "feat_x"], "duplicate"),
    ([""], "nonempty"), ([None], "nonempty"), ([3], "nonempty"),
    (["missing"], "missing"), (["label_response"], "label_|target"),
    (["label_copy"], "label_"),
])
def test_invalid_explicit_features(dataset_csv, before_training, features, match):
    with pytest.raises(ValueError, match=match):
        train_model(dataset_csv, target="label_response", features=features)


@pytest.mark.parametrize("target", ["max_force", "absent", "", None, 3])
def test_missing_or_invalid_target_is_actionable(dataset_csv, before_training, target):
    with pytest.raises(ValueError, match="target") as exc:
        train_model(dataset_csv, target=target, features=["feat_x"])
    if target in ("max_force", "absent"):
        assert "label_response" in str(exc.value)
        assert "target=" in str(exc.value)


@pytest.mark.parametrize("header", [
    ["x", "x", "y"], ["x", "y", "y"], ["x", "y", "unused", "unused"],
    ["x,quoted", "x,quoted", "y"], ["x", "", "y"],
])
@pytest.mark.parametrize("automatic", [False, True])
def test_raw_headers_before_pandas_mangling(tmp_path, before_training, monkeypatch, header, automatic):
    path = tmp_path / "raw.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        handle.write("\n  \n")
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows([[1] * len(header)] * 10)
    def forbidden(*args, **kwargs):
        pytest.fail("invalid raw header reached pandas")
    monkeypatch.setattr(pd, "read_csv", forbidden)
    with pytest.raises(ValueError, match="header"):
        train_model(path, target="y", features=None if automatic else [header[0]])


@pytest.mark.parametrize("column", ["x", "y"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, "text", "1+2j"])
@pytest.mark.parametrize("model_type", ["lr", "rf", "gb"])
def test_selected_invalid_values_before_split(tmp_path, before_training, column, bad, model_type):
    frame = pd.DataFrame({"x": np.arange(20, dtype=float), "y": np.arange(20, dtype=float)})
    frame[column] = frame[column].astype(object)
    # Index 0 is in the existing held-out split: validation covers ALL rows.
    assert 0 in train_test_split(np.arange(20), test_size=.2, random_state=42)[1]
    frame.loc[0, column] = bad
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=column):
        train_model(path, target="y", features=["x"], model_type=model_type)


@pytest.mark.parametrize("mutation", ["arbitrary", "batch", "missing_grid", "extra", "empty_feature_suffix", "empty_label_suffix", "non_label_target"])
def test_ambiguous_auto_refuses(dataset_csv, before_training, mutation):
    frame = pd.read_csv(dataset_csv)
    if mutation == "arbitrary":
        frame = pd.DataFrame({"x": range(64), "label_response": range(64)})
    elif mutation == "batch":
        frame = pd.DataFrame({"grid_x": range(64), "energy": range(64), "label_response": range(64)})
    elif mutation == "missing_grid":
        frame = frame.drop(columns="grid_y")
    elif mutation == "extra":
        frame["energy"] = 1
    elif mutation == "empty_feature_suffix":
        frame = frame.rename(columns={"feat_x": "feat_"})
    elif mutation == "empty_label_suffix":
        frame = frame.rename(columns={"label_copy": "label_"})
    target = "feat_x" if mutation == "non_label_target" else "label_response"
    frame.to_csv(dataset_csv, index=False)
    with pytest.raises(ValueError, match="features="):
        train_model(dataset_csv, target=target)


@pytest.mark.parametrize("model_type,cls,kwargs", [
    ("lr", LinearRegression, {}),
    ("rf", RandomForestRegressor, {"n_estimators": 100, "random_state": 42}),
    ("gb", GradientBoostingRegressor, {"n_estimators": 100, "random_state": 42}),
])
def test_explicit_path_preserves_original_estimator_and_artifact(dataset_csv, tmp_path, model_type, cls, kwargs):
    frame = pd.read_csv(dataset_csv)
    features = ["feat_x", "feat_independent"]  # Deliberately reverse CSV order.
    X, y = frame[features].values, frame["label_response"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)
    original = cls(**kwargs).fit(X_train, y_train)
    prediction = original.predict(X_test)
    path = tmp_path / f"{model_type}.pkl"
    metrics = train_model(dataset_csv, target="label_response", features=features,
                          model_type=model_type, test_size=.25, save_path=path)
    assert metrics == {"r2": r2_score(y_test, prediction),
                       "rmse": np.sqrt(mean_squared_error(y_test, prediction)),
                       "n_samples": len(frame), "features": features}
    with path.open("rb") as handle:
        artifact = pickle.load(handle)
    assert set(artifact) == {"model", "features", "metrics", "artifact_version", "target"}
    assert artifact["features"] == features
    assert artifact["metrics"] == metrics
    assert type(artifact["model"]) is cls
    np.testing.assert_array_equal(artifact["model"].predict(X), original.predict(X))


def test_explicit_arbitrary_numeric_names_ignore_unselected_metadata(tmp_path):
    path = tmp_path / "arbitrary.csv"
    pd.DataFrame({"sample_id": range(20), "x,quoted": range(20), "y": range(20),
                  "unused": [None, "text"] * 10, "label_unused": "bad"}).to_csv(path, index=False)
    metrics = train_model(path, target="y", features=["sample_id", "x,quoted"], model_type="lr")
    assert metrics["features"] == ["sample_id", "x,quoted"]


def test_explicit_reserved_label_on_arbitrary_csv(tmp_path, before_training):
    path = tmp_path / "arbitrary.csv"
    pd.DataFrame({"x": range(20), "y": range(20), "label_measurement": range(20)}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="label_"):
        train_model(path, target="y", features=["label_measurement"])


def test_explicit_target_on_arbitrary_csv(tmp_path, before_training):
    path = tmp_path / "arbitrary.csv"
    pd.DataFrame({"x": range(20), "y": range(20)}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="target"):
        train_model(path, target="y", features=["y"])


@pytest.mark.parametrize("row", ["1,2,3", "1", '""', '"  "'])
def test_invalid_record_width_before_pandas(tmp_path, before_training, monkeypatch, row):
    path = tmp_path / "width.csv"
    path.write_text("x,y\n" + row + "\n")
    def forbidden(*args, **kwargs):
        pytest.fail("invalid record width reached pandas")
    monkeypatch.setattr(pd, "read_csv", forbidden)
    with pytest.raises(ValueError, match="record 2.*fields.*header"):
        train_model(path, target="y", features=["x"])


def test_valid_bom_blanks_quoted_comma_and_multiline(tmp_path):
    path = tmp_path / "quoted.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        handle.write("\n  \n")
        writer = csv.writer(handle)
        writer.writerow(["x,quoted", "y", "metadata"])
        for i in range(20):
            writer.writerow([i, 2 * i, "multiple, fields\nand lines"])
            handle.write(" \n\n")
    metrics = train_model(path, target="y", features=["x,quoted"], model_type="lr")
    assert metrics["n_samples"] == 20
    assert metrics["features"] == ["x,quoted"]
    assert metrics["r2"] == 1


def test_dataset_header_order_not_fixed(dataset_csv):
    frame = pd.read_csv(dataset_csv)
    frame = frame[list(reversed(frame.columns))]
    frame.to_csv(dataset_csv, index=False)
    metrics = train_model(dataset_csv, target="label_response", model_type="lr")
    assert metrics["features"] == ["feat_x", "feat_independent"]


def test_malformed_quote_rejected(tmp_path, before_training):
    path = tmp_path / "quote.csv"
    path.write_text('x,y\n"unclosed,2\n')
    with pytest.raises(ValueError, match="Malformed CSV record"):
        train_model(path, target="y", features=["x"])
