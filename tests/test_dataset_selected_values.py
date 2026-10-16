"""Selected dataset values must not fabricate successful training labels."""
import csv

import numpy as np
import pytest

from fibernet.ml.dataset import DatasetSample, FiberNetDataset


def make_dataset():
    return FiberNetDataset([
        DatasetSample("zero-0", "audit", (1, 1), {},
                      {"x": 0.0, "unused": None}, {"y": 0.0, "unused": np.nan},
                      {"contact": {"status": "not_requested", "value": None},
                       "fracture": {"status": "unavailable", "value": None}}),
        DatasetSample("row-1", "audit", (1, 1), {}, {"x": np.int64(2)},
                      {"y": np.float32(3)}),
    ], feature_names=["x"], target_names=["y"])


def read_axis(dataset, axis):
    return dataset.features if axis == "features" else dataset.all_labels


@pytest.mark.parametrize("axis", ["features", "labels"])
@pytest.mark.parametrize("value", [None, np.nan, np.inf, -np.inf, 1 + 0j, "1", [1], object(), 10**1000])
def test_invalid_selected_values_rejected_with_context(axis, value):
    ds = make_dataset()
    name = "x" if axis == "features" else "y"
    getattr(ds.samples[1], axis)[name] = value
    with pytest.raises(ValueError, match=rf"row-1.*row 1.*'{name}'"):
        read_axis(ds, axis)


@pytest.mark.parametrize("axis", ["features", "labels"])
def test_missing_selected_value_rejected_after_mutation(axis):
    ds = make_dataset()
    read_axis(ds, axis)
    name = "x" if axis == "features" else "y"
    del getattr(ds.samples[1], axis)[name]
    with pytest.raises(ValueError, match=rf"row-1.*'{name}'.*missing"):
        read_axis(ds, axis)


@pytest.mark.parametrize("axis", ["features", "labels"])
@pytest.mark.parametrize("bad_name", [None, 1, "", " \t", []])
def test_invalid_selected_names_rejected_even_without_samples(axis, bad_name):
    ds = FiberNetDataset([], ["x"], ["y"])
    names = ds.feature_names if axis == "features" else ds.target_names
    names.append(bad_name)
    with pytest.raises(ValueError, match="Invalid selected.*column 1"):
        read_axis(ds, axis)


@pytest.mark.parametrize("axis", ["features", "labels"])
def test_duplicate_names_rejected_after_selection_mutation(axis):
    ds = make_dataset()
    read_axis(ds, axis)
    names = ds.feature_names if axis == "features" else ds.target_names
    names.append(names[0])
    with pytest.raises(ValueError, match="Duplicate selected"):
        read_axis(ds, axis)


@pytest.mark.parametrize("axis", ["features", "labels"])
def test_string_is_not_a_selected_name_sequence(axis):
    ds = make_dataset()
    setattr(ds, "feature_names" if axis == "features" else "target_names", "xy")
    with pytest.raises(ValueError, match="sequence of names"):
        read_axis(ds, axis)


@pytest.mark.parametrize("axis", ["features", "labels"])
@pytest.mark.parametrize("names", [None, {"x": 1}, {"x"}, 1])
def test_malformed_mutated_selection_rejected(axis, names):
    ds = make_dataset()
    setattr(ds, "feature_names" if axis == "features" else "target_names", names)
    with pytest.raises(ValueError, match="sequence of names"):
        read_axis(ds, axis)


@pytest.mark.parametrize("axis", ["features", "labels"])
@pytest.mark.parametrize("mapping", [None, [1]])
def test_malformed_selected_mapping_rejected_with_context(axis, mapping):
    ds = make_dataset()
    setattr(ds.samples[1], axis, mapping)
    with pytest.raises(ValueError, match="row-1.*row 1.*mapping"):
        read_axis(ds, axis)


@pytest.mark.parametrize("value,expected", [(False, 0), (True, 1),
                                            (np.bool_(False), 0), (np.bool_(True), 1)])
def test_boolean_values_preserve_explicit_binary_convention(value, expected):
    ds = make_dataset()
    ds.samples[0].features["x"] = value
    ds.samples[0].labels["y"] = value
    assert ds.features[0, 0] == ds.all_labels[0, 0] == expected


@pytest.mark.parametrize("axis", ["features", "labels"])
@pytest.mark.parametrize("operation", ["save", "to_csv"])
@pytest.mark.parametrize("existing", [False, True])
def test_invalid_export_preflight_preserves_files(tmp_path, axis, operation, existing):
    ds = make_dataset()
    name = "x" if axis == "features" else "y"
    del getattr(ds.samples[1], axis)[name]
    dest = tmp_path / ("saved" if operation == "save" else "dataset.csv")
    if existing:
        if operation == "save":
            dest.mkdir()
            for filename in ("features.npy", "labels.npy", "metadata.json"):
                (dest / filename).write_bytes(b"existing contents")
        else:
            dest.write_bytes(b"existing CSV")
    before = {str(path.relative_to(tmp_path)): path.read_bytes()
              for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match=rf"row-1.*'{name}'.*missing"):
        getattr(ds, operation)(str(dest))
    after = {str(path.relative_to(tmp_path)): path.read_bytes()
             for path in tmp_path.rglob("*") if path.is_file()}
    assert before == after
    assert dest.exists() == existing


def test_true_zero_unselected_unavailable_and_metadata_roundtrip(tmp_path):
    ds = make_dataset()
    np.testing.assert_array_equal(ds.features, [[0], [2]])
    np.testing.assert_array_equal(ds.all_labels, [[0], [3]])
    np.testing.assert_array_equal(ds.labels("y"), [0, 3])
    assert ds.features.dtype == ds.all_labels.dtype == np.float64
    ds.save(str(tmp_path / "saved"))
    loaded = FiberNetDataset.load(str(tmp_path / "saved"))
    np.testing.assert_array_equal(loaded.features, ds.features)
    np.testing.assert_array_equal(loaded.all_labels, ds.all_labels)
    assert loaded.samples[0].metadata == ds.samples[0].metadata
    loaded.to_csv(str(tmp_path / "data.csv"))
    with (tmp_path / "data.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert float(rows[0]["feat_x"]) == float(rows[0]["label_y"]) == 0


@pytest.mark.parametrize("feature_names,target_names", [([], []), (["x"], ["y"]), (["a", "b"], ["y"])])
def test_empty_declared_matrices_roundtrip(tmp_path, feature_names, target_names):
    ds = FiberNetDataset([], feature_names, target_names)
    assert ds.features.shape == (0, len(feature_names))
    assert ds.all_labels.shape == (0, len(target_names))
    if target_names:
        assert ds.labels(target_names[0]).shape == (0,)
    ds.save(str(tmp_path / "saved"))
    loaded = FiberNetDataset.load(str(tmp_path / "saved"))
    assert loaded.features.shape == ds.features.shape
    assert loaded.all_labels.shape == ds.all_labels.shape
    ds.to_csv(str(tmp_path / "empty.csv"))
    with (tmp_path / "empty.csv").open() as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 1
    assert len(rows[0]) == 4 + len(feature_names) + len(target_names)


def test_empty_selections_with_samples_are_two_dimensional():
    ds = FiberNetDataset(make_dataset().samples, [], [])
    assert ds.features.shape == ds.all_labels.shape == (2, 0)


def test_empty_split_preserves_declared_shapes():
    ds = make_dataset()
    ds.samples = ds.samples[:1]
    train, val, test = ds.split()
    assert train.features.shape == val.features.shape == (0, 1)
    assert train.labels("y").shape == val.labels("y").shape == (0,)
    assert test.features.shape == (1, 1)


def test_features_do_not_consume_unavailable_targets():
    ds = make_dataset()
    ds.samples[0].labels["y"] = None
    np.testing.assert_array_equal(ds.features, [[0], [2]])
    with pytest.raises(ValueError, match="target 'y'.*None"):
        ds.labels("y")


def test_first_sample_inference_rejects_later_missing_key():
    ds = FiberNetDataset([
        DatasetSample("first", "audit", (1, 1), {}, {"x": 0}, {"y": 0}),
        DatasetSample("later", "audit", (1, 1), {}, {}, {}),
    ])
    with pytest.raises(ValueError, match="later.*feature 'x'.*missing"):
        ds.features
    with pytest.raises(ValueError, match="later.*target 'y'.*missing"):
        ds.all_labels


def test_loading_legacy_zero_does_not_invent_missingness(tmp_path):
    ds = make_dataset()
    ds.save(str(tmp_path))
    # The format has no per-field provenance. Genuine and historic imputed zeros
    # are indistinguishable; the loader must not claim to restore missingness.
    loaded = FiberNetDataset.load(str(tmp_path))
    assert loaded.samples[0].labels["y"] == 0.0
    assert loaded.all_labels[0, 0] == 0.0
