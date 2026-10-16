"""Saved triples must be consistent before a dataset is returned.

Historical byte fixtures below were captured from the trusted original
FiberNetDataset.save implementation by the 2026-10-16 read-only audit. They are
plain numeric NPY bytes and JSON, never pickle or an external runtime dependency.
"""
import base64
import json
import warnings

import numpy as np
import pytest

import fibernet.ml.dataset as module
from fibernet.ml.dataset import FiberNetDataset


FILES = ("features.npy", "labels.npy", "metadata.json")
BASE_X = np.array([[1., 11.], [2., 22.]])
BASE_Y = np.array([[101., 1001.], [202., 2002.]])


def metadata(n=2, features=None, targets=None):
    return {
        "feature_names": ["x", "z"] if features is None else features,
        "target_names": ["y", "w"] if targets is None else targets,
        "n_samples": n,
        "samples": [{"id": f"sample-{i}", "unit": "audit", "grid": [1, 1],
                     "params": {"arbitrary": None},
                     "metadata": {"availability": {"status": "unavailable", "value": None},
                                  "legacy_nan": float("nan")}} for i in range(n)],
    }


def write_dataset(path, meta=None, X=None, Y=None):
    path.mkdir(exist_ok=True)
    np.save(path / "features.npy", BASE_X if X is None else X, allow_pickle=False)
    np.save(path / "labels.npy", BASE_Y if Y is None else Y, allow_pickle=False)
    write_metadata(path, metadata() if meta is None else meta)
    return path


def write_metadata(path, meta):
    (path / "metadata.json").write_text(json.dumps(meta, indent=2))


def snapshot(path):
    return {p.name: p.read_bytes() for p in path.iterdir() if p.is_file()}


def reject(path, match):
    class CannotReturnDataset(FiberNetDataset):
        def __init__(self, *args, **kwargs):
            raise AssertionError("Malformed input reached the dataset constructor")

    before = snapshot(path)
    with pytest.raises(ValueError, match=match):
        CannotReturnDataset.load(path)
    assert snapshot(path) == before


@pytest.mark.parametrize("axis", ["features", "labels"])
@pytest.mark.parametrize("names", ["xz", {"x": 1}, None, False, 2])
@pytest.mark.parametrize("empty", [False, True])
def test_name_containers_rejected_even_without_rows(tmp_path, axis, names, empty):
    meta = metadata(0 if empty else 2)
    meta["feature_names" if axis == "features" else "target_names"] = names
    path = write_dataset(tmp_path, meta)
    reject(path, "metadata.json:.*names must be an array")


@pytest.mark.parametrize("axis", ["features", "labels"])
@pytest.mark.parametrize("bad", ["", " \t", 1, None, False, [], {}])
@pytest.mark.parametrize("empty", [False, True])
def test_name_elements_rejected_before_dictionary_construction(tmp_path, axis, bad, empty):
    meta = metadata(0 if empty else 2)
    meta["feature_names" if axis == "features" else "target_names"] = ["first", bad]
    path = write_dataset(tmp_path, meta)
    reject(path, "metadata.json: invalid.*column 1")


@pytest.mark.parametrize("axis", ["features", "labels"])
@pytest.mark.parametrize("empty", [False, True])
def test_duplicate_names_rejected_before_dictionary_overwrite(tmp_path, monkeypatch, axis, empty):
    meta = metadata(0 if empty else 2)
    meta["feature_names" if axis == "features" else "target_names"] = ["same", "same"]
    path = write_dataset(tmp_path, meta)

    def no_sample(*args, **kwargs):
        raise AssertionError("Name validation must precede sample construction")

    monkeypatch.setattr(module, "DatasetSample", no_sample)
    reject(path, "metadata.json: duplicate.*'same'.*column 1")


@pytest.mark.parametrize("value", [-1, True, False, 2.0, "2", None, [], {}, float("nan")])
def test_count_requires_nonnegative_nonbool_integer(tmp_path, value):
    meta = metadata(); meta["n_samples"] = value
    reject(write_dataset(tmp_path, meta), "n_samples must be a nonnegative integer")


@pytest.mark.parametrize("count", [0, 1, 3])
def test_count_equals_number_of_records(tmp_path, count):
    meta = metadata(); meta["n_samples"] = count
    reject(write_dataset(tmp_path, meta), "n_samples=.*does not match samples length 2")


@pytest.mark.parametrize("field", ["feature_names", "target_names", "n_samples", "samples"])
def test_required_top_level_fields(tmp_path, field):
    meta = metadata(); del meta[field]
    reject(write_dataset(tmp_path, meta), f"missing required field '{field}'")


@pytest.mark.parametrize("value", [None, [], "text", 2, False])
def test_root_requires_object(tmp_path, value):
    path = write_dataset(tmp_path)
    write_metadata(path, value)
    reject(path, "metadata.json: expected an object")


@pytest.mark.parametrize("value", [None, {}, "rows", 2, False])
def test_samples_requires_array(tmp_path, value):
    meta = metadata(); meta["samples"] = value
    reject(write_dataset(tmp_path, meta), "metadata.json: samples must be an array")


@pytest.mark.parametrize("value", [None, [], "row", 2, False])
def test_sample_records_require_objects(tmp_path, value):
    meta = metadata(); meta["samples"][1] = value
    reject(write_dataset(tmp_path, meta), r"metadata.json: samples\[1\] must be an object")


@pytest.mark.parametrize("field", ["id", "unit", "grid"])
def test_sample_required_fields(tmp_path, field):
    meta = metadata(); del meta["samples"][1][field]
    reject(write_dataset(tmp_path, meta), rf"samples\[1\].*missing required field '{field}'")


@pytest.mark.parametrize("value", [None, "11", {"x": 1, "y": 2}, 2, False])
def test_grid_container_is_not_silently_reinterpreted(tmp_path, value):
    meta = metadata(); meta["samples"][1]["grid"] = value
    reject(write_dataset(tmp_path, meta), r"samples\[1\].grid must be an array")


@pytest.mark.parametrize("axis", ["features", "labels"])
@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (3, 2), (0, 2), (2, 0),
                                   (2, 1), (2, 3), (2, 2, 1), (2, 2, 2)])
def test_nonempty_array_requires_exact_rank_rows_and_columns(tmp_path, monkeypatch, axis, shape):
    path = write_dataset(tmp_path)
    np.save(path / (axis + ".npy"), np.zeros(shape), allow_pickle=False)

    def no_sample(*args, **kwargs):
        raise AssertionError("Array validation must precede sample construction")

    monkeypatch.setattr(module, "DatasetSample", no_sample)
    reject(path, rf"{axis}.npy: expected shape \(2, 2\), got")


@pytest.mark.parametrize("axis", ["features", "labels"])
@pytest.mark.parametrize("shape", [(), (1,), (0, 0), (0, 1), (0, 3), (1, 2), (0, 2, 1)])
def test_declared_empty_requires_exact_shape_or_legacy_vector(tmp_path, axis, shape):
    path = write_dataset(tmp_path, metadata(0), X=np.zeros((0, 2)), Y=np.zeros((0, 2)))
    np.save(path / (axis + ".npy"), np.zeros(shape), allow_pickle=False)
    reject(path, rf"{axis}.npy: expected shape \(0, 2\), got")


@pytest.mark.parametrize("axis", ["features", "labels"])
@pytest.mark.parametrize("shape", [(0,), (1, 0), (3, 0), (2, 1), (2, 0, 1)])
def test_zero_columns_still_requires_exact_row_count(tmp_path, axis, shape):
    path = write_dataset(tmp_path, metadata(features=[], targets=[]),
                         X=np.zeros((2, 0)), Y=np.zeros((2, 0)))
    np.save(path / (axis + ".npy"), np.zeros(shape), allow_pickle=False)
    reject(path, rf"{axis}.npy: expected shape \(2, 0\), got")


@pytest.mark.parametrize("axis", ["features", "labels"])
@pytest.mark.parametrize("dtype", ["U20", "S20", "complex128", "datetime64[D]",
                                   "timedelta64[D]", [("value", "f8")]])
@pytest.mark.parametrize("shape", [(2, 2), (0, 2), (0,), (2, 0)])
def test_nonnumeric_dtypes_rejected_even_with_no_values(tmp_path, axis, dtype, shape):
    n = shape[0]
    names = [] if shape == (2, 0) else ["a", "b"]
    meta = metadata(n, features=names, targets=names)
    path = write_dataset(tmp_path, meta, X=np.zeros((n, len(names))), Y=np.zeros((n, len(names))))
    values = np.ones(shape, dtype=dtype) if dtype in ("U20", "S20", "complex128") else np.zeros(shape, dtype=dtype)
    np.save(path / (axis + ".npy"), values, allow_pickle=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reject(path, rf"{axis}.npy: expected a real numeric or boolean dtype")


@pytest.mark.parametrize("axis", ["features", "labels"])
def test_nonzero_complex_imaginary_values_are_never_discarded(tmp_path, axis):
    path = write_dataset(tmp_path)
    np.save(path / (axis + ".npy"), BASE_X + 7j, allow_pickle=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reject(path, "expected a real numeric or boolean dtype, got complex128")


@pytest.mark.parametrize("axis", ["features", "labels"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_values_rejected_at_load_with_row_and_column(tmp_path, axis, value):
    path = write_dataset(tmp_path)
    values = BASE_X.copy(); values[1, 1] = value
    np.save(path / (axis + ".npy"), values, allow_pickle=False)
    kind, name = ("feature", "z") if axis == "features" else ("target", "w")
    reject(path, rf"{axis}.npy: sample 'sample-1' \(row 1\), {kind} '{name}':.*finite after float conversion")


@pytest.mark.parametrize("axis", ["features", "labels"])
def test_finite_extended_value_must_remain_finite_after_float_conversion(tmp_path, axis):
    if np.finfo(np.longdouble).max <= np.finfo(float).max:
        pytest.skip("longdouble has no wider exponent range than Python float")
    path = write_dataset(tmp_path)
    values = BASE_X.astype(np.longdouble); values[1, 1] = np.finfo(np.longdouble).max
    assert np.isfinite(values).all()
    np.save(path / (axis + ".npy"), values, allow_pickle=False)
    reject(path, rf"{axis}.npy: sample 'sample-1'.*finite after float conversion")


@pytest.mark.parametrize("axis", ["features", "labels"])
@pytest.mark.parametrize("empty", [False, True])
def test_npz_disguised_as_npy_is_rejected_and_closed(tmp_path, monkeypatch, axis, empty):
    n = 0 if empty else 2
    path = write_dataset(tmp_path, metadata(n), X=np.zeros((n, 2)), Y=np.zeros((n, 2)))
    with (path / (axis + ".npy")).open("wb") as handle:
        np.savez(handle, values=np.zeros((n, 2)))
    loaded_archives = []
    real_load = np.load

    def capture(*args, **kwargs):
        assert kwargs.get("allow_pickle") is False
        result = real_load(*args, **kwargs)
        if isinstance(result, np.lib.npyio.NpzFile):
            loaded_archives.append(result)
        return result

    monkeypatch.setattr(np, "load", capture)
    reject(path, rf"{axis}.npy: expected a single NPY array, got an NPZ archive")
    assert len(loaded_archives) == 1
    assert loaded_archives[0].zip is None
    assert loaded_archives[0].fid is None


@pytest.mark.parametrize("axis", ["features", "labels"])
@pytest.mark.parametrize("shape", [(2, 2), (0, 2)])
def test_object_header_rejected_without_pickle_payload(tmp_path, monkeypatch, axis, shape):
    path = write_dataset(tmp_path, metadata(shape[0]), X=np.zeros(shape), Y=np.zeros(shape))
    with (path / (axis + ".npy")).open("wb") as handle:
        np.lib.format.write_array_header_1_0(handle,
            {"descr": "|O", "fortran_order": False, "shape": shape})
    real_load = np.load

    def no_pickle(*args, **kwargs):
        assert kwargs.get("allow_pickle") is False
        return real_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", no_pickle)
    reject(path, "Object arrays cannot be loaded when allow_pickle=False")


@pytest.mark.parametrize("axis", ["features", "labels"])
@pytest.mark.parametrize("dtype", ["bool", "int8", "int64", "uint64", "float16", "float32", "float64", ">f8"])
@pytest.mark.parametrize("empty", [False, True])
def test_real_numeric_and_boolean_types_preserve_conversion(tmp_path, axis, dtype, empty):
    n = 0 if empty else 2
    values = BASE_X[:n].astype(dtype)
    X, Y = np.zeros((n, 2)), np.zeros((n, 2))
    if axis == "features": X = values
    else: Y = values
    path = write_dataset(tmp_path, metadata(n), X=X, Y=Y)
    before = snapshot(path)
    ds = FiberNetDataset.load(path)
    np.testing.assert_array_equal(ds.features if axis == "features" else ds.all_labels,
                                  values.astype(float))
    assert snapshot(path) == before


def test_fortran_layout_ordered_names_and_same_name_across_axes(tmp_path):
    meta = metadata(features=["z", " x "], targets=["z", " x "])
    path = write_dataset(tmp_path, meta, X=np.asfortranarray(BASE_X), Y=np.asfortranarray(BASE_Y))
    ds = FiberNetDataset.load(path)
    assert ds.feature_names == ds.target_names == ["z", " x "]
    np.testing.assert_array_equal(ds.features, BASE_X)
    np.testing.assert_array_equal(ds.all_labels, BASE_Y)


@pytest.mark.parametrize("feature_names,target_names", [([], []), (["a", "b"], ["y"]), ([], ["y"])])
@pytest.mark.parametrize("legacy_axes", [(False, False), (True, False), (False, True), (True, True)])
def test_modern_and_actual_legacy_empty_shapes_are_accepted_independently(tmp_path, feature_names,
                                                                        target_names, legacy_axes):
    meta = metadata(0, feature_names, target_names)
    path = write_dataset(tmp_path, meta,
                         X=np.zeros((0,) if legacy_axes[0] else (0, len(feature_names))),
                         Y=np.zeros((0,) if legacy_axes[1] else (0, len(target_names))))
    before = snapshot(path)
    ds = FiberNetDataset.load(path)
    assert ds.features.shape == (0, len(feature_names))
    assert ds.all_labels.shape == (0, len(target_names))
    if target_names: assert ds.labels(target_names[0]).shape == (0,)
    assert snapshot(path) == before


def test_duplicate_ids_opaque_fields_and_optional_metadata_are_unchanged(tmp_path):
    meta = metadata()
    meta["samples"][1]["id"] = meta["samples"][0]["id"]
    meta["samples"][0].update(unit=None, grid=["opaque", None, -1], params=[None, "opaque"], metadata="note")
    del meta["samples"][1]["params"]
    del meta["samples"][1]["metadata"]
    meta["ignored_extra_field"] = {"nested": None}
    meta["samples"][0]["ignored_row_field"] = float("nan")
    path = write_dataset(tmp_path, meta)
    before = snapshot(path)
    ds = FiberNetDataset.load(path)
    assert [s.sample_id for s in ds.samples] == ["sample-0", "sample-0"]
    assert ds.samples[0].unit is None
    assert ds.samples[0].grid == ("opaque", None, -1)
    assert ds.samples[0].params == [None, "opaque"]
    assert ds.samples[0].metadata == "note"
    assert ds.samples[1].params == ds.samples[1].metadata == {}
    np.testing.assert_array_equal(ds.features, BASE_X)
    assert snapshot(path) == before


def test_unselected_unavailable_values_remain_permissive(tmp_path):
    meta = metadata()
    meta["samples"][0]["metadata"]["infinity"] = float("inf")
    meta["samples"][0]["params"] = None
    ds = FiberNetDataset.load(write_dataset(tmp_path, meta))
    assert ds.samples[0].params is None
    assert ds.samples[0].metadata["availability"]["value"] is None
    assert np.isnan(ds.samples[0].metadata["legacy_nan"])
    assert np.isinf(ds.samples[0].metadata["infinity"])
    np.testing.assert_array_equal(ds.all_labels, BASE_Y)


def test_same_shaped_mix_and_row_permutation_are_not_claimed_detectable(tmp_path):
    meta = metadata(); meta["samples"].reverse()
    ds = FiberNetDataset.load(write_dataset(tmp_path, meta, Y=BASE_Y + 100000.))
    assert ds.samples[0].sample_id == "sample-1"
    np.testing.assert_array_equal(ds.features, BASE_X)
    np.testing.assert_array_equal(ds.all_labels, BASE_Y + 100000.)


@pytest.mark.parametrize("filename", FILES)
def test_missing_files_preserve_filesystem_error_and_bytes(tmp_path, filename):
    path = write_dataset(tmp_path); (path / filename).unlink()
    before = snapshot(path)
    with pytest.raises(FileNotFoundError):
        FiberNetDataset.load(path)
    assert snapshot(path) == before


def test_invalid_json_preserves_parser_error(tmp_path):
    path = write_dataset(tmp_path); (path / "metadata.json").write_text("{bad")
    before = snapshot(path)
    with pytest.raises(json.JSONDecodeError):
        FiberNetDataset.load(path)
    assert snapshot(path) == before


@pytest.mark.parametrize("axis", ["features", "labels"])
def test_truncated_npy_is_still_rejected(tmp_path, axis):
    path = write_dataset(tmp_path); file = path / (axis + ".npy")
    file.write_bytes(file.read_bytes()[:-8])
    reject(path, "")


def test_load_does_not_allocate_dense_validation_matrices(tmp_path, monkeypatch):
    path = write_dataset(tmp_path)

    def no_matrix(*args, **kwargs):
        raise AssertionError("Loading must validate during reconstruction, without dense copies")

    monkeypatch.setattr(FiberNetDataset, "_selected_matrix", no_matrix)
    ds = FiberNetDataset.load(path)
    assert ds.samples[0].features == {"x": 1., "z": 11.}


def test_valid_scalar_does_not_format_unused_error_context():
    class NoRepresentation:
        def __repr__(self):
            raise AssertionError("Valid scalar must not format the sample ID")

    assert module._load_dataset_value(np.float64(1.5), "features.npy",
                                      NoRepresentation(), 0, "feature", "x") == 1.5


def test_load_preserves_subclass_constructor(tmp_path):
    class ChildDataset(FiberNetDataset):
        pass

    assert isinstance(ChildDataset.load(write_dataset(tmp_path)), ChildDataset)


@pytest.mark.parametrize("case", ["ordinary", "empty", "empty_declared", "empty_selection",
                                  "historic_imputed_zero", "duplicate_id"])
def test_actual_old_writer_captured_bytes_load_unchanged(tmp_path, case):
    encoded = LEGACY_FILES[case]
    for filename, contents in encoded.items():
        (tmp_path / filename).write_bytes(base64.b64decode(contents))
    before = snapshot(tmp_path)
    meta = json.loads((tmp_path / "metadata.json").read_text())
    ds = FiberNetDataset.load(tmp_path)
    assert ds.n_samples == meta["n_samples"]
    assert ds.feature_names == meta["feature_names"]
    assert ds.target_names == meta["target_names"]
    assert [s.sample_id for s in ds.samples] == [s["id"] for s in meta["samples"]]
    for filename, names, actual in [("features.npy", ds.feature_names, ds.features),
                                    ("labels.npy", ds.target_names, ds.all_labels)]:
        saved = np.load(tmp_path / filename, allow_pickle=False)
        expected = np.empty((0, len(names))) if saved.shape == (0,) else saved
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == (ds.n_samples, len(names))
    if case == "historic_imputed_zero": assert ds.samples[0].labels["y"] == 0.0
    if case == "duplicate_id": assert ds.samples[0].sample_id == ds.samples[1].sample_id
    if ds.samples:
        assert ds.samples[0].metadata["availability"]["value"] is None
        assert np.isnan(ds.samples[0].metadata["legacy_nan"])
    assert snapshot(tmp_path) == before


# Static original-writer bytes. Base64 is only textual storage for numeric NPY
# and JSON fixtures; decoding is not object deserialization or pickle loading.
LEGACY_FILES = {'ordinary': {'features.npy': 'k05VTVBZAQB2AHsnZGVzY3InOiAnPGY4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsIDIpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAADwPwAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAQA==',
              'labels.npy': 'k05VTVBZAQB2AHsnZGVzY3InOiAnPGY4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsIDIpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAIQAAAAAAAAAAAAAAAAAAAEEAAAAAAAAAQQA==',
              'metadata.json': 'ewogICJmZWF0dXJlX25hbWVzIjogWwogICAgInoiLAogICAgIngiCiAgXSwKICAidGFyZ2V0X25hbWVzIjogWwogICAgInciLAogICAgInkiCiAgXSwKICAibl9zYW1wbGVzIjogMiwKICAic2FtcGxlcyI6IFsKICAgIHsKICAgICAgImlkIjogIm9sZC0wIiwKICAgICAgInVuaXQiOiAiYXVkaXQiLAogICAgICAiZ3JpZCI6IFsKICAgICAgICAxLAogICAgICAgIDEKICAgICAgXSwKICAgICAgInBhcmFtcyI6IHsKICAgICAgICAiYXJiaXRyYXJ5IjogbnVsbAogICAgICB9LAogICAgICAibWV0YWRhdGEiOiB7CiAgICAgICAgImF2YWlsYWJpbGl0eSI6IHsKICAgICAgICAgICJ2YWx1ZSI6IG51bGwsCiAgICAgICAgICAic3RhdHVzIjogInVuYXZhaWxhYmxlIgogICAgICAgIH0sCiAgICAgICAgImxlZ2FjeV9uYW4iOiBOYU4KICAgICAgfQogICAgfSwKICAgIHsKICAgICAgImlkIjogIm9sZC0xIiwKICAgICAgInVuaXQiOiAiYXVkaXQiLAogICAgICAiZ3JpZCI6IFsKICAgICAgICAxLAogICAgICAgIDEKICAgICAgXSwKICAgICAgInBhcmFtcyI6IHsKICAgICAgICAiYXJiaXRyYXJ5IjogbnVsbAogICAgICB9LAogICAgICAibWV0YWRhdGEiOiB7CiAgICAgICAgImF2YWlsYWJpbGl0eSI6IHsKICAgICAgICAgICJ2YWx1ZSI6IG51bGwsCiAgICAgICAgICAic3RhdHVzIjogInVuYXZhaWxhYmxlIgogICAgICAgIH0sCiAgICAgICAgImxlZ2FjeV9uYW4iOiBOYU4KICAgICAgfQogICAgfQogIF0KfQ=='},
 'empty': {'features.npy': 'k05VTVBZAQB2AHsnZGVzY3InOiAnPGY4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDAsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAo=',
           'labels.npy': 'k05VTVBZAQB2AHsnZGVzY3InOiAnPGY4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDAsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAo=',
           'metadata.json': 'ewogICJmZWF0dXJlX25hbWVzIjogW10sCiAgInRhcmdldF9uYW1lcyI6IFtdLAogICJuX3NhbXBsZXMiOiAwLAogICJzYW1wbGVzIjogW10KfQ=='},
 'empty_declared': {'features.npy': 'k05VTVBZAQB2AHsnZGVzY3InOiAnPGY4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDAsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAo=',
                    'labels.npy': 'k05VTVBZAQB2AHsnZGVzY3InOiAnPGY4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDAsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAo=',
                    'metadata.json': 'ewogICJmZWF0dXJlX25hbWVzIjogWwogICAgInoiLAogICAgIngiCiAgXSwKICAidGFyZ2V0X25hbWVzIjogWwogICAgInciLAogICAgInkiCiAgXSwKICAibl9zYW1wbGVzIjogMCwKICAic2FtcGxlcyI6IFtdCn0='},
 'empty_selection': {'features.npy': 'k05VTVBZAQB2AHsnZGVzY3InOiAnPGY4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsIDApLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAo=',
                     'labels.npy': 'k05VTVBZAQB2AHsnZGVzY3InOiAnPGY4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsIDApLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAo=',
                     'metadata.json': 'ewogICJmZWF0dXJlX25hbWVzIjogW10sCiAgInRhcmdldF9uYW1lcyI6IFtdLAogICJuX3NhbXBsZXMiOiAyLAogICJzYW1wbGVzIjogWwogICAgewogICAgICAiaWQiOiAib2xkLTAiLAogICAgICAidW5pdCI6ICJhdWRpdCIsCiAgICAgICJncmlkIjogWwogICAgICAgIDEsCiAgICAgICAgMQogICAgICBdLAogICAgICAicGFyYW1zIjogewogICAgICAgICJhcmJpdHJhcnkiOiBudWxsCiAgICAgIH0sCiAgICAgICJtZXRhZGF0YSI6IHsKICAgICAgICAiYXZhaWxhYmlsaXR5IjogewogICAgICAgICAgInZhbHVlIjogbnVsbCwKICAgICAgICAgICJzdGF0dXMiOiAidW5hdmFpbGFibGUiCiAgICAgICAgfSwKICAgICAgICAibGVnYWN5X25hbiI6IE5hTgogICAgICB9CiAgICB9LAogICAgewogICAgICAiaWQiOiAib2xkLTEiLAogICAgICAidW5pdCI6ICJhdWRpdCIsCiAgICAgICJncmlkIjogWwogICAgICAgIDEsCiAgICAgICAgMQogICAgICBdLAogICAgICAicGFyYW1zIjogewogICAgICAgICJhcmJpdHJhcnkiOiBudWxsCiAgICAgIH0sCiAgICAgICJtZXRhZGF0YSI6IHsKICAgICAgICAiYXZhaWxhYmlsaXR5IjogewogICAgICAgICAgInZhbHVlIjogbnVsbCwKICAgICAgICAgICJzdGF0dXMiOiAidW5hdmFpbGFibGUiCiAgICAgICAgfSwKICAgICAgICAibGVnYWN5X25hbiI6IE5hTgogICAgICB9CiAgICB9CiAgXQp9'},
 'historic_imputed_zero': {'features.npy': 'k05VTVBZAQB2AHsnZGVzY3InOiAnPGY4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsIDIpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAADwPwAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAQA==',
                           'labels.npy': 'k05VTVBZAQB2AHsnZGVzY3InOiAnPGY4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsIDIpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAIQAAAAAAAAAAAAAAAAAAAEEAAAAAAAAAQQA==',
                           'metadata.json': 'ewogICJmZWF0dXJlX25hbWVzIjogWwogICAgInoiLAogICAgIngiCiAgXSwKICAidGFyZ2V0X25hbWVzIjogWwogICAgInciLAogICAgInkiCiAgXSwKICAibl9zYW1wbGVzIjogMiwKICAic2FtcGxlcyI6IFsKICAgIHsKICAgICAgImlkIjogIm9sZC0wIiwKICAgICAgInVuaXQiOiAiYXVkaXQiLAogICAgICAiZ3JpZCI6IFsKICAgICAgICAxLAogICAgICAgIDEKICAgICAgXSwKICAgICAgInBhcmFtcyI6IHsKICAgICAgICAiYXJiaXRyYXJ5IjogbnVsbAogICAgICB9LAogICAgICAibWV0YWRhdGEiOiB7CiAgICAgICAgImF2YWlsYWJpbGl0eSI6IHsKICAgICAgICAgICJ2YWx1ZSI6IG51bGwsCiAgICAgICAgICAic3RhdHVzIjogInVuYXZhaWxhYmxlIgogICAgICAgIH0sCiAgICAgICAgImxlZ2FjeV9uYW4iOiBOYU4KICAgICAgfQogICAgfSwKICAgIHsKICAgICAgImlkIjogIm9sZC0xIiwKICAgICAgInVuaXQiOiAiYXVkaXQiLAogICAgICAiZ3JpZCI6IFsKICAgICAgICAxLAogICAgICAgIDEKICAgICAgXSwKICAgICAgInBhcmFtcyI6IHsKICAgICAgICAiYXJiaXRyYXJ5IjogbnVsbAogICAgICB9LAogICAgICAibWV0YWRhdGEiOiB7CiAgICAgICAgImF2YWlsYWJpbGl0eSI6IHsKICAgICAgICAgICJ2YWx1ZSI6IG51bGwsCiAgICAgICAgICAic3RhdHVzIjogInVuYXZhaWxhYmxlIgogICAgICAgIH0sCiAgICAgICAgImxlZ2FjeV9uYW4iOiBOYU4KICAgICAgfQogICAgfQogIF0KfQ=='},
 'duplicate_id': {'features.npy': 'k05VTVBZAQB2AHsnZGVzY3InOiAnPGY4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsIDIpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAADwPwAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAQA==',
                  'labels.npy': 'k05VTVBZAQB2AHsnZGVzY3InOiAnPGY4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsIDIpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAIQAAAAAAAAAAAAAAAAAAAEEAAAAAAAAAQQA==',
                  'metadata.json': 'ewogICJmZWF0dXJlX25hbWVzIjogWwogICAgInoiLAogICAgIngiCiAgXSwKICAidGFyZ2V0X25hbWVzIjogWwogICAgInciLAogICAgInkiCiAgXSwKICAibl9zYW1wbGVzIjogMiwKICAic2FtcGxlcyI6IFsKICAgIHsKICAgICAgImlkIjogIm9sZC0wIiwKICAgICAgInVuaXQiOiAiYXVkaXQiLAogICAgICAiZ3JpZCI6IFsKICAgICAgICAxLAogICAgICAgIDEKICAgICAgXSwKICAgICAgInBhcmFtcyI6IHsKICAgICAgICAiYXJiaXRyYXJ5IjogbnVsbAogICAgICB9LAogICAgICAibWV0YWRhdGEiOiB7CiAgICAgICAgImF2YWlsYWJpbGl0eSI6IHsKICAgICAgICAgICJ2YWx1ZSI6IG51bGwsCiAgICAgICAgICAic3RhdHVzIjogInVuYXZhaWxhYmxlIgogICAgICAgIH0sCiAgICAgICAgImxlZ2FjeV9uYW4iOiBOYU4KICAgICAgfQogICAgfSwKICAgIHsKICAgICAgImlkIjogIm9sZC0wIiwKICAgICAgInVuaXQiOiAiYXVkaXQiLAogICAgICAiZ3JpZCI6IFsKICAgICAgICAxLAogICAgICAgIDEKICAgICAgXSwKICAgICAgInBhcmFtcyI6IHsKICAgICAgICAiYXJiaXRyYXJ5IjogbnVsbAogICAgICB9LAogICAgICAibWV0YWRhdGEiOiB7CiAgICAgICAgImF2YWlsYWJpbGl0eSI6IHsKICAgICAgICAgICJ2YWx1ZSI6IG51bGwsCiAgICAgICAgICAic3RhdHVzIjogInVuYXZhaWxhYmxlIgogICAgICAgIH0sCiAgICAgICAgImxlZ2FjeV9uYW4iOiBOYU4KICAgICAgfQogICAgfQogIF0KfQ=='}}
