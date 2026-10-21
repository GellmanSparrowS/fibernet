"""Bounded real saved-artifact journey, including its current consumer gaps."""
import json
import pickle

import numpy as np
import pytest

pytest.importorskip("sklearn")
meshio = pytest.importorskip("meshio")

from examples.reference_research_journey import FEATURE_NAMES, TARGET_NAME, run
from fibernet.ml.dataset import FiberNetDataset
from fibernet import predict_named


@pytest.fixture(scope="module")
def journey(tmp_path_factory):
    folder = tmp_path_factory.mktemp("reference-parent") / "journey"
    summary = run(folder)
    return folder, summary


def test_saved_dataset_and_actual_training_artifact(journey):
    folder, summary = journey
    dataset = FiberNetDataset.load(str(folder / "dataset"))
    assert dataset.n_samples == summary["n_samples"] == 10
    assert dataset.feature_names == FEATURE_NAMES
    assert dataset.target_names == [TARGET_NAME]
    for sample in dataset.samples:
        assert sample.metadata["contact"] == {"status": "not_requested", "value": None}
        assert sample.metadata["fracture_strength"]["value"] is None
        assert sample.labels[TARGET_NAME] > 0
        assert "electrical" not in " ".join(dataset.target_names)
        np.testing.assert_allclose(sample.labels[TARGET_NAME],
                                   1e6 * sample.features["mean_cross_section_area_m2"] * .01,
                                   rtol=1e-9)
    # This is our own generated local pickle, never an untrusted fixture.
    with (folder / "model.pkl").open("rb") as handle:
        artifact = pickle.load(handle)
    assert sorted(artifact) == ["artifact_version", "features", "metrics", "model", "target"]
    assert artifact["features"] == FEATURE_NAMES
    assert artifact["model"].n_features_in_ == 3
    assert artifact["artifact_version"] == 1
    assert artifact["target"] == TARGET_NAME
    assert summary["training_input_kind"] == "FiberNetDataset"
    assert summary["split_policy"] == "explicit_row_holdout_not_origin_validation"
    provenance = artifact["metrics"]["training_data"]
    assert provenance["metadata_keys"] == ["length_unit", "target_unit", "target_meaning"]
    assert provenance["split"] == {"method": "row", "random_state": 42,
                                    "train_rows": [0, 7, 2, 9, 4, 3, 6], "test_rows": [8, 1, 5]}
    assert [row["sample_id"] for row in provenance["rows"]] == [s.sample_id for s in dataset.samples]
    assert [row["metadata"] for row in provenance["rows"]] == [
        {key: sample.metadata[key] for key in provenance["metadata_keys"]} for sample in dataset.samples]
    assert all(row["metadata"]["target_unit"] == "N" for row in provenance["rows"])
    assert summary["named_inference_validation"] == "named_columns_only"
    assert summary["equal_width_reordered_input_accepted"] is True
    assert summary["equal_width_reordered_input_consumer"] == "bare_estimator"
    assert summary["max_prediction_change_from_wrong_order_N"] > 1
    assert np.isfinite(summary["metrics"]["rmse"])


def test_named_inference_uses_saved_order_and_exposes_legacy_boundary(journey):
    folder, _ = journey
    with (folder / "model.pkl").open("rb") as handle:
        artifact = pickle.load(handle)
    dataset = FiberNetDataset.load(folder / "dataset")
    names, model = artifact["features"], artifact["model"]
    matrix = np.array([[sample.features[name] for name in names] for sample in dataset.samples])
    expected = model.predict(matrix)
    shuffled = [{"sample_id": sample.sample_id,
                 **{name: sample.features[name] for name in reversed(names)}}
                for sample in dataset.samples]
    np.testing.assert_array_equal(expected, predict_named(artifact, shuffled, extra_columns="ignore")["predictions"])
    with pytest.raises(ValueError, match="extra"):
        predict_named(artifact, shuffled)
    with pytest.raises(ValueError, match="Missing"):
        predict_named(artifact, [{name: value for name, value in row.items() if name != names[0]}
                                 for row in shuffled], extra_columns="ignore")
    # The existing bare estimator enforces width, not units/column semantics.
    wrong = model.predict(matrix[:, [1, 2, 0]])
    assert not np.allclose(wrong, expected)
    with pytest.raises(ValueError):
        model.predict(matrix[:, :2])


def test_csv_export_keeps_legacy_prefixed_names(journey):
    import csv
    folder, _ = journey
    with (folder / "dataset.csv").open() as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames == ["sample_id", "unit", "grid_x", "grid_y"] + [
        "feat_" + name for name in FEATURE_NAMES] + ["label_" + TARGET_NAME]
    assert len(rows) == 10
    dataset = FiberNetDataset.load(folder / "dataset")
    for row, sample in zip(rows, dataset.samples):
        assert row["sample_id"] == sample.sample_id
        for name in FEATURE_NAMES:
            assert float(row["feat_" + name]) == sample.features[name]
        assert float(row["label_" + TARGET_NAME]) == sample.labels[TARGET_NAME]


def test_reordered_native_result_ids_survive_vtu_reload(journey):
    folder, summary = journey
    assert all(item["reloaded_fields_match_ids"] for item in summary["vtu_checks"])
    sample = folder / "square-01"
    mech = json.loads((sample / "mechanical.json").read_text())
    electrical = json.loads((sample / "electrical.json").read_text())
    assert mech["node_ids"] == [40, 29, 18, 7]
    assert electrical["node_ids"] == [7, 18, 29, 40]
    assert mech["edge_ids"] == electrical["edge_ids"] == [56, 43, 30, 17]
    mesh = meshio.read(sample / "vtu/network.vtu")
    np.testing.assert_array_equal(mesh.point_data["node_id"], electrical["node_ids"])
    displacements = dict(zip(mech["node_ids"], mech["displacements"]))
    np.testing.assert_allclose(mesh.point_data["displacement_m"],
        [displacements[int(nid)] for nid in mesh.point_data["node_id"]])
    np.testing.assert_allclose(mesh.cell_data["current_A"][0], electrical["current_A"])
    assert electrical["provenance"]["geometry_conversion"] == "none"
    manifest = json.loads((sample / "vtu/manifest.json").read_text())
    assert manifest["length_unit"] == "m"
    native = json.loads((sample / "graph-reordered.json").read_text())
    assert all(edge["material"]["youngs_modulus"] == 1e6 for edge in native["edges"])
    for material_index in mesh.cell_data["material_index"][0]:
        assert manifest["materials"][int(material_index)]["youngs_modulus"] == 1e6
        assert manifest["materials"][int(material_index)]["poissons_ratio"] == .3
    assert manifest["field_metadata"]["cell"]["axial_force_N"]["units"] == "N"


def test_existing_output_is_not_overwritten(journey):
    folder, _ = journey
    before = (folder / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        run(folder)
    assert (folder / "summary.json").read_bytes() == before
