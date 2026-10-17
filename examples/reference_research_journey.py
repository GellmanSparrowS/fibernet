"""Bounded existing-API research journey; this is not a schema framework.

Run from the repository: python -m examples.reference_research_journey \
    --output-dir /a/new/private/directory
Only load the pickle created by this invocation; never load untrusted pickles.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

import fibernet as fn
from fibernet.analysis import SnapshotFeatureExtractor
from fibernet.io.meshio_adapter import write_vtu_bundle
from fibernet.ml.dataset import DatasetSample, FiberNetDataset


# Explicit caller-maintained scientific choices, not a library schema.
FEATURE_NAMES = ["total_length_m", "mean_cross_section_area_m2", "pore_area_mean_m2"]
TARGET_NAME = "max_axial_force_N"


def run(output_dir):
    """Execute ten small samples, refuse existing output, and return evidence."""
    import meshio

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    samples, export_checks, export_inputs = [], [], []
    for index in range(10):
        sample_id = f"square-{index:02d}"
        folder = out / sample_id
        folder.mkdir()
        side = .01 + .001 * index
        radius = .0001 + .00001 * ((index * 3) % 10)
        graph = fn.pattern_2d(unit="square", box=(side, side), grid=(1, 1),
                              radius=radius, n_pts_per_side=0, seed=23,
                              material=fn.Material(name="reference_linear", youngs_modulus=1e6,
                                                   poissons_ratio=.3))
        if index % 2:
            # Native dict editing is an explicit fixture preparation, not a new API.
            data = graph.to_dict()
            remap = {nid: 7 + 11 * nid for nid in graph.nodes}
            for node in data["nodes"]:
                node["id"] = remap[node["id"]]
            for edge in data["edges"]:
                edge["node_i"] = remap[edge["node_i"]]
                edge["node_j"] = remap[edge["node_j"]]
                edge["id"] = 17 + 13 * edge["id"]
            data["nodes"].reverse()
            data["edges"].reverse()
            data.pop("next_node_id")
            data.pop("next_edge_id")
            graph = fn.StructureGraph.from_dict(data)
        graph.save_json(folder / "graph.json")
        reload_path = folder / "graph.json"
        if index % 2:
            # Native save canonicalizes records. Explicitly reorder a second
            # native JSON fixture to exercise differing solver row orders.
            saved_graph = json.loads(reload_path.read_text())
            saved_graph["nodes"].reverse()
            saved_graph["edges"].reverse()
            reload_path = folder / "graph-reordered.json"
            reload_path.write_text(json.dumps(saved_graph, indent=2) + "\n")
        graph = fn.StructureGraph.load_json(reload_path)
        assert all(edge.material.youngs_modulus == 1e6 and edge.material.poissons_ratio == .3
                   for edge in graph.edges.values())

        # E is supplied globally, NOT inferred from graph Material. Geometry is m.
        # strain=1.01 is stretch ratio, i.e. engineering strain=.01.
        # pct=.5 selects both nodes of each side on these four-node squares.
        mechanical = fn.simulate(graph, backend="fem", mode="stretch",
                                 strain=1.01, nonlinear=False, dim=2, pct=.5,
                                 E=1e6, nu=.3)
        mechanical.save(folder / "mechanical.json", detailed=True)
        mechanical = fn.SimResult.load(folder / "mechanical.json")
        np.testing.assert_allclose(mechanical.max_force, 1e6 * np.pi * radius**2 * .01,
                                   rtol=1e-9, atol=1e-15)

        # Conductance is supplied in S, independent of length/radius/material.
        conductance = {eid: .2 + .01 * index for eid in graph.edges}
        boundary = [(nid, 1. if np.isclose(node.position[0], 0) else 0.)
                    for nid, node in graph.nodes.items()]
        electrical = fn.solve_linear_transport(graph, conductance, boundary,
                                                length_unit="m")
        (folder / "electrical.json").write_text(
            json.dumps(electrical.to_dict(), indent=2, allow_nan=False) + "\n")
        electrical_saved = json.loads((folder / "electrical.json").read_text())
        assert electrical_saved["provenance"]["geometry_conversion"] == "none"

        snapshot = SnapshotFeatureExtractor(include_contact=False).extract_graph(graph)
        snapshot_json = {key: value.tolist() if isinstance(value, np.ndarray) else value
                         for key, value in snapshot.items()}
        (folder / "snapshot.json").write_text(
            json.dumps(snapshot_json, indent=2, allow_nan=False) + "\n")
        # The extractor uses caller length units. Rename them explicitly for ML.
        features = {
            "total_length_m": float(snapshot["total_length"]),
            "mean_cross_section_area_m2": float(np.mean(
                [np.pi * edge.radius**2 for edge in graph.edges.values()])),
            "pore_area_mean_m2": float(snapshot["pore_area_mean"]),
        }
        # Maximum member axial force, not the summed grip reaction.
        labels = {TARGET_NAME: float(mechanical.max_force)}
        # Keep fixture intent explicit; Dataset consumption now also rejects
        # missing, nonfinite or invalid selected feature/label values.
        assert set(features) == set(FEATURE_NAMES)
        assert all(np.isfinite(value) for value in [*features.values(), *labels.values()])
        metadata = {
            "length_unit": "m", "target_unit": "N", "E_Pa": 1e6, "nu": .3,
            "stretch_ratio": 1.01, "engineering_strain": .01, "boundary_percentile": .5,
            "target_meaning": "maximum absolute member axial force, not summed grip reaction",
            "material_source": "explicit global E/nu; graph Material is not consumed",
            "feature_source": "snapshot geometry plus mean graph cross-section area",
            "contact": {"status": "not_requested", "value": None},
            "fracture_strength": {"status": "unavailable_in_linear_FEM", "value": None},
            "mechanical_result": f"{sample_id}/mechanical.json",
            "electrical_result": f"{sample_id}/electrical.json",
            "electrical_role": "separate supplied-conductance demonstration; not ML target",
        }
        samples.append(DatasetSample(sample_id, "square", (1, 1),
                                     {"side_m": side, "radius_m": radius},
                                     features, labels, metadata))

        export_inputs.append((sample_id, folder, graph, mechanical, electrical,
                              electrical_saved, metadata))

    dataset = FiberNetDataset(samples, feature_names=FEATURE_NAMES, target_names=[TARGET_NAME])
    dataset.save(str(out / "dataset"))
    loaded = FiberNetDataset.load(str(out / "dataset"))
    np.testing.assert_array_equal(loaded.features, dataset.features)
    np.testing.assert_array_equal(loaded.all_labels, dataset.all_labels)
    assert loaded.samples[0].metadata["contact"]["value"] is None
    loaded.to_csv(str(out / "dataset.csv"))
    # Keep the CSV export for interoperability, but train directly on raw names.
    # Explicit row splitting is only an API fixture, not unseen-origin validation.
    metadata_keys = ["length_unit", "target_unit", "target_meaning"]
    metrics = fn.train_model(loaded, target=TARGET_NAME,
                            model_type="lr", test_size=.3, split="row",
                            metadata_keys=metadata_keys,
                            save_path=str(out / "model.pkl"))
    # Trusted local artifact produced above, not an external pickle.
    with (out / "model.pkl").open("rb") as handle:
        artifact = pickle.load(handle)
    # Stored feature names, not this script's constant, drive inference selection.
    # Reverse mapping order and retain extras to exercise named selection.
    records = [{"sample_id": sample.sample_id, TARGET_NAME: sample.labels[TARGET_NAME],
                **{name: sample.features[name] for name in reversed(loaded.feature_names)}}
               for sample in loaded.samples]
    inference_matrix = np.array([[row[name] for name in artifact["features"]] for row in records])
    named_result = fn.predict_named(artifact, records, extra_columns="ignore")
    predictions = np.asarray(named_result["predictions"])
    assert named_result["target"] == TARGET_NAME
    assert named_result["validation"] == "named_columns_only"
    np.testing.assert_allclose(predictions, artifact["model"].predict(inference_matrix))
    # Deliberately expose the legacy bare-estimator boundary: wrong semantic order.
    wrong_order_predictions = artifact["model"].predict(inference_matrix[:, [1, 2, 0]])
    assert not np.allclose(wrong_order_predictions, predictions, rtol=1e-6, atol=1e-12)
    prediction_rows = [{"sample_id": sample.sample_id, "observed_max_axial_force_N":
                        sample.labels[TARGET_NAME], "predicted_max_axial_force_N": float(prediction)}
                       for sample, prediction in zip(loaded.samples, predictions)]
    (out / "predictions.json").write_text(json.dumps(prediction_rows, indent=2) + "\n")
    # Final consumer: export the retained simulation results after ML inference.
    for (sample_id, folder, graph, mechanical, electrical,
         electrical_saved, metadata) in export_inputs:
        # Result row order differs between FEM (insertion) and transport (sorted).
        # Pass actual identities, never guessed or positional alignment.
        displacement = dict(zip(mechanical.metadata["node_ids"], mechanical.displacements))
        axial_force = dict(zip(mechanical.metadata["edge_ids"], mechanical.edge_forces))
        voltage = dict(zip(map(int, electrical.node_ids), electrical.voltage_V))
        current = dict(zip(map(int, electrical.edge_ids), electrical.current_A))
        bundle = write_vtu_bundle(folder / "vtu", graph, length_unit="m",
            point_data_by_id={"displacement_m": displacement, "potential_V": voltage},
            cell_data_by_id={"axial_force_N": axial_force, "current_A": current},
            field_metadata={
                "point": {"displacement_m": {"units": "m", "meaning": "mechanical displacement"},
                          "potential_V": {"units": "V", "meaning": "electric potential"}},
                "cell": {"axial_force_N": {"units": "N", "meaning": "signed axial force"},
                         "current_A": {"units": "A", "meaning": "signed node_i to node_j current"}}},
            provenance={"sample_id": sample_id, "mechanics": metadata,
                        "electrical_schema": electrical_saved["schema"]})
        mesh = meshio.read(bundle / "network.vtu")
        manifest = json.loads((bundle / "manifest.json").read_text())
        for material_index in mesh.cell_data["material_index"][0]:
            declared_material = manifest["materials"][int(material_index)]
            assert declared_material["youngs_modulus"] == 1e6
            assert declared_material["poissons_ratio"] == .3
        node_ids = mesh.point_data["node_id"]
        edge_ids = mesh.cell_data["edge_id"][0]
        np.testing.assert_array_equal(node_ids, sorted(graph.nodes))
        np.testing.assert_array_equal(edge_ids, list(graph.edges))
        for name, values in (("displacement_m", displacement), ("potential_V", voltage)):
            np.testing.assert_allclose(mesh.point_data[name], [values[int(n)] for n in node_ids])
        for name, values in (("axial_force_N", axial_force), ("current_A", current)):
            np.testing.assert_allclose(mesh.cell_data[name][0], [values[int(e)] for e in edge_ids])
        rows = {int(nid): i for i, nid in enumerate(node_ids)}
        np.testing.assert_array_equal(mesh.cells_dict["line"],
            [[rows[e.node_i], rows[e.node_j]] for e in graph.edges.values()])
        export_checks.append({"sample_id": sample_id, "node_ids": node_ids.tolist(),
                              "edge_ids": edge_ids.tolist(), "reloaded_fields_match_ids": True})

    summary = {
        "n_samples": len(samples), "metrics": metrics,
        "training_input_kind": "FiberNetDataset",
        "split_policy": "explicit_row_holdout_not_origin_validation",
        "selected_metadata_keys": metadata_keys,
        "inference_feature_order_from_artifact": artifact["features"],
        "artifact_keys": sorted(artifact),
        "schema_version_guard": "absent; artifact envelope version is not a physical schema",
        "named_inference_validation": named_result["validation"],
        "artifact_target": named_result["target"],
        "equal_width_reordered_input_accepted": True,
        "equal_width_reordered_input_consumer": "bare_estimator",
        "max_prediction_change_from_wrong_order_N": float(np.max(np.abs(predictions - wrong_order_predictions))),
        "unavailable_metrics": "contact/fracture null with status retained in dataset metadata; not zero labels",
        "evidence_scope": "tiny synthetic API usability only; not science, calibration, or generalization",
        "prediction_scope": "all ten training-source rows for reload checks; held-out metrics are train_model's split only",
        "next_consumer_boundary": "named columns validated using saved order; units and scientific meaning remain caller responsibility",
        "vtu_checks": export_checks,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="new directory; existing output is never overwritten")
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))
