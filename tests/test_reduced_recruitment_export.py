"""Saved reduced recruitment can be reanalyzed using only native public APIs."""

from dataclasses import asdict
import json

import numpy as np
import pytest

from examples import reduced_recruitment_workflow as workflow
from fibernet.analysis import analyze_tensile_recruitment, compute_percolation
from fibernet.core.structure_graph import StructureGraph
from fibernet.gen import PlanarManufacturingConfig
from fibernet.sim import ReducedBeamConfig, ReducedBeamSolver


def _sparse_graph():
    # Parallel fibers and intentionally shuffled sparse edge IDs exercise the
    # native graph serializer's sorting versus the solver's insertion order.
    return StructureGraph.from_dict(
        {
            "dimension": 3,
            "nodes": [
                {"id": 42, "position": [2.0, 0.0, 7.0]},
                {"id": 3, "position": [0.0, 0.0, 7.0]},
                {"id": 11, "position": [1.0, 0.2, 7.0]},
            ],
            "edges": [
                {"id": 91, "node_i": 3, "node_j": 11},
                {"id": 5, "node_i": 11, "node_j": 42},
                {"id": 27, "node_i": 3, "node_j": 11},
            ],
            "metadata": {"weld_triplets": [[3, 11, 42]]},
            "next_node_id": 100,
            "next_edge_id": 200,
        }
    )


def _json(saved, key):
    assert saved[key].shape == ()
    assert saved[key].dtype.kind == "U"
    return json.loads(saved[key].item())


def test_export_replays_native_analysis_and_solver_without_hidden_defaults(
    tmp_path, monkeypatch
):
    graph = _sparse_graph()
    monkeypatch.setattr(workflow, "manufacturable_graph", lambda config: graph)
    summary = workflow.ReducedRecruitmentWorkflow(tmp_path, alpha=0.17).run()
    with np.load(tmp_path / "trajectory.npz", allow_pickle=False) as saved:
        # Every field must be readable without pickle, not just known arrays.
        assert all(saved[key].dtype.kind != "O" for key in saved.files)
        assert saved["trajectory_schema_version"].item() == 1
        np.testing.assert_array_equal(saved["node_ids"], [3, 11, 42])
        np.testing.assert_array_equal(saved["edge_ids"], [91, 5, 27])
        assert _json(saved, "graph_config_json") == asdict(
            PlanarManufacturingConfig(
                unit="kagome",
                grid_x=3,
                grid_y=3,
                n_pts_per_side=2,
                perturbation=0.25,
                seed=23,
            )
        )
        cfg_data = _json(saved, "solver_config_json")
        assert cfg_data == asdict(
            ReducedBeamConfig(
                target_stretch=1.4,
                n_increments=20,
                num_steps=2000,
                use_contact=False,
                use_bending=True,
            )
        )
        analysis_config = _json(saved, "recruitment_config_json")
        assert analysis_config == dict(
            alpha=0.17,
            hysteresis=0.6,
            grip_connected=True,
            quantile=None,
            min_active=1e-4,
        )
        analyzed = analyze_tensile_recruitment(
            saved["edge_strain"],
            saved["edges"],
            saved["left_nodes"],
            saved["right_nodes"],
            n_nodes=len(saved["node_ids"]),
            **analysis_config,
        )
        for key, attr in [
            ("active_edges", "active_edges"),
            ("spanning_edges", "edge_in_spanning"),
            ("spanning_frac", "spanning_frac"),
            ("backbone_frac", "backbone_frac"),
            ("edge_depth_norm", "edge_depth_norm"),
            ("threshold_curve", "threshold_curve"),
        ]:
            np.testing.assert_array_equal(saved[key], getattr(analyzed, attr))
        assert analyzed.perc_frame == saved["first_spanning_frame"].item()
        assert analyzed.perc_frame == summary["first_spanning_frame"]

        restored = StructureGraph.from_dict(_json(saved, "graph_json"))
        assert restored.to_dict() == graph.to_dict()
        assert list(restored.edges) == [91, 5, 27]
        replay = ReducedBeamSolver(restored, ReducedBeamConfig(**cfg_data)).run()
        for key, attr in [
            ("frames_xy", "frames_xy"),
            ("edges", "edges"),
            ("edge_rest", "edge_rest"),
            ("edge_strain", "edge_strain"),
            ("force_curve", "force_curve"),
            ("stretch_levels", "strain_levels"),
            ("left_nodes", "left_nodes"),
            ("right_nodes", "right_nodes"),
        ]:
            np.testing.assert_array_equal(saved[key], getattr(replay, attr))
        for name, values in replay.energies.items():
            np.testing.assert_array_equal(saved["energy_" + name], values)
        metadata = _json(saved, "result_metadata_json")
        for key in metadata.keys() - {"wall_seconds"}:
            assert metadata[key] == replay.metadata[key]
        expected = compute_percolation(replay, **analysis_config)
        np.testing.assert_array_equal(saved["active_edges"], expected.active_edges)
        np.testing.assert_array_equal(
            saved["spanning_edges"], expected.edge_in_spanning
        )


def test_default_export_keeps_legacy_arrays_and_real_graph_roundtrip(tmp_path):
    summary = workflow.ReducedRecruitmentWorkflow(tmp_path).run()
    assert (summary["nodes"], summary["edges"], summary["frames"]) == (436, 576, 21)
    with np.load(tmp_path / "trajectory.npz", allow_pickle=False) as saved:
        graph = StructureGraph.from_dict(_json(saved, "graph_json"))
        cfg = ReducedBeamConfig(**_json(saved, "solver_config_json"))
        original = ReducedBeamSolver(graph, cfg).run()
        recruited = compute_percolation(original, alpha=0.05)
        legacy = dict(
            frames_xy=original.frames_xy,
            edges=original.edges,
            edge_strain=original.edge_strain,
            force_curve=original.force_curve,
            stretch_levels=original.strain_levels,
            active_edges=recruited.active_edges,
            spanning_edges=recruited.edge_in_spanning,
        )
        for key, values in legacy.items():
            assert saved[key].dtype == values.dtype
            np.testing.assert_array_equal(saved[key], values)
        assert all(saved[key].dtype.kind != "O" for key in saved.files)
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    assert not list(tmp_path.glob("*.tmp.*"))


def test_failed_archive_replace_preserves_previous_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        workflow, "manufacturable_graph", lambda config: _sparse_graph()
    )
    target = tmp_path / "trajectory.npz"
    summary = tmp_path / "summary.json"
    target.write_bytes(b"previous trajectory")
    summary.write_text("previous summary")

    def fail_replace(source, destination):
        raise OSError("replacement failed")

    monkeypatch.setattr(workflow.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement failed"):
        workflow.ReducedRecruitmentWorkflow(tmp_path).run()
    assert target.read_bytes() == b"previous trajectory"
    assert summary.read_text() == "previous summary"
    assert not list(tmp_path.glob("*.tmp.*"))


@pytest.mark.parametrize("invalid", [float("nan"), object()])
def test_non_json_metadata_does_not_modify_outputs(tmp_path, monkeypatch, invalid):
    graph = _sparse_graph()
    graph.metadata["unserializable"] = invalid
    monkeypatch.setattr(workflow, "manufacturable_graph", lambda config: graph)
    target = tmp_path / "trajectory.npz"
    summary = tmp_path / "summary.json"
    target.write_bytes(b"previous trajectory")
    summary.write_text("previous summary")

    with pytest.raises((ValueError, TypeError)):
        workflow.ReducedRecruitmentWorkflow(tmp_path).run()
    assert target.read_bytes() == b"previous trajectory"
    assert summary.read_text() == "previous summary"
    assert not list(tmp_path.glob("*.tmp.*"))
