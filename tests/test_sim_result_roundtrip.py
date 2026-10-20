"""FEM resultants and canonical result fields have distinct JSON contracts."""
import json

import numpy as np
import pytest

from fibernet import StructureGraph, simulate
from fibernet.sim.accelerated import SimResult


def _axial_result():
    graph = StructureGraph(dimension=2)
    graph.add_node([0, 0])
    graph.add_node([1, 0])
    graph.add_edge(0, 1, radius=.05)
    graph.add_edge(0, 1, radius=.10, allow_parallel=True)
    return simulate(graph, backend="fem", strain=1.01, E=1000,
                    nonlinear=False)


def test_fem_summary_does_not_reinterpret_resultants_as_axial_forces(tmp_path):
    result = _axial_result()
    path = tmp_path / "summary.json"
    result.save(path)
    data = json.loads(path.read_text())
    assert "edge_forces" not in data  # Canonical arrays require detailed=True.
    assert np.asarray(data["beam_resultants"]).shape == (2, 3)
    restored = SimResult.load(path)
    assert restored.edge_forces is None
    assert restored.max_force == result.max_force
    assert restored.energy == result.energy
    np.testing.assert_allclose(np.asarray(restored.metadata["beam_resultants"])[:, 0],
                               result.edge_forces)


def test_fem_detailed_roundtrip_preserves_forces_and_resultants(tmp_path):
    result = _axial_result()
    result.positions_trajectory = [result.deformed_positions.copy()]
    path = tmp_path / "detailed.json"
    result.save(path, detailed=True)
    restored = SimResult.load(path)
    assert restored.edge_forces.shape == (2,)
    np.testing.assert_allclose(restored.edge_forces, result.edge_forces)
    np.testing.assert_allclose(restored.edge_stretches, result.edge_stretches)
    np.testing.assert_allclose(restored.displacements, result.displacements)
    np.testing.assert_allclose(restored.metadata["beam_resultants"],
                               result.metadata["beam_resultants"])
    np.testing.assert_allclose(restored.positions_trajectory,
                               result.positions_trajectory)
    assert "positions_trajectory" not in restored.metadata
    assert "positions_trajectory" not in restored.to_dict()


def test_legacy_fem_summary_recovers_signed_axial_forces(tmp_path):
    path = tmp_path / "legacy.json"
    resultants = [[-2., 3., 4.], [5., -6., 7.]]
    incomplete = dict(mode="fem_beam_frame", n_edges=2,
                      edge_forces=resultants, max_force=5.)
    path.write_text(json.dumps(incomplete), encoding='utf-8')
    with pytest.raises(ValueError, match='recognizable mechanical'):
        SimResult.load(path)
    # Supply the complete synthetic mechanical envelope; do not infer zeros.
    complete = dict(incomplete, energy=1., n_nodes=3)
    path.write_text(json.dumps(complete), encoding='utf-8')
    restored = SimResult.load(path)
    np.testing.assert_array_equal(restored.edge_forces, [-2., 5.])
    np.testing.assert_array_equal(restored.metadata["beam_resultants"], resultants)
    restored.save(path, detailed=True)
    again = SimResult.load(path)
    np.testing.assert_array_equal(again.edge_forces, restored.edge_forces)
    np.testing.assert_array_equal(again.metadata["beam_resultants"], resultants)


def test_metadata_cannot_override_canonical_or_detail_only_fields():
    result = SimResult(mode="spring", energy=2., max_force=3., metadata={
        "mode": "fem_beam_frame", "energy": 99., "max_force": 100.,
        "edge_forces": [[1., 2., 3.]], "positions_trajectory": [[[0, 0, 0]]],
        "provenance": "synthetic",
    })
    summary = result.to_dict()
    assert summary["mode"] == "spring"
    assert summary["energy"] == 2.
    assert summary["max_force"] == 3.
    assert summary["provenance"] == "synthetic"
    assert "edge_forces" not in summary
    assert "positions_trajectory" not in summary
