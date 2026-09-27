"""Public reduced solver matches the APP and feeds recruitment directly."""
from pathlib import Path
import sys

import numpy as np

from fibernet.gen import PlanarManufacturingConfig, manufacturable_graph
from fibernet.sim import ReducedBeamConfig, ReducedBeamSolver
from fibernet.analysis import compute_percolation
from examples.reduced_recruitment_workflow import ReducedRecruitmentWorkflow

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "FiberScope"))
from fslab.engine2 import Engine2, Engine2Config


def test_public_reduced_solver_matches_app():
    assert ReducedBeamSolver is Engine2
    assert ReducedBeamConfig is Engine2Config
    graph = manufacturable_graph(PlanarManufacturingConfig(
        unit="kagome", grid_x=2, grid_y=2, n_pts_per_side=1,
        perturbation=0.04, seed=11))
    result = ReducedBeamSolver(graph, ReducedBeamConfig(
        target_stretch=1.2, n_increments=4, num_steps=300,
        use_contact=False)).run()
    assert result.n_frames == 5
    assert result.n_edges == graph.num_edges
    assert np.isfinite(result.force_curve).all()
    recruited = compute_percolation(result, alpha=0.05)
    assert recruited.active_edges.shape == result.edge_strain.shape


def test_public_stretch_workflow_saves_replayable_arrays(tmp_path):
    summary = ReducedRecruitmentWorkflow(tmp_path, grid=2).run()
    with np.load(tmp_path / "trajectory.npz", allow_pickle=False) as saved:
        assert saved["frames_xy"].shape[0] == summary["frames"]
        assert saved["edges"].shape[0] == summary["edges"]
        assert saved["active_edges"].shape == saved["edge_strain"].shape
