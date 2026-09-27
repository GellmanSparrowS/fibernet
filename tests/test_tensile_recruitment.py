"""Public API and APP compatibility checks for tensile recruitment."""
from types import SimpleNamespace

import numpy as np
import pytest

from fibernet.analysis import analyze_tensile_recruitment, compute_percolation


def test_small_spanning_path_and_hysteresis():
    edges = np.array([[0, 1], [1, 2], [0, 2]])
    strain = np.array([[0.2, 0.3, -0.1], [0.02, 0.3, 0.0]])
    result = analyze_tensile_recruitment(
        strain, edges, [0], [2], 3, alpha=0.1, hysteresis=0.5)
    assert result.active_edges.tolist() == [[True, True, False],
                                            [True, True, False]]
    assert result.perc_frame == 0
    np.testing.assert_allclose(result.backbone_frac, [2 / 3, 2 / 3])
    assert not result.edge_in_spanning[:, 2].any()


def test_app_run_adapter_matches_independent_api():
    edges = np.array([[0, 1], [1, 2]])
    strain = np.array([[0.0, 0.0], [0.2, 0.3]])
    run = SimpleNamespace(edge_strain=strain, edges=edges,
                          left_nodes=np.array([0]), right_nodes=np.array([2]),
                          frames_xy=np.zeros((2, 3, 2)))
    direct = analyze_tensile_recruitment(strain, edges, [0], [2], 3)
    adapted = compute_percolation(run)
    for field in ('active_edges', 'edge_in_spanning', 'spanning_frac',
                  'backbone_frac', 'edge_depth_norm', 'threshold_curve'):
        np.testing.assert_array_equal(getattr(direct, field),
                                      getattr(adapted, field))
    assert direct.perc_frame == adapted.perc_frame == 1


@pytest.mark.parametrize('strain,edges,left,right,nodes', [
    ([[0.1]], [[0, 2]], [0], [1], 2),
    ([[float('nan')]], [[0, 1]], [0], [1], 2),
    ([[0.1, 0.2]], [[0, 1]], [0], [1], 2),
    ([[0.1]], [[0, 1]], [], [1], 2),
])
def test_invalid_input_is_rejected(strain, edges, left, right, nodes):
    with pytest.raises(ValueError):
        analyze_tensile_recruitment(strain, edges, left, right, nodes)


def test_app_import_uses_same_calculation():
    from pathlib import Path
    import sys

    app_root = Path(__file__).resolve().parents[1] / 'FiberScope'
    sys.path.insert(0, str(app_root))
    try:
        from fslab.percolation import compute_percolation as app_compute
        assert app_compute is compute_percolation
    finally:
        sys.path.remove(str(app_root))
