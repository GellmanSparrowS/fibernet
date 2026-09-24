"""Exercise the documented three-line FEM user journey."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_fem_quickstart():
    import fibernet as fn
    from fibernet.ml import BeamFrameFEM

    graph = fn.pattern_2d(unit='honeycomb', box=(10, 10), grid=(2, 2),
                          radius=0.05)
    result = BeamFrameFEM(E=1e9, nu=0.3).stretch_test(
        graph, target_stretch=1.01)
    expected = np.linalg.norm(np.asarray(result['u'])[:, :2], axis=1).max()
    assert result['max_displacement'] > 0
    assert np.isclose(result['max_displacement'], expected)
    assert np.isfinite(result['sigma_total']).all()


if __name__ == '__main__':
    test_fem_quickstart()
    print('[test_fem_quickstart] PASS')
