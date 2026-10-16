"""The easy FEM API must not silently project generated 3D networks to 2D."""
import numpy as np
import pytest

import fibernet as fn


@pytest.mark.parametrize('unit', ['cubic', 'octet', 'bcc'])
def test_3d_fem_default_matches_explicit_3d_solver(unit):
    graph = fn.pattern_3d(unit=unit, box=(1, 1, 1), grid=(1, 1, 1))
    default = fn.simulate(graph, backend='fem', strain=1.01, nonlinear=False)
    explicit = fn.simulate(graph, backend='fem', strain=1.01, nonlinear=False, dim=3)
    assert default.metadata['reactions'].shape == (graph.num_nodes, 6)
    np.testing.assert_array_equal(default.edge_forces, explicit.edge_forces)
    np.testing.assert_array_equal(default.displacements, explicit.displacements)
    np.testing.assert_array_equal(default.edge_stretches, explicit.edge_stretches)
    assert np.isfinite(default.edge_forces).all()


def test_2d_fem_default_is_unchanged():
    graph = fn.pattern_2d(unit='honeycomb', grid=(2, 2))
    default = fn.simulate(graph, backend='fem', strain=1.01)
    explicit = fn.simulate(graph, backend='fem', strain=1.01, dim=2)
    assert default.metadata['reactions'].shape == (graph.num_nodes, 3)
    np.testing.assert_array_equal(default.edge_forces, explicit.edge_forces)
    np.testing.assert_array_equal(default.displacements, explicit.displacements)


def test_explicit_dimension_override_is_forwarded():
    graph = fn.pattern_3d(unit='octet', grid=(1, 1, 1))
    explicit = fn.simulate(graph, backend='fem', strain=1.01, dim=2)
    assert explicit.metadata['reactions'].shape == (graph.num_nodes, 3)


@pytest.mark.parametrize('dim', [0, 1, 4, 2.0, None, '3'])
def test_invalid_fem_dimension_rejected(dim):
    graph = fn.pattern_3d(unit='cubic', grid=(1, 1, 1))
    with pytest.raises(ValueError, match='dimension must be 2 or 3'):
        fn.simulate(graph, backend='fem', strain=1.01, dim=dim)
