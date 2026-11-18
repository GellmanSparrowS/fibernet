"""Missing measurements and identities cannot define a recruitment path."""
from collections import deque
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from fibernet.analysis import analyze_tensile_recruitment, compute_percolation


def inputs():
    return dict(
        edge_strain=np.array([[.02, .03], [.01, .025]]),
        edges=np.array([[0, 1], [1, 2]]),
        left_nodes=np.array([0]), right_nodes=np.array([2]), n_nodes=3,
    )


@pytest.mark.parametrize('field', ['edge_strain', 'edges', 'left_nodes', 'right_nodes'])
@pytest.mark.parametrize('form', ['array', 'nested', 'object'])
def test_missing_values_are_rejected_before_numpy_conversion(field, form):
    data = inputs()
    masked = np.ma.array(data[field], mask=False, copy=True)
    masked.mask.flat[0] = True
    if form == 'array':
        value = masked
    elif form == 'nested':
        value = deque(masked if masked.ndim == 2 else [masked])
    else:
        value = masked.data.astype(object)
        value.flat[0] = np.ma.masked
    data[field] = value
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(ValueError, match=field + '.*masked'):
            analyze_tensile_recruitment(**data)
    np.testing.assert_array_equal(masked.data, inputs()[field])
    assert masked.mask.sum() == 1


@pytest.mark.parametrize('quantile', [None, .25])
def test_unmasked_arrays_keep_the_complete_output(quantile):
    data = inputs()
    expected = analyze_tensile_recruitment(**data, quantile=quantile)
    for field in ('edge_strain', 'edges', 'left_nodes', 'right_nodes'):
        data[field] = np.ma.array(data[field], mask=False)
    actual = analyze_tensile_recruitment(**data, quantile=quantile)
    for key, value in vars(expected).items():
        np.testing.assert_array_equal(getattr(actual, key), value)


def test_run_adapter_rejects_a_hidden_grip():
    data = inputs()
    data.pop('n_nodes')
    data['right_nodes'] = np.ma.array([2], mask=True)
    run = SimpleNamespace(**data, frames_xy=np.zeros((2, 3, 2)))
    with pytest.raises(ValueError, match='right_nodes.*masked'):
        compute_percolation(run)


def test_generated_octet_missing_cut_cannot_determine_spanning():
    import fibernet as fn
    from fibernet.ml.beam_frame_fem import BeamFrameFEM

    graph = fn.pattern_3d(unit='octet', grid=(2, 2, 2), box=(.01, .015, .02),
                          radius=.0005, n_pts_per_side=0, seed=23)
    grips = BeamFrameFEM.graph_to_fem_input(graph, dim=3, pct=.1)['boundaries']
    result = fn.simulate(graph, backend='fem', mode='stretch', stretch_ratio=1.002,
                         nonlinear=False, E=1e6, nu=.3, pct=.1)
    node_ids = result.metadata['node_ids']
    rows = {nid: row for row, nid in enumerate(node_ids)}
    members = [graph.edges[eid] for eid in result.metadata['edge_ids']]
    edges = np.array([[rows[e.node_i], rows[e.node_j]] for e in members])
    strain = np.stack([np.zeros(len(members)), result.edge_stretches - 1.])
    args = edges, grips['left'], grips['right'], len(node_ids)
    complete = analyze_tensile_recruitment(strain, *args)

    x = np.array([graph.nodes[nid].position[0] for nid in node_ids])
    left_of_cut = x < .0075
    assert left_of_cut[grips['left']].all() and not left_of_cut[grips['right']].any()
    crossing = left_of_cut[edges[:, 0]] != left_of_cut[edges[:, 1]]
    assert complete.perc_frame == 1 and complete.edge_in_spanning[1, crossing].any()
    mask = np.zeros_like(strain, dtype=bool)
    mask[1, crossing] = True
    for hidden in (0., .002):
        backing = np.where(mask, hidden, strain)
        observations = np.ma.array(backing, mask=mask, copy=True)
        np.testing.assert_array_equal(observations.compressed(), strain[~mask])
        with pytest.raises(ValueError, match='edge_strain.*masked'):
            analyze_tensile_recruitment(observations, *args)
        np.testing.assert_array_equal(observations.data, backing)
        np.testing.assert_array_equal(observations.mask, mask)
