"""Finals regression suite. Run: python tests/selftest_finals.py."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fslab.structure import StructureFactory, CELL
from fslab.cell_rules import RULES, graph_health


def main():
    keys = set()
    for rule in RULES:
        f = StructureFactory(unit='octagon', grid_x=3, grid_y=2,
                             n_pts_per_side=2, expansion_rule=rule)
        g = f.build()
        pos, edges = g.node_positions(), g.edge_array()
        assert np.isfinite(pos).all()
        assert graph_health(pos, edges)['components'] == 1
        assert np.array_equal(pos, f.build().node_positions())
        keys.add(f.key())
    assert len(keys) == len(RULES)
    from fslab.cell_rules import transform_cell
    p = np.array([[0.2, 0.3]])
    assert np.allclose(transform_cell(p, 1, 0, 'mirror', 1), [[1.8, .3]])
    assert np.allclose(transform_cell(p, 1, 0, 'rotate', 1), [[1.8, .7]])
    from fslab.cell_rules import validate_rule
    table = [[{'turn': 1, 'flip': True}, {'turn': 0, 'flip': False}]]
    assert np.allclose(transform_cell(p, 0, 0, 'custom', 1, table), [[.7, .8]])
    a = StructureFactory(expansion_rule='custom', custom_rule=table)
    assert a.key() != StructureFactory(expansion_rule='custom').key()
    assert np.isfinite(a.build().node_positions()).all()
    try:
        validate_rule([[{'turn': 9}]])
        raise AssertionError('invalid period accepted')
    except ValueError:
        pass
    print('[finals] expansion determinism, connectivity and cache identity ok')
    from fslab.contact import ContactConfig, candidate_pairs
    from fslab.features import compute_features
    p = np.array([[0., 0.], [2., 0.], [0., .15], [2., .15]])
    e = np.array([[0, 1], [2, 3]])
    narrow = ContactConfig(width=.1).compute(p, e)
    wide = ContactConfig(width=.3).compute(p, e)
    assert narrow['contact_pair_count'] == 0
    assert wide['contact_pair_count'] == 1
    assert wide['contact_edge_ratio'] == 1
    assert wide['contact_graph_max'] == 2
    assert wide['contact_patch_count'] == 1
    assert abs(wide['contact_overlap_area'] - .3276) < .02
    finer = ContactConfig(width=.3, resolution=1024).compute(p, e)
    assert abs(wide['contact_overlap_area']-finer['contact_overlap_area']) < .01
    assert not list(candidate_pairs(p, np.array([[0, 1], [1, 3]]), .3))
    collinear = np.array([[0., 0.], [2., 0.], [.5, 0.], [1.5, 0.]])
    feats = compute_features(collinear, e, include_contact=True)
    assert feats['cross_count'] == 0
    assert feats['contact_cluster_max'] == 2
    assert feats['overlap_len_mean'] == 1
    try:
        ContactConfig(width=.3, max_pixels=1).compute(p, e)
        raise AssertionError('pixel memory budget ignored')
    except MemoryError:
        pass
    print('[finals] finite width, adjacency exclusion, overlap clusters and budgets ok')


if __name__ == '__main__':
    main()
