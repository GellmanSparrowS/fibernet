"""Math-core selftest (no Qt). Run: python tests/selftest.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fslab import StructureFactory, RunConfig, run_stretch, compute_percolation


def main():
    factory = StructureFactory(unit="reentrant", grid_x=3, grid_y=3,
                               n_pts_per_side=2, seed=7)
    cfg = RunConfig(target_stretch=2.0, save_interval=800)
    run = run_stretch(factory, cfg, cache_dir=None)

    F, N, _ = run.frames_xy.shape
    E = run.n_edges
    assert run.edge_strain.shape == (F, E), run.edge_strain.shape
    assert run.strain_levels.shape == (F,)
    assert run.force_curve.shape == (F,)
    assert run.strain_levels[-1] > run.strain_levels[0], "sample did not stretch"
    assert run.left_nodes.size > 0 and run.right_nodes.size > 0
    assert np.all(run.force_curve >= 0)
    assert run.force_curve[-1] > 0, "no reaction force at final strain"
    print(f"[selftest] run ok: F={F} N={N} E={E} "
          f"stretch {run.strain_levels[0]:.3f}->{run.strain_levels[-1]:.3f} "
          f"F_end={run.force_curve[-1]:.1f}")

    perc = compute_percolation(run)
    assert perc.spanning_frac.shape == (F,)
    assert perc.edge_in_spanning.shape == (F, E)
    assert 0.0 <= perc.spanning_frac.max() <= 1.0
    assert perc.edge_depth_norm.min() >= 0.0 and perc.edge_depth_norm.max() <= 1.0
    assert perc.perc_frame >= 0, "no spanning cluster found in stretched reentrant grid"
    assert perc.spanning_frac[-1] > 0
    print(f"[selftest] percolation ok: perc_frame={perc.perc_frame} "
          f"P_end={perc.spanning_frac[-1]:.3f} backbone_end={perc.backbone_frac[-1]:.3f}")
    # M1a: classic path - reference spectrum is rotated into every
    # original edge of the unit (diamond/star/hexagon/voronoi), and only
    # the intermediate degree-2 nodes move (corners stay put).
    ld = [[0.05, 0.10], [-0.05, -0.06], [0.02, 0.08]]
    for unit, n_lines in (('diamond', 4), ('star', 8), ('hexagon', 6)):
        kw = dict(unit=unit, grid_x=1, grid_y=1, n_pts_per_side=3, seed=1)
        g0 = StructureFactory(**kw).build()
        g1 = StructureFactory(**kw, line_displacements=ld).build()
        p0 = np.asarray(g0.node_positions(), float)[:, :2]
        p1 = np.asarray(g1.node_positions(), float)[:, :2]
        edges = np.asarray(g0.edge_array(), int)[:, :2]
        deg = np.bincount(edges.ravel(), minlength=len(p0))
        delta = p1 - p0
        assert np.abs(delta).max() > 1e-3, 'line disp not applied: %s' % unit
        n_moved = int((np.abs(delta).max(axis=1) > 1e-6).sum())
        assert n_moved == n_lines * 3, \
            'expected %d moved intermediates, got %d: %s' % (
                n_lines * 3, n_moved, unit)
    # M1b: P1 path - welded bulk is periodic cell-to-cell
    g4 = StructureFactory(unit='square', grid_x=4, grid_y=4,
                          n_pts_per_side=3, seed=1,
                          line_displacements=ld).build()
    p4 = np.asarray(g4.node_positions(), float)[:, :2]
    boxA = p4[(p4[:, 0] >= 10) & (p4[:, 0] < 20) &
              (p4[:, 1] >= 10) & (p4[:, 1] < 20)]
    boxB = p4[(p4[:, 0] >= 20) & (p4[:, 0] < 30) &
              (p4[:, 1] >= 10) & (p4[:, 1] < 20)]
    assert len(boxA) == len(boxB) > 0, 'bulk boxes empty'
    for q in boxA:
        tgt = q + np.array([10.0, 0.0])
        assert np.abs(boxB - tgt).max(axis=1).min() < 1e-6, \
            'P1 bulk not periodic'
    print('[selftest] periodic replication ok (classic + P1 bulk)')

    # M1: perturbation jitter is seeded and live
    fa = StructureFactory(unit="square", grid_x=2, grid_y=2,
                          n_pts_per_side=2, perturbation=0.3, seed=5)
    fb = StructureFactory(unit="square", grid_x=2, grid_y=2,
                          n_pts_per_side=2, perturbation=0.3, seed=5)
    fc = StructureFactory(unit="square", grid_x=2, grid_y=2,
                          n_pts_per_side=2, perturbation=0.3, seed=6)
    pa = np.asarray(fa.build().node_positions(), float)
    pb = np.asarray(fb.build().node_positions(), float)
    pc = np.asarray(fc.build().node_positions(), float)
    assert np.allclose(pa, pb), "jitter not deterministic in seed"
    assert not np.allclose(pa, pc), "seed has no effect"
    print("[selftest] seeded jitter ok")
    print("[selftest] PASS")


if __name__ == "__main__":
    main()
