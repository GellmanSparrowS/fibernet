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
    # M1: line profile replicated periodically (same unit position ->
    # same displacement) across the whole lattice
    from fslab.structure import CELL as _CELL
    ld = [[0.05, 0.10], [-0.05, -0.06], [0.02, 0.08]]
    kw = dict(unit="hexagon", grid_x=3, grid_y=3, n_pts_per_side=3, seed=1)
    p0 = np.asarray(StructureFactory(**kw).build().node_positions(),
                    float)[:, :2]
    p1 = np.asarray(StructureFactory(**kw,
                        line_displacements=ld).build().node_positions(),
                    float)[:, :2]
    delta = p1 - p0
    assert np.abs(delta).max() > 1e-3, "line disp not applied"
    groups = {}
    for i in range(p0.shape[0]):
        key = (round(float(p0[i, 0]) % _CELL, 2),
               round(float(p0[i, 1]) % _CELL, 2))
        groups.setdefault(key, []).append(i)
    n_moved = 0
    for key, ids in groups.items():
        d0 = delta[ids[0]]
        assert np.allclose(delta[ids], d0, atol=1e-6), \
            f"replication not periodic at {key}"
        if np.hypot(d0[0], d0[1]) > 1e-3:
            n_moved += 1
    assert n_moved >= 6, "too few displaced unit positions"
    print(f"[selftest] line replication ok: {n_moved} unit positions "
          f"displaced periodically")

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
