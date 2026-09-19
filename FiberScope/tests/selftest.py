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
        assert n_moved == n_lines * 2 * 3, \
            'expected %d moved intermediates, got %d: %s' % (
                n_lines * 2 * 3, n_moved, unit)
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
                          line_displacements=[[.02,.04],[.03,.05]], n_pts_per_side=2, perturbation=0.3, seed=5)
    fb = StructureFactory(unit="square", grid_x=2, grid_y=2,
                          line_displacements=[[.02,.04],[.03,.05]], n_pts_per_side=2, perturbation=0.3, seed=5)
    fc = StructureFactory(unit="square", grid_x=2, grid_y=2,
                          line_displacements=[[.02,.04],[.03,.05]], n_pts_per_side=2, perturbation=0.3, seed=6)
    pa = np.asarray(fa.build().node_positions(), float)
    pb = np.asarray(fb.build().node_positions(), float)
    pc = np.asarray(fc.build().node_positions(), float)
    assert np.allclose(pa, pb), "jitter not deterministic in seed"
    assert not np.allclose(pa, pc), "seed has no effect"
    print("[selftest] seeded jitter ok")
    # S0: cache identity + robustness (a cache problem must never break a run)
    import io
    import re
    import shutil
    import tempfile

    from fslab.simcache import (ENGINE_SRC_HASH, cache_tag, default_cache_dir,
                                engine_src_hash)
    src_h = engine_src_hash()
    # read the recorded constant from disk: a same-second rewrite of equal
    # size can leave a stale .pyc behind and fake a mismatch
    _sim = io.open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "fslab", "simcache.py"),
        encoding="utf-8").read()
    rec_h = re.search(r'^ENGINE_SRC_HASH = "([^"]*)"', _sim, re.M)
    rec_h = rec_h.group(1) if rec_h else ENGINE_SRC_HASH
    assert src_h is None or src_h == rec_h, (
        "numeric core changed without a cache-identity bump: run "
        "scripts/bump_engine_version.py (src=%s recorded=%s)" % (src_h, rec_h))
    scfg = RunConfig(target_stretch=1.6, num_steps=1200, n_increments=12,
                     save_interval=400)
    small = StructureFactory(unit="square", grid_x=2, grid_y=2,
                             n_pts_per_side=1, seed=7).clamped()
    tag = cache_tag(small, scfg)
    assert cache_tag(small, RunConfig(target_stretch=1.8, num_steps=1200,
                                      n_increments=12,
                                      save_interval=400)) != tag, \
        "cache tag ignores run config"
    other = StructureFactory(unit="square", grid_x=2, grid_y=2,
                             n_pts_per_side=1, seed=8).clamped()
    assert cache_tag(other, scfg) != tag, "cache tag ignores structure spec"

    d = tempfile.mkdtemp(prefix="fscache")
    try:
        fresh = run_stretch(small, scfg, cache_dir=d)
        assert os.path.exists(os.path.join(d, "run_%s.npz" % tag)), \
            "cache file not written"
        cached = run_stretch(small, scfg, cache_dir=d)
        assert np.array_equal(fresh.force_curve, cached.force_curve), \
            "cached run differs from fresh run"
        blocker = os.path.join(d, "blocker")
        with open(blocker, "w", encoding="utf-8") as fh:
            fh.write("x")
        ro = run_stretch(small, scfg, cache_dir=os.path.join(blocker, "sub"))
        assert np.array_equal(fresh.force_curve, ro.force_curve), \
            "unwritable cache path changed the result"
    finally:
        shutil.rmtree(d, ignore_errors=True)
    cdir = default_cache_dir()
    assert cdir and os.path.isdir(cdir), "no writable cache dir found"
    print(f"[selftest] cache ok: tag={tag} dir={cdir}")

    # S0: retired units still resolve, and the shipped log stays loadable
    from fslab.structure import resolve_unit
    assert resolve_unit("square") == ("square", None)
    # kagome is live again as a parity-correct cell-graph unit (F8a); the
    # fibernet original (odd-degree midpoints) is no longer reachable
    assert resolve_unit("kagome") == ("kagome", None)
    assert resolve_unit("honeycomb")[0] == "hexagon"
    assert resolve_unit("nonexistent_unit")[0] == "square"
    log = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data", "exploration_log.jsonl")
    if os.path.exists(log):
        import json
        units = set()
        with io.open(log, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    units.add(json.loads(line)["unit"])
        unknown = sorted(u for u in units
                         if (resolve_unit(u)[1] or "").startswith("unknown"))
        legacy = sorted(u for u in units if resolve_unit(u)[1])
        assert not unknown, "exploration log has unloadable units: %s" % unknown
        print(f"[selftest] replay units ok: {len(units)} units, "
              f"legacy remap {legacy or 'none'}")

    # S4e: CSV exporters must round-trip the arrays they claim to write
    import csv
    from fslab.exporter import (export_features_csv, export_hist_csv,
                                export_inverse_csv, export_run_csv)
    from fslab.features import compute_features
    from fslab.inverse import InverseRecord

    def _rows(p):
        with io.open(p, encoding="utf-8-sig", newline="") as fh:
            return list(csv.reader(fh))

    g = factory.build()
    pos = np.asarray(g.node_positions(), float)[:, :2]
    edg = np.asarray(g.edge_array(), int)[:, :2]
    feats = compute_features(pos, edg)
    g2 = StructureFactory(unit="reentrant", grid_x=3, grid_y=3,
                          n_pts_per_side=2, seed=8).build()
    feats_b = compute_features(
        np.asarray(g2.node_positions(), float)[:, :2],
        np.asarray(g2.edge_array(), int)[:, :2])
    d = tempfile.mkdtemp(prefix="fsexport")
    try:
        rows = _rows(export_run_csv(run, os.path.join(d, "run.csv"),
                                    perc=perc))
        assert rows[0][:3] == ["frame", "strain", "force"], rows[0][:3]
        assert len(rows) == run.n_frames + 1, "not one row per frame"
        assert abs(float(rows[-1][2]) - run.force_curve[-1]) < 1e-6
        assert abs(float(rows[-1][9]) - perc.spanning_frac[-1]) < 1e-6

        rows = _rows(export_features_csv(feats, os.path.join(d, "f.csv"),
                                         batch=[feats, feats_b]))
        assert rows[0][:5] == ["key", "name", "group", "unit", "value"]
        assert "batch_mean" in rows[0], "batch stats missing"
        keys = [r[0] for r in rows[1:]]
        assert len(keys) == len(set(keys)), "duplicate feature rows"
        assert len(keys) >= 15, f"only {len(keys)} scalar features"
        table = {r[0]: r for r in rows[1:]}
        for k in keys:
            assert abs(float(table[k][4]) - float(feats[k])) < 1e-6, k

        n_hist = len(_rows(export_hist_csv(feats, os.path.join(d, "h.csv"))))
        assert n_hist > 100, f"histogram export too small ({n_hist})"

        recs = [InverseRecord(1, "screen", "square", [], 0.5, 0.5),
                InverseRecord(2, "refine", "square", [0.1, -0.2], 0.3, 0.3)]
        rows = _rows(export_inverse_csv(recs, os.path.join(d, "i.csv")))
        assert rows[0] == ["eval_id", "stage", "label", "dist", "best_dist",
                           "p0", "p1"], rows[0]
        assert rows[1][5:] == ["", ""], "short params not padded"
        assert float(rows[2][5]) == 0.1 and float(rows[2][6]) == -0.2
        print(f"[selftest] export ok: run={run.n_frames} rows "
              f"feat={len(keys)} hist={n_hist - 1} inv={len(rows) - 1}")
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("[selftest] PASS")


if __name__ == "__main__":
    main()
