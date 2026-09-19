"""Performance baseline harness (before/after numbers for optimization work).

Usage:
    python scripts/bench_perf.py [--quick] [--profile] [--gui]

Measures: structure build, engine2 wall time, inverse-design single eval,
feature extraction, optional cProfile hot spots, optional GUI/canvas cost.
"""
import argparse
import cProfile
import io
import os
import pstats
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from fslab import StructureFactory  # noqa: E402
from fslab.engine2 import Engine2, _grips  # noqa: E402
from fslab.features import compute_features  # noqa: E402
from fslab.inverse import fast_cfg  # noqa: E402
from fslab.simcache import RunConfig  # noqa: E402

BUILD_CASES = [("square", 3, 5), ("square", 6, 6), ("square", 8, 6),
               ("chiral", 3, 2), ("voronoi", 3, 2), ("reentrant", 4, 4)]
ENGINE_CASES = [(3, 5), (6, 6)]


def _timed(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return time.perf_counter() - t0, out


def bench_build():
    print("[build]")
    for unit, gx, pts in BUILD_CASES:
        f = StructureFactory(unit=unit, grid_x=gx, grid_y=gx,
                             n_pts_per_side=pts, seed=7).clamped()
        dt, g = _timed(f.build)
        print(f"  {unit:9s} {gx}x{gx} pts{pts}: {dt * 1000:7.1f} ms "
              f"N={g.num_nodes:5d} E={g.num_edges:5d}")


def bench_engine(quick=False):
    print("[engine]")
    for gx, pts in ENGINE_CASES:
        f = StructureFactory(unit="square", grid_x=gx, grid_y=gx,
                             n_pts_per_side=pts, seed=7).clamped()
        g = f.build()
        cfg = RunConfig().engine_cfg(g.num_nodes)
        dt, res = _timed(Engine2(g, cfg).run)
        print(f"  N={g.num_nodes:5d} steps={cfg.num_steps}: {dt:6.2f} s "
              f"frames={res.n_frames} E={res.n_edges}")
    f = StructureFactory(unit="square", grid_x=3, grid_y=3,
                         n_pts_per_side=2, seed=7).clamped()
    g = f.build()
    dt, _ = _timed(Engine2(g, fast_cfg(2.0).engine_cfg()).run)
    print(f"  inverse-fast N={g.num_nodes}: {dt * 1000:7.1f} ms/eval")
    if quick:
        return
    f = StructureFactory(unit="voronoi", grid_x=3, grid_y=3,
                         n_pts_per_side=2, seed=7).clamped()
    g = f.build()
    cfg = RunConfig().engine_cfg(g.num_nodes)
    dt, _ = _timed(Engine2(g, cfg).run)
    print(f"  voronoi N={g.num_nodes} steps={cfg.num_steps}: {dt:6.2f} s")


def bench_profile():
    f = StructureFactory(unit="square", grid_x=6, grid_y=6,
                         n_pts_per_side=6, seed=7).clamped()
    g = f.build()
    pr = cProfile.Profile()
    pr.enable()
    Engine2(g, RunConfig().engine_cfg(g.num_nodes)).run()
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(9)
    print("[profile engine N=%d]" % g.num_nodes)
    print("\n".join(s.getvalue().splitlines()[4:19]))


def bench_features():
    print("[features]")
    f = StructureFactory(unit="square", grid_x=6, grid_y=6,
                         n_pts_per_side=5, seed=7).clamped()
    g = f.build()
    pos = np.asarray(g.node_positions(), float)[:, :2]
    edges = np.asarray(g.edge_array(), int)[:, :2]
    dt, feat = _timed(compute_features, pos, edges)
    n_scal = len([k for k, v in feat.items() if np.ndim(v) == 0])
    print(f"  N={pos.shape[0]} E={edges.shape[0]}: {dt * 1000:7.1f} ms "
          f"scalars={n_scal}")


def bench_gui():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from studio.network_canvas import NetworkCanvas
    app = QApplication.instance() or QApplication([])
    print("[gui]")
    t0 = time.perf_counter()
    from studio.main import MainWindow
    win = MainWindow(lang="zh", mode="light")
    t1 = time.perf_counter()
    win.show()
    app.processEvents()
    print(f"  MainWindow construct {(t1 - t0) * 1000:.0f} ms, "
          f"show+paint {(time.perf_counter() - t1) * 1000:.0f} ms")
    for i in range(win.tabs.count()):
        win.tabs.setCurrentIndex(i)
        t3 = time.perf_counter()
        app.processEvents()
        print(f"  tab {i} {win.tabs.tabText(i):8s} first show "
              f"{(time.perf_counter() - t3) * 1000:6.0f} ms")
    print("[canvas]")
    for gx, pts in [(3, 5), (6, 6), (8, 6)]:
        f = StructureFactory(unit="square", grid_x=gx, grid_y=gx,
                             n_pts_per_side=pts, seed=7).clamped()
        g = f.build()
        pos = np.asarray(g.node_positions(), float)[:, :2]
        edges = np.asarray(g.edge_array(), int)[:, :2]
        left, right = _grips(pos[:, 0], 0.10)
        cv = NetworkCanvas(mode="light")
        cv.resize(900, 700)
        cv.set_static(pos, edges, left, right)
        dt, _ = _timed(cv._render_scene)
        run = Engine2(g, RunConfig(target_stretch=2.0, num_steps=4000,
                                   n_increments=40, save_interval=200
                                   ).engine_cfg(pos.shape[0])).run()
        cv.set_data(run, None)
        ts = []
        step = max(1, run.n_frames // 8)
        for fr in range(0, run.n_frames, step):
            cv.set_frame(fr)
            dt2, _ = _timed(cv._render_scene)
            ts.append(dt2)
        print(f"  N={pos.shape[0]:5d}: static {dt * 1000:5.1f} ms, "
              f"frame mean {np.mean(ts) * 1000:5.1f} ms "
              f"max {np.max(ts) * 1000:5.1f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip the slow voronoi engine case")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()
    bench_build()
    bench_engine(quick=args.quick)
    if args.profile:
        bench_profile()
    bench_features()
    if args.gui:
        bench_gui()


if __name__ == "__main__":
    main()