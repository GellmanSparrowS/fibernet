"""Freeze reference engine2 outputs so any numerics change can be verified.

Usage:
    python scripts/make_golden_engine.py            # write tests/golden_engine.npz
    python scripts/make_golden_engine.py --check    # compare only, do not write

Each case is a (unit, grid, pts, RunConfig) tuple run straight through
Engine2 (no cache).  The npz stores force curve, the five energy channels,
contact counts and the final frame, so tests/selftest_engine2.py can assert
that a refactor is numerically identical.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from fslab import StructureFactory
from fslab.engine2 import Engine2
from fslab.simcache import ENGINE_VERSION, RunConfig

OUT = os.path.join(ROOT, "tests", "golden_engine.npz")

# (name, unit, grid, pts, seed, kwargs) - kept small so the check stays fast
CASES = [
    ("qs_square", "square", 3, 5, 7,
     dict(target_stretch=2.0, num_steps=3000, n_increments=30,
          save_interval=250)),
    # deformed square path: locks the twin-fiber (lens) shared boundaries
    ("qs_bow", "auxetic_bow", 3, 5, 7,
     dict(target_stretch=1.8, num_steps=3000, n_increments=30,
          save_interval=250)),
    ("qs_chiral", "chiral", 3, 2, 7,
     dict(target_stretch=2.2, num_steps=3000, n_increments=30,
          save_interval=250)),
    ("ramp_reentrant", "reentrant", 3, 2, 7,
     dict(target_stretch=1.8, num_steps=3000, n_increments=0,
          save_interval=250, ramp_fraction=0.3)),
]


def run_case(case):
    name, unit, grid, pts, seed, kw = case
    f = StructureFactory(unit=unit, grid_x=grid, grid_y=grid,
                         n_pts_per_side=pts, seed=seed, topology='legacy').clamped()
    g = f.build()
    cfg = RunConfig(**kw).engine_cfg(g.num_nodes)
    t0 = time.time()
    res = Engine2(g, cfg).run()
    return res, time.time() - t0


def collect():
    arrays, meta = {}, []
    for case in CASES:
        res, dt = run_case(case)
        n = case[0]
        arrays[n + "_force"] = res.force_curve
        arrays[n + "_strain"] = res.strain_levels
        arrays[n + "_ccounts"] = res.contact_counts
        arrays[n + "_final"] = res.frames_xy[-1]
        arrays[n + "_estrain"] = res.edge_strain[-1]
        for k, v in res.energies.items():
            arrays["%s_e_%s" % (n, k)] = np.asarray(v, float)
        meta.append(dict(name=n, unit=case[1], grid=case[2], pts=case[3],
                         seed=case[4], kw=case[5], seconds=round(dt, 2),
                         n_frames=res.n_frames, n_edges=res.n_edges,
                         n_nodes=int(res.frames_xy.shape[1])))
        print("  %-16s F=%3d N=%4d E=%4d  %5.2fs  F_end=%.6e"
              % (n, res.n_frames, res.frames_xy.shape[1], res.n_edges, dt,
                 res.force_curve[-1]))
    return arrays, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    print("[golden] engine version", ENGINE_VERSION)
    arrays, meta = collect()
    if args.check:
        return 0
    arrays["meta"] = np.array(json.dumps(
        {"engine_version": ENGINE_VERSION, "cases": meta}))
    tmp = OUT + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    _atomic_write(tmp, OUT)
    print("[golden] wrote %s (%.1f KB)" % (OUT, os.path.getsize(OUT) / 1024))
    return 0


def _atomic_write(tmp, dst):
    try:
        os.replace(tmp, dst)
    except OSError:
        import shutil
        shutil.move(tmp, dst)


if __name__ == "__main__":
    sys.exit(main())
