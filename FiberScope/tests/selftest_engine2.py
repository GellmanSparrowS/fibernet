"""engine2 validation: physics must be right before any GUI uses it.

Run: python tests/selftest_engine2.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fslab import StructureFactory
from fslab.engine2 import Engine2, Engine2Config


def build(unit, stretch, pts=2, ld=None, **kw):
    g = StructureFactory(unit=unit, grid_x=3, grid_y=3,
                         n_pts_per_side=pts, seed=7,
                         line_displacements=ld).build()
    cfg = Engine2Config(target_stretch=stretch, num_steps=8000,
                        save_interval=1000, **kw)
    return Engine2(g, cfg).run()


def frac(r):
    e = r.energies
    tot = e["axial"] + e["bend"] + e["contact"] + 1e-12
    return e["bend"] / tot, e["axial"] / tot, e["contact"] / tot


def main():
    # 1) mode contrast: rigid straight beams (pts=0) are stretch-dominated,
    #    resolvable fibers (pts=5) engage bending along their length
    r0 = build('square', 1.3, pts=0)
    r5 = build('square', 1.3, pts=5)
    b0, a0, _ = frac(r0)
    b5, a5, _ = frac(r5)
    print(f'[e2] mode contrast: pts0 bend_frac={b0[-1]:.2f} '
          f'pts5 bend_frac={b5[-1]:.2f}')
    assert b5[-1] > b0[-1], 'resolvable fibers should engage bending'
    assert a0[-1] > a5[-1], 'rigid straight beams should be stretch-dominated'

    # 2) contact engages progressively on a bowed, laterally-contracting unit
    #    (pristine primitives carry no deformation by design).  chiral's
    #    rotating arms interlock as the sample narrows: 39 pairs in contact
    #    at stretch 1.4 vs 112 at 2.2.  cross served before the F6 grip
    #    change; with whole-column width bands its fibers no longer meet in
    #    ramp mode.  Amplitudes are fractions of the edge length, matching
    #    what LineEditor emits.
    bow = [[0.0, 0.0], [0.0, -0.225], [0.0, -0.45], [0.0, -0.225],
           [0.0, 0.0]]
    rc = build("chiral", 2.2, pts=5, use_contact=True, ld=bow)
    print(f"[e2] contact PE end={rc.energies['contact'][-1]:.1f} "
          f"(axial {rc.energies['axial'][-1]:.1f})")
    assert rc.energies["contact"][-1] > 0, "no contact events at high stretch"

    r_low = build("chiral", 1.4, pts=5, use_contact=True, ld=bow)
    print(f"[e2] contact engagement: pairs@1.4={r_low.contact_counts[-1]} "
          f"pairs@2.2={rc.contact_counts[-1]}")
    assert rc.contact_counts[-1] > r_low.contact_counts[-1], \
        "contact should engage progressively with stretch"

    # 3) energy balance (undamped): work in == KE + PE
    g = StructureFactory(unit="reentrant", grid_x=3, grid_y=3,
                         n_pts_per_side=2, seed=7).build()
    cfg = Engine2Config(target_stretch=1.5, num_steps=3000, save_interval=500,
                        drag=0.0)
    r = Engine2(g, cfg).run()
    pe = (r.energies["axial"] + r.energies["bend"] + r.energies["contact"])[-1]
    ke = r.energies["kinetic"][-1]
    w = r.energies["work"][-1]
    err = abs(w - (pe + ke)) / max(abs(w), 1.0)
    print(f"[e2] energy balance: W={w:.3e} PE+KE={pe + ke:.3e} err={err:.2%}")
    assert err < 0.15, "energy not conserved in undamped run"

    # 4) damped run settles (quasi-static end state)
    assert rc.energies["kinetic"][-1] < 0.05 * rc.energies["axial"][-1] + 1.0, \
        "damped run did not settle"
    print(f"[e2] settling ok: KE_end={rc.energies['kinetic'][-1]:.1f}")

    # 5) determinism + speed
    r_a = build("reentrant", 2.0)
    r_b = build("reentrant", 2.0)
    assert np.allclose(r_a.frames_xy, r_b.frames_xy), "non-deterministic engine"
    t0 = time.time()
    build("reentrant", 1.5)
    dt = time.time() - t0
    print(f"[e2] determinism ok; speed 8000 steps {dt:.1f}s "
          f"(wall {r_a.metadata['wall_seconds']}s)")
    assert dt < 15, "engine too slow for interactive use"
    # 6) quasi-static mode: uniform strain field + progress callback
    g = StructureFactory(unit="chiral", grid_x=3, grid_y=3,
                           n_pts_per_side=3, seed=7).build()
    pos0 = np.asarray(g.node_positions(), dtype=float)[:, :2]
    cfg = Engine2Config(target_stretch=1.8, num_steps=6000,
                        n_increments=60, save_interval=1000)
    prog = []
    r = Engine2(g, cfg).run(progress_cb=lambda d, t: prog.append((d, t)))
    xmid = 0.5 * (pos0[:, 0].min() + pos0[:, 0].max())
    em = pos0[r.edges].mean(axis=1)[:, 0]
    s = np.abs(r.edge_strain[-1])
    sl, sr = s[em < xmid].mean(), s[em >= xmid].mean()
    print(f"[e2] quasi-static: F={r.n_frames} left_strain={sl:.3f} "
          f"right_strain={sr:.3f} prog_calls={len(prog)}")
    assert 0.5 < sr / max(sl, 1e-6) < 2.0, "strain field not uniform"
    assert len(prog) == 61 and prog[-1][0] == prog[-1][1], "progress cb broken"

    # 7) golden reference: numerics must not drift when the solver is
    #    refactored (regenerate with scripts/make_golden_engine.py)
    golden_check()
    print("[e2] PASS")


def golden_check():
    """Re-run the frozen cases and compare every stored array."""
    import json

    from fslab.simcache import ENGINE_VERSION
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "golden_engine.npz")
    if not os.path.exists(ref_path):
        print("[e2] golden: no reference file, skipped")
        return
    sys.path.insert(0, os.path.join(root, "scripts"))
    from make_golden_engine import CASES, run_case
    worst_key, worst = "", 0.0
    n_arr = 0
    t0 = time.time()
    with np.load(ref_path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        assert meta["engine_version"] == ENGINE_VERSION, (
            "golden reference was made with engine %s, code is %s: run "
            "scripts/make_golden_engine.py" % (meta["engine_version"],
                                               ENGINE_VERSION))
        for case in CASES:
            res, _dt = run_case(case)
            n = case[0]
            pairs = [(n + "_force", res.force_curve),
                     (n + "_strain", res.strain_levels),
                     (n + "_ccounts", res.contact_counts),
                     (n + "_final", res.frames_xy[-1]),
                     (n + "_estrain", res.edge_strain[-1])]
            pairs += [("%s_e_%s" % (n, k), np.asarray(v, float))
                      for k, v in res.energies.items()]
            for key, arr in pairs:
                ref = np.asarray(z[key], float)
                got = np.asarray(arr, float)
                assert got.shape == ref.shape, (key, got.shape, ref.shape)
                scale = max(float(np.abs(ref).max()), 1e-12)
                dev = float(np.abs(got - ref).max()) / scale
                n_arr += 1
                if dev > worst:
                    worst, worst_key = dev, key
                assert dev < 1e-9, (
                    "golden mismatch %s: rel dev %.3e (refactor changed the "
                    "physics)" % (key, dev))
    rec = sum(c["seconds"] for c in meta["cases"])
    print("[e2] golden ok: %d arrays, max rel dev %.2e (%s), "
          "%.1fs vs %.1fs recorded" % (n_arr, worst, worst_key,
                                       time.time() - t0, rec))


if __name__ == "__main__":
    main()
