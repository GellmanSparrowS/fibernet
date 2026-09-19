"""Pre-build gate: prove the app never touches modules we plan to exclude.

Installs a meta_path blocker for every candidate module, then runs each test
suite in a child process.  A suite that still passes means the frozen bundle
can safely ship without those packages (size win, no runtime surprise).

Usage:
    python scripts/check_runtime_deps.py                 # all suites
    python scripts/check_runtime_deps.py --only gui_smoke
    python scripts/check_runtime_deps.py --list
"""
import argparse
import os
import re
import runpy
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SUITES = ["probe", "selftest", "selftest_engine2", "selftest_inverse",
          "gui_smoke", "selftest_learning", "selftest_surface", "selftest_manufacturing"]

def build_excludes():
    """The EXCLUDES list of scripts/build_exe.py, parsed from source.

    The gate must mirror the build exactly: anything listed there has to be
    provably unused at runtime, and anything needed at runtime must never be
    listed.  Parsing (instead of importing) keeps this script dependency-free.
    """
    src = open(os.path.join(ROOT, "scripts", "build_exe.py"),
               encoding="utf-8").read()
    m = re.search(r"^EXCLUDES = \[(.*?)^\]", src, re.S | re.M)
    if not m:
        raise SystemExit("cannot parse EXCLUDES from scripts/build_exe.py")
    return re.findall(r"[\"\']([A-Za-z_][\w.]*)[\"\']", m.group(1))


# extra dev-machine packages that are not in EXCLUDES but must still be absent
# from the runtime path (they are the reason EXCLUDES exists)
EXTRA_BLOCK = ["matplotlib", "IPython", "jupyter_client", "notebook"]

BLOCK = sorted(set(build_excludes()) | set(EXTRA_BLOCK))



class _Blocker:
    """meta_path finder that makes blocked imports look uninstalled."""

    def __init__(self, names):
        self.names = list(names)
        self.hits = []

    def _blocked(self, name):
        return any(name == b or name.startswith(b + ".") for b in self.names)

    def find_spec(self, name, path=None, target=None):
        if self._blocked(name):
            self.hits.append(name)
            raise ImportError("blocked by check_runtime_deps: %s" % name)
        return None


def install(extra=()):
    blocker = _Blocker(list(BLOCK) + list(extra))
    sys.meta_path.insert(0, blocker)
    return blocker


def _probe():
    """Exercise every runtime path with the blockers installed.

    Covers all 15 units (fibernet pattern_2d incl. the classic per-edge
    spectrum rotation), percolation, features, exporters, the bundled OBJ
    surfaces, a micro inverse-design run and a micro MLP train - i.e.
    everything a judge can click in the frozen app.
    """
    import tempfile
    import numpy as np
    from fslab import (StructureFactory, RunConfig, run_stretch,
                       compute_percolation)
    from fslab.structure import UNIT_PRESETS, SPECTRUM_PRESETS
    from fslab import exporter as EXP
    from fslab import features as FEAT
    from fslab import inverse as INV
    from fslab import mlmodel as MLM
    from fslab import surface3d as S3D

    cfg = RunConfig(target_stretch=1.5, save_interval=800)
    for unit in sorted(UNIT_PRESETS):
        fac = StructureFactory(unit=unit, grid_x=2, grid_y=2,
                               n_pts_per_side=2, seed=7)
        g = fac.build()
        pos = np.asarray(g.node_positions(), float)[:, :2]
        edges = np.asarray(g.edge_array(), int)[:, :2]
        run = run_stretch(fac, cfg, cache_dir=None)
        perc = compute_percolation(run)
        feats = FEAT.compute_features(pos, edges)
        assert run.frames_xy.shape[0] > 1, "no frames: %s" % unit
        print("  unit %-13s nodes=%4d edges=%4d frames=%3d perc=%3d feats=%2d"
              % (unit, len(pos), len(edges), run.frames_xy.shape[0],
                 perc.perc_frame, len(feats)))

    ld = [[0.05, 0.10], [-0.05, -0.06], [0.02, 0.08]]
    fac = StructureFactory(unit="hexagon", grid_x=1, grid_y=1,
                           n_pts_per_side=3, seed=1, line_displacements=ld)
    fac.build()
    tmp = tempfile.mkdtemp(prefix="fsprobe_")
    EXP.export_json(fac, os.path.join(tmp, "s.json"))
    EXP.export_svg(fac, os.path.join(tmp, "s.svg"))
    print("  exporters ok ->", tmp)

    objdir = os.path.join(ROOT, "assets", "obj")
    spec = SPECTRUM_PRESETS["zigzag"]
    names = sorted(fn for fn in os.listdir(objdir) if fn.lower().endswith('.obj'))
    for fn in names:
        V, F = S3D.parse_obj(os.path.join(objdir, fn))
        P, segs = S3D.deform_surface(V, F, spec)
        assert len(P) and len(segs), "empty drape: %s" % fn
    print("  surface ok: %d OBJ draped (last P=%d segs=%d)"
          % (len(names), len(P), len(segs)))
    from fslab.surface_mapping import coarsen_quads
    reduced_v, reduced_f = coarsen_quads(V, F, 300)
    assert len(reduced_f) <= 300 and np.isfinite(reduced_v).all()

    def builder(unit, pert, ldisp):
        return StructureFactory(unit=unit, grid_x=2, grid_y=2,
                                n_pts_per_side=3, seed=7, perturbation=pert,
                                line_displacements=ldisp).build()

    res = INV.run_inverse(builder, "J", budget=4, seed=0, stretch=1.5,
                          fixed_unit="square", pts=3)
    print("  inverse ok: %d records best=%s dist=%.4f"
          % (len(res["records"]), res["best_label"], res["best_dist"]))

    # real light-feature vector + the 3 documented targets (peak/stiffness/
    # toughness) so the surrogate sees the same shapes as the ML tab
    x0 = np.asarray(MLM.light_features(pos, edges), float).ravel()
    rng = np.random.default_rng(0)
    X = np.tile(x0, (24, 1)) + rng.normal(scale=1e-3, size=(24, x0.size))
    Y = rng.normal(size=(24, len(MLM.TARGET_NAMES)))
    net = MLM.MLP(hidden=8, seed=0)
    hist = net.train(X, Y, epochs=3, batch=8)
    n_out = len(MLM.TARGET_NAMES)
    assert np.asarray(net.predict(X[:1])).shape == (1, n_out), "predict shape"
    print("  ml ok: x_dim=%d out=%d epochs_run=%s"
          % (x0.size, n_out, hist.get("epochs_run")))
    print("[probe] all runtime paths clean")
    return 0


def _child(suite):
    blocker = install()
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    if suite == "probe":
        try:
            return _probe()
        finally:
            sys.stderr.write("[blocked imports] %s\n"
                             % (", ".join(sorted(set(blocker.hits))) or "-"))
    script = os.path.join(ROOT, "tests", suite + ".py")
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        sys.stderr.write("[blocked imports] %s\n"
                         % ", ".join(sorted(set(blocker.hits))) or "-")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma separated suite names")
    ap.add_argument("--list", action="store_true", help="print block list")
    ap.add_argument("--_child", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args._child:
        sys.exit(_child(args._child))
    if args.list:
        print("\n".join(BLOCK))
        return 0
    want = args.only.split(",") if args.only else SUITES
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen",
               PYTHONDONTWRITEBYTECODE="1")
    failed = []
    for suite in want:
        t0 = time.time()
        r = subprocess.run([sys.executable, os.path.abspath(__file__),
                            "--_child", suite], cwd=ROOT, env=env,
                           capture_output=True, text=True)
        dt = time.time() - t0
        tail = [l for l in (r.stdout or "").strip().splitlines() if l][-3:]
        blk = [l for l in (r.stderr or "").splitlines()
               if l.startswith("[blocked imports]")]
        print("== %-20s exit=%d  %.1fs" % (suite, r.returncode, dt))
        for l in tail:
            print("   ", l)
        if blk:
            print("   ", blk[-1])
        if r.returncode != 0:
            failed.append(suite)
            for l in [x for x in (r.stderr or "").strip().splitlines()
                      if not x.startswith("[blocked imports]")][-12:]:
                print("   !", l)
    print("SUMMARY: %d/%d suites clean without %d blocked packages"
          % (len(want) - len(failed), len(want), len(BLOCK)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
