"""Inverse design validation. Run: python tests/selftest_inverse.py"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fslab import StructureFactory
from fslab.inverse import run_inverse, target_curve, metrics_of


def main():
    def builder(unit, pert, ld):
        return StructureFactory(unit=unit, grid_x=3, grid_y=3,
                                n_pts_per_side=2, seed=7,
                                perturbation=pert,
                                line_displacements=ld).build()

    t0 = time.time()
    res = run_inverse(builder, "J", budget=40, seed=3)
    dt = time.time() - t0
    screen = [r for r in res["records"] if r.stage == "screen"]
    refine = [r for r in res["records"] if r.stage == "refine"]
    d_screen_best = min(r.dist for r in screen)
    print(f"[inv] screen_best={d_screen_best:.3f} "
          f"final={res['best_dist']:.3f} best={res['best_label']} ({dt:.1f}s)")
    assert res["best_dist"] <= d_screen_best + 1e-9
    assert refine and refine[-1].best_dist <= refine[0].best_dist + 1e-9
    assert res["best_spec"] is not None and res["best_spec"]["unit"]

    # scalar objective: maximize peak force must beat pristine screen best
    res2 = run_inverse(builder, "max_peak", budget=30, seed=3)
    m_best = metrics_of(res2["best_run"])
    scr = [r for r in res2["records"] if r.stage == "screen"]
    print(f"[inv] max_peak: obj={res2['best_dist']:.3e} "
          f"peak_best={m_best['peak']:.3e} screen_min_obj={min(r.dist for r in scr):.3e}")
    assert res2["best_dist"] <= min(r.dist for r in scr) + 1e-9
    print("[inv] PASS")


if __name__ == "__main__":
    main()
