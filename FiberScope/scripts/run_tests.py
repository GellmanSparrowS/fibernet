"""Run every self-test suite and report a compact summary.

Usage: python scripts/run_tests.py [--only selftest] [--gui]
Exit code is non-zero when any suite fails.  QT_QPA_PLATFORM=offscreen is
forced so the GUI suite never needs a display.
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITES = ["selftest_bridge", "selftest_graph_contract", "selftest", "selftest_engine2", "selftest_inverse", "selftest_finals", "selftest_learning", "selftest_search", "selftest_surface",
          "gui_default_language", "gui_smoke", "gui_finals", "gui_upgrade", "gui_refinement24", "selftest_manufacturing", "gui_manufacturing", "selftest_topnet26", "selftest_obj26", "gui_workflow26", "selftest_geometry27", "gui_geometry27", "selftest_upgrade28", "gui_upgrade28", "gui_precision29", "gui_final30"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma separated suite names")
    ap.add_argument("--tail", type=int, default=4)
    args = ap.parse_args()
    want = args.only.split(",") if args.only else SUITES
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen",
               PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    failed = []
    for name in want:
        script = os.path.join("tests", name + ".py")
        t0 = time.time()
        r = subprocess.run([sys.executable, script], cwd=ROOT, env=env,
                           capture_output=True, text=True, encoding="utf-8")
        dt = time.time() - t0
        lines = [l for l in (r.stdout or "").strip().splitlines() if l]
        print("== %-20s exit=%d  %.1fs" % (name, r.returncode, dt))
        for l in lines[-args.tail:]:
            print("   ", l)
        if r.returncode != 0:
            failed.append(name)
            err = (r.stderr or "").strip().splitlines()
            for l in err[-12:]:
                print("   !", l)
    print("SUMMARY: %d/%d passed" % (len(want) - len(failed), len(want)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
