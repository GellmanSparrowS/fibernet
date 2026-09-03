"""FiberScope entry point.

Usage:
    python run.py [--lang zh|en] [--theme dark|light] [--smoke]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description="FiberScope")
    ap.add_argument("--lang", default="zh", choices=["zh", "en"])
    ap.add_argument("--theme", default="light", choices=["dark", "light"])
    ap.add_argument("--smoke", action="store_true",
                    help="offscreen smoke mode: quit after 1.5s")
    args = ap.parse_args()
    if args.smoke:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication
    from studio.main import MainWindow
    from fslab.structure import unit_display

    app = QApplication(sys.argv)
    app.setApplicationName("FiberScope")
    win = MainWindow(lang=args.lang, mode=args.theme)
    win.show()
    if args.smoke:
        sys.exit(_smoke_checks(app, win))
    sys.exit(app.exec())


def _emit(msg):
    data = (msg + "\n").encode("utf-8", "replace")
    try:
        os.write(1, data)
    except OSError:
        pass
    try:
        side = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
                            "smoke_result.txt")
        with open(side, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")
    except OSError:
        pass


def _smoke_checks(app, win):
    from fslab.structure import unit_display
    """Headless end-to-end check of all four tabs (also valid in the frozen
    exe): percolation, stretch+contact, exploration replay, inverse design."""
    import traceback

    try:
        # 0) structure studio drives the shared spec
        ts = win.tab_struct
        ts.unit_combo.setCurrentText(unit_display("reentrant"))
        ts.grid_x.setValue(3); ts.grid_y.setValue(3)
        ts.pts.setValue(2); ts.seed.setValue(7)
        ts.push_spec()
        app.processEvents()
        assert ts.factory is not None, "structure preview failed"

        # 1) percolation: fibernet pattern -> engine2 -> two-sided blue front
        tab = win.tab_perc
        tab.stretch.setValue(2.0)
        tab.quant.setValue(5)
        tab.run_sync()
        app.processEvents()
        tab._play_timer.stop()
        assert tab.run is not None, "percolation sim produced no result"
        assert tab.perc.perc_frame >= 0, "expected percolation in smoke config"

        # 2) stretch: chiral rotation-induced contacts
        ts.unit_combo.setCurrentText(unit_display("chiral"))
        ts.push_spec()
        app.processEvents()
        st = win.tab_stretch
        st.stretch.setValue(2.2)
        st.run_sync()
        app.processEvents()
        st._play_timer.stop()
        assert st.run is not None, "stretch sim produced no result"
        cmax = int(st.run.contact_counts.max())
        assert cmax > 0, "chiral should show contact events"

        # 3) replay: shipped exploration log must load (>=100 records)
        rp = win.tab_replay
        assert len(rp.recs) >= 100, f"exploration log too small: {len(rp.recs)}"

        # 4) inverse design: micro budget run must finish with a best unit
        dg = win.tab_design
        dg.budget.setValue(24)
        dg.run_sync()
        app.processEvents()
        assert dg.best_run is not None, "design produced no best run"

        msg = (f"[smoke] PASS perc_frame={tab.perc.perc_frame} "
               f"contacts={cmax} recs={len(rp.recs)} "
               f"design={dg.status.text()}")
        _emit(msg)
        return 0
    except Exception:
        _emit("[smoke] FAIL\n" + traceback.format_exc())
        return 1


if __name__ == "__main__":
    main()
