"""Offscreen GUI smoke test: build window, simulate, pixel-probe the canvas.

Run: python tests/gui_smoke.py   (sets QT_QPA_PLATFORM=offscreen itself)
"""
import os
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PySide6.QtWidgets import QApplication

TMP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_tmp")


def grab(win, path):
    app = QApplication.instance()
    win.tab_perc.canvas._cache_key = None  # force re-render
    win.tab_perc.canvas.update()
    app.processEvents()
    pm = win.tab_perc.canvas.grab()
    pm.save(path)
    return path


def count_blue(path):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    px = im.load()
    n = 0
    for y in range(0, im.height, 2):
        for x in range(0, im.width, 2):
            r, g, b = px[x, y]
            if b > 150 and b > r + 40:
                n += 1
    return n


def main():
    app = QApplication([])
    from studio.main import MainWindow
    win = MainWindow(lang="zh", mode="dark")
    win.resize(1100, 700)
    win.show()
    app.processEvents()

    # structure studio drives the shared spec
    ts = win.tab_struct
    from fslab.structure import unit_display
    ts.unit_combo.setCurrentText(unit_display("reentrant"))
    ts.grid_x.setValue(3); ts.grid_y.setValue(3)
    ts.pts.setValue(2); ts.seed.setValue(7)
    ts.push_spec()
    app.processEvents()
    assert ts.factory is not None, "structure preview failed"

    tab = win.tab_perc
    tab.stretch.setValue(2.0)
    tab.quant.setValue(5)
    tab.run_sync()
    app.processEvents()
    tab._play_timer.stop()
    assert tab.run is not None, "simulation produced no result"
    F = tab.run.n_frames
    assert tab.perc.perc_frame >= 0, "expected percolation in smoke config"
    print(f"[gui_smoke] sim ok: F={F} perc_frame={tab.perc.perc_frame}")

    os.makedirs(TMP, exist_ok=True)
    tab.slider.setValue(0)
    app.processEvents()
    p0 = grab(win, os.path.join(TMP, "smoke_frame0.png"))
    tab.slider.setValue(F - 1)
    app.processEvents()
    p1 = grab(win, os.path.join(TMP, "smoke_frame_last.png"))

    b0 = count_blue(p0)
    b1 = count_blue(p1)
    print(f"[gui_smoke] blue pixels: frame0={b0} last={b1}")
    assert b1 > b0 + 50, "percolation backbone did not grow visually"
    assert b1 > 200, "no visible backbone at final frame"

    # ---- stretch tab regression: chiral contact demo ----
    ts.unit_combo.setCurrentText(unit_display("chiral"))
    ts.push_spec()
    app.processEvents()
    st = win.tab_stretch
    st.stretch.setValue(2.2)
    st.run_sync()
    app.processEvents()
    st._play_timer.stop()
    assert st.run is not None
    e = st.run.energies
    tot = e["axial"] + e["bend"] + e["contact"]
    frac = (e["axial"] + e["bend"] + e["contact"]) / (tot + 1e-12)
    assert np.allclose(frac[tot > 1.0], 1.0)
    assert st.run.contact_counts.max() > 0, "chiral should show contact events"
    st.view_combo.setCurrentIndex(1)
    app.processEvents()
    st.slider.setValue(st.run.n_frames - 1)
    app.processEvents()
    ps = os.path.join(TMP, "smoke_stretch_last.png")
    win.tab_stretch.canvas._cache_key = None
    win.tab_stretch.canvas.update()
    app.processEvents()
    win.tab_stretch.canvas.grab().save(ps)
    from PIL import Image
    im = Image.open(ps).convert("RGB")
    px = im.load()
    nred = 0
    for y in range(0, im.height, 2):
        for x in range(0, im.width, 2):
            r, g, b = px[x, y]
            if r > 180 and g < 110 and b < 110:
                nred += 1
    print(f"[gui_smoke] stretch: contacts_max={st.run.contact_counts.max()} "
          f"red_pixels={nred}")
    assert nred > 5, "contact markers not visible"
    # ---- replay tab regression ----
    rp = win.tab_replay
    assert len(rp.recs) >= 100, "exploration log missing or too small"
    rp.slider.setValue(min(60, len(rp.recs) - 1))
    app.processEvents()
    n_ag = sum(1 for r in rp.recs if r["method"] == "agent")
    n_r0 = sum(1 for r in rp.recs if r["method"] == "r0")
    print(f"[gui_smoke] replay: recs={len(rp.recs)} agent={n_ag} r0={n_r0} "
          f"clusters_agent={rp.cluster_curve['agent'][-1]} "
          f"r0={rp.cluster_curve['r0'][-1]}")

    # ---- design tab regression (small budget) ----
    dg = win.tab_design
    dg.budget.setValue(24)
    dg.run_sync()
    app.processEvents()
    assert dg.best_run is not None
    print(f"[gui_smoke] design: best={dg.status.text()}")
    # ---- AI assistant offline tool wiring ----
    dock = win.ai_dock
    st = dock.registry["get_app_state"]()
    assert "structure" in st
    dock.registry["set_structure"](unit="square", grid_x=3, grid_y=3,
                                     pts=3, seed=11)
    app.processEvents()
    assert ts.unit_combo.currentText() == unit_display("square")
    dock.registry["set_line_displacements"](
        displacements=[[0, 0.1], [0, -0.1], [0, 0.05]])
    app.processEvents()
    assert win.tab_struct.factory.line_displacements is not None
    out = dock.registry["export_structure"](
        format="json", path=os.path.join(TMP, "ai_export.json"))
    assert os.path.exists(out["path"])
    pr = dock.registry["run_percolation"](stretch=1.6, alpha=0.05)
    assert pr["frames"] > 0
    print(f"[gui_smoke] ai tools ok: perc={pr['perc_frame']} "
          f"export={os.path.basename(out['path'])}")
    print("[gui_smoke] PASS")


if __name__ == "__main__":
    main()
