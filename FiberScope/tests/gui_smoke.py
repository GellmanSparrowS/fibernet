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


def _pixels(path):
    """(r, g, b) on a 2px stride, read with Qt only (PIL is not bundled)."""
    from PySide6.QtGui import QImage
    im = QImage(path)
    assert not im.isNull(), f"cannot read {path}"
    for y in range(0, im.height(), 2):
        for x in range(0, im.width(), 2):
            v = im.pixel(x, y)          # 0xAARRGGBB
            yield (v >> 16) & 255, (v >> 8) & 255, v & 255


def count_blue(path):
    return sum(1 for r, _g, b in _pixels(path) if b > 150 and b > r + 40)


def main():
    import tempfile
    from fslab import structure as structure_module
    custom_dir = tempfile.TemporaryDirectory(prefix='fs_gui_custom_')
    structure_module.CUSTOM_CELL_FILE = os.path.join(custom_dir.name, 'custom_units.json')
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

    # the big canvas must stay on screen while the spectrum editor is open:
    # watching the whole network deform is the point of editing one line
    h_gallery = ts.canvas.height()
    ts._toggle_edit()
    app.processEvents()
    assert ts.deck.currentIndex() == 1, "spectrum editor did not open"
    assert ts.canvas.height() >= 150, f"canvas collapsed to {ts.canvas.height()}"
    pm = ts.unit_preview.pixmap()
    assert pm is not None and not pm.isNull(), "single-cell preview empty"
    ts._toggle_edit()
    app.processEvents()
    assert ts.deck.currentIndex() == 0, "editor did not close"
    print(f"[gui_smoke] structure layout ok: canvas {h_gallery}px, "
          f"{ts.canvas.height()}px while editing")

    # F8b: a custom base unit must round-trip editor -> json -> gallery -> build
    import os as _os
    from fslab.structure import CUSTOM_CELL_FILE, delete_custom_cell
    from studio.cell_editor import CellEditorDialog, tiled_parity
    n_before = ts.gallery.count()
    dlg = CellEditorDialog("dark", None, None)   # top-level: out of tab tree
    dlg.canvas.set_spec(
        [(0, 0), (1, 0), (1, 1), (0, 1), (.5, 0), (1, .5), (.5, 1), (0, .5)],
        [(0, 4), (4, 1), (1, 5), (5, 2), (2, 6), (6, 3), (3, 7), (7, 0),
         (4, 5), (5, 6), (6, 7), (7, 4)])
    dlg.name_zh.setText("冒烟环")
    dlg.name_en.setText("smoke_ring")
    assert tiled_parity(dlg.canvas.nodes, dlg.canvas.edges)[0] == 0
    dlg._save()
    assert dlg.saved_key, "custom cell did not save"
    ts._rebuild_gallery()
    assert ts.gallery.count() == n_before == 12
    assert ts.custom_combo.findData(dlg.saved_key) >= 0
    combo = [ts.unit_combo.itemText(i)
             for i in range(ts.unit_combo.count())]
    assert unit_display(dlg.saved_key) in combo, "custom unit missing in combo"
    ts.unit_combo.setCurrentText(unit_display(dlg.saved_key))
    ts.push_spec()
    assert ts.factory.unit == dlg.saved_key, "custom unit did not build"
    ts.unit_combo.setCurrentText(unit_display("square"))
    ts.push_spec()
    delete_custom_cell(dlg.saved_key)
    ts._rebuild_gallery()
    assert ts.gallery.count() == n_before, "gallery did not shrink"
    if _os.path.exists(CUSTOM_CELL_FILE):
        _os.remove(CUSTOM_CELL_FILE)
    dlg.deleteLater()   # transient dialog must not leak into the i18n scan
    app.processEvents()
    dlg = None
    print("[gui_smoke] custom unit round-trip ok")

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
    st.chk_contact.setChecked(True)
    st.chk_weld.setChecked(False)
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
    nred = sum(1 for r, g, b in _pixels(ps)
               if r > 180 and g < 110 and b < 110)
    print(f"[gui_smoke] stretch: contacts_max={st.run.contact_counts.max()} "
          f"red_pixels={nred}")
    assert nred > 5, "contact markers not visible"

    # playback: loop wraps to the first frame, non-loop stops at the last
    st._on_speed(2)
    assert st._play_timer.interval() == round(1000.0 / 24), "24 fps interval"
    st.loop_btn.setChecked(True)
    st.slider.setValue(st.slider.maximum() - 1)
    st._play_step(); st._play_step()
    assert st.slider.value() == 0, "loop did not wrap to frame 0"
    st.loop_btn.setChecked(False)
    st.slider.setValue(st.slider.maximum() - 1)
    st._play_step(); st._play_step()
    assert st.slider.value() == st.slider.maximum(), "non-loop overran"
    assert not st._play_timer.isActive(), "non-loop kept playing"
    st.loop_btn.setChecked(True)
    print(f"[gui_smoke] playback ok: 24fps default, loop wrap verified")

    # exports: per-frame CSV + plot/canvas PNG, dialogs bypassed via silent
    from PySide6.QtGui import QImage

    def _ink(path, step=3):
        im = QImage(path)
        assert not im.isNull(), f"unreadable png {path}"
        bg = im.pixel(im.width() - 3, im.height() - 3)
        return sum(1 for y in range(0, im.height(), step)
                   for x in range(0, im.width(), step)
                   if im.pixel(x, y) != bg), im

    pcsv = st._export_csv(silent=os.path.join(TMP, "smoke_curve.csv"))
    with open(pcsv, encoding="utf-8-sig") as fh:
        rows = fh.read().strip().splitlines()
    assert len(rows) == st.run.n_frames + 1, f"{len(rows)} csv rows"
    assert rows[0].split(",")[1] == "strain", rows[0]
    pf = st._export_force_png(silent=os.path.join(TMP, "smoke_force.png"))
    assert pf, f"force png export failed: {st.status.text()}"
    n1, im1 = _ink(pf)
    assert im1.width() >= 1200, f"force png only {im1.width()}px wide"
    assert n1 > 500, f"force curve png looks blank ({n1} ink px)"
    pc = st._export_frame_png(silent=os.path.join(TMP, "smoke_canvas.png"))
    assert pc, f"canvas png export failed: {st.status.text()}"
    n2, _ = _ink(pc)
    assert n2 > 500, f"canvas png looks blank ({n2} ink px)"
    print(f"[gui_smoke] export ok: csv={len(rows) - 1} rows "
          f"force_png={im1.width()}x{im1.height()} ink={n1}/{n2}")
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

    # exports on the design / features / structure tabs
    pinv = dg._export_inv_csv(silent=os.path.join(TMP, "smoke_inv.csv"))
    assert pinv, f"inverse csv export failed: {dg.status.text()}"
    with open(pinv, encoding="utf-8-sig") as fh:
        inv_rows = fh.read().strip().splitlines()
    assert len(inv_rows) >= 25, f"inverse log has {len(inv_rows)} rows"
    n, _ = _ink(dg._export_best_png(
        silent=os.path.join(TMP, "smoke_best.png")))
    assert n > 200, f"best-vs-target png blank ({n})"

    ft = win.tab_features
    ft.set_factory(ts.collect_factory())
    ft.refresh()
    app.processEvents()
    assert ft._feats is not None, ft.status.text()
    pfeat = ft._export_feat_csv(silent=os.path.join(TMP, "smoke_feat.csv"))
    assert pfeat, f"feature csv export failed: {ft.status.text()}"
    with open(pfeat, encoding="utf-8-sig") as fh:
        feat_rows = fh.read().strip().splitlines()
    assert len(feat_rows) >= 16, f"feature table has {len(feat_rows)} rows"
    n, _ = _ink(ft._export_fp_png(silent=os.path.join(TMP, "smoke_fp.png")))
    assert n > 100, f"fingerprint png blank ({n})"

    n, _ = _ink(ts._do_export_png(
        silent=os.path.join(TMP, "smoke_struct.png")))
    assert n > 500, f"structure canvas png blank ({n})"
    print(f"[gui_smoke] tab exports ok: inv={len(inv_rows) - 1} rows "
          f"feat={len(feat_rows) - 1} rows")

    # inverse cancel: stop_cb aborts between evaluations, best so far survives
    from studio.design_tab import DesignWorker
    from studio.i18n import tr
    from fslab.inverse import run_inverse
    w = DesignWorker(dg.target_combo.currentText(), budget=40, seed=3, pts=3)
    n_ev = {"n": 0}
    res_stop = run_inverse(
        w._builder, w.target, budget=40, seed=3, pts=3,
        callback=lambda rec, run: n_ev.__setitem__("n", n_ev["n"] + 1),
        stop_cb=lambda: n_ev["n"] >= 2)
    assert res_stop["stopped"] is True, "stop_cb did not abort the search"
    assert n_ev["n"] <= 3, f"kept evaluating after stop ({n_ev['n']})"
    assert res_stop["best_spec"] is not None, "stopped run lost its best"
    assert not dg.stop_btn.isEnabled(), "stop enabled while idle"
    dg._on_done(res_stop)
    app.processEvents()
    assert tr("stopped_note") in dg.status.text(), dg.status.text()
    assert dg.run_btn.isEnabled(), "run button stuck disabled after stop"
    assert not dg.stop_btn.isEnabled(), "stop still enabled after finish"
    print(f"[gui_smoke] inverse cancel ok: evals={n_ev['n']} "
          f"best={res_stop['best_label']} obj={res_stop['best_dist']:.3f}")
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
    er = dock.registry["export_result"](
        kind="curve_csv", path=os.path.join(TMP, "ai_curve.csv"))
    assert er.get("ok") and os.path.exists(er["path"]), er
    er = dock.registry["export_result"](
        kind="canvas_png", path=os.path.join(TMP, "ai_canvas.png"))
    assert er.get("ok"), er
    bad = dock.registry["export_result"](kind="nope")
    assert not bad.get("ok"), "unknown kind must fail"
    print(f"[gui_smoke] ai tools ok: perc={pr['perc_frame']} "
          f"export={os.path.basename(out['path'])} result_export=ok")

    # i18n: English mode must leave no CJK in static widget text
    import re as _re
    from PySide6.QtWidgets import QComboBox, QWidget
    CJK = _re.compile(r"[\u4e00-\u9fff]")
    DYNAMIC = {"subtitle", "hint"}     # status lines keep runtime text

    def _cjk(w):
        bad = []
        for c in [w] + w.findChildren(QWidget):
            if c.objectName() in DYNAMIC:
                continue
            for getter in ("text", "title"):
                f = getattr(c, getter, None)
                if not callable(f):
                    continue
                try:
                    s = f()
                except TypeError:
                    continue
                if isinstance(s, str) and CJK.search(s):
                    bad.append(f"{c.metaObject().className()}: {s}")
            if isinstance(c, QComboBox):
                bad += [c.itemText(i) for i in range(c.count())
                        if CJK.search(c.itemText(i))]
        return bad

    win._on_lang(1)
    app.processEvents()
    for tab in (win.tab_struct, win.tab_sim, win.tab_features, win.tab_ml,
                win.tab_design, win.tab_surface, win.tab_replay):
        bad = _cjk(tab)
        assert not bad, f"CJK left in en mode: {bad[:6]}"
    win._on_lang(0)
    app.processEvents()
    print("[gui_smoke] i18n ok: en mode free of CJK on 7 tabs")
    print("[gui_smoke] PASS")


if __name__ == "__main__":
    main()
