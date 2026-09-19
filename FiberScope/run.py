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
    ap.add_argument('--solid-worker', default=None)
    args = ap.parse_args()
    if args.solid_worker:
        from fslab.solid_process import worker
        return worker(args.solid_worker)
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
        if getattr(sys, "frozen", False):
            base = os.path.dirname(os.path.abspath(sys.executable))
        else:
            base = os.path.dirname(os.path.abspath(sys.argv[0]))
        side = os.path.join(base, "smoke_result.txt")
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
        if ts.factory is None:
            # push_spec swallows the cause into the status label; rebuild
            # headless so the real traceback lands in smoke_result.txt
            from fslab import StructureFactory
            StructureFactory(unit="square", grid_x=2, grid_y=2,
                             n_pts_per_side=2, seed=7).build()
            raise AssertionError("structure preview failed: %s"
                                 % ts.status.text())

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
        st.chk_contact.setChecked(True)
        st.chk_weld.setChecked(False)
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

        # 5) exports: CSV + captioned PNG must work headless (frozen too)
        import shutil
        import tempfile
        d = tempfile.mkdtemp(prefix="fssmoke")
        try:
            csv_p = st._export_csv(silent=os.path.join(d, "curve.csv"))
            assert csv_p and os.path.getsize(csv_p) > 500, "curve csv empty"
            png_p = st._export_force_png(silent=os.path.join(d, "force.png"))
            assert png_p and os.path.getsize(png_p) > 8000, "force png small"
            net_p = ts._do_export_png(silent=os.path.join(d, "net.png"))
            from PySide6.QtGui import QImage
            net_image = QImage(net_p or '')
            assert not net_image.isNull() and net_image.width() >= 200, 'canvas export invalid'
            samples = {net_image.pixel(x, y)
                       for x in range(0, net_image.width(), 3)
                       for y in range(0, net_image.height(), 3)}
            assert len(samples) > 4, 'canvas export is blank'
            png_kb = os.path.getsize(png_p) // 1024
        finally:
            shutil.rmtree(d, ignore_errors=True)

        msg = (f"[smoke] PASS perc_frame={tab.perc.perc_frame} "
               f"contacts={cmax} recs={len(rp.recs)} "
               f"export_png={png_kb}KB "
               f"design={dg.status.text()}")
        # Finals features must also work from a self-contained frozen bundle.
        import numpy as np
        from fslab.contact import ContactConfig
        from fslab import structure as structures
        from studio.cell_editor import CellEditorDialog
        from studio.ai_assistant import _ToolRequest
        import json
        pair = ContactConfig(width=.3).compute(
            np.array([[0., 0.], [2., 0.], [0., .15], [2., .15]]),
            np.array([[0, 1], [2, 3]]))
        assert pair['contact_pair_count'] == 1
        old_path, old_cells = structures.CUSTOM_CELL_FILE, dict(structures.CUSTOM_CELLS)
        with tempfile.TemporaryDirectory(prefix='fsfinals_') as room:
            try:
                structures.CUSTOM_CELL_FILE = os.path.join(room, 'cells.json')
                dlg = CellEditorDialog(mode=win.mode)
                dlg.name_zh.setText('Smoke cell')
                dlg.name_en.setText('smoke_cell')
                dlg.rule.setCurrentIndex(1)
                dlg._save()
                assert dlg.saved_factory is not None
                structures.load_custom_cells()
                assert structures.CUSTOM_CELLS[dlg.saved_key]['settings']['expansion_rule'] == 'rotate'
                assert dlg.saved_factory.build().num_nodes > 0
                dlg.deleteLater()
            finally:
                structures.CUSTOM_CELL_FILE = old_path
                structures.CUSTOM_CELLS.clear()
                structures.CUSTOM_CELLS.update(old_cells)
                structures._CELLS_REGISTERED.clear()
        request = _ToolRequest('set_expansion', dict(rule='mirror', grid_x=2, grid_y=2))
        win.ai_panel._run_tool(request)
        assert json.loads(request.result)['ok']
        win.ai_panel.registry['set_tab'](tab='features')
        assert win.tabs.currentWidget() is win.tab_features
        from fslab.learning import Regressor, MODEL_SPECS
        import time
        from fslab.mlmodel import gen_dataset
        X, Y = gen_dataset('square', n=12)
        ml = win.tab_ml
        ml.X, ml.Y, ml.label_mode = X, Y, 'physics'
        win.tabs.setCurrentWidget(ml)
        for key in MODEL_SPECS:
            ml.algorithm = key
            ml.parameters = {k: v[0] for k, v in MODEL_SPECS[key][2].items()}
            ml.model = None
            ml.start_train()
            deadline = time.monotonic()+60
            while ml.train_worker is not None:
                app.processEvents()
                time.sleep(.005)
                assert time.monotonic() < deadline, 'ML worker timed out'
            assert ml.model is not None, ml.status.text()
            for target in range(3):
                ml.target_combo.setCurrentIndex(target)
                ml.scatter_plot.setRange(xRange=(-2, -1), yRange=(-2, -1))
                ml._update_scatter()
                x, y = ml.scatter.getData()
                assert len(x) == 2 and np.isfinite(y).all()
                bounds = ml.scatter_plot.viewRange()
                assert min(x) >= bounds[0][0] and max(x) <= bounds[0][1]
                assert min(y) >= bounds[1][0] and max(y) <= bounds[1][1]
        from fslab.model_inverse import run_model_inverse
        from fslab.structure import StructureFactory
        predicted = run_model_inverse(StructureFactory(unit='square', grid_x=1, grid_y=1),
                                      ml.model, 'max_peak', budget=3)
        assert predicted['best_preview'] is not None and predicted['best_run'] is None
        win.tab_design._on_done(predicted)
        assert win.tab_design.canvas.static is not None
        surface = win.tab_surface
        surface.obj_combo.setCurrentIndex(surface._obj_files.index(
            next(p for p in surface._obj_files if p.endswith('6_vans_500.obj'))))
        surface.unit_combo.setCurrentIndex(surface.unit_combo.findData('octagon'))
        surface.chk_follow.setChecked(False)
        surface._recompute()
        assert len(surface.network[1]) > 0 and not surface.canvas.show_mesh
        from fslab.surface_mapping import load_obj, coarsen_quads
        v, f = load_obj(surface._obj_files[-1])
        v, f = coarsen_quads(v, f, 300)
        assert len(f) <= 300 and np.isfinite(v).all()
        if getattr(sys, 'frozen', False):
            from pathlib import Path
            assert (Path(sys._MEIPASS) / 'training_runtime' / 'scripts' / 'rl_worker.py').is_file()
        from studio.recording_command import recording_options
        options=recording_options("请执行决赛录像流程：生成方形网络，生成300个样本，以J型曲线为目标进行200次预算")
        assert options["demo_unit"]=="square" and options["iterations"]==200
        from fslab.manufacturing import compile_planar
        from fslab.print_export import build_solid, export_solid
        import tempfile
        from pathlib import Path
        manufacturing = compile_planar(StructureFactory(unit='hexagon', grid_x=2, grid_y=2, n_pts_per_side=2))
        from fslab.solid_process import build_isolated
        from fslab.print_export import PrintSettings
        solid = build_isolated(manufacturing, PrintSettings())
        assert manufacturing.health['odd'] == 0 and np.allclose(np.ptp(solid.vertices,axis=0), [100,100,2])
        with tempfile.TemporaryDirectory(prefix='fs_frozen_solid_') as directory:
            export_solid(Path(directory)/'test.3mf',solid)
            export_solid(Path(directory)/'test.stl',solid)
            gif_path = Path(directory)/'simulation.gif'
            tab._export_gif(str(gif_path))
            until=time.monotonic()+90
            while tab._gif.active:
                app.processEvents()
                time.sleep(.005)
                assert time.monotonic()<until, 'GIF export timed out'
            from PIL import Image
            with Image.open(gif_path) as animation:
                assert animation.width==1600 and animation.n_frames==tab.run.n_frames
        msg += ' finals=PASS upgrade25=PASS learned_inverse=PASS physical_ml_gui=6/6 mesh_reduce=PASS manufacturing_solid=PASS isolated_worker=PASS gif1600=PASS'
        _emit(msg)
        return 0
    except Exception:
        _emit("[smoke] FAIL\n" + traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
