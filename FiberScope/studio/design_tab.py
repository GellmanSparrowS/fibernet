"""Inverse design tab: live two-stage optimizer (topology screen + CEM over
the reference-line point values) streaming convergence, best curve vs
target, stretched best structure plus its pre-stretch inset."""
from dataclasses import replace
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (QComboBox, QCheckBox, QDoubleSpinBox, QFrame, QGridLayout, QGroupBox,
                               QHBoxLayout, QLabel, QListWidget, QMenu,
                               QPushButton, QSizePolicy, QSpinBox, QSplitter,
                               QVBoxLayout, QWidget)

from fslab import StructureFactory
from fslab.exporter import export_inverse_csv
from fslab.inverse import (run_inverse, target_curve, curve_of, metrics_of,
                           TARGETS, SCALARS, PTS)
from fslab.structure import UNIT_PRESETS, all_unit_keys, unit_display
from .exports import (add_caption, ask_save, save_plot_png, save_widget_png,
                      unique_stem)
from .i18n import tr, get_lang
from fslab.search_algorithms import SEARCH_SPECS
from .network_canvas import NetworkCanvas
from .count_input import CountInput
from fslab.model_inverse import PredictedStructure, run_model_inverse
from .theme import apply_plot_theme, colors


class DesignWorker(QThread):
    progress = Signal(object, object)   # record, run_or_None
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, target, budget, seed, fixed_unit=None, pts=None,
                 parent=None, factory=None, algorithm='cem', parameters=None, amplitude=.6, resume=False, model=None):
        super().__init__(parent)
        self.target = target
        self.budget = budget
        self.seed = seed
        self.fixed_unit = fixed_unit
        self.pts = int(pts) if pts else PTS
        self._stop = False
        self.factory = replace(factory, n_pts_per_side=self.pts) if factory else StructureFactory(unit=fixed_unit or 'square', n_pts_per_side=self.pts)
        self.algorithm, self.parameters = algorithm, parameters or {}
        self.amplitude, self.resume = amplitude, resume
        self.model = model

    def request_stop(self):
        self._stop = True

    def _builder(self, unit, pert, ld):
        return replace(self.factory, unit=unit,
                                n_pts_per_side=self.pts, seed=self.seed,
                                perturbation=pert,
                                line_displacements=ld).build()

    def run(self):
        try:
            if self.algorithm != 'cem':
                from fslab.rl_process import run_external
                res = run_external(self.factory, self.target, self.algorithm, self.parameters,
                                   self.budget, self.seed,
                                   callback=lambda rec, run: self.progress.emit(rec, run),
                                   stop_cb=lambda: self._stop, amplitude=self.amplitude, resume=self.resume, model=self.model)
                self.finished_ok.emit(res)
                return
            if self.model is not None:
                res = run_model_inverse(self.factory, self.model, self.target, self.budget,
                    self.seed, self.amplitude, lambda rec, run: self.progress.emit(rec, run),
                    lambda: self._stop)
                self.finished_ok.emit(res)
                return
            res = run_inverse(self._builder, self.target, budget=self.budget,
                              seed=self.seed, fixed_unit=self.fixed_unit,
                              pts=self.pts, initial_spec=self.factory, amplitude=self.amplitude,
                              stop_cb=lambda: self._stop,
                              callback=lambda rec, run: self.progress.emit(
                                  rec, run))
            self.finished_ok.emit(res)
        except Exception as e:
            self.failed.emit(f"{e.__class__.__name__}: {e}")


class DesignTab(QWidget):
    apply_structure = Signal(object)

    def __init__(self, mode="dark", parent=None):
        super().__init__(parent)
        self.mode = mode
        self.worker = None
        self.model_provider = None
        self.selected_model = None
        self.best_run = None
        self.fixed_unit = "square"
        self.fixed_pts = 5
        self.factory = StructureFactory(unit=self.fixed_unit)
        self.algorithm, self.parameters = 'cem', {}
        self._build()
        self.retranslate()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)
        head = QHBoxLayout()
        self.replay_toggle = QPushButton()
        self.replay_toggle.setCheckable(True)
        self.replay_toggle.setChecked(False)
        self.replay_toggle.clicked.connect(self._toggle_replay)
        head.addStretch(1)
        self.algorithm_btn = QPushButton()
        self.algorithm_btn.clicked.connect(self._configure_search)
        self.model_btn = QCheckBox()
        self.model_btn.setCheckable(True)
        self.model_btn.clicked.connect(self._choose_model)
        head.addWidget(self.algorithm_btn)
        head.addWidget(self.replay_toggle)
        outer.addLayout(head)
        self.vsplit = QSplitter(Qt.Vertical)
        splitter = QSplitter(Qt.Horizontal)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.gb_cfg = QGroupBox()
        g = QGridLayout(self.gb_cfg)
        self.lbl_target = QLabel()
        self.target_combo = QComboBox()
        self.target_combo.addItems(list(TARGETS) + list(SCALARS))
        self.lbl_budget = QLabel()
        self.budget = CountInput(100)
        self.target_combo.setCurrentText("C")
        self.lbl_seed = QLabel()
        self.seed = QSpinBox(); self.seed.setRange(0, 9999); self.seed.setValue(7)
        g.addWidget(self.lbl_target, 0, 0); g.addWidget(self.target_combo, 0, 1)
        g.addWidget(self.lbl_budget, 1, 0); g.addWidget(self.budget, 1, 1)
        g.addWidget(self.lbl_seed, 2, 0); g.addWidget(self.seed, 2, 1)
        self.lbl_unit = QLabel()
        self.unit_combo = QComboBox()
        for key in all_unit_keys():
            self.unit_combo.addItem(unit_display(key, get_lang()), key)
        self.unit_combo.setCurrentIndex(self.unit_combo.findData('square'))
        self.unit_combo.currentIndexChanged.connect(self._select_unit)
        g.addWidget(self.lbl_unit, 3, 0); g.addWidget(self.unit_combo, 3, 1)
        self.lbl_amplitude = QLabel()
        self.amplitude = QDoubleSpinBox()
        self.amplitude.setRange(.05, 1.)
        self.amplitude.setSingleStep(.05)
        self.amplitude.setValue(.6)
        g.addWidget(self.lbl_amplitude, 4, 0); g.addWidget(self.amplitude, 4, 1)
        self.resume = QCheckBox()
        self.resume.setEnabled(False)
        self.resume.hide()
        self.follow_type = QCheckBox()
        self.follow_type.setChecked(True)
        self.follow_type.toggled.connect(self._sync_type)
        g.addWidget(self.model_btn, 5, 0)
        g.addWidget(self.follow_type, 5, 1)
        self.unit_combo.setEnabled(False)
        lv.addWidget(self.gb_cfg)
        self.run_btn = QPushButton()
        self.run_btn.setProperty("primary", True)
        self.run_btn.clicked.connect(self.start_run)
        self.stop_btn = QPushButton()
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._request_stop)
        self.apply_btn = QPushButton()
        self.apply_btn.clicked.connect(self._apply_best)
        self.status = QLabel(); self.status.setObjectName("subtitle")
        self.status.setWordWrap(True)
        brow = QHBoxLayout()
        brow.addWidget(self.run_btn, 1)
        brow.addWidget(self.stop_btn)
        lv.addLayout(brow); lv.addWidget(self.apply_btn)
        self.export_btn = QPushButton()
        self.export_menu = QMenu(self.export_btn)
        self.act_inv_csv = self.export_menu.addAction("")
        self.act_best_png = self.export_menu.addAction("")
        self.act_conv_png = self.export_menu.addAction("")
        self.export_menu.addSeparator()
        self.act_struct_png = self.export_menu.addAction("")
        self.export_btn.setMenu(self.export_menu)
        self.act_inv_csv.triggered.connect(self._export_inv_csv)
        self.act_best_png.triggered.connect(self._export_best_png)
        self.act_conv_png.triggered.connect(self._export_conv_png)
        self.act_struct_png.triggered.connect(self._export_struct_png)
        lv.addWidget(self.export_btn)
        lv.addWidget(self.status)
        self.gb_log = QGroupBox()
        lg = QVBoxLayout(self.gb_log)
        self.log = QListWidget()
        lg.addWidget(self.log)
        lv.addWidget(self.gb_log, 1)
        left.setMinimumWidth(240)
        left.setMaximumWidth(320)
        splitter.addWidget(left)

        center = QWidget()
        cv = QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(6)
        self.lbl_init = QLabel()
        self.lbl_init.setObjectName("hint")
        cv.addWidget(self.lbl_init)
        self.canvas0 = NetworkCanvas(mode=self.mode)
        self.canvas0.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cv.addWidget(self.canvas0, 2)
        self.lbl_def = QLabel()
        self.lbl_def.setObjectName("hint")
        cv.addWidget(self.lbl_def)
        self.canvas = NetworkCanvas(mode=self.mode)
        self.canvas.set_color_mode("strain")
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cv.addWidget(self.canvas, 3)
        splitter.addWidget(center)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.gb_curve = QGroupBox()
        cvv = QVBoxLayout(self.gb_curve)
        self.plot_c = pg.PlotWidget(); self.plot_c.setMenuEnabled(False)
        self.curve_target = self.plot_c.plot(
            pen=pg.mkPen("#888888", width=2, style=Qt.DashLine))
        self.curve_best = self.plot_c.plot(
            pen=pg.mkPen(colors(self.mode)["accent"], width=2))
        cvv.addWidget(self.plot_c)
        self.lbl_target_info = QLabel()
        self.lbl_target_info.setObjectName('hint')
        self.lbl_target_info.setWordWrap(True)
        cvv.addWidget(self.lbl_target_info)
        rv.addWidget(self.gb_curve)
        self.gb_conv = QGroupBox()
        kv = QVBoxLayout(self.gb_conv)
        self.plot_k = pg.PlotWidget(); self.plot_k.setMenuEnabled(False)
        self.plot_k.setMinimumHeight(130)
        self.curve_conv = self.plot_k.plot(
            pen=pg.mkPen(colors(self.mode)["accent2"], width=2))
        kv.addWidget(self.plot_k)
        rv.addWidget(self.gb_conv)
        right.setMinimumWidth(230)
        right.setMaximumWidth(320)
        splitter.addWidget(right)
        splitter.setSizes([280, 600, 260])
        self.vsplit.addWidget(splitter)
        outer.addWidget(self.vsplit, 1)

    def embed_replay(self, widget):
        """Exploration replay lives as a collapsible bottom panel."""
        self._replay_widget = widget
        self.vsplit.addWidget(widget)
        widget.hide()
        self.vsplit.setSizes([660, 160])
        self.vsplit.setStretchFactor(0, 1)
        self.vsplit.setStretchFactor(1, 0)
        self._toggle_replay()

    def _toggle_replay(self):
        if getattr(self, "_replay_widget", None) is None:
            return
        self._replay_widget.setVisible(self.replay_toggle.isChecked())
        self.retranslate()

    # ---------------- run ----------------
    def _configure_search(self):
        from .algorithm_dialog import AlgorithmDialog
        dialog = AlgorithmDialog(SEARCH_SPECS, self.algorithm, {self.algorithm: self.parameters}, parent=self)
        if dialog.exec():
            self.algorithm, self.parameters = dialog.selection()
            self.resume.setEnabled(self.algorithm != 'cem')
            self.algorithm_btn.setText(SEARCH_SPECS[self.algorithm][0 if get_lang() == 'zh' else 1] + ' · …')

    def _choose_model(self, checked):
        if not checked:
            self.selected_model = None
            self._sync_type()
            self.target_combo.setCurrentText("C")
            self._preview_target()
            return
        from PySide6.QtWidgets import QInputDialog
        supplied = self.model_provider() if self.model_provider else None
        if not supplied or supplied[0] is None:
            self.model_btn.setChecked(False)
            self.status.setText("请先在机器学习中训练模型" if get_lang() == "zh" else "Train a model in Machine Learning first")
            return
        target, ok = QInputDialog.getItem(self, "模型预测" if get_lang() == "zh" else "Model prediction",
            "选择标量目标（模型不预测完整曲线）" if get_lang() == "zh" else "Scalar target (no full curve prediction)",
            list(SCALARS), 0, False)
        if not ok:
            self.model_btn.setChecked(False)
            return
        import copy
        self.selected_model = copy.deepcopy(supplied[0])
        self.unit_combo.setCurrentIndex(self.unit_combo.findData(supplied[1]))
        self.model_unit = supplied[1]
        self._sync_type()
        self.target_combo.setCurrentText(target)
        self._preview_target()

    def start_run(self):
        if self.worker is not None and self.worker.isRunning():
            return
        if self.selected_model is not None and self.model_provider:
            supplied = self.model_provider()
            if supplied and supplied[0] is not None:
                import copy
                self.selected_model = copy.deepcopy(supplied[0])
                self.model_unit = supplied[1]
                self._sync_type()
        if self.selected_model is not None and self.target_combo.currentText() not in SCALARS:
            self.status.setText("模型仅支持标量目标 / Model supports scalar targets only")
            return
        self.model_btn.setEnabled(False)
        self.best_run = None
        self.canvas.clear_static()
        self.canvas0.clear_static()
        self.canvas.set_color_mode("plain" if self.selected_model is not None else "strain")
        self.lbl_def.setText(("最优候选 · 模型预测" if get_lang() == "zh" else "Best candidate · model prediction") if self.selected_model is not None else ("当前候选 · 拉伸形态" if get_lang() == "zh" else "Current candidate · stretched"))
        self.curve_best.setData([], [])
        self.run_btn.setEnabled(False)
        self.algorithm_btn.setEnabled(False)
        self.gb_cfg.setEnabled(False)
        self.apply_btn.setEnabled(False)
        self._run_factory = replace(self.factory, unit=self.unit_combo.currentData(), seed=self.seed.value())
        self.fixed_unit = self._run_factory.unit
        self._result = None
        self.stop_btn.setEnabled(True)
        self.status.setText(tr("running"))
        self.log.clear()
        self._evals, self._bests = [], []
        self._preview_target()
        self.worker = DesignWorker(self.target_combo.currentText(),
                                   self.budget.value(), self.seed.value(),
                                   fixed_unit=self.fixed_unit,
                                   pts=self.fixed_pts, factory=self._run_factory,
                                   algorithm=self.algorithm, parameters=self.parameters,
                                   amplitude=self.amplitude.value(), resume=False, model=self.selected_model)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.failed.connect(self._on_fail)
        self.worker.start()

    def _request_stop(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_stop()
            self.stop_btn.setEnabled(False)
            self.status.setText(tr("stopping"))

    def _on_progress(self, rec, run):
        self._evals.append(rec.eval_id)
        self._bests.append(rec.best_dist)
        del self._evals[:-2000]
        del self._bests[:-2000]
        self.curve_conv.setData(self._evals, self._bests)
        self.plot_k.enableAutoRange()
        if isinstance(run, PredictedStructure):
            self.canvas0.set_static(run.positions, run.edges)
            if rec.dist <= rec.best_dist:
                self._show_prediction(run)
        elif run is not None:
            self.canvas.clear_static()
            self.canvas.set_color_mode("strain")
            self.canvas.set_data(run, None)
            self.canvas.set_frame(run.n_frames - 1)
            self.canvas0.set_static(run.frames_xy[0], run.edges,
                                    run.left_nodes, run.right_nodes)
            if rec.dist <= rec.best_dist:
                self.best_run = run
                self.curve_best.setData(np.linspace(0, 1, 24), curve_of(run, 24))
        self.log.addItem(f"#{rec.eval_id} [{rec.stage}] {rec.label} "
                         f"obj={rec.dist:.3f} best={rec.best_dist:.3f}")
        while self.log.count() > 2000:
            self.log.takeItem(0)
        self.log.scrollToBottom()
        self.status.setText(f"{tr('running')} #{rec.eval_id}/"
                            f"{self.budget.value()}")

    def _show_prediction(self, preview):
        self.canvas.set_color_mode("plain")
        self.canvas.set_static(preview.positions, preview.edges)
        self.curve_best.setData([], [])
        self.lbl_target_info.setText(("模型预测" if get_lang() == "zh" else "Model prediction") +
            " | peak=%.3g stiff=%.3g tough=%.3g" % tuple(preview.prediction))

    def _preview_target(self):
        tname = self.target_combo.currentText()
        if tname in TARGETS:
            self.curve_target.setData(np.linspace(0, 1, 24),
                                      target_curve(tname, 24))
            self.lbl_target_info.setText(tr('target_label') + ': ' + tname)
        elif tname in SCALARS:
            key, sign = SCALARS[tname]
            self.curve_target.setData([], [])
            kind = 'maximize' if sign < 0 else 'minimize'
            self.lbl_target_info.setText(
                tr('target_label') + f': {tname} ({kind} {key})')
        else:
            self.curve_target.setData([], [])
            self.lbl_target_info.setText('')

    def _on_done(self, res):
        self._result = res
        self.gb_cfg.setEnabled(True)
        self.apply_btn.setEnabled(bool(res.get('best_spec')))
        if res.get('best_run') is not None:
            run = res['best_run']
            self.best_run = run
            self.canvas.set_data(run, None)
            self.canvas.set_frame(run.n_frames - 1)
            self.canvas0.set_static(run.frames_xy[0], run.edges, run.left_nodes, run.right_nodes)
            self.curve_best.setData(np.linspace(0, 1, 24), curve_of(run, 24))
        tname = self.target_combo.currentText()
        self._preview_target()
        if res.get("best_preview") is not None:
            self._show_prediction(res["best_preview"])
        self.stop_btn.setEnabled(False)
        note = tr("stopped_note") + " · " if res.get("stopped") else ""
        if res.get("best_run") is not None:
            m = metrics_of(res["best_run"])
            self.status.setText(
                f"{note}{tr('ready')}: {res['best_label']} "
                f"obj={res['best_dist']:.3f}"
                f" | peak={m['peak']:.2e} stiff={m['stiff']:.2e} "
                f"tough={m['tough']:.2e}")
        else:
            self.status.setText(f"{note}{tr('ready')}: {res['best_label']}")
        self.run_btn.setEnabled(True)
        self.algorithm_btn.setEnabled(True)
        self.model_btn.setEnabled(True)

        if getattr(self, '_auto_apply', False):
            self._auto_apply = False
            self._apply_best()

    def _on_fail(self, msg):
        self.gb_cfg.setEnabled(True)
        self.status.setText(f"{tr('sim_failed')}: {msg}")
        self.run_btn.setEnabled(True)
        self.algorithm_btn.setEnabled(True)
        self.model_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    # ---------------- export ----------------
    def _export_as(self, writer, stem, ext, title_key, silent=None,
                   caption=()):
        if getattr(self, "_result", None) is None:
            self.status.setText(tr("export_nothing"))
            return None
        path = ask_save(self, tr(title_key), unique_stem(stem, "", ext),
                        f"{ext.upper()} (*.{ext})", silent)
        if not path:
            return None
        try:
            writer(path)
            if caption:
                add_caption(path, list(caption), self.mode)
        except Exception as e:
            self.status.setText(f"{tr('sim_failed')}: {e}")
            return None
        self.status.setText(f"{tr('exported')}: {path}")
        return path

    def _export_inv_csv(self, silent=None):
        return self._export_as(
            lambda p: export_inverse_csv(self._result["records"], p),
            "inverse_log", "csv", "export_csv_btn", silent)

    def _export_best_png(self, silent=None):
        return self._export_as(lambda p: save_plot_png(self.plot_c, p),
                               "best_vs_target", "png", "export_png_btn",
                               silent,
                               caption=[f"FiberScope · "
                                        f"{self.target_combo.currentText()}",
                                        tr("ex_best_png")])

    def _export_conv_png(self, silent=None):
        return self._export_as(lambda p: save_plot_png(self.plot_k, p),
                               "convergence", "png", "export_png_btn", silent,
                               caption=[f"FiberScope · "
                                        f"{self.target_combo.currentText()}",
                                        tr("ex_conv_png")])

    def _export_struct_png(self, silent=None):
        return self._export_as(lambda p: save_widget_png(self.canvas, p),
                               "best_structure", "png", "export_png_btn",
                               silent,
                               caption=[f"FiberScope · "
                                        f"{self.target_combo.currentText()}",
                                        tr("ex_struct_png")])

    def _apply_best(self):
        res = getattr(self, "_result", None)
        if not res or not res.get("best_spec"):
            return
        sp = res["best_spec"]
        f = replace(getattr(self, '_run_factory', self.factory), unit=sp["unit"],
                             n_pts_per_side=len(sp.get("line_displacements") or []) or getattr(self, "_run_factory", self.factory).n_pts_per_side,
                             perturbation=sp["pert"],
                             line_displacements=sp["line_displacements"])
        self.apply_structure.emit(f.clamped())

    def _select_unit(self, *_):
        self.fixed_unit = self.unit_combo.currentData()
        self.factory = replace(self.factory, unit=self.fixed_unit)

    def _sync_type(self, *_):
        unit = getattr(self, 'model_unit', self.unit_combo.currentData()) if self.selected_model is not None else (
            getattr(self, '_current_factory', self.factory).unit if self.follow_type.isChecked() else self.unit_combo.currentData())
        if self.unit_combo.findData(unit)<0:
            self.unit_combo.addItem(unit_display(unit,get_lang()),unit)
        self.unit_combo.setCurrentIndex(self.unit_combo.findData(unit))
        self.unit_combo.setEnabled(self.selected_model is None and not self.follow_type.isChecked())
        self.fixed_unit = unit
        self.factory = replace(self.factory,unit=unit)

    def set_spec(self, f):
        self._current_factory = replace(f)
        self.factory = replace(f, unit=self.unit_combo.currentData())
        self.fixed_pts = max(1, min(24, int(f.n_pts_per_side)))
        self._sync_type()

    def show_external(self, res, target_name):
        """Populate plots/log from an externally run inverse design (AI)."""
        items = [self.target_combo.itemText(i)
                 for i in range(self.target_combo.count())]
        if target_name in items:
            self.target_combo.setCurrentText(target_name)
        self.log.clear()
        self._evals, self._bests = [], []
        for rec in res.get("records", []):
            self._evals.append(rec.eval_id)
            self._bests.append(rec.best_dist)
            self.log.addItem(f"#{rec.eval_id} [{rec.stage}] {rec.label} "
                             f"obj={rec.dist:.3f} best={rec.best_dist:.3f}")
        self.curve_conv.setData(self._evals, self._bests)
        run = res.get("best_run")
        if run is not None:
            self.best_run = run
            self.canvas.set_data(run, None)
            self.canvas.set_frame(run.n_frames - 1)
            self.canvas0.set_static(run.frames_xy[0], run.edges,
                                    run.left_nodes, run.right_nodes)
            self.curve_best.setData(np.linspace(0, 1, 24), curve_of(run, 24))
        self._on_done(res)

    def run_sync(self):
        """Test hook."""
        self.log.clear()
        self._evals, self._bests = [], []
        w = DesignWorker(self.target_combo.currentText(),
                         self.budget.value(), self.seed.value(),
                         fixed_unit=self.fixed_unit, pts=self.fixed_pts)
        res = run_inverse(w._builder, w.target, budget=w.budget, seed=w.seed,
                          fixed_unit=w.fixed_unit, pts=w.pts,
                          callback=lambda rec, run: self._on_progress(rec, run))
        self._on_done(res)

    # ---------------- theme / i18n ----------------
    def set_mode(self, mode):
        self.mode = mode
        self.canvas.set_mode(mode)
        self.canvas0.set_mode(mode)
        c = colors(mode)
        for pw in (self.plot_c, self.plot_k):
            apply_plot_theme(pw, mode)
        self.curve_best.setPen(pg.mkPen(c["accent"], width=2))
        self.curve_conv.setPen(pg.mkPen(c["accent2"], width=2))

    def retranslate(self):
        zh = get_lang() == 'zh'
        self.model_btn.setText('用机器学习模型加速' if zh else 'Use ML model')
        self.follow_type.setText('跟随当前结构类型' if zh else 'Follow current type')
        self.lbl_unit.setText('结构类型' if zh else 'Structure type')
        self.lbl_amplitude.setText('探索幅度' if zh else 'Amplitude')
        self.amplitude.setToolTip('参考边长的比例；初期广泛探索，后期细化。' if zh else 'Fraction of reference edge length; broad exploration followed by refinement.')
        for i in range(self.unit_combo.count()):
            self.unit_combo.setItemText(i, unit_display(self.unit_combo.itemData(i), get_lang()))
        self.algorithm_btn.setText(SEARCH_SPECS[self.algorithm][0 if get_lang() == 'zh' else 1] + ' · …')
        self.gb_cfg.setTitle(tr("design_cfg"))
        self.gb_log.setTitle(tr("eval_log"))
        self.gb_curve.setTitle(tr("best_curve"))
        self.gb_conv.setTitle(tr("convergence"))
        self.lbl_target.setText(tr("target_label"))
        self.lbl_budget.setText(tr("budget"))
        self.lbl_seed.setText(tr("seed"))
        self.run_btn.setText(tr("run_btn"))
        self.stop_btn.setText(tr("stop_btn"))
        self.apply_btn.setText(tr("apply_struct"))
        self.export_btn.setText(tr("export_btn"))
        self.act_inv_csv.setText(tr("ex_inv_csv"))
        self.act_best_png.setText(tr("ex_best_png"))
        self.act_conv_png.setText(tr("ex_conv_png"))
        self.act_struct_png.setText(tr("ex_struct_png"))
        self.lbl_init.setText('当前候选 · 初始形态（完成后显示最优）' if zh else 'Current candidate · initial (best on completion)')
        self.lbl_def.setText('当前候选 · 拉伸形态' if zh else 'Current candidate · stretched')
        self.replay_toggle.setText(tr("replay_toggle"))
