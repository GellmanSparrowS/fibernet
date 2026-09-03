"""Inverse design tab: live two-stage optimizer (topology screen + CEM over
the reference-line point values) streaming convergence, best curve vs
target, stretched best structure plus its pre-stretch inset."""
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (QComboBox, QFrame, QGridLayout, QGroupBox,
                               QHBoxLayout, QLabel, QListWidget, QPushButton,
                               QSizePolicy, QSpinBox, QSplitter,
                               QVBoxLayout, QWidget)

from fslab import StructureFactory
from fslab.inverse import (run_inverse, target_curve, curve_of, metrics_of,
                           TARGETS, SCALARS, PTS)
from fslab.structure import UNIT_PRESETS
from .i18n import tr
from .network_canvas import NetworkCanvas
from .theme import apply_plot_theme, colors


class DesignWorker(QThread):
    progress = Signal(object, object)   # record, run_or_None
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, target, budget, seed, fixed_unit=None, pts=None,
                 parent=None):
        super().__init__(parent)
        self.target = target
        self.budget = budget
        self.seed = seed
        self.fixed_unit = fixed_unit
        self.pts = int(pts) if pts else PTS

    def _builder(self, unit, pert, ld):
        return StructureFactory(unit=unit, grid_x=3, grid_y=3,
                                n_pts_per_side=self.pts, seed=self.seed,
                                perturbation=pert,
                                line_displacements=ld).build()

    def run(self):
        try:
            res = run_inverse(self._builder, self.target, budget=self.budget,
                              seed=self.seed, fixed_unit=self.fixed_unit,
                              pts=self.pts,
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
        self.best_run = None
        self.fixed_unit = "reentrant"
        self.fixed_pts = 5
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
        self.budget = QSpinBox(); self.budget.setRange(24, 120)
        self.budget.setValue(40)
        self.lbl_seed = QLabel()
        self.seed = QSpinBox(); self.seed.setRange(0, 9999); self.seed.setValue(7)
        g.addWidget(self.lbl_target, 0, 0); g.addWidget(self.target_combo, 0, 1)
        g.addWidget(self.lbl_budget, 1, 0); g.addWidget(self.budget, 1, 1)
        g.addWidget(self.lbl_seed, 2, 0); g.addWidget(self.seed, 2, 1)
        lv.addWidget(self.gb_cfg)
        self.run_btn = QPushButton()
        self.run_btn.setProperty("primary", True)
        self.run_btn.clicked.connect(self.start_run)
        self.apply_btn = QPushButton()
        self.apply_btn.clicked.connect(self._apply_best)
        self.status = QLabel(); self.status.setObjectName("subtitle")
        self.status.setWordWrap(True)
        lv.addWidget(self.run_btn); lv.addWidget(self.apply_btn)
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
    def start_run(self):
        if self.worker is not None and self.worker.isRunning():
            return
        self.run_btn.setEnabled(False)
        self.status.setText(tr("running"))
        self.log.clear()
        self._evals, self._bests = [], []
        self._preview_target()
        self.worker = DesignWorker(self.target_combo.currentText(),
                                   self.budget.value(), self.seed.value(),
                                   fixed_unit=self.fixed_unit,
                                   pts=self.fixed_pts)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.failed.connect(self._on_fail)
        self.worker.start()

    def _on_progress(self, rec, run):
        self._evals.append(rec.eval_id)
        self._bests.append(rec.best_dist)
        self.curve_conv.setData(self._evals, self._bests)
        self.plot_k.enableAutoRange()
        if run is not None:
            self.best_run = run
            self.canvas.set_data(run, None)
            self.canvas.set_frame(run.n_frames - 1)
            self.canvas0.set_static(run.frames_xy[0], run.edges,
                                    run.left_nodes, run.right_nodes)
            self.curve_best.setData(np.linspace(0, 1, 24), curve_of(run, 24))
        self.log.addItem(f"#{rec.eval_id} [{rec.stage}] {rec.label} "
                         f"obj={rec.dist:.3f} best={rec.best_dist:.3f}")
        self.log.scrollToBottom()
        self.status.setText(f"{tr('running')} #{rec.eval_id}/"
                            f"{self.budget.value()}")

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
        tname = self.target_combo.currentText()
        self._preview_target()
        if res.get("best_run") is not None:
            m = metrics_of(res["best_run"])
            self.status.setText(
                f"{tr('ready')}: {res['best_label']} obj={res['best_dist']:.3f}"
                f" | peak={m['peak']:.2e} stiff={m['stiff']:.2e} "
                f"tough={m['tough']:.2e}")
        else:
            self.status.setText(f"{tr('ready')}: {res['best_label']}")
        self.run_btn.setEnabled(True)

    def _on_fail(self, msg):
        self.status.setText(f"{tr('sim_failed')}: {msg}")
        self.run_btn.setEnabled(True)

    def _apply_best(self):
        res = getattr(self, "_result", None)
        if not res or not res.get("best_spec"):
            return
        sp = res["best_spec"]
        f = StructureFactory(unit=sp["unit"], grid_x=3, grid_y=3,
                             n_pts_per_side=self.fixed_pts,
                             seed=self.seed.value(),
                             perturbation=sp["pert"],
                             line_displacements=sp["line_displacements"])
        self.apply_structure.emit(f.clamped())

    def set_spec(self, f):
        self.fixed_unit = f.unit
        self.fixed_pts = max(1, min(6, int(f.n_pts_per_side)))

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
        self.gb_cfg.setTitle(tr("design_cfg"))
        self.gb_log.setTitle(tr("eval_log"))
        self.gb_curve.setTitle(tr("best_curve"))
        self.gb_conv.setTitle(tr("convergence"))
        self.lbl_target.setText(tr("target_label"))
        self.lbl_budget.setText(tr("budget"))
        self.lbl_seed.setText(tr("seed"))
        self.run_btn.setText(tr("run_btn"))
        self.apply_btn.setText(tr("apply_struct"))
        self.lbl_init.setText(tr("initial_struct"))
        self.lbl_def.setText(tr("stretched_struct"))
        self.replay_toggle.setText(tr("replay_toggle"))
