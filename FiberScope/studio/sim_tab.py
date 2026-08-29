"""Unified live-simulation tab: one stretch run drives two synchronized
views (load-path percolation / strain+contact) plus three curves
(percolation order parameter, force-stretch, mode energy split).

Red dots = fiber-fiber contact events (strain view); amber = grips.
"""
import os

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QGroupBox, QHBoxLayout, QLabel, QProgressBar,
                               QPushButton, QScrollArea, QSizePolicy, QSlider,
                               QSplitter, QVBoxLayout, QWidget)

from fslab import StructureFactory, RunConfig, run_stretch, compute_percolation
from .i18n import tr
from .network_canvas import NetworkCanvas
from .structure_tab import spec_text
from .theme import apply_plot_theme, colors

import sys as _sys


def _cache_dir():
    if getattr(_sys, "frozen", False):
        return os.path.join(os.path.dirname(_sys.executable), "_cache")
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_cache")


CACHE_DIR = _cache_dir()
SPEEDS = [0.5, 1.0, 2.0, 4.0]


class SimWorker(QThread):
    finished_ok = Signal(object, object, bool)  # run, perc, cache_hit
    failed = Signal(str)
    progress = Signal(int)

    def __init__(self, factory, cfg, quantile, parent=None):
        super().__init__(parent)
        self.factory = factory
        self.cfg = cfg
        self.quantile = quantile

    def run(self):
        try:
            hit = False
            if os.path.isdir(CACHE_DIR):
                import json, hashlib
                from dataclasses import asdict
                raw = json.dumps({"s": self.factory.key(),
                                  "c": asdict(self.cfg)}, sort_keys=True)
                tag = hashlib.sha1(raw.encode()).hexdigest()[:16]
                hit = os.path.exists(os.path.join(CACHE_DIR,
                                                  f"run_{tag}.npz"))

            def cb(done, total):
                self.progress.emit(int(100 * done // max(int(total), 1)))
            run = run_stretch(self.factory, self.cfg, cache_dir=CACHE_DIR,
                              progress_cb=cb)
            perc = compute_percolation(run, alpha=self.quantile)
            self.finished_ok.emit(run, perc, hit)
        except Exception as e:
            self.failed.emit(f"{e.__class__.__name__}: {e}")


class SimTab(QWidget):
    def __init__(self, mode="dark", parent=None):
        super().__init__(parent)
        self.mode = mode
        self.run = None
        self.perc = None
        self.worker = None
        self._never_txt = None
        self.spec = StructureFactory(unit="reentrant", grid_x=3, grid_y=3,
                                     n_pts_per_side=5, seed=7).clamped()
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(120)
        self._play_timer.timeout.connect(self._play_step)
        self._build()
        self.retranslate()

    # ---------------- UI ----------------
    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        chip_row = QHBoxLayout()
        self.struct_chip = QLabel()
        self.struct_chip.setObjectName("chip_accent")
        self.struct_hint = QLabel()
        self.struct_hint.setObjectName("hint")
        chip_row.addWidget(self.struct_chip)
        chip_row.addWidget(self.struct_hint)
        chip_row.addStretch(1)
        outer.addLayout(chip_row)

        body = QSplitter(Qt.Horizontal)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(10)

        self.gb_fixed = QGroupBox()
        fv = QVBoxLayout(self.gb_fixed)
        fv.setSpacing(10)
        row1 = QHBoxLayout()
        self.lbl_stretch = QLabel()
        self.stretch = QDoubleSpinBox()
        self.stretch.setRange(1.1, 3.0); self.stretch.setSingleStep(0.1)
        self.stretch.setValue(2.0)
        row1.addWidget(self.lbl_stretch); row1.addWidget(self.stretch)
        fv.addLayout(row1)
        row2 = QHBoxLayout()
        self.lbl_quant = QLabel()
        self.quant = QSlider(Qt.Horizontal); self.quant.setRange(2, 15)
        self.quant.setValue(5)
        self.quant_val = QLabel("0.05")
        self.quant_val.setObjectName("chip")
        self.quant.valueChanged.connect(
            lambda v: self.quant_val.setText(f"{v / 100:.2f}"))
        row2.addWidget(self.lbl_quant); row2.addWidget(self.quant)
        row2.addWidget(self.quant_val)
        fv.addLayout(row2)
        self.chk_bend = QCheckBox(); self.chk_bend.setChecked(True)
        self.chk_contact = QCheckBox(); self.chk_contact.setChecked(True)
        fv.addWidget(self.chk_bend); fv.addWidget(self.chk_contact)
        self.lbl_phys = QLabel(); self.lbl_phys.setObjectName("hint")
        self.lbl_phys.setWordWrap(True)
        self.lbl_bc = QLabel(); self.lbl_bc.setObjectName("hint")
        self.lbl_bc.setWordWrap(True)
        fv.addWidget(self.lbl_phys); fv.addWidget(self.lbl_bc)
        lv.addWidget(self.gb_fixed)

        self.run_btn = QPushButton()
        self.run_btn.setProperty("primary", True)
        self.run_btn.clicked.connect(self.start_run)
        lv.addWidget(self.run_btn)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFixedHeight(16)
        self.progress.setFormat("%p%")
        lv.addWidget(self.progress)
        self.status = QLabel(); self.status.setObjectName("subtitle")
        self.status.setWordWrap(True)
        lv.addWidget(self.status)
        lv.addStretch(1)
        left.setMinimumWidth(240)
        left.setMaximumWidth(320)
        body.addWidget(left)

        center = QWidget()
        cv = QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(8)
        vrow = QHBoxLayout()
        self.lbl_view = QLabel()
        self.lbl_view.setObjectName("hint")
        self.view_combo = QComboBox()
        self.view_combo.setFixedWidth(150)
        self.view_combo.currentIndexChanged.connect(self._set_view)
        vrow.addWidget(self.lbl_view)
        vrow.addWidget(self.view_combo)
        vrow.addStretch(1)
        cv.addLayout(vrow)
        self.canvas = NetworkCanvas(mode=self.mode)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cv.addWidget(self.canvas, 1)

        transport = QHBoxLayout()
        transport.setSpacing(8)
        self.play_btn = QPushButton()
        self.play_btn.setFixedWidth(76)
        self.play_btn.clicked.connect(self._toggle_play)
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5×", "1×", "2×", "4×"])
        self.speed_combo.setCurrentIndex(1)
        self.speed_combo.setFixedWidth(64)
        self.speed_combo.currentIndexChanged.connect(self._on_speed)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._on_slider)
        self.lbl_frame = QLabel(); self.lbl_frame.setObjectName("chip")
        self.lbl_strain = QLabel(); self.lbl_strain.setObjectName("chip")
        self.lbl_contacts = QLabel(); self.lbl_contacts.setObjectName("chip")
        transport.addWidget(self.play_btn)
        transport.addWidget(self.speed_combo)
        transport.addWidget(self.slider, 1)
        transport.addWidget(self.lbl_frame)
        transport.addWidget(self.lbl_strain)
        transport.addWidget(self.lbl_contacts)
        cv.addLayout(transport)
        body.addWidget(center)

        right_wrap = QScrollArea()
        right_wrap.setWidgetResizable(True)
        right_wrap.setFrameShape(QScrollArea.NoFrame)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(8)

        self.gb_curve = QGroupBox()
        kv = QVBoxLayout(self.gb_curve)
        self.plot = pg.PlotWidget()
        self.plot.setMenuEnabled(False)
        self.plot.setMinimumHeight(140)
        self.curve_p = self.plot.plot(
            pen=pg.mkPen(colors(self.mode)["accent"], width=2))
        self.curve_b = self.plot.plot(
            pen=pg.mkPen(colors(self.mode)["accent2"], width=1.5,
                         style=Qt.DashLine))
        self.marker = pg.InfiniteLine(angle=90, movable=False,
                                      pen=pg.mkPen("#888888", width=1,
                                                   style=Qt.DotLine))
        self.plot.addItem(self.marker)
        kv.addWidget(self.plot)
        self.lbl_perc_at = QLabel(); self.lbl_perc_at.setObjectName("chip_accent")
        self.lbl_legend = QLabel(); self.lbl_legend.setObjectName("hint")
        self.lbl_legend.setWordWrap(True)
        kv.addWidget(self.lbl_perc_at)
        kv.addWidget(self.lbl_legend)

        self.gb_force = QGroupBox()
        fvv = QVBoxLayout(self.gb_force)
        self.plot_f = pg.PlotWidget()
        self.plot_f.setMenuEnabled(False)
        self.plot_f.setMinimumHeight(130)
        self.curve_f = self.plot_f.plot(
            pen=pg.mkPen(colors(self.mode)["accent"], width=2))
        self.marker_f = pg.InfiniteLine(angle=90, movable=False,
                                        pen=pg.mkPen("#888888", width=1,
                                                     style=Qt.DotLine))
        self.plot_f.addItem(self.marker_f)
        fvv.addWidget(self.plot_f)
        rv.addWidget(self.gb_force)

        self.gb_mode = QGroupBox()
        mv = QVBoxLayout(self.gb_mode)
        self.plot_m = pg.PlotWidget()
        self.plot_m.setMenuEnabled(False)
        self.plot_m.setMinimumHeight(130)
        self.plot_m.setYRange(0, 1)
        self.fill_a = None
        self.m_c1 = self.plot_m.plot(pen=pg.mkPen(colors(self.mode)["accent"],
                                                  width=1.5))
        self.m_c2 = self.plot_m.plot(pen=pg.mkPen(colors(self.mode)["warn"],
                                                  width=1.5))
        self.m_c3 = self.plot_m.plot(pen=pg.mkPen("#ff5d47", width=1.5))
        self.marker_m = pg.InfiniteLine(angle=90, movable=False,
                                        pen=pg.mkPen("#888888", width=1,
                                                     style=Qt.DotLine))
        self.plot_m.addItem(self.marker_m)
        mv.addWidget(self.plot_m)
        self.lbl_legend_m = QLabel(); self.lbl_legend_m.setObjectName("hint")
        mv.addWidget(self.lbl_legend_m)
        rv.addWidget(self.gb_mode)
        rv.addWidget(self.gb_curve)

        self.gb_explain = QGroupBox()
        evx = QVBoxLayout(self.gb_explain)
        self.lbl_explain = QLabel()
        self.lbl_explain.setObjectName("hint")
        self.lbl_explain.setWordWrap(True)
        evx.addWidget(self.lbl_explain)
        rv.addWidget(self.gb_explain)
        rv.addStretch(1)
        right.setMinimumWidth(260)
        right.setMaximumWidth(400)
        right_wrap.setWidget(right)
        body.addWidget(right_wrap)

        body.setSizes([270, 760, 330])
        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)
        body.setStretchFactor(2, 0)
        outer.addWidget(body, 1)

    # ---------------- shared structure ----------------
    def set_spec(self, factory):
        self.spec = factory
        self.struct_chip.setText("⬡ " + spec_text(factory))

    def collect_factory(self) -> StructureFactory:
        return StructureFactory(
            unit=self.spec.unit, grid_x=self.spec.grid_x,
            grid_y=self.spec.grid_y, n_pts_per_side=self.spec.n_pts_per_side,
            perturbation=self.spec.perturbation,
            seed=self.spec.seed,
            line_displacements=self.spec.line_displacements).clamped()

    def collect_cfg(self) -> RunConfig:
        return RunConfig(target_stretch=self.stretch.value(),
                         use_bending=self.chk_bend.isChecked(),
                         use_contact=self.chk_contact.isChecked())

    # ---------------- run ----------------
    def start_run(self):
        if self.worker is not None and self.worker.isRunning():
            return
        self._play_timer.stop()
        self.run_btn.setEnabled(False)
        self.status.setText(tr("running"))
        self.progress.setValue(0)
        self.worker = SimWorker(self.collect_factory(), self.collect_cfg(),
                                self.quant.value() / 100.0)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.failed.connect(self._on_fail)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.start()

    def run_sync(self):
        """Test hook: run without a worker thread."""
        run = run_stretch(self.collect_factory(), self.collect_cfg(),
                          cache_dir=CACHE_DIR)
        perc = compute_percolation(run, alpha=self.quant.value() / 100.0)
        self._on_done(run, perc, False)

    def _on_done(self, run, perc, cache_hit):
        self.run, self.perc = run, perc
        F = run.n_frames
        self.slider.blockSignals(True)
        self.slider.setRange(0, F - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)

        x = np.arange(F)
        self.curve_p.setData(x, perc.spanning_frac)
        self.curve_b.setData(x, perc.backbone_frac)
        ymax = max(0.12, float(perc.spanning_frac.max()) * 1.25)
        self.plot.setYRange(0, ymax)
        if perc.perc_frame >= 0:
            s = run.strain_levels[perc.perc_frame]
            self.lbl_perc_at.setText(
                f"{tr('perc_at')}: #{perc.perc_frame} ({s:.2f})")
        else:
            self.lbl_perc_at.setText(tr("never"))
        if perc.perc_frame < 0:
            if self._never_txt is None:
                self._never_txt = pg.TextItem("", anchor=(0.5, 0.5))
                self.plot.addItem(self._never_txt)
            self._never_txt.setText(tr("never"))
            self._never_txt.setColor(colors(self.mode)["sub"])
            self._never_txt.setPos((F - 1) / 2.0, ymax / 2.0)
            self._never_txt.setVisible(True)
        elif self._never_txt is not None:
            self._never_txt.setVisible(False)

        self.curve_f.setData(run.strain_levels, run.force_curve)
        e = run.energies
        tot = e["axial"] + e["bend"] + e["contact"] + 1e-12
        fa = e["axial"] / tot
        fab = fa + e["bend"] / tot
        self.m_c1.setData(x, fa)
        self.m_c2.setData(x, fab)
        self.m_c3.setData(x, np.ones(F))
        if self.fill_a is None:
            zero = pg.PlotDataItem(x, np.zeros(F))
            self.plot_m.addItem(zero)
            self._zero = zero
            self.fill_a = pg.FillBetweenItem(
                self.m_c1, zero,
                brush=pg.mkColor(colors(self.mode)["accent"] + "55"))
            self.fill_b = pg.FillBetweenItem(
                self.m_c2, self.m_c1,
                brush=pg.mkColor(colors(self.mode)["warn"] + "55"))
            self.fill_c = pg.FillBetweenItem(
                self.m_c3, self.m_c2, brush=pg.mkColor("#ff5d4755"))
            for it in (self.fill_a, self.fill_b, self.fill_c):
                self.plot_m.addItem(it)

        msg = tr("ready") + (f" ({tr('cached')})" if cache_hit else "")
        msg += f"  N={run.frames_xy.shape[1]} E={run.n_edges}"
        self.status.setText(msg)
        self.run_btn.setEnabled(True)
        self.progress.setValue(100)
        self._set_view(self.view_combo.currentIndex())
        self._sync_frame(0)
        self._start_play()

    def _on_fail(self, msg):
        self.status.setText(f"{tr('sim_failed')}: {msg}")
        self.run_btn.setEnabled(True)

    # ---------------- views / playback ----------------
    def _set_view(self, idx):
        if self.run is None:
            return
        if idx <= 0:
            self.canvas.set_color_mode("percolation")
            self.canvas.set_data(self.run, self.perc)
            self._set_legend_perc()
            self.lbl_explain.setText(tr("explain_perc"))
        else:
            self.canvas.set_color_mode("strain")
            self.canvas.set_data(self.run, None)
            self._set_legend_strain()
            self.lbl_explain.setText(tr("explain_stretch"))
        self.canvas.set_frame(self.slider.value())

    def _on_speed(self, idx):
        self._play_timer.setInterval(int(120 / SPEEDS[idx]))

    def _on_slider(self, v):
        self._sync_frame(v)

    def _sync_frame(self, v):
        if self.run is None:
            return
        self.canvas.set_frame(v)
        self.marker.setValue(v)
        self.marker_m.setValue(v)
        self.marker_f.setValue(self.run.strain_levels[v])
        self.lbl_frame.setText(f"{tr('frame')} {v + 1}/{self.run.n_frames}")
        self.lbl_strain.setText(
            f"{tr('strain')} {self.run.strain_levels[v]:.3f}")
        cc = self.run.contact_counts[v] if v < len(self.run.contact_counts) \
            else 0
        self.lbl_contacts.setText(f"{tr('m_contact')}: {cc}")

    def _start_play(self):
        if self.run is None:
            return
        self.slider.setValue(0)
        self._play_timer.start()
        self.retranslate()

    def _toggle_play(self):
        if self.run is None:
            return
        if self._play_timer.isActive():
            self._play_timer.stop()
        else:
            if self.slider.value() >= self.slider.maximum():
                self.slider.setValue(0)
            self._play_timer.start()
        self.retranslate()

    def _play_step(self):
        v = self.slider.value() + 1
        if v > self.slider.maximum():
            self._play_timer.stop()
            self.retranslate()
            return
        self.slider.setValue(v)

    # ---------------- theme / i18n ----------------
    def set_mode(self, mode):
        self.mode = mode
        self.canvas.set_mode(mode)
        c = colors(mode)
        for pw in (self.plot, self.plot_f, self.plot_m):
            apply_plot_theme(pw, mode)
        self.curve_p.setPen(pg.mkPen(c["accent"], width=2))
        self.curve_b.setPen(pg.mkPen(c["accent2"], width=1.5,
                                     style=Qt.DashLine))
        self.curve_f.setPen(pg.mkPen(c["accent"], width=2))
        self.m_c1.setPen(pg.mkPen(c["accent"], width=1.5))
        self.m_c2.setPen(pg.mkPen(c["warn"], width=1.5))
        if self._never_txt is not None:
            self._never_txt.setColor(c["sub"])
        self._set_view(self.view_combo.currentIndex())

    def _set_legend_perc(self):
        c = colors(self.mode)
        self.canvas.set_legend([
            (c["edge_inactive"], tr("legend_inactive")),
            ("#5b7fb9", tr("legend_load")),
            (c["accent"], tr("legend_span")),
            (c["warn"], tr("legend_grip")),
        ])

    def _set_legend_strain(self):
        c = colors(self.mode)
        self.canvas.set_legend([
            ("#4dd0e1", tr("legend_comp")),
            (c["edge_inactive"], tr("legend_neutral")),
            ("#ff5d47", tr("legend_tens")),
            ("#ff3b30", tr("legend_contact")),
            (c["warn"], tr("legend_grip")),
        ])

    def retranslate(self):
        self.struct_hint.setText(tr("struct_hint"))
        if self.spec is not None:
            self.struct_chip.setText("⬡ " + spec_text(self.spec))
        self.gb_fixed.setTitle(tr("fixed_card"))
        self.gb_explain.setTitle(tr("explain_card"))
        self.gb_curve.setTitle(tr("perc_curve_card"))
        self.gb_force.setTitle(tr("force_curve"))
        self.gb_mode.setTitle(tr("mode_split"))
        self.lbl_view.setText(tr("view_label"))
        if self.view_combo.count() == 0:
            self.view_combo.addItems([tr("view_perc"), tr("view_strain")])
        else:
            self.view_combo.setItemText(0, tr("view_perc"))
            self.view_combo.setItemText(1, tr("view_strain"))
        self.lbl_stretch.setText(tr("stretch"))
        self.lbl_quant.setText(tr("quantile"))
        self.chk_bend.setText(tr("phys_bending"))
        self.chk_contact.setText(tr("phys_contact"))
        self.lbl_phys.setText(tr("fixed_physics"))
        self.lbl_bc.setText(tr("fixed_bc"))
        self.run_btn.setText(tr("run_btn"))
        self.lbl_legend.setText(
            f"— {tr('perc_order')}   - - {tr('backbone')}")
        self.lbl_legend_m.setText(
            f"— {tr('m_axial')}  — {tr('m_bend')}  — {tr('m_contact')}")
        self.play_btn.setText(tr("pause_btn") if self._play_timer.isActive()
                              else tr("play_btn"))
        self._set_view(self.view_combo.currentIndex())
