"""Exploration replay tab: plays back the shipped exploration log
(novelty-search agent vs R0 random, same budget) in behavior space.

Interactive: click scatter points or feed entries to jump, hover for
details, load any replayed structure into the simulation tabs, and
discovery events are marked on the cluster-discovery curve.
"""
import json
import os
import sys

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (QCheckBox, QGroupBox, QHBoxLayout, QLabel,
                               QListWidget, QListWidgetItem, QPushButton,
                               QSizePolicy, QSlider, QSplitter, QToolTip,
                               QVBoxLayout, QWidget)

from fslab import StructureFactory
from .i18n import tr
from .theme import apply_plot_theme, colors


def _app_root():
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


ROOT = _app_root()
LOG = os.path.join(ROOT, "data", "exploration_log.jsonl")
CLUSTER_R = 0.35


def load_log():
    recs = []
    if os.path.exists(LOG):
        with open(LOG, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    return recs


class ReplayTab(QWidget):
    apply_structure = Signal(object)

    def __init__(self, mode="dark", parent=None):
        super().__init__(parent)
        self.mode = mode
        self.recs = load_log()
        self._shown_a, self._shown_r = [], []
        self._prepare()
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(80)
        self._play_timer.timeout.connect(self._play_step)
        self._build()
        self.retranslate()
        if self.recs:
            self.slider.setValue(len(self.recs) - 1)

    def _prepare(self):
        if not self.recs:
            self.B = np.zeros((0, 2))
            return
        X = np.array([r["behavior"] for r in self.recs], dtype=np.float64)
        Xc = X - X.mean(0)
        u, s, vt = np.linalg.svd(Xc, full_matrices=False)
        self.B = Xc @ vt[:2].T
        self.clusters = {"agent": [], "r0": []}
        self.cluster_curve = {"agent": [], "r0": []}
        self.novelty_curve = []
        self.discoveries = {"agent": [], "r0": []}
        counts = {"agent": 0, "r0": 0}
        best = 0.0
        for i, r in enumerate(self.recs):
            b = X[i]
            cl = self.clusters[r["method"]]
            new = (not cl or min(np.linalg.norm(b - a) for a in cl)
                   > CLUSTER_R)
            if new:
                cl.append(b)
                self.discoveries[r["method"]].append(i)
            counts[r["method"]] = len(cl)
            self.cluster_curve["agent"].append(counts["agent"])
            self.cluster_curve["r0"].append(counts["r0"])
            best = max(best, r["novelty"])
            self.novelty_curve.append(best)

    # ---------------- UI ----------------
    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)
        self.lbl_explain = QLabel()
        self.lbl_explain.setObjectName("explain")
        self.lbl_explain.setWordWrap(True)
        outer.addWidget(self.lbl_explain)
        splitter = QSplitter(Qt.Horizontal)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.gb_stats = QGroupBox()
        sv = QVBoxLayout(self.gb_stats)
        self.lbl_stats = QLabel(); self.lbl_stats.setWordWrap(True)
        sv.addWidget(self.lbl_stats)
        row = QHBoxLayout()
        self.chk_agent = QCheckBox("agent"); self.chk_agent.setChecked(True)
        self.chk_r0 = QCheckBox("R0"); self.chk_r0.setChecked(True)
        self.chk_agent.stateChanged.connect(lambda _: self._sync())
        self.chk_r0.stateChanged.connect(lambda _: self._sync())
        row.addWidget(self.chk_agent); row.addWidget(self.chk_r0)
        sv.addLayout(row)
        self.apply_btn = QPushButton()
        self.apply_btn.clicked.connect(self._apply_current)
        sv.addWidget(self.apply_btn)
        lv.addWidget(self.gb_stats)
        self.gb_feed = QGroupBox()
        fv = QVBoxLayout(self.gb_feed)
        self.feed = QListWidget()
        self.feed.itemClicked.connect(self._on_feed_click)
        fv.addWidget(self.feed)
        lv.addWidget(self.gb_feed, 1)
        left.setMinimumWidth(240)
        left.setMaximumWidth(320)
        splitter.addWidget(left)

        center = QWidget()
        cv = QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0)
        self.scatter = pg.PlotWidget()
        self.scatter.setMenuEnabled(False)
        self.scatter.setMinimumHeight(260)
        self.pts_r0 = pg.ScatterPlotItem(size=7, brush=pg.mkColor("#88888899"),
                                         pen=pg.mkPen(None))
        self.pts_agent = pg.ScatterPlotItem(size=8,
                                            brush=pg.mkColor("#3aa0ffcc"),
                                            pen=pg.mkPen(None))
        self.pts_cur = pg.ScatterPlotItem(
            size=16, pen=pg.mkPen(color="#ffb454", width=2),
            brush=pg.mkColor("#00000000"))
        for it in (self.pts_r0, self.pts_agent):
            it.sigClicked.connect(self._on_scatter_click)
            it.setZValue(1)
        for it in (self.pts_r0, self.pts_agent, self.pts_cur):
            self.scatter.addItem(it)
        cv.addWidget(self.scatter, 1)
        self.lbl_step = QLabel()
        self.lbl_step.setObjectName("chip_accent")
        self.lbl_step.setWordWrap(True)
        cv.addWidget(self.lbl_step)
        splitter.addWidget(center)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.gb_cl = QGroupBox()
        cvv = QVBoxLayout(self.gb_cl)
        self.plot_cl = pg.PlotWidget(); self.plot_cl.setMenuEnabled(False)
        self.plot_cl.setMinimumHeight(150)
        self.cl_agent = self.plot_cl.plot(
            pen=pg.mkPen(colors(self.mode)["accent"], width=2))
        self.cl_r0 = self.plot_cl.plot(pen=pg.mkPen("#888888", width=2))
        self.disc_a = pg.ScatterPlotItem(size=8, symbol="t",
                                         brush=pg.mkColor(
                                             colors(self.mode)["accent"]),
                                         pen=pg.mkPen(None))
        self.plot_cl.addItem(self.disc_a)
        cvv.addWidget(self.plot_cl)
        rv.addWidget(self.gb_cl)
        self.gb_nv = QGroupBox()
        nv = QVBoxLayout(self.gb_nv)
        self.plot_nv = pg.PlotWidget(); self.plot_nv.setMenuEnabled(False)
        self.plot_nv.setMinimumHeight(150)
        self.nv_curve = self.plot_nv.plot(
            pen=pg.mkPen(colors(self.mode)["accent2"], width=2))
        nv.addWidget(self.plot_nv)
        rv.addWidget(self.gb_nv)
        right.setMinimumWidth(240)
        right.setMaximumWidth(320)
        splitter.addWidget(right)
        splitter.setSizes([280, 620, 280])
        outer.addWidget(splitter, 1)

        bottom = QWidget()
        bv = QHBoxLayout(bottom)
        bv.setContentsMargins(0, 0, 0, 0)
        self.play_btn = QPushButton(); self.play_btn.setFixedWidth(70)
        self.play_btn.clicked.connect(self._toggle_play)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, max(0, len(self.recs) - 1))
        self.slider.valueChanged.connect(lambda v: self._sync())
        self.lbl_at = QLabel()
        bv.addWidget(self.play_btn); bv.addWidget(self.slider, 1)
        bv.addWidget(self.lbl_at)
        outer.addWidget(bottom)
        self._sync()

    # ---------------- sync ----------------
    def _sync(self):
        if not self.recs:
            return
        k = self.slider.value()
        show_a = self.chk_agent.isChecked()
        show_r = self.chk_r0.isChecked()
        self._shown_r = [i for i in range(k + 1)
                         if self.recs[i]["method"] == "r0" and show_r]
        self._shown_a = [i for i in range(k + 1)
                         if self.recs[i]["method"] == "agent" and show_a]
        self.pts_r0.setData([self.B[i, 0] for i in self._shown_r],
                            [self.B[i, 1] for i in self._shown_r])
        self.pts_agent.setData([self.B[i, 0] for i in self._shown_a],
                               [self.B[i, 1] for i in self._shown_a])
        self.pts_cur.setData([self.B[k, 0]], [self.B[k, 1]])
        x = np.arange(k + 1)
        self.cl_agent.setData(x, self.cluster_curve["agent"][:k + 1])
        self.cl_r0.setData(x, self.cluster_curve["r0"][:k + 1])
        da = [i for i in self.discoveries["agent"] if i <= k]
        self.disc_a.setData(da, [self.cluster_curve["agent"][i] for i in da])
        self.nv_curve.setData(x, self.novelty_curve[:k + 1])
        n_acc = sum(1 for i in range(k + 1)
                    if self.recs[i]["method"] == "agent"
                    and self.recs[i]["accepted"])
        n_ag = sum(1 for i in range(k + 1) if self.recs[i]["method"] == "agent")
        self.lbl_stats.setText(
            f"evals={k + 1}/{len(self.recs)}  agent={n_ag} "
            f"accepted={n_acc}  clusters: agent="
            f"{self.cluster_curve['agent'][k]} r0={self.cluster_curve['r0'][k]}")
        r = self.recs[k]
        self.lbl_step.setText(
            f"#{k + 1} {r['method']} · {r['unit']} · pert={r['pert']:.2f} · "
            f"nov={r['novelty']:.2f} · "
            + ("accepted" if r["accepted"] else "rejected"))
        self.lbl_at.setText(f"#{k + 1}/{len(self.recs)}")
        self.feed.clear()
        recent = []
        for i in range(k, -1, -1):
            rec = self.recs[i]
            if rec["method"] == "agent" and rec["accepted"]:
                recent.append(i)
                if len(recent) >= 30:
                    break
        for i in reversed(recent):
            rec = self.recs[i]
            it = QListWidgetItem(f"#{rec['id']} {rec['unit']} "
                                 f"pert={rec['pert']} "
                                 f"nov={rec['novelty']:.2f}")
            it.setData(Qt.UserRole, i)
            self.feed.addItem(it)
        self.feed.scrollToBottom()

    # ---------------- interactions ----------------
    def _on_scatter_click(self, item, pts):
        if not pts:
            return
        mapping = self._shown_a if item is self.pts_agent else self._shown_r
        i = mapping[pts[0].index()]
        self.slider.setValue(i)
        r = self.recs[i]
        QToolTip.showText(
            QCursor.pos(),
            f"#{i + 1} {r['method']} {r['unit']} pert={r['pert']} "
            f"nov={r['novelty']:.2f}")

    def _on_feed_click(self, it):
        i = it.data(Qt.UserRole)
        if i is not None:
            self.slider.setValue(int(i))

    def _apply_current(self):
        if not self.recs:
            return
        r = self.recs[self.slider.value()]
        f = StructureFactory(unit=r["unit"], grid_x=3, grid_y=3,
                             n_pts_per_side=2, seed=int(r.get("seed", 7)),
                             perturbation=float(r["pert"])).clamped()
        self.apply_structure.emit(f)

    def _toggle_play(self):
        if not self.recs:
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
        c = colors(mode)
        for pw in (self.scatter, self.plot_cl, self.plot_nv):
            apply_plot_theme(pw, mode)
        self.cl_agent.setPen(pg.mkPen(c["accent"], width=2))
        self.nv_curve.setPen(pg.mkPen(c["accent2"], width=2))
        self.disc_a.setBrush(pg.mkColor(c["accent"]))

    def retranslate(self):
        self.gb_stats.setTitle(tr("replay_stats"))
        self.gb_feed.setTitle(tr("replay_feed"))
        self.gb_cl.setTitle(tr("replay_clusters"))
        self.gb_nv.setTitle(tr("replay_novelty"))
        self.chk_agent.setText(tr("show_agent"))
        self.chk_r0.setText(tr("show_r0"))
        self.play_btn.setText(tr("pause_btn") if self._play_timer.isActive()
                              else tr("play_btn"))
        self.lbl_explain.setText(tr("explain_replay"))
        self.apply_btn.setText(tr("replay_load"))
        self.scatter.setLabel("bottom", tr("behav1"))
        self.scatter.setLabel("left", tr("behav2"))
