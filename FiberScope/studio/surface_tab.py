"""3D surface tab: drape the fiber network over quad OBJ surfaces.

Custom numpy orthographic renderer (yaw/pitch rotation matrices drawn
with QPainter; no painter's-algorithm sort, depth modulates alpha for
a sense of volume).  The left panel controls the OBJ model, interior
point count, in-face pattern, spectrum preset and deformation
amplitude; recomputes are debounced by 120 ms.

Structure link: apply_structure(factory) consumes the structure tab's
StructureFactory (unit + reference-edge displacement spectrum) and
re-drapes the surface with it; a non-square unit falls back to square
edge fibers while keeping the spectrum shape.  A small non-interactive
"before" canvas (zero spectrum) overlays the main canvas for direct
comparison and mirrors the main view rotation.
"""
import os
import sys
import time

import numpy as np
from PySide6.QtCore import QLineF, QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QGroupBox,
                               QHBoxLayout, QLabel, QPushButton, QSlider,
                               QSpinBox, QVBoxLayout, QWidget)

from fslab.structure import SPECTRUM_PRESETS, all_unit_keys, unit_display
from fslab.surface_mapping import MappingConfig, front_basis, load_obj
from fslab import surface3d
from .i18n import tr, get_lang
from .theme import colors

OBJ_NAMES = ['3_Lung_quad_500.obj', '4_Heart_quad_500.obj', '6_vans_500.obj',
             'reference_shirt.obj', 'reference_shoe.obj', 'reference_paris.obj', 'demo_pyramid.obj']


def _obj_dir():
    """Bundled assets/obj first (frozen or repo); dev fallback last."""
    if getattr(sys, 'frozen', False):
        base = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cand = os.path.join(base, 'assets', 'obj')
    if os.path.isdir(cand):
        return cand
    return cand      # bundled assets missing: the tab shows a hint


OBJ_DIR = _obj_dir()
OBJ_FILES = [os.path.join(OBJ_DIR, n) for n in OBJ_NAMES]

UNIT_ITEMS = all_unit_keys()
PRESET_KEYS = ['square', 'auxetic_bow', 'rhombic_bow', 'swirl']
CUSTOM_KEY = '__custom__'
# tune: preset match acceptance (rms residual after best-fit scale)
MATCH_RES = 0.02
MATCH_SMAX = 1.5
BEFORE_W, BEFORE_H = 250, 190


def resample_spectrum(spec, n):
    """Linearly resample a (dx, dy) spectrum to n interior points."""
    spec = np.asarray(spec, dtype=float).reshape(-1, 2)
    if spec.shape[0] == 0:
        return np.zeros((n, 2))
    if spec.shape[0] == n:
        return spec.copy()
    xs = np.linspace(0.0, 1.0, spec.shape[0])
    xt = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(xt, xs, spec[:, 0]),
                     np.interp(xt, xs, spec[:, 1])], axis=1)


class SurfaceCanvas(QWidget):
    """Orthographic 3D viewer: drag rotates, wheel zooms, dblclick resets."""

    YAW0, PITCH0, ZOOM0 = 0.0, 0.0, 1.0

    def __init__(self, mode='dark', parent=None):
        super().__init__(parent)
        self.mode = mode
        self.center = np.zeros(3)
        self.radius = 1.0
        self.Vc = None            # centered mesh vertices
        self.Pc = None            # centered fiber points
        self.edge_idx = None      # (E,2) unique mesh edges
        self.segs = None          # (k,2) fiber segments
        self.show_nodes = False
        self.show_mesh = False
        self.basis = np.eye(3)
        self.yaw, self.pitch, self.zoom = self.YAW0, self.PITCH0, self.ZOOM0
        self._rm = None           # rotated mesh vertices
        self._rf = None           # rotated fiber points
        self._drag = None
        self._view_listeners = []
        self.setMinimumSize(320, 320)
        self.setFocusPolicy(Qt.StrongFocus)

    # ---------------- data ----------------
    def set_mesh(self, V, F):
        V = np.asarray(V, dtype=float)
        self.center = V.mean(axis=0)
        self.basis = front_basis(V)
        self.yaw, self.pitch = self.YAW0, self.PITCH0
        self.Vc = V - self.center
        r = float(np.max(np.linalg.norm(self.Vc, axis=1)))
        self.radius = r if r > 1e-12 else 1.0
        F4 = np.asarray(F, dtype=int)
        eds = np.concatenate([F4[:, [0, 1]], F4[:, [1, 2]],
                              F4[:, [2, 3]], F4[:, [3, 0]]], axis=0)
        self.edge_idx = np.unique(np.sort(eds, axis=1), axis=0)
        self._rotate()

    def set_fibers(self, P, segs):
        self.Pc = np.asarray(P, dtype=float) - self.center
        self.segs = np.asarray(segs, dtype=np.int64)
        self._rotate()
        self.update()

    def set_show_nodes(self, on):
        self.show_nodes = bool(on)
        self.update()

    def add_view_listener(self, fn):
        """fn(canvas) after user-driven yaw/pitch/zoom changes."""
        self._view_listeners.append(fn)

    def _emit_view(self):
        for fn in list(self._view_listeners):
            fn(self)

    # ---------------- view ----------------
    def _rotate(self):
        cy, sy = np.cos(np.radians(self.yaw)), np.sin(np.radians(self.yaw))
        cp, sp = np.cos(np.radians(self.pitch)), np.sin(np.radians(self.pitch))
        Rz = np.array([[cy, 0., sy], [0., 1., 0.], [-sy, 0., cy]])
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
        rot = Rx @ Rz @ self.basis
        if self.Vc is not None:
            self._rm = self.Vc @ rot.T
        if self.Pc is not None:
            self._rf = self.Pc @ rot.T

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag = e.position().toPoint()

    def mouseMoveEvent(self, e):
        if self._drag is None:
            return
        pos = e.position().toPoint()
        dx, dy = pos.x() - self._drag.x(), pos.y() - self._drag.y()
        self._drag = pos
        self.yaw += dx * 0.4
        self.pitch = float(np.clip(self.pitch + dy * 0.4, -89.0, 89.0))
        self._rotate()
        self.update()
        self._emit_view()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag = None

    def wheelEvent(self, e):
        self.zoom = float(np.clip(self.zoom * 1.0015 ** e.angleDelta().y(),
                                  0.05, 40.0))
        self.update()
        self._emit_view()

    def mouseDoubleClickEvent(self, e):
        self.reset_view()

    def reset_view(self):
        self.yaw, self.pitch, self.zoom = self.YAW0, self.PITCH0, self.ZOOM0
        self._rotate()
        self.update()
        self._emit_view()

    # ---------------- painting ----------------
    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c = colors(self.mode)
        p.fillRect(self.rect(), QColor(c['bg']))
        if self.Pc is None or len(self.Pc) == 0 or self._rf is None:
            p.end()
            return
        w, h = self.width(), self.height()
        s = self.zoom * 0.45 * min(w, h) / self.radius
        cx, cy = w / 2.0, h / 2.0

        # mesh wireframe (faint)
        if self.show_mesh and self._rm is not None and self.edge_idx is not None \
                and len(self.edge_idx):
            xm = self._rm[:, 0] * s + cx
            ym = cy - self._rm[:, 1] * s
            lines = [QLineF(xm[a], ym[a], xm[b], ym[b])
                     for a, b in self.edge_idx]
            pen = QPen(QColor(c['faint']))
            pen.color().setAlpha(85 if self.mode == 'dark' else 130)
            pen.setWidthF(1.0)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.drawLines(lines)

        # fiber segments, depth -> alpha buckets
        rf = self._rf
        xf = rf[:, 0] * s + cx
        yf = cy - rf[:, 1] * s
        segs = self.segs
        dz = 0.5 * (rf[segs[:, 0], 2] + rf[segs[:, 1], 2])
        zmin, zmax = float(dz.min()), float(dz.max())
        span = max(zmax - zmin, 1e-9)
        nb = 10
        ks = np.clip(((dz - zmin) / span * nb).astype(int), 0, nb - 1)
        alphas = np.linspace(55, 235, nb)
        acc = QColor(c['accent2'])
        for bkt in range(nb):
            sel = np.flatnonzero(ks == bkt)
            if len(sel) == 0:
                continue
            pen = QPen(QColor(acc.red(), acc.green(), acc.blue(),
                              int(alphas[bkt])))
            pen.setWidthF(1.6)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.drawLines([QLineF(xf[a], yf[a], xf[b], yf[b])
                         for a, b in segs[sel]])

        # fiber nodes
        if self.show_nodes:
            pen = QPen(QColor(c['accent']))
            pen.color().setAlpha(150)
            pen.setWidthF(2.0)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.drawPoints(QPolygonF([QPointF(x, y)
                                    for x, y in zip(xf, yf)]))
        p.end()


class SurfaceTab(QWidget):
    def __init__(self, mode='dark', parent=None):
        super().__init__(parent)
        self.mode = mode
        self._mesh_cache = {}
        self._mesh_settings = {}
        self._edge_cache = {}
        self._cur_path = None
        self._obj_files = list(OBJ_FILES)
        self.last_ms = 0.0
        self._follow_spec = None      # raw (n,2) spectrum from structure
        self._follow_unit = 'square'  # resolved surface unit
        self._follow_hint = None      # preset key if spectrum came from one
        self._custom_spec = None      # spectrum behind the custom item
        self._syncing = False         # guard: programmatic combo updates
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(120)
        self._debounce.timeout.connect(self._recompute)
        self._build()
        self._style_overlay()
        self.retranslate()
        self._recompute()

    # ---------------- UI ----------------
    def _build(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        left = QWidget()
        left.setFixedWidth(280)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(10)

        self.gb = QGroupBox()
        gv = QVBoxLayout(self.gb)
        gv.setSpacing(10)

        self.lbl_obj = QLabel()
        self.obj_combo = QComboBox()
        for path in self._obj_files:
            self.obj_combo.addItem(
                os.path.splitext(os.path.basename(path))[0])
        gv.addWidget(self.lbl_obj)
        gv.addWidget(self.obj_combo)
        objrow = QHBoxLayout()
        self.import_btn = QPushButton()
        self.reset_btn = QPushButton()
        self.import_btn.clicked.connect(self._import_obj)
        self.reset_btn.clicked.connect(self._reset_view)
        objrow.addWidget(self.import_btn)
        objrow.addWidget(self.reset_btn)
        self.mesh_btn = QPushButton()
        self.mesh_btn.clicked.connect(self._configure_mesh)
        gv.addWidget(self.mesh_btn)
        gv.addLayout(objrow)

        row = QHBoxLayout()
        self.lbl_pts = QLabel()
        self.spin_pts = QSpinBox()
        self.spin_pts.setRange(1, 24)
        self.spin_pts.setValue(5)
        row.addWidget(self.lbl_pts)
        row.addWidget(self.spin_pts)
        gv.addLayout(row)

        self.lbl_unit = QLabel()
        self.unit_combo = QComboBox()
        for key in UNIT_ITEMS:
            self.unit_combo.addItem(unit_display(key, get_lang()), key)
        gv.addWidget(self.lbl_unit)
        gv.addWidget(self.unit_combo)

        self.lbl_preset = QLabel()
        self.preset_combo = QComboBox()
        for key in PRESET_KEYS:
            self.preset_combo.addItem(tr("surf_p_" + key), key)
        gv.addWidget(self.lbl_preset)
        gv.addWidget(self.preset_combo)

        row2 = QHBoxLayout()
        self.lbl_amp = QLabel()
        self.amp_slider = QSlider(Qt.Horizontal)
        self.amp_slider.setRange(0, 500)
        self.amp_slider.setValue(100)
        self.amp_val = QLabel('100%')
        self.amp_val.setObjectName('chip')
        row2.addWidget(self.lbl_amp)
        row2.addWidget(self.amp_slider, 1)
        row2.addWidget(self.amp_val)
        gv.addLayout(row2)

        self.chk_follow = QCheckBox()
        self.chk_follow.setChecked(True)
        gv.addWidget(self.chk_follow)

        self.chk_nodes = QCheckBox()
        self.chk_nodes.setChecked(False)
        gv.addWidget(self.chk_nodes)
        self.overscale = QSpinBox()
        self.overscale.setRange(101, 130)
        self.overscale.setValue(106)
        self.overscale.setSuffix('%')
        self.lbl_overlap = QLabel()
        overlap_row = QHBoxLayout()
        overlap_row.addWidget(self.lbl_overlap)
        overlap_row.addWidget(self.overscale)
        gv.addLayout(overlap_row)
        self.overscale.valueChanged.connect(self._schedule)
        self.view_combo = QComboBox()
        for key in ('front', 'back', 'left', 'top', 'iso'):
            self.view_combo.addItem(key, key)
        self.view_combo.currentIndexChanged.connect(self._choose_view)
        gv.addWidget(self.view_combo)
        self.manufacturing_btn = QPushButton()
        self.manufacturing_btn.clicked.connect(self._open_manufacturing)
        self.export_network_btn = QPushButton()
        self.export_network_btn.clicked.connect(self._export_network)
        gv.addWidget(self.export_network_btn)
        gv.addWidget(self.manufacturing_btn)

        self.status = QLabel('')
        self.status.setObjectName('hint')
        self.status.setWordWrap(True)
        gv.addWidget(self.status)

        lv.addWidget(self.gb)
        lv.addStretch(1)

        # main canvas with floating before/after overlays
        self.canvas = SurfaceCanvas(self.mode)
        ov = QVBoxLayout(self.canvas)
        ov.setContentsMargins(8, 8, 8, 8)
        ov.setSpacing(0)
        top = QHBoxLayout()
        top.addStretch(1)
        self.before_panel = QWidget(self.canvas)
        self.before_panel.setObjectName('before_panel')
        bv = QVBoxLayout(self.before_panel)
        bv.setContentsMargins(6, 4, 6, 6)
        bv.setSpacing(2)
        self.lbl_before = QLabel()
        self.lbl_before.setObjectName('before_lbl')
        self.before_canvas = SurfaceCanvas(self.mode, self.canvas)
        self.before_canvas.setMinimumSize(0, 0)
        self.before_canvas.setFixedSize(BEFORE_W, BEFORE_H)
        self.before_canvas.setAttribute(Qt.WA_TransparentForMouseEvents)
        bv.addWidget(self.lbl_before, 0, Qt.AlignHCenter)
        bv.addWidget(self.before_canvas)
        top.addWidget(self.before_panel)
        ov.addLayout(top)
        ov.addStretch(1)
        bot = QHBoxLayout()
        self.after_chip = QLabel(self.canvas)
        self.after_chip.setObjectName('after_chip')
        bot.addWidget(self.after_chip)
        bot.addStretch(1)
        ov.addLayout(bot)

        outer.addWidget(left)
        outer.addWidget(self.canvas, 1)

        self.obj_combo.currentIndexChanged.connect(self._schedule)
        self.spin_pts.valueChanged.connect(self._schedule)
        self.unit_combo.currentIndexChanged.connect(self._on_unit_changed)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        self.amp_slider.valueChanged.connect(self._on_amp)
        self.chk_follow.toggled.connect(self._on_follow_toggled)
        self.chk_nodes.toggled.connect(self._on_nodes_toggled)
        self.canvas.add_view_listener(self._sync_before_view)

    def _style_overlay(self):
        c = colors(self.mode)
        self.before_panel.setStyleSheet(
            '#before_panel { background: %s; border: 1px solid %s;'
            ' border-radius: 8px; }' % (c['panel'], c['line']))
        self.lbl_before.setStyleSheet(
            '#before_lbl { color: %s; background: transparent;'
            ' border: none; }' % c['faint'])
        self.after_chip.setStyleSheet(
            '#after_chip { color: %s; background: %s;'
            ' border: 1px solid %s; border-radius: 6px; padding: 2px 8px; }'
            % (c['faint'], c['panel'], c['line']))

    def retranslate(self):
        was_syncing = self._syncing
        self._syncing = True
        zh = get_lang() == 'zh'
        names = [('肺', 'Lung'), ('心脏', 'Heart'), ('运动鞋', 'Sneaker'),
                 ('衣服轮廓', 'Shirt envelope'), ('鞋轮廓', 'Shoe envelope'), ('铁塔轮廓', 'Tower envelope'),
                 ('金字塔 · 打印演示', 'Pyramid · print demo')]
        for i, pair in enumerate(names):
            self.obj_combo.setItemText(i, pair[0 if zh else 1])
        self.lbl_overlap.setText('映射覆盖比例' if zh else 'Patch coverage')
        self.manufacturing_btn.setText('曲面连续制造 · 3D' if zh else 'Surface fabrication · 3D')
        self.export_network_btn.setText('导出纤维网络 OBJ' if zh else 'Export fiber OBJ')
        for i, names in enumerate((('正视', 'Front'), ('背面', 'Back'), ('侧面', 'Side'), ('顶部', 'Top'), ('立体', 'Isometric'))):
            self.view_combo.setItemText(i, names[0 if zh else 1])
        for i in range(self.unit_combo.count()):
            self.unit_combo.setItemText(i, unit_display(self.unit_combo.itemData(i), get_lang()))
        self.gb.setTitle(tr("surf_group"))
        self.lbl_obj.setText(tr("surf_obj"))
        self.import_btn.setText(tr("surf_import"))
        self.mesh_btn.setText('网格设置…' if zh else 'Mesh settings…')
        self.import_btn.setToolTip('三角面自动转为四边面；高密度三角网格自动减面，原文件不变。' if zh else 'Triangles convert to quads; dense triangle meshes are reduced. Source files are preserved.')
        self.reset_btn.setText(tr("surf_reset"))
        self.lbl_pts.setText(tr("surf_pts"))
        self.lbl_unit.setText(tr("surf_unit"))
        self.lbl_preset.setText(tr("surf_preset"))
        for i in range(self.preset_combo.count()):
            key = self.preset_combo.itemData(i)
            if key == CUSTOM_KEY:
                self.preset_combo.setItemText(i, tr("surf_custom"))
            elif key in PRESET_KEYS:
                self.preset_combo.setItemText(i, tr("surf_p_" + key))
        self.lbl_amp.setText(tr("surf_amp"))
        self.chk_follow.setText(tr("surf_follow"))
        self.chk_nodes.setText(tr("surf_nodes"))
        self.lbl_before.setText(tr("surf_before"))
        self.after_chip.setText(tr("surf_after"))
        self._syncing = was_syncing

    def set_mode(self, mode):
        self.mode = mode
        self.canvas.mode = mode
        self.before_canvas.mode = mode
        self._style_overlay()
        self.canvas.update()
        self.before_canvas.update()

    # ---------------- public API ----------------
    def apply_structure(self, factory):
        """Re-drape from a StructureFactory (structure_changed payload).

        Uses the factory's unit and reference-edge spectrum
        (line_displacements or spectrum()), resampled to spin_pts at
        recompute time.  Units not supported by surface3d fall back to
        square edge fibers but keep the spectrum shape.  UI combos are
        synced only while "follow current structure" is checked.
        """
        if factory is None:
            return
        unit = getattr(factory, 'unit', 'square') or 'square'
        surf_unit = 'square' if unit in SPECTRUM_PRESETS else unit
        if self.unit_combo.findData(surf_unit) < 0:
            self.unit_combo.addItem(unit_display(surf_unit, get_lang()), surf_unit)
        spec = None
        ld = getattr(factory, 'line_displacements', None)
        if ld:
            spec = np.asarray(ld, dtype=float).reshape(-1, 2)
        else:
            fn = getattr(factory, 'spectrum', None)
            try:
                s = fn() if callable(fn) else None
            except Exception:
                s = None
            if s is not None:
                s = np.asarray(s, dtype=float).reshape(-1, 2)
                if len(s):
                    spec = s
        if spec is None:
            spec = np.zeros((max(1, self.spin_pts.value()), 2))
        if factory.topology == 'topnet26':
            spec = factory.effective_spectrum()
        # deterministic hint: P1 units ARE spectrum presets on square
        preset_hint = unit if (not ld and unit in SPECTRUM_PRESETS) else None
        self._follow_unit = surf_unit
        self._follow_spec = spec
        self._follow_hint = preset_hint
        if self.chk_follow.isChecked():
            self._syncing = True
            try:
                self.unit_combo.setCurrentIndex(
                    self.unit_combo.findData(surf_unit))
                self._sync_preset_for(spec, preset_hint)
                self.amp_slider.setValue(100)
                self.spin_pts.setValue(max(1,int(factory.n_pts_per_side)))
            finally:
                self._syncing = False
            self._schedule()

    # ---------------- events ----------------
    def _on_follow_toggled(self, on):
        if on and self._follow_spec is not None:
            self._syncing = True
            try:
                self.unit_combo.setCurrentIndex(
                    self.unit_combo.findData(self._follow_unit))
                self._sync_preset_for(self._follow_spec, self._follow_hint)
                self.amp_slider.setValue(100)
            finally:
                self._syncing = False
        elif on:
            self.status.setText(tr('surf_follow_on'))
        self._schedule()

    def _on_unit_changed(self, idx):
        if not self._syncing and self.chk_follow.isChecked():
            self.chk_follow.setChecked(False)
        self._schedule()

    def _on_preset_changed(self, idx):
        if not self._syncing and self.chk_follow.isChecked():
            self.chk_follow.setChecked(False)
        self._schedule()

    def _on_nodes_toggled(self, on):
        self.canvas.set_show_nodes(on)
        self.before_canvas.set_show_nodes(on)

    def _sync_before_view(self, src):
        bc = self.before_canvas
        bc.yaw, bc.pitch, bc.zoom = src.yaw, src.pitch, src.zoom
        bc._rotate()
        bc.update()

    # ---------------- preset matching ----------------
    def _ensure_custom_item(self):
        for i in range(self.preset_combo.count()):
            if self.preset_combo.itemData(i) == CUSTOM_KEY:
                return i
        self.preset_combo.addItem(tr('surf_custom'), CUSTOM_KEY)
        return self.preset_combo.count() - 1

    def _match_preset(self, spec):
        """Best-fit (key, rms_residual, scale) of spec onto the presets."""
        a0 = np.asarray(spec, dtype=float).reshape(-1, 2)
        m = max(6, a0.shape[0])
        # interior-node parameterization, same as fslab.structure
        def _rs(x, k):
            x = np.asarray(x, dtype=float).reshape(-1, 2)
            if x.shape[0] == k:
                return x
            ts = (np.arange(x.shape[0]) + 1.0) / (x.shape[0] + 1.0)
            tt = (np.arange(k) + 1.0) / (k + 1.0)
            return np.stack([np.interp(tt, ts, x[:, 0]),
                             np.interp(tt, ts, x[:, 1])], axis=1)
        a = _rs(a0, m)
        best = (None, float('inf'), 0.0)
        for key in PRESET_KEYS:
            b = _rs(SPECTRUM_PRESETS[key], m)
            bb = float((b * b).sum())
            s = float((a * b).sum()) / bb if bb > 1e-12 else 0.0
            res = float(np.sqrt(((a - s * b) ** 2).sum() / m))
            if res < best[1]:
                best = (key, res, s)
        return best

    def _sync_preset_for(self, spec, hint=None):
        if hint is not None:
            for i in range(self.preset_combo.count()):
                if self.preset_combo.itemData(i) == hint:
                    self.preset_combo.setCurrentIndex(i)
                    break
            self._custom_spec = None
            return
        key, res, s = self._match_preset(spec)
        matched = key is not None and res < MATCH_RES \
            and 0.0 <= s <= MATCH_SMAX
        if matched:
            for i in range(self.preset_combo.count()):
                if self.preset_combo.itemData(i) == key:
                    self.preset_combo.setCurrentIndex(i)
                    break
            self._custom_spec = None
            if key != 'square' and s > 1e-6:
                self.amp_slider.setValue(int(round(s * 100)))
        else:
            ci = self._ensure_custom_item()
            self.preset_combo.setCurrentIndex(ci)
            self._custom_spec = np.asarray(spec, dtype=float).copy()

    def _import_obj(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr('surf_import'), OBJ_DIR, 'OBJ (*.obj)')
        if not path:
            return
        path = os.path.abspath(path)
        if path in self._obj_files:
            self.obj_combo.setCurrentIndex(self._obj_files.index(path))
            return
        self._obj_files.append(path)
        self.obj_combo.addItem(os.path.splitext(os.path.basename(path))[0])
        self.obj_combo.setCurrentIndex(self.obj_combo.count() - 1)
        self._schedule()

    def _configure_mesh(self):
        from .mesh_dialog import MeshDialog
        path = self._obj_files[self.obj_combo.currentIndex()]
        dialog = MeshDialog(*self._mesh_settings.get(path,(1500,0)),parent=self)
        if dialog.exec():
            self._mesh_settings[path] = dialog.values()
            self._mesh_cache.pop(path,None)
            self._edge_cache.pop(path,None)
            self._cur_path = None
            self._schedule()

    def _reset_view(self):
        self.canvas.reset_view()
        self.before_canvas.reset_view()

    # ---------------- logic ----------------
    def _on_amp(self, v):
        self.amp_val.setText(f'{v}%')
        self._schedule()

    def _schedule(self, *_):
        self._debounce.start()

    def _choose_view(self):
        angles = {'front': (0., 0.), 'back': (180., 0.), 'left': (90., 0.), 'top': (0., 90.), 'iso': (-35., 25.)}
        self.canvas.yaw, self.canvas.pitch = angles[self.view_combo.currentData()]
        self.canvas._rotate()
        self.canvas.update()
        self.canvas._emit_view()

    def _open_manufacturing(self):
        if hasattr(self, "manufacturing_host"):
            return self.manufacturing_host.open_source(True)
        dialog = getattr(self, 'manufacturing_dialog', None)
        if dialog is not None and (dialog.isVisible() or (dialog.worker is not None and dialog.worker.isRunning())):
            dialog.show()
            dialog.raise_()
            return
        if not hasattr(self, '_manufacturing_source') or self._mapping_error:
            return
        from .manufacturing_dialog import ManufacturingDialog
        vertices, faces, factory, overscale = self._manufacturing_source
        self.manufacturing_dialog = ManufacturingDialog(factory, (vertices, faces, overscale), self.mode, self,
                                                         network=self.manufacturing_network)
        self.manufacturing_dialog.show()

    def _export_network(self):
        if not hasattr(self, 'network'):
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Export fiber network', 'fibers.obj', 'OBJ (*.obj)')
        if path:
            points, edges = self.network
            try:
                with open(path, 'w', encoding='utf-8') as stream:
                    stream.write('# Mapped fibers and seam stitches; no support mesh\n')
                    for p in points:
                        stream.write('v %.9g %.9g %.9g\n' % tuple(p))
                    for a, b in edges:
                        stream.write('l %d %d\n' % (a+1, b+1))
            except OSError as exc:
                self.status.setText(str(exc))

    def _recompute(self):
        self._mapping_error = None
        try:
            self._compute_mapping()
        except (ValueError, MemoryError, OSError) as exc:
            self._mapping_error = str(exc)
            self.export_network_btn.setEnabled(False)
            self.status.setText(str(exc))

    def _compute_mapping(self):
        idx = self.obj_combo.currentIndex()
        if idx < 0 or idx >= len(self._obj_files):
            return
        path = self._obj_files[idx]
        if not os.path.exists(path):
            self.status.setText('OBJ missing: %s' % path)
            return
        if path not in self._mesh_cache:
            target, levels = self._mesh_settings.get(path,(1500,0))
            v,f,info = load_obj(path, return_info=True, target_faces=target)
            if levels:
                from fslab.surface_mapping import subdivide_quads
                v,f = subdivide_quads(v,f,levels)
                info['quad_faces'] = len(f)
            self._mesh_cache[path] = v,f,info
        V, F, info = self._mesh_cache[path]
        n = self.spin_pts.value()
        amp = self.amp_slider.value() / 100.0
        follow_on = (self.chk_follow.isChecked()
                     and self._follow_spec is not None)
        if follow_on:
            spec = resample_spectrum(self._follow_spec, n) * amp
            unit = self._follow_unit
            src = tr('surf_src_struct')
        else:
            key = self.preset_combo.currentData()
            if key == CUSTOM_KEY and self._custom_spec is not None:
                base = self._custom_spec
            elif key in SPECTRUM_PRESETS:
                base = SPECTRUM_PRESETS[key]
            else:
                base = SPECTRUM_PRESETS['square']
            spec = resample_spectrum(base, n) * amp
            unit = self.unit_combo.currentData()
            src = tr('surf_src_preset')
        t0 = time.perf_counter()
        from fslab.structure import StructureFactory
        self._manufacturing_source = (V, F, StructureFactory(unit=unit, n_pts_per_side=n, line_displacements=np.asarray(spec).tolist()), self.overscale.value()/100.)
        config = MappingConfig(overscale=self.overscale.value()/100.)
        from dataclasses import replace
        from fslab.manufacturing import compile_surface
        factory = self._manufacturing_source[2]
        mapped = compile_surface(V, F, factory, config.overscale)
        reference = compile_surface(V, F, replace(factory, line_displacements=np.zeros((n,2)).tolist()), config.overscale)
        P, segs = mapped.positions, mapped.edges
        P0, segs0 = reference.positions, reference.edges
        config.mapped_faces = mapped.mapped_faces
        config.source_faces = len(F)
        self.network = P, segs
        mapped.up_axis = 'z' if os.path.basename(path)=='demo_pyramid.obj' else 'y'
        self.manufacturing_network = mapped
        self.export_network_btn.setEnabled(True)
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        if path != self._cur_path:
            self.canvas.set_mesh(V, F)
            self.before_canvas.set_mesh(V, F)
            if os.path.basename(path)=='demo_pyramid.obj':
                self.canvas.basis = np.array([[1.,0,0],[0,0,1.],[0,-1.,0]])
                self.before_canvas.basis = self.canvas.basis.copy()
                self.canvas.yaw = self.before_canvas.yaw = -25.
                self.canvas.pitch = self.before_canvas.pitch = 15.
            if os.path.basename(path).startswith('real_'):
                basis = np.eye(3)
                self.canvas.basis = basis
                self.before_canvas.basis = basis.copy()
            self.view_combo.blockSignals(True)
            self.view_combo.setCurrentIndex(0)
            self.view_combo.blockSignals(False)
            self._cur_path = path
        self.canvas.set_fibers(P, segs)
        self.before_canvas.set_fibers(P0, segs0)
        if path not in self._edge_cache:
            self._edge_cache[path] = len(surface3d.build_edge_faces(F))
        self.status.setText(
            tr('surf_stat') % (len(V), self._edge_cache[path], len(segs),
                               self.last_ms, src))
        if info['converted']:
            note = ('自动转换：%d 个原始面 → %d 个四边面' if get_lang() == 'zh' else 'Converted: %d source faces → %d quads')
            self.status.setText(self.status.text() + '\n' + note % (info['source_faces'], info['quad_faces']))
        if config.source_faces != config.mapped_faces:
            note = ('为适配复杂单元，展示曲面降至 %d 个面' if get_lang() == 'zh' else 'Display surface reduced to %d patches for this cell')
            self.status.setText(self.status.text() + '\n' + note % config.mapped_faces)
