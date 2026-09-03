'''Feature analysis dashboard: card grid of scalar tiles and mini
histograms, single snapshot or seeded batch mode, region selection.

All metrics live in fslab.features; this file is presentation only.
Batch computation runs in a QThread worker with progress reporting.
'''
import time

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, Qt, QThread, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog,
                               QDoubleSpinBox, QFrame, QGridLayout,
                               QHBoxLayout, QLabel, QProgressBar,
                               QPushButton, QScrollArea, QSpinBox,
                               QVBoxLayout, QWidget)

from fslab.structure import CELL, StructureFactory
from fslab.features import (FEATURE_GROUPS, FEATURE_INT, FEATURE_RANGE,
                            FEATURE_UNIT, FEATURE_ZH, HIST_KEYS,
                            compute_features)
from .i18n import get_lang
from .structure_tab import spec_text
from .theme import colors

NCOLS = 4
GROUP_ORDER = ('structure', 'pore', 'contact')
GROUP_COLOR = {'structure': 'accent', 'pore': 'violet', 'contact': 'warn'}
GROUP_NAME = {'structure': ('结构', 'Structure'),
              'pore': ('孔隙', 'Pore'),
              'contact': ('接触', 'Contact')}

_TEXT = {
    'region': ('区域', 'Region'),
    'batch': ('批量统计', 'Batch'),
    'samples': ('样本数', 'Samples'),
    'refresh': ('刷新', 'Refresh'),
    'no_struct': ('暂无结构 · 先在「结构生成」板块生成', 'No structure yet'),
    'computing': ('批量计算中…', 'Batch running…'),
    'done_single': ('单样本 · %d 特征 · %.0f ms',
                    'Single · %d features · %.0f ms'),
    'done_batch': ('批量 N=%d · seeds 0..%d · %.1f s',
                   'Batch N=%d · seeds 0..%d · %.1f s'),
}


def _t(key):
    zh, en = _TEXT[key]
    return zh if get_lang() == 'zh' else en


def _fmt(v):
    if v is None:
        return '-'
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    v = float(v)
    a = abs(v)
    if a >= 1000:
        return '%.0f' % v
    if a >= 1 or v == 0.0:
        return '%.3f' % v
    return '%.4g' % v


def _hist_bins(key, data):
    '''(counts, edges) for a histogram card, or None when empty.'''
    if data is None:
        return None
    arr = np.asarray(data, float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    if key in FEATURE_INT:
        lo, hi = int(arr.min()), int(arr.max())
        edges = np.arange(lo, hi + 2) - 0.5
        counts, _ = np.histogram(arr, edges)
        return counts.astype(float), edges
    rng = FEATURE_RANGE.get(key)
    if rng is not None:
        nb = 18 if key == 'orientations' else 14
        counts, edges = np.histogram(arr, bins=nb, range=rng)
        return counts.astype(float), edges
    nb = int(min(28, max(6, 2 * np.sqrt(arr.size))))
    counts, edges = np.histogram(arr, bins=nb)
    return counts.astype(float), edges


RADAR_KEYS = [
    ('orient_entropy', '取向熵', 'orient', 4.2),
    ('degree_entropy', '度熵', 'degree', 3.2),
    ('anisotropy', '各向异', 'aniso', 1.0),
    ('mean_edge_len', '边长', 'edge', CELL),
    ('porosity', '孔隙率', 'pore', 1.0),
    ('straightness_mean', '直线度', 'straight', 1.0),
    ('cross_per_edge', '交叉密', 'cross', 1.0),
    ('boundary_ratio', '边界比', 'boundary', 1.0),
]


class FingerprintWidget(QWidget):
    """Radar fingerprint: eight normalized descriptors of the current net."""

    def __init__(self, mode, parent=None):
        super().__init__(parent)
        self.mode = mode
        self.values = None
        self.pos = None
        self.edges = None
        self.setMinimumSize(360, 180)

    def set_features(self, feats, pos=None, edges=None):
        self.values = feats
        if pos is not None and len(pos):
            self.pos = np.asarray(pos, float)[:, :2]
            self.edges = np.asarray(edges, int)[:, :2]
        else:
            self.pos = None
            self.edges = None
        self.update()

    def _norm(self, key, scale):
        if not self.values:
            return 0.0
        v = self.values.get(key, 0.0)
        if v is None or not np.isfinite(float(v)):
            return 0.0
        return float(np.clip(abs(float(v)) / max(scale, 1e-9), 0.0, 1.0))

    def _draw_structure(self, p, c, x, y, w, h):
        p.setPen(QPen(QColor(c['line']), 1.0))
        p.drawRect(x, y, max(w, 1), max(h, 1))
        if self.pos is None or len(self.pos) == 0:
            return
        pos = self.pos
        lo = pos.min(0)
        hi = pos.max(0)
        span = np.maximum(hi - lo, 1e-9)
        scale = min(w, h) / max(span[0], span[1]) * 0.9
        px = x + (w - span[0] * scale) / 2.0
        py = y + (h - span[1] * scale) / 2.0
        p.setPen(QPen(QColor(c['accent2']), 1.1))
        for a, b in self.edges:
            p.drawLine(QPointF(px + (pos[a, 0] - lo[0]) * scale,
                               py + (pos[a, 1] - lo[1]) * scale),
                       QPointF(px + (pos[b, 0] - lo[0]) * scale,
                               py + (pos[b, 1] - lo[1]) * scale))

    def paintEvent(self, ev):
        c = colors(self.mode)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QColor(c['card2']))
        w = self.width()
        h = self.height()
        # left: small structure snapshot (same scale as the radar panel)
        left_w = int(w * 0.42)
        self._draw_structure(p, c, 8, 8, left_w - 10, h - 16)
        # right: radar fingerprint
        cx = left_w + (w - left_w) / 2.0
        cy = h / 2.0
        r = 0.34 * min(w - left_w, h)
        n = len(RADAR_KEYS)
        ang = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n, endpoint=False)

        def pt(i, radius):
            return (cx + radius * np.cos(ang[i]), cy - radius * np.sin(ang[i]))
        # grid rings
        for frac in (0.25, 0.5, 0.75, 1.0):
            poly = [QPointF(*pt(i, r * frac)) for i in range(n)]
            p.setPen(QPen(QColor(c['line']), 1.0))
            p.drawPolygon(poly)
        # spokes
        for i in range(n):
            x, y = pt(i, r)
            p.setPen(QPen(QColor(c['line']), 1.0))
            p.drawLine(QPointF(cx, cy), QPointF(x, y))
        # data polygon
        if self.values:
            poly = [QPointF(*pt(i, r * self._norm(key, scale)))
                    for i, (key, _, _, scale) in enumerate(RADAR_KEYS)]
            p.setPen(QPen(QColor(c['accent']), 1.8))
            p.setBrush(QColor(c['accent']))
            p.setBrush(QColor(int(QColor(c['accent']).red()),
                              int(QColor(c['accent']).green()),
                              int(QColor(c['accent']).blue()), 70))
            p.drawPolygon(poly)
        # labels
        zh = get_lang() == 'zh'
        p.setPen(QPen(QColor(c['sub'])))
        for i, (_, z, e, _) in enumerate(RADAR_KEYS):
            x, y = pt(i, r + 18)
            p.drawText(int(x - 20), int(y + 4), 40, 16,
                       Qt.AlignHCenter, z if zh else e)
        p.end()

class FeatureCard(QFrame):
    '''One dashboard tile: scalar value or mini histogram + chips.'''

    def __init__(self, key, group, mode, parent=None):
        super().__init__(parent)
        self.key, self.group, self.mode = key, group, mode
        self.is_hist = key in HIST_KEYS
        self._single = None
        self._batch = None
        self._counts = None
        self._edges = None
        self._vline = None
        self.setObjectName('card')
        self.setCursor(Qt.PointingHandCursor)
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(5)
        head = QHBoxLayout()
        head.setSpacing(6)
        self.title = QLabel()
        self.tag = QLabel()
        head.addWidget(self.title, 1)
        head.addWidget(self.tag)
        v.addLayout(head)
        if self.is_hist:
            self.plot = pg.PlotWidget()
            self.plot.setMenuEnabled(False)
            self.plot.setMouseEnabled(False, False)
            self.plot.setFixedHeight(92)
            self.plot.hideAxis('left')
            ax = self.plot.getPlotItem().getAxis('bottom')
            ax.setStyle(showValues=False)
            ax.setHeight(8)
            v.addWidget(self.plot)
            self.hover_lbl = QLabel(' ')
            self.hover_lbl.setObjectName('hint')
            v.addWidget(self.hover_lbl)
            self.plot.scene().sigMouseMoved.connect(self._on_hover)
            self.plot.scene().sigMouseClicked.connect(self._zoom)
        else:
            self.value_lbl = QLabel('-')
            v.addWidget(self.value_lbl)
        v.addStretch(1)
        chips = QHBoxLayout()
        chips.setSpacing(5)
        self.chips = []
        for _ in range(3):
            chip = QLabel()
            chip.setObjectName('chip')
            chip.setVisible(False)
            chips.addWidget(chip)
            self.chips.append(chip)
        chips.addStretch(1)
        v.addLayout(chips)
        for w in [self.title, self.tag] + self.chips:
            w.setAttribute(Qt.WA_TransparentForMouseEvents)
        if self.is_hist:
            self.hover_lbl.setAttribute(Qt.WA_TransparentForMouseEvents)
        else:
            self.value_lbl.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._c = colors(mode)
        self.apply_theme(self._c)
        self.retranslate()

    # ---------------- theme / i18n ----------------
    def apply_theme(self, c):
        self._c = c
        gc = c[GROUP_COLOR[self.group]]
        self.title.setStyleSheet(
            'color:%s; font-weight:600; font-size:13px;' % c['text'])
        self.tag.setStyleSheet(
            'color:%s; background:%s1f; border:1px solid %s55;'
            ' border-radius:8px; padding:1px 7px; font-size:10px;'
            % (gc, gc, gc))
        for chip in self.chips:
            chip.setStyleSheet(
                'background:%s; border:1px solid %s; border-radius:8px;'
                ' padding:1px 7px; color:%s; font-size:10px;'
                % (c['card2'], c['line'], c['sub']))
        if self.is_hist:
            self.plot.setBackground(c['card2'])
            self.plot.getPlotItem().getAxis('bottom').setPen(
                pg.mkPen(c['line2']))
            self._draw(self._batch if self._batch is not None
                       else self._single)
        else:
            self.value_lbl.setStyleSheet(
                'color:%s; font-size:21px; font-weight:700;' % c['accent2'])

    def retranslate(self):
        zh = get_lang() == 'zh'
        self.title.setText(FEATURE_ZH.get(self.key, self.key)
                           if zh else self.key)
        self.tag.setText(GROUP_NAME[self.group][0 if zh else 1])

    # ---------------- data ----------------
    def set_single(self, value):
        self._single = value
        self._batch = None
        self._refresh()

    def set_batch(self, batch):
        self._batch = batch
        self._refresh()

    def clear_batch(self):
        if self._batch is not None:
            self._batch = None
            self._refresh()

    def _refresh(self):
        if self.is_hist:
            data = self._batch if self._batch is not None else self._single
            self._draw(data)
            if data is not None and np.size(data):
                a = np.asarray(data, float)
                self._set_chips(['n=%d' % a.size,
                                 'μ=%s' % _fmt(a.mean()),
                                 'σ=%s' % _fmt(a.std())])
            else:
                self._set_chips(['n=0', '', ''])
        elif self._batch is not None and np.size(self._batch):
            a = np.asarray(self._batch, float)
            self.value_lbl.setText(_fmt(float(a.mean())))
            self._set_chips(['±%s' % _fmt(a.std()),
                             'q90 %s' % _fmt(np.quantile(a, 0.9)),
                             FEATURE_UNIT.get(self.key, '')])
        else:
            self.value_lbl.setText(_fmt(self._single))
            self._set_chips([FEATURE_UNIT.get(self.key, ''), '', ''])

    def _set_chips(self, texts):
        for chip, txt in zip(self.chips, texts):
            chip.setText(txt)
            chip.setVisible(bool(txt))

    # ---------------- histogram ----------------
    def _draw(self, data):
        self.plot.clear()
        self._counts = self._edges = None
        self._vline = None
        res = _hist_bins(self.key, data)
        if res is None:
            return
        counts, edges = res
        self._counts, self._edges = counts, edges
        centers = 0.5 * (edges[1:] + edges[:-1])
        width = float((edges[1:] - edges[:-1]).min()) * 0.9
        brush = pg.mkBrush(self._c[GROUP_COLOR[self.group]])
        self.plot.addItem(pg.BarGraphItem(x=centers, height=counts,
                                          width=width, brush=brush,
                                          pen=None))
        self.plot.disableAutoRange()
        self.plot.setXRange(edges[0], edges[-1], padding=0.03)
        self.plot.setYRange(0, max(float(counts.max()), 1.0) * 1.15,
                            padding=0)

    def _on_hover(self, scene_pos):
        if self._counts is None:
            return
        vb = self.plot.getPlotItem().vb
        pt = vb.mapSceneToView(scene_pos)
        idx = int(np.searchsorted(self._edges, pt.x()) - 1)
        if 0 <= idx < self._counts.size:
            lo, hi = self._edges[idx], self._edges[idx + 1]
            self.hover_lbl.setText('[%s, %s) · n=%d'
                                   % (_fmt(lo), _fmt(hi),
                                      int(self._counts[idx])))
            if self._vline is None:
                self._vline = pg.InfiniteLine(
                    pen=pg.mkPen(self._c['faint'], style=Qt.DashLine),
                    movable=False)
                self.plot.addItem(self._vline)
            self._vline.setValue(0.5 * (lo + hi))
            self._vline.show()
        else:
            self.hover_lbl.setText(' ')
            if self._vline is not None:
                self._vline.hide()

    # ---------------- zoom ----------------
    def mousePressEvent(self, ev):
        self._zoom()

    def _zoom(self):
        FeatureDialog(self.key, self.group, self._c,
                      self._single, self._batch, self).exec()


class FeatureDialog(QDialog):
    '''Click-to-zoom: large plot (hist keys) or value + full stats.'''

    def __init__(self, key, group, c, single, batch, parent=None):
        super().__init__(parent)
        self.key = key
        zh = get_lang() == 'zh'
        name = FEATURE_ZH.get(key, key) if zh else key
        self.setWindowTitle('%s · %s' % (name, key))
        self.resize(580, 440)
        self.setStyleSheet('QDialog { background:%s; }' % c['bg'])
        v = QVBoxLayout(self)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(10)
        head = QLabel(name)
        head.setStyleSheet('color:%s; font-size:15px; font-weight:700;'
                           % c['text'])
        sub = QLabel('%s · %s' % (GROUP_NAME[group][0 if zh else 1], key))
        sub.setStyleSheet('color:%s; font-size:11px;' % c['faint'])
        v.addWidget(head)
        v.addWidget(sub)
        if key in HIST_KEYS:
            data = batch if batch is not None else single
            pw = pg.PlotWidget()
            pw.setMinimumHeight(250)
            pw.setBackground(c['card2'])
            for axname in ('left', 'bottom'):
                ax = pw.getPlotItem().getAxis(axname)
                ax.setPen(pg.mkPen(c['sub']))
                ax.setTextPen(pg.mkPen(c['sub']))
            res = _hist_bins(key, data)
            if res is not None:
                counts, edges = res
                centers = 0.5 * (edges[1:] + edges[:-1])
                width = float((edges[1:] - edges[:-1]).min()) * 0.9
                pw.addItem(pg.BarGraphItem(
                    x=centers, height=counts, width=width,
                    brush=pg.mkBrush(c[GROUP_COLOR[group]]), pen=None))
            v.addWidget(pw, 1)
        else:
            val = (float(np.mean(batch))
                   if batch is not None and np.size(batch) else single)
            big = QLabel(_fmt(val))
            big.setStyleSheet('color:%s; font-size:34px; font-weight:800;'
                              % c['accent2'])
            v.addWidget(big)
            v.addStretch(1)
        lbl = QLabel(self._stats_text(single, batch))
        lbl.setStyleSheet('color:%s; font-size:12px;' % c['sub'])
        lbl.setWordWrap(True)
        v.addWidget(lbl)

    def _stats_text(self, single, batch):
        parts = []
        if batch is not None and np.size(batch):
            a = np.asarray(batch, float)
            parts = ['N=%d' % a.size, 'mean=%s' % _fmt(a.mean()),
                     'std=%s' % _fmt(a.std()),
                     'q10=%s' % _fmt(np.quantile(a, 0.1)),
                     'q50=%s' % _fmt(np.quantile(a, 0.5)),
                     'q90=%s' % _fmt(np.quantile(a, 0.9)),
                     'min=%s' % _fmt(a.min()), 'max=%s' % _fmt(a.max())]
        elif single is not None:
            s = np.asarray(single, float)
            if s.size > 1:
                parts = ['n=%d' % s.size, 'mean=%s' % _fmt(s.mean()),
                         'std=%s' % _fmt(s.std()),
                         'min=%s' % _fmt(s.min()), 'max=%s' % _fmt(s.max())]
            else:
                parts = ['value=%s' % _fmt(single)]
        unit = FEATURE_UNIT.get(self.key, '')
        if unit:
            parts.append('unit=%s' % unit)
        return '   '.join(parts)


class BatchWorker(QThread):
    '''Seed-loop batch computation off the GUI thread.'''
    progress = Signal(int, int)
    done = Signal(list)
    failed = Signal(str)

    def __init__(self, factory, n, rect, token, parent=None):
        super().__init__(parent)
        self.factory = factory
        self.n = int(n)
        self.rect = rect
        self.token = token

    def run(self):
        try:
            results = []
            for i in range(self.n):
                f = StructureFactory(
                    unit=self.factory.unit, grid_x=3, grid_y=3,
                    n_pts_per_side=self.factory.n_pts_per_side,
                    seed=i, perturbation=0.1).clamped()
                g = f.build()
                pos = np.asarray(g.node_positions(), float)[:, :2]
                edges = np.asarray(g.edge_array(), int)[:, :2]
                results.append(compute_features(pos, edges, rect=self.rect))
                self.progress.emit(i + 1, self.n)
            self.done.emit(results)
        except Exception as e:
            self.failed.emit('%s: %s' % (e.__class__.__name__, e))


class FeaturesTab(QWidget):
    def __init__(self, mode='dark', parent=None):
        super().__init__(parent)
        self.mode = mode
        self.factory = None
        self._feats = None
        self._worker = None
        self._batch_token = 0
        self._t_batch = 0.0
        self.cards = {}
        self.fingerprint = FingerprintWidget(mode)
        self._build()
        self.retranslate()

    # ---------------- UI construction ----------------
    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        bar = QFrame()
        bar.setObjectName('card')
        bh = QHBoxLayout(bar)
        bh.setContentsMargins(12, 8, 12, 8)
        bh.setSpacing(8)
        self.chk_region = QCheckBox()
        self.chk_region.toggled.connect(self._on_region)
        bh.addWidget(self.chk_region)
        self.rect_spins = {}
        defaults = dict(x0=0.0, y0=0.0, x1=CELL, y1=CELL)
        for name in ('x0', 'y0', 'x1', 'y1'):
            lab = QLabel(name)
            lab.setObjectName('hint')
            sp = QDoubleSpinBox()
            sp.setRange(-1e3, 1e3)
            sp.setSingleStep(1.0)
            sp.setDecimals(2)
            sp.setValue(defaults[name])
            sp.setEnabled(False)
            bh.addWidget(lab)
            bh.addWidget(sp)
            self.rect_spins[name] = sp
        bh.addSpacing(6)
        self.chk_batch = QCheckBox()
        bh.addWidget(self.chk_batch)
        self.lbl_n = QLabel()
        self.lbl_n.setObjectName('hint')
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(2, 50)
        self.batch_spin.setValue(20)
        bh.addWidget(self.lbl_n)
        bh.addWidget(self.batch_spin)
        self.btn_refresh = QPushButton()
        self.btn_refresh.setProperty('primary', True)
        self.btn_refresh.clicked.connect(self.refresh)
        bh.addWidget(self.btn_refresh)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(150)
        self.progress.setTextVisible(False)
        self.progress.hide()
        bh.addWidget(self.progress)
        self.status = QLabel()
        self.status.setObjectName('hint')
        bh.addWidget(self.status, 1)
        outer.addWidget(bar)

        fp = QFrame()
        fp.setObjectName('card')
        fpv = QVBoxLayout(fp)
        fpv.setContentsMargins(10, 8, 10, 8)
        self.fp_title = QLabel()
        self.fp_title.setObjectName('subtitle')
        fpv.addWidget(self.fp_title)
        fpv.addWidget(self.fingerprint)
        outer.addWidget(fp)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        mv = QVBoxLayout(content)
        mv.setContentsMargins(2, 2, 8, 2)
        mv.setSpacing(12)
        self.group_lbls = {}
        self.grids = {}
        for g in GROUP_ORDER:
            lbl = QLabel()
            lbl.setObjectName('subtitle')
            self.group_lbls[g] = lbl
            grid = QGridLayout()
            grid.setSpacing(10)
            for ci in range(NCOLS):
                grid.setColumnStretch(ci, 1)
            self.grids[g] = grid
            mv.addWidget(lbl)
            mv.addLayout(grid)
        mv.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)
        self._scroll = scroll
        self._content = content

        for g in GROUP_ORDER:
            for i, key in enumerate(FEATURE_GROUPS[g]):
                card = FeatureCard(key, g, self.mode)
                self.cards[key] = card
                self.grids[g].addWidget(card, i // NCOLS, i % NCOLS)

    # ---------------- data ----------------
    def set_factory(self, factory):
        '''Slot: main window pushes a new StructureFactory here.'''
        self.factory = factory
        self._feats = None
        self._batch_token += 1
        if factory is None:
            self.status.setText(_t('no_struct'))
            return
        self.refresh()

    def _pos_edges(self, factory):
        g = factory.build()
        pos = np.asarray(g.node_positions(), float)[:, :2]
        edges = np.asarray(g.edge_array(), int)[:, :2]
        return pos, edges

    def _rect(self):
        if not self.chk_region.isChecked():
            return None
        s = self.rect_spins
        return (s['x0'].value(), s['y0'].value(),
                s['x1'].value(), s['y1'].value())

    def _on_region(self, on):
        for sp in self.rect_spins.values():
            sp.setEnabled(on)

    def refresh(self):
        '''Recompute the current snapshot (and the batch if enabled).'''
        if self.factory is None:
            self.status.setText(_t('no_struct'))
            return
        if self._worker is not None and self._worker.isRunning():
            return
        rect = self._rect()
        self._batch_token += 1
        try:
            t0 = time.perf_counter()
            pos, edges = self._pos_edges(self.factory)
            self._feats = compute_features(pos, edges, rect=rect)
            dt = (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            self.status.setText('%s: %s' % (e.__class__.__name__, e))
            return
        for key, card in self.cards.items():
            card.set_single(self._feats.get(key))
        self.fingerprint.set_features(self._feats, pos, edges)
        self.status.setText(_t('done_single') % (len(self.cards), dt))
        if self.chk_batch.isChecked():
            self._start_batch(rect)

    # ---------------- batch ----------------
    def _start_batch(self, rect):
        n = int(self.batch_spin.value())
        self._batch_token += 1
        self._worker = BatchWorker(self.factory, n, rect,
                                   self._batch_token, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_batch_done)
        self._worker.failed.connect(self._on_batch_failed)
        self.progress.setRange(0, n)
        self.progress.setValue(0)
        self.progress.show()
        self.btn_refresh.setEnabled(False)
        self.status.setText(_t('computing'))
        self._t_batch = time.perf_counter()
        self._worker.start()

    def _on_progress(self, i, n):
        self.progress.setValue(i)

    def _on_batch_done(self, results):
        worker = self.sender()
        self._worker = None
        self.progress.hide()
        self.btn_refresh.setEnabled(True)
        if worker is None or worker.token != self._batch_token:
            return
        for key, card in self.cards.items():
            if card.is_hist:
                parts = [np.asarray(r.get(key, ()), float)
                         for r in results]
                card.set_batch(np.concatenate(parts))
            else:
                card.set_batch(np.asarray(
                    [float(r.get(key, 0.0)) for r in results]))
        n = len(results)
        self.status.setText(
            _t('done_batch') % (n, n - 1,
                                time.perf_counter() - self._t_batch))

    def _on_batch_failed(self, msg):
        self._worker = None
        self.progress.hide()
        self.btn_refresh.setEnabled(True)
        self.status.setText(msg)

    # ---------------- theme / i18n ----------------
    def set_mode(self, mode):
        self.mode = mode
        c = colors(mode)
        # inline-styled descendants pin the content palette (Qt QSS
        # quirk); re-apply the app palette so theme switches propagate
        pal = QApplication.palette()
        self._content.setPalette(pal)
        self._scroll.viewport().setPalette(pal)
        for card in self.cards.values():
            card.mode = mode
            card.apply_theme(c)

    def retranslate(self):
        zh = get_lang() == 'zh'
        self.chk_region.setText(_t('region'))
        self.chk_batch.setText(_t('batch'))
        self.lbl_n.setText(_t('samples'))
        self.btn_refresh.setText(_t('refresh'))
        self.fp_title.setText('结构指纹 · 当前结构' if zh
                               else 'Structural fingerprint · current net')
        for g in GROUP_ORDER:
            self.group_lbls[g].setText(
                '%s · %s' % GROUP_NAME[g] if zh else GROUP_NAME[g][1])
        for card in self.cards.values():
            card.retranslate()
        if self.factory is None:
            self.status.setText(_t('no_struct'))
        elif self._feats is None:
            self.status.setText('⬡ ' + spec_text(self.factory))
