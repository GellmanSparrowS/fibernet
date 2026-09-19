"""Authoring dialog for user-defined square-periodic base units (F8b).

A cell is a corner graph with finite nodes and an edge list. The unit box
sets its repeat period; geometry can extend outside it.  Tiling is handled by fslab.structure (register_unit), so the only
thing this dialog has to get right is the graph itself; the live 2x2
preview and the odd-degree readout show the paper's manufacturability
check (all-even degree <= 8) while the user draws.
"""
import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal, QTimer
from PySide6.QtGui import QColor, QPainter, QPen, QShortcut, QKeySequence
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QHBoxLayout,
                               QLabel, QLineEdit, QPushButton, QVBoxLayout,
                               QWidget, QSpinBox, QDoubleSpinBox, QFormLayout)

from fslab.structure import (CELL_UNITS, CUSTOM_CELLS, save_custom_cell,
                             unit_display, valid_cell_spec)
from .i18n import tr, get_lang
from .theme import colors
from fslab.cell_rules import RULES, graph_health
from fslab import structure as structures

NODE_R = 8.0
EDGE_HIT = 6.0


def _tiled_graph(nodes, edges, gx=2, gy=2):
    """Merge-translated copies of one cell; returns (pos, edges) arrays."""
    idx, pos = {}, []
    out = []
    for j in range(gy):
        for i in range(gx):
            remap = []
            for x, y in nodes:
                key = (round(float(x) + i, 4), round(float(y) + j, 4))
                if key not in idx:
                    idx[key] = len(pos)
                    pos.append(key)
                remap.append(idx[key])
            for a, b in edges:
                out.append((remap[int(a)], remap[int(b)]))
    return np.asarray(pos, float), np.asarray(out, int).reshape(-1, 2)


def tiled_parity(nodes, edges, gx=2, gy=2):
    """(odd_degree_count, max_degree) of the tiled cell graph."""
    if not nodes or not edges:
        return 0, 0
    pos, ed = _tiled_graph(nodes, edges, gx, gy)
    deg = np.zeros(len(pos), int)
    for a, b in ed:
        deg[a] += 1
        deg[b] += 1
    return int((deg % 2 == 1).sum()), int(deg.max())


class CellCanvas(QWidget):
    """Draw / drag / connect / delete the corner graph of one cell."""
    changed = Signal()

    def __init__(self, mode="dark", parent=None):
        super().__init__(parent)
        self.mode = mode
        self.nodes = []
        self.edges = []
        self.tool = "select"
        self.snap = True
        self._drag = None
        self._link_from = None
        self._hover_node = None
        self._hover_edge = None
        self._undo = []
        self._redo = []
        self.view_scale = .55
        self.view_offset = QPointF(0., 0.)
        self._pan_start = None
        self.setMinimumSize(300, 300)
        self.setMouseTracking(True)

    def set_spec(self, nodes, edges):
        self.nodes = [[float(a), float(b)] for a, b in (nodes or [])]
        self.edges = [[int(a), int(b)] for a, b in (edges or [])]
        self._link_from = None
        self.fit_view()
        self.update()
        self.changed.emit()

    def spec(self):
        return ([list(n) for n in self.nodes], [list(e) for e in self.edges])

    # ---------------- geometry ----------------
    def _geom(self):
        m = 26
        side = max(40.0, min(self.width(), self.height()) - 2 * m) * self.view_scale
        x0 = (self.width() - side) / 2.0 + self.view_offset.x()
        y0 = (self.height() - side) / 2.0 + self.view_offset.y()
        return x0, y0, side

    def _px(self, x, y):
        x0, y0, s = self._geom()
        return QPointF(x0 + float(x) * s, y0 + (1.0 - float(y)) * s)

    def _unit(self, p):
        x0, y0, s = self._geom()
        x = (p.x() - x0) / s
        y = 1.0 - (p.y() - y0) / s
        if self.snap:
            x = round(x * 20.0) / 20.0
            y = round(y * 20.0) / 20.0
        return float(x), float(y)

    def fit_view(self):
        points = np.asarray(self.nodes + [[0., 0.], [1., 1.]], float)
        low, high = points.min(0), points.max(0)
        self.view_scale = .65 / max(1., float((high-low).max()))
        self.view_offset = QPointF(0., 0.)
        _, _, side = self._geom()
        center = (low+high)/2
        self.view_offset = QPointF((.5-center[0])*side, (center[1]-.5)*side)
        self.update()

    def wheelEvent(self, ev):
        self.view_scale = max(.02, min(8., self.view_scale * (1.15 if ev.angleDelta().y() > 0 else 1/1.15)))
        self.update()
        ev.accept()

    def mouseDoubleClickEvent(self, ev):
        self.fit_view()

    def _hit_node(self, p):
        best, bd = None, NODE_R + 4.0
        for k, (x, y) in enumerate(self.nodes):
            d = (self._px(x, y) - p).manhattanLength()
            if d < bd:
                best, bd = k, d
        return best

    def _hit_edge(self, p):
        best, bd = None, EDGE_HIT
        for k, (a, b) in enumerate(self.edges):
            pa, pb = self._px(*self.nodes[a]), self._px(*self.nodes[b])
            v = pb - pa
            L2 = v.x() ** 2 + v.y() ** 2
            t = 0.0 if L2 < 1e-9 else max(0.0, min(
                1.0, ((p.x() - pa.x()) * v.x() + (p.y() - pa.y()) * v.y()) / L2))
            q = pa + v * t
            d = (q - p).manhattanLength()
            if d < bd:
                best, bd = k, d
        return best

    # ---------------- events ----------------
    def mousePressEvent(self, ev):
        if ev.button() == Qt.MiddleButton:
            self._pan_start = ev.position()
            return
        if ev.button() == Qt.RightButton:
            self._link_from = None
            self.update()
            return
        self._undo.append(self.spec())
        self._undo = self._undo[-100:]
        self._redo.clear()
        p = ev.position()
        if self.tool == "select":
            self._drag = self._hit_node(p)
        elif self.tool == "add":
            if self._hit_node(p) is None and len(self.nodes) < 48:
                self.nodes.append(list(self._unit(p)))
                self._emit()
        elif self.tool == 'draw':
            k = self._hit_node(p)
            if k is None:
                if len(self.nodes) >= 48:
                    return
                k = len(self.nodes)
                self.nodes.append(list(self._unit(p)))
            a = self._link_from
            if a is not None and a != k and [a, k] not in self.edges and [k, a] not in self.edges:
                self.edges.append([a, k])
            self._link_from = k
            self._emit()
        elif self.tool == "link":
            k = self._hit_node(p)
            if k is None:
                return
            if self._link_from is None:
                self._link_from = k
            elif self._link_from != k:
                a, b = sorted((self._link_from, k))
                if [a, b] not in self.edges and [b, a] not in self.edges:
                    self.edges.append([self._link_from, k])
                self._link_from = None
                self._emit()
            else:
                self._link_from = None
            self.update()
        elif self.tool == "del":
            k = self._hit_node(p)
            if k is not None:
                self.edges = [[a - (1 if a > k else 0), b - (1 if b > k else 0)]
                              for a, b in self.edges if k not in (a, b)]
                self.nodes.pop(k)
                self._link_from = None
                self._emit()
            else:
                e = self._hit_edge(p)
                if e is not None:
                    self.edges.pop(e)
                    self._emit()

    def mouseMoveEvent(self, ev):
        p = ev.position()
        if self._pan_start is not None:
            self.view_offset += p - self._pan_start
            self._pan_start = p
            self.update()
            return
        if self._drag is not None:
            self.nodes[self._drag] = list(self._unit(p))
            self.update()
            self.changed.emit()
            return
        self._hover_node = self._hit_node(p)
        self._hover_edge = None if self._hover_node is not None \
            else self._hit_edge(p)
        self.update()

    def mouseReleaseEvent(self, ev):
        self._pan_start = None
        self._drag = None

    def _emit(self):
        self.update()
        self.changed.emit()

    def undo(self):
        if self._undo:
            self._redo.append(self.spec())
            self.set_spec(*self._undo.pop())

    def redo(self):
        if self._redo:
            self._undo.append(self.spec())
            self.set_spec(*self._redo.pop())

    # ---------------- paint ----------------
    def paintEvent(self, ev):
        p = QPainter(self)
        c = colors(self.mode)
        p.fillRect(self.rect(), QColor(c["bg"]))
        p.setRenderHint(QPainter.Antialiasing, True)
        x0, y0, s = self._geom()
        p.setPen(QPen(QColor(c["faint"]), 1.0))
        p.drawRect(int(x0), int(y0), int(s), int(s))
        p.setPen(QPen(QColor(c["faint"]), 0.6, Qt.DotLine))
        for k in range(1, 4):
            t = k / 4.0
            p.drawLine(self._px(t, 0.0), self._px(t, 1.0))
            p.drawLine(self._px(0.0, t), self._px(1.0, t))
        for k, (a, b) in enumerate(self.edges):
            hot = (k == self._hover_edge)
            p.setPen(QPen(QColor(c["accent"] if hot else c["accent2"]),
                          2.4 if hot else 1.8))
            p.drawLine(self._px(*self.nodes[a]), self._px(*self.nodes[b]))
        for k, (x, y) in enumerate(self.nodes):
            q = self._px(x, y)
            ring = c["accent"]
            if k == self._link_from:
                ring = c["accent2"]
            elif k == self._hover_node:
                ring = c["text"]
            p.setBrush(QColor(c["bg"]))
            p.setPen(QPen(QColor(ring), 2.2))
            p.drawEllipse(q, NODE_R, NODE_R)
        p.end()


class TilePreview(QWidget):
    """Small 2x2 tiling of the cell so periodicity is visible while drawing."""

    def __init__(self, mode="dark", parent=None):
        super().__init__(parent)
        self.mode = mode
        self.nodes, self.edges = [], []
        self.graph_arrays = None
        self.setMinimumSize(180, 180)

    def set_spec(self, nodes, edges):
        self.graph_arrays = None
        self.nodes, self.edges = nodes, edges
        self.update()

    def set_graph(self, pos, edges):
        self.graph_arrays = (np.asarray(pos), np.asarray(edges))
        self.nodes = pos.tolist()
        self.edges = edges.tolist()
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        c = colors(self.mode)
        p.fillRect(self.rect(), QColor(c["bg"]))
        p.setRenderHint(QPainter.Antialiasing, True)
        if not self.nodes or not self.edges:
            p.end()
            return
        pos, ed = (self.graph_arrays if self.graph_arrays is not None else
                   _tiled_graph(self.nodes, self.edges, 2, 2))
        m = 10
        lo, hi = pos.min(0), pos.max(0)
        span = np.maximum(hi - lo, 1e-9)
        sx = min((self.width() - 2*m) / span[0],
                 (self.height() - 2*m) / span[1])
        ox = (self.width() - span[0]*sx) / 2
        oy = (self.height() - span[1]*sx) / 2

        def w2s(x, y):
            return QPointF(ox + (x-lo[0]) * sx, oy + (hi[1]-y) * sx)
        p.setPen(QPen(QColor(c["accent2"]), 1.2))
        for a, b in ed:
            p.drawLine(w2s(*pos[a]), w2s(*pos[b]))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(c["sub"]))
        for x, y in pos:
            p.drawEllipse(w2s(x, y), 1.4, 1.4)
        p.end()


class CellEditorDialog(QDialog):
    """Modal authoring of one custom base unit."""

    def __init__(self, mode="dark", key=None, parent=None):
        super().__init__(parent)
        self.mode = mode
        self.edit_key = key if key in CUSTOM_CELLS else None
        self.saved_key = None
        self.saved_factory = None
        self._preview_factory = None
        self.custom_rule = None
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(120)
        self._preview_timer.timeout.connect(self._build_preview)
        c = colors(mode)
        self.setWindowTitle(tr("cell_editor"))
        self.setStyleSheet("QDialog { background:%s; }" % c["bg"])
        self.resize(820, 650)

        root = QHBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(12)

        left = QVBoxLayout()
        left.setSpacing(8)
        left.addWidget(QLabel(tr('cell_step_draw')))
        self.canvas = CellCanvas(mode)
        self.canvas.changed.connect(self._sync)
        self.undo_shortcut = QShortcut(QKeySequence.Undo, self)
        self.undo_shortcut.activated.connect(self.canvas.undo)
        self.redo_shortcut = QShortcut(QKeySequence.Redo, self)
        self.redo_shortcut.activated.connect(self.canvas.redo)
        left.addWidget(self.canvas, 1)
        self.hint = QLabel(tr("cell_hint"))
        self.hint.setObjectName("hint")
        self.hint.setWordWrap(True)
        left.addWidget(self.hint)
        root.addLayout(left, 3)

        right = QVBoxLayout()
        right.setSpacing(8)
        tools = QHBoxLayout()
        tools.setSpacing(4)
        self.tool_btns = {}
        for name in ("select", "draw", "add", "link", "del"):
            b = QPushButton(tr("cell_tool_" + name))
            b.setCheckable(True)
            b.setChecked(name == "select")
            b.clicked.connect(lambda _=False, n=name: self._set_tool(n))
            self.tool_btns[name] = b
            tools.addWidget(b)
        left.insertLayout(1, tools)
        history = QHBoxLayout()
        self.undo_btn = QPushButton(tr('cell_undo'))
        self.redo_btn = QPushButton(tr('cell_redo'))
        self.undo_btn.clicked.connect(self.canvas.undo)
        self.redo_btn.clicked.connect(self.canvas.redo)
        history.addWidget(self.undo_btn)
        history.addWidget(self.redo_btn)
        self.fit_btn = QPushButton('适应画布' if get_lang() == 'zh' else 'Fit')
        self.fit_btn.clicked.connect(self.canvas.fit_view)
        history.addWidget(self.fit_btn)
        left.insertLayout(2, history)
        self.snap_box = QCheckBox(tr("cell_snap"))
        self.snap_box.setChecked(True)
        self.snap_box.toggled.connect(self._set_snap)
        left.insertWidget(3, self.snap_box)

        self.start_lbl = QLabel(tr("cell_start"))
        self.start_lbl.setObjectName("hint")
        self.start = QComboBox()
        self.start.addItem(tr("cell_blank"), "")
        self.seeds = dict(CELL_UNITS)
        self.seeds['square'] = dict(nodes=[(0,0), (1,0), (1,1), (0,1)],
                                   edges=[(0,1), (1,2), (2,3), (3,0)])
        self.seeds.update(CUSTOM_CELLS)
        for k in self.seeds:
            if k == key:
                continue
            self.start.addItem(unit_display(k), k)
        self.start.currentIndexChanged.connect(self._load_start)
        left.insertWidget(1, self.start_lbl)
        left.insertWidget(2, self.start)
        right.addWidget(QLabel(tr('cell_step_expand')))

        self.name_lbl = QLabel(tr("cell_name"))
        self.name_lbl.setObjectName("hint")
        self.name_zh = QLineEdit()
        self.name_en = QLineEdit()
        right.addWidget(self.name_lbl)
        names = QHBoxLayout()
        self.name_zh.setPlaceholderText(tr('cell_name_zh'))
        self.name_en.setPlaceholderText('English')
        names.addWidget(self.name_zh)
        names.addWidget(self.name_en)
        right.addLayout(names)

        form = QFormLayout()
        self.rule = QComboBox()
        for rule in RULES:
            self.rule.addItem(tr('rule_' + rule), rule)
        self.gx, self.gy = QSpinBox(), QSpinBox()
        for spin in (self.gx, self.gy):
            spin.setRange(1, 8)
            spin.setValue(3)
            spin.valueChanged.connect(self._sync)
        grid = QHBoxLayout()
        grid.addWidget(self.gx)
        grid.addWidget(QLabel('×'))
        grid.addWidget(self.gy)
        self.profile = QComboBox()
        for k in structures.SPECTRUM_PRESETS:
            self.profile.addItem(unit_display(k), k)
        self.rule.hide()
        self.rule_edit = QPushButton(tr('expansion_rule') + '…')
        self.rule_edit.setMaximumWidth(150)
        self.rule_edit.clicked.connect(self._edit_rule)
        form.addRow('', self.rule_edit)
        form.addRow(tr('grid'), grid)
        form.addRow(tr('spectrum'), self.profile)
        self.amplitude = QDoubleSpinBox()
        self.amplitude.setRange(0.0, 3.0)
        self.amplitude.setSingleStep(0.1)
        self.amplitude.setValue(1.0)
        form.addRow(tr('cell_amplitude'), self.amplitude)
        self.amplitude.valueChanged.connect(self._sync)
        self.rule.currentIndexChanged.connect(self._sync)
        self.profile.currentIndexChanged.connect(self._sync)
        right.addLayout(form)

        self.prev_lbl = QLabel(tr("cell_preview"))
        self.prev_lbl.setObjectName("hint")
        self.preview = TilePreview(mode)
        self.preview.setMinimumSize(180, 140)
        right.addWidget(self.prev_lbl)
        right.addWidget(self.preview, 1)
        self.stats = QLabel()
        self.stats.setWordWrap(True)
        right.addWidget(self.stats)

        btns = QHBoxLayout()
        btns.addStretch(1)
        self.save_btn = QPushButton(tr("cell_save"))
        self.save_btn.clicked.connect(self._save)
        self.cancel_btn = QPushButton(tr("cancel"))
        self.cancel_btn.clicked.connect(self.reject)
        btns.addWidget(self.cancel_btn)
        btns.addWidget(self.save_btn)
        right.addLayout(btns)
        root.addLayout(right, 2)

        if key and key in CUSTOM_CELLS:
            spec = CUSTOM_CELLS[key]
            self.canvas.set_spec(spec["nodes"], spec["edges"])
            self.name_zh.setText(spec.get("zh", ""))
            self.name_en.setText(spec.get("en", ""))
            self.start.hide()
            self.start_lbl.hide()
            settings = spec.get('settings', {})
            self.custom_rule = settings.get('custom_rule')
            self.gx.setValue(settings.get('grid_x', 3))
            self.gy.setValue(settings.get('grid_y', 3))
            self.rule.setCurrentIndex(max(0, self.rule.findData(settings.get('expansion_rule', 'translate'))))
            self.profile.setCurrentIndex(max(0, self.profile.findData(settings.get('profile', 'square'))))
            self.amplitude.setValue(settings.get('amplitude', 1.0))
        else:
            self.start.setCurrentIndex(self.start.findData(key if key in self.seeds else 'square'))
        self._sync()

    # ---------------- slots ----------------
    def _edit_rule(self):
        from .rule_dialog import ExpansionDialog
        dialog = ExpansionDialog(self.rule.currentData(), self.custom_rule, self)
        if dialog.exec():
            self.custom_rule = dialog.custom_rule
            self.rule.setCurrentIndex(self.rule.findData(dialog.rule.currentData()))
            self._sync()

    def _set_tool(self, name):
        self.canvas.tool = name
        self.hint.setText(tr('cell_draw_help') if name == 'draw' else tr('cell_hint'))
        self.canvas._link_from = None
        for n, b in self.tool_btns.items():
            b.setChecked(n == name)
        self.canvas.update()

    def _set_snap(self, on):
        self.canvas.snap = bool(on)

    def _load_start(self, _idx):
        k = self.start.currentData()
        if not k:
            self.canvas.set_spec([], [])
            return
        spec = self.seeds.get(k)
        if spec:
            self.canvas.set_spec(spec["nodes"], spec["edges"])

    def _sync(self):
        if not hasattr(self, 'stats'):
            return
        nodes, edges = self.canvas.spec()
        self.preview.set_spec(nodes, edges)
        self._preview_factory = None
        self.stats.setText(tr('cell_checking'))
        self._preview_timer.start()

    def _build_preview(self):
        nodes, edges = self.canvas.spec()
        spec = dict(nodes=nodes, edges=edges)
        if not valid_cell_spec(spec):
            self.stats.setText(tr('cell_invalid'))
            return
        key = '__editor_preview__'
        structures.CUSTOM_CELLS[key] = spec
        structures._CELLS_REGISTERED.discard(key)
        structures._INTERMEDIATE_CACHE.clear()
        try:
            profile = structures.SPECTRUM_PRESETS[self.profile.currentData()]
            profile = (np.asarray(profile, float) * self.amplitude.value()).tolist()
            f = structures.StructureFactory(
                unit=key, grid_x=self.gx.value(), grid_y=self.gy.value(),
                n_pts_per_side=5, line_displacements=list(profile),
                expansion_rule=self.rule.currentData(), custom_rule=self.custom_rule)
            graph = f.build()
            pos = np.asarray(graph.node_positions(), float)[:, :2]
            ed = np.asarray(graph.edge_array(), int)[:, :2]
            health = graph_health(pos, ed)
            self.preview.set_graph(pos, ed)
            self.stats.setText(tr('cell_health') % (
                health['nodes'], health['edges'], health['components'],
                health['odd'], health['max_degree']))
            self._preview_factory = f
        except Exception as exc:
            self.stats.setText(str(exc))
        finally:
            structures.CUSTOM_CELLS.pop(key, None)
            structures._CELLS_REGISTERED.discard(key)
            structures._INTERMEDIATE_CACHE.clear()

    def _save(self):
        self._preview_timer.stop()
        self._build_preview()
        nodes, edges = self.canvas.spec()
        zh = self.name_zh.text().strip()
        en = self.name_en.text().strip() or zh
        if not zh:
            self.stats.setText(tr("cell_need_name"))
            return
        if not valid_cell_spec({"nodes": nodes, "edges": edges}):
            self.stats.setText(tr("cell_invalid"))
            return
        if self._preview_factory is None:
            return
        import re
        key = self.edit_key or re.sub(r'[^a-z0-9_]+', '_', en.lower()).strip('_') or 'cell'
        if not self.edit_key:
            stem, suffix = key, 2
            while key in structures.all_unit_keys() or key in structures.SPECTRUM_PRESETS:
                key = stem + '_' + str(suffix)
                suffix += 1
        try:
            self.saved_key = save_custom_cell(key, nodes, edges, zh, en,
                settings=dict(grid_x=self.gx.value(), grid_y=self.gy.value(),
                              expansion_rule=self.rule.currentData(),
                              custom_rule=self.custom_rule,
                              profile=self.profile.currentData(),
                              amplitude=self.amplitude.value()))
        except OSError as exc:
            self.stats.setText(str(exc))
            return
        from dataclasses import replace
        self.saved_factory = replace(self._preview_factory, unit=self.saved_key)
        self.accept()
