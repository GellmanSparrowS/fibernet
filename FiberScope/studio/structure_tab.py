"""Structure studio: standalone structure-generation panel.

Owns the shared StructureFactory; simulation tabs consume the emitted spec.
The primitive editor edits ONE reference fiber line (control points as
fractions of the line length); the profile is replicated onto every fiber
line of the lattice, so dragging one line reshapes the whole unit.
"""
import numpy as np
from PySide6.QtCore import QPointF, QSize, Qt, QTimer, Signal, QSignalBlocker
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout, QGroupBox,
                               QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                               QSlider, QSpinBox, QSplitter, QStackedWidget,
                               QListWidget, QListWidgetItem, QMenu,
                               QMessageBox, QVBoxLayout, QWidget)

from fslab.structure import (StructureFactory, UNIT_PRESETS, BASE_UNIT_KEYS,
                                 SPECTRUM_PRESETS,
                                 all_unit_keys, load_custom_cells,
                                 CUSTOM_CELLS, delete_custom_cell,
                                 resolve_unit, unit_display, unit_key)
from .cell_editor import CellEditorDialog
from fslab.engine2 import _grips
from fslab.exporter import export_json, export_svg
from .exports import add_caption, ask_save, save_widget_png
from .i18n import tr, get_lang
from .network_canvas import NetworkCanvas
from .theme import colors


def spec_text(f: StructureFactory) -> str:
    return (f"{unit_display(f.unit, get_lang())} · {f.grid_x}×{f.grid_y} · "
            f"pts {f.n_pts_per_side} · seed {f.seed}")


class LineEditor(QWidget):
    """Drag control points of one reference fiber line (fractions of len)."""
    changed = Signal()

    def __init__(self, mode="dark", parent=None):
        super().__init__(parent)
        self.mode = mode
        self.pts = 0
        self.disp = []
        self._drag = None
        self.zoom = 1.0
        self.setMinimumSize(320, 150)
        self.setMouseTracking(True)

    def set_pts(self, n):
        n = int(max(0, n))
        old = self.disp
        self.disp = [[old[k][0], old[k][1]] if k < len(old) else [0.0, 0.0]
                     for k in range(n)]
        self.pts = n
        self.update()

    def set_displacements(self, ld):
        self.disp = [[float(x), float(y)] for x, y in (ld or [])]
        self.pts = len(self.disp)
        self.update()

    def clear(self):
        self.disp = [[0.0, 0.0] for _ in range(self.pts)]
        self.update()
        self.changed.emit()

    def displacements(self):
        if self.pts <= 0:
            return None
        if all(abs(x) < 1e-6 and abs(y) < 1e-6 for x, y in self.disp):
            return None
        return [[float(x), float(y)] for x, y in self.disp]

    # ---------------- geometry ----------------
    def _geom(self):
        m = 30
        w, h = self.width(), self.height()
        base = max(140.0, min(w - 2 * m, (h - 16) / 0.6))
        L = base * self.zoom
        x0 = (w - L) / 2.0
        return x0, x0 + L, h * 0.5

    def _handle_at(self, p):
        best, bd = None, 20.0
        for k in range(self.pts):
            d = (self._handle_pos(k) - p).manhattanLength()
            if d < bd:
                best, bd = k, d
        return best

    def _handle_pos(self, k):
        x0, x1, y = self._geom()
        L = x1 - x0
        t = (k + 1) / (self.pts + 1)
        dx, dy = self.disp[k]
        return QPointF(x0 + t * L + dx * L, y + dy * L)

    def mousePressEvent(self, ev):
        self._drag = self._handle_at(ev.position())
        if self._drag is not None:
            self._move(ev.position())

    def mouseMoveEvent(self, ev):
        if self._drag is not None:
            self._move(ev.position())

    def mouseReleaseEvent(self, ev):
        self._drag = None

    def wheelEvent(self, ev):
        delta = 1.0016 ** ev.angleDelta().y()
        self.zoom = float(max(0.35, min(6.0, self.zoom * delta)))
        self.update()

    def _move(self, p):
        k = self._drag
        if k is None:
            return
        x0, x1, y = self._geom()
        L = x1 - x0
        t = (k + 1) / (self.pts + 1)
        dx = (p.x() - (x0 + t * L)) / L
        dy = (p.y() - y) / L
        self.disp[k] = [float(max(-1.0, min(1.0, dx))),
                        float(max(-1.0, min(1.0, dy)))]
        self.update()
        self.changed.emit()

    # ---------------- paint ----------------
    def paintEvent(self, ev):
        p = QPainter(self)
        c = colors(self.mode)
        p.fillRect(self.rect(), QColor(c["bg"]))
        p.setRenderHint(QPainter.Antialiasing, True)
        x0, x1, y = self._geom()
        # original straight line
        p.setPen(QPen(QColor(c["faint"]), 1.2, Qt.DashLine))
        p.drawLine(QPointF(x0, y), QPointF(x1, y))
        # displaced profile
        p.setPen(QPen(QColor(c["accent2"]), 2.0))
        prev = QPointF(x0, y)
        for k in range(self.pts):
            q = self._handle_pos(k)
            p.drawLine(prev, q)
            prev = q
        p.drawLine(prev, QPointF(x1, y))
        # fixed ends
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(c["sub"]))
        for x in (x0, x1):
            p.drawRect(int(x) - 4, int(y) - 4, 8, 8)
        # handles
        for k in range(self.pts):
            q = self._handle_pos(k)
            p.setBrush(QColor(c["bg"]))
            p.setPen(QPen(QColor(c["accent"]), 2.2))
            p.drawEllipse(q, 9.0, 9.0)
        p.end()


class DispRow(QWidget):
    """Numeric per-point dx/dy inputs (% of line length), synced w/ editor."""
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lay = QGridLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(6)
        self._widgets = []
        self.spins = []
        self._guard = False

    def set_pts(self, n):
        self._guard = True
        for w in self._widgets:
            self._lay.removeWidget(w)
            w.deleteLater()
        self._widgets, self.spins = [], []
        for k in range(int(max(0, n))):
            row, col = divmod(k, 3)
            lbl = QLabel(f"P{k + 1}")
            lbl.setObjectName("hint")
            self._lay.addWidget(lbl, row, col * 3)
            self._widgets.append(lbl)
            for axis in range(2):
                sp = QDoubleSpinBox()
                sp.setRange(-100, 100)
                sp.setSingleStep(1.0)
                sp.setDecimals(1)
                sp.setSuffix("%")
                sp.setProperty("k", k)
                sp.setProperty("axis", axis)
                sp.valueChanged.connect(self._on_spin)
                self._lay.addWidget(sp, row, col * 3 + 1 + axis)
                self._widgets.append(sp)
                self.spins.append(sp)
        self._guard = False

    def set_values(self, disp):
        self._guard = True
        for sp in self.spins:
            k = sp.property("k")
            ax = sp.property("axis")
            if k < len(disp):
                sp.setValue(round(disp[k][ax] * 100.0, 1))
        self._guard = False

    def values(self):
        out = {}
        for sp in self.spins:
            out.setdefault(sp.property("k"), [0.0, 0.0])[
                sp.property("axis")] = sp.value() / 100.0
        return [out[k] for k in sorted(out)]

    def _on_spin(self, *a):
        if not self._guard:
            self.changed.emit()


class StructureTab(QWidget):
    structure_changed = Signal(object)

    def __init__(self, mode="dark", parent=None):
        super().__init__(parent)
        self.mode = mode
        self.factory = None
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(120)
        self._debounce.timeout.connect(self.push_spec)
        self._build()
        self.retranslate()
        self.push_spec()

    # ---------------- UI ----------------
    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        body = QSplitter(Qt.Horizontal)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(10)

        self.gb_params = QGroupBox()
        g = QGridLayout(self.gb_params)
        g.setHorizontalSpacing(8); g.setVerticalSpacing(10)
        load_custom_cells()   # user-authored cells join the gallery
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(
            [unit_display(k, get_lang()) for k in all_unit_keys()])
        self.unit_combo.setCurrentText(unit_display("square", get_lang()))
        self.spectrum_combo = QComboBox()
        self.spectrum_combo.addItems(
            [unit_display(k, get_lang()) for k in SPECTRUM_PRESETS])
        self.spectrum_combo.setCurrentText(unit_display("square", get_lang()))
        self.grid_x = QSpinBox(); self.grid_x.setRange(1, 128); self.grid_x.setValue(3)
        self.grid_y = QSpinBox(); self.grid_y.setRange(1, 128); self.grid_y.setValue(3)
        self.lbl_x = QLabel("×")
        self.pts = QSpinBox(); self.pts.setRange(0, 24); self.pts.setValue(5)
        self.pert = QSlider(Qt.Horizontal); self.pert.setRange(0, 100)
        self.pert.setValue(0)
        self.seed = QSpinBox(); self.seed.setRange(0, 9999); self.seed.setValue(7)
        self.dice_btn = QPushButton()
        self.lbl_unit = QLabel(); self.lbl_grid = QLabel()
        self.lbl_pts = QLabel(); self.lbl_pert = QLabel(); self.lbl_seed = QLabel()
        self.lbl_spectrum = QLabel()
        g.addWidget(self.lbl_unit, 0, 0); g.addWidget(self.unit_combo, 0, 1, 1, 3)
        g.addWidget(self.lbl_spectrum, 1, 0)
        g.addWidget(self.spectrum_combo, 1, 1, 1, 3)
        g.addWidget(self.lbl_grid, 2, 0); g.addWidget(self.grid_x, 2, 1)
        g.addWidget(self.lbl_x, 2, 2, Qt.AlignCenter)
        g.addWidget(self.grid_y, 2, 3)
        g.addWidget(self.lbl_pts, 3, 0); g.addWidget(self.pts, 3, 1, 1, 3)
        g.addWidget(self.lbl_pert, 4, 0); g.addWidget(self.pert, 4, 1, 1, 3)
        g.addWidget(self.lbl_seed, 5, 0); g.addWidget(self.seed, 5, 1, 1, 3)
        g.addWidget(self.dice_btn, 6, 0, 1, 4)
        self.dice_btn.clicked.connect(self._random_seed)
        from fslab.cell_rules import RULES
        self.custom_rule = None
        self.rule_combo = QComboBox()
        for rule in RULES:
            self.rule_combo.addItem(tr('rule_' + rule), rule)
        self.lbl_rule = QLabel(tr('expansion_rule'))
        g.addWidget(self.lbl_rule, 7, 0)
        g.addWidget(self.rule_combo, 7, 1, 1, 2)
        self.rule_edit = QPushButton(tr('rule_edit'))
        self.rule_edit.clicked.connect(self._edit_rule)
        g.addWidget(self.rule_edit, 7, 3)
        self.rule_combo.currentIndexChanged.connect(lambda: self._debounce.start())
        for control in (self.lbl_rule, self.rule_combo, self.rule_edit):
            control.hide()
        lv.addWidget(self.gb_params)

        self.workbench_btn = QPushButton(tr('cell_workbench'))
        self.workbench_btn.clicked.connect(lambda: self._open_cell_editor(None))
        lv.addWidget(self.workbench_btn)

        self.edit_btn = QPushButton()
        self.edit_btn.setProperty("primary", True)
        self.edit_btn.clicked.connect(self._toggle_edit)
        lv.addWidget(self.edit_btn)

        exrow = QHBoxLayout()
        self.export_json_btn = QPushButton()
        self.export_svg_btn = QPushButton()
        self.export_png_btn = QPushButton()
        self.export_json_btn.clicked.connect(self._do_export_json)
        self.export_svg_btn.clicked.connect(self._do_export_svg)
        self.export_png_btn.clicked.connect(self._do_export_png)
        exrow.addWidget(self.export_json_btn)
        exrow.addWidget(self.export_svg_btn)
        exrow.addWidget(self.export_png_btn)
        lv.addLayout(exrow)
        self.manufacturing_btn = QPushButton()
        self.manufacturing_btn.clicked.connect(self._open_manufacturing)
        lv.addWidget(self.manufacturing_btn)

        self.gb_stats = QGroupBox()
        sv = QHBoxLayout(self.gb_stats)
        sv.setSpacing(6)
        self.stat_n_v = QLabel(); self.stat_n_v.setObjectName("stat_value")
        self.stat_n_l = QLabel(); self.stat_n_l.setObjectName("stat_label")
        self.stat_e_v = QLabel(); self.stat_e_v.setObjectName("stat_value")
        self.stat_e_l = QLabel(); self.stat_e_l.setObjectName("stat_label")
        self.stat_u_v = QLabel(); self.stat_u_v.setObjectName("stat_value")
        self.stat_u_l = QLabel(); self.stat_u_l.setObjectName("stat_label")
        for v, l in ((self.stat_n_v, self.stat_n_l),
                     (self.stat_e_v, self.stat_e_l),
                     (self.stat_u_v, self.stat_u_l)):
            col = QVBoxLayout()
            col.addWidget(v); col.addWidget(l)
            sv.addLayout(col)
        lv.addWidget(self.gb_stats)

        self.status = QLabel(); self.status.setObjectName("hint")
        self.status.setWordWrap(True)
        lv.addWidget(self.status)
        lv.addStretch(1)
        left.setFixedWidth(300)
        body.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(10)

        # The canvas never hides: editing the spectrum is only worth seeing
        # while the whole network deforms live, so the gallery/editor deck
        # sits BELOW the canvas instead of replacing it.
        self.canvas = NetworkCanvas(mode=self.mode)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumHeight(170)
        self.deck = QStackedWidget()
        self.gb_gallery = QGroupBox()
        gal = QVBoxLayout(self.gb_gallery)
        gal.setContentsMargins(8, 16, 8, 8)
        self.gallery = QListWidget()
        self.gallery.setObjectName("gallery")
        self.gallery.setViewMode(QListWidget.IconMode)
        self.gallery.setWrapping(True)
        self.gallery.setResizeMode(QListWidget.Adjust)
        self.gallery.setMovement(QListWidget.Static)
        self.gallery.setIconSize(QSize(70, 44))
        self.gallery.setGridSize(QSize(96, 76))
        self.gallery.setSpacing(4)
        # Built-ins stay in two rows; saved cells have their own compact shelf.
        self.gallery.setFixedWidth(6 * (96 + 4) + 8)
        self.gallery.setFixedHeight(176)
        self.gallery.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.gallery.setSelectionMode(QListWidget.SingleSelection)
        self._gallery_guard = False
        self._fill_gallery_items()
        self.gallery.currentItemChanged.connect(self._on_gallery_pick)
        self.gallery.setContextMenuPolicy(Qt.CustomContextMenu)
        self.gallery.customContextMenuRequested.connect(self._gallery_menu)
        gal.addWidget(self.gallery)
        gal.setAlignment(self.gallery, Qt.AlignHCenter)
        self.custom_shelf = QWidget()
        shelf = QHBoxLayout(self.custom_shelf)
        shelf.setContentsMargins(0, 0, 0, 0)
        self.custom_label = QLabel(tr('cell_saved'))
        self.custom_combo = QComboBox()
        self.custom_edit = QPushButton(tr('cell_edit'))
        self.custom_delete = QPushButton(tr('cell_delete'))
        shelf.addWidget(self.custom_label)
        shelf.addWidget(self.custom_combo, 1)
        shelf.addWidget(self.custom_edit)
        shelf.addWidget(self.custom_delete)
        self.custom_edit.clicked.connect(lambda: self._open_cell_editor(self.custom_combo.currentData()))
        self.custom_delete.clicked.connect(self._delete_saved_cell)
        self.custom_combo.activated.connect(self._choose_saved_cell)
        gal.addWidget(self.custom_shelf)
        self._refresh_custom_shelf()
        self.deck.addWidget(self.gb_gallery)

        page1 = QWidget()
        p1 = QVBoxLayout(page1)
        p1.setContentsMargins(0, 0, 0, 0)
        p1.setSpacing(10)
        self.editor = LineEditor(mode=self.mode)
        self.editor.set_pts(self.pts.value())
        self.editor.changed.connect(self._on_edit_changed)
        top = QHBoxLayout()
        top.setSpacing(10)
        top.addWidget(self.editor, 1)
        side = QVBoxLayout()
        side.setSpacing(4)
        self.lbl_preview = QLabel()
        self.lbl_preview.setObjectName("subtitle")
        self.unit_preview = QLabel()
        self.unit_preview.setAlignment(Qt.AlignCenter)
        self.unit_preview.setObjectName("well")
        self.unit_preview.setFixedSize(168, 148)
        side.addWidget(self.lbl_preview)
        side.addWidget(self.unit_preview, 1)
        top.addLayout(side)
        p1.addLayout(top, 1)
        self.lbl_disp_row = QLabel()
        self.lbl_disp_row.setObjectName("subtitle")
        p1.addWidget(self.lbl_disp_row)
        self.disp_row = DispRow()
        self.disp_row.set_pts(self.pts.value())
        p1.addWidget(self.disp_row)
        erow = QHBoxLayout()
        self.edit_hint = QLabel()
        self.edit_hint.setObjectName("hint")
        self.edit_hint.setWordWrap(True)
        self.clear_btn = QPushButton()
        self.clear_btn.clicked.connect(self._clear_line)
        self.done_btn = QPushButton()
        self.done_btn.setProperty("primary", True)
        self.done_btn.clicked.connect(self._toggle_edit)
        erow.addWidget(self.edit_hint, 1)
        erow.addWidget(self.clear_btn)
        erow.addWidget(self.done_btn)
        p1.addLayout(erow)
        self.deck.addWidget(page1)
        self.deck.setMinimumHeight(214)

        self.vsplit = QSplitter(Qt.Vertical)
        self.vsplit.setChildrenCollapsible(False)
        self.vsplit.addWidget(self.canvas)
        self.vsplit.addWidget(self.deck)
        self.vsplit.setStretchFactor(0, 3)
        self.vsplit.setStretchFactor(1, 1)
        self.vsplit.setSizes([430, 250])
        rv.addWidget(self.vsplit, 1)
        body.addWidget(right)
        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)
        outer.addWidget(body, 1)

        for w in (self.grid_x, self.grid_y):
            w.valueChanged.connect(lambda *a: self._on_discrete())
        self.pts.valueChanged.connect(self._on_points_changed)
        self.seed.valueChanged.connect(self._on_seed_changed)
        self.disp_row.changed.connect(self._on_row_changed)
        self._sync_row()
        self.unit_combo.currentTextChanged.connect(self._on_unit_changed)
        self.spectrum_combo.currentTextChanged.connect(self._on_spectrum_changed)
        self.pert.valueChanged.connect(self._apply_seeded_parameters)
        self._refresh_gallery()

    # ---------------- actions ----------------
    def _on_discrete(self):
        self.push_spec()

    def _on_unit_changed(self, *a):
        self._refresh_gallery()
        self._sync_spectrum_visibility()
        self.push_spec()

    def _on_spectrum_changed(self, *a):
        key = unit_key(self.spectrum_combo.currentText(), get_lang())
        if key in SPECTRUM_PRESETS:
            self.editor.set_displacements(SPECTRUM_PRESETS[key])
            self._parameter_basis=np.asarray(self.editor.disp,float)/(self.pert.value()/100. or 1.)
            self._sync_row()
        self._refresh_gallery()
        self.push_spec()

    def _sync_spectrum_visibility(self):
        self.spectrum_combo.setEnabled(
            unit_key(self.unit_combo.currentText(), get_lang()) == 'square')

    def _on_points_changed(self, value):
        self.editor.set_pts(value)
        self._sync_row()
        if self.pert.value():
            self._apply_seeded_parameters()
        else:
            self.push_spec()

    def _on_seed_changed(self, *_):
        self._parameter_basis=None
        if self.pert.value()==0:
            blocker=QSignalBlocker(self.pert)
            self.pert.setValue(25)
            del blocker
        self._apply_seeded_parameters()

    def _apply_seeded_parameters(self, *_):
        shape=(self.pts.value(),2)
        basis=getattr(self,'_parameter_basis',None)
        if basis is None or basis.shape!=shape:
            basis=np.random.default_rng(self.seed.value()).uniform(-.25,.25,shape)
            self._parameter_basis=basis
        self.editor.disp=np.round(np.clip(basis*self.pert.value()/100.,-1.,1.),3).tolist()
        self.editor.update()
        self._sync_row()
        self._update_preview()
        self._debounce.start()

    def _random_seed(self):
        self.seed.setValue((self.seed.value()+int(np.random.randint(1,10000)))%10000)

    def _on_gallery_pick(self, cur, prev):
        if self._gallery_guard or cur is None:
            return
        self._pick_unit(cur)

    def _pick_unit(self, name):
        key = name.data(Qt.UserRole) if hasattr(name, "data") else name
        if hasattr(name, "data") and name.data(Qt.UserRole):
            key = name.data(Qt.UserRole)
        if key == "__new__":
            self._open_cell_editor(None)
            return
        self.unit_combo.setCurrentText(unit_display(key, get_lang()))

    # ---------------- custom base units ----------------
    def _fill_gallery_items(self):
        for name in BASE_UNIT_KEYS:
            it = QListWidgetItem(unit_display(name, get_lang()))
            it.setData(Qt.UserRole, name)
            it.setTextAlignment(Qt.AlignHCenter | Qt.AlignBottom)
            self.gallery.addItem(it)
        it = QListWidgetItem(tr("cell_new"))
        it.setData(Qt.UserRole, "__new__")
        it.setTextAlignment(Qt.AlignHCenter | Qt.AlignBottom)
        self.gallery.addItem(it)

    def _refresh_custom_shelf(self):
        key = self.custom_combo.currentData()
        self.custom_combo.blockSignals(True)
        self.custom_combo.clear()
        for k in sorted(CUSTOM_CELLS):
            self.custom_combo.addItem(unit_display(k, get_lang()), k)
        index = self.custom_combo.findData(key)
        if index >= 0:
            self.custom_combo.setCurrentIndex(index)
        self.custom_combo.blockSignals(False)
        self.custom_shelf.setVisible(bool(CUSTOM_CELLS))

    def _choose_saved_cell(self):
        key = self.custom_combo.currentData()
        if key not in CUSTOM_CELLS:
            return
        settings = CUSTOM_CELLS[key].get('settings', {})
        from fslab.structure import resample_spectrum
        self.load_spec(StructureFactory(unit=key,
            grid_x=settings.get('grid_x', 3), grid_y=settings.get('grid_y', 3),
            expansion_rule=settings.get('expansion_rule', 'translate'),
            custom_rule=settings.get('custom_rule'),
            line_displacements=(np.asarray(resample_spectrum(SPECTRUM_PRESETS.get(
                settings.get('profile', 'square'), SPECTRUM_PRESETS['square']), 5))
                * settings.get('amplitude', 1.0)).tolist()))

    def _delete_saved_cell(self):
        key = self.custom_combo.currentData()
        if key not in CUSTOM_CELLS:
            return
        if QMessageBox.question(self, tr('cell_delete'),
                tr('cell_confirm_del') % key) != QMessageBox.Yes:
            return
        delete_custom_cell(key)
        self._rebuild_gallery()
        self.load_spec(StructureFactory())

    def _select_gallery(self, key):
        self._gallery_guard = True
        for i in range(self.gallery.count()):
            it = self.gallery.item(i)
            selected = (it.data(Qt.UserRole) or it.text()) == key
            it.setSelected(selected)
            if selected:
                self.gallery.setCurrentItem(it)
        self._gallery_guard = False

    def _plus_icon(self):
        pm = QPixmap(70, 44)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(QPen(QColor(colors(self.mode)["faint"]), 2.0))
        p.drawLine(QPointF(35, 10), QPointF(35, 34))
        p.drawLine(QPointF(23, 22), QPointF(47, 22))
        p.end()
        return pm

    def _rebuild_gallery(self):
        self._refresh_custom_shelf()
        cur = unit_key(self.unit_combo.currentText(), get_lang())
        self.gallery.clear()
        self._fill_gallery_items()
        self.unit_combo.clear()
        self.unit_combo.addItems(
            [unit_display(k, get_lang()) for k in all_unit_keys()])
        self.unit_combo.setCurrentText(unit_display(cur, get_lang()))
        self._refresh_gallery()
        self._select_gallery(cur)

    def _open_cell_editor(self, key):
        dlg = CellEditorDialog(self.mode, key, self)
        dlg.exec()
        saved = dlg.saved_key
        prev = self.factory.unit if self.factory is not None else 'square'
        self._rebuild_gallery()
        target = saved or prev
        self._select_gallery(target)
        if saved:
            self.load_spec(dlg.saved_factory)
        dlg.deleteLater()

    def _gallery_menu(self, pos):
        it = self.gallery.itemAt(pos)
        if it is None:
            return
        key = it.data(Qt.UserRole)
        if key not in CUSTOM_CELLS:
            return
        menu = QMenu(self)
        act_edit = menu.addAction(tr("cell_edit"))
        act_del = menu.addAction(tr("cell_delete"))
        act = menu.exec(self.gallery.mapToGlobal(pos))
        if act == act_edit:
            self._open_cell_editor(key)
        elif act == act_del:
            if QMessageBox.question(
                    self, tr("cell_delete"),
                    tr("cell_confirm_del") % key) != QMessageBox.Yes:
                return
            delete_custom_cell(key)
            prev = self.factory.unit if self.factory is not None else 'square'
            if prev == key:
                prev = 'square'
            self._rebuild_gallery()
            self._select_gallery(prev)
            self.unit_combo.setCurrentText(unit_display(prev, get_lang()))

    def _toggle_edit(self):
        if self.deck.currentIndex() == 0:
            self._update_preview()
            self.deck.setCurrentIndex(1)
        else:
            self.deck.setCurrentIndex(0)
            self.push_spec()
        self.retranslate()

    def _clear_line(self):
        self.editor.clear()

    def _sync_row(self, *a):
        if len(self.disp_row.spins)!=2*self.pts.value():
            self.disp_row.set_pts(self.pts.value())
        self.disp_row.set_values(self.editor.disp)

    def _on_row_changed(self):
        self.editor.disp = self.disp_row.values()
        self.editor.update()
        self._on_edit_changed()

    def _on_edit_changed(self):
        self._parameter_basis=np.asarray(self.editor.disp,float)/(self.pert.value()/100. or 1.)
        self.disp_row.set_values(self.editor.disp)
        self._update_preview()
        self._debounce.start()

    # ---------------- data ----------------
    def _edit_rule(self):
        from .rule_dialog import RuleDialog
        dialog = RuleDialog(self.custom_rule, self)
        if dialog.exec():
            self.custom_rule = dialog.pattern()
            self.rule_combo.setCurrentIndex(self.rule_combo.findData('custom'))
            self._debounce.start()

    def collect_factory(self) -> StructureFactory:
        return StructureFactory(
            unit=unit_key(self.unit_combo.currentText(), get_lang()),
            grid_x=self.grid_x.value(), grid_y=self.grid_y.value(),
            n_pts_per_side=self.pts.value(),
            perturbation=self.pert.value() / 100.0,
            seed=self.seed.value(),
            expansion_rule=self.rule_combo.currentData(), custom_rule=self.custom_rule,
            line_displacements=self.editor.displacements(), spectrum_resolved=True).clamped()

    def load_spec(self, f: StructureFactory):
        """Programmatic entry (AI assistant / inverse design hand-off)."""
        blockers = [QSignalBlocker(w) for w in (self.unit_combo, self.spectrum_combo,
            self.rule_combo, self.grid_x, self.grid_y, self.pts, self.pert,
            self.seed, self.editor, self.disp_row)]
        self.rule_combo.setCurrentIndex(max(0, self.rule_combo.findData(f.expansion_rule)))
        self.custom_rule = f.custom_rule
        key, note = resolve_unit(f.unit)
        if note:
            f.unit = key       # retired unit: load the closest current one
        if key in SPECTRUM_PRESETS and key != 'square':
            self.unit_combo.setCurrentText(unit_display('square', get_lang()))
            self.spectrum_combo.setCurrentText(unit_display(key, get_lang()))
        else:
            self.unit_combo.setCurrentText(unit_display(key, get_lang()))
        self.grid_x.setValue(f.grid_x); self.grid_y.setValue(f.grid_y)
        self.pts.setValue(f.n_pts_per_side)
        self.pert.setValue(int(round(f.perturbation * 100)))
        self.seed.setValue(f.seed)
        self.editor.set_pts(f.n_pts_per_side)
        self.editor.disp = f.effective_spectrum().tolist()
        self._parameter_basis=np.asarray(self.editor.disp,float)/(self.pert.value()/100. or 1.)
        self._sync_row()
        self._sync_spectrum_visibility()
        self._update_preview()
        del blockers
        self.push_spec()
        if note:
            self.status.setText(tr("struct_synced") + " · " + note)

    def push_spec(self):
        self._debounce.stop()
        f = self.collect_factory()
        try:
            g = f.build()
            pos = np.asarray(g.node_positions(), float)[:, :2]
            edges = np.asarray(g.edge_array(), int)[:, :2]
            left, right = _grips(pos[:, 0], 0.05)
            self.canvas.set_static(pos, edges, left, right)
            self.factory = f
            self.stat_n_v.setText(str(pos.shape[0]))
            self.stat_e_v.setText(str(edges.shape[0]))
            self.stat_u_v.setText(unit_display(f.unit, get_lang()))
            self.status.setText(tr("struct_synced"))
            self._select_gallery(f.unit)
            self.structure_changed.emit(f)
        except Exception as e:
            self.status.setText(f"{tr('sim_failed')}: {e}")

    # ---------------- previews ----------------
    def _draw_network(self, pos, edges, w, h, line_w=1.1):
        c = colors(self.mode)
        pm = QPixmap(w, h)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        pad = 0.4
        x0, y0 = pos[:, 0].min() - pad, pos[:, 1].min() - pad
        s = min(w / (pos[:, 0].ptp() + 2 * pad),
                h / (pos[:, 1].ptp() + 2 * pad)) * 0.92

        def w2s(x, y):
            return ((x - x0) * s + (w - pos[:, 0].ptp() * s) / 2,
                    (y - y0) * s + (h - pos[:, 1].ptp() * s) / 2)
        p.setPen(QPen(QColor(c["accent2"]), line_w))
        for a, b in edges:
            xa, ya = w2s(*pos[a])
            xb, yb = w2s(*pos[b])
            p.drawLine(QPointF(xa, ya), QPointF(xb, yb))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(c["sub"]))
        for x, y in pos:
            px, py = w2s(x, y)
            p.drawEllipse(QPointF(px, py), 1.3, 1.3)
        p.end()
        return pm

    def _refresh_gallery(self):
        spec_key = unit_key(self.spectrum_combo.currentText(), get_lang())
        for i in range(self.gallery.count()):
            it = self.gallery.item(i)
            key = it.data(Qt.UserRole) or it.text()
            if key == "__new__":
                it.setIcon(self._plus_icon())
                continue
            try:
                ld = (SPECTRUM_PRESETS[spec_key]
                      if key == 'square' and spec_key != 'square' else None)
                f = StructureFactory(
                    unit=key, grid_x=1, grid_y=1,
                    n_pts_per_side=self.pts.value(),
                    perturbation=0.0,
                    seed=self.seed.value(),
                    line_displacements=ld).clamped()
                g = f.build()
                pos = np.asarray(g.node_positions(), float)[:, :2]
                edges = np.asarray(g.edge_array(), int)[:, :2]
                pm = self._draw_network(pos, edges, 70, 44)
            except Exception:
                continue
            it.setIcon(pm.scaled(70, 44, Qt.KeepAspectRatio,
                                 Qt.SmoothTransformation))

    def _update_preview(self):
        try:
            f = StructureFactory(
                unit=unit_key(self.unit_combo.currentText(), get_lang()),
                grid_x=1, grid_y=1,
                n_pts_per_side=self.pts.value(),
                perturbation=self.pert.value() / 100.0,
                seed=self.seed.value(),
                line_displacements=self.editor.displacements(), spectrum_resolved=True).clamped()
            g = f.build()
            pos = np.asarray(g.node_positions(), float)[:, :2]
            edges = np.asarray(g.edge_array(), int)[:, :2]
            w = max(90, self.unit_preview.width() - 8)
            h = max(70, self.unit_preview.height() - 8)
            pm = self._draw_unit_preview(pos, edges, w, h, line_w=1.4)
            self.unit_preview.setPixmap(pm)
        except Exception:
            self.unit_preview.setPixmap(QPixmap())

    def _draw_unit_preview(self, pos, edges, w, h, line_w=1.4):
        """Draw the deformed 1x1 cell over a dashed reference square.

        The combined bounds (base square + deformed nodes) are fitted,
        so a 100% (full-edge-length) point displacement remains visible
        relative to the undeformed square instead of being auto-zoomed away.
        """
        from fslab.structure import CELL as _CELL
        c = colors(self.mode)
        pm = QPixmap(w, h)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        base = np.array([[0.0, 0.0], [_CELL, 0.0], [_CELL, _CELL],
                         [0.0, _CELL]])
        if pos.size:
            lo = np.minimum(pos.min(0), base.min(0))
            hi = np.maximum(pos.max(0), base.max(0))
        else:
            lo, hi = base.min(0), base.max(0)
        span = np.maximum(hi - lo, 1e-9)
        scale = min(w, h) / max(span[0], span[1]) * 0.86
        cx = (w - span[0] * scale) / 2.0
        cy = (h - span[1] * scale) / 2.0

        def w2s(x, y):
            return (cx + (x - lo[0]) * scale, cy + (y - lo[1]) * scale)
        # dashed reference square
        p.setPen(QPen(QColor(c["faint"]), 1.0, Qt.DashLine))
        sq = [w2s(*q) for q in base]
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
            p.drawLine(QPointF(*sq[a]), QPointF(*sq[b]))
        # deformed edges
        p.setPen(QPen(QColor(c["accent2"]), line_w))
        for a, b in edges:
            p.drawLine(QPointF(*w2s(*pos[a])), QPointF(*w2s(*pos[b])))
        # nodes
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(c["sub"]))
        for q in pos:
            x, y = w2s(*q)
            p.drawEllipse(QPointF(x, y), 1.6, 1.6)
        p.end()
        return pm

    def _open_manufacturing(self):
        if hasattr(self, "manufacturing_host"):
            return self.manufacturing_host.open_source(False)
        dialog = getattr(self, 'manufacturing_dialog', None)
        if dialog is not None and (dialog.isVisible() or (dialog.worker is not None and dialog.worker.isRunning())):
            dialog.show()
            dialog.raise_()
            return
        from .manufacturing_dialog import ManufacturingDialog
        self.manufacturing_dialog = ManufacturingDialog(self.collect_factory(), mode=self.mode, parent=self)
        self.manufacturing_dialog.show()

    # ---------------- export ----------------
    def _do_export_json(self):
        path, _ = QFileDialog.getSaveFileName(
            self, tr("export_json_btn"),
            f"fiberscope_{unit_key(self.unit_combo.currentText(), get_lang())}.json",
            "JSON (*.json)")
        if not path:
            return
        try:
            export_json(self.collect_factory(), path)
            self.status.setText(f"{tr('exported')}: {path}")
        except Exception as e:
            self.status.setText(f"{tr('sim_failed')}: {e}")

    def _do_export_svg(self):
        path, _ = QFileDialog.getSaveFileName(
            self, tr("export_svg_btn"),
            f"fiberscope_{unit_key(self.unit_combo.currentText(), get_lang())}.svg",
            "SVG (*.svg)")
        if not path:
            return
        try:
            export_svg(self.collect_factory(), path, dark=self.mode == "dark")
            self.status.setText(f"{tr('exported')}: {path}")
        except Exception as e:
            self.status.setText(f"{tr('sim_failed')}: {e}")

    def _do_export_png(self, silent=None):
        """Raster of the big network canvas exactly as it is framed now."""
        stem = unit_key(self.unit_combo.currentText(), get_lang())
        path = ask_save(self, tr("export_png_btn"),
                        f"fiberscope_{stem}_network.png", "PNG (*.png)",
                        silent)
        if not path:
            return None
        try:
            save_widget_png(self.canvas, path)
            add_caption(path,
                        [f"FiberScope · {spec_text(self.collect_factory())}",
                         tr("ex_canvas_png")], self.mode)
        except Exception as e:
            self.status.setText(f"{tr('sim_failed')}: {e}")
            return None
        self.status.setText(f"{tr('exported')}: {path}")
        return path

    # ---------------- theme / i18n ----------------
    def set_mode(self, mode):
        self.mode = mode
        self.canvas.set_mode(mode)
        self.editor.mode = mode
        self.editor.update()
        self._refresh_gallery()
        self._update_preview()

    def retranslate(self):
        self.custom_label.setText(tr('cell_saved'))
        self.custom_edit.setText(tr('cell_edit'))
        self.custom_delete.setText(tr('cell_delete'))
        self._refresh_custom_shelf()
        self.lbl_rule.setText(tr('expansion_rule'))
        self.rule_edit.setText(tr('rule_edit'))
        self.workbench_btn.setText(tr('cell_workbench'))
        for i in range(self.rule_combo.count()):
            self.rule_combo.setItemText(i, tr('rule_' + self.rule_combo.itemData(i)))
        self.gb_params.setTitle(tr("struct_params"))
        self.gb_stats.setTitle(tr("struct_stats"))
        self.gb_gallery.setTitle(tr("struct_gallery"))
        self.lbl_unit.setText(tr("unit"))
        self.lbl_spectrum.setText(tr("spectrum"))
        self.lbl_grid.setText(tr("grid"))
        self.lbl_pts.setText(tr("pts_per_side"))
        self.lbl_pert.setText(tr("perturbation"))
        self.lbl_seed.setText(tr("seed"))
        self.dice_btn.setText(tr("dice_btn"))
        self.edit_btn.setText(tr("edit_line"))
        self.done_btn.setText(tr("edit_done"))
        self.clear_btn.setText(tr("disp_clear"))
        self.edit_hint.setText(tr("edit_line_hint"))
        self.lbl_disp_row.setText(tr("disp_row"))
        self.manufacturing_btn.setText("连续制造 · 2D / 3D" if get_lang() == "zh" else "Fabrication · 2D / 3D")
        self.export_json_btn.setText(tr("export_json_btn"))
        self.export_svg_btn.setText(tr("export_svg_btn"))
        self.export_png_btn.setText(tr("export_png_btn"))
        self.lbl_preview.setText(tr("prim_preview"))
        self.stat_n_l.setText("NODES")
        self.stat_e_l.setText("EDGES")
        self.stat_u_l.setText("UNIT")
        # unit display names are language-dependent: rewrite in place so the
        # current selection survives a language switch
        cur_u = unit_key(self.unit_combo.currentText(), get_lang())
        for i, k in enumerate(all_unit_keys()):
            self.unit_combo.setItemText(i, unit_display(k, get_lang()))
        self.unit_combo.setCurrentText(unit_display(cur_u, get_lang()))
        cur_s = unit_key(self.spectrum_combo.currentText(), get_lang())
        for i, k in enumerate(SPECTRUM_PRESETS):
            self.spectrum_combo.setItemText(i, unit_display(k, get_lang()))
        self.spectrum_combo.setCurrentText(unit_display(cur_s, get_lang()))
        for i in range(self.gallery.count()):
            it = self.gallery.item(i)
            key = it.data(Qt.UserRole)
            it.setText(tr("cell_new") if key == "__new__"
                       else unit_display(key, get_lang()))
        if self.factory is not None:
            self.stat_u_v.setText(
                unit_display(self.factory.unit, get_lang()))
        if self.deck.currentIndex() == 0:
            self.edit_btn.setText(tr("edit_line"))
