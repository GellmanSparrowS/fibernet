"""MainWindow: branded header + tab container + shared structure wiring."""
import os
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QApplication, QComboBox, QFrame, QHBoxLayout,
                               QLabel, QMainWindow, QPushButton, QSplitter,
                               QTabWidget, QVBoxLayout, QWidget)


def _icon_path():
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(base, "assets", "icon.ico")
    return p if os.path.exists(p) else None

from .i18n import set_lang, tr
from .structure_tab import StructureTab
from .sim_tab import SimTab
from .design_tab import DesignTab
from .replay_tab import ReplayTab
from .features_tab import FeaturesTab
from .surface_tab import SurfaceTab
from .ml_tab import MLTab
from .ai_assistant import AIPanel
from .theme import build_palette, build_qss


class MainWindow(QMainWindow):
    def __init__(self, lang: str = "zh", mode: str = "light"):
        super().__init__()
        set_lang(lang)
        self.mode = mode
        self.resize(1360, 840)
        self.setMinimumSize(1120, 700)

        # ---- header ----
        header = QFrame()
        header.setObjectName("header")
        hh = QHBoxLayout(header)
        hh.setContentsMargins(18, 10, 18, 10)
        hh.setSpacing(12)
        self.logo = QLabel("⬡")
        self.logo.setObjectName("logo")
        titles = QVBoxLayout()
        titles.setSpacing(1)
        self.title = QLabel()
        self.title.setObjectName("title")
        self.tagline = QLabel()
        self.tagline.setObjectName("tagline")
        titles.addWidget(self.title)
        titles.addWidget(self.tagline)
        hh.addWidget(self.logo)
        hh.addLayout(titles)
        hh.addStretch(1)
        self.ver_chip = QLabel()
        self.ver_chip.setObjectName("chip")
        hh.addWidget(self.ver_chip)
        self.ai_btn = QPushButton()
        self.ai_btn.setProperty("primary", True)
        self.ai_btn.clicked.connect(self._toggle_ai)
        hh.addWidget(self.ai_btn)
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["中文", "English"])
        self.lang_combo.setCurrentIndex(0 if lang == "zh" else 1)
        self.lang_combo.currentIndexChanged.connect(self._on_lang)
        self.theme_btn = QPushButton()
        self.theme_btn.clicked.connect(self._toggle_theme)
        hh.addWidget(self.lang_combo)
        hh.addWidget(self.theme_btn)

        # ---- tabs ----
        self.tabs = QTabWidget()
        self.tab_struct = StructureTab(mode=self.mode)
        self.tab_sim = SimTab(mode=self.mode)
        self.tab_perc = self.tab_sim   # compat alias
        self.tab_stretch = self.tab_sim
        self.tab_design = DesignTab(mode=self.mode)
        self.tab_replay = ReplayTab(mode=self.mode)
        self.tab_features = FeaturesTab(mode=self.mode)
        self.tab_surface = SurfaceTab(mode=self.mode)
        self.tab_ml = MLTab(mode=self.mode)
        # order: structure / sim / features / ml / design(+replay) / surface
        self.tabs.addTab(self.tab_struct, "")
        self.tabs.addTab(self.tab_sim, "")
        self.tabs.addTab(self.tab_features, "")
        self.tabs.addTab(self.tab_ml, "")
        self.tabs.addTab(self.tab_design, "")
        self.tabs.addTab(self.tab_surface, "")
        self.tab_design.embed_replay(self.tab_replay)
        self.tab_struct.structure_changed.connect(
            self.tab_sim.set_spec)
        self.tab_struct.structure_changed.connect(
            self.tab_design.set_spec)
        self.tab_design.apply_structure.connect(
            self.tab_struct.load_spec)
        self.tab_replay.apply_structure.connect(
            self.tab_struct.load_spec)
        self.tab_struct.structure_changed.connect(
            self.tab_features.set_factory)
        if hasattr(self.tab_surface, "apply_structure"):
            self.tab_struct.structure_changed.connect(
                self.tab_surface.apply_structure)

        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(header)
        self.ai_panel = AIPanel(self)
        self.ai_dock = self.ai_panel  # legacy alias (tests)
        self.split = QSplitter(Qt.Horizontal)
        self.split.addWidget(self.tabs)
        self.split.addWidget(self.ai_panel)
        self.split.setStretchFactor(0, 1)
        self.split.setStretchFactor(1, 0)
        self.split.setSizes([860, 500])
        self.ai_panel.hide()
        outer.addWidget(self.split, 1)
        self.setCentralWidget(central)

        ic = _icon_path()
        if ic:
            self.setWindowIcon(QIcon(ic))
        self._apply_theme()
        self.retranslate()

    def _apply_theme(self):
        app = QApplication.instance()
        app.setPalette(build_palette(self.mode))
        app.setStyleSheet(build_qss(self.mode))
        for t in (self.tab_struct, self.tab_sim,
                  self.tab_design, self.tab_replay, self.tab_features,
                  self.tab_surface, self.tab_ml):
            if t is not None:
                t.set_mode(self.mode)
        self.ai_panel.refresh_theme()

    def _toggle_theme(self):
        self.mode = "light" if self.mode == "dark" else "dark"
        self._apply_theme()
        self.retranslate()

    def _toggle_ai(self):
        self.ai_panel.setVisible(not self.ai_panel.isVisible())

    def _on_lang(self, idx):
        set_lang("zh" if idx == 0 else "en")
        self.retranslate()

    def retranslate(self):
        self.setWindowTitle(tr("app_title"))
        self.title.setText("FiberScope")
        self.tagline.setText(tr("tagline"))
        self.ver_chip.setText(tr("version_chip"))
        self.tabs.setTabText(0, tr("tab_structure"))
        self.tabs.setTabText(1, tr("tab_sim"))
        self.tabs.setTabText(2, tr("tab_features"))
        self.tabs.setTabText(3, tr("tab_ml"))
        self.tabs.setTabText(4, tr("tab_design"))
        self.tabs.setTabText(5, tr("tab_surface"))
        self.theme_btn.setText(tr("theme_btn") if self.mode == "dark"
                               else "Dark")
        self.ai_btn.setText(tr("ai_btn"))
        self.ai_panel.retranslate()
        self.tab_struct.retranslate()
        self.tab_sim.retranslate()
        self.tab_design.retranslate()
        self.tab_replay.retranslate()
        self.tab_features.retranslate()
        self.tab_surface.retranslate()
        self.tab_ml.retranslate()
