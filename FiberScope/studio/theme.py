"""Visual system: layered dark/light palettes + one comprehensive QSS.

Design language: deep graphite-blue surfaces, cyan accent, rounded cards,
pill chips, gradient primary actions. Artwork colors (canvas) live in
network_canvas and read from here.
"""
from PySide6.QtGui import QColor, QPalette

DARK = dict(
    bg="#0a0e15", panel="#0f1520", card="#131b29", card2="#182233",
    line="#22304a", line2="#31446a", text="#e8edf6", sub="#93a1b8",
    faint="#5c6b84", accent="#38bdf8", accent2="#22d3ee", violet="#818cf8",
    warn="#fbbf24", hot="#ff5d47", ok="#34d399", edge_inactive="#3a465c",
    handle="#f2f6fc",
)
LIGHT = dict(
    bg="#eef2f8", panel="#ffffff", card="#ffffff", card2="#e8eef7",
    line="#d5deeb", line2="#b7c6dd", text="#16233a", sub="#5a6b85",
    faint="#8fa0b8", accent="#0284c7", accent2="#0891b2", violet="#6366f1",
    warn="#d97706", hot="#e11d48", ok="#059669", edge_inactive="#c3cdda",
    handle="#ffffff",
)


def colors(mode: str) -> dict:
    return DARK if mode == "dark" else LIGHT


def build_palette(mode: str) -> QPalette:
    c = colors(mode)
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(c["bg"]))
    pal.setColor(QPalette.WindowText, QColor(c["text"]))
    pal.setColor(QPalette.Base, QColor(c["card2"]))
    pal.setColor(QPalette.AlternateBase, QColor(c["card"]))
    pal.setColor(QPalette.Text, QColor(c["text"]))
    pal.setColor(QPalette.Button, QColor(c["card"]))
    pal.setColor(QPalette.ButtonText, QColor(c["text"]))
    pal.setColor(QPalette.Highlight, QColor(c["accent"]))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ToolTipBase, QColor(c["card"]))
    pal.setColor(QPalette.ToolTipText, QColor(c["text"]))
    pal.setColor(QPalette.PlaceholderText, QColor(c["faint"]))
    return pal


def apply_plot_theme(pw, mode: str):
    """Theme a pyqtgraph PlotWidget: card background + sub-colored axes."""
    import pyqtgraph as pg
    c = colors(mode)
    pw.setBackground(c["card2"])
    pi = pw.getPlotItem()
    for name in ("left", "bottom"):
        ax = pi.getAxis(name)
        ax.setPen(pg.mkPen(c["sub"]))
        ax.setTextPen(pg.mkPen(c["sub"]))


def build_qss(mode: str) -> str:
    c = colors(mode)
    return f"""
    * {{ font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif; }}
    QMainWindow, QDialog {{ background: {c['bg']}; }}
    QWidget {{ color: {c['text']}; font-size: 13px; }}

    /* ---------- header ---------- */
    QFrame#header {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 {c['panel']}, stop:1 {c['bg']});
        border-bottom: 1px solid {c['line']};
    }}
    QLabel#logo {{ color: {c['accent2']}; font-size: 26px; font-weight: 800; }}
    QLabel#title {{ font-size: 19px; font-weight: 700; letter-spacing: 0.5px; }}
    QLabel#tagline {{ color: {c['sub']}; font-size: 12px; }}
    QLabel#chip {{
        background: {c['card2']}; border: 1px solid {c['line']};
        border-radius: 9px; padding: 3px 10px; color: {c['sub']};
        font-size: 11px;
    }}
    QLabel#chip_accent {{
        background: {c['accent']}22; border: 1px solid {c['accent']}66;
        border-radius: 9px; padding: 3px 10px; color: {c['accent']};
        font-size: 11px; font-weight: 600;
    }}

    /* ---------- tabs ---------- */
    QTabWidget::pane {{ border: none; background: {c['bg']}; }}
    QTabBar {{ background: transparent; }}
    QTabBar::tab {{
        background: transparent; border: none; padding: 11px 20px 9px 20px;
        color: {c['sub']}; font-size: 13px; font-weight: 600;
        border-bottom: 2px solid transparent;
    }}
    QTabBar::tab:hover {{ color: {c['text']}; }}
    QTabBar::tab:selected {{
        color: {c['accent2']};
        border-bottom: 2px solid {c['accent2']};
    }}

    /* ---------- cards ---------- */
    QGroupBox {{
        background: {c['card']}; border: 1px solid {c['line']};
        border-radius: 10px; margin-top: 16px; padding: 14px 10px 10px 10px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin; left: 12px; padding: 0 6px;
        color: {c['sub']}; font-size: 11px; font-weight: 700;
        letter-spacing: 1px;
    }}
    QFrame#card {{
        background: {c['card']}; border: 1px solid {c['line']};
        border-radius: 10px;
    }}
    QFrame#well {{
        background: {c['card2']}; border: 1px solid {c['line']};
        border-radius: 8px;
    }}

    /* ---------- buttons ---------- */
    QPushButton {{
        background: transparent; color: {c['text']};
        border: 1px solid {c['line2']}; border-radius: 8px;
        padding: 7px 14px; font-weight: 600;
    }}
    QPushButton:hover {{ border-color: {c['accent']}; color: {c['accent']}; }}
    QPushButton:disabled {{ color: {c['faint']}; border-color: {c['line']}; }}
    QPushButton[primary="true"] {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {c['accent']}, stop:1 {c['accent2']});
        border: none; color: #04121f; font-weight: 700; padding: 9px 18px;
    }}
    QPushButton[primary="true"]:hover {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {c['accent2']}, stop:1 {c['accent']});
        color: #04121f;
    }}
    QPushButton[primary="true"]:disabled {{
        background: {c['card2']}; color: {c['faint']};
    }}
    QPushButton#gallery {{
        background: {c['card2']}; border: 1px solid {c['line']};
        border-radius: 8px; padding: 2px;
    }}
    QPushButton#gallery:hover {{ border-color: {c['line2']}; }}
    QPushButton#gallery[selected="true"] {{
        border: 1px solid {c['accent2']}; background: {c['accent2']}18;
    }}

    /* ---------- inputs ---------- */
    QComboBox, QSpinBox, QDoubleSpinBox {{
        background: {c['card2']}; border: 1px solid {c['line']};
        border-radius: 7px; padding: 4px 8px; min-height: 22px;
        selection-background-color: {c['accent']};
    }}
    QSpinBox, QDoubleSpinBox {{ min-width: 68px; }}
    QSpinBox::up-button, QSpinBox::down-button,
    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
        width: 20px; background: transparent; border: none;
    }}
    QSpinBox::up-button:hover, QSpinBox::down-button:hover,
    QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
        background: {c['accent']}22;
    }}
    QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {c['accent']};
    }}
    QComboBox::drop-down {{ border: none; width: 22px; }}
    QComboBox QAbstractItemView {{
        background: {c['card']}; border: 1px solid {c['line2']};
        selection-background-color: {c['accent']}33;
        selection-color: {c['text']}; padding: 4px; outline: none;
    }}
    QCheckBox {{ spacing: 8px; }}
    QCheckBox::indicator {{
        width: 16px; height: 16px; border-radius: 5px;
        border: 1px solid {c['line2']}; background: {c['card2']};
    }}
    QCheckBox::indicator:checked {{
        background: {c['accent']}; border-color: {c['accent']};
    }}

    /* ---------- slider ---------- */
    QSlider::groove:horizontal {{
        height: 6px; background: {c['card2']};
        border: 1px solid {c['line']}; border-radius: 3px;
    }}
    QSlider::sub-page:horizontal {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {c['accent']}, stop:1 {c['accent2']});
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        width: 16px; height: 16px; margin: -6px 0; border-radius: 9px;
        background: {c['handle']}; border: 3px solid {c['accent']};
    }}
    QSlider::handle:horizontal:hover {{ border-color: {c['accent2']}; }}

    /* ---------- lists / scroll ---------- */
    QListWidget {{
        background: {c['card2']}; border: 1px solid {c['line']};
        border-radius: 8px; padding: 4px; outline: none;
    }}
    QListWidget::item {{ padding: 5px 8px; border-radius: 5px; }}
    QListWidget::item:selected {{ background: {c['accent']}26; color: {c['accent']}; }}
    QScrollBar:vertical {{
        background: transparent; width: 10px; margin: 2px;
    }}
    QScrollBar::handle:vertical {{
        background: {c['line2']}; border-radius: 4px; min-height: 28px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {c['faint']}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
    QScrollBar::handle:horizontal {{
        background: {c['line2']}; border-radius: 4px; min-width: 28px;
    }}
    QSplitter::handle {{ background: {c['line']}; width: 2px; height: 2px; }}
    QWidget#aipanel {{
        background: {c['panel']}; border-left: 1px solid {c['line']};
    }}
    QTextBrowser#aichat {{
        background: {c['card2']}; border: 1px solid {c['line']};
        border-radius: 10px; padding: 8px;
    }}
    QListWidget#gallery {{
        background: {c['card2']}; border: 1px solid {c['line']};
        border-radius: 8px; padding: 4px; outline: none;
    }}
    QListWidget#gallery::item {{ border-radius: 8px; }}
    QListWidget#gallery::item:selected {{ background: {c['accent2']}22; }}

    /* ---------- progress ---------- */
    QProgressBar {{
        background: {c['card2']}; border: 1px solid {c['line']};
        border-radius: 6px; text-align: center; color: {c['sub']};
        font-size: 10px;
    }}
    QProgressBar::chunk {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {c['accent']}, stop:1 {c['accent2']});
        border-radius: 5px;
    }}

    /* ---------- misc ---------- */
    QLabel {{ background: transparent; }}
    QLabel#subtitle {{ color: {c['sub']}; font-size: 12px; }}
    QLabel#hint {{ color: {c['faint']}; font-size: 11px; }}
    QLabel#explain {{
        background: {c['card']}; border: 1px solid {c['line']};
        border-radius: 10px; padding: 10px 12px; color: {c['sub']};
        font-size: 12px;
    }}
    QLabel#stat_value {{ font-size: 17px; font-weight: 700; color: {c['accent2']}; }}
    QLabel#stat_label {{ color: {c['faint']}; font-size: 10px;
        letter-spacing: 1px; }}
    QLabel#statusbar {{ color: {c['sub']}; font-size: 12px; padding: 2px 8px; }}
    QStatusBar {{ background: {c['panel']}; border-top: 1px solid {c['line']}; }}
    QToolTip {{
        background: {c['card']}; color: {c['text']};
        border: 1px solid {c['line2']}; border-radius: 6px; padding: 6px;
    }}
    """
