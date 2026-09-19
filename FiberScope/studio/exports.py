"""Shared save-dialog and raster export helpers for the GUI tabs.

fslab.exporter stays Qt-free (pure CSV/JSON/SVG writers); this module is the
thin Qt layer on top: one dialog helper that tests and AI tools can bypass
with `silent`, plus PNG grabbing for pyqtgraph plots and plain widgets.
"""
import os

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QFileDialog

PNG_W = 1600          # plot raster width for slides; height follows aspect


def ask_save(parent, title, default_name, filt, silent=None):
    """Chosen path, or None if cancelled. `silent` skips the dialog."""
    if silent:
        d = os.path.dirname(os.path.abspath(silent))
        if d:
            os.makedirs(d, exist_ok=True)
        return silent
    path, _ = QFileDialog.getSaveFileName(parent, title, default_name, filt)
    return path or None


def save_plot_png(plot_widget, path, width=PNG_W):
    """Rasterize a pyqtgraph PlotWidget at a fixed resolution.

    Deliberately NOT pyqtgraph.exporters.ImageExporter: importing that
    package pulls HDF5Exporter -> h5py, which the frozen build excludes
    (check_runtime_deps gate).  The PlotItem gets a temporary fixed size so
    the output is identical whether or not the tab is on screen (a hidden
    tab never lays out, and its auto-range sceneRect is meaningless).
    """
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtWidgets import QApplication
    item = plot_widget.plotItem
    w = int(width)
    h = int(round(w * 0.62))
    item.setMinimumSize(w, h)
    item.setMaximumSize(w, h)
    QApplication.processEvents()
    try:
        src = item.sceneBoundingRect()
        img = QImage(w, h, QImage.Format_ARGB32)
        brush = plot_widget.backgroundBrush()
        img.fill(brush.color() if brush.style() != Qt.NoBrush
                 else QColor("#0f1520"))
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        item.scene().render(p, QRectF(img.rect()), src)
        p.end()
    finally:
        # hand the layout back to the live view
        item.setMinimumSize(0, 0)
        item.setMaximumSize(16777215, 16777215)
    if not img.save(path):
        raise IOError(f"cannot write {path}")
    return path


def save_widget_png(widget, path):
    """Pixel-exact grab of any widget (network canvas, fingerprint, ...)."""
    pm = widget.grab()
    if pm.isNull() or not pm.save(path):
        raise IOError(f"cannot write {path}")
    return path


def unique_stem(prefix, unit="", ext=""):
    """`fiberscope_chiral_curve.csv` style default file name."""
    parts = [p for p in ("fiberscope", unit, prefix) if p]
    return "_".join(parts) + (f".{ext}" if ext else "")


def add_caption(path, lines, mode="dark"):
    """Re-save `path` with a caption band on top so an exported figure is
    self-describing in a slide deck (structure spec + what the panel shows).
    Text is drawn with the app font; offscreen test runs have no fonts, so
    tests must not assert on the caption glyphs."""
    from PySide6.QtGui import QColor, QFont, QImage, QPainter
    from .theme import colors
    src = QImage(path)
    if src.isNull():
        raise IOError(f"cannot read {path}")
    c = colors(mode)
    fs = max(14, src.width() // 90)
    step = int(fs * 1.75)
    band = step * len(lines) + 12
    out = QImage(src.width(), src.height() + band, QImage.Format_ARGB32)
    out.fill(QColor(c["bg"]))
    p = QPainter(out)
    f = QFont()
    f.setPixelSize(fs)
    for i, text in enumerate(lines):
        f.setBold(i == 0)
        p.setFont(f)
        p.setPen(QColor(c["text"] if i == 0 else c["sub"]))
        p.drawText(12, step * (i + 1) + 4, text)
    p.drawImage(0, band, src)
    p.end()
    if not out.save(path):
        raise IOError(f"cannot write {path}")
    return path
