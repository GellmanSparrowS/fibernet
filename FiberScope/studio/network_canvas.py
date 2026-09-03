"""QPainter network canvas with pixmap scene cache.

Modes:
  static       : as-designed structure preview (structure tab)
  percolation  : inactive gray / active dim blue / spanning backbone
                 blue at grips -> cyan mid-sample (two-sided front)
  strain       : diverging colormap (cyan compression -> gray -> red tension)
                 plus red markers at fresh contact points
A small legend chip is drawn in the top-right corner.
"""
import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from .theme import colors


def _lerp_color(c1: QColor, c2: QColor, t: float) -> QColor:
    t = float(max(0.0, min(1.0, t)))
    return QColor(int(c1.red() + (c2.red() - c1.red()) * t),
                  int(c1.green() + (c2.green() - c1.green()) * t),
                  int(c1.blue() + (c2.blue() - c1.blue()) * t))


class NetworkCanvas(QWidget):
    def __init__(self, mode: str = "dark", parent=None):
        super().__init__(parent)
        self.mode = mode
        self.color_mode = "percolation"
        self.run = None
        self.perc = None
        self.frame = 0
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.static = None          # dict(pos, edges, left, right)
        self.legend = []            # list of (color_hex, text)
        self._cache = None
        self._cache_key = None
        self.setMinimumSize(360, 300)
        self.setMouseTracking(True)

    # ---------------- data ----------------
    def set_mode(self, mode: str):
        self.mode = mode
        self._cache_key = None
        self.update()

    def set_color_mode(self, cm: str):
        self.color_mode = cm
        self._cache_key = None
        self.update()

    def set_legend(self, entries):
        self.legend = list(entries)
        self.update()

    def set_static(self, pos, edges, left=None, right=None):
        self.static = dict(pos=np.asarray(pos, float),
                           edges=np.asarray(edges, int).reshape(-1, 2),
                           left=None if left is None else np.asarray(left),
                           right=None if right is None else np.asarray(right))
        self.run = None
        self.perc = None
        self._cache_key = None
        self.update()

    def clear_static(self):
        self.static = None
        self._cache_key = None
        self.update()

    def set_data(self, run, perc):
        self.run = run
        self.perc = perc
        self.frame = 0
        self.zoom, self.pan_x, self.pan_y = 1.0, 0.0, 0.0
        self._contact_cum = None
        allxy = np.concatenate([np.asarray(f, float)
                                for f in run.frames_xy])
        pad = 0.6
        self._run_rect = QRectF(
            float(allxy[:, 0].min()) - pad, float(allxy[:, 1].min()) - pad,
            float(allxy[:, 0].ptp()) + 2 * pad,
            float(allxy[:, 1].ptp()) + 2 * pad)
        cf = getattr(run, "contact_frames", None)
        if cf is not None:
            cum, acc = [], set()
            for pairs in cf:
                acc |= set((int(a), int(b)) for a, b in np.asarray(pairs,
                           dtype=int).reshape(-1, 2)) if len(pairs) else acc
                cum.append(frozenset(acc))
            self._contact_cum = cum
        self._cache_key = None
        self.update()

    def set_frame(self, f: int):
        if self.run is None:
            return
        f = max(0, min(self.run.n_frames - 1, int(f)))
        if f != self.frame:
            self.frame = f
            self._cache_key = None
            self.update()

    # ---------------- transform ----------------
    def _world_rect(self) -> QRectF:
        if self.run is not None:
            # fixed viewport over ALL frames: playback shows the sample
            # stretching left->right inside a stable window (no re-zoom)
            return self._run_rect
        if self.static is not None:
            xy = self.static["pos"]
            pad = 0.6
            return QRectF(xy[:, 0].min() - pad, xy[:, 1].min() - pad,
                          xy[:, 0].ptp() + 2 * pad, xy[:, 1].ptp() + 2 * pad)
        return QRectF(0, 0, 1, 1)

    def _transform(self):
        wr = self._world_rect()
        w, h = self.width(), self.height()
        s = min(w / wr.width(), h / wr.height()) * 0.94 * self.zoom
        cx = wr.x() + wr.width() / 2
        cy = wr.y() + wr.height() / 2
        ox = w / 2 - cx * s + self.pan_x
        oy = h / 2 - cy * s + self.pan_y
        return s, ox, oy

    # ---------------- render ----------------
    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(colors(self.mode)["bg"]))
        if self.run is None and self.static is None:
            p.end()
            return
        key = (id(self.run), id(self.perc), id(self.static), self.frame,
               self.width(), self.height(), self.zoom, self.pan_x,
               self.pan_y, self.mode, self.color_mode)
        if self._cache is None or self._cache_key != key:
            self._cache = self._render_scene()
            self._cache_key = key
        p.drawPixmap(0, 0, self._cache)
        self._draw_overlay(p)
        self._draw_legend(p)
        p.end()

    def _render_scene(self) -> QPixmap:
        dpr = self.devicePixelRatioF()
        pm = QPixmap(int(self.width() * dpr), int(self.height() * dpr))
        pm.setDevicePixelRatio(dpr)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        s, ox, oy = self._transform()
        c = colors(self.mode)

        def w2s(x, y):
            return QPointF(float(x) * s + ox, float(y) * s + oy)

        if self.run is None:
            self._draw_static(p, w2s, s, c)
            p.end()
            return pm

        run, perc = self.run, self.perc
        f = self.frame
        xy = run.frames_xy[f]
        edges = run.edges

        def pt(i):
            return w2s(xy[i, 0], xy[i, 1])

        if self.color_mode == "strain":
            self._draw_strain_edges(p, pt, run, f)
        else:
            blue = QColor(c["accent"])
            cyan = QColor("#7ef0ff")
            dim = QColor("#5b7fb9")
            inactive = QColor(c["edge_inactive"])
            in_span = perc.edge_in_spanning[f] if perc is not None else None
            active = perc.active_mask(f) if perc is not None else None
            depth = perc.edge_depth_norm[f] if perc is not None else None

            p.setPen(QPen(inactive, 1.2))
            for k in range(run.n_edges):
                if active is not None and active[k]:
                    continue
                p.drawLine(pt(edges[k, 0]), pt(edges[k, 1]))
            if active is not None:
                p.setPen(QPen(dim, 1.8))
                for k in range(run.n_edges):
                    if active[k] and not in_span[k]:
                        p.drawLine(pt(edges[k, 0]), pt(edges[k, 1]))
                for k in range(run.n_edges):
                    if in_span[k]:
                        col = _lerp_color(blue, cyan, depth[k])
                        p.setPen(QPen(col, 2.6))
                        p.drawLine(pt(edges[k, 0]), pt(edges[k, 1]))

        # nodes (skip individual dots on large nets; they overlap and
        # dominate paint time for no visual gain)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(c["sub"]))
        r = max(1.6, 0.05 * s)
        if xy.shape[0] <= 800:
            for i in range(xy.shape[0]):
                p.drawEllipse(pt(i), r, r)
        # grips
        p.setBrush(QColor(c["warn"]))
        for i in np.concatenate([run.left_nodes, run.right_nodes]):
            p.drawEllipse(pt(int(i)), r * 1.7, r * 1.7)
        # contacts: history dim, fresh bright
        cf = getattr(run, "contact_frames", None)
        if cf is not None and f < len(cf):
            fresh = set((int(a), int(b)) for a, b in
                        np.asarray(cf[f], dtype=int).reshape(-1, 2)) \
                if len(cf[f]) else set()
            if self._contact_cum is not None and f < len(self._contact_cum):
                old_pairs = set(self._contact_cum[f]) - fresh
                p.setBrush(QColor(255, 59, 48, 80))
                for a, b in old_pairs:
                    mx = (xy[a, 0] + xy[b, 0]) / 2
                    my = (xy[a, 1] + xy[b, 1]) / 2
                    p.drawEllipse(w2s(mx, my), 3.5, 3.5)
            if fresh:
                p.setBrush(QColor("#ff3b30"))
                for a, b in fresh:
                    mx = (xy[a, 0] + xy[b, 0]) / 2
                    my = (xy[a, 1] + xy[b, 1]) / 2
                    p.drawEllipse(w2s(mx, my), 4.5, 4.5)
        p.end()
        return pm

    def _draw_static(self, p, w2s, s, c):
        st = self.static
        p.setPen(QPen(QColor(c["line2"]), max(1.2, 0.035 * s)))
        for a, b in st["edges"]:
            p.drawLine(w2s(*st["pos"][a]), w2s(*st["pos"][b]))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(c["sub"]))
        r = max(1.4, 0.045 * s)
        if len(st["pos"]) <= 800:
            for x, y in st["pos"]:
                p.drawEllipse(w2s(x, y), r, r)
        if st["left"] is not None:
            p.setBrush(QColor(c["warn"]))
            for i in np.concatenate([st["left"], st["right"]]):
                x, y = st["pos"][int(i)]
                p.drawEllipse(w2s(x, y), r * 1.7, r * 1.7)

    def _draw_strain_edges(self, p, pt, run, f):
        sf = run.edge_strain[f]
        scale = float(np.percentile(np.abs(sf), 95)) or 1.0
        cold = QColor("#4dd0e1")
        mid = QColor(colors(self.mode)["edge_inactive"])
        hot = QColor("#ff5d47")
        for k in range(run.n_edges):
            v = max(-1.0, min(1.0, float(sf[k]) / scale))
            col = _lerp_color(mid, hot, v) if v >= 0 else \
                _lerp_color(mid, cold, -v)
            p.setPen(QPen(col, 1.4 + 1.6 * abs(v)))
            p.drawLine(pt(run.edges[k, 0]), pt(run.edges[k, 1]))

    def _draw_overlay(self, p: QPainter):
        if self.run is None:
            return
        c = colors(self.mode)
        lines = [f"{self.run.strain_levels[self.frame]:.3f}"]
        if self.perc is not None and self.color_mode == "percolation":
            lines.append(f"P={self.perc.spanning_frac[self.frame]:.3f}")
        p.setPen(QColor(c["sub"]))
        f = p.font()
        f.setPointSize(10)
        p.setFont(f)
        for i, t in enumerate(lines):
            p.drawText(10, 18 + 16 * i, t)

    def _draw_legend(self, p: QPainter):
        if not self.legend:
            return
        c = colors(self.mode)
        p.save()
        f = p.font()
        f.setPointSize(9)
        p.setFont(f)
        fm = p.fontMetrics()
        w = max(fm.horizontalAdvance(t) for _, t in self.legend) + 34
        h = 18 * len(self.legend) + 10
        x0 = self.width() - w - 10
        y0 = 10
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(c["card"] + "dd") if self.mode == "dark"
                   else QColor("#ffffffee"))
        p.drawRoundedRect(x0, y0, w, h, 7, 7)
        p.setPen(QColor(c["line2"]))
        p.drawRoundedRect(x0, y0, w, h, 7, 7)
        for i, (col, t) in enumerate(self.legend):
            y = y0 + 14 + 18 * i
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(col))
            p.drawRoundedRect(x0 + 8, y - 7, 14, 4, 2, 2)
            p.setPen(QColor(c["sub"]))
            p.drawText(x0 + 28, y, t)
        p.restore()

    # ---------------- interaction ----------------
    def wheelEvent(self, ev):
        factor = 1.15 if ev.angleDelta().y() > 0 else 1 / 1.15
        self.zoom = max(0.2, min(20.0, self.zoom * factor))
        self._cache_key = None
        self.update()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._drag = ev.position()

    def mouseMoveEvent(self, ev):
        if hasattr(self, "_drag") and ev.buttons() & Qt.LeftButton:
            d = ev.position() - self._drag
            self.pan_x += d.x()
            self.pan_y += d.y()
            self._drag = ev.position()
            self._cache_key = None
            self.update()

    def mouseReleaseEvent(self, ev):
        if hasattr(self, "_drag"):
            del self._drag
