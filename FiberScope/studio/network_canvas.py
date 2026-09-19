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
from PySide6.QtCore import Qt, QLineF, QPointF, QRectF
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
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
        c = colors(self.mode)
        img = QImage(max(1, int(round(self.width() * dpr))),
                     max(1, int(round(self.height() * dpr))),
                     QImage.Format_RGB32)
        img.setDevicePixelRatio(dpr)
        img.fill(QColor(c["bg"]).rgb())
        p = QPainter(img)
        # thousands of thin edges: source-over blending on an alpha pixmap
        # plus antialiasing costs ~5-20x the raster time of an opaque
        # non-AA pass and buys nothing visible at dpr >= 1; the few dots
        # and contact markers below keep antialiasing
        p.setRenderHint(QPainter.Antialiasing, False)
        s, ox, oy = self._transform()

        if self.run is None:
            self._draw_static(p, s, ox, oy, c)
            p.end()
            return QPixmap.fromImage(img)

        run, perc = self.run, self.perc
        f = self.frame
        S = np.asarray(run.frames_xy[f], float) * s + (ox, oy)
        if self.color_mode == "strain":
            groups = self._strain_groups(run, f, c)
        else:
            groups = self._perc_groups(run, perc, f, c)
        self._draw_edge_groups(p, S, run.edges, groups)

        # nodes (skip individual dots on large nets; they overlap and
        # dominate paint time for no visual gain)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(c["sub"]))
        r = max(1.6, 0.05 * s)
        if S.shape[0] <= 800:
            for i in range(S.shape[0]):
                p.drawEllipse(QPointF(S[i, 0], S[i, 1]), r, r)
        # grips
        p.setBrush(QColor(c["warn"]))
        for i in np.concatenate([run.left_nodes, run.right_nodes]):
            p.drawEllipse(QPointF(S[int(i), 0], S[int(i), 1]),
                          r * 1.7, r * 1.7)
        # contacts: history dim, fresh bright
        cf = getattr(run, "contact_frames", None)
        if cf is not None and f < len(cf):
            fresh = set((int(a), int(b)) for a, b in
                        np.asarray(cf[f], dtype=int).reshape(-1, 2)) \
                if len(cf[f]) else set()
            if self._contact_cum is not None and f < len(self._contact_cum):
                old = np.asarray(sorted(set(self._contact_cum[f]) - fresh),
                                 dtype=int).reshape(-1, 2)
                if len(old):
                    M = 0.5 * (S[old[:, 0]] + S[old[:, 1]])
                    p.setBrush(QColor(255, 59, 48, 80))
                    for x, y in M:
                        p.drawEllipse(QPointF(x, y), 3.5, 3.5)
            if fresh:
                fr = np.asarray(sorted(fresh), dtype=int).reshape(-1, 2)
                M = 0.5 * (S[fr[:, 0]] + S[fr[:, 1]])
                p.setBrush(QColor("#ff3b30"))
                for x, y in M:
                    p.drawEllipse(QPointF(x, y), 4.5, 4.5)
        p.end()
        return QPixmap.fromImage(img)

    # ---------------- batched edge painting ----------------
    @staticmethod
    def _draw_edge_groups(p, S, edges, groups):
        """One drawLines call per colour bucket.

        Per-edge drawLine + per-edge QPen was the playback bottleneck on
        big networks (37 ms/frame at dpr 1 for voronoi 3x3, ~4x worse on
        HiDPI); bucketing by colour turns it into a handful of batched
        raster calls.
        """
        for pen, idx in groups:
            if not len(idx):
                continue
            p.setPen(pen)
            e = edges[idx].tolist()
            p.drawLines([QLineF(S[a, 0], S[a, 1], S[b, 0], S[b, 1])
                         for a, b in e])

    def _perc_groups(self, run, perc, f, c):
        nE = run.n_edges
        allidx = np.arange(nE)
        if perc is None:
            return [(QPen(QColor(c["edge_inactive"]), 1.2), allidx)]
        blue = QColor(c["accent"])
        cyan = QColor("#7ef0ff")
        in_span = perc.edge_in_spanning[f]
        active = perc.active_mask(f)
        depth = perc.edge_depth_norm[f]
        groups = [(QPen(QColor(c["edge_inactive"]), 1.2), allidx[~active]),
                  (QPen(QColor("#5b7fb9"), 1.8), allidx[active & ~in_span])]
        sp = allidx[active & in_span]
        K = 8
        if len(sp):
            b = np.clip((depth[sp] * K).astype(int), 0, K - 1)
            for k in range(K):
                idx = sp[b == k]
                if len(idx):
                    groups.append((QPen(_lerp_color(blue, cyan,
                                                    (k + 0.5) / K), 2.6),
                                   idx))
        return groups

    def _strain_groups(self, run, f, c):
        sf = run.edge_strain[f]
        scale = float(np.percentile(np.abs(sf), 95)) or 1.0
        cold = QColor("#4dd0e1")
        mid = QColor(c["edge_inactive"])
        hot = QColor("#ff5d47")
        v = np.clip(sf / scale, -1.0, 1.0)
        K = 12
        b = np.clip(((v + 1.0) * K).astype(int), 0, 2 * K - 1)
        groups = []
        for k in range(2 * K):
            idx = np.flatnonzero(b == k)
            if not len(idx):
                continue
            t = (k + 0.5) / K - 1.0
            col = _lerp_color(mid, hot, t) if t >= 0 else \
                _lerp_color(mid, cold, -t)
            groups.append((QPen(col, 1.4 + 1.6 * abs(t)), idx))
        return groups

    def _draw_static(self, p, s, ox, oy, c):
        st = self.static
        S = np.asarray(st["pos"], float) * s + (ox, oy)
        e = np.asarray(st["edges"], int).tolist()
        p.setPen(QPen(QColor(c["accent2"]), max(1.2, 0.035 * s)))
        p.drawLines([QLineF(S[a, 0], S[a, 1], S[b, 0], S[b, 1])
                     for a, b in e])
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(c["sub"]))
        r = max(1.4, 0.045 * s)
        if len(st["pos"]) <= 800:
            for x, y in S:
                p.drawEllipse(QPointF(x, y), r, r)
        if st["left"] is not None:
            p.setBrush(QColor(c["warn"]))
            for i in np.concatenate([st["left"], st["right"]]):
                x, y = S[int(i)]
                p.drawEllipse(QPointF(x, y), r * 1.7, r * 1.7)

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
