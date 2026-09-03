'''Parametric structure factory (square-primitive model, P1-consistent).

Two generation paths share one deformation language (the displacement
spectrum of one reference fiber line):

P1 path (unit in SPECTRUM_PRESETS, default 'square'):
  base primitive : square unit cell, n_pts_per_side interior nodes per edge
  spectrum       : per-index offsets of the interior nodes of one reference
                   edge, as fractions of the edge length
  replication    : 4-fold rotational symmetry onto the four cell edges,
                   periodic tiling, then lens welding so shared boundaries
                   are single fibers (same math as the P1 dataset code)

classic path (fibernet units: hexagon, honeycomb, kagome, ...):
  base primitive : the canonical undeformed unit (no baked-in randomness)
  replication    : the spectrum is rotated into every fiber line's own
                   orientation via a prototype displacement table

Named spectrum presets are fixed spectra on the square base; the
undeformed square is the neutral element.  The editor, the preview and the
canvas all consume the same spectrum, so what you edit is what you get.
'''
from dataclasses import dataclass, asdict
import hashlib
import json
import re

import numpy as np

from .fibernet_bridge import ensure_fibernet

MAX_NODES = 6000      # hard cap: canvas + memory guard
MAX_GRID = 8
MAX_PTS = 6
CELL = 10.0           # unit-cell edge length (box=(10,10))

_PRESET_PTS = 5


def _zero():
    return [(0.0, 0.0)] * _PRESET_PTS


SPECTRUM_PRESETS = {
    'square': _zero(),
    'auxetic_bow': [(0.0, -0.34), (0.0, -0.50), (0.0, -0.56),
                  (0.0, -0.50), (0.0, -0.34)],
    'rhombic_bow': [(0.0, 0.34), (0.0, 0.50), (0.0, 0.56),
                (0.0, 0.50), (0.0, 0.34)],
    'swirl': [(-0.46, -0.08), (-0.28, 0.22), (0.0, 0.38),
               (0.28, 0.22), (0.46, -0.08)],
    'zigzag': [(0.0, 0.48), (0.0, -0.48), (0.0, 0.48),
             (0.0, -0.48), (0.0, 0.48)],
    'pinwheel': [(0.38, 0.22), (0.16, 0.42), (0.0, -0.28),
             (-0.16, 0.42), (-0.38, 0.22)],
}

CLASSIC_UNITS = ['triangle', 'hexagon', 'voronoi', 'reentrant',
                 'chiral', 'star', 'cross', 'missing_rib', 'diamond']

UNIT_PRESETS = {k: k for k in list(SPECTRUM_PRESETS) + CLASSIC_UNITS}

# base-unit list for the UI (square spectrum presets are folded into square)
BASE_UNIT_KEYS = ['square'] + CLASSIC_UNITS

# internal key -> (zh, en) display names (no underscores in UI)
UNIT_DISPLAY = {
    'square': ('方形', 'Square'),
    'auxetic_bow': ('内凹弓形', 'Auxetic bow'),
    'rhombic_bow': ('外凸弓形', 'Rhombic bow'),
    'swirl': ('旋涡', 'Swirl'),
    'zigzag': ('锯齿', 'Zigzag'),
    'pinwheel': ('风车', 'Pinwheel'),
    'triangle': ('三角形', 'Triangle'),
    'hexagon': ('六边形', 'Hexagon'),
    'voronoi': ('Voronoi', 'Voronoi'),
    'reentrant': ('内凹蜂窝', 'Reentrant'),
    'chiral': ('手性', 'Chiral'),
    'star': ('星形', 'Star'),
    'cross': ('十字形', 'Cross'),
    'missing_rib': ('缺肋', 'Missing rib'),
    'diamond': ('菱形', 'Diamond'),
}


def unit_display(key, lang='zh'):
    """Human-readable unit name; falls back to the raw key."""
    names = UNIT_DISPLAY.get(key, (key, key))
    return names[0] if lang == 'zh' else names[1]


def unit_key(display, lang='zh'):
    """Resolve a display name back to its internal key."""
    for key, names in UNIT_DISPLAY.items():
        if display == names[0] or display == names[1]:
            return key
    return display

_INTERMEDIATE_CACHE = {}


def n_intermediate_for(unit, n_pts_per_side):
    '''Number of intermediate nodes pattern_2d expects for a classic unit.'''
    if n_pts_per_side <= 0:
        return 0
    key = (unit, n_pts_per_side)
    if key in _INTERMEDIATE_CACHE:
        return _INTERMEDIATE_CACHE[key]
    ensure_fibernet()
    from fibernet.gen.pattern import pattern_2d
    # node-count delta: robust for builders that do not validate the
    # displacement list length (they would silently auto-perturb otherwise)
    base = pattern_2d(unit=unit, grid=(1, 1), n_pts_per_side=0,
                      box=(CELL, CELL), seed=0)
    withd = pattern_2d(unit=unit, grid=(1, 1),
                       n_pts_per_side=n_pts_per_side,
                       box=(CELL, CELL), seed=0)
    n = max(0, int(withd.num_nodes) - int(base.num_nodes))
    _INTERMEDIATE_CACHE[key] = n
    return n


def resample_spectrum(spec, pts):
    '''Resample a preset spectrum (any length) onto pts interior nodes.'''
    pts = int(pts)
    if pts <= 0:
        return []
    src = np.asarray(spec, float)
    n = src.shape[0]
    ts = (np.arange(n) + 1.0) / (n + 1.0)
    tt = (np.arange(pts) + 1.0) / (pts + 1.0)
    x = np.interp(tt, ts, src[:, 0])
    y = np.interp(tt, ts, src[:, 1])
    return [[float(a), float(b)] for a, b in zip(x, y)]


def fit_spectrum(spec, pts):
    '''Pad or resample an arbitrary spectrum list to exactly pts entries.'''
    spec = list(spec or [])
    pts = int(pts)
    if len(spec) == pts:
        return [[float(a), float(b)] for a, b in spec]
    if not spec:
        return [[0.0, 0.0] for _ in range(pts)]
    return resample_spectrum(spec, pts)


def rotated_displacements(spectrum):
    '''Replicate one-edge spectrum onto the 4 cell edges with C4 symmetry.

    Order matches the closed square polyline edges AB, BC, CD, DA.
    '''
    ab = [(float(dx), float(dy)) for dx, dy in spectrum]
    bc = [(-dy, dx) for dx, dy in ab]
    cd = [(-dx, -dy) for dx, dy in ab]
    da = [(dy, -dx) for dx, dy in ab]
    return ab + bc + cd + da


def chains_of(pos, edges):
    '''Maximal degree-2 walks (fiber lines) as ordered node-id lists.'''
    n = pos.shape[0]
    deg = np.zeros(n, int)
    adj = [[] for _ in range(n)]
    for a, b in edges:
        a = int(a)
        b = int(b)
        deg[a] += 1
        deg[b] += 1
        adj[a].append(b)
        adj[b].append(a)
    seen = np.zeros(n, bool)
    chains = []
    for v in range(n):
        if deg[v] != 2 or seen[v]:
            continue
        seen[v] = True
        seq = [[], []]
        for si, first in enumerate(adj[v]):
            prev, cur = v, first
            while deg[cur] == 2 and not seen[cur] and cur != v:
                seq[si].append(cur)
                seen[cur] = True
                nxt = [w for w in adj[cur] if w != prev][0]
                prev, cur = cur, nxt
        chains.append(list(reversed(seq[0])) + [v] + seq[1])
    return chains, deg


def weld_lens(g):
    '''Merge twin boundary polylines (lens pairs) into one welded line.

    Tiling the C4-deformed cell makes adjacent cells draw their own copy of
    each shared boundary (a thin lens).  Physically the boundary is a single
    fiber, so each twin pair is welded: inner nodes are merged at their
    midpoint and the duplicate chain is removed; the graph is rebuilt with
    the same node/edge API.
    '''
    pos = np.asarray(g.node_positions(), float)[:, :2]
    edges = np.asarray(g.edge_array(), int)[:, :2]
    chains, deg = chains_of(pos, edges)
    n = pos.shape[0]
    adj = [[] for _ in range(n)]
    for a, b in edges:
        a = int(a)
        b = int(b)
        adj[a].append(b)
        adj[b].append(a)

    def end_key(ch):
        ea = [x for x in adj[ch[0]] if deg[x] != 2]
        eb = [x for x in adj[ch[-1]] if deg[x] != 2]
        u = ea[0] if ea else ch[0]
        v = eb[0] if eb else ch[-1]
        return (min(u, v), max(u, v))

    by_ends = {}
    for ch in chains:
        if len(ch) < 3:
            continue
        by_ends.setdefault(end_key(ch), []).append(ch)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = []
    for key in sorted(by_ends):
        grp = by_ends[key]
        if len(grp) != 2:
            continue
        c1, c2 = grp
        if len(c1) != len(c2):
            continue
        i1, i2 = list(c1), list(c2)
        fwd = float(np.abs(pos[i1] - pos[i2]).sum())
        rev = float(np.abs(pos[i1] - pos[i2[::-1]]).sum())
        if rev < fwd:
            i2 = i2[::-1]
        for a, b in zip(i1, i2):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
                pairs.append((a, b))
    if not pairs:
        return g
    newpos = pos.copy()
    for a, b in pairs:
        newpos[a] = 0.5 * (pos[a] + pos[b])
    g2 = type(g)(dimension=2, tolerance=1e-9)
    ids = {}
    for i in range(n):
        r = find(i)
        if r not in ids:
            ids[r] = g2.add_node(np.array([newpos[r][0], newpos[r][1], 0.0]))
    seen = set()
    for a, b in edges:
        ra, rb = find(int(a)), find(int(b))
        if ra == rb:
            continue
        ia, ib = ids[ra], ids[rb]
        key = (min(ia, ib), max(ia, ib))
        if key in seen:
            continue
        seen.add(key)
        g2.add_edge(ia, ib, radius=0.1)
    return g2


@dataclass
class StructureFactory:
    unit: str = 'square'
    grid_x: int = 3
    grid_y: int = 3
    n_pts_per_side: int = 5
    perturbation: float = 0.0   # 0..1 seeded node-jitter fraction
    radius: float = 0.1
    seed: int = 7
    wave: bool = False          # True = fibernet seeded auto displacements
    line_displacements: list = None   # spectrum of reference edge
    node_offsets: list = None         # [[bx, by, dx, dy]] abs units

    def clamped(self):
        self.grid_x = int(max(1, min(MAX_GRID, self.grid_x)))
        self.grid_y = int(max(1, min(MAX_GRID, self.grid_y)))
        self.n_pts_per_side = int(max(0, min(MAX_PTS, self.n_pts_per_side)))
        self.perturbation = float(max(0.0, min(1.0, self.perturbation)))
        return self

    def is_p1(self):
        return self.unit in SPECTRUM_PRESETS

    def spectrum(self):
        '''Effective spectrum (pts entries) for the reference edge.'''
        pts = self.n_pts_per_side
        if self.line_displacements:
            return fit_spectrum(self.line_displacements, pts)
        if self.is_p1():
            return resample_spectrum(SPECTRUM_PRESETS[self.unit], pts)
        return None

    def build(self):
        ensure_fibernet()
        from fibernet.gen.pattern import pattern_2d
        self.clamped()
        pts = self.n_pts_per_side
        if self.is_p1():
            kwargs = dict(
                points=[(0.0, 0.0), (CELL, 0.0), (CELL, CELL), (0.0, CELL)],
                closed=True,
                grid=(self.grid_x, self.grid_y),
                n_pts_per_side=pts,
                radius=self.radius,
                seed=self.seed,
                box=(CELL, CELL),
            )
            if pts > 0:
                if self.wave:
                    kwargs['point_displacements'] = None
                else:
                    kwargs['point_displacements'] = rotated_displacements(
                        self.spectrum())
            g = pattern_2d(**kwargs)
            g = weld_lens(g)
        else:
            kwargs = dict(
                unit=self.unit,
                grid=(self.grid_x, self.grid_y),
                n_pts_per_side=pts,
                radius=self.radius,
                seed=self.seed,
                box=(CELL, CELL),
            )
            if pts > 0 and not self.wave:
                if self.line_displacements:
                    kwargs['point_displacements'] = \
                        self._classic_point_displacements()
                else:
                    n = n_intermediate_for(self.unit, pts)
                    if n > 0:
                        kwargs['point_displacements'] = [(0.0, 0.0)] * n
            g = pattern_2d(**kwargs)
        if g.num_nodes > MAX_NODES:
            raise MemoryError('structure too large: %d > %d'
                              % (g.num_nodes, MAX_NODES))
        self._apply_offsets(g)
        self._apply_jitter(g)
        return g

    # ---------------- classic-path spectrum replication ----------------
    def _classic_point_displacements(self):
        """Rotate the reference-line spectrum into every base-unit edge.

        Each classic unit is described by its zero-point corner graph;
        every original edge receives the same (dx along edge, dy normal)
        profile, exactly like the square P1 path.  The flattened list
        follows base.edge_array() order, which matches the unit factory
        edge-insertion order used when n_pts_per_side > 0.
        """
        ensure_fibernet()
        from fibernet.gen.pattern import pattern_2d
        spec = fit_spectrum(self.line_displacements, self.n_pts_per_side)
        base = pattern_2d(unit=self.unit, grid=(1, 1), n_pts_per_side=0,
                          box=(CELL, CELL), seed=self.seed)
        pos = np.asarray(base.node_positions(), float)[:, :2]
        edges = np.asarray(base.edge_array(), int)[:, :2]
        out = []
        for a, b in edges:
            d = pos[b] - pos[a]
            L = max(float(np.hypot(d[0], d[1])), 1e-9)
            ca, sa = d[0] / L, d[1] / L
            for dx, dy in spec:
                rx = dx * CELL * ca - dy * CELL * sa
                ry = dx * CELL * sa + dy * CELL * ca
                out.append((float(rx), float(ry)))
        return out

    def _line_disp_table(self):
        '''Prototype (3x3) displacement table keyed by base pos mod CELL.

        Maximal fiber lines of the prototype are sampled along their own
        chord; the profile is rotated into each line's orientation.  Keys
        are base positions mod CELL, so replication is exactly periodic on
        any grid size.
        '''
        ld = self.line_displacements
        pts = len(ld)
        ts = np.concatenate(([0.0], [(k + 1) / (pts + 1) for k in range(pts)],
                             [1.0]))
        prof = np.vstack(([0.0, 0.0], np.asarray(ld, float) * CELL,
                          [0.0, 0.0]))
        proto = StructureFactory(
            unit=self.unit, grid_x=3, grid_y=3,
            n_pts_per_side=self.n_pts_per_side, radius=self.radius,
            seed=self.seed, wave=self.wave, perturbation=0.0).clamped()
        g3 = proto.build()
        pos = np.asarray(g3.node_positions(), float)[:, :2]
        edges = np.asarray(g3.edge_array(), int)[:, :2]
        chains, _ = chains_of(pos, edges)
        chains.sort(key=len, reverse=True)
        table = {}
        for mem in chains:
            m = len(mem)
            if m == 0:
                continue
            d = pos[mem[-1]] - pos[mem[0]]
            L = float(np.hypot(d[0], d[1]))
            if L < 1e-9:
                continue
            ca, sa = d[0] / L, d[1] / L
            tt = np.array([(j + 1) / (m + 1) for j in range(m)])
            ax = np.interp(tt, ts, prof[:, 0])
            ay = np.interp(tt, ts, prof[:, 1])
            rx = ax * ca - ay * sa
            ry = ax * sa + ay * ca
            for nid, dx, dy in zip(mem, rx, ry):
                q = pos[nid]
                key = (round(float(q[0]) % CELL, 2),
                       round(float(q[1]) % CELL, 2))
                table.setdefault(key, (float(dx), float(dy)))
        return table

    def _apply_line_displacements(self, g):
        if not self.line_displacements or self.n_pts_per_side <= 0:
            return
        table = self._line_disp_table()
        pos = self._pos(g)
        for nid, p in enumerate(pos):
            key = (round(float(p[0]) % CELL, 2),
                   round(float(p[1]) % CELL, 2))
            d = table.get(key)
            if d is not None:
                g.set_node_position(int(nid),
                                    [float(p[0] + d[0]), float(p[1] + d[1]),
                                     0.0])

    # ---------------- post-build adjustments ----------------
    def _pos(self, g):
        return np.asarray(g.node_positions(), float)[:, :2]

    def _apply_offsets(self, g):
        if not self.node_offsets:
            return
        pos = self._pos(g)
        table = {}
        for bx, by, dx, dy in self.node_offsets:
            key = (round(float(bx) % CELL, 3), round(float(by) % CELL, 3))
            table[key] = (float(dx), float(dy))
        for nid, p in enumerate(pos):
            key = (round(float(p[0]) % CELL, 3), round(float(p[1]) % CELL, 3))
            d = table.get(key)
            if d is not None:
                g.set_node_position(int(nid),
                                    [float(p[0] + d[0]), float(p[1] + d[1]),
                                     0.0])

    def _apply_jitter(self, g):
        if self.perturbation <= 0:
            return
        pos = self._pos(g)
        rng = np.random.default_rng(self.seed)
        amp = self.perturbation * 0.35 * CELL
        j = rng.uniform(-amp, amp, size=pos.shape)
        for nid, (x, y) in enumerate(pos + j):
            g.set_node_position(int(nid), [float(x), float(y), 0.0])

    def key(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha1(raw.encode()).hexdigest()[:16]
