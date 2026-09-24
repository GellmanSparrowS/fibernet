'''Parametric structure factory (square-primitive model, P1-consistent).

Two generation paths share one deformation language (the displacement
spectrum of one reference fiber line):

Scale convention (both paths, single rule):
  the spectrum is a list of (dx along the line, dy normal to it) offsets
  expressed as FRACTIONS OF THAT LINE'S OWN LENGTH.  Each fiber line of the
  network receives the same profile scaled by its own length, so one drag in
  the editor means the same relative deformation on every unit and on every
  edge of a unit, regardless of how long that edge happens to be.  This is
  what LineEditor emits and what `resample_spectrum` preserves.

P1 path (unit in SPECTRUM_PRESETS, default 'square'):
  base primitive : square unit cell, n_pts_per_side interior nodes per edge
  spectrum       : per-index offsets of the interior nodes of one reference
                   edge, as fractions of the edge length (every square edge
                   is CELL long, so the scale factor is CELL)
  replication    : 4-fold rotational symmetry onto the four cell edges,
                   then periodic tiling.  Adjacent cells each draw their own
                   copy of a shared boundary, so every interior boundary is
                   a pair of overlapping fibers (a thin lens) - exactly the
                   paper's construction: the twins are identified as two
                   distinct edges (node degrees stay even, <= 8) and are
                   bonded where they overlap, which is what makes them
                   anchors.  They are NOT merged: merging at the midpoint
                   flattened every interior boundary back to a straight line
                   for symmetric spectra, i.e. only the outer frame deformed.

classic path (fibernet units: hexagon, reentrant, chiral, ...):
  base primitive : the canonical undeformed unit (no baked-in randomness)
  replication    : the spectrum is rotated into every base-unit edge's own
                   orientation and scaled by that edge's own length

Named spectrum presets are fixed spectra on the square base; the
undeformed square is the neutral element.  The editor, the preview and the
canvas all consume the same spectrum, so what you edit is what you get.
'''
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import os
import re

import numpy as np

from .fibernet_bridge import ensure_fibernet

ensure_fibernet()
from fibernet.gen.spectrum import (FiberSpectrum, fit_spectrum,
                                   resample_spectrum, rotated_displacements)
from fibernet.gen.custom_cells import CustomCell, CustomCellRegistry

MAX_NODES = 40000      # hard cap: canvas + memory guard
MAX_GRID = 128
MAX_PTS = 24
CELL = 10.0           # unit-cell edge length (box=(10,10))

_PRESET_PTS = 5


def _zero():
    return [(0.0, 0.0)] * _PRESET_PTS


# Values are fractions of the reference line's own length.  They were
# originally authored as absolute units on a CELL=10 square, so they carry a
# 1/10 factor here; the rendered square geometry is unchanged.
SPECTRUM_PRESETS = {
    'square': _zero(),
    'auxetic_bow': [(0.0, -0.034), (0.0, -0.050), (0.0, -0.056),
                    (0.0, -0.050), (0.0, -0.034)],
    'rhombic_bow': [(0.0, 0.034), (0.0, 0.050), (0.0, 0.056),
                    (0.0, 0.050), (0.0, 0.034)],
    'swirl': [(-0.046, -0.008), (-0.028, 0.022), (0.0, 0.038),
              (0.028, 0.022), (0.046, -0.008)],
    'zigzag': [(0.0, 0.048), (0.0, -0.048), (0.0, 0.048),
               (0.0, -0.048), (0.0, 0.048)],
    'pinwheel': [(0.038, 0.022), (0.016, 0.042), (0.0, -0.028),
                 (-0.016, 0.042), (-0.038, 0.022)],
}

CLASSIC_UNITS = ['triangle', 'hexagon', 'reentrant',
                 'chiral', 'star', 'cross', 'diamond']

# truncated-square cut fraction (octagon cell below)
T8 = 1.0 / (2.0 + math.sqrt(2.0))

# Square-periodic corner-graph cells: nodes in unit-box coordinates and
# edges listed as one counter-clockwise walk, so a shared boundary is
# traversed in opposite senses by the two neighbouring cells and survives
# tiling as a twin-fiber lens (paper TOPNet: all-even degree <= 8).
CELL_UNITS = {
    'ring': {
        'nodes': [(0.5+0.36*math.cos(k*math.pi/2), 0.5+0.36*math.sin(k*math.pi/2)) for k in range(4)]
                 + [(1.,.5),(.5,1.),(0.,.5),(.5,0.)],
        'edges': [(k,(k+1)%4) for k in range(4)] + [(0,4),(1,5),(2,6),(3,7)],
    },
    'kagome': {
        'nodes': [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0),
                  (0.5, 0.0), (1.0, 0.5), (0.5, 1.0), (0.0, 0.5)],
        'edges': [(0, 4), (4, 1), (1, 5), (5, 2), (2, 6), (6, 3), (3, 7),
                  (7, 0), (4, 5), (5, 6), (6, 7), (7, 4)],
    },
    'octagon': {
        'nodes': [(T8, 0.0), (1.0 - T8, 0.0), (1.0, T8), (1.0, 1.0 - T8),
                  (1.0 - T8, 1.0), (T8, 1.0), (0.0, 1.0 - T8), (0.0, T8)],
        'edges': [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
                  (7, 0)],
    },
}

import sys
_CUSTOM_ROOT = (os.path.join(os.environ.get('LOCALAPPDATA', os.path.expanduser('~')),
                             'FiberScope') if getattr(sys, 'frozen', False)
                else os.path.join(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__))), 'data'))
CUSTOM_CELL_FILE = os.path.join(_CUSTOM_ROOT, 'custom_units.json')
CUSTOM_CELLS = {}        # key -> {'nodes', 'edges', 'zh', 'en'}
_CELLS_REGISTERED = set()

UNIT_PRESETS = {k: k for k in
                list(SPECTRUM_PRESETS) + CLASSIC_UNITS + list(CELL_UNITS)}

# base-unit list for the UI (square spectrum presets are folded into square)
BASE_UNIT_KEYS = ['square'] + CLASSIC_UNITS + list(CELL_UNITS)

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
    'ring': ('圆环晶格', 'Ring lattice'),
    'voronoi': ('Voronoi（旧版）', 'Voronoi (legacy)'),
    'reentrant': ('内凹蜂窝', 'Reentrant'),
    'chiral': ('手性', 'Chiral'),
    'star': ('星形', 'Star'),
    'cross': ('十字形', 'Cross'),
    'diamond': ('菱形', 'Diamond'),
    'kagome': ('笼目', 'Kagome'),
    'octagon': ('八角方形', 'Truncated square'),
}


def unit_display(key, lang='zh'):
    """Human-readable unit name; falls back to the raw key."""
    names = UNIT_DISPLAY.get(key, (key, key))
    return names[0] if lang == 'zh' else names[1]


# units that appear in older exploration logs but were retired from the UI
# (missing_rib was hexagon-derived: identical base edge lengths 5.00/5.59)
LEGACY_UNIT_MAP = {'voronoi': 'ring', 'honeycomb': 'hexagon',
                   'missing_rib': 'hexagon'}


def resolve_unit(key):
    """Map a possibly retired unit key onto a supported one.

    Returns (resolved_key, note); note is None for a currently listed unit.
    """
    if key in UNIT_PRESETS or key in CUSTOM_CELLS:
        return key, None
    if key in LEGACY_UNIT_MAP:
        return LEGACY_UNIT_MAP[key], 'legacy:%s->%s' % (
            key, LEGACY_UNIT_MAP[key])
    return 'square', 'unknown:%s->square' % key


def unit_key(display, lang='zh'):
    """Resolve a display name back to its internal key."""
    for key, names in UNIT_DISPLAY.items():
        if display == names[0] or display == names[1]:
            return key
    for key, spec in CUSTOM_CELLS.items():
        if display in (spec.get('zh'), spec.get('en')):
            return key
    return display


# ---------------- cell-graph units (built-in + user authored) ----------------
def _cell_factory(spec):
    """pattern_2d unit factory for one corner-graph cell spec."""
    nodes = [(float(a), float(b)) for a, b in spec['nodes']]
    edges = [(int(a), int(b)) for a, b in spec['edges']]

    def factory(box=(10.0, 10.0), n_internal=0, radius=0.1, material=None,
                n_pts_per_side=0, point_displacements=None,
                perturbation=0.0, seed=None):
        from fibernet.core.structure_graph import StructureGraph
        from fibernet.gen.pattern import _add_edge_with_intermediates
        w, h = box
        g = StructureGraph(dimension=2, box_size=[w, h])
        pos = [(nx * w, ny * h) for nx, ny in nodes]
        for p in pos:
            g.add_node(list(p))
        n = int(n_pts_per_side)
        disp = list(point_displacements) if point_displacements else None
        for k, (a, b) in enumerate(edges):
            sl = disp[k * n:(k + 1) * n] if (disp and n) else None
            _add_edge_with_intermediates(g, pos[a], pos[b], n, sl,
                                         radius, material, n_internal)
        g._metadata['unit_type'] = 'cell'
        g._metadata['n_pts_per_side'] = n
        return g

    return factory


def _all_cell_specs():
    specs = dict(CELL_UNITS)
    specs.update(CUSTOM_CELLS)
    return specs


def _ensure_cells():
    """Register built-in + custom cell specs as pattern_2d units."""
    ensure_fibernet()
    from fibernet.gen.pattern import register_unit
    for key, spec in _all_cell_specs().items():
        if key not in _CELLS_REGISTERED:
            register_unit(key, _cell_factory(spec))
            _CELLS_REGISTERED.add(key)


def valid_cell_spec(spec):
    """Validate a connected graph; the unit box is a period, not a clipping boundary."""
    try:
        CustomCell.from_mapping(spec)
    except ValueError:
        return False
    return True


def load_custom_cells(path=None):
    """Read data/custom_units.json; returns the loaded key -> spec dict."""
    global CUSTOM_CELLS
    path = path or CUSTOM_CELL_FILE
    try:
        data = CustomCellRegistry(path).load(strict=False)
    except (OSError, ValueError, MemoryError):
        data = {}
    CUSTOM_CELLS.clear()
    CUSTOM_CELLS.update({k: v.to_mapping() for k, v in data.items()})
    _CELLS_REGISTERED.clear()
    return dict(CUSTOM_CELLS)


def _write_custom_cells():
    CustomCellRegistry(CUSTOM_CELL_FILE).replace_all(CUSTOM_CELLS)


def save_custom_cell(key, nodes, edges, zh=None, en=None, settings=None):
    """Persist one user cell and make it buildable; returns the final key."""
    spec = {'nodes': [[float(a), float(b)] for a, b in nodes],
            'edges': [[int(a), int(b)] for a, b in edges],
            'zh': zh or key, 'en': en or key}
    if settings is not None:
        spec['settings'] = dict(settings)
    if not valid_cell_spec(spec):
        raise ValueError('invalid cell spec')
    key = re.sub(r'[^a-z0-9_]+', '_', str(key).lower()).strip('_') or 'cell'
    while key in CELL_UNITS or key in SPECTRUM_PRESETS \
            or key in CLASSIC_UNITS:
        key += '_x'
    previous = CUSTOM_CELLS.get(key)
    CUSTOM_CELLS[key] = spec
    try:
        _write_custom_cells()
    except (OSError, ValueError, MemoryError):
        if previous is None:
            CUSTOM_CELLS.pop(key, None)
        else:
            CUSTOM_CELLS[key] = previous
        raise
    _CELLS_REGISTERED.discard(key)
    _INTERMEDIATE_CACHE.clear()
    return key


def delete_custom_cell(key):
    if key not in CUSTOM_CELLS:
        return False
    previous = CUSTOM_CELLS.pop(key)
    try:
        _write_custom_cells()
    except (OSError, ValueError, MemoryError):
        CUSTOM_CELLS[key] = previous
        raise
    _CELLS_REGISTERED.discard(key)
    return True


def all_unit_keys():
    """Every buildable base unit: built-ins first, then user cells."""
    return BASE_UNIT_KEYS + sorted(CUSTOM_CELLS)

_INTERMEDIATE_CACHE = {}


def n_intermediate_for(unit, n_pts_per_side):
    '''Number of intermediate nodes pattern_2d expects for a classic unit.'''
    if n_pts_per_side <= 0:
        return 0
    key = (unit, n_pts_per_side)
    if key in _INTERMEDIATE_CACHE:
        return _INTERMEDIATE_CACHE[key]
    ensure_fibernet()
    _ensure_cells()
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


@dataclass
class StructureFactory:
    unit: str = 'square'
    grid_x: int = 3
    grid_y: int = 3
    n_pts_per_side: int = 5
    perturbation: float = 0.0   # relative perturbation of shared displacement parameters
    radius: float = 0.1
    seed: int = 7
    wave: bool = False          # True = fibernet seeded auto displacements
    line_displacements: list = None   # spectrum of reference edge
    node_offsets: list = None         # [[bx, by, dx, dy]] abs units
    expansion_rule: str = 'translate'
    custom_rule: list = None
    topology: str = 'topnet26'
    spectrum_resolved: bool = False

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

    def effective_spectrum(self):
        spectrum = np.asarray(fit_spectrum(self.spectrum(), self.n_pts_per_side), float).reshape(-1, 2)
        if self.spectrum_resolved:
            return spectrum
        rng = np.random.default_rng(self.seed)
        if not self.line_displacements and not np.any(spectrum) and self.perturbation and not self.wave:
            return np.round(rng.uniform(-.25,.25,spectrum.shape)*self.perturbation,3)
        if self.wave and self.n_pts_per_side:
            spectrum = rng.uniform(-.08, .08, spectrum.shape)
        if self.perturbation:
            spectrum *= 1. + rng.uniform(-self.perturbation, self.perturbation, spectrum.shape)
        return spectrum

    def build(self):
        if self.topology == 'topnet26':
            from fibernet.gen.manufacturing import manufacturable_graph
            return manufacturable_graph(self)
        if self.topology != 'legacy':
            raise ValueError('unknown topology version')
        ensure_fibernet()
        _ensure_cells()
        from fibernet.gen.pattern import pattern_2d
        self.clamped()
        spec = _all_cell_specs().get(self.unit)
        if spec and (len(spec['nodes']) + len(spec['edges'])*self.n_pts_per_side) * self.grid_x*self.grid_y > MAX_NODES:
            raise MemoryError('custom cell expansion exceeds node budget; reduce grid or points')
        if self.expansion_rule != 'translate':
            from dataclasses import replace
            from .cell_rules import expand_graph
            base = replace(self, grid_x=1, grid_y=1,
                           expansion_rule='translate', perturbation=0.0,
                           node_offsets=None).build()
            g = expand_graph(base, self.grid_x, self.grid_y,
                             self.expansion_rule, CELL, MAX_NODES, self.custom_rule)
            self._apply_offsets(g)
            self._apply_jitter(g)
            return g
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
        profile scaled by THAT EDGE'S OWN length, exactly like the square P1
        path.  Classic units mix edge lengths freely (chiral: 1.15 and 8.57;
        voronoi: 0.01 to 3.58), so a single CELL-wide scale made short edges
        deform several times their own length while long edges barely moved.
        The flattened list
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
                rx = dx * L * ca - dy * L * sa
                ry = dx * L * sa + dy * L * ca
                out.append((float(rx), float(ry)))
        return out

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
        spec = asdict(self)
        if self.unit in CUSTOM_CELLS:
            spec['custom_cell'] = CUSTOM_CELLS[self.unit]
        raw = json.dumps(spec, sort_keys=True)
        return hashlib.sha1(raw.encode()).hexdigest()[:16]
