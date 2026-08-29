"""Parametric structure factory wrapping fibernet.gen.pattern_2d.

Tunables live in StructureFactory fields; everything else stays fixed so the
exploration environment has an explicit fixed/explored split.

Deformation model (post-build, unit-cell periodic):
  line_displacements : one reference fiber line's control-point offsets
    (fractions of the cell edge, along/perp to the line). The profile is
    replicated onto EVERY fiber line of the lattice, rotated into each
    line's own orientation -- editing one line edits the whole unit.
  node_offsets       : explicit per-point manual offsets [[bx,by,dx,dy]].
  perturbation       : seeded node jitter (any pts), amplitude fraction.
wave=True falls back to fibernet's seeded auto displacements.
"""
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

# display name -> fibernet unit key
UNIT_PRESETS = {
    "square": "square",
    "triangle": "triangle",
    "hexagon": "hexagon",
    "honeycomb": "honeycomb",
    "kagome": "kagome",
    "reentrant": "reentrant",
    "chiral": "chiral",
    "star": "star",
    "cross": "cross",
    "missing_rib": "missing_rib",
    "diamond": "diamond",
}

_INTERMEDIATE_CACHE = {}


def n_intermediate_for(unit: str, n_pts_per_side: int) -> int:
    """Number of manually adjustable intermediate nodes for (unit, pts).

    Probed via pattern_2d's own length validation (fast, fails before tiling).
    """
    if n_pts_per_side <= 0:
        return 0
    key = (unit, n_pts_per_side)
    if key in _INTERMEDIATE_CACHE:
        return _INTERMEDIATE_CACHE[key]
    ensure_fibernet()
    from fibernet.gen.pattern import pattern_2d
    n = 0
    try:
        pattern_2d(unit=UNIT_PRESETS.get(unit, unit), grid=(1, 1),
                   n_pts_per_side=n_pts_per_side,
                   point_displacements=[(0.0, 0.0)], box=(CELL, CELL), seed=0)
    except ValueError as e:
        m = re.search(r"Expected (\d+) displacements", str(e))
        n = int(m.group(1)) if m else 0
    except Exception:
        n = 0
    _INTERMEDIATE_CACHE[key] = n
    return n


def chains_of(pos: np.ndarray, edges: np.ndarray):
    """Maximal degree-2 walks (fiber lines) as ordered node-id lists."""
    n = pos.shape[0]
    deg = np.zeros(n, int)
    adj = [[] for _ in range(n)]
    for a, b in edges:
        a, b = int(a), int(b)
        deg[a] += 1; deg[b] += 1
        adj[a].append(b); adj[b].append(a)
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
    unit: str = "square"
    grid_x: int = 3
    grid_y: int = 3
    n_pts_per_side: int = 5
    perturbation: float = 0.0   # 0..1 seeded node-jitter fraction
    radius: float = 0.1
    seed: int = 7
    wave: bool = False          # True = seeded auto displacements
    line_displacements: list = None   # [[dx,dy] x pts], fractions of CELL
    node_offsets: list = None         # [[bx, by, dx, dy]] abs units

    def clamped(self):
        self.grid_x = int(max(1, min(MAX_GRID, self.grid_x)))
        self.grid_y = int(max(1, min(MAX_GRID, self.grid_y)))
        self.n_pts_per_side = int(max(0, min(MAX_PTS, self.n_pts_per_side)))
        self.perturbation = float(max(0.0, min(1.0, self.perturbation)))
        return self

    def build(self):
        ensure_fibernet()
        from fibernet.gen.pattern import pattern_2d
        self.clamped()
        kwargs = dict(
            unit=UNIT_PRESETS.get(self.unit, self.unit),
            grid=(self.grid_x, self.grid_y),
            n_pts_per_side=self.n_pts_per_side,
            radius=self.radius,
            seed=self.seed,
            box=(CELL, CELL),
        )
        if self.n_pts_per_side > 0 and not self.wave:
            n = n_intermediate_for(self.unit, self.n_pts_per_side)
            if n > 0:
                kwargs["point_displacements"] = [(0.0, 0.0)] * n
        g = pattern_2d(**kwargs)
        if g.num_nodes > MAX_NODES:
            raise MemoryError(f"structure too large: {g.num_nodes} > {MAX_NODES}")
        self._apply_line_displacements(g)
        self._apply_offsets(g)
        self._apply_jitter(g)
        return g

    # ---------------- post-build deformation ----------------
    def _pos(self, g):
        return np.asarray(g.node_positions(), float)[:, :2]

    def _line_disp_table(self):
        """Prototype build (3x3): per-unit-position displacement table.

        Maximal fiber lines of the prototype are sampled along their own
        chord; longer chains win key conflicts so clipped boundary copies
        never override a complete fiber. Keys are base positions mod CELL,
        which makes the replication exactly periodic on any grid size.
        """
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
                                    [float(p[0] + d[0]), float(p[1] + d[1]), 0.0])

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
