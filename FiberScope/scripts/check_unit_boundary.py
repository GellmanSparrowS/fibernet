"""Unit-cell boundary audit for fslab.structure.

The paper's construction (TOPNet step 1-2) says:
  * one reference edge carries a 10-dim control-point vector; the identical
    pattern is applied to all four cell edges, so editing ONE line deforms
    EVERY fiber line of the whole network;
  * after tiling, a shared boundary is drawn twice (once by each cell); the
    two copies are kept as TWO DISTINCT edges, so every node keeps an even
    degree (<= 8) and the network stays Eulerian / single-path printable.

This script measures, per unit and per spectrum:
  A. scale   - what |displacement| actually reaches pattern_2d, as a % of
               CELL, on the square path vs the classic path
  B. parity  - odd-degree node count and max degree (Eulerian check)
  C. reach   - perpendicular deviation of nodes sitting on each ideal grid
               line, split into BORDER lines (domain outline) and INNER
               lines (shared cell boundaries).  If inner == 0 while
               border != 0, the deformation only survives on the outer
               frame: the structure has degenerated into "just an outer
               frame".
  D. contact - as-printed overlapping non-neighbour pairs are the paper's
               thermal-bonded anchors: the engine keeps them bonded for the
               whole run (they never enter the contact candidate set), so
               the LIVE as-printed preload must be exactly zero

Usage:
    python scripts/check_unit_boundary.py            # full report
    python scripts/check_unit_boundary.py --strict   # exit 1 on violations
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fslab.structure import (CELL, SPECTRUM_PRESETS, StructureFactory,
                             chains_of, rotated_displacements, CELL_UNITS,
                             CLASSIC_UNITS as CLASSIC_BUILTIN)
from fslab.fibernet_bridge import ensure_fibernet

SQUARE_UNITS = list(SPECTRUM_PRESETS)
# cell-graph units ride the same classic path, so audit them identically
CLASSIC_UNITS = list(CLASSIC_BUILTIN) + list(CELL_UNITS)

# reference spectrum used by the consistency / preload probes
REF_SPEC = [[0.0, -0.034], [0.0, -0.050], [0.0, -0.056],
            [0.0, -0.050], [0.0, -0.034]]


def _arrays(g):
    pos = np.asarray(g.node_positions(), float)[:, :2]
    edges = np.asarray(g.edge_array(), int)[:, :2]
    return pos, edges


def parity(g):
    """(n_nodes, n_odd_degree, max_degree)."""
    pos, edges = _arrays(g)
    deg = np.zeros(len(pos), int)
    for a, b in edges:
        deg[int(a)] += 1
        deg[int(b)] += 1
    return len(pos), int((deg % 2 == 1).sum()), int(deg.max()) if len(deg) else 0


def grid_line_reach(g, gx, gy, tol=0.75):
    """Deviation of nodes from the ideal lattice lines, border vs inner.

    Returns dict(inner=[...], border=[...]) of max |perpendicular deviation|
    per lattice line.  A line whose nodes have all migrated out of the
    `tol` window is reported as None (heavily deformed / not recognisable).
    """
    pos, _ = _arrays(g)
    inner, border = [], []
    for axis, n_lines in ((0, gx + 1), (1, gy + 1)):
        other = 1 - axis
        for k in range(n_lines):
            m = np.abs(pos[:, other] - k * CELL) < tol
            if m.sum() == 0:
                (border if k in (0, n_lines - 1) else inner).append(None)
                continue
            dev = float(np.max(np.abs(pos[m, other] - k * CELL)))
            (border if k in (0, n_lines - 1) else inner).append(dev)
    return dict(inner=inner, border=border)


def _fmt(vals):
    got = [v for v in vals if v is not None]
    lost = sum(1 for v in vals if v is None)
    if not got:
        return 'all lost'
    s = 'max %.3f' % max(got)
    if lost:
        s += ' (+%d lost)' % lost
    return s


def contact_preload(g, r_contact=0.8):
    """As-printed overlap pairs: bonded anchors vs live engine preload.

    Pairs already inside r_contact in the as-printed state are anchors
    (overlapping fibers joined by bonding, per the paper); Engine2 excludes
    them from the contact candidate set permanently, so they can never push.
    Returns (n_anchor, n_live, fmax_live, dmin).
    """
    from fslab.engine2 import Engine2, Engine2Config
    eng = Engine2(g, Engine2Config(n_increments=0))
    cand = eng._contact_candidates(eng.pos0)
    if not len(cand):
        return 0, 0, 0.0, float('nan')
    d = np.linalg.norm(eng.pos0[cand[:, 1]] - eng.pos0[cand[:, 0]], axis=1)
    inside = d < r_contact
    codes = cand[:, 0] * eng.n + cand[:, 1]
    live = inside & ~np.isin(codes, eng._rest_codes)
    fmax = float((eng.cfg.k_contact * (r_contact - d[live]).clip(0)).max()) \
        if live.any() else 0.0
    return int(inside.sum() - live.sum()), int(live.sum()), fmax, \
        float(d.min())


def twin_pairs(g):
    """Lens pairs: degree-2 chains that share both end nodes.

    Each shared cell boundary is drawn once per adjacent cell, so the two
    copies form a thin lens; counting them is counting the anchor lines of
    the printed structure.
    """
    pos, edges = _arrays(g)
    chains, deg = chains_of(pos, edges)
    adj = {}
    for a, b in edges:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    cnt = {}
    for ch in chains:
        if len(ch) < 3:
            continue
        ea = [x for x in adj[ch[0]] if deg[x] != 2]
        eb = [x for x in adj[ch[-1]] if deg[x] != 2]
        u = ea[0] if ea else ch[0]
        v = eb[0] if eb else ch[-1]
        key = (min(u, v), max(u, v))
        cnt[key] = cnt.get(key, 0) + 1
    return sum(1 for v in cnt.values() if v == 2)


def scale_of(unit, spec):
    """|displacement| actually handed to pattern_2d, as % of CELL."""
    f = StructureFactory(unit=unit, grid_x=1, grid_y=1, n_pts_per_side=len(spec),
                         perturbation=0.0, seed=7,
                         line_displacements=[list(s) for s in spec]).clamped()
    if f.is_p1():
        d = rotated_displacements(f.spectrum())
    else:
        d = f._classic_point_displacements()
    mx = max((float(np.hypot(a, b)) for a, b in d), default=0.0)
    return 100.0 * mx / CELL


def base_edge_lengths(unit):
    """Undeformed corner-graph edge lengths of one classic base unit."""
    ensure_fibernet()
    from fibernet.gen.pattern import pattern_2d
    g = pattern_2d(unit=unit, grid=(1, 1), n_pts_per_side=0,
                   box=(CELL, CELL), seed=7)
    pos = np.asarray(g.node_positions(), float)[:, :2]
    ed = np.asarray(g.edge_array(), int)[:, :2]
    return np.linalg.norm(pos[ed[:, 1]] - pos[ed[:, 0]], axis=1)


def relative_deformation(unit, spec):
    """|displacement| / (that edge's own length), per base-unit edge.

    This is the quantity that must agree across units for one editor drag to
    mean the same physical deformation everywhere.  Returns (min, max).
    """
    f = StructureFactory(unit=unit, grid_x=1, grid_y=1,
                         n_pts_per_side=len(spec), perturbation=0.0, seed=7,
                         line_displacements=[list(s) for s in spec]).clamped()
    pts = len(spec)
    if f.is_p1():
        # every square edge is CELL long
        d = rotated_displacements(f.spectrum())
        per_edge = [max(np.hypot(a, b) for a, b in d[i * pts:(i + 1) * pts])
                    / CELL for i in range(4)]
    else:
        d = f._classic_point_displacements()
        Ls = base_edge_lengths(unit)
        per_edge = []
        for i, L in enumerate(Ls):
            blk = d[i * pts:(i + 1) * pts]
            if not blk or L < 1e-9:
                continue
            per_edge.append(max(np.hypot(a, b) for a, b in blk) / L)
    if not per_edge:
        return 0.0, 0.0
    return float(min(per_edge)), float(max(per_edge))


def raw_tile(spec, gx, gy, pts):
    """Tiled square cell before weld_lens (what pattern_2d alone returns)."""
    ensure_fibernet()
    from fibernet.gen.pattern import pattern_2d
    return pattern_2d(
        points=[(0., 0.), (CELL, 0.), (CELL, CELL), (0., CELL)], closed=True,
        grid=(gx, gy), n_pts_per_side=pts, radius=0.1, seed=7,
        box=(CELL, CELL), point_displacements=rotated_displacements(spec))


def check_unit(unit, spec, gx=3, gy=3, pts=5, verbose=True):
    """Build one structure and return its audit dict."""
    f = StructureFactory(unit=unit, grid_x=gx, grid_y=gy, n_pts_per_side=pts,
                         perturbation=0.0, seed=7,
                         line_displacements=None if unit in SPECTRUM_PRESETS
                         and spec is SPECTRUM_PRESETS.get(unit)
                         else [list(s) for s in spec]).clamped()
    g = f.build()
    n, odd, dmax = parity(g)
    reach = grid_line_reach(g, gx, gy)
    anchors, nlive, fmax, dmin = contact_preload(g)
    rec = dict(unit=unit, n=n, odd=odd, degmax=dmax, reach=reach,
               anchors=anchors, live=nlive, fmax=fmax, dmin=dmin,
               neutral=(unit == 'square'),
               scale=scale_of(unit, spec))
    if verbose:
        inner_max = max([v for v in reach['inner'] if v is not None],
                        default=0.0)
        border_max = max([v for v in reach['border'] if v is not None],
                         default=0.0)
        flag = ''
        if border_max > 1e-6 and inner_max < 0.05 * border_max:
            flag = '  <-- FRAME ONLY (inner boundaries flattened)'
        print('  %-13s N=%-5d odd=%-3d degmax=%d  scale=%5.1f%%CELL  '
              'inner %-22s border %-22s anchors=%-4d live=%d Fmax=%.0f%s'
              % (unit, n, odd, dmax, rec['scale'], _fmt(reach['inner']),
                 _fmt(reach['border']), anchors, nlive, fmax, flag))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strict', action='store_true',
                    help='exit 1 when an invariant is violated')
    ap.add_argument('--grid', type=int, default=3)
    ap.add_argument('--pts', type=int, default=5)
    a = ap.parse_args()
    gx = gy = a.grid

    spec = SPECTRUM_PRESETS['auxetic_bow']
    print('reference spectrum (auxetic_bow):', spec)
    print()

    print('A. displacement scale reaching pattern_2d (%% of CELL=%.1f)' % CELL)
    print('   square  path (rotated_displacements)        : %.1f%%'
          % scale_of('square', spec))
    for u in ('hexagon', 'triangle', 'chiral'):
        print('   classic path %-8s (_classic_point_displacements): %.1f%%'
              % (u, scale_of(u, spec)))
    print()

    print('B/C/D. square path, preset spectra, grid %dx%d pts %d' % (gx, gy, a.pts))
    recs = []
    for u in SQUARE_UNITS:
        recs.append(check_unit(u, SPECTRUM_PRESETS[u], gx, gy, a.pts))
    print()

    print('   shared boundaries of auxetic_bow (kept as twin fibers)')
    g = raw_tile(spec, gx, gy, a.pts)
    n, odd, dmax = parity(g)
    r = grid_line_reach(g, gx, gy)
    print('     built   N=%-5d odd=%-3d degmax=%d  inner %-20s border %s'
          % (n, odd, dmax, _fmt(r['inner']), _fmt(r['border'])))
    print('     twin (lens) fiber pairs = %d -> bonded anchor lines'
          % twin_pairs(g))
    print()

    print('   classic units + same spectrum')
    for u in CLASSIC_UNITS:
        try:
            recs.append(check_unit(u, spec, gx, gy, a.pts))
        except Exception as e:
            print('  %-13s BUILD FAIL %r' % (u, e))
    print()

    print('E. deformation consistency: |disp| / that edge\'s OWN length')
    print('   one editor drag must mean the same relative bow on every unit')
    print('   %-13s %-18s %s' % ('unit', '|d|/L range', 'verdict'))
    ratios = {}
    for u in ['square'] + CLASSIC_UNITS:
        try:
            lo, hi = relative_deformation(u, REF_SPEC)
        except Exception as e:
            print('   %-13s FAIL %r' % (u, e))
            continue
        ratios[u] = (lo, hi)
        tgt = max(abs(v) for p in REF_SPEC for v in p)
        ok = abs(hi - tgt) < 0.05 * tgt and abs(lo - tgt) < 0.35 * tgt
        print('   %-13s [%.3f, %.3f]      %s (target %.3f)'
              % (u, lo, hi, 'ok' if ok else 'OFF', tgt))
    spread = 0.0
    if ratios:
        allmax = max(h for _, h in ratios.values())
        allmin = min(l for l, _ in ratios.values() if l > 0) if any(
            l > 0 for l, _ in ratios.values()) else 0.0
        spread = allmax / allmin if allmin > 0 else float('inf')
    print('   cross-unit spread (max/min of |d|/L) = %.2fx' % spread)
    print()

    bad_frame = [r['unit'] for r in recs
                 if max([v for v in r['reach']['border'] if v is not None],
                        default=0.0) > 1e-6
                 and max([v for v in r['reach']['inner'] if v is not None],
                         default=0.0) < 0.05 * max(
                             [v for v in r['reach']['border']
                              if v is not None], default=1.0)]
    # the paper's even-degree guarantee covers the TOPNet square-primitive
    # path once a spectrum is applied; the neutral 'square' preset is the
    # undeformed template (twins coincide exactly, graph merges them), and
    # classic units carry fibernet's own topology - both reported separately
    bad_parity = [r['unit'] for r in recs if r['unit'] in SQUARE_UNITS
                  and r['odd'] and not r['neutral']]
    neutral_odd = [r['unit'] for r in recs if r.get('neutral') and r['odd']]
    classic_odd = [r['unit'] for r in recs
                   if r['unit'] not in SQUARE_UNITS and r['odd']]
    # our own cell specs must stay perfectly Eulerian once a spectrum bends
    # them; the neutral template merges coincident twins and is exempt
    cell_odd = [r['unit'] for r in recs
                if r['unit'] in CELL_UNITS and r['odd'] and not r['neutral']]
    bad_live = [r['unit'] for r in recs if r['live'] or r['fmax'] > 0]
    bad_scale = [u for u, (lo, hi) in ratios.items()
                 if abs(hi - max(abs(v) for p in REF_SPEC for v in p))
                 > 0.05 * max(abs(v) for p in REF_SPEC for v in p)]
    print('SUMMARY')
    print('  frame-only (interior boundaries flattened): %s'
          % (bad_frame or 'none'))
    print('  odd-degree nodes on square path (Eulerian): %s'
          % (bad_parity or 'none'))
    print('  odd-degree on neutral template (expected): %s'
          % (neutral_odd or 'none'))
    print('  odd-degree nodes on classic path (fibernet): %s'
          % (classic_odd or 'none'))
    print('  odd-degree nodes on cell-graph units        : %s'
          % (cell_odd or 'none'))
    print('  live as-printed contact preload           : %s'
          % (bad_live or 'none'))
    print('  units whose |d|/L misses the drag target : %s'
          % (bad_scale or 'none'))
    print('  cross-unit deformation spread             : %.2fx' % spread)
    if a.strict and (bad_frame or bad_parity or bad_live or bad_scale
                     or cell_odd or spread > 2.0):
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
