"""Export the current structure as JSON (spec + graph) or SVG (vector art)."""
import json

import numpy as np

from .structure import CELL, chains_of

_PALETTE = ["#22d3ee", "#818cf8", "#fbbf24", "#ff5d47",
            "#34d399", "#38bdf8", "#f472b6", "#a3e635"]


def _arrays(factory):
    g = factory.build()
    pos = np.asarray(g.node_positions(), float)[:, :2]
    edges = np.asarray(g.edge_array(), int)[:, :2]
    return pos, edges


def export_json(factory, path):
    pos, edges = _arrays(factory)
    doc = {
        "app": "FiberScope", "version": 1,
        "spec": {k: getattr(factory, k) for k in
                 ("unit", "grid_x", "grid_y", "n_pts_per_side",
                  "perturbation", "seed", "wave",
                  "line_displacements", "node_offsets")},
        "cell": CELL,
        "nodes": [[round(float(x), 4), round(float(y), 4)] for x, y in pos],
        "edges": [[int(a), int(b)] for a, b in edges],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)
    return path


def export_svg(factory, path, width=720, height=720, dark=True):
    pos, edges = _arrays(factory)
    chains, _ = chains_of(pos, edges)
    chain_id = {}
    for ci, mem in enumerate(chains):
        for v in mem:
            chain_id[int(v)] = ci
    lo = pos.min(0); hi = pos.max(0)
    pad = 0.5
    s = min((width - 20) / (hi[0] - lo[0] + 2 * pad),
            (height - 20) / (hi[1] - lo[1] + 2 * pad))

    def X(x):
        return 10 + (x - lo[0] + pad) * s

    def Y(y):
        return 10 + (y - lo[1] + pad) * s

    bg = "#0a0e15" if dark else "#ffffff"
    fg = "#e8edf6" if dark else "#16233a"
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
           f'height="{height}" viewBox="0 0 {width} {height}">',
           f'<rect width="100%" height="100%" fill="{bg}"/>']
    for a, b in edges:
        a, b = int(a), int(b)
        cid = chain_id.get(a, chain_id.get(b, -1))
        col = _PALETTE[cid % len(_PALETTE)] if cid >= 0 else fg
        out.append(f'<line x1="{X(pos[a, 0]):.1f}" y1="{Y(pos[a, 1]):.1f}" '
                   f'x2="{X(pos[b, 0]):.1f}" y2="{Y(pos[b, 1]):.1f}" '
                   f'stroke="{col}" stroke-width="1.6" '
                   f'stroke-linecap="round"/>')
    for x, y in pos:
        out.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="1.8" '
                   f'fill="{fg}"/>')
    out.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))
    return path
