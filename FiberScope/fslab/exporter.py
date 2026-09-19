"""Export helpers: structure as JSON/SVG, results as CSV.

Every writer here is Qt-free and returns the path it wrote, so the GUI,
the AI tool surface and the tests all share one implementation.  CSV files
are written as utf-8-sig so Excel opens the Chinese feature names correctly.
"""
import csv
import json

import numpy as np

from .structure import CELL, chains_of

_PALETTE = ["#22d3ee", "#818cf8", "#fbbf24", "#ff5d47",
            "#34d399", "#38bdf8", "#f472b6", "#a3e635"]


def _n(x, sig=10):
    """Compact but round-trippable float for CSV cells."""
    return float(f"{float(x):.{sig}g}")


def _write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


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
                  "line_displacements", "node_offsets", "radius", "expansion_rule", "custom_rule", "topology")},
        "cell": CELL,
        "nodes": [[round(float(x), 4), round(float(y), 4)] for x, y in pos],
        "edges": [[int(a), int(b)] for a, b in edges],
    }
    from .structure import CUSTOM_CELLS
    if factory.unit in CUSTOM_CELLS:
        doc['custom_cell'] = CUSTOM_CELLS[factory.unit]
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


# ---------------------------------------------------------------- results
def export_run_csv(run, path, perc=None):
    """Per-frame simulation log: strain, grip force, the five energy
    channels and the contact count.  Pass a PercolationResult to append the
    spanning/backbone fractions and the per-frame strain threshold."""
    e = run.energies
    header = ["frame", "strain", "force", "e_axial", "e_bend", "e_contact",
              "e_kinetic", "e_work", "contacts"]
    if perc is not None:
        header += ["spanning_frac", "backbone_frac", "strain_threshold"]
    rows = []
    for f in range(run.n_frames):
        r = [f, _n(run.strain_levels[f]), _n(run.force_curve[f]),
             _n(e["axial"][f]), _n(e["bend"][f]), _n(e["contact"][f]),
             _n(e["kinetic"][f]), _n(e["work"][f]),
             int(run.contact_counts[f])]
        if perc is not None:
            r += [_n(perc.spanning_frac[f]), _n(perc.backbone_frac[f]),
                  _n(perc.threshold_curve[f])]
        rows.append(r)
    return _write_csv(path, header, rows)


def export_features_csv(feats, path, batch=None):
    """Scalar feature table, one row per feature, in FEATURE_GROUPS order.
    `batch` is a list of feature dicts (batch mode) -> mean/std/min/max."""
    from .features import FEATURE_GROUPS, FEATURE_UNIT, FEATURE_ZH, HIST_KEYS
    stats = {}
    if batch:
        for key in feats:
            if key in HIST_KEYS:
                continue
            vals = [float(b[key]) for b in batch
                    if key in b and np.ndim(b[key]) == 0]
            if len(vals) >= 2:
                a = np.asarray(vals, float)
                stats[key] = (len(a), a.mean(), a.std(), a.min(), a.max())
    header = ["key", "name", "group", "unit", "value"]
    if stats:
        header += ["batch_n", "batch_mean", "batch_std", "batch_min",
                   "batch_max"]
    rows = []
    for group, keys in FEATURE_GROUPS.items():
        for key in keys:
            if key not in feats or key in HIST_KEYS:
                continue
            v = feats[key]
            if np.ndim(v) != 0:
                continue
            r = [key, FEATURE_ZH.get(key, key), group,
                 FEATURE_UNIT.get(key, ""), _n(v)]
            if stats:
                s = stats.get(key)
                r += ([s[0]] + [_n(x) for x in s[1:]] if s else [""] * 5)
            rows.append(r)
    return _write_csv(path, header, rows)


def export_hist_csv(feats, path):
    """Long-format raw samples behind the histogram cards (key,index,value),
    so a reader can rebin without re-running the analysis."""
    from .features import HIST_KEYS
    rows = []
    for key in HIST_KEYS:
        if key not in feats:
            continue
        arr = np.asarray(feats[key], float).ravel()
        rows += [[key, i, _n(v)] for i, v in enumerate(arr)]
    return _write_csv(path, ["key", "index", "value"], rows)


def export_inverse_csv(records, path):
    """Optimizer log, one row per evaluation; the CEM vector is spread over
    p0..pN so a spreadsheet can chart convergence directly."""
    npar = max((len(r.params) for r in records), default=0)
    header = (["eval_id", "stage", "label", "dist", "best_dist"] +
              [f"p{i}" for i in range(npar)])
    rows = []
    for r in records:
        p = [_n(x) for x in r.params]
        rows.append([r.eval_id, r.stage, r.label, _n(r.dist), _n(r.best_dist)]
                    + p + [""] * (npar - len(p)))
    return _write_csv(path, header, rows)
