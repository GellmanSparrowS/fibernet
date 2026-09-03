"""Drape a fiber network onto a quad OBJ surface (pure numpy).

Math mirrors the reference implementation (batch_process.py): for every
mesh edge and each adjacent face, len(spectrum) interior points are
inserted at

    t   = k / (n + 1)
    pos = p_u + t * Lvec + dx * L * e1 + dy * L * e2

where e1 is the edge tangent, e2 = cross(n_face, e1) for dy < 0
(in-face direction) and e2 = cross(n_avg, e1) for dy > 0 (outward,
n_avg = mean normal of the faces sharing the edge).

Optional in-face patterns (unit='triangle' / 'hexagon' / 'reentrant')
add per-face fibers whose dy is applied along the face normal.
"""
import numpy as np


def parse_obj(path):
    """Parse an OBJ file keeping quad faces only. Returns (V, F)."""
    verts, faces = [], []
    with open(path, encoding='utf-8', errors='ignore') as fh:
        for ln in fh:
            if ln.startswith('v '):
                parts = ln.split()
                verts.append((float(parts[1]), float(parts[2]),
                              float(parts[3])))
            elif ln.startswith('f '):
                idx = [int(tok.split('/')[0]) - 1 for tok in ln.split()[1:]]
                if len(idx) == 4:
                    faces.append(idx)
    return np.asarray(verts, dtype=float), faces


def build_edge_faces(F):
    """Map undirected edge (min, max) -> list of adjacent face ids."""
    e2f = {}
    for fid, f in enumerate(F):
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[3]), (f[3], f[0])):
            key = (a, b) if a < b else (b, a)
            e2f.setdefault(key, []).append(fid)
    return e2f


def _unit_vecs(A):
    ln = np.linalg.norm(A, axis=-1, keepdims=True)
    ln = np.where(ln < 1e-12, 1.0, ln)
    return A / ln


def _face_normals(V, F4):
    pts = V[F4]
    return _unit_vecs(np.cross(pts[:, 1] - pts[:, 0],
                               pts[:, 2] - pts[:, 0]))


def _chain_segs(start_ids, inner_ids, end_ids):
    """Segment index pairs along polylines start -> inner row -> end."""
    n = inner_ids.shape[1]
    parts = [np.stack([start_ids, inner_ids[:, 0]], axis=1)]
    if n > 1:
        parts.append(np.stack([inner_ids[:, :-1].reshape(-1),
                               inner_ids[:, 1:].reshape(-1)], axis=1))
    parts.append(np.stack([inner_ids[:, -1], end_ids], axis=1))
    return parts


def _line_inner(p0, p1, nrm, S):
    """Interior points of lines p0->p1 displaced along +-nrm by dy."""
    n = len(S)
    Lvec = p1 - p0
    L = np.linalg.norm(Lvec, axis=1)
    e1 = _unit_vecs(Lvec)
    t = np.arange(1, n + 1) / (n + 1)
    dx, dy = S[:, 0], S[:, 1]
    sgn = np.where(dy[None, :, None] > 0, 1.0, -1.0)
    pts = (p0[:, None, :]
           + t[None, :, None] * Lvec[:, None, :]
           + dx[None, :, None] * L[:, None, None] * e1[:, None, :]
           + dy[None, :, None] * L[:, None, None] * sgn * nrm[:, None, :])
    return pts.reshape(-1, 3)


def deform_surface(V, F, spectrum, unit='square'):
    """Drape fibers over the quad mesh.

    Returns (P, segs): P (m,3) all points (mesh vertices first),
    segs (k,2) integer index pairs of adjacent polyline points.
    """
    V = np.asarray(V, dtype=float)
    S = np.asarray(spectrum, dtype=float).reshape(-1, 2)
    n = len(S)
    F4 = np.asarray(F, dtype=int)
    nF = len(F4)
    Fn = _face_normals(V, F4)

    P_list, seg_list = [V], []

    # ---- fibers along mesh edges, one polyline per (edge, face) ----
    nxt = np.roll(F4, -1, axis=1)
    U = F4.reshape(-1)                    # directed edges in face winding
    W = nxt.reshape(-1)
    FID = np.repeat(np.arange(nF), 4)
    R = len(U)
    if n > 0 and R > 0:
        e2f = build_edge_faces(F4.tolist())
        nsum = {key: Fn[fl].sum(axis=0) for key, fl in e2f.items()}
        lo = np.minimum(U, W)
        hi = np.maximum(U, W)
        n_avg = np.array([nsum[(lo[r], hi[r])] for r in range(R)])
        n_avg = _unit_vecs(n_avg)

        Lvec = V[W] - V[U]
        L = np.linalg.norm(Lvec, axis=1)
        e1 = _unit_vecs(Lvec)
        n_face = Fn[FID]
        e2_in = _unit_vecs(np.cross(n_face, e1))
        e2_out = _unit_vecs(np.cross(n_avg, e1))

        t = np.arange(1, n + 1) / (n + 1)
        dx, dy = S[:, 0], S[:, 1]
        e2 = np.where(dy[None, :, None] > 0,
                      e2_out[:, None, :], e2_in[:, None, :])
        pts = (V[U][:, None, :]
               + t[None, :, None] * Lvec[:, None, :]
               + dx[None, :, None] * L[:, None, None] * e1[:, None, :]
               + dy[None, :, None] * L[:, None, None] * e2)
        base = len(V)
        P_list.append(pts.reshape(-1, 3))
        inner = base + np.arange(R)[:, None] * n + np.arange(n)[None, :]
        seg_list.extend(_chain_segs(U, inner, W))

    # ---- in-face pattern fibers (dy applied along the face normal) ----
    if unit == 'triangle' and n > 0:
        p0, p1 = V[F4[:, 0]], V[F4[:, 2]]
        pts = _line_inner(p0, p1, Fn, S)
        base = sum(p.shape[0] for p in P_list)
        P_list.append(pts)
        inner = base + np.arange(nF)[:, None] * n + np.arange(n)[None, :]
        seg_list.extend(_chain_segs(F4[:, 0], inner, F4[:, 2]))
    elif unit in ('hexagon', 'reentrant') and n > 0:
        Sr = -S if unit == 'reentrant' else S
        m = 0.5 * (V[F4] + V[nxt])                 # edge midpoints per face
        base_m = sum(p.shape[0] for p in P_list)
        P_list.append(m.reshape(-1, 3))
        mid_ids = base_m + np.arange(nF)[:, None] * 4 + np.arange(4)[None, :]
        p0 = np.concatenate([m[:, j] for j in range(4)])
        p1 = np.concatenate([m[:, (j + 1) % 4] for j in range(4)])
        nrm = np.tile(Fn, (4, 1))
        pts = _line_inner(p0, p1, nrm, Sr)
        base_i = base_m + nF * 4
        P_list.append(pts)
        M4 = 4 * nF
        inner = base_i + np.arange(M4)[:, None] * n + np.arange(n)[None, :]
        start = np.concatenate([mid_ids[:, j] for j in range(4)])
        end = np.concatenate([mid_ids[:, (j + 1) % 4] for j in range(4)])
        seg_list.extend(_chain_segs(start, inner, end))

    P = np.concatenate(P_list, axis=0)
    segs = (np.concatenate(seg_list, axis=0).astype(np.int64)
            if seg_list else np.zeros((0, 2), dtype=np.int64))
    return P, segs

