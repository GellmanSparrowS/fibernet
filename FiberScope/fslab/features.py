'''Structure feature extraction for fiber networks (pure numpy).

compute_features(pos, edges, rect=None) -> dict of scalar metrics plus
histogram arrays, organised in three groups (structure / pore /
contact).  Keys listed in HIST_KEYS hold per-element arrays consumed by
the dashboard cards; every other key is a scalar float/int.  When
rect=(x0, y0, x1, y1) is given, only edges whose midpoint lies inside
the rectangle are analysed.

Pores are the bounded faces of the planarized network: segments are
split at pairwise crossings and at nodes lying on other segments, then
faces are enumerated through the angular rotation system (cycle_basis
returns fundamental cycles, which are NOT faces once edges are
subdivided).  Contact detection is a pairwise segment test with a bbox
prefilter (O(E^2); fine for E < 800).
'''
import numpy as np

ORIENT_BINS = 12

# keys whose value is a per-element array (histogram cards)
HIST_KEYS = ('edge_lengths', 'orientations', 'degree_hist', 'node_spacing',
             'segment_straightness', 'pore_areas', 'pore_aspect',
             'pore_neighbor', 'overlap_len', 'cross_angle')

FEATURE_GROUPS = {
    'structure': ['n_node', 'n_edge', 'total_length', 'mean_edge_len',
                  'len_cv', 'edge_len_q90', 'degree_mean', 'degree_max',
                  'junction_count', 'degree_entropy', 'orient_entropy',
                  'orient_bias', 'anisotropy', 'radius_gyration',
                  'boundary_ratio', 'node_density', 'edge_density',
                  'nn_dist_mean', 'nn_dist_cv', 'straightness_mean',
                  'edge_lengths', 'orientations', 'degree_hist',
                  'node_spacing', 'segment_straightness'],
    'pore': ['pore_count', 'pore_area_mean', 'pore_area_cv',
             'largest_pore_ratio', 'porosity', 'pore_aspect_mean',
             'pore_perim_mean', 'pore_neighbor_mean',
             'pore_areas', 'pore_aspect', 'pore_neighbor'],
    'contact': ['cross_count', 'cross_per_edge', 'cross_pos_ratio',
                'cross_angle_mean', 'overlap_len_mean',
                'contact_cluster_max', 'contact_cluster_mean',
                'overlap_len', 'cross_angle'],
}

FEATURE_ZH = {
    'n_node': '节点数',
    'n_edge': '边数',
    'total_length': '总边长',
    'mean_edge_len': '平均边长',
    'len_cv': '边长变异系数',
    'edge_len_q90': '边长90分位',
    'degree_mean': '平均度',
    'degree_max': '最大度',
    'junction_count': '三叉结点数',
    'degree_entropy': '度熵',
    'orient_entropy': '取向熵',
    'orient_bias': '取向集中度',
    'anisotropy': '各向异性',
    'radius_gyration': '回转半径',
    'boundary_ratio': '边界节点占比',
    'node_density': '节点密度',
    'edge_density': '纤维线密度',
    'nn_dist_mean': '最近邻距离均值',
    'nn_dist_cv': '最近邻距离变异',
    'straightness_mean': '链直线度均值',
    'edge_lengths': '边长分布',
    'orientations': '取向分布',
    'degree_hist': '度分布',
    'node_spacing': '最近邻距离分布',
    'segment_straightness': '链直线度分布',
    'pore_count': '孔隙数',
    'pore_area_mean': '平均孔隙面积',
    'pore_area_cv': '孔隙面积变异系数',
    'largest_pore_ratio': '最大孔隙占比',
    'porosity': '孔隙率',
    'pore_aspect_mean': '孔隙长宽比均值',
    'pore_perim_mean': '孔隙周长均值',
    'pore_neighbor_mean': '孔隙邻域数均值',
    'pore_areas': '孔隙面积分布',
    'pore_aspect': '孔隙长宽比分布',
    'pore_neighbor': '孔隙邻域数分布',
    'cross_count': '交叉点数',
    'cross_per_edge': '每边交叉数',
    'cross_pos_ratio': '有交叉边占比',
    'cross_angle_mean': '平均交叉角',
    'overlap_len_mean': '平均重叠长度',
    'contact_cluster_max': '最大接触簇',
    'contact_cluster_mean': '平均接触簇',
    'overlap_len': '重叠长度分布',
    'cross_angle': '交叉角分布',
}

# display units for scalar cards ('°' values are stored in degrees)
FEATURE_UNIT = {
    'total_length': 'u', 'mean_edge_len': 'u', 'edge_len_q90': 'u',
    'radius_gyration': 'u', 'nn_dist_mean': 'u', 'pore_area_mean': 'u²',
    'pore_perim_mean': 'u', 'overlap_len_mean': 'u',
    'cross_angle_mean': '°', 'node_density': 'u⁻²', 'edge_density': 'u⁻¹',
}

# fixed histogram ranges; other keys use data-driven bins
FEATURE_RANGE = {
    'orientations': (0.0, np.pi),
    'cross_angle': (0.0, 90.0),
    'segment_straightness': (0.0, 1.0),
}

# integer-valued histograms (one bin per integer)
FEATURE_INT = frozenset(('degree_hist', 'pore_neighbor'))


def scalar_keys(groups=None):
    '''Ordered scalar feature keys of the given (or all) groups.'''
    keys = []
    for g in (groups or FEATURE_GROUPS):
        keys.extend(k for k in FEATURE_GROUPS[g] if k not in HIST_KEYS)
    return keys


def _filter_region(pos, edges, rect):
    '''Keep only edges whose midpoint lies inside rect; reindex nodes.'''
    pos = np.asarray(pos, float)
    edges = np.asarray(edges, int)
    if rect is None or edges.size == 0:
        return pos, edges
    x0, y0, x1, y1 = (float(v) for v in rect)
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    mid = 0.5 * (pos[edges[:, 0]] + pos[edges[:, 1]])
    keep = ((mid[:, 0] >= x0) & (mid[:, 0] <= x1)
            & (mid[:, 1] >= y0) & (mid[:, 1] <= y1))
    edges = edges[keep]
    if edges.size == 0:
        return pos[:0], edges
    used = np.unique(edges)
    remap = np.full(pos.shape[0], -1, int)
    remap[used] = np.arange(used.size)
    return pos[used], remap[edges]


def _region_area(pos, rect):
    '''Analysis region area (rect if given, else node bbox).'''
    if rect is not None:
        x0, y0, x1, y1 = (float(v) for v in rect)
        return max(abs(x1 - x0) * abs(y1 - y0), 1e-12)
    if pos.shape[0] < 2:
        return 1e-12
    lo = pos.min(0)
    hi = pos.max(0)
    return max((hi[0] - lo[0]) * (hi[1] - lo[1]), 1e-12)


def _nn_distances(pos, budget=2000000):
    '''Nearest-neighbour distance of every node (chunked, O(N^2)).'''
    n = pos.shape[0]
    if n < 2:
        return np.zeros(0)
    out = np.empty(n)
    step = max(1, budget // n)
    for i in range(0, n, step):
        j = min(i + step, n)
        d = np.hypot(pos[i:j, None, 0] - pos[None, :, 0],
                     pos[i:j, None, 1] - pos[None, :, 1])
        d[np.arange(j - i), np.arange(i, j)] = np.inf
        out[i:j] = d.min(axis=1)
    return out


def _chain_straightness(pos, edges):
    '''Chord/arc-length of every maximal degree-2 walk (fiber line).'''
    if edges.shape[0] == 0:
        return np.zeros(0)
    n = pos.shape[0]
    deg = np.bincount(edges.ravel(), minlength=n)
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    seen = np.zeros(n, bool)
    out = []
    for v in range(n):
        if deg[v] != 2 or seen[v]:
            continue
        seen[v] = True
        arms = []
        for u in adj[v]:
            arm = []
            prev, cur = v, u
            while deg[cur] == 2 and not seen[cur]:
                seen[cur] = True
                arm.append(cur)
                a, b = adj[cur]
                prev, cur = cur, (b if a == prev else a)
            arms.append(arm)
        chain = arms[0][::-1] + [v] + arms[1]
        p = pos[chain]
        arclen = float(np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1])).sum())
        chord = float(np.hypot(p[-1, 0] - p[0, 0], p[-1, 1] - p[0, 1]))
        if arclen > 1e-12:
            out.append(min(1.0, chord / arclen))
    return np.asarray(out, float)


def _structure_features(pos, edges):
    n_node = int(pos.shape[0])
    n_edge = int(edges.shape[0])
    if n_edge == 0:
        return dict(n_node=n_node, n_edge=0, total_length=0.0,
                    mean_edge_len=0.0, len_cv=0.0, degree_entropy=0.0,
                    orient_entropy=0.0, anisotropy=0.0,
                    radius_gyration=0.0), np.zeros(0), np.zeros(0), \
            np.zeros(n_node, int)
    a = pos[edges[:, 0]]
    b = pos[edges[:, 1]]
    d = b - a
    lens = np.hypot(d[:, 0], d[:, 1])
    total = float(lens.sum())
    mean = float(lens.mean())
    len_cv = float(lens.std() / mean) if mean > 0 else 0.0
    # degree-distribution entropy (matches reference Features.py: raw
    # probability of each degree, entropy base 2, no normalization)
    deg = np.bincount(edges.ravel(), minlength=n_node)
    deg_dist = np.bincount(deg).astype(float)
    p_deg = deg_dist / max(float(deg_dist.sum()), 1.0)
    p_deg = p_deg[p_deg > 0]
    degree_entropy = float(-(p_deg * np.log2(p_deg)).sum())
    # orientation entropy: signed angles in [-pi, pi), 18 bins, base 2
    ang_full = np.arctan2(d[:, 1], d[:, 0])
    hist, _ = np.histogram(ang_full, bins=18, range=(-np.pi, np.pi))
    hist = hist[hist > 0].astype(float)
    hist = hist / max(float(hist.sum()), 1.0)
    orient_entropy = float(-(hist * np.log2(hist)).sum())
    # orientation tensor (unweighted, reference formula)
    cs = np.column_stack([np.cos(ang_full), np.sin(ang_full)])
    q = (cs.T @ cs) / max(float(ang_full.size), 1.0)
    w = np.linalg.eigvalsh(q)
    anisotropy = float((w[-1] - w[0]) / max(w[-1] + w[0], 1e-12))
    # histograms keep undirected angles in [0, pi)
    ang = np.mod(ang_full, np.pi)
    centroid = pos.mean(axis=0)
    radius_gyration = float(np.sqrt(((pos - centroid) ** 2).sum(1).mean()))
    feats = dict(n_node=n_node, n_edge=n_edge, total_length=total,
                 mean_edge_len=mean, len_cv=len_cv,
                 degree_entropy=float(degree_entropy),
                 orient_entropy=float(orient_entropy),
                 anisotropy=anisotropy, radius_gyration=radius_gyration)
    return feats, lens, ang, deg


def _structure_extra(pos, edges, lens, ang, deg, rect):
    '''Extended structure scalars + histogram arrays.'''
    n_node = pos.shape[0]
    feats = {}
    feats['degree_mean'] = float(deg.mean()) if n_node else 0.0
    feats['degree_max'] = int(deg.max()) if n_node else 0
    feats['junction_count'] = int((deg >= 3).sum())
    feats['degree_hist'] = deg.astype(float)
    feats['edge_len_q90'] = (float(np.quantile(lens, 0.9))
                             if lens.size else 0.0)
    if ang.size:
        hist, _ = np.histogram(ang, bins=ORIENT_BINS, range=(0.0, np.pi))
        feats['orient_bias'] = float(hist.max() / float(hist.sum()))
    else:
        feats['orient_bias'] = 0.0
    if n_node:
        if rect is not None:
            x0, y0, x1, y1 = (float(v) for v in rect)
            x0, x1 = min(x0, x1), max(x0, x1)
            y0, y1 = min(y0, y1), max(y0, y1)
        else:
            (x0, y0), (x1, y1) = pos.min(0), pos.max(0)
        tol = 1e-6 * max(x1 - x0, y1 - y0, 1e-12) + 1e-9
        on = ((np.abs(pos[:, 0] - x0) <= tol)
              | (np.abs(pos[:, 0] - x1) <= tol)
              | (np.abs(pos[:, 1] - y0) <= tol)
              | (np.abs(pos[:, 1] - y1) <= tol))
        feats['boundary_ratio'] = float(on.mean())
        area = max((x1 - x0) * (y1 - y0), 1e-12)
        feats['node_density'] = float(n_node / area)
        feats['edge_density'] = float(lens.sum() / area) if lens.size else 0.0
    else:
        feats['boundary_ratio'] = 0.0
        feats['node_density'] = 0.0
        feats['edge_density'] = 0.0
    nn = _nn_distances(pos)
    feats['node_spacing'] = nn
    feats['nn_dist_mean'] = float(nn.mean()) if nn.size else 0.0
    feats['nn_dist_cv'] = (float(nn.std() / nn.mean())
                           if nn.size and nn.mean() > 0 else 0.0)
    st = _chain_straightness(pos, edges)
    feats['segment_straightness'] = st
    feats['straightness_mean'] = float(st.mean()) if st.size else 0.0
    return feats


def _segment_interactions(pos, edges):
    '''Pairwise segment test with bbox prefilter (vectorized).

    Returns dict with cross_pts (K,2), cross_pairs (K,2) edge ids,
    cross_t (K,2) intersection parameters along each edge, and
    overlap_lens (collinear shared-stretch lengths).  Pairs sharing a
    node are skipped, so only contacts between distinct fibers count;
    T-junctions (endpoint on another segment) are proper crossings.
    '''
    n_edge = int(edges.shape[0])
    out = dict(cross_pts=np.zeros((0, 2)),
               cross_pairs=np.zeros((0, 2), int),
               cross_t=np.zeros((0, 2)), overlap_lens=[])
    if n_edge < 2:
        return out
    p1 = pos[edges[:, 0]]
    p2 = pos[edges[:, 1]]
    lo = np.minimum(p1, p2)
    hi = np.maximum(p1, p2)
    ia, ib = np.triu_indices(n_edge, 1)
    share = ((edges[ia, 0] == edges[ib, 0]) | (edges[ia, 0] == edges[ib, 1])
             | (edges[ia, 1] == edges[ib, 0]) | (edges[ia, 1] == edges[ib, 1]))
    pad = 1e-9
    box = ((lo[ia, 0] <= hi[ib, 0] + pad) & (lo[ib, 0] <= hi[ia, 0] + pad)
           & (lo[ia, 1] <= hi[ib, 1] + pad) & (lo[ib, 1] <= hi[ia, 1] + pad))
    ia, ib = ia[box & ~share], ib[box & ~share]
    if ia.size == 0:
        return out
    a, b = p1[ia], p2[ia]
    c, d = p1[ib], p2[ib]
    r = b - a
    s = d - c
    lr = np.hypot(r[:, 0], r[:, 1])
    ls = np.hypot(s[:, 0], s[:, 1])
    den = r[:, 0] * s[:, 1] - r[:, 1] * s[:, 0]
    qp = c - a
    qxs = qp[:, 0] * s[:, 1] - qp[:, 1] * s[:, 0]
    qxr = qp[:, 0] * r[:, 1] - qp[:, 1] * r[:, 0]
    parallel = np.abs(den) <= 1e-12 * np.maximum(lr * ls, 1.0)
    den_safe = np.where(parallel, 1.0, den)
    t = qxs / den_safe
    u = qxr / den_safe
    tol = 1e-6
    cross = (~parallel & (t >= -tol) & (t <= 1 + tol)
             & (u >= -tol) & (u <= 1 + tol))
    ci = np.where(cross)[0]
    out['cross_pts'] = a[ci] + t[ci][:, None] * r[ci]
    out['cross_pairs'] = np.column_stack((ia[ci], ib[ci]))
    out['cross_t'] = np.column_stack((t[ci], u[ci]))
    if parallel.any():
        dist = np.abs(qxr[parallel]) / np.maximum(lr[parallel], 1e-30)
        collin = dist <= 1e-6 * np.maximum(lr[parallel], 1.0)
        idx = np.where(parallel)[0][collin]
        if idx.size:
            lr2 = np.maximum(lr[idx] ** 2, 1e-30)
            tc = ((c[idx] - a[idx]) * r[idx]).sum(1) / lr2
            td = ((d[idx] - a[idx]) * r[idx]).sum(1) / lr2
            ov = (np.minimum(1.0, np.maximum(tc, td))
                  - np.maximum(0.0, np.minimum(tc, td)))
            good = ov > 1e-9
            out['overlap_lens'] = (ov[good] * lr[idx][good]).tolist()
    return out


def _planarize(pos, edges, inter):
    '''Split segments at crossings / T-junctions -> planar line graph.'''
    n_edge = int(edges.shape[0])
    nodes = [tuple(map(float, p)) for p in pos]
    coord_id = {(round(x, 6), round(y, 6)): i
                for i, (x, y) in enumerate(nodes)}

    def node_at(pt):
        key = (round(float(pt[0]), 6), round(float(pt[1]), 6))
        nid = coord_id.get(key)
        if nid is None:
            nid = len(nodes)
            coord_id[key] = nid
            nodes.append((float(pt[0]), float(pt[1])))
        return nid

    splits = [[] for _ in range(n_edge)]
    cp = inter['cross_pairs']
    ct = inter['cross_t']
    for k in range(cp.shape[0]):
        nid = node_at(inter['cross_pts'][k])
        t0 = min(1.0, max(0.0, float(ct[k, 0])))
        t1 = min(1.0, max(0.0, float(ct[k, 1])))
        splits[int(cp[k, 0])].append((t0, nid))
        splits[int(cp[k, 1])].append((t1, nid))
    # nodes lying on an edge interior (T-junctions) also split it
    chunk = 200
    for e0 in range(0, n_edge, chunk):
        e1 = min(e0 + chunk, n_edge)
        aa = pos[edges[e0:e1, 0]]
        rr = pos[edges[e0:e1, 1]] - aa
        l2 = np.maximum((rr ** 2).sum(1), 1e-30)
        t_all = (((pos[:, None, :] - aa[None]) * rr[None]).sum(2)
                 / l2[None])
        proj = aa[None] + t_all[:, :, None] * rr[None]
        dist = np.hypot(pos[:, None, 0] - proj[:, :, 0],
                        pos[:, None, 1] - proj[:, :, 1])
        hit = (dist <= 1e-6) & (t_all > 1e-6) & (t_all < 1 - 1e-6)
        for nid, ei in zip(*np.where(hit)):
            ge = e0 + int(ei)
            if nid in (edges[ge, 0], edges[ge, 1]):
                continue
            splits[ge].append((float(t_all[nid, ei]), int(nid)))
    new_edges = []
    for e in range(n_edge):
        seq = sorted(set([(0.0, int(edges[e, 0])), (1.0, int(edges[e, 1]))]
                         + splits[e]), key=lambda z: z[0])
        clean = [seq[0]]
        for t, nid in seq[1:]:
            if t - clean[-1][0] > 1e-9:
                clean.append((t, nid))
        for (_, n1), (_, n2) in zip(clean, clean[1:]):
            if n1 != n2:
                new_edges.append(sorted((n1, n2)))
    new_pos = np.asarray(nodes, float)
    if new_edges:
        new_edges = np.asarray(sorted(set(map(tuple, new_edges))), int)
    else:
        new_edges = np.zeros((0, 2), int)
    return new_pos, new_edges


def _poly_area(q):
    x, y = q[:, 0], q[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _poly_perim(q):
    dx = np.roll(q[:, 0], -1) - q[:, 0]
    dy = np.roll(q[:, 1], -1) - q[:, 1]
    return float(np.hypot(dx, dy).sum())


def _faces(pos, edges):
    '''Bounded faces of a planar straight-line graph.

    Half-edge walk over the angular rotation system: from directed edge
    (u, v) the next edge is (v, w) where w is the neighbour of v just
    clockwise of the reverse direction.  Bounded faces come out CCW
    (positive signed shoelace area); the unbounded face is discarded.
    Returns per-face vertex-id arrays and per-face directed edges.
    '''
    if edges.shape[0] == 0:
        return [], []
    adj = [[] for _ in range(pos.shape[0])]
    for a, b in edges:
        a, b = int(a), int(b)
        adj[a].append(b)
        adj[b].append(a)
    order = {}
    index = {}
    for v in range(pos.shape[0]):
        nb = adj[v]
        if not nb:
            order[v] = []
            index[v] = {}
            continue
        ang = np.arctan2(pos[nb, 1] - pos[v, 1], pos[nb, 0] - pos[v, 0])
        nb = [nb[i] for i in np.argsort(ang)]
        order[v] = nb
        index[v] = {w: i for i, w in enumerate(nb)}
    guard = 4 * edges.shape[0] + 8
    faces = []
    dired = []
    seen = set()
    for a, b in edges:
        for u0, v0 in ((int(a), int(b)), (int(b), int(a))):
            if (u0, v0) in seen:
                continue
            u, v = u0, v0
            face = []
            closed = False
            for _ in range(guard):
                seen.add((u, v))
                face.append(u)
                nb = order[v]
                w = nb[(index[v][u] - 1) % len(nb)]
                u, v = v, w
                if (u, v) == (u0, v0):
                    closed = True
                    break
            if not closed or len(face) < 3:
                continue
            q = pos[face]
            if _poly_area(q) > 1e-9:
                faces.append(np.asarray(face, int))
                dired.append(list(zip(face, face[1:] + face[:1])))
    return faces, dired


def _pore_features(pos, edges, inter, area):
    '''Bounded faces of the planarized network + shape/adjacency stats.'''
    ppos, pedges = _planarize(pos, edges, inter)
    faces, dired = _faces(ppos, pedges)
    empty = np.zeros(0)
    if not faces:
        feats = dict(pore_count=0, pore_area_mean=0.0, pore_area_cv=0.0,
                     largest_pore_ratio=0.0, porosity=0.0,
                     pore_aspect_mean=0.0, pore_perim_mean=0.0,
                     pore_neighbor_mean=0.0)
        return feats, dict(areas=empty, aspects=empty, neighbors=empty)
    polys = [ppos[f] for f in faces]
    areas = np.asarray([_poly_area(q) for q in polys], float)
    perims = np.asarray([_poly_perim(q) for q in polys], float)
    ws = np.asarray([q[:, 0].max() - q[:, 0].min() for q in polys])
    hs = np.asarray([q[:, 1].max() - q[:, 1].min() for q in polys])
    aspects = np.maximum(ws, hs) / np.maximum(np.minimum(ws, hs), 1e-12)
    dir2face = {}
    for fi, de in enumerate(dired):
        for uv in de:
            dir2face[uv] = fi
    nbr = [set() for _ in faces]
    for (u, v), fi in dir2face.items():
        fj = dir2face.get((v, u))
        if fj is not None and fj != fi:
            nbr[fi].add(fj)
    neighbors = np.asarray([len(s) for s in nbr], float)
    mean = float(areas.mean())
    total = float(areas.sum())
    feats = dict(pore_count=int(areas.size), pore_area_mean=mean,
                 pore_area_cv=float(areas.std() / mean) if mean > 0 else 0.0,
                 largest_pore_ratio=float(areas.max() / total)
                 if total > 0 else 0.0,
                 porosity=float(min(total / area, 1.0)),
                 pore_aspect_mean=float(aspects.mean()),
                 pore_perim_mean=float(perims.mean()),
                 pore_neighbor_mean=float(neighbors.mean()))
    return feats, dict(areas=areas, aspects=aspects, neighbors=neighbors)


def _contact_features(pos, edges, inter):
    '''Crossing/overlap stats plus crossing angles and contact clusters.'''
    n_edge = int(edges.shape[0])
    cp = inter['cross_pairs']
    cross_count = int(cp.shape[0])
    overlap_lens = np.asarray(inter['overlap_lens'], float)
    feats = dict(cross_count=cross_count,
                 cross_per_edge=float(cross_count / n_edge) if n_edge else 0.0,
                 overlap_len_mean=float(overlap_lens.mean())
                 if overlap_lens.size else 0.0)
    if cross_count:
        d = pos[edges[:, 1]] - pos[edges[:, 0]]
        ea = np.mod(np.arctan2(d[:, 1], d[:, 0]), np.pi)
        diff = np.abs(ea[cp[:, 0]] - ea[cp[:, 1]])
        angles = np.degrees(np.minimum(diff, np.pi - diff))
    else:
        angles = np.zeros(0)
    feats['cross_angle'] = angles
    feats['cross_angle_mean'] = float(angles.mean()) if angles.size else 0.0
    feats['cross_pos_ratio'] = (float(np.unique(cp).size / n_edge)
                                if n_edge and cross_count else 0.0)
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in cp:
        a, b = int(a), int(b)
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    if parent:
        sizes = np.bincount([find(x) for x in parent])
        sizes = sizes[sizes > 0]
    else:
        sizes = np.zeros(0, int)
    feats['contact_cluster_max'] = int(sizes.max()) if sizes.size else 0
    feats['contact_cluster_mean'] = float(sizes.mean()) if sizes.size else 0.0
    feats['overlap_len'] = overlap_lens
    return feats


def compute_features(pos, edges, rect=None):
    '''Scalar feature dict + histogram arrays for one network snapshot.'''
    pos, edges = _filter_region(pos, edges, rect)
    feats, lens, ang, deg = _structure_features(pos, edges)
    feats.update(_structure_extra(pos, edges, lens, ang, deg, rect))
    inter = _segment_interactions(pos, edges)
    area = _region_area(pos, rect)
    pore_feats, pore_arr = _pore_features(pos, edges, inter, area)
    feats.update(pore_feats)
    feats.update(_contact_features(pos, edges, inter))
    feats['edge_lengths'] = lens
    feats['orientations'] = ang
    feats['pore_areas'] = pore_arr['areas']
    feats['pore_aspect'] = pore_arr['aspects']
    feats['pore_neighbor'] = pore_arr['neighbors']
    return feats
