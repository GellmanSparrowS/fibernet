"""Load-path percolation analysis on stretch trajectories.

At each strain level, edges carrying above-threshold tension form an active
subgraph. The spanning cluster (connected to BOTH gripped sides) is the load
backbone; BFS depth from the two boundaries gives the two-sided blue front
used by the canvas.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class PercolationResult:
    spanning_frac: np.ndarray    # (F,) node fraction in spanning cluster
    backbone_frac: np.ndarray    # (F,) edge fraction: spanning-cluster edges / all edges
    active_edges: np.ndarray     # (F, E) bool: load-bearing edges
    edge_in_spanning: np.ndarray # (F, E) bool
    edge_depth_norm: np.ndarray  # (F, E) float32, 0 at grips -> 1 mid-sample
    perc_frame: int              # first frame with spanning cluster, -1 if none
    threshold_curve: np.ndarray  # (F,) strain threshold used per frame

    def active_mask(self, f: int) -> np.ndarray:
        return self.active_edges[f]


class _UnionFind:
    __slots__ = ("parent", "size")

    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)

    def find(self, x):
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def _frame_analysis(strain_f, edges, n_nodes, left, right, thr,
                    active_override=None):
    E = edges.shape[0]
    active = active_override if active_override is not None else \
        (strain_f > thr) & (strain_f > 0.0)
    idx = np.nonzero(active)[0]
    spanning_mask = np.zeros(E, dtype=bool)
    depth_norm = np.zeros(E, dtype=np.float32)
    if idx.size == 0:
        return 0.0, 0.0, spanning_mask, depth_norm

    uf = _UnionFind(n_nodes)
    ea, eb = edges[idx, 0].astype(np.int64), edges[idx, 1].astype(np.int64)
    for a, b in zip(ea, eb):
        uf.union(a, b)

    roots = np.array([uf.find(i) for i in range(n_nodes)], dtype=np.int64)
    span_roots = set(roots[left]) & set(roots[right])
    if not span_roots:
        return 0.0, 0.0, spanning_mask, depth_norm

    in_span_node = np.isin(roots, list(span_roots))
    spanning_mask[idx] = in_span_node[ea] & in_span_node[eb]

    # multi-source BFS depth from each gripped side, spanning edges only
    adj = [[] for _ in range(n_nodes)]
    sidx = np.nonzero(spanning_mask)[0]
    for k in sidx:
        a, b = int(edges[k, 0]), int(edges[k, 1])
        adj[a].append(b)
        adj[b].append(a)

    def bfs(sources):
        dist = np.full(n_nodes, -1, dtype=np.int64)
        q = []
        for s in sources:
            s = int(s)
            if in_span_node[s] and dist[s] < 0:
                dist[s] = 0
                q.append(s)
        head = 0
        while head < len(q):
            v = q[head]
            head += 1
            dv = dist[v] + 1
            for w in adj[v]:
                if dist[w] < 0:
                    dist[w] = dv
                    q.append(w)
        return dist

    dL = bfs(left)
    dR = bfs(right)
    # nodes in the spanning cluster are reachable from both sides
    dnode = np.where((dL >= 0) & (dR >= 0), np.minimum(dL, dR), -1)
    dedge = np.maximum(dnode[edges[sidx, 0]], dnode[edges[sidx, 1]])
    dmax = dedge.max() if dedge.size else 0
    if dmax > 0:
        depth_norm[sidx] = dedge.astype(np.float32) / float(dmax)
    span_node_frac = float(in_span_node.sum()) / n_nodes
    span_edge_frac = float(spanning_mask.sum()) / E
    return span_node_frac, span_edge_frac, spanning_mask, depth_norm


def _grip_prune(active, edges, n_nodes, left, right):
    """Keep only active edges whose connected component touches a grip.

    Force can only flow along continuous paths anchored at the clamped
    sides; isolated stressed edges are not load-bearing.
    """
    idx = np.nonzero(active)[0]
    if idx.size == 0:
        return active
    uf = _UnionFind(n_nodes)
    for k in idx:
        uf.union(int(edges[k, 0]), int(edges[k, 1]))
    keep_roots = set(int(uf.find(int(i))) for i in left)
    keep_roots |= set(int(uf.find(int(i))) for i in right)
    out = np.zeros_like(active)
    for k in idx:
        if int(uf.find(int(edges[k, 0]))) in keep_roots:
            out[k] = True
    return out


def compute_percolation(run, alpha: float = 0.05, hysteresis: float = 0.6,
                        grip_connected: bool = True,
                        quantile: float = None,
                        min_active: float = 1e-4) -> PercolationResult:
    """Edge activation threshold for the load backbone.

    alpha: edge recruits when strain >= alpha * max strain of the frame.
    hysteresis: once recruited, an edge stays load-bearing while its strain
      remains above hysteresis * alpha * max (release threshold). This
      hysteresis band makes the growing force path stable instead of
      flickering.
    grip_connected: prune active edges to components anchored at a grip, so
      the visible load path always grows in from the clamped sides.
    quantile (optional): per-frame strain quantile instead of alpha.
    """
    F, E = run.edge_strain.shape
    n_nodes = run.frames_xy.shape[1]
    spanning_frac = np.zeros(F, dtype=np.float32)
    backbone_frac = np.zeros(F, dtype=np.float32)
    active_edges = np.zeros((F, E), dtype=bool)
    edge_in_spanning = np.zeros((F, E), dtype=bool)
    edge_depth_norm = np.zeros((F, E), dtype=np.float32)
    thr_curve = np.zeros(F, dtype=np.float32)
    perc_frame = -1
    prev = np.zeros(E, dtype=bool)

    for f in range(F):
        sf = run.edge_strain[f]
        if quantile is not None:
            pos = sf[sf > 0]
            thr = max(float(np.quantile(pos, quantile)), min_active) \
                if pos.size else min_active
        else:
            mx = float(sf.max())
            thr = max(alpha * mx, min_active) if mx > 0 else min_active
        thr_curve[f] = thr
        raw_on = (sf > thr) & (sf > 0.0)
        act = raw_on | (prev & (sf > hysteresis * thr) & (sf > 0.0))
        if grip_connected:
            act = _grip_prune(act, run.edges, n_nodes,
                              run.left_nodes, run.right_nodes)
        active_edges[f] = act
        prev = act
        nf, bf, sm, dn = _frame_analysis(sf, run.edges, n_nodes,
                                         run.left_nodes, run.right_nodes, thr,
                                         active_override=act)
        spanning_frac[f] = nf
        backbone_frac[f] = bf
        edge_in_spanning[f] = sm
        edge_depth_norm[f] = dn
        if perc_frame < 0 and nf > 0:
            perc_frame = f

    return PercolationResult(spanning_frac, backbone_frac, active_edges,
                             edge_in_spanning, edge_depth_norm, perc_frame,
                             thr_curve)
