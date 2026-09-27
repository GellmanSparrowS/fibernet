"""Finite-width geometric overlap, with explicit resolution and memory budgets."""
from dataclasses import dataclass
import numpy as np


def candidate_pairs(pos, edges, padding=0.0, limit=300000):
    """Yield nonadjacent bbox pairs without allocating an E x E matrix."""
    ends = pos[edges]
    lo, hi = ends.min(1), ends.max(1)
    count = 0
    for a in range(len(edges)):
        ids = np.arange(a + 1, len(edges))
        hit = np.all(lo[ids] <= hi[a] + padding, axis=1)
        hit &= np.all(hi[ids] + padding >= lo[a], axis=1)
        hit &= np.all(edges[ids] != edges[a, 0], axis=1)
        hit &= np.all(edges[ids] != edges[a, 1], axis=1)
        ids = ids[hit]
        count += len(ids)
        if count > limit:
            raise MemoryError('contact candidate budget exceeded; reduce grid size')
        for b in ids:
            yield a, int(b)


@dataclass
class ContactConfig:
    width: float = 0.2
    resolution: int = 512
    max_pixels: int = 2000000

    def compute(self, pos, edges):
        width = float(self.width)
        resolution = int(self.resolution)
        if not np.isfinite(width) or width < 0 or not 64 <= resolution <= 1024:
            raise ValueError('contact width must be nonnegative; resolution 64..1024')
        out = dict(contact_width=width, contact_resolution=resolution,
                   contact_pair_count=0, contact_edge_ratio=0.0,
                   contact_overlap_area=0.0, contact_coverage_ratio=0.0,
                   contact_patch_count=0, contact_pixel_size=0.0,
                   contact_graph_max=0)
        if not len(edges) or width == 0:
            return out
        lo = pos.min(0) - width
        span = float(np.max(pos.max(0) - lo) + width)
        step = span / (resolution - 1)
        out['contact_pixel_size'] = step
        union = np.zeros(resolution * resolution, bool)
        overlap = np.zeros_like(union)
        pixels = []
        budget = 0
        for a, b in edges:
            p, q = pos[a], pos[b]
            start = np.maximum(0, np.floor((np.minimum(p, q)-width/2-lo)/step).astype(int))
            stop = np.minimum(resolution-1, np.ceil((np.maximum(p, q)+width/2-lo)/step).astype(int))
            x, y = np.meshgrid(np.arange(start[0], stop[0]+1),
                               np.arange(start[1], stop[1]+1))
            xy = np.column_stack((x.ravel(), y.ravel())) * step + lo
            v = q-p
            t = np.clip((xy-p) @ v / max(float(v @ v), 1e-30), 0, 1)
            hit = np.sum((xy-p-t[:, None]*v)**2, axis=1) <= (width/2)**2
            ids = (y.ravel()*resolution+x.ravel())[hit].astype(np.int32)
            budget += len(ids)
            if budget > self.max_pixels:
                raise MemoryError('fiber coverage budget exceeded; reduce resolution or grid')
            pixels.append(ids)
            union[ids] = True
        degree = np.zeros(len(edges), int)
        parent = np.arange(len(edges))

        def root(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for a, b in candidate_pairs(pos, edges, width):
            common = np.intersect1d(pixels[a], pixels[b], assume_unique=True)
            if len(common):
                overlap[common] = True
                degree[[a, b]] += 1
                parent[root(a)] = root(b)
                out['contact_pair_count'] += 1
        active = np.flatnonzero(degree)
        if len(active):
            counts = np.bincount([root(a) for a in active])
            out['contact_graph_max'] = int(counts.max())
        # Four-neighbour connected overlap patches; fixed canvas bounds memory.
        pending = set(np.flatnonzero(overlap).tolist())
        patches = 0
        while pending:
            patches += 1
            stack = [pending.pop()]
            while stack:
                k = stack.pop()
                x, y = k % resolution, k // resolution
                neighbours = []
                if x: neighbours.append(k-1)
                if x+1 < resolution: neighbours.append(k+1)
                if y: neighbours.append(k-resolution)
                if y+1 < resolution: neighbours.append(k+resolution)
                for j in neighbours:
                    if j in pending:
                        pending.remove(j)
                        stack.append(j)
        out.update(contact_edge_ratio=float(len(active)/len(edges)),
                   contact_overlap_area=float(overlap.sum()*step**2),
                   contact_coverage_ratio=float(overlap.sum()/max(1, union.sum())),
                   contact_patch_count=patches)
        return out
