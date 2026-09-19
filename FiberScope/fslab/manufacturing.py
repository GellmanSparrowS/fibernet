"""Canonical multigraph compilation and deformation-invariant continuous routes."""
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import numpy as np
from .cell_rules import transform_cell, graph_health

MAX_MANUFACTURING_POINTS = 180000
MAX_MANUFACTURING_EDGES = 240000


@lru_cache(maxsize=4)
def _route(edge_bytes, count):
    edges = np.frombuffer(edge_bytes, dtype=np.int64).reshape(-1, 2)
    adjacency = [[] for _ in range(count)]
    for eid, (a, b) in enumerate(edges):
        adjacency[a].append((int(b), eid))
        adjacency[b].append((int(a), eid))
    used = np.zeros(len(edges), bool)
    stack, incoming, vertices, ids = [int(edges[0, 0])], [-1], [], []
    while stack:
        a = stack[-1]
        while adjacency[a] and used[adjacency[a][-1][1]]:
            adjacency[a].pop()
        if adjacency[a]:
            b, eid = adjacency[a].pop()
            used[eid] = True
            stack.append(b)
            incoming.append(eid)
        else:
            vertices.append(stack.pop())
            ids.append(incoming.pop())
    vertices, ids = vertices[::-1], ids[::-1][1:]
    if len(ids) != len(edges) or vertices[0] != vertices[-1]:
        raise ValueError('graph has no closed all-edge manufacturing route')
    return tuple(vertices), tuple(ids)


@dataclass
class ManufacturingNetwork:
    reference: np.ndarray
    positions: np.ndarray
    edges: np.ndarray
    added_edges: int = 0
    bridges: int = 0

    def __post_init__(self):
        self.reference = np.asarray(self.reference, float).reshape(-1, 3)
        self.positions = np.asarray(self.positions, float).reshape(-1, 3)
        self.edges = np.asarray(self.edges, np.int64).reshape(-1, 2)
        if len(self.positions) > MAX_MANUFACTURING_POINTS or len(self.edges) > MAX_MANUFACTURING_EDGES:
            raise MemoryError('manufacturing graph budget exceeded; reduce cells or surface faces')
        if self.positions.shape != self.reference.shape or not np.isfinite(self.positions).all():
            raise ValueError('invalid manufacturing coordinates')
        self.health = graph_health(self.positions, self.edges)
        if self.health['components'] != 1 or self.health['odd']:
            raise ValueError('manufacturing graph must be connected and all-even')
        vertices, edges = _route(self.edges.tobytes(), len(self.positions))
        self.route_nodes = np.asarray(vertices, np.int64)
        self.route_edges = np.asarray(edges, np.int64)
        self.topology_id = hashlib.sha256(self.edges.tobytes()).hexdigest()


def _connect(reference, edges):
    """Connect components in the reference geometry, never in deformed space."""
    from scipy.spatial import cKDTree
    parent = list(range(len(reference)))
    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def join(a, b):
        a, b = root(a), root(b)
        if a == b:
            return False
        parent[b] = a
        return True
    for a, b in edges:
        join(a, b)
    added = 0
    if len({root(i) for i in range(len(reference))}) == 1:
        return 0
    tree = cKDTree(reference)
    distances, neighbors = tree.query(reference, k=min(12, len(reference)))
    candidates = [(float(d), a, int(b)) for a, (ds, ns) in enumerate(zip(distances, neighbors))
                  for d, b in zip(np.atleast_1d(ds)[1:], np.atleast_1d(ns)[1:]) if a < b]
    for _, a, b in sorted(candidates):
        if join(a, b):
            edges.extend([(a, b), (b, a)])
            added += 2
    representatives = {}
    for i in range(len(reference)):
        representatives.setdefault(root(i), i)
    # Sparse fallback for well-separated components; no quadratic distance matrix.
    reps = sorted(representatives.values(), key=lambda i: tuple(reference[i]))
    for a, b in zip(reps, reps[1:]):
        edges.extend([(a, b), (b, a)])
        added += 2
    return added


def _even_edges(reference, edges):
    """Parity correction by a deterministic tree T-join, not edge deduplication."""
    n = len(reference)
    adjacency = [[] for _ in range(n)]
    parity = np.zeros(n, np.uint8)
    for eid, (a, b) in enumerate(edges):
        adjacency[a].append((b, eid))
        adjacency[b].append((a, eid))
        parity[a] ^= 1
        parity[b] ^= 1
    parents = [-1] * n
    order, stack = [], [0]
    parents[0] = 0
    while stack:
        a = stack.pop()
        order.append(a)
        for b, _ in adjacency[a]:
            if parents[b] < 0:
                parents[b] = a
                stack.append(b)
    added = 0
    for b in reversed(order[1:]):
        if parity[b]:
            a = parents[b]
            edges.append((b, a))
            parity[a] ^= 1
            added += 1
    return added


def _split_junctions(reference, edges, positions=None):
    """Resolve even high-degree junctions into degree-four rings before deformation."""
    reference = list(np.asarray(reference, float))
    actual = list(np.asarray(positions, float)) if positions is not None else None
    incidence = [[] for _ in reference]
    for eid, (a, b) in enumerate(edges):
        incidence[a].append((eid, 0, b))
        incidence[b].append((eid, 1, a))
    edges = [list(e) for e in edges]
    for center, hits in enumerate(incidence):
        if len(hits) <= 8:
            continue
        origin = reference[center]
        hits.sort(key=lambda h: (np.arctan2(*(reference[h[2]]-origin)[[1, 0]]), h[0]))
        length = min(np.linalg.norm(reference[h[2]]-origin) for h in hits)
        radius = max(length * .025, 1e-5)
        ports = []
        for k in range(len(hits)//2):
            angle = 2*np.pi*k/(len(hits)//2)
            ports.append(len(reference))
            offset = radius*np.array([np.cos(angle), np.sin(angle), 0.])
            reference.append(origin + offset)
            if actual is not None:
                actual.append(actual[center] + offset)
            for eid, side, _ in hits[2*k:2*k+2]:
                edges[eid][side] = ports[-1]
        edges.extend([[a, b] for a, b in zip(ports, ports[1:]+ports[:1])])
    used = np.unique(np.asarray(edges))
    remap = np.full(len(reference), -1, int)
    remap[used] = np.arange(len(used))
    result = (np.asarray(reference)[used], remap[np.asarray(edges)])
    return (*result, np.asarray(actual)[used]) if actual is not None else result


def compile_planar(factory, boundary_twins=True):
    """One canonical topology for every spectrum/overlap of this unit and tiling."""
    from .structure import CELL, MAX_NODES, fit_spectrum
    from .cell_cycles import closed_cell_edges
    factory = replace(factory).clamped()
    base = replace(factory, grid_x=1, grid_y=1, n_pts_per_side=0,
                   line_displacements=None, wave=False, node_offsets=None,
                   perturbation=0., expansion_rule='translate', topology='legacy').build()
    nodes = np.asarray(base.node_positions(), float)
    base_edges = closed_cell_edges(nodes, np.asarray(base.edge_array(), int)[:, :2])
    if not len(base_edges):
        raise ValueError('cell has no fiber edges')
    if len(nodes)*factory.grid_x*factory.grid_y > MAX_NODES:
        raise MemoryError('canonical cell budget exceeded')
    lookup, reference, edges, handedness, arc_centers = {}, [], [], [], []
    for j in range(factory.grid_y):
        for i in range(factory.grid_x):
            positions = transform_cell(nodes, i, j, factory.expansion_rule, CELL, factory.custom_rule)
            ids = []
            for p in positions:
                key = tuple(np.round(p, 8))
                if key not in lookup:
                    lookup[key] = len(reference)
                    reference.append(p)
                ids.append(lookup[key])
            basis = transform_cell(np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),
                                   i, j, factory.expansion_rule, CELL, factory.custom_rule)
            sign = np.sign(np.linalg.det((basis[1:,:2]-basis[0,:2]).T))
            center = np.array([CELL*.5,CELL*.5,0.])
            mapped_center = transform_cell(center[None,:],i,j,factory.expansion_rule,CELL,factory.custom_rule)[0]
            for a, b in base_edges:
                if ids[a] != ids[b]:
                    edges.append((ids[a], ids[b]))
                    handedness.append(sign)
                    circular = factory.unit=='ring' and np.allclose(
                        [np.linalg.norm(nodes[a]-center),np.linalg.norm(nodes[b]-center)], CELL*.36)
                    arc_centers.append(mapped_center if circular else None)
    reference = np.asarray(reference)
    used = np.unique(edges)
    remap = np.full(len(reference), -1, int)
    remap[used] = np.arange(len(used))
    reference, edges = reference[used], remap[np.asarray(edges)].tolist()
    # Complete single boundary fibers with independent oppositely directed twins.
    groups = {}
    for eid, (a, b) in enumerate(edges):
        groups.setdefault(tuple(sorted((a,b))), []).append(eid)
    boundary_count = 0
    for hits in groups.values():
        if boundary_twins and len(hits) % 2:
            eid = hits[0]
            a, b = edges[eid]
            edges.append((b,a))
            handedness.append(handedness[eid])
            arc_centers.append(arc_centers[eid])
            boundary_count += 1
    bridges = _connect(reference, edges)
    added = boundary_count + (len(base_edges)-len(base.edge_array()))*factory.grid_x*factory.grid_y
    handedness.extend([1.] * bridges)
    arc_centers.extend([None] * bridges)
    positions = reference.copy()
    for x, y, dx, dy in factory.node_offsets or []:
        mask = np.linalg.norm(reference[:, :2]-[x, y], axis=1) < 1e-6
        positions[mask, :2] += [dx, dy]
    rng = np.random.default_rng(factory.seed)
    pts = factory.n_pts_per_side
    if len(reference)+len(edges)*pts > MAX_MANUFACTURING_POINTS or len(edges)*(pts+1) > MAX_MANUFACTURING_EDGES:
        raise MemoryError('manufacturing graph budget exceeded; reduce grid or control points')
    spectrum = factory.effective_spectrum()
    ref_out, out, segments = list(reference), list(positions), []
    for (a, b), sign, center in zip(edges, handedness, arc_centers):
        delta = positions[b]-positions[a]
        normal = sign * np.array([-delta[1], delta[0], 0.])
        chain = [int(a)]
        for k, (dx, dy) in enumerate(spectrum):
            t = (k+1)/(pts+1)
            chain.append(len(out))
            linear = (1-t)*reference[a]+t*reference[b]
            curve = linear if center is None else (center+np.cos(t*np.pi/2)*(reference[a]-center)
                                                    +np.sin(t*np.pi/2)*(reference[b]-center))
            ref_out.append(curve)
            out.append((1-t)*positions[a]+t*positions[b]+curve-linear+dx*delta+dy*normal)
        chain.append(int(b))
        segments.extend(zip(chain[:-1], chain[1:]))
    out = np.asarray(out)
    ref_out, segments, out = _split_junctions(ref_out, segments, out)
    return ManufacturingNetwork(ref_out, out, segments, added, bridges)


def compile_surface(vertices, faces, factory, overscale=1.06):
    """Map closed units; share only prescribed mesh-corner identities."""
    from .surface_mapping import bilinear, coarsen_quads
    cell = compile_planar(replace(factory, grid_x=1, grid_y=1, node_offsets=None),
                          boundary_twins=False)
    vertices, faces = np.asarray(vertices,float), np.asarray(faces,int)
    if not 1. <= overscale <= 1.3 or faces.ndim!=2 or faces.shape[1]!=4:
        raise ValueError('surface requires quad faces and overscale 1..1.3')
    allowed = min(MAX_MANUFACTURING_POINTS//(len(cell.positions)+8),
                  MAX_MANUFACTURING_EDGES//(len(cell.edges)+32))
    if len(faces)>allowed:
        vertices,faces=coarsen_quads(vertices,faces,allowed)
        faces=np.asarray(faces,int)
    low,span=cell.reference[:,:2].min(0),np.ptp(cell.reference[:,:2],axis=0)
    if np.any(span<1e-8):
        raise ValueError('surface unit must have nonzero width and height')
    raw=(cell.reference[:,:2]-low)/span
    uv0=(raw-.5)*overscale+.5
    uv=((cell.positions[:,:2]-low)/span-.5)*overscale+.5
    corners=np.array([[0,0],[1,0],[1,1],[0,1]],float)
    corner_ids={}
    for k,c in enumerate(corners):
        for node in np.flatnonzero(np.linalg.norm(raw-c,axis=1)<1e-8):
            corner_ids[int(node)]=k
            uv0[node]=c
            uv[node]=c
    reference,positions,edges,patches=[],[],[],[]
    shared={}
    for face in faces:
        ref,actual=bilinear(vertices[face],uv0),bilinear(vertices[face],uv)
        ids=[]
        for node in range(len(raw)):
            key=int(face[corner_ids[node]]) if node in corner_ids else None
            if key is not None and key in shared:
                ids.append(shared[key])
            else:
                ids.append(len(positions))
                reference.append(ref[node]); positions.append(actual[node])
                if key is not None: shared[key]=ids[-1]
        patches.append(ids)
        edges.extend((ids[a],ids[b]) for a,b in cell.edges)
    seams={}
    for fid,face in enumerate(faces):
        for side in range(4):
            seams.setdefault(tuple(sorted((int(face[side]),int(face[(side+1)%4])))),[]).append((fid,side))
    sides=np.array([[.5,0],[1,.5],[.5,1],[0,.5]])
    targets=[int(np.argmin(np.linalg.norm(raw-point,axis=1))) for point in sides]
    bridges=0
    for seam,adjacent in seams.items():
        if len(adjacent)<2 or all(k in shared for k in seam):
            continue
        anchor=len(positions)
        point=vertices[list(seam)].mean(0)
        positions.append(point); reference.append(point)
        for fid,side in adjacent:
            node=patches[fid][targets[side]]
            edges.extend(((node,anchor),(anchor,node))); bridges+=2
    bridges+=_connect(np.asarray(reference),edges)
    reference,edges,positions=_split_junctions(reference,edges,positions)
    result=ManufacturingNetwork(reference,positions,edges,cell.added_edges*len(faces),bridges)
    result.mapped_faces=len(faces)
    return result
