"""
Patches for known generator bugs.
Applied at import time via api.py.
"""
import numpy as np
from typing import Optional, Tuple
from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork, Crosslink
from fibernet.core.material import Material


def _edges_to_network_safe(nodes, edges, radius=0.1, material=None, dimension=2):
    """Build FiberNetwork from node/edge dict with ALL crosslink pairs at junctions."""
    fibers = []
    valid_edges = []
    for (u, v) in edges:
        if u not in nodes or v not in nodes:
            continue
        p1 = np.array(nodes[u], dtype=float)
        p2 = np.array(nodes[v], dtype=float)
        if len(p1) == 2:
            p1 = np.append(p1, 0.0)
        if len(p2) == 2:
            p2 = np.append(p2, 0.0)
        length = np.linalg.norm(p2 - p1)
        if length < 1e-10:
            continue
        fiber = Fiber(centerline=np.array([p1, p2]), radius=radius, material=material)
        fibers.append(fiber)
        valid_edges.append((u, v))

    net = FiberNetwork(fibers=fibers, dimension=dimension)

    node_to_fibers = {}
    for i, (u, v) in enumerate(valid_edges):
        for node in [u, v]:
            node_to_fibers.setdefault(node, []).append(i)

    crosslinks = []
    for node, fiber_indices in node_to_fibers.items():
        if len(fiber_indices) < 2:
            continue
        pos = np.array(nodes[node], dtype=float)
        if len(pos) == 2:
            pos = np.append(pos, 0.0)
        seen = set()
        for ii in range(len(fiber_indices)):
            for jj in range(ii + 1, len(fiber_indices)):
                pair = (min(fiber_indices[ii], fiber_indices[jj]),
                        max(fiber_indices[ii], fiber_indices[jj]))
                if pair in seen:
                    continue
                seen.add(pair)
                crosslinks.append(Crosslink(
                    position=pos, fiber_i=pair[0], fiber_j=pair[1],
                    param_i=0.5, param_j=0.5,
                ))

    net.crosslinks = crosslinks
    return net


def lattice_3d_fixed(
    topology: str = "octet",
    cell_size: float = 10.0,
    grid_size: Tuple[int, int, int] = (3, 3, 3),
    perturbation: float = 0.0,
    radius: float = 0.1,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> FiberNetwork:
    """Fixed 3D lattice with correct node indexing and multi-junction crosslinks."""
    topology = topology.lower()
    rng = np.random.RandomState(seed)
    s = cell_size
    nx, ny, nz = grid_size

    all_nodes = {}
    all_edges = []

    def add_node(pos):
        nid = len(all_nodes)
        p = np.array(pos, dtype=float)
        if perturbation > 0:
            p += rng.uniform(-perturbation, perturbation, 3) * s
        all_nodes[nid] = tuple(p)
        return nid

    if topology == 'cubic':
        idx_map = {}
        for ix in range(nx + 1):
            for iy in range(ny + 1):
                for iz in range(nz + 1):
                    nid = add_node((ix*s, iy*s, iz*s))
                    idx_map[(ix, iy, iz)] = nid

        for (ix, iy, iz), nid in idx_map.items():
            for di, dj, dk in [(1,0,0), (0,1,0), (0,0,1)]:
                key = (ix+di, iy+dj, iz+dk)
                if key in idx_map:
                    all_edges.append((nid, idx_map[key]))

    elif topology == 'octet':
        idx_map = {}
        for ix in range(nx + 1):
            for iy in range(ny + 1):
                for iz in range(nz + 1):
                    nid = add_node((ix*s, iy*s, iz*s))
                    idx_map[(ix, iy, iz)] = nid

        for (ix, iy, iz), nid in idx_map.items():
            for di, dj, dk in [(1,0,0), (0,1,0), (0,0,1),
                                (1,1,0), (1,-1,0), (1,0,1), (1,0,-1), (0,1,1), (0,1,-1)]:
                key = (ix+di, iy+dj, iz+dk)
                if key in idx_map:
                    all_edges.append((nid, idx_map[key]))

    elif topology == 'diamond':
        # Diamond: FCC basis with tetrahedral bonds
        node_positions = []
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    for pos in [
                        (ix*s, iy*s, iz*s),
                        ((ix+0.5)*s, (iy+0.5)*s, iz*s),
                        ((ix+0.5)*s, iy*s, (iz+0.5)*s),
                        (ix*s, (iy+0.5)*s, (iz+0.5)*s),
                    ]:
                        add_node(pos)

        # Connect nearest neighbors within threshold
        threshold = s * 0.8
        node_list = list(all_nodes.keys())
        for i in range(len(node_list)):
            for j in range(i+1, min(i+20, len(node_list))):
                p1 = np.array(all_nodes[node_list[i]])
                p2 = np.array(all_nodes[node_list[j]])
                dist = np.linalg.norm(p1 - p2)
                if 1e-6 < dist < threshold:
                    all_edges.append((node_list[i], node_list[j]))

    elif topology in ('gyroid', 'plate'):
        from fibernet.gen.metamaterials import gyroid_lattice_3d, plate_lattice_3d
        if topology == 'gyroid':
            return gyroid_lattice_3d(cell_size=cell_size, **kwargs)
        else:
            return plate_lattice_3d(cell_size=cell_size, **kwargs)
    else:
        raise ValueError(f"Unknown topology: {topology}")

    return _edges_to_network_safe(all_nodes, all_edges, radius=radius,
                                  material=material, dimension=3)
