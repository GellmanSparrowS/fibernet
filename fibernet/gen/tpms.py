"""
Triply Periodic Minimal Surface (TPMS) generators for fiber networks.

All generators now use FiberGraph to ensure proper connectivity
and crosslink creation at shared vertices.

Available TPMS types:
- Gyroid (Schwarz G-surface)
- Schwarz Diamond (D-surface)
- Schwarz Primitive (P-surface)
- I-WP (Wrapped Package)
- Neovius
- Lidinoid
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from fibernet.core.network import FiberNetwork
from fibernet.core.material import Material
from fibernet.gen._graph_builder import FiberGraph

def _merge_components(net, max_dist=5.0):
    """Merge disconnected components by adding bridging fibers."""
    import networkx as nx
    
    if net.num_fibers < 2:
        return net
    
    G = nx.Graph()
    for i in range(net.num_fibers):
        G.add_node(i)
    for cl in net.crosslinks:
        G.add_edge(cl.fiber_i, cl.fiber_j)
    
    components = list(nx.connected_components(G))
    if len(components) <= 1:
        return net
    
    # Find largest component
    sizes = [len(c) for c in components]
    largest_idx = np.argmax(sizes)
    largest = components[largest_idx]
    
    # For each small component, find closest pair to largest and add bridge
    from fibernet.core.fiber import Fiber
    from fibernet.core.network import Crosslink
    
    new_fid = net.num_fibers
    large_fibers = list(largest)
    
    for comp in components:
        if comp == largest:
            continue
        
        comp_list = list(comp)
        min_dist = float('inf')
        best_pair = None
        
        # Find closest fiber pair between component and largest
        for fi in comp_list[:min(10, len(comp_list))]:
            for fj in large_fibers[:min(20, len(large_fibers))]:
                pi = net.fibers[fi].centerline[0]
                pj = net.fibers[fj].centerline[0]
                d = np.linalg.norm(pi - pj)
                if d < min_dist:
                    min_dist = d
                    best_pair = (fi, fj)
        
        if best_pair and min_dist < max_dist:
            fi, fj = best_pair
            p1 = net.fibers[fi].centerline[0]
            p2 = net.fibers[fj].centerline[0]
            bridge = Fiber.straight(p1, p2, radius=net.fibers[fi].radius,
                                   material=net.fibers[fi].material, fiber_id=new_fid)
            net.add_fiber(bridge)
            net.add_crosslink(Crosslink(
                fiber_i=fi, fiber_j=new_fid, param_i=0.0, param_j=0.0,
                position=p1,
            ))
            net.add_crosslink(Crosslink(
                fiber_i=fj, fiber_j=new_fid, param_i=0.0, param_j=1.0,
                position=p2,
            ))
            large_fibers.append(new_fid)
            new_fid += 1
    
    return net



# ============================================================================
# TPMS field functions
# ============================================================================

def _tpms_field(kind: str, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Compute TPMS level-set field."""
    k = kind.lower()
    
    if k in ('gyroid', 'g'):
        return (np.sin(X) * np.cos(Y) +
                np.sin(Y) * np.cos(Z) +
                np.sin(Z) * np.cos(X))
    elif k in ('diamond', 'd'):
        return (np.sin(X) * np.sin(Y) * np.sin(Z) +
                np.sin(X) * np.cos(Y) * np.cos(Z) +
                np.cos(X) * np.sin(Y) * np.cos(Z) +
                np.cos(X) * np.cos(Y) * np.sin(Z))
    elif k in ('primitive', 'p'):
        return np.cos(X) + np.cos(Y) + np.cos(Z)
    elif k in ('iwp', 'i-wp'):
        return (2.0 * (np.cos(X) * np.cos(Y) +
                       np.cos(Y) * np.cos(Z) +
                       np.cos(Z) * np.cos(X)) -
                (np.cos(2*X) + np.cos(2*Y) + np.cos(2*Z)))
    elif k == 'neovius':
        return (3.0 * (np.cos(X) + np.cos(Y) + np.cos(Z)) +
                4.0 * np.cos(X) * np.cos(Y) * np.cos(Z))
    elif k == 'lidinoid':
        return (np.cos(2*X) * np.sin(Y) * np.cos(Z) +
                np.cos(2*Y) * np.sin(Z) * np.cos(X) +
                np.cos(2*Z) * np.sin(X) * np.cos(Y) -
                np.cos(2*X) * np.cos(2*Y) -
                np.cos(2*Y) * np.cos(2*Z) -
                np.cos(2*Z) * np.cos(2*X) + 0.3)
    else:
        raise ValueError(f"Unknown TPMS type: {kind}")


# ============================================================================
# Marching cubes edge extraction
# ============================================================================

def _extract_tpms_edges(
    kind: str,
    box_size: Tuple[float, float, float],
    num_periods: Tuple[int, int, int],
    resolution: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract edges from TPMS using marching cubes.
    
    Returns (vertices, edges) arrays.
    """
    Lx, Ly, Lz = box_size
    nx, ny, nz = num_periods
    
    res_x = max(resolution * nx, 8)
    res_y = max(resolution * ny, 8)
    res_z = max(resolution * nz, 8)
    
    x = np.linspace(0, 2 * np.pi * nx, res_x)
    y = np.linspace(0, 2 * np.pi * ny, res_y)
    z = np.linspace(0, 2 * np.pi * nz, res_z)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    field = _tpms_field(kind, X, Y, Z)
    
    # Try skimage marching cubes first
    try:
        from skimage.measure import marching_cubes
        verts, faces, _, _ = marching_cubes(
            field, level=0.0,
            spacing=(Lx / res_x, Ly / res_y, Lz / res_z)
        )
        
        # Extract unique edges from faces
        edge_set = set()
        for face in faces:
            for i in range(len(face)):
                e = tuple(sorted([int(face[i]), int(face[(i + 1) % len(face)])]))
                edge_set.add(e)
        
        edges = np.array(list(edge_set), dtype=int)
        return verts, edges
    
    except ImportError:
        pass
    
    # Fallback: simple zero-crossing detection
    dx = Lx / res_x
    dy = Ly / res_y
    dz = Lz / res_z
    
    verts = []
    edges = []
    vert_map = {}
    
    def add_vertex(pos):
        key = tuple(np.round(pos, 6))
        if key not in vert_map:
            vert_map[key] = len(verts)
            verts.append(pos)
        return vert_map[key]
    
    # Find zero-crossings on grid edges
    for i in range(res_x - 1):
        for j in range(res_y - 1):
            for k in range(res_z - 1):
                v = field[i, j, k]
                
                # Check x-edge
                if i + 1 < res_x:
                    v2 = field[i + 1, j, k]
                    if v * v2 < 0:
                        t = -v / (v2 - v)
                        p1 = np.array([i * dx + t * dx, j * dy, k * dz])
                        vid1 = add_vertex(p1)
                        
                        # Connect to nearby vertices
                        for vi, vi2 in [(i, j), (i, k)]:
                            if vi + 1 < res_x if vi == i else vi + 1 < res_y:
                                pass  # simplified
        
        # This is getting complex. Let me use a simpler approach:
        # Just extract edges from the field gradient
    
    # Simple approach: find all pairs of adjacent cells with opposite signs
    edge_list = []
    for i in range(res_x - 1):
        for j in range(res_y - 1):
            for k in range(res_z - 1):
                corners = [
                    (i, j, k), (i+1, j, k), (i+1, j+1, k), (i, j+1, k),
                    (i, j, k+1), (i+1, j, k+1), (i+1, j+1, k+1), (i, j+1, k+1),
                ]
                vals = [field[c[0], c[1], c[2]] for c in corners]
                
                # Check each edge of the cube
                cube_edges = [(0,1), (1,2), (2,3), (3,0),
                             (4,5), (5,6), (6,7), (7,4),
                             (0,4), (1,5), (2,6), (3,7)]
                
                cell_verts = []
                for e1, e2 in cube_edges:
                    v1, v2 = vals[e1], vals[e2]
                    if v1 * v2 < 0:
                        # Zero crossing
                        t = -v1 / (v2 - v1)
                        c1, c2 = corners[e1], corners[e2]
                        pos = np.array([
                            c1[0] * dx + t * (c2[0] - c1[0]) * dx,
                            c1[1] * dy + t * (c2[1] - c1[1]) * dy,
                            c1[2] * dz + t * (c2[2] - c1[2]) * dz,
                        ])
                        vid = add_vertex(pos)
                        cell_verts.append(vid)
                
                # Connect vertices in this cell
                for a in range(len(cell_verts)):
                    for b in range(a + 1, min(a + 3, len(cell_verts))):
                        edge_list.append((cell_verts[a], cell_verts[b]))
    
    if len(verts) == 0:
        return np.zeros((0, 3)), np.zeros((0, 2), dtype=int)
    
    return np.array(verts), np.array(edge_list, dtype=int)


# ============================================================================
# TPMS generators using FiberGraph
# ============================================================================

def tpms_sheet(
    kind: str = "gyroid",
    box_size: Tuple[float, float, float] = (10.0, 10.0, 10.0),
    resolution: int = 20,
    num_periods: Tuple[int, int, int] = (1, 1, 1),
    radius: float = 0.05,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> FiberNetwork:
    """Generate fiber network from TPMS sheet (level-set surface).
    
    Uses FiberGraph to ensure proper connectivity at shared vertices.
    """
    mat = material or Material(name=f"tpms_{kind}")
    
    if seed is not None:
        np.random.seed(seed)
    
    verts, edges = _extract_tpms_edges(kind, box_size, num_periods, resolution)
    
    if len(edges) == 0:
        return FiberNetwork(dimension=3, box_size=np.array(box_size),
                          metadata={"generator": "tpms_sheet", "kind": kind})
    
    # Build FiberGraph
    g = FiberGraph(dimension=3, tolerance=0.01)
    
    for e in edges:
        p1 = verts[e[0]]
        p2 = verts[e[1]]
        g.add_edge_by_pos(p1, p2, radius=radius, material=mat)
    
    return g.to_network(
        material=mat,
        box_size=np.array(box_size),
        metadata={"generator": "tpms_sheet", "kind": kind, "resolution": resolution},
    )


def tpms_lattice(
    kind: str = "primitive",
    box_size: Tuple[float, float, float] = (10.0, 10.0, 10.0),
    resolution: int = 20,
    num_periods: Tuple[int, int, int] = (1, 1, 1),
    radius: float = 0.1,
    threshold: float = 0.5,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> FiberNetwork:
    """Generate fiber network from TPMS lattice (skeleton).
    
    Extracts the medial axis of the TPMS solid phase and creates
    a fiber network along the skeleton.
    """
    mat = material or Material(name=f"tpms_lattice_{kind}")
    
    if seed is not None:
        np.random.seed(seed)
    
    Lx, Ly, Lz = box_size
    nx, ny, nz = num_periods
    
    res_x = max(resolution * nx, 8)
    res_y = max(resolution * ny, 8)
    res_z = max(resolution * nz, 8)
    
    x = np.linspace(0, 2 * np.pi * nx, res_x)
    y = np.linspace(0, 2 * np.pi * ny, res_y)
    z = np.linspace(0, 2 * np.pi * nz, res_z)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    field = _tpms_field(kind, X, Y, Z)
    
    # Threshold to create solid phase
    binary = field < threshold
    
    # Extract skeleton using distance transform
    try:
        from scipy.ndimage import distance_transform_edt, label
        from scipy.spatial import cKDTree
        
        # Distance transform of solid phase
        dist = distance_transform_edt(binary)
        
        # Find skeleton points (local maxima of distance)
        skeleton_mask = np.zeros_like(binary)
        for i in range(1, res_x - 1):
            for j in range(1, res_y - 1):
                for k in range(1, res_z - 1):
                    if binary[i, j, k]:
                        local = dist[i-1:i+2, j-1:j+2, k-1:k+2]
                        if dist[i, j, k] >= np.max(local) * 0.9 and dist[i, j, k] > 1.0:
                            skeleton_mask[i, j, k] = True
        
        # Label connected components of skeleton
        labeled, n_comp = label(skeleton_mask)
        
        if n_comp == 0:
            return FiberNetwork(dimension=3, box_size=np.array(box_size),
                              metadata={"generator": "tpms_lattice", "kind": kind})
        
        # Build graph from skeleton points
        g = FiberGraph(dimension=3, tolerance=0.1)
        
        dx = Lx / res_x
        dy = Ly / res_y
        dz = Lz / res_z
        
        # Get skeleton point positions
        skel_coords = np.argwhere(skeleton_mask)
        
        if len(skel_coords) < 2:
            return FiberNetwork(dimension=3, box_size=np.array(box_size),
                              metadata={"generator": "tpms_lattice", "kind": kind})
        
        # Use KDTree to find nearby points and create edges
        positions = skel_coords * np.array([dx, dy, dz])
        tree = cKDTree(positions)
        
        max_dist = np.sqrt(dx**2 + dy**2 + dz**2) * 2.0
        pairs = tree.query_pairs(max_dist)
        
        for i, j in pairs:
            g.add_edge_by_pos(positions[i], positions[j], radius=radius, material=mat)
        
        net = g.to_network(
            material=mat,
            box_size=np.array(box_size),
            metadata={"generator": "tpms_lattice", "kind": kind},
        )
        
        # Merge small components into the largest one
        net = _merge_components(net, max_dist=max_dist * 3.0)
        return net
    
    except ImportError:
        # Fallback: use marching cubes edges
        return tpms_sheet(kind, box_size, resolution, num_periods, radius, material, seed)


def tpms_gradient(
    kind: str = "gyroid",
    box_size: Tuple[float, float, float] = (30.0, 10.0, 10.0),
    gradient_axis: int = 0,
    gradient_range: Tuple[float, float] = (0.1, 0.5),
    num_segments: int = 4,
    resolution: int = 8,
    radius: float = 0.1,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> FiberNetwork:
    """Generate TPMS with gradient density along one axis.
    
    Divides the domain into segments along gradient_axis,
    each with different strut thickness.
    """
    mat = material or Material(name=f"tpms_gradient_{kind}")
    
    all_nets = []
    
    seg_length = box_size[gradient_axis] / num_segments
    
    for seg_idx in range(num_segments):
        # Interpolate thickness
        t = seg_idx / max(num_segments - 1, 1)
        strut_r = radius * (gradient_range[0] + t * (gradient_range[1] - gradient_range[0]))
        
        # Create segment
        seg_box = list(box_size)
        seg_box[gradient_axis] = seg_length
        
        seg_net = tpms_sheet(
            kind=kind,
            box_size=tuple(seg_box),
            resolution=resolution,
            num_periods=(1, 1, 1),
            radius=strut_r,
            material=mat,
            seed=seed + seg_idx if seed is not None else None,
        )
        
        # Offset segment position
        offset = np.zeros(3)
        offset[gradient_axis] = seg_idx * seg_length
        
        # Translate all fibers
        for fiber in seg_net.fibers:
            fiber.centerline += offset
        
        all_nets.append(seg_net)
    
    # Merge all segments
    from fibernet.core.transform import merge
    merged = merge(all_nets)
    merged.box_size = np.array(box_size)
    merged.metadata = {
        "generator": "tpms_gradient",
        "kind": kind,
        "gradient_axis": gradient_axis,
    }
    
    return merged
