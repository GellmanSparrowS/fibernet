"""
Unified fiber network generators.

These generators consolidate many specific generators into powerful,
parameterized ones. Each supports multiple modes via a `topology` or
`mode` parameter, with full control over geometry, tiling, perturbation,
and more.

Design Philosophy:
- One generator per concept family (not per specific structure)
- Parameters control the continuous deformation space
- All outputs are FiberNetwork objects ready for simulation
"""

import numpy as np
from typing import Optional, Tuple, List, Union
from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork
from fibernet.core.material import Material


# ============================================================================
# Helper: Build FiberNetwork from edges + nodes
# ============================================================================

def _edges_to_network(
    nodes: dict,
    edges: list,
    radius: float = 0.1,
    material: Optional[Material] = None,
    dimension: int = 2,
) -> FiberNetwork:
    """Convert node/edge graph to FiberNetwork.
    
    Each edge becomes a Fiber. Nodes where edges meet become crosslinks.
    """
    from fibernet.core.network import Crosslink
    
    fibers = []
    for (u, v) in edges:
        p1 = np.array(nodes[u], dtype=float)
        p2 = np.array(nodes[v], dtype=float)
        # Ensure 3D coordinates
        if len(p1) == 2:
            p1 = np.append(p1, 0.0)
        if len(p2) == 2:
            p2 = np.append(p2, 0.0)
        centerline = np.array([p1, p2])
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < 1e-10:
            continue
        direction = direction / length
        fiber = Fiber(
            centerline=centerline,
            radius=radius,
            material=material,
        )
        fibers.append(fiber)
    
    net = FiberNetwork(fibers=fibers, dimension=dimension)
    
    # Detect crosslinks at shared nodes
    node_to_fibers = {}
    for i, (u, v) in enumerate(edges):
        for node in [u, v]:
            if node not in node_to_fibers:
                node_to_fibers[node] = []
            node_to_fibers[node].append(i)
    
    crosslinks = []
    for node, fiber_indices in node_to_fibers.items():
        if len(fiber_indices) < 2:
            continue
        pos = np.array(nodes[node], dtype=float)
        if len(pos) == 2:
            pos = np.append(pos, 0.0)
        # create crosslinks for all pairs at this junction
        seen = set()
        for ii in range(len(fiber_indices)):
            for jj in range(ii + 1, len(fiber_indices)):
                pair = (min(fiber_indices[ii], fiber_indices[jj]),
                        max(fiber_indices[ii], fiber_indices[jj]))
                if pair in seen:
                    continue
                seen.add(pair)
                cl = Crosslink(
                    position=pos,
                    fiber_i=pair[0], fiber_j=pair[1],
                    param_i=0.5, param_j=0.5,
                )
                crosslinks.append(cl)
    
    net.crosslinks = crosslinks
    return net


def _tile_unit_cell(
    cell_nodes: dict,
    cell_edges: list,
    grid_size: Tuple[int, int],
    cell_width: float,
    cell_height: float,
    perturbation: float = 0.0,
    rotation: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[dict, list]:
    """Tile a 2D unit cell into a grid.
    
    Parameters
    ----------
    cell_nodes : dict
        Node positions in unit cell (relative to origin).
    cell_edges : list
        Edge list (node name pairs).
    grid_size : (nx, ny)
        Number of tiles.
    cell_width, cell_height : float
        Unit cell dimensions.
    perturbation : float
        Random displacement magnitude (fraction of cell_size).
    rotation : float
        Rotation angle in radians.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    nodes, edges
        Tiled graph.
    """
    rng = np.random.RandomState(seed)
    nx_total, ny_total = grid_size
    
    all_nodes = {}
    all_edges = []
    
    for ix in range(nx_total):
        for iy in range(ny_total):
            prefix = f"t{ix}_{iy}_"
            for name, pos in cell_nodes.items():
                new_name = prefix + str(name)
                new_pos = (
                    pos[0] + ix * cell_width,
                    pos[1] + iy * cell_height,
                )
                if perturbation > 0:
                    new_pos = (
                        new_pos[0] + rng.uniform(-perturbation, perturbation) * cell_width,
                        new_pos[1] + rng.uniform(-perturbation, perturbation) * cell_height,
                    )
                all_nodes[new_name] = new_pos
            for (u, v) in cell_edges:
                all_edges.append((prefix + str(u), prefix + str(v)))
    
    # Connect adjacent tiles (shared boundary nodes)
    for ix in range(nx_total):
        for iy in range(ny_total):
            # Right neighbor
            if ix < nx_total - 1:
                _connect_tiles(all_nodes, all_edges, cell_nodes, cell_edges,
                              ix, iy, ix+1, iy, cell_width, cell_height, 'x')
            # Top neighbor
            if iy < ny_total - 1:
                _connect_tiles(all_nodes, all_edges, cell_nodes, cell_edges,
                              ix, iy, ix, iy+1, cell_width, cell_height, 'y')
    
    # Apply rotation
    if abs(rotation) > 1e-10:
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        for name in all_nodes:
            x, y = all_nodes[name]
            all_nodes[name] = (x * cos_r - y * sin_r, x * sin_r + y * cos_r)
    
    return all_nodes, all_edges


def _connect_tiles(all_nodes, all_edges, cell_nodes, cell_edges,
                   ix1, iy1, ix2, iy2, cw, ch, direction):
    """Merge boundary nodes between adjacent tiles by position.
    
    Nodes from tile (ix2, iy2) that sit at the same position as nodes
    from tile (ix1, iy1) get merged: edges referencing the tile-2 node
    are rewritten to point to the tile-1 node instead.
    """
    tol = 1e-6
    prefix1 = f"t{ix1}_{iy1}_"
    prefix2 = f"t{ix2}_{iy2}_"
    
    # Build position -> name map for tile 1 boundary nodes
    pos_to_name1 = {}
    for name, pos in all_nodes.items():
        if name.startswith(prefix1):
            key = (round(pos[0] / tol) * tol, round(pos[1] / tol) * tol)
            pos_to_name1[key] = name
    
    # For each tile 2 node, check if it coincides with a tile 1 node
    merge_map = {}  # old_name -> new_name
    for name2, pos2 in list(all_nodes.items()):
        if not name2.startswith(prefix2):
            continue
        key = (round(pos2[0] / tol) * tol, round(pos2[1] / tol) * tol)
        if key in pos_to_name1:
            merge_map[name2] = pos_to_name1[key]
    
    # Rewrite edges that reference merged nodes
    new_edges = []
    for (u, v) in all_edges:
        nu = merge_map.get(u, u)
        nv = merge_map.get(v, v)
        if nu != nv:
            new_edges.append((nu, nv))
    all_edges.clear()
    all_edges.extend(new_edges)


# ============================================================================
# Unit Cell Definitions
# ============================================================================

def _unit_cell_square(cell_size: float) -> Tuple[dict, list, float, float]:
    """Square unit cell."""
    s = cell_size
    nodes = {0: (0, 0), 1: (s, 0), 2: (s, s), 3: (0, s)}
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    return nodes, edges, s, s


def _unit_cell_honeycomb(cell_size: float) -> Tuple[dict, list, float, float]:
    """Honeycomb unit cell."""
    s = cell_size
    h = s * np.sqrt(3) / 2
    nodes = {
        0: (0, 0), 1: (s/2, 0), 2: (s, h),
        3: (s/2, 2*h), 4: (0, 2*h), 5: (-s/2, h),
    }
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    return nodes, edges, s, 2*h


def _unit_cell_triangular(cell_size: float) -> Tuple[dict, list, float, float]:
    """Triangular (equilateral) unit cell."""
    s = cell_size
    h = s * np.sqrt(3) / 2
    nodes = {0: (0, 0), 1: (s, 0), 2: (s/2, h)}
    edges = [(0, 1), (1, 2), (2, 0)]
    return nodes, edges, s, h


def _unit_cell_kagome(cell_size: float) -> Tuple[dict, list, float, float]:
    """Kagome unit cell."""
    s = cell_size
    h = s * np.sqrt(3) / 2
    nodes = {
        0: (0, 0), 1: (s, 0), 2: (s*1.5, h), 3: (s, 2*h),
        4: (0, 2*h), 5: (-s*0.5, h),
        6: (s/2, 0), 7: (s*1.25, h/2), 8: (s*1.25, 1.5*h),
        9: (s/2, 2*h), 10: (-s*0.25, 1.5*h), 11: (-s*0.25, h/2),
    }
    edges = [
        (0, 6), (6, 1), (1, 7), (7, 2), (2, 8), (8, 3),
        (3, 9), (9, 4), (4, 10), (10, 5), (5, 11), (11, 0),
    ]
    return nodes, edges, s, 2*h


def _unit_cell_reentrant(cell_size: float, angle_deg: float = 150.0) -> Tuple[dict, list, float, float]:
    """Reentrant honeycomb unit cell."""
    s = cell_size
    angle = np.radians(angle_deg)
    dx = s * np.cos(angle)
    dy = s * np.sin(angle)
    
    w = 2 * s + 2 * abs(dx)
    h = 2 * s
    
    nodes = {
        0: (0, s), 1: (s, 0), 2: (s + abs(dx), s - dy),
        3: (w, s), 4: (s + abs(dx), s + dy),
        5: (s, 2*s), 6: (0, s),  # connects back
    }
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    return nodes, edges, w, h


def _unit_cell_star(cell_size: float) -> Tuple[dict, list, float, float]:
    """Star honeycomb unit cell."""
    s = cell_size
    nodes = {
        0: (0, s), 1: (s/2, s/2), 2: (s, 0), 3: (s*1.5, s/2),
        4: (2*s, s), 5: (s*1.5, s*1.5), 6: (s, 2*s), 7: (s/2, s*1.5),
    }
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (4, 5), (5, 6), (6, 7), (7, 0),
    ]
    return nodes, edges, 2*s, 2*s


def _unit_cell_arrowhead(cell_size: float) -> Tuple[dict, list, float, float]:
    """Arrowhead auxetic unit cell — rhombus that tiles by translation."""
    s = cell_size
    nodes = {
        0: (0, s/2), 1: (s/2, 0), 2: (s, s/2), 3: (s/2, s),
    }
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    return nodes, edges, s, s


def _unit_cell_chiral(cell_size: float) -> Tuple[dict, list, float, float]:
    """Chiral honeycomb unit cell — central hub with tangent ligaments to adjacent cells."""
    s = cell_size
    r = s * 0.25
    cx, cy = s/2, s/2
    n_ring = 6
    angles = np.linspace(0, 2*np.pi, n_ring + 1)[:-1]
    
    # Center hub node
    nodes = {0: (cx, cy)}
    # Ring nodes around center
    for i, a in enumerate(angles):
        nodes[i+1] = (cx + r*np.cos(a), cy + r*np.sin(a))
    
    edges = []
    # Ring ring connections
    for i in range(n_ring):
        edges.append((i+1, (i+1) % n_ring + 1))
    # Hub to ring (radial struts)
    for i in range(n_ring):
        edges.append((0, i+1))
    # Tangent ligaments extending to adjacent cells (boundary nodes)
    # These connect ring nodes to the boundary for inter-cell connectivity
    return nodes, edges, s, s


def _unit_cell_missing_rib(cell_size: float) -> Tuple[dict, list, float, float]:
    """Missing-rib auxetic unit cell."""
    s = cell_size
    nodes = {
        0: (0, 0), 1: (s, 0), 2: (s, s), 3: (0, s),
        4: (s/2, s/2),
    }
    # Square with diagonal missing ribs (alternating)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (2, 4)]
    return nodes, edges, s, s



# ============================================================================
# Direct Construction Builders for Lattice 2D
# ============================================================================

def _build_square_direct(nx: int, ny: int, cell_size: float) -> Tuple[dict, list, float, float]:
    """Square lattice: grid of nodes with horizontal + vertical edges."""
    nodes = {}
    edges = []
    s = cell_size
    
    for i in range(nx + 1):
        for j in range(ny + 1):
            nid = i * (ny + 1) + j
            nodes[nid] = (i * s, j * s)
    
    for i in range(nx + 1):
        for j in range(ny + 1):
            nid = i * (ny + 1) + j
            if i < nx:
                right = (i + 1) * (ny + 1) + j
                edges.append((nid, right))
            if j < ny:
                up = i * (ny + 1) + (j + 1)
                edges.append((nid, up))
    
    return nodes, edges, nx * s, ny * s


def _build_triangular_direct(nx: int, ny: int, cell_size: float) -> Tuple[dict, list, float, float]:
    """Triangular lattice: each node connects right, up, and diagonal."""
    nodes = {}
    edges = []
    s = cell_size
    h = s * np.sqrt(3) / 2
    
    for i in range(nx + 1):
        for j in range(ny + 1):
            nid = i * (ny + 1) + j
            x_offset = (j % 2) * (s / 2)
            x = i * s + x_offset
            y = j * h
            nodes[nid] = (x, y)
    
    for i in range(nx + 1):
        for j in range(ny + 1):
            nid = i * (ny + 1) + j
            if i < nx:
                right = (i + 1) * (ny + 1) + j
                edges.append((nid, right))
            if j < ny:
                up = i * (ny + 1) + (j + 1)
                edges.append((nid, up))
            if i < nx and j < ny:
                diag = (i + 1) * (ny + 1) + (j + 1)
                edges.append((nid, diag))
    
    xs = [p[0] for p in nodes.values()]
    ys = [p[1] for p in nodes.values()]
    return nodes, edges, max(xs) - min(xs), max(ys) - min(ys)


def _build_honeycomb_direct(nx: int, ny: int, cell_size: float) -> Tuple[dict, list, float, float]:
    """Honeycomb lattice: two-sublattice approach with staggered hexagons."""
    nodes = {}
    edges = []
    s = cell_size
    h = s * np.sqrt(3)
    
    node_id = 0
    node_map = {}
    
    for i in range(nx + 1):
        for j in range(ny + 1):
            x_a = i * 1.5 * s
            y_a = j * h
            node_map[(i, j, 'A')] = node_id
            nodes[node_id] = (x_a, y_a)
            node_id += 1
            
            x_b = x_a + s * 0.5
            y_b = y_a + h * 0.5
            node_map[(i, j, 'B')] = node_id
            nodes[node_id] = (x_b, y_b)
            node_id += 1
    
    for i in range(nx + 1):
        for j in range(ny + 1):
            a = node_map.get((i, j, 'A'))
            b = node_map.get((i, j, 'B'))
            if a is not None and b is not None:
                edges.append((a, b))
            
            if i < nx:
                b_next = node_map.get((i + 1, j, 'A'))
                if b is not None and b_next is not None:
                    edges.append((b, b_next))
            
            if j < ny:
                b_up = node_map.get((i, j + 1, 'A'))
                if b is not None and b_up is not None:
                    edges.append((b, b_up))
    
    xs = [p[0] for p in nodes.values()]
    ys = [p[1] for p in nodes.values()]
    return nodes, edges, max(xs) - min(xs), max(ys) - min(ys)

# ============================================================================
# Unified 2D Lattice Generator
# ============================================================================

def lattice_2d(
    topology: str = "honeycomb",
    cell_size: float = 10.0,
    grid_size: Tuple[int, int] = (6, 6),
    perturbation: float = 0.0,
    rotation: float = 0.0,
    radius: float = 0.1,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> FiberNetwork:
    """Unified 2D lattice generator.
    
    Generates periodic 2D lattices using direct coordinate construction
    (no tiling) for guaranteed connectivity.
    
    Parameters
    ----------
    topology : str
        Unit cell type: 'square', 'honeycomb', 'triangular', 'kagome'.
    cell_size : float
        Unit cell characteristic length.
    grid_size : (int, int)
        Grid dimensions (nx, ny). Meaning depends on topology.
    perturbation : float
        Random node displacement (fraction of cell_size). 0 = perfect lattice.
    rotation : float
        Global rotation angle in radians.
    radius : float
        Fiber radius.
    material : Material, optional
        Fiber material.
    seed : int, optional
        Random seed for perturbation.
    
    Returns
    -------
    FiberNetwork
    
    Examples
    --------
    >>> net = fn.create("lattice_2d", topology="honeycomb", cell_size=8.0)
    >>> net = fn.create("lattice_2d", topology="kagome", perturbation=0.2)
    """
    topology = topology.lower()
    nx, ny = grid_size
    
    # Direct construction functions
    direct_builders = {
        'square': _build_square_direct,
        'triangular': _build_triangular_direct,
        'honeycomb': _build_honeycomb_direct,
    }
    
    if topology in direct_builders:
        nodes, edges, cw, ch = direct_builders[topology](nx, ny, cell_size)
    elif topology == 'kagome':
        # Kagome works with original tiling approach
        cell_nodes, cell_edges, cw, ch = _unit_cell_kagome(cell_size)
        nodes, edges = _tile_unit_cell(
            cell_nodes, cell_edges, grid_size, cw, ch,
            perturbation=0.0, rotation=0.0, seed=None,
        )
    else:
        raise ValueError(f"Unknown topology '{topology}'. Available: square, triangular, honeycomb, kagome")
    
    # Apply perturbation if requested
    if perturbation > 0 and topology != 'kagome':
        rng = np.random.RandomState(seed)
        for nid in nodes:
            x, y = nodes[nid]
            dx = rng.uniform(-perturbation, perturbation) * cell_size
            dy = rng.uniform(-perturbation, perturbation) * cell_size
            nodes[nid] = (x + dx, y + dy)
    
    # Apply rotation if requested
    if abs(rotation) > 1e-10:
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        for nid in nodes:
            x, y = nodes[nid]
            nodes[nid] = (x * cos_r - y * sin_r, x * sin_r + y * cos_r)
    
    return _edges_to_network(nodes, edges, radius=radius, material=material, dimension=2)



def _build_chiral_direct(nx: int, ny: int, cell_size: float) -> Tuple[dict, list, float, float]:
    """
    Chiral honeycomb: rings on hexagonal grid connected by tangent ligaments.
    Each ring has 6 nodes, connected to 6 neighboring rings.
    """
    nodes = {}
    edges = []
    s = cell_size
    r = s * 0.2  # ring radius
    n_ring = 6
    
    # Place ring centers on hexagonal grid
    ring_centers = []
    for i in range(nx + 1):
        for j in range(ny + 1):
            x_offset = (j % 2) * (s * 0.75)
            cx = i * s * 1.5 + x_offset
            cy = j * s * np.sqrt(3) / 2
            ring_centers.append((cx, cy))
    
    # Create ring nodes
    node_id = 0
    ring_node_ids = []  # list of lists, one per ring
    for cx, cy in ring_centers:
        ring_ids = []
        for k in range(n_ring):
            angle = k * (2 * np.pi / n_ring)
            x = cx + r * np.cos(angle)
            y = cy + r * np.sin(angle)
            nodes[node_id] = (x, y)
            ring_ids.append(node_id)
            node_id += 1
        ring_node_ids.append(ring_ids)
    
    # Connect nodes within each ring
    for ring_ids in ring_node_ids:
        for k in range(n_ring):
            edges.append((ring_ids[k], ring_ids[(k + 1) % n_ring]))
    
    # Connect adjacent rings with tangent ligaments
    # Each ring connects to its 6 nearest neighbors
    for i, (cx1, cy1) in enumerate(ring_centers):
        for j, (cx2, cy2) in enumerate(ring_centers):
            if i >= j:
                continue
            dist = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
            if dist < s * 1.6:  # nearest neighbors
                # Connect closest ring nodes
                min_dist = float('inf')
                best_pair = None
                for n1 in ring_node_ids[i]:
                    for n2 in ring_node_ids[j]:
                        p1 = nodes[n1]
                        p2 = nodes[n2]
                        d = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                        if d < min_dist:
                            min_dist = d
                            best_pair = (n1, n2)
                if best_pair and min_dist < s * 0.8:
                    edges.append(best_pair)
    
    xs = [p[0] for p in nodes.values()]
    ys = [p[1] for p in nodes.values()]
    return nodes, edges, max(xs) - min(xs), max(ys) - min(ys)


# ============================================================================
# Unified 2D Metamaterial Generator
# ============================================================================

def metamaterial_2d(
    mode: str = "reentrant",
    cell_size: float = 10.0,
    grid_size: Tuple[int, int] = (5, 5),
    angle: float = 150.0,
    perturbation: float = 0.0,
    rotation: float = 0.0,
    radius: float = 0.2,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> FiberNetwork:
    """Unified 2D metamaterial generator.
    
    Generates auxetic and other metamaterial structures from
    parametric unit cells.
    
    Parameters
    ----------
    mode : str
        Unit cell mode: 'reentrant', 'star', 'arrowhead', 'chiral', 'missing_rib'.
    cell_size : float
        Unit cell size.
    grid_size : (int, int)
        Tiling dimensions.
    angle : float
        Key geometric angle (degrees). Meaning depends on mode:
        - reentrant: reentrant angle (120-170°)
        - star: inner angle
    perturbation : float
        Random displacement (fraction of cell_size).
    rotation : float
        Global rotation (radians).
    radius : float
        Fiber radius.
    material : Material, optional
        Fiber material.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    FiberNetwork
    
    Examples
    --------
    >>> net = fn.create("metamaterial_2d", mode="reentrant", angle=135)
    >>> net = fn.create("metamaterial_2d", mode="chiral", cell_size=8.0)
    """
    mode = mode.lower()
    
    if mode == 'reentrant':
        cell_nodes, cell_edges, cw, ch = _unit_cell_reentrant(cell_size, angle)
    elif mode == 'star':
        cell_nodes, cell_edges, cw, ch = _unit_cell_star(cell_size)
    elif mode == 'arrowhead':
        cell_nodes, cell_edges, cw, ch = _unit_cell_arrowhead(cell_size)
    elif mode == 'chiral':
        # Use direct construction for guaranteed connectivity
        nodes, edges, cw, ch = _build_chiral_direct(grid_size[0], grid_size[1], cell_size)
        return _edges_to_network(nodes, edges, radius=radius, material=material, dimension=2)
    elif mode == 'missing_rib':
        cell_nodes, cell_edges, cw, ch = _unit_cell_missing_rib(cell_size)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Available: reentrant, star, arrowhead, chiral, missing_rib")
    
    nodes, edges = _tile_unit_cell(
        cell_nodes, cell_edges, grid_size, cw, ch,
        perturbation=perturbation, rotation=rotation, seed=seed,
    )
    
    return _edges_to_network(nodes, edges, radius=radius, material=material, dimension=2)


# ============================================================================
# Curved Random 2D Generator
# ============================================================================

def curved_random_2d(
    num_fibers: int = 100,
    fiber_length: float = 10.0,
    box_size: Optional[float] = None,
    curvature_type: str = "sinusoidal",
    curvature_amplitude: float = 2.0,
    curvature_frequency: float = 1.0,
    num_points: int = 20,
    angle_std: float = 1.57,
    mean_angle: float = 0.0,
    radius: float = 0.1,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> FiberNetwork:
    """Generate 2D random network with curved fibers.
    
    Parameters
    ----------
    num_fibers : int
        Number of fibers.
    fiber_length : float
        Nominal fiber length (end-to-end distance).
    box_size : float, optional
        Domain size. Auto-computed from percolation if None.
    curvature_type : str
        'sinusoidal', 'bezier', 'arc', 'random_walk'.
    curvature_amplitude : float
        Amplitude of curvature (perpendicular to fiber axis).
    curvature_frequency : float
        Number of oscillations (for sinusoidal).
    num_points : int
        Number of points per fiber centerline.
    angle_std : float
        Orientation standard deviation (radians). 1.57 = isotropic.
    mean_angle : float
        Mean fiber orientation.
    radius : float
        Fiber radius.
    material : Material, optional
        Fiber material.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    FiberNetwork
    
    Examples
    --------
    >>> net = fn.create("curved_random_2d", curvature_type="sinusoidal")
    >>> net = fn.create("curved_random_2d", curvature_type="bezier",
    ...                 curvature_amplitude=5.0, angle_std=0.3)
    """
    rng = np.random.RandomState(seed)
    
    # Auto-compute box size for percolation
    if box_size is None:
        rho_c = 5.71 / (fiber_length ** 2)
        density = max(num_fibers * fiber_length / 50.0, 2.0 * rho_c * 50.0)
        box_size = np.sqrt(num_fibers * fiber_length / density)
        box_size = max(box_size, 2.0 * fiber_length)
    
    fibers = []
    for _ in range(num_fibers):
        # Random center
        cx = rng.uniform(0, box_size)
        cy = rng.uniform(0, box_size)
        
        # Random orientation
        if angle_std < 0.01:
            angle = mean_angle
        else:
            angle = mean_angle + rng.normal(0, angle_std)
        
        # Generate curved centerline
        t = np.linspace(0, 1, num_points)
        
        # Base direction
        dx = np.cos(angle)
        dy = np.sin(angle)
        nx_dir = -np.sin(angle)  # normal
        ny_dir = np.cos(angle)
        
        half_len = fiber_length / 2
        
        if curvature_type == 'sinusoidal':
            # Sinusoidal deviation along fiber
            s = (t - 0.5) * fiber_length
            perp = curvature_amplitude * np.sin(2 * np.pi * curvature_frequency * t)
            px = cx + s * dx + perp * nx_dir
            py = cy + s * dy + perp * ny_dir
            
        elif curvature_type == 'bezier':
            # Quadratic Bezier with random control point
            ctrl_offset = rng.uniform(-curvature_amplitude, curvature_amplitude)
            p0 = np.array([cx - half_len * dx, cy - half_len * dy])
            p1 = np.array([cx + ctrl_offset * nx_dir, cy + ctrl_offset * ny_dir])
            p2 = np.array([cx + half_len * dx, cy + half_len * dy])
            # Bezier formula: B(t) = (1-t)²P0 + 2t(1-t)P1 + t²P2
            px = (1-t)**2 * p0[0] + 2*t*(1-t)*p1[0] + t**2 * p2[0]
            py = (1-t)**2 * p0[1] + 2*t*(1-t)*p1[1] + t**2 * p2[1]
            
        elif curvature_type == 'arc':
            # Circular arc
            arc_angle = curvature_amplitude / fiber_length  # curvature
            s = (t - 0.5) * fiber_length
            if abs(arc_angle) > 1e-6:
                R = fiber_length / (2 * np.sin(arc_angle))
                theta = s / R
                px = cx + R * (np.sin(theta) * dx + (1 - np.cos(theta)) * nx_dir)
                py = cy + R * (np.sin(theta) * dy + (1 - np.cos(theta)) * ny_dir)
            else:
                px = cx + s * dx
                py = cy + s * dy
            
        elif curvature_type == 'random_walk':
            # Smooth random walk (Brownian bridge)
            s = (t - 0.5) * fiber_length
            noise = rng.normal(0, curvature_amplitude * 0.3, num_points)
            # Cumulative sum with bridge constraint (returns to 0 at endpoints)
            cum_noise = np.cumsum(noise)
            cum_noise = cum_noise - t * cum_noise[-1]  # bridge
            px = cx + s * dx + cum_noise * nx_dir
            py = cy + s * dy + cum_noise * ny_dir
        else:
            raise ValueError(f"Unknown curvature_type: {curvature_type}")
        
        centerline = np.column_stack([px, py, np.zeros(num_points)])
        
        fiber = Fiber(
            centerline=centerline,
            radius=radius,
            material=material,
        )
        fibers.append(fiber)
    
    # Build network
    net = FiberNetwork(fibers=fibers, dimension=2)
    
    # Detect crosslinks (simplified: check minimum distance between fiber segments)
    from fibernet.core.network import Crosslink
    crosslinks = []
    threshold = fiber_length * 0.1  # 10% of fiber length
    
    for i in range(len(fibers)):
        for j in range(i+1, min(i+20, len(fibers))):  # limit checks for speed
            fi = fibers[i].centerline
            fj = fibers[j].centerline
            # Check each segment pair
            for si in range(len(fi)-1):
                for sj in range(len(fj)-1):
                    mid_i = (fi[si] + fi[si+1]) / 2
                    mid_j = (fj[sj] + fj[sj+1]) / 2
                    dist = np.linalg.norm(mid_i - mid_j)
                    if dist < threshold:
                        cl = Crosslink(
                            position=(mid_i + mid_j) / 2,
                            fiber_i=i, fiber_j=j, param_i=0.5, param_j=0.5,
    
                        )
                        crosslinks.append(cl)
                        break
                else:
                    continue
                break
    
    net.crosslinks = crosslinks
    return net


# ============================================================================
# Unified 3D Lattice Generator
# ============================================================================

def lattice_3d(
    topology: str = "octet",
    cell_size: float = 10.0,
    grid_size: Tuple[int, int, int] = (3, 3, 3),
    perturbation: float = 0.0,
    radius: float = 0.1,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> FiberNetwork:
    """Unified 3D lattice generator.
    
    Parameters
    ----------
    topology : str
        'cubic', 'octet', 'diamond', 'gyroid', 'plate'.
    cell_size : float
        Unit cell size.
    grid_size : (int, int, int)
        Number of cells in each dimension.
    perturbation : float
        Random displacement fraction.
    radius : float
        Fiber radius.
    material : Material, optional
        Fiber material.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    FiberNetwork
    """
    from fibernet.gen._fixes import lattice_3d_fixed
    return lattice_3d_fixed(
        topology=topology, cell_size=cell_size, grid_size=grid_size,
        perturbation=perturbation, radius=radius, material=material,
        seed=seed, **kwargs,
    )



def entangled_3d(
    num_fibers: int = 60,
    fiber_length: float = 30.0,
    box_size: Optional[float] = None,
    curvature: float = 0.3,
    num_points: int = 15,
    radius: float = 0.1,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> FiberNetwork:
    """Generate 3D entangled fiber network.
    
    Creates fibers that follow curved paths through 3D space,
    producing natural entanglement (not woven, not lattice).
    
    Parameters
    ----------
    num_fibers : int
        Number of fibers.
    fiber_length : float
        Nominal fiber arc length.
    box_size : float, optional
        Domain size. Auto-computed if None.
    curvature : float
        Curvature intensity (0=straight, 1=highly curved).
    num_points : int
        Points per fiber.
    radius : float
        Fiber radius.
    material : Material, optional
        Fiber material.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    FiberNetwork
    
    Examples
    --------
    >>> net = fn.create("entangled_3d", num_fibers=100, curvature=0.5)
    """
    rng = np.random.RandomState(seed)
    
    if box_size is None:
        # Dense packing: box = 2x fiber length
        box_size = fiber_length
    
    fibers = []
    
    fibers = []
    for _ in range(num_fibers):
        # Random start point
        start = rng.uniform(0, box_size, 3)
        
        # Random initial direction
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0, np.pi)
        direction = np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi),
        ])
        
        # Generate curved path (smooth random walk in 3D)
        points = [start.copy()]
        step_length = fiber_length / num_points
        
        current_pos = start.copy()
        current_dir = direction.copy()
        
        for step in range(num_points - 1):
            # Add random perturbation to direction
            perturbation = rng.normal(0, curvature, 3)
            current_dir = current_dir + perturbation
            current_dir = current_dir / np.linalg.norm(current_dir)
            
            current_pos = current_pos + step_length * current_dir
            # Wrap around box
            current_pos = current_pos % box_size
            points.append(current_pos.copy())
        
        centerline = np.array(points)
        
        fiber = Fiber(
            centerline=centerline,
            radius=radius,
            material=material,
        )
        fibers.append(fiber)
    
    net = FiberNetwork(fibers=fibers, dimension=3)
    
    # Detect crosslinks using spatial hashing for efficiency
    from fibernet.core.network import Crosslink
    crosslinks = []
    threshold = fiber_length * 0.1
    
    # Build spatial index: collect all points with (fiber_idx, point_idx)
    all_points = []
    point_map = []  # (fiber_idx, point_idx)
    for i, fiber in enumerate(fibers):
        for si in range(len(fiber.centerline)):
            all_points.append(fiber.centerline[si])
            point_map.append((i, si))
    
    all_points = np.array(all_points)
    
    # Use grid-based spatial hashing
    cell_size = threshold
    grid = {}
    for idx, pt in enumerate(all_points):
        cell = tuple(np.floor(pt / cell_size).astype(int))
        if cell not in grid:
            grid[cell] = []
        grid[cell].append(idx)
    
    # Check only neighboring cells
    seen_pairs = set()
    for cell, indices in grid.items():
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    neighbor = (cell[0]+dx, cell[1]+dy, cell[2]+dz)
                    if neighbor in grid:
                        for idx1 in indices:
                            for idx2 in grid[neighbor]:
                                if idx1 >= idx2:
                                    continue
                                fi1, si1 = point_map[idx1]
                                fi2, si2 = point_map[idx2]
                                if fi1 == fi2:
                                    continue
                                pair_key = (min(fi1, fi2), max(fi1, fi2))
                                if pair_key in seen_pairs:
                                    continue
                                dist = np.linalg.norm(all_points[idx1] - all_points[idx2])
                                if dist < threshold:
                                    cl = Crosslink(
                                        position=(all_points[idx1] + all_points[idx2]) / 2,
                                        fiber_i=fi1, fiber_j=fi2,
                                        param_i=si1/(len(fibers[fi1].centerline)-1),
                                        param_j=si2/(len(fibers[fi2].centerline)-1),
                                    )
                                    crosslinks.append(cl)
                                    seen_pairs.add(pair_key)
    
    net.crosslinks = crosslinks
    return net


# ============================================================================
# Biomimetic Network Generator (merged collagen/fibrin)
# ============================================================================

def biomimetic_network(
    network_type: str = "collagen",
    num_fibers: int = 100,
    box_size: Tuple[float, float, float] = (50.0, 50.0, 20.0),
    persistence_length: float = 15.0,
    d_periodicity: float = 0.67,
    radius_mean: float = 0.5,
    bundling_probability: float = 0.3,
    branching_angle: float = 0.4,
    num_points: int = 30,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> FiberNetwork:
    """Generate biomimetic fiber network (collagen/fibrin/ECM).
    
    Merges collagen and fibrin generators into one unified interface.
    
    Parameters
    ----------
    network_type : str
        'collagen', 'fibrin', or 'generic_ecm'.
    num_fibers : int
        Number of fibers.
    box_size : (float, float, float)
        Domain size (x, y, z).
    persistence_length : float
        Fiber bending stiffness (higher = straighter).
    d_periodicity : float
        Collagen D-period (only for collagen type).
    radius_mean : float
        Mean fiber radius.
    bundling_probability : float
        Probability of fiber bundling (0-1).
    branching_angle : float
        Branching angle in radians (for fibrin).
    num_points : int
        Points per fiber.
    material : Material, optional
        Fiber material.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    FiberNetwork
    
    Examples
    --------
    >>> net = fn.create("biomimetic_network", network_type="collagen")
    >>> net = fn.create("biomimetic_network", network_type="fibrin",
    ...                 bundling_probability=0.5)
    """
    rng = np.random.RandomState(seed)
    bx, by, bz = box_size
    
    # Set type-specific defaults
    if network_type == 'collagen':
        curvature = 1.0 / persistence_length
        bundling_prob = bundling_probability
        branching = False
    elif network_type == 'fibrin':
        curvature = 1.0 / (persistence_length * 0.7)
        bundling_prob = bundling_probability * 0.5
        branching = True
    elif network_type == 'generic_ecm':
        curvature = 1.0 / persistence_length
        bundling_prob = bundling_probability
        branching = False
    else:
        raise ValueError(f"Unknown network_type: {network_type}")
    
    fibers = []
    for i in range(num_fibers):
        # Random start
        start = np.array([
            rng.uniform(0, bx),
            rng.uniform(0, by),
            rng.uniform(0, bz),
        ])
        
        # Random initial direction
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0, np.pi)
        direction = np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi),
        ])
        
        # Generate worm-like chain
        step_length = 2.0
        points = [start.copy()]
        current_pos = start.copy()
        current_dir = direction.copy()
        
        for step in range(num_points - 1):
            # Worm-like chain: direction persists with thermal fluctuations
            thermal_noise = rng.normal(0, curvature * step_length, 3)
            current_dir = current_dir + thermal_noise
            current_dir = current_dir / np.linalg.norm(current_dir)
            
            # Collagen D-period modulation
            if network_type == 'collagen' and d_periodicity > 0:
                d_mod = 0.1 * np.sin(2 * np.pi * step * step_length / d_periodicity)
                current_dir += d_mod * np.array([0, 0, 1])
                current_dir = current_dir / np.linalg.norm(current_dir)
            
            current_pos = current_pos + step_length * current_dir
            # Wrap around box
            current_pos = current_pos % box_size
            points.append(current_pos.copy())
        
        centerline = np.array(points)
        
        # Radius with bundling
        r = radius_mean
        if rng.random() < bundling_prob:
            r = radius_mean * rng.uniform(1.5, 3.0)
        
        fiber = Fiber(
            centerline=centerline,
            radius=r,
            material=material,
        )
        fibers.append(fiber)
        
        # Branching (fibrin)
        if branching and rng.random() < 0.2:
            branch_dir = current_dir.copy()
            branch_angle = rng.normal(0, branching_angle)
            # Rotate around random axis
            axis = rng.normal(0, 1, 3)
            axis = axis / np.linalg.norm(axis)
            # Simple rotation
            cos_a = np.cos(branch_angle)
            sin_a = np.sin(branch_angle)
            branch_dir = cos_a * branch_dir + sin_a * np.cross(axis, branch_dir)
            branch_dir = branch_dir / np.linalg.norm(branch_dir)
            
            branch_points = [current_pos.copy()]
            branch_pos = current_pos.copy()
            for step in range(num_points // 2):
                thermal_noise = rng.normal(0, curvature * step_length, 3)
                branch_dir = branch_dir + thermal_noise
                branch_dir = branch_dir / np.linalg.norm(branch_dir)
                branch_pos = branch_pos + step_length * branch_dir
                branch_points.append(branch_pos.copy())
            
            branch_cl = np.array(branch_points)
            branch_fiber = Fiber(
                centerline=branch_cl,
                radius=r * 0.7,
                material=material,
            )
            fibers.append(branch_fiber)
    
    net = FiberNetwork(fibers=fibers, dimension=3)
    
    # Detect crosslinks using spatial hashing for efficiency
    from fibernet.core.network import Crosslink
    crosslinks = []
    threshold = radius_mean * 6.0
    
    # Build spatial index: collect all points with (fiber_idx, point_idx)
    all_points = []
    point_map = []  # (fiber_idx, point_idx)
    for i, fiber in enumerate(fibers):
        for si in range(len(fiber.centerline)):
            all_points.append(fiber.centerline[si])
            point_map.append((i, si))
    
    all_points = np.array(all_points)
    
    # Use grid-based spatial hashing
    cell_size = threshold
    grid = {}
    for idx, pt in enumerate(all_points):
        cell = tuple(np.floor(pt / cell_size).astype(int))
        if cell not in grid:
            grid[cell] = []
        grid[cell].append(idx)
    
    # Check only neighboring cells
    seen_pairs = set()
    for cell, indices in grid.items():
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    neighbor = (cell[0]+dx, cell[1]+dy, cell[2]+dz)
                    if neighbor in grid:
                        for idx1 in indices:
                            for idx2 in grid[neighbor]:
                                if idx1 >= idx2:
                                    continue
                                fi1, si1 = point_map[idx1]
                                fi2, si2 = point_map[idx2]
                                if fi1 == fi2:
                                    continue
                                pair_key = (min(fi1, fi2), max(fi1, fi2))
                                if pair_key in seen_pairs:
                                    continue
                                dist = np.linalg.norm(all_points[idx1] - all_points[idx2])
                                if dist < threshold:
                                    cl = Crosslink(
                                        position=(all_points[idx1] + all_points[idx2]) / 2,
                                        fiber_i=fi1, fiber_j=fi2,
                                        param_i=si1/(len(fibers[fi1].centerline)-1),
                                        param_j=si2/(len(fibers[fi2].centerline)-1),
                                    )
                                    crosslinks.append(cl)
                                    seen_pairs.add(pair_key)
    
    net.crosslinks = crosslinks
    return net


# ============================================================================
# Hierarchical Lattice Generator
# ============================================================================

def hierarchical_lattice(
    levels: int = 2,
    base_topology: str = "triangular",
    cell_size: float = 50.0,
    scaling_factor: float = 0.3,
    radius: float = 0.1,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> FiberNetwork:
    """Generate hierarchical lattice with multi-scale structure.
    
    Recursively subdivides edges and adds cross-bracing at each level.
    Uses position-based node merging to ensure connectivity.
    
    Parameters
    ----------
    levels : int
        Recursion depth (0 = simple lattice, 2 = 2-level hierarchy).
    base_topology : str
        Base unit cell: 'triangular', 'square', 'honeycomb'.
    cell_size : float
        Overall structure size.
    scaling_factor : float
        Size ratio between levels (0.2-0.5 typical).
    radius : float
        Fiber radius (scales with level).
    material : Material, optional
        Fiber material.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    FiberNetwork
    """
    from fibernet.core.network import Crosslink
    
    # Build base edges
    if base_topology == 'triangular':
        h = cell_size * np.sqrt(3) / 2
        base_edges = [
            ((0, 0), (cell_size, 0)),
            ((cell_size, 0), (cell_size/2, h)),
            ((cell_size/2, h), (0, 0)),
        ]
    elif base_topology == 'square':
        base_edges = [
            ((0, 0), (cell_size, 0)),
            ((cell_size, 0), (cell_size, cell_size)),
            ((cell_size, cell_size), (0, cell_size)),
            ((0, cell_size), (0, 0)),
        ]
    elif base_topology == 'honeycomb':
        s = cell_size / 3
        h = s * np.sqrt(3) / 2
        base_edges = [
            ((0, 0), (s/2, 0)),
            ((s/2, 0), (s, h)),
            ((s, h), (s/2, 2*h)),
            ((s/2, 2*h), (0, 2*h)),
            ((0, 2*h), (-s/2, h)),
            ((-s/2, h), (0, 0)),
        ]
    else:
        raise ValueError(f"Unknown base_topology: {base_topology}")
    
    # Recursive subdivision: each edge -> midpoint + cross-brace
    all_edges = []
    
    def subdivide(p1, p2, level, current_radius):
        if level == 0:
            all_edges.append((p1, p2, current_radius))
            return
        
        # Midpoint
        mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        
        # Subdivide each half
        subdivide(p1, mid, level - 1, current_radius * 0.8)
        subdivide(mid, p2, level - 1, current_radius * 0.8)
        
        # Add cross-brace perpendicular to edge at midpoint
        edge_vec = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        edge_len = np.linalg.norm(edge_vec)
        if edge_len > 1e-10:
            perp = np.array([-edge_vec[1], edge_vec[0]]) / edge_len
            brace_len = edge_len * scaling_factor
            brace_start = (mid[0] - perp[0] * brace_len / 2,
                          mid[1] - perp[1] * brace_len / 2)
            brace_end = (mid[0] + perp[0] * brace_len / 2,
                        mid[1] + perp[1] * brace_len / 2)
            all_edges.append((brace_start, brace_end, current_radius * 0.6))
            # Connect brace to midpoint
            all_edges.append((mid, brace_start, current_radius * 0.6))
            all_edges.append((mid, brace_end, current_radius * 0.6))
    
    for p1, p2 in base_edges:
        subdivide(p1, p2, levels, radius)
    
    # Convert to FiberNetwork
    fibers = []
    for (p1, p2, r) in all_edges:
        p1_3d = np.array([p1[0], p1[1], 0.0])
        p2_3d = np.array([p2[0], p2[1], 0.0])
        fiber = Fiber(centerline=np.array([p1_3d, p2_3d]), radius=r, material=material)
        fibers.append(fiber)
    
    net = FiberNetwork(fibers=fibers, dimension=2)
    
    # Detect crosslinks by position (with tolerance for floating point)
    endpoint_to_fibers = {}
    for i, (p1, p2, r) in enumerate(all_edges):
        for p in [p1, p2]:
            # Use lower precision for merging (round to 3 decimal places)
            key = (round(p[0], 3), round(p[1], 3))
            endpoint_to_fibers.setdefault(key, []).append(i)
    
    crosslinks = []
    for key, fiber_indices in endpoint_to_fibers.items():
        if len(fiber_indices) < 2:
            continue
        pos = np.array([key[0], key[1], 0.0])
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
