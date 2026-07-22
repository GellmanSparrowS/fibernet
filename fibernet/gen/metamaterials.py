"""
Mechanics metamaterial structure generators (v2 — node-based, properly connected).

All generators use a node-edge graph approach ensuring full connectivity
when unit cells are tiled. A shared _graph_to_network helper converts
node positions and edge lists into FiberNetwork objects with proper crosslinks.

References:
- Gibson & Ashby, Cellular Solids (1997)
- Evans et al., Nature Materials 2001 (auxetic)
- Lakes, Science 1987 (re-entrant foam)
- Spadoni & Ruzzene, J Intell Mat Syst Str 2012 (chiral honeycomb)
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Set
from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork, Crosslink
from fibernet.core.material import Material


def _graph_to_network(
    nodes: Dict[int, np.ndarray],
    edges: Set[Tuple[int, int]],
    radius: float = 0.2,
    material: Optional[Material] = None,
    dimension: int = 2,
    metadata: Optional[dict] = None,
    box_size: Optional[np.ndarray] = None,
) -> FiberNetwork:
    """Convert a node-edge graph to a properly connected FiberNetwork.
    
    Each edge becomes a fiber. Shared nodes become crosslinks.
    """
    mat = material or Material(name="metamaterial")
    
    if box_size is not None:
        box = np.array(box_size, dtype=float)
    else:
        all_pos = np.array(list(nodes.values())) if nodes else np.zeros((1, 3))
        bb_min = all_pos.min(axis=0)
        bb_max = all_pos.max(axis=0)
        box = bb_max - bb_min
        if dimension == 2:
            box[2] = 0.0
    
    net = FiberNetwork(dimension=dimension, box_size=box, metadata=metadata or {})
    
    node_to_fibers: Dict[int, List[Tuple[int, float]]] = {nid: [] for nid in nodes}
    
    for edge in edges:
        n1, n2 = edge
        p1, p2 = nodes[n1], nodes[n2]
        dist = np.linalg.norm(p2 - p1)
        if dist < 1e-12:
            continue
        
        fid = net.num_fibers
        n_seg = max(4, int(dist / max(radius * 2, 0.1)))
        fiber = Fiber.straight(p1, p2, radius=radius, material=mat,
                              fiber_id=fid, segments=n_seg)
        net.add_fiber(fiber)
        
        node_to_fibers[n1].append((fid, 0.0))
        node_to_fibers[n2].append((fid, 1.0))
    
    for nid, fiber_list in node_to_fibers.items():
        if len(fiber_list) < 2:
            continue
        pos = nodes[nid]
        for i in range(len(fiber_list)):
            for j in range(i + 1, len(fiber_list)):
                fi, pi = fiber_list[i]
                fj, pj = fiber_list[j]
                net.add_crosslink(Crosslink(
                    fiber_i=fi, fiber_j=fj,
                    param_i=pi, param_j=pj,
                    position=pos.copy(),
                    crosslink_type="welded",
                ))
    
    return net


class _NodeGraph:
    """Helper for building node-edge graphs with automatic node merging."""
    
    def __init__(self, tolerance: float = 1e-6):
        self.nodes: Dict[int, np.ndarray] = {}
        self.edges: Set[Tuple[int, int]] = set()
        self._next_id = 0
        self._tolerance = tolerance
        self._pos_to_id: Dict[Tuple, int] = {}
    
    def add_node(self, pos: np.ndarray) -> int:
        """Add node, merging with existing nodes within tolerance."""
        pos = np.asarray(pos, dtype=float)
        # Round for fast lookup
        key = tuple(np.round(pos / self._tolerance).astype(int))
        if key in self._pos_to_id:
            return self._pos_to_id[key]
        # Also check nearby keys (for boundary cases)
        for dk in [(0,0,0), (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
            nkey = (key[0]+dk[0], key[1]+dk[1], key[2]+dk[2])
            if nkey in self._pos_to_id:
                existing_pos = self.nodes[self._pos_to_id[nkey]]
                if np.linalg.norm(existing_pos - pos) < self._tolerance:
                    return self._pos_to_id[nkey]
        
        nid = self._next_id
        self.nodes[nid] = pos.copy()
        self._pos_to_id[key] = nid
        self._next_id += 1
        return nid
    
    def add_edge(self, n1: int, n2: int):
        """Add edge between two node IDs."""
        if n1 != n2:
            self.edges.add((min(n1, n2), max(n1, n2)))
    
    def add_edge_by_pos(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[int, int]:
        """Add edge by positions (auto-creates nodes)."""
        n1 = self.add_node(p1)
        n2 = self.add_node(p2)
        self.add_edge(n1, n2)
        return n1, n2
    
    def to_network(self, radius=0.2, material=None, dimension=2,
                   metadata=None, box_size=None) -> FiberNetwork:
        return _graph_to_network(self.nodes, self.edges, radius=radius,
                                material=material, dimension=dimension,
                                metadata=metadata, box_size=box_size)




def _ensure_connected(net: FiberNetwork, max_gap_factor: float = 3.0) -> FiberNetwork:
    """Post-process to ensure all components are connected.
    
    If the network has multiple disconnected components, bridges them
    using connect_components with a generous gap tolerance.
    """
    from collections import defaultdict
    adj = defaultdict(set)
    for cl in net.crosslinks:
        adj[cl.fiber_i].add(cl.fiber_j)
        adj[cl.fiber_j].add(cl.fiber_i)
    
    visited = set()
    n_components = 0
    for start in range(net.num_fibers):
        if start not in visited:
            n_components += 1
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                queue.extend(adj[node] - visited)
    
    if n_components > 1:
        # Bridge with generous max_gap
        max_gap = max_gap_factor * net.mean_fiber_length if net.mean_fiber_length > 0 else 50.0
        net.connect_components(max_gap=max_gap)
    
    return net

# ===========================================================================
# Re-entrant Honeycomb (2D Auxetic) — properly tessellating
# ===========================================================================

def reentrant_honeycomb_2d(
    cell_height: float = 10.0,
    cell_width: float = 10.0,
    reentrant_angle: float = 150.0,
    grid_size: Tuple[int, int] = (5, 5),
    radius: float = 0.2,
    material: Optional[Material] = None,
    vertical_thickness: float = 1.0,
) -> FiberNetwork:
    """Generate a 2D re-entrant honeycomb auxetic structure.
    
    Uses a properly tessellating unit cell. The cell is a hexagonal shape
    where re-entrant (inward-pointing) struts create negative Poisson's ratio.
    
    The reentrant_angle controls the inward strut angle:
    - 120° = standard honeycomb (positive Poisson's ratio)
    - >120° = re-entrant (auxetic, negative Poisson's ratio)
    - 150° = strongly auxetic
    """
    mat = material or Material(name="reentrant")
    theta = np.radians(reentrant_angle)
    nx, ny = grid_size
    
    l = cell_width / 2  # inclined strut length (half-width)
    h = cell_height / 2  # vertical strut half-height
    
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    # Proper tessellation dimensions
    unit_w = 2 * l * abs(cos_t)
    unit_h = h + l * abs(sin_t)
    
    graph = _NodeGraph(tolerance=radius * 0.1)
    
    for ci in range(nx):
        for cj in range(ny):
            ox = ci * unit_w
            oy = cj * unit_h
            
            # Unit cell with properly shared boundary nodes
            # Layout (for re-entrant, theta > 90°, cos_t < 0):
            #
            #        n3 (left-top, extends left)
            #       /  \
            #      n2    \
            #      |      n4 (right-top)
            #      |      |
            #      n7     n5
            #       \    /
            #        n6 (bottom-center)
            #
            # Actually, the proper tessellating re-entrant cell is:
            # n1=bottom-left, n2=left-mid, n3=top-left (re-entrant arm)
            # n4=top-right (re-entrant arm), n5=right-mid, n6=bottom-right
            # The top of this cell connects to the bottom of the cell above
            # via shared nodes at the vertical strut endpoints
            
            # Vertices of the hexagonal cell
            p1 = np.array([ox, oy, 0.0])                       # bottom-left
            p2 = np.array([ox, oy + h, 0.0])                   # left-mid-bottom
            p3 = np.array([ox + l * cos_t, oy + unit_h, 0.0])  # top-left (re-entrant)
            p4 = np.array([ox + unit_w, oy + unit_h, 0.0])     # top-right (re-entrant)  
            p5 = np.array([ox + unit_w, oy + h, 0.0])          # right-mid-bottom
            p6 = np.array([ox + unit_w, oy, 0.0])              # bottom-right
            
            # Internal re-entrant nodes
            p7 = np.array([ox + l * cos_t, oy + l * abs(sin_t), 0.0])  # inner-left
            p8 = np.array([ox + unit_w - l * abs(cos_t), oy + l * abs(sin_t), 0.0])  # inner-right (same x as p7 by symmetry if cos_t is symmetric)
            # Actually for proper re-entrant:
            p8 = np.array([ox + unit_w + l * cos_t, oy + l * abs(sin_t), 0.0])
            
            n1 = graph.add_node(p1)
            n2 = graph.add_node(p2)
            n3 = graph.add_node(p3)
            n4 = graph.add_node(p4)
            n5 = graph.add_node(p5)
            n6 = graph.add_node(p6)
            n7 = graph.add_node(p7)
            n8 = graph.add_node(p8)
            
            # Edges: hexagonal ring + internal re-entrant struts
            graph.add_edge(n1, n2)  # left vertical (bottom half)
            graph.add_edge(n2, n3)  # upper-left strut
            graph.add_edge(n3, n4)  # top horizontal
            graph.add_edge(n4, n5)  # upper-right strut
            graph.add_edge(n5, n6)  # right vertical (bottom half)
            graph.add_edge(n6, n1)  # bottom horizontal (shared with cell below)
            
            # Re-entrant internal struts
            graph.add_edge(n2, n7)  # left vertical to inner
            graph.add_edge(n7, n8)  # inner horizontal
            graph.add_edge(n8, n5)  # inner to right vertical
    
    Lx = nx * unit_w
    Ly = ny * unit_h
    
    net = graph.to_network(
        radius=radius, material=mat, dimension=2,
        metadata={
            "generator": "reentrant_honeycomb_2d",
            "reentrant_angle_deg": reentrant_angle,
            "cell_height": cell_height,
            "cell_width": cell_width,
            "grid_size": grid_size,
        },
        box_size=np.array([Lx, Ly, 0.0]),
    )
    return net


# ===========================================================================
# Chiral Honeycomb — properly connected
# ===========================================================================

def chiral_honeycomb_2d(
    node_radius: float = 3.0,
    ligament_length: float = 8.0,
    grid_size: Tuple[int, int] = (5, 5),
    radius: float = 0.2,
    material: Optional[Material] = None,
    num_node_points: int = 6,
) -> FiberNetwork:
    """Generate a 2D chiral honeycomb with rotating nodes connected by ligaments."""
    mat = material or Material(name="chiral")
    nx, ny = grid_size
    r = node_radius
    L = ligament_length
    
    cell_size = 2 * r + L
    dx = cell_size
    dy = cell_size * np.sqrt(3) / 2
    
    graph = _NodeGraph(tolerance=r * 0.1)
    
    # Place cell centers on hexagonal grid
    cell_centers = []
    for ci in range(nx):
        for cj in range(ny):
            cx = ci * dx + (cj % 2) * dx / 2
            cy = cj * dy
            cell_centers.append((ci, cj, cx, cy))
    
    # For each cell, create ring nodes and connect to neighbors
    for idx, (ci, cj, cx, cy) in enumerate(cell_centers):
        # 6 ring attachment points
        ring_nids = []
        for k in range(6):
            angle = np.pi / 6 + k * np.pi / 3
            px = cx + r * np.cos(angle)
            py = cy + r * np.sin(angle)
            nid = graph.add_node(np.array([px, py, 0.0]))
            ring_nids.append(nid)
        
        # Ring edges
        for k in range(6):
            graph.add_edge(ring_nids[k], ring_nids[(k+1)%6])
        
        # Connect to 6 neighbors
        for k in range(6):
            angle = k * np.pi / 3
            # Find neighbor in this direction
            for jdx, (c2, d2, ncx, ncy) in enumerate(cell_centers):
                if jdx == idx:
                    continue
                ddx = ncx - cx
                ddy = ncy - cy
                dist = np.sqrt(ddx**2 + ddy**2)
                if abs(dist - cell_size) < cell_size * 0.2:
                    # Check if this neighbor is in direction k
                    target_dx = np.cos(angle) * cell_size
                    target_dy = np.sin(angle) * cell_size
                    err = np.sqrt((ddx - target_dx)**2 + (ddy - target_dy)**2)
                    if err < cell_size * 0.3:
                        # Find the corresponding attachment point on neighbor
                        opp_angle = angle + np.pi
                        # Find which ring index on the neighbor faces back
                        for m in range(6):
                            nb_angle = np.pi / 6 + m * np.pi / 3
                            nb_px = ncx + r * np.cos(nb_angle)
                            nb_py = ncy + r * np.sin(nb_angle)
                            # Check if this point is close to the ligament target
                            src_px = cx + r * np.cos(np.pi/6 + k * np.pi/3)
                            src_py = cy + r * np.sin(np.pi/6 + k * np.pi/3)
                            lig_dist = np.sqrt((src_px - nb_px)**2 + (src_py - nb_py)**2)
                            if lig_dist < L * 1.5:
                                tgt_nid = graph.add_node(np.array([nb_px, nb_py, 0.0]))
                                graph.add_edge(ring_nids[k], tgt_nid)
                                break
                        break
    
    Lx = nx * dx + dx / 2
    Ly = ny * dy + dy
    
    return graph.to_network(
        radius=radius, material=mat, dimension=2,
        metadata={
            "generator": "chiral_honeycomb_2d",
            "node_radius": node_radius,
            "ligament_length": ligament_length,
        },
        box_size=np.array([Lx, Ly, 0.0]),
    )


# ===========================================================================
# Star Honeycomb — properly tiled
# ===========================================================================

def star_honeycomb_2d(
    cell_size: float = 10.0,
    star_angle: float = 60.0,
    grid_size: Tuple[int, int] = (5, 5),
    radius: float = 0.2,
    material: Optional[Material] = None,
    num_arms: int = 4,
    star_inner_angle: float = 30.0,
) -> FiberNetwork:
    """Generate a 2D star-shaped honeycomb (star tessellation)."""
    mat = material or Material(name="star")
    nx, ny = grid_size
    a = cell_size
    alpha = np.radians(star_angle)
    
    graph = _NodeGraph(tolerance=a * 0.01)
    
    unit_w = a
    unit_h = a
    star_arm = a / 2 * np.sin(alpha / 2)
    
    n_arms = max(num_arms, 3)
    for ci in range(nx):
        for cj in range(ny):
            cx = (ci + 0.5) * unit_w
            cy = (cj + 0.5) * unit_h
            
            # Inner star vertices
            inner_nids = []
            for k in range(n_arms):
                angle = k * 2 * np.pi / n_arms
                px = cx + star_arm * np.cos(angle)
                py = cy + star_arm * np.sin(angle)
                inner_nids.append(graph.add_node(np.array([px, py, 0.0])))
            
            # Corner vertices (shared between cells)
            corner_nids = []
            for k, (dx_c, dy_c) in enumerate([(0,0), (1,0), (1,1), (0,1)]):
                px = (ci + dx_c) * unit_w
                py = (cj + dy_c) * unit_h
                corner_nids.append(graph.add_node(np.array([px, py, 0.0])))
            
            # Star inner ring
            for k in range(n_arms):
                graph.add_edge(inner_nids[k], inner_nids[(k+1) % n_arms])
            
            # Connect inner to nearest corners
            for k in range(n_arms):
                graph.add_edge(inner_nids[k], corner_nids[k % 4])
    
    return graph.to_network(
        radius=radius, material=mat, dimension=2,
        metadata={"generator": "star_honeycomb_2d", "cell_size": cell_size, "star_angle": star_angle},
        box_size=np.array([nx * unit_w, ny * unit_h, 0.0]),
    )


# ===========================================================================
# Arrowhead Auxetic — properly tiled
# ===========================================================================

def arrowhead_auxetic_2d(
    cell_size: float = 10.0,
    arrow_angle: float = 60.0,
    grid_size: Tuple[int, int] = (5, 5),
    radius: float = 0.2,
    material: Optional[Material] = None,
    arm_angle: float = None,
) -> FiberNetwork:
    """Generate a 2D arrowhead auxetic structure."""
    mat = material or Material(name="arrowhead")
    nx, ny = grid_size
    a = cell_size
    theta = np.radians(arrow_angle)
    
    graph = _NodeGraph(tolerance=a * 0.01)
    
    unit_w = a
    unit_h = a * np.tan(theta / 2)
    
    for ci in range(nx):
        for cj in range(ny):
            ox = ci * unit_w
            oy = cj * unit_h
            
            # Arrow: V-shape pointing right
            # Nodes shared between adjacent cells
            tip = np.array([ox + unit_w, oy + unit_h / 2, 0.0])
            bl = np.array([ox, oy, 0.0])
            tl = np.array([ox, oy + unit_h, 0.0])
            mid_l = np.array([ox, oy + unit_h / 2, 0.0])
            
            n_tip = graph.add_node(tip)
            n_bl = graph.add_node(bl)
            n_tl = graph.add_node(tl)
            n_ml = graph.add_node(mid_l)
            
            graph.add_edge(n_ml, n_tip)
            graph.add_edge(n_bl, n_tip)
            graph.add_edge(n_tl, n_tip)
            graph.add_edge(n_bl, n_ml)
            graph.add_edge(n_ml, n_tl)
    
    return graph.to_network(
        radius=radius, material=mat, dimension=2,
        metadata={"generator": "arrowhead_auxetic_2d", "cell_size": cell_size},
        box_size=np.array([nx * unit_w, ny * unit_h, 0.0]),
    )


# ===========================================================================
# Hierarchical Lattice — properly connected
# ===========================================================================

def hierarchical_lattice_2d(
    cell_size: float = 20.0,
    grid_size: Tuple[int, int] = (4, 4),
    levels: int = 2,
    radius: float = 0.2,
    material: Optional[Material] = None,
    base_type: str = "triangular",
) -> FiberNetwork:
    """Generate a 2D hierarchical lattice (self-similar structure).
    
    Uses alternating up/down triangles for proper edge sharing,
    then applies recursive subdivision at each hierarchy level.
    """
    mat = material or Material(name="hierarchical")
    nx, ny = grid_size
    a = cell_size
    h = a * np.sqrt(3) / 2
    
    graph = _NodeGraph(tolerance=a * 0.0001)
    
    # Build proper triangular tessellation with shared edges
    # Collect all unique edges first
    edge_set = set()
    
    for ci in range(nx):
        for cj in range(ny):
            ox = ci * a
            oy = cj * h
            
            # Up triangle
            p1 = (ox, oy)
            p2 = (ox + a, oy)
            p3 = (ox + a/2, oy + h)
            
            # Down triangle (offset)
            p4 = (ox + a/2, oy + h)
            p5 = (ox + a*3/2, oy + h)
            p6 = (ox + a, oy)
            
            # Add edges (as sorted tuples of rounded positions)
            for e in [(p1,p2), (p2,p3), (p3,p1), (p4,p5), (p5,p6), (p6,p4)]:
                key = tuple(sorted([tuple(np.round(e[0], 8)), tuple(np.round(e[1], 8))]))
                edge_set.add(key)
    
    def subdivide(p1_2d, p2_2d, level):
        """Recursively add hierarchical sub-structure."""
        pp1 = np.array([p1_2d[0], p1_2d[1], 0.0])
        pp2 = np.array([p2_2d[0], p2_2d[1], 0.0])
        n1 = graph.add_node(pp1)
        n2 = graph.add_node(pp2)
        graph.add_edge(n1, n2)
        
        if level <= 0:
            return
        
        mid = (pp1 + pp2) / 2
        n_mid = graph.add_node(mid)
        graph.add_edge(n1, n_mid)
        graph.add_edge(n_mid, n2)
        
        subdivide(tuple(mid[:2]), p2_2d, level - 1)
        subdivide(p1_2d, tuple(mid[:2]), level - 1)
    
    for (p1, p2) in edge_set:
        subdivide(p1, p2, levels - 1)
    
    all_pos = np.array(list(graph.nodes.values())) if graph.nodes else np.zeros((1, 3))
    bb = all_pos.max(axis=0) - all_pos.min(axis=0)
    bb[2] = 0.0
    
    net = graph.to_network(
        radius=radius, material=mat, dimension=2,
        metadata={"generator": "hierarchical_lattice_2d", "cell_size": cell_size, "levels": levels},
        box_size=bb,
    )
    return _ensure_connected(net)


# ===========================================================================
# Missing-Rib Auxetic — properly tiled
# ===========================================================================

def missing_rib_auxetic_2d(
    cell_size: float = 10.0,
    grid_size: Tuple[int, int] = (5, 5),
    radius: float = 0.2,
    rib_angle: float = 45.0,
    material: Optional[Material] = None,
) -> FiberNetwork:
    """Generate a 2D missing-rib auxetic structure."""
    mat = material or Material(name="missing_rib")
    nx, ny = grid_size
    a = cell_size
    
    graph = _NodeGraph(tolerance=a * 0.01)
    
    grid_nodes = {}
    for i in range(nx + 1):
        for j in range(ny + 1):
            nid = graph.add_node(np.array([i * a, j * a, 0.0]))
            grid_nodes[(i, j)] = nid
    
    for i in range(nx + 1):
        for j in range(ny + 1):
            if i < nx:
                graph.add_edge(grid_nodes[(i, j)], grid_nodes[(i+1, j)])
            if j < ny:
                graph.add_edge(grid_nodes[(i, j)], grid_nodes[(i, j+1)])
    
    for i in range(nx):
        for j in range(ny):
            if (i + j) % 2 == 0:
                graph.add_edge(grid_nodes[(i, j)], grid_nodes[(i+1, j+1)])
    
    return graph.to_network(
        radius=radius, material=mat, dimension=2,
        metadata={"generator": "missing_rib_auxetic_2d", "cell_size": cell_size},
        box_size=np.array([nx * a, ny * a, 0.0]),
    )


# ===========================================================================
# Kagome Lattice — properly tiled
# ===========================================================================

def kagome_lattice_2d(
    spacing: float = 5.0,
    grid_size: Tuple[int, int] = (8, 8),
    radius: float = 0.1,
    material: Optional[Material] = None,
) -> FiberNetwork:
    """Generate a 2D Kagome lattice (trihexagonal tiling) — properly connected.
    
    The Kagome lattice consists of corner-sharing triangles arranged on
    a triangular lattice. Each node has coordination number 4.
    """
    mat = material or Material(name="kagome")
    nx, ny = grid_size
    a = spacing
    
    graph = _NodeGraph(tolerance=a * 0.01)
    
    # Kagome = triangular lattice of edge-midpoints
    # Place nodes at edge midpoints of a triangular lattice
    # Then connect nodes that share a vertex
    
    # Triangular lattice vertices
    tri_nodes = {}
    for i in range(nx + 2):
        for j in range(ny + 2):
            x = i * a + (j % 2) * a / 2
            y = j * a * np.sqrt(3) / 2
            tri_nodes[(i, j)] = np.array([x, y, 0.0])
    
    # Edge midpoints become kagome nodes
    # Each triangle vertex connects to 6 neighbors via edges
    # Kagome node = midpoint of each triangular lattice edge
    midpoints = {}
    
    def get_midpoint(key_i, key_j):
        """Get or create midpoint between two triangular lattice nodes."""
        edge_key = (min(key_i, key_j), max(key_i, key_j))
        if edge_key not in midpoints:
            p1 = tri_nodes[key_i]
            p2 = tri_nodes[key_j]
            mid = (p1 + p2) / 2
            midpoints[edge_key] = graph.add_node(mid)
        return midpoints[edge_key]
    
    # Create edges between midpoints that share a triangular lattice vertex
    for (i, j), pos in tri_nodes.items():
        # Find all neighbors of this vertex in the triangular lattice
        neighbor_keys = [
            (i+1, j), (i-1, j),
            (i, j+1), (i, j-1),
            (i+1, j-1) if j % 2 == 0 else (i+1, j+1),
            (i-1, j-1) if j % 2 == 0 else (i-1, j+1),
        ]
        
        # Get midpoints of all edges from this vertex
        vertex_mids = []
        for nk in neighbor_keys:
            if nk in tri_nodes:
                mp = get_midpoint((i, j), nk)
                vertex_mids.append(mp)
        
        # Connect consecutive midpoints (they form the Kagome pattern)
        for k in range(len(vertex_mids)):
            graph.add_edge(vertex_mids[k], vertex_mids[(k+1) % len(vertex_mids)])
    
    all_pos = np.array(list(graph.nodes.values())) if graph.nodes else np.zeros((1, 3))
    bb = all_pos.max(axis=0) - all_pos.min(axis=0)
    bb[2] = 0.0
    
    return graph.to_network(
        radius=radius, material=mat, dimension=2,
        metadata={"generator": "kagome_lattice_2d", "spacing": spacing},
        box_size=bb,
    )


# ===========================================================================
# Proper Octet Truss 3D — node-based
# ===========================================================================

def proper_octet_truss_3d(
    spacing: float = 5.0,
    grid_size: Tuple[int, int, int] = (3, 3, 3),
    radius: float = 0.15,
    material: Optional[Material] = None,
) -> FiberNetwork:
    """Generate a 3D octet truss with all 12 struts per unit cell."""
    mat = material or Material(name="octet")
    nx, ny, nz = grid_size
    a = spacing
    
    graph = _NodeGraph(tolerance=a * 0.01)
    
    nodes = {}
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                pos = np.array([i * a, j * a, k * a])
                nodes[(i, j, k)] = graph.add_node(pos)
    
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                n = (i, j, k)
                if n not in nodes:
                    continue
                
                # 12 edges per octet truss node
                edge_dirs = [
                    (1,0,0), (0,1,0), (0,0,1),       # cube edges
                    (1,1,0), (1,0,1), (0,1,1),        # face diagonals
                    (1,-1,0), (1,0,-1), (0,1,-1),     # negative face diagonals
                    (1,1,1),                            # body diagonal
                    (1,1,-1), (1,-1,1),                # negative body diagonals
                ]
                
                for di, dj, dk in edge_dirs:
                    ni, nj, nk = i+di, j+dj, k+dk
                    if (ni, nj, nk) in nodes:
                        graph.add_edge(nodes[n], nodes[(ni, nj, nk)])
    
    return graph.to_network(
        radius=radius, material=mat, dimension=3,
        metadata={"generator": "proper_octet_truss_3d", "spacing": spacing},
        box_size=np.array([nx * a, ny * a, nz * a]),
    )


# ===========================================================================
# Diamond Lattice 3D
# ===========================================================================

def diamond_lattice_3d(
    spacing: float = 5.0,
    grid_size: Tuple[int, int, int] = (3, 3, 3),
    radius: float = 0.15,
    material: Optional[Material] = None,
) -> FiberNetwork:
    """Generate a 3D diamond lattice (tetrahedral coordination)."""
    mat = material or Material(name="diamond")
    nx, ny, nz = grid_size
    a = spacing
    
    graph = _NodeGraph(tolerance=a * 0.01)
    
    # Diamond = two interpenetrating FCC lattices
    # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    fcc_basis = [
        np.array([0, 0, 0]),
        np.array([0.5, 0.5, 0]),
        np.array([0.5, 0, 0.5]),
        np.array([0, 0.5, 0.5]),
    ]
    
    # Second sublattice offset
    offset = np.array([0.25, 0.25, 0.25])
    
    node_positions = {}
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cell_origin = np.array([i, j, k]) * a
                for b in fcc_basis:
                    # Sublattice A
                    pos_a = cell_origin + b * a
                    key_a = tuple(np.round(pos_a, 6))
                    node_positions[key_a] = pos_a
                    
                    # Sublattice B
                    pos_b = cell_origin + (b + offset) * a
                    key_b = tuple(np.round(pos_b, 6))
                    node_positions[key_b] = pos_b
    
    # Add all nodes
    node_ids = {}
    for key, pos in node_positions.items():
        node_ids[key] = graph.add_node(pos)
    
    # Connect nearest neighbors (distance = a * sqrt(3) / 4)
    nn_dist = a * np.sqrt(3) / 4 * 1.1  # with tolerance
    keys = list(node_positions.keys())
    for i_k in range(len(keys)):
        for j_k in range(i_k + 1, len(keys)):
            d = np.linalg.norm(node_positions[keys[i_k]] - node_positions[keys[j_k]])
            if d < nn_dist:
                graph.add_edge(node_ids[keys[i_k]], node_ids[keys[j_k]])
    
    all_pos = np.array(list(graph.nodes.values()))
    bb = all_pos.max(axis=0) - all_pos.min(axis=0)
    
    return graph.to_network(
        radius=radius, material=mat, dimension=3,
        metadata={"generator": "diamond_lattice_3d", "spacing": spacing},
        box_size=bb,
    )


# ===========================================================================
# Gyroid Lattice 3D — node-based
# ===========================================================================

def gyroid_lattice_3d(
    cell_size: float = 10.0,
    grid_size: Tuple[int, int, int] = (2, 2, 2),
    radius: float = 0.15,
    resolution: int = 8,
    material: Optional[Material] = None,
    threshold: float = 0.0,
) -> FiberNetwork:
    """Generate a 3D gyroid lattice (TPMS-inspired).
    
    Uses marching-cubes-style surface extraction to trace the
    gyroid zero-isosurface and create a connected fiber network.
    
    Parameters
    ----------
    cell_size : float
        Unit cell edge length.
    grid_size : tuple
        (nx, ny, nz) number of unit cells.
    resolution : int
        Grid points per unit cell edge (higher = smoother).
    """
    mat = material or Material(name="gyroid")
    nx, ny, nz = grid_size
    a = cell_size
    
    graph = _NodeGraph(tolerance=a / resolution * 0.5)
    
    def gyroid_value(x, y, z):
        kx = 2 * np.pi * x / a
        ky = 2 * np.pi * y / a
        kz = 2 * np.pi * z / a
        return (np.sin(kx)*np.cos(ky) + np.sin(ky)*np.cos(kz) 
                + np.sin(kz)*np.cos(kx))
    
    # Build 3D grid of gyroid values
    total_x = nx * resolution
    total_y = ny * resolution
    total_z = nz * resolution
    
    dx = nx * a / total_x
    dy = ny * a / total_y
    dz = nz * a / total_z
    
    values = np.zeros((total_x + 1, total_y + 1, total_z + 1))
    for ix in range(total_x + 1):
        for iy in range(total_y + 1):
            for iz in range(total_z + 1):
                values[ix, iy, iz] = gyroid_value(ix * dx, iy * dy, iz * dz)
    
    # For each cell, find edges that cross the zero-isosurface
    # and create fibers connecting crossing points
    for ix in range(total_x):
        for iy in range(total_y):
            for iz in range(total_z):
                # Get 8 corner values
                v = [
                    values[ix, iy, iz],
                    values[ix+1, iy, iz],
                    values[ix+1, iy+1, iz],
                    values[ix, iy+1, iz],
                    values[ix, iy, iz+1],
                    values[ix+1, iy, iz+1],
                    values[ix+1, iy+1, iz+1],
                    values[ix, iy+1, iz+1],
                ]
                
                # 12 edges of the cube
                edge_pairs = [
                    (0,1), (1,2), (2,3), (3,0),  # bottom face
                    (4,5), (5,6), (6,7), (7,4),  # top face
                    (0,4), (1,5), (2,6), (3,7),  # vertical
                ]
                
                # Corner positions
                corners = [
                    (ix, iy, iz), (ix+1, iy, iz),
                    (ix+1, iy+1, iz), (ix, iy+1, iz),
                    (ix, iy, iz+1), (ix+1, iy, iz+1),
                    (ix+1, iy+1, iz+1), (ix, iy+1, iz+1),
                ]
                
                # Find zero-crossings on edges
                crossing_pts = []
                for ci, cj in edge_pairs:
                    if v[ci] * v[cj] < 0:
                        t = abs(v[ci]) / (abs(v[ci]) + abs(v[cj]))
                        xi = corners[ci][0] + t * (corners[cj][0] - corners[ci][0])
                        yi = corners[ci][1] + t * (corners[cj][1] - corners[ci][1])
                        zi = corners[ci][2] + t * (corners[cj][2] - corners[ci][2])
                        p = np.array([xi * dx, yi * dy, zi * dz])
                        crossing_pts.append(graph.add_node(p))
                
                # Connect crossing points within this cell
                for i in range(len(crossing_pts)):
                    for j in range(i + 1, len(crossing_pts)):
                        graph.add_edge(crossing_pts[i], crossing_pts[j])
    
    if not graph.nodes:
        return FiberNetwork(dimension=3, box_size=np.array([nx*a, ny*a, nz*a]))
    
    all_pos = np.array(list(graph.nodes.values()))
    bb = all_pos.max(axis=0) - all_pos.min(axis=0)
    
    return graph.to_network(
        radius=radius, material=mat, dimension=3,
        metadata={"generator": "gyroid_lattice_3d", "cell_size": cell_size, "resolution": resolution},
        box_size=bb,
    )


# ===========================================================================
# Re-entrant Honeycomb 3D
# ===========================================================================

def reentrant_honeycomb_3d(
    cell_size: float = 10.0,
    reentrant_angle: float = 150.0,
    grid_size: Tuple[int, int, int] = (3, 3, 3),
    radius: float = 0.2,
    material: Optional[Material] = None,
) -> FiberNetwork:
    """Generate a 3D re-entrant auxetic lattice."""
    mat = material or Material(name="reentrant_3d")
    theta = np.radians(reentrant_angle)
    nx, ny, nz = grid_size
    a = cell_size
    
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    d = a * abs(cos_t)
    unit_h = a * (1 + abs(sin_t))
    
    graph = _NodeGraph(tolerance=a * 0.01)
    
    for ci in range(nx):
        for cj in range(ny):
            for ck in range(nz):
                ox = ci * 2 * d
                oy = cj * 2 * d
                oz = ck * unit_h
                
                # 8 corners of the cube
                corners = []
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            x = ox + di * 2 * d
                            y = oy + dj * 2 * d
                            z = oz + dk * unit_h
                            corners.append(np.array([x, y, z]))
                
                corner_nids = [graph.add_node(c) for c in corners]
                
                # Cube edges (12 edges)
                edge_pairs = [
                    (0,1), (0,2), (0,4),  # from corner 0
                    (1,3), (1,5),          # from corner 1
                    (2,3), (2,6),          # from corner 2
                    (3,7),                 # from corner 3
                    (4,5), (4,6),          # from corner 4
                    (5,7),                 # from corner 5
                    (6,7),                 # from corner 6
                ]
                
                for i, j in edge_pairs:
                    graph.add_edge(corner_nids[i], corner_nids[j])
                
                # Re-entrant internal struts
                center = np.array([ox + d, oy + d, oz + unit_h/2])
                n_center = graph.add_node(center)
                
                for cn in corner_nids:
                    graph.add_edge(n_center, cn)
    
    Lx = nx * 2 * d
    Ly = ny * 2 * d
    Lz = nz * unit_h
    
    return graph.to_network(
        radius=radius, material=mat, dimension=3,
        metadata={"generator": "reentrant_honeycomb_3d", "reentrant_angle": reentrant_angle},
        box_size=np.array([Lx, Ly, Lz]),
    )


# ===========================================================================
# Plate/Shell Lattice 3D
# ===========================================================================

def plate_lattice_3d(
    spacing: float = 10.0,
    grid_size: Tuple[int, int, int] = (3, 3, 3),
    plate_thickness: float = 0.5,
    material: Optional[Material] = None,
    radius: float = 0.1,
) -> FiberNetwork:
    """Generate a 3D plate lattice (cubic + octahedral plates)."""
    mat = material or Material(name="plate_lattice")
    nx, ny, nz = grid_size
    a = spacing
    
    graph = _NodeGraph(tolerance=a * 0.02)
    
    for ci in range(nx):
        for cj in range(ny):
            for ck in range(nz):
                ox, oy, oz = ci * a, cj * a, ck * a
                
                # 8 cube corners
                corners = []
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            corners.append(np.array([ox + di*a, oy + dj*a, oz + dk*a]))
                
                # Face diagonals (representing plates)
                for face in range(6):
                    if face == 0:  # bottom face (z=oz)
                        pts = [corners[i] for i in [0,1,2,3]]
                    elif face == 1:  # top face (z=oz+a)
                        pts = [corners[i] for i in [4,5,6,7]]
                    elif face == 2:  # front face (y=oy)
                        pts = [corners[i] for i in [0,1,4,5]]
                    elif face == 3:  # back face (y=oy+a)
                        pts = [corners[i] for i in [2,3,6,7]]
                    elif face == 4:  # left face (x=ox)
                        pts = [corners[i] for i in [0,2,4,6]]
                    else:  # right face (x=ox+a)
                        pts = [corners[i] for i in [1,3,5,7]]
                    
                    nids = [graph.add_node(p) for p in pts]
                    # Connect face edges + diagonals
                    for k in range(4):
                        graph.add_edge(nids[k], nids[(k+1)%4])
                    graph.add_edge(nids[0], nids[2])
                    graph.add_edge(nids[1], nids[3])
    
    return graph.to_network(
        radius=radius, material=mat, dimension=3,
        metadata={"generator": "plate_lattice_3d", "spacing": spacing},
        box_size=np.array([nx * a, ny * a, nz * a]),
    )
