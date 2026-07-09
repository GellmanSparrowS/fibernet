"""
FiberGraph Builder — shared infrastructure for all generators.

Provides a node-edge graph abstraction that all generators build upon,
ensuring proper connectivity, crosslink creation, and standardized
parameter handling.

Architecture
------------
Every generator follows the same pattern:

    1. Build a FiberGraph (nodes + edges in space)
    2. Call ``to_network()`` to get a FiberNetwork with proper crosslinks

The FiberGraph is the *authoritative* topological representation:
- Nodes = junction points in 3D space
- Edges = fiber segments connecting two nodes
- A node shared by multiple edges becomes a crosslink

This separates *topology* (which edges connect to which nodes)
from *simulation* (FEM beam elements, dynamics springs).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Set, Any, Callable
from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork, Crosslink
from fibernet.core.material import Material


# ============================================================================
# FiberGraph: the authoritative graph model
# ============================================================================

@dataclass
class GraphNode:
    """A node in the fiber graph — a junction point in space."""
    position: np.ndarray
    node_id: int = -1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """An edge in the fiber graph — a fiber segment between two nodes."""
    node_i: int
    node_j: int
    edge_id: int = -1
    radius: float = 0.1
    material: Optional[Material] = None
    segments: int = 4
    metadata: Dict[str, Any] = field(default_factory=dict)


class FiberGraph:
    """Explicit node-edge graph for fiber network construction.
    
    This is the foundational data structure. All generators build
    a FiberGraph, then convert to FiberNetwork for simulation.
    
    Parameters
    ----------
    dimension : int
        2 or 3.
    tolerance : float
        Node merging tolerance (fraction of characteristic length).
    """
    
    def __init__(self, dimension: int = 2, tolerance: float = 1e-6):
        self.dimension = dimension
        self._tolerance = tolerance
        self._nodes: Dict[int, GraphNode] = {}
        self._edges: Dict[int, GraphEdge] = {}
        self._next_node_id = 0
        self._next_edge_id = 0
        # Spatial index for fast node merging
        self._pos_to_id: Dict[Tuple, int] = {}
    
    # ---- Node operations ----
    
    def add_node(self, pos: np.ndarray, **metadata) -> int:
        """Add a node, merging with existing nodes within tolerance.
        
        Returns the node ID (existing or new).
        """
        pos = np.asarray(pos, dtype=float)
        if len(pos) == 2:
            pos = np.append(pos, 0.0)
        
        # Fast lookup via rounded position key
        key = self._pos_key(pos)
        if key in self._pos_to_id:
            existing_id = self._pos_to_id[key]
            existing_pos = self._nodes[existing_id].position
            if np.linalg.norm(existing_pos - pos) < self._tolerance:
                return existing_id
        
        # Check nearby keys for boundary cases
        for dk in self._nearby_keys():
            nkey = (key[0]+dk[0], key[1]+dk[1], key[2]+dk[2])
            if nkey in self._pos_to_id:
                existing_id = self._pos_to_id[nkey]
                existing_pos = self._nodes[existing_id].position
                if np.linalg.norm(existing_pos - pos) < self._tolerance:
                    return existing_id
        
        # New node
        nid = self._next_node_id
        self._nodes[nid] = GraphNode(position=pos.copy(), node_id=nid, metadata=metadata)
        self._pos_to_id[key] = nid
        self._next_node_id += 1
        return nid
    
    def _pos_key(self, pos: np.ndarray) -> Tuple:
        """Rounded position key for fast lookup."""
        scale = 1.0 / max(self._tolerance, 1e-12)
        return tuple(int(round(pos[i] * scale)) for i in range(3))
    
    def _nearby_keys(self) -> List[Tuple]:
        """Offsets for checking neighboring grid cells."""
        return [(0,0,0), (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    
    # ---- Edge operations ----
    
    def add_edge(self, node_i: int, node_j: int, radius: float = 0.1,
                 material: Optional[Material] = None, segments: int = 4,
                 **metadata) -> int:
        """Add an edge between two nodes. Returns edge ID."""
        if node_i == node_j:
            return -1
        
        eid = self._next_edge_id
        self._edges[eid] = GraphEdge(
            node_i=node_i, node_j=node_j, edge_id=eid,
            radius=radius, material=material, segments=segments,
            metadata=metadata,
        )
        self._next_edge_id += 1
        return eid
    
    def add_edge_by_pos(self, p1: np.ndarray, p2: np.ndarray,
                        radius: float = 0.1, material: Optional[Material] = None,
                        segments: int = 4, **metadata) -> Tuple[int, int, int]:
        """Add edge by positions (auto-creates nodes). Returns (node_i, node_j, edge_id)."""
        n1 = self.add_node(p1)
        n2 = self.add_node(p2)
        eid = self.add_edge(n1, n2, radius=radius, material=material,
                           segments=segments, **metadata)
        return n1, n2, eid
    
    # ---- Properties ----
    
    @property
    def num_nodes(self) -> int:
        return len(self._nodes)
    
    @property
    def num_edges(self) -> int:
        return len(self._edges)
    
    @property
    def nodes(self) -> Dict[int, GraphNode]:
        return self._nodes
    
    @property
    def edges(self) -> Dict[int, GraphEdge]:
        return self._edges
    
    def node_positions(self) -> np.ndarray:
        """All node positions as (N, 3) array."""
        if not self._nodes:
            return np.zeros((0, 3))
        return np.array([n.position for n in self._nodes.values()])
    
    def degree(self, node_id: int) -> int:
        """Degree of a node."""
        return sum(1 for e in self._edges.values()
                  if e.node_i == node_id or e.node_j == node_id)
    
    def degree_distribution(self) -> Dict[int, int]:
        """Degree distribution as {degree: count}."""
        from collections import Counter
        degs = [self.degree(nid) for nid in self._nodes]
        return dict(Counter(degs))
    
    def is_connected(self) -> bool:
        """Check if the graph is fully connected."""
        if not self._nodes:
            return True
        adj = self._adjacency()
        visited = set()
        start = next(iter(self._nodes))
        queue = [start]
        while queue:
            n = queue.pop(0)
            if n in visited:
                continue
            visited.add(n)
            queue.extend(adj[n] - visited)
        return len(visited) == len(self._nodes)
    
    def num_components(self) -> int:
        """Number of connected components."""
        adj = self._adjacency()
        visited = set()
        count = 0
        for start in self._nodes:
            if start not in visited:
                count += 1
                queue = [start]
                while queue:
                    n = queue.pop(0)
                    if n in visited:
                        continue
                    visited.add(n)
                    queue.extend(adj[n] - visited)
        return count
    
    def _adjacency(self) -> Dict[int, Set[int]]:
        """Node adjacency list."""
        adj: Dict[int, Set[int]] = {nid: set() for nid in self._nodes}
        for e in self._edges.values():
            adj[e.node_i].add(e.node_j)
            adj[e.node_j].add(e.node_i)
        return adj
    
    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Bounding box (min, max)."""
        positions = self.node_positions()
        if len(positions) == 0:
            return np.zeros(3), np.zeros(3)
        return positions.min(axis=0), positions.max(axis=0)
    
    # ---- Conversion to FiberNetwork ----
    
    def to_network(
        self,
        material: Optional[Material] = None,
        box_size: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None,
        default_radius: float = 0.1,
        default_segments: int = 4,
    ) -> FiberNetwork:
        """Convert to FiberNetwork with proper crosslinks.
        
        Each edge becomes a Fiber. Shared nodes become Crosslinks.
        
        Parameters
        ----------
        material : Material, optional
            Default material for edges without one.
        box_size : ndarray, optional
            Explicit box size. If None, computed from node positions.
        metadata : dict, optional
            Network metadata.
        default_radius : float
            Radius for edges without explicit radius.
        default_segments : int
            Segments for edges without explicit segment count.
        
        Returns
        -------
        FiberNetwork
        """
        mat = material or Material(name="default")
        
        # Compute box size
        if box_size is not None:
            box = np.array(box_size, dtype=float)
        else:
            bb_min, bb_max = self.bounding_box()
            box = bb_max - bb_min
            if self.dimension == 2:
                box[2] = 0.0
            box = np.maximum(box, 1e-6)
        
        net = FiberNetwork(
            dimension=self.dimension,
            box_size=box,
            metadata=metadata or {},
        )
        
        # Map: node_id -> list of (fiber_id, parametric_position)
        node_to_fibers: Dict[int, List[Tuple[int, float]]] = {
            nid: [] for nid in self._nodes
        }
        
        for eid, edge in self._edges.items():
            p1 = self._nodes[edge.node_i].position
            p2 = self._nodes[edge.node_j].position
            dist = np.linalg.norm(p2 - p1)
            if dist < 1e-12:
                continue
            
            fid = net.num_fibers
            r = edge.radius if edge.radius > 0 else default_radius
            e_mat = edge.material or mat
            n_seg = max(edge.segments or default_segments,
                       max(4, int(dist / max(r * 2, 0.05))))
            
            fiber = Fiber.straight(
                p1, p2, radius=r, material=e_mat,
                fiber_id=fid, segments=n_seg,
            )
            net.add_fiber(fiber)
            
            node_to_fibers[edge.node_i].append((fid, 0.0))
            node_to_fibers[edge.node_j].append((fid, 1.0))
        
        # Create crosslinks at shared nodes
        for nid, fiber_list in node_to_fibers.items():
            if len(fiber_list) < 2:
                continue
            pos = self._nodes[nid].position
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
    
    # ---- Post-processing ----
    
    def ensure_connected(self, net: FiberNetwork, max_gap_factor: float = 3.0) -> FiberNetwork:
        """Post-process: if disconnected, bridge components.
        
        Only bridges if the graph had connectivity issues.
        """
        if not net.is_connected() if hasattr(net, 'is_connected') else False:
            pass
        
        from collections import defaultdict
        adj = defaultdict(set)
        for cl in net.crosslinks:
            adj[cl.fiber_i].add(cl.fiber_j)
            adj[cl.fiber_j].add(cl.fiber_i)
        
        visited = set()
        n_comp = 0
        for s in range(net.num_fibers):
            if s not in visited:
                n_comp += 1
                q = [s]
                while q:
                    n = q.pop(0)
                    if n in visited: continue
                    visited.add(n)
                    q.extend(adj[n] - visited)
        
        if n_comp > 1:
            max_gap = max_gap_factor * net.mean_fiber_length if net.mean_fiber_length > 0 else 50.0
            net.connect_components(max_gap=max_gap)
        
        return net


# ============================================================================
# Standardized Generator Parameters
# ============================================================================

@dataclass
class GeneratorParams:
    """Standardized parameters for all generators.
    
    Every generator should accept these core parameters, plus
    generator-specific parameters via **kwargs.
    
    Parameters
    ----------
    size : float
        Characteristic physical size (length of one side of bounding box).
    density : float or None
        Target volume/area fraction. If None, determined by geometry.
    resolution : tuple or int
        Grid resolution or number of elements.
    radius : float
        Fiber radius.
    material : Material or None
        Fiber material. If None, uses default.
    seed : int or None
        Random seed for reproducibility.
    """
    size: float = 10.0
    density: Optional[float] = None
    resolution: Any = None  # int or tuple
    radius: float = 0.1
    material: Optional[Material] = None
    seed: Optional[int] = None
    
    def get_material(self, name: str = "default") -> Material:
        return self.material or Material(name=name)
    
    def get_grid(self, default: Tuple = (5, 5)) -> Tuple:
        if self.resolution is None:
            return default
        if isinstance(self.resolution, int):
            return (self.resolution,) * len(default)
        return tuple(self.resolution)


def _ensure_connected_fallback(net: FiberNetwork, max_gap_factor: float = 3.0) -> FiberNetwork:
    """Convenience: bridge disconnected components if needed."""
    from collections import defaultdict
    adj = defaultdict(set)
    for cl in net.crosslinks:
        adj[cl.fiber_i].add(cl.fiber_j)
        adj[cl.fiber_j].add(cl.fiber_i)
    
    visited = set()
    n_comp = 0
    for s in range(net.num_fibers):
        if s not in visited:
            n_comp += 1
            q = [s]
            while q:
                n = q.pop(0)
                if n in visited: continue
                visited.add(n)
                q.extend(adj[n] - visited)
    
    if n_comp > 1:
        max_gap = max_gap_factor * net.mean_fiber_length if net.mean_fiber_length > 0 else 50.0
        net.connect_components(max_gap=max_gap)
    
    return net
