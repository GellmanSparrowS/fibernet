"""
Fractal and Self-Similar Network Generator Module

Provides generators for fractal and self-similar fiber networks:
- Sierpinski triangle networks
- Koch curve networks
- Fractal tree networks
- Hilbert curve networks

All generators use a node-edge graph approach ensuring proper connectivity
through shared nodes at junction points.

References:
- Mandelbrot, B.B., "The Fractal Geometry of Nature", Freeman, 1982
- Falconer, K., "Fractal Geometry: Mathematical Foundations and Applications", Wiley, 2014
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Set

from fibernet.core.network import FiberNetwork, Crosslink
from fibernet.core.fiber import Fiber
from fibernet.core.material import Material


class _FractalGraph:
    """Lightweight node-edge graph for fractal construction."""
    
    def __init__(self, tolerance: float = 1e-6):
        self.nodes: Dict[int, np.ndarray] = {}
        self.edges: Set[Tuple[int, int]] = set()
        self._next_id = 0
        self._tolerance = tolerance
    
    def add_node(self, pos: np.ndarray) -> int:
        pos = np.asarray(pos, dtype=float)
        for nid, npos in self.nodes.items():
            if np.linalg.norm(npos - pos) < self._tolerance:
                return nid
        nid = self._next_id
        self.nodes[nid] = pos.copy()
        self._next_id += 1
        return nid
    
    def add_edge(self, n1: int, n2: int):
        if n1 != n2:
            self.edges.add((min(n1, n2), max(n1, n2)))
    
    def to_network(self, radius=0.1, material=None, dimension=2,
                   metadata=None, box_size=None, variable_radius=None) -> FiberNetwork:
        """Convert to FiberNetwork with crosslinks at shared nodes."""
        mat = material or Material(name="fractal")
        
        if box_size is not None:
            box = np.array(box_size, dtype=float)
        else:
            all_pos = np.array(list(self.nodes.values())) if self.nodes else np.zeros((1, 3))
            bb = all_pos.max(axis=0) - all_pos.min(axis=0)
            box = bb if dimension == 3 else np.array([bb[0], bb[1], 0.0])
        
        net = FiberNetwork(dimension=dimension, box_size=box, metadata=metadata or {})
        
        node_to_fibers: Dict[int, List[Tuple[int, float]]] = {nid: [] for nid in self.nodes}
        
        for edge in self.edges:
            n1, n2 = edge
            p1, p2 = self.nodes[n1], self.nodes[n2]
            dist = np.linalg.norm(p2 - p1)
            if dist < 1e-12:
                continue
            
            fid = net.num_fibers
            r = radius
            if variable_radius is not None:
                r = variable_radius(edge)
            
            n_seg = max(4, int(dist / max(r * 2, 0.1)))
            fiber = Fiber.straight(p1, p2, radius=r, material=mat,
                                  fiber_id=fid, segments=n_seg)
            net.add_fiber(fiber)
            
            node_to_fibers[n1].append((fid, 0.0))
            node_to_fibers[n2].append((fid, 1.0))
        
        for nid, fiber_list in node_to_fibers.items():
            if len(fiber_list) < 2:
                continue
            pos = self.nodes[nid]
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


def sierpinski_triangle(
    iterations: int = 3,
    size: float = 10.0,
    origin: Tuple[float, float] = (0.0, 0.0),
    radius: float = 0.1,
    material: Optional[Material] = None,
) -> FiberNetwork:
    """Generate a Sierpinski triangle fiber network.
    
    Parameters
    ----------
    iterations : int
        Number of iterations (recursion depth).
    size : float
        Initial triangle side length.
    origin : tuple
        (x, y) position of bottom-left corner.
    radius : float
        Fiber radius.
    material : Material, optional
        Fiber material.
    
    Returns
    -------
    FiberNetwork with proper connectivity at shared vertices.
    """
    mat = material or Material(name="sierpinski")
    graph = _FractalGraph(tolerance=size * 1e-6)
    
    h = size * np.sqrt(3) / 2
    p1 = np.array([origin[0], origin[1], 0.0])
    p2 = np.array([origin[0] + size, origin[1], 0.0])
    p3 = np.array([origin[0] + size/2, origin[1] + h, 0.0])
    
    def add_triangle(p1, p2, p3, depth):
        if depth == 0:
            n1 = graph.add_node(p1)
            n2 = graph.add_node(p2)
            n3 = graph.add_node(p3)
            graph.add_edge(n1, n2)
            graph.add_edge(n2, n3)
            graph.add_edge(n3, n1)
        else:
            m1 = (p1 + p2) / 2
            m2 = (p2 + p3) / 2
            m3 = (p3 + p1) / 2
            add_triangle(p1, m1, m3, depth - 1)
            add_triangle(m1, p2, m2, depth - 1)
            add_triangle(m3, m2, p3, depth - 1)
    
    add_triangle(p1, p2, p3, iterations)
    
    return graph.to_network(
        radius=radius, material=mat, dimension=2,
        metadata={"generator": "sierpinski_triangle", "iterations": iterations, "size": size},
    )


def koch_curve(
    iterations: int = 3,
    start: Tuple[float, float] = (0.0, 0.0),
    end: Tuple[float, float] = (10.0, 0.0),
    radius: float = 0.1,
    material: Optional[Material] = None,
) -> FiberNetwork:
    """Generate a Koch curve fiber network.
    
    The Koch curve is a continuous path, so all segments are naturally connected.
    """
    mat = material or Material(name="koch")
    graph = _FractalGraph(tolerance=1e-8)
    
    p_start = np.array([start[0], start[1], 0.0])
    p_end = np.array([end[0], end[1], 0.0])
    
    def koch_points(p1, p2, depth):
        if depth == 0:
            return [p1, p2]
        
        v = p2 - p1
        p3 = p1 + v / 3
        p5 = p1 + 2 * v / 3
        
        angle = np.pi / 3
        rot = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        p4 = p3 + rot @ (v / 3)
        
        points = []
        points.extend(koch_points(p1, p3, depth - 1)[:-1])
        points.extend(koch_points(p3, p4, depth - 1)[:-1])
        points.extend(koch_points(p4, p5, depth - 1)[:-1])
        points.extend(koch_points(p5, p2, depth - 1))
        return points
    
    points = koch_points(p_start, p_end, iterations)
    
    prev_nid = graph.add_node(points[0])
    for i in range(1, len(points)):
        curr_nid = graph.add_node(points[i])
        graph.add_edge(prev_nid, curr_nid)
        prev_nid = curr_nid
    
    return graph.to_network(
        radius=radius, material=mat, dimension=2,
        metadata={"generator": "koch_curve", "iterations": iterations},
    )


def fractal_tree(
    iterations: int = 5,
    trunk_length: float = 10.0,
    branch_ratio: float = 0.7,
    branch_angle: float = np.pi / 6,
    origin: Tuple[float, float] = (0.0, 0.0),
    radius: float = 0.1,
    material: Optional[Material] = None,
) -> FiberNetwork:
    """Generate a fractal tree fiber network.
    
    Each branch shares its starting point with its parent's endpoint,
    ensuring full connectivity through the tree hierarchy.
    """
    mat = material or Material(name="fractal_tree")
    graph = _FractalGraph(tolerance=trunk_length * 1e-6)
    
    p_start = np.array([origin[0], origin[1], 0.0])
    
    def add_branch(start, angle, length, depth, current_radius):
        if depth == 0:
            return
        
        end = start + length * np.array([np.cos(angle), np.sin(angle), 0.0])
        
        n_start = graph.add_node(start)
        n_end = graph.add_node(end)
        graph.add_edge(n_start, n_end)
        
        child_length = length * branch_ratio
        add_branch(end, angle + branch_angle, child_length, depth - 1, current_radius * 0.8)
        add_branch(end, angle - branch_angle, child_length, depth - 1, current_radius * 0.8)
    
    add_branch(p_start, np.pi / 2, trunk_length, iterations, radius)
    
    # Variable radius: thinner branches at higher depth
    edge_depths = {}
    def compute_depths(start, angle, length, depth):
        if depth == 0:
            return
        end = start + length * np.array([np.cos(angle), np.sin(angle), 0.0])
        n_start = graph.add_node(start)
        n_end = graph.add_node(end)
        edge_key = (min(n_start, n_end), max(n_start, n_end))
        edge_depths[edge_key] = iterations - depth
        
        child_length = length * branch_ratio
        compute_depths(end, angle + branch_angle, child_length, depth - 1)
        compute_depths(end, angle - branch_angle, child_length, depth - 1)
    
    compute_depths(p_start, np.pi / 2, trunk_length, iterations)
    
    def variable_radius(edge):
        d = edge_depths.get(edge, 0)
        return radius * (0.8 ** d)
    
    return graph.to_network(
        radius=radius, material=mat, dimension=2,
        metadata={"generator": "fractal_tree", "iterations": iterations},
        variable_radius=variable_radius,
    )


def hilbert_curve(
    order: int = 3,
    size: float = 10.0,
    origin: Tuple[float, float] = (0.0, 0.0),
    radius: float = 0.1,
    material: Optional[Material] = None,
) -> FiberNetwork:
    """Generate a Hilbert curve fiber network.
    
    The Hilbert curve is a single continuous path, ensuring full connectivity.
    """
    mat = material or Material(name="hilbert")
    graph = _FractalGraph(tolerance=size * 1e-6)
    
    n = 2 ** order
    points = []
    
    for i in range(n * n):
        x, y = 0, 0
        t = i
        
        for s in range(order):
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            
            if ry == 0:
                if rx == 1:
                    x = (1 << s) - 1 - x
                    y = (1 << s) - 1 - y
                x, y = y, x
            
            x += rx << s
            y += ry << s
            t //= 4
        
        px = origin[0] + size * x / (n - 1)
        py = origin[1] + size * y / (n - 1)
        points.append(np.array([px, py, 0.0]))
    
    prev_nid = graph.add_node(points[0])
    for i in range(1, len(points)):
        curr_nid = graph.add_node(points[i])
        graph.add_edge(prev_nid, curr_nid)
        prev_nid = curr_nid
    
    return graph.to_network(
        radius=radius, material=mat, dimension=2,
        metadata={"generator": "hilbert_curve", "order": order, "size": size},
        box_size=np.array([size, size, 0.0]),
    )
