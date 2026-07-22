"""
ZigZag lattice generator.

Creates zigzag fiber networks by mirroring and tiling a base zigzag pattern.
Based on pattern.ipynb (ZigZagLattice).
"""

from typing import List, Tuple, Optional

import numpy as np

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class ZigZagGenerator:
    """Generate zigzag fiber networks.
    
    Creates a zigzag pattern from base points, then tiles it with
    optional mirroring to create complex periodic structures.
    
    Parameters
    ----------
    base_points : list of (float, float)
        Control points defining one zigzag period.
        Example: [(0, 31.7), (75, 75), (31.7, 0), (161.6, 75), (118.3, 0), (193, 43.3)]
    n_cols : int
        Number of columns in the tiled array.
    n_rows : int
        Number of rows in the tiled array.
    mirror_x : bool
        If True, mirror every other column horizontally.
    mirror_y : bool
        If True, mirror every other row vertically.
    
    Examples
    --------
    >>> gen = ZigZagGenerator(
    ...     base_points=[(0, 31.7), (75, 75), (31.7, 0)],
    ...     n_cols=4, n_rows=10,
    ...     mirror_x=True, mirror_y=True,
    ... )
    >>> G = gen.generate()
    >>> print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    """
    
    def __init__(
        self,
        base_points: Optional[List[Tuple[float, float]]] = None,
        n_cols: int = 4,
        n_rows: int = 2,
        mirror_x: bool = True,
        mirror_y: bool = True,
    ):
        if not HAS_NETWORKX:
            raise ImportError("networkx is required: pip install networkx")
        
        # Default base points from pattern.ipynb
        if base_points is None:
            self.base_points = [
                (0.0, 31.7),
                (75.0, 75.0),
                (31.7, 0.0),
                (161.6, 75.0),
                (118.3, 0.0),
                (193.0, 43.3),
            ]
        else:
            self.base_points = base_points
        
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.mirror_x = mirror_x
        self.mirror_y = mirror_y
        
        # Compute bounding box
        xs, ys = zip(*self.base_points)
        self.width = max(xs) - min(xs)
        self.height = max(ys) - min(ys)
        
        # Tolerance for deduplication
        self.eps = 1e-6
    
    def generate(self) -> "nx.Graph":
        """Generate the zigzag fiber network.
        
        Returns
        -------
        nx.Graph
            Graph with 'pos' attribute on nodes.
        """
        G = nx.Graph()
        node_map = {}  # (x, y) -> node_id
        
        for j in range(self.n_rows):
            for i in range(self.n_cols):
                # Transform base points for this tile
                transformed = [
                    self._transform_point(x, y, i, j)
                    for x, y in self.base_points
                ]
                
                # Create edges between consecutive points
                for k in range(len(transformed) - 1):
                    x1, y1 = transformed[k]
                    x2, y2 = transformed[k + 1]
                    
                    # Deduplicate nodes by position
                    key1 = self._pos_key(x1, y1)
                    key2 = self._pos_key(x2, y2)
                    
                    if key1 not in node_map:
                        node_id = len(node_map)
                        node_map[key1] = node_id
                        G.add_node(node_id, pos=(x1, y1, 0.0))
                    
                    if key2 not in node_map:
                        node_id = len(node_map)
                        node_map[key2] = node_id
                        G.add_node(node_id, pos=(x2, y2, 0.0))
                    
                    n1 = node_map[key1]
                    n2 = node_map[key2]
                    
                    if n1 != n2:
                        G.add_edge(n1, n2)
        
        return G
    
    def _transform_point(self, x: float, y: float, 
                         col: int, row: int) -> Tuple[float, float]:
        """Apply mirror and translation to a point."""
        # Mirror
        if self.mirror_x and col % 2 == 1:
            x = self.width - x
        if self.mirror_y and row % 2 == 1:
            y = self.height - y
        
        # Translate
        x += col * self.width
        y += row * self.height
        
        return x, y
    
    def _pos_key(self, x: float, y: float) -> Tuple[int, int]:
        """Create a hash key for position deduplication."""
        return (round(x / self.eps), round(y / self.eps))
    

    def to_fiber_network(self, radius: float = 0.1, material=None):
        """Convert the nx.Graph to a FiberNetwork.
        
        Parameters
        ----------
        radius : float
            Fiber radius.
        material : Material, optional
            Fiber material.
        
        Returns
        -------
        FiberNetwork
            Connected fiber network ready for simulation.
        """
        from fibernet.core.network import FiberNetwork, Crosslink
        from fibernet.core.fiber import Fiber
        from fibernet.core.material import Material
        
        G = self.generate()
        mat = material or Material(name="zigzag")
        
        net = FiberNetwork(dimension=2, metadata={"generator": "zigzag"})
        
        node_to_fibers = {}
        
        for edge in G.edges():
            p1 = G.nodes[edge[0]]['pos']
            p2 = G.nodes[edge[1]]['pos']
            
            if len(p1) == 2:
                p1 = (*p1, 0.0)
            if len(p2) == 2:
                p2 = (*p2, 0.0)
            
            fiber = Fiber.straight(
                np.array(p1), np.array(p2),
                radius=radius, material=mat,
                fiber_id=net.num_fibers,
            )
            net.add_fiber(fiber)
            
            for node in edge:
                if node not in node_to_fibers:
                    node_to_fibers[node] = []
                node_to_fibers[node].append(net.num_fibers - 1)
        
        # Add crosslinks at shared nodes
        for node, fiber_ids in node_to_fibers.items():
            if len(fiber_ids) < 2:
                continue
            pos = G.nodes[node]['pos']
            if len(pos) == 2:
                pos = (*pos, 0.0)
            for i, fi in enumerate(fiber_ids):
                for fj in fiber_ids[i+1:]:
                    net.add_crosslink(Crosslink(
                        fiber_i=fi, fiber_j=fj,
                        param_i=0.5, param_j=0.5,
                        position=np.array(pos),
                        crosslink_type="welded",
                    ))
        
        return net


    @staticmethod
    def simple_zigzag(amplitude: float = 50.0, 
                      wavelength: float = 100.0,
                      n_periods: int = 3) -> "ZigZagGenerator":
        """Create a simple zigzag with triangular teeth.
        
        Parameters
        ----------
        amplitude : float
            Height of zigzag peaks.
        wavelength : float
            Horizontal distance per period.
        n_periods : int
            Number of zigzag periods.
        
        Returns
        -------
        ZigZagGenerator
            Configured generator.
        """
        points = []
        for i in range(n_periods + 1):
            x = i * wavelength
            points.append((x, 0.0))
            if i < n_periods:
                points.append((x + wavelength / 2, amplitude))
        
        return ZigZagGenerator(base_points=points, n_cols=1, n_rows=1)
