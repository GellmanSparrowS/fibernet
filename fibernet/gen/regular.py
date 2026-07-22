"""
Regular fiber network generators.

Implements structured fiber networks based on square/hexagonal/triangular
unit cells with optional perturbations and tiling.

Based on P1_Gen_dataset_regular_net.py (SquareGraphGenerator).
"""

import random
from typing import List, Tuple, Optional

import numpy as np

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class RegularNetworkGenerator:
    """Generate regular fiber networks from unit cells.
    
    Creates a square frame graph with optional midpoint perturbations
    on each edge, then tiles it into an N×N array.
    
    Parameters
    ----------
    side_length : float
        Length of the square unit cell side.
    num_points_per_side : int
        Number of intermediate points on each edge.
    perturbations : list of (dx, dy), optional
        Perturbation offsets for each midpoint. Range [-0.5, 0.5].
        If None, no perturbation.
    tiling : int
        Number of tiles per dimension (creates tiling × tiling array).
    scale_to_unit : bool
        If True, scale the final graph to fit in [0, 1] × [0, 1].
    
    Examples
    --------
    >>> gen = RegularNetworkGenerator(
    ...     side_length=10, num_points_per_side=2,
    ...     perturbations=[(0.1, -0.2), (0.3, 0.1), (-0.15, 0.25)],
    ...     tiling=3,
    ... )
    >>> G = gen.generate()
    >>> print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    """
    
    def __init__(
        self,
        side_length: float = 10.0,
        num_points_per_side: int = 1,
        perturbations: Optional[List[Tuple[float, float]]] = None,
        tiling: int = 3,
        scale_to_unit: bool = True,
    ):
        if not HAS_NETWORKX:
            raise ImportError("networkx is required: pip install networkx")
        
        self.side_length = side_length
        self.num_points_per_side = num_points_per_side
        self.perturbations = perturbations or []
        self.tiling = tiling
        self.scale_to_unit = scale_to_unit
    
    def generate(self) -> "nx.Graph":
        """Generate the regular fiber network.
        
        Returns
        -------
        nx.Graph
            Graph with 'pos' attribute on nodes.
        """
        # Step 1: Create base square graph
        G = self._generate_square_graph()
        
        # Step 2: Apply perturbations
        for idx, (dx, dy) in enumerate(self.perturbations):
            G = self._move_midpoints(G, idx + 1, dx, dy)
        
        # Step 3: Tile into array
        G = self._scale_and_tile(G)
        
        # Step 4: Optionally scale to unit square
        if self.scale_to_unit:
            G = self._scale_to_unit_square(G)
        
        return G
    
    def _generate_square_graph(self) -> "nx.Graph":
        """Create a square frame with midpoints on each edge."""
        G = nx.Graph()
        s = self.side_length
        
        vertices = {
            'A': (0.0, 0.0), 'B': (s, 0.0),
            'C': (s, s), 'D': (0.0, s)
        }
        
        for vname, pos in vertices.items():
            G.add_node(vname, pos=pos)
        
        edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]
        
        for start, end in edges:
            G.add_edge(start, end)
            sp = np.array(vertices[start])
            ep = np.array(vertices[end])
            
            for j in range(1, self.num_points_per_side + 1):
                t = j / (self.num_points_per_side + 1)
                pt = (1 - t) * sp + t * ep
                name = f"{start}{end}{j}"
                G.add_node(name, pos=tuple(pt))
                
                if j == 1:
                    G.add_edge(start, name)
                if j == self.num_points_per_side:
                    G.add_edge(name, end)
                if j > 1:
                    prev = f"{start}{end}{j-1}"
                    G.add_edge(prev, name)
        
        return G
    
    def _move_midpoints(self, G: "nx.Graph", num: int, 
                        dx: float, dy: float) -> "nx.Graph":
        """Perturb midpoint nodes on each edge of the square.
        
        Applies the same perturbation pattern as P1's move_AB:
        - AB edge: (dx, dy)
        - BC edge: (-dy, dx)
        - CD edge: (-dx, -dy)  
        - DA edge: (dy, -dx)
        """
        new_G = G.copy()
        s = self.side_length
        dx_scaled = dx * s
        dy_scaled = dy * s
        
        offsets = {
            f'AB{num}': (dx_scaled, dy_scaled),
            f'BC{num}': (-dy_scaled, dx_scaled),
            f'CD{num}': (-dx_scaled, -dy_scaled),
            f'DA{num}': (dy_scaled, -dx_scaled),
        }
        
        for node, (odx, ody) in offsets.items():
            if node in new_G.nodes:
                cx, cy = new_G.nodes[node]['pos']
                new_G.nodes[node]['pos'] = (cx + odx, cy + ody)
        
        # Reconnect edges in order
        new_G.remove_edges_from(list(new_G.edges))
        node_list = sorted(new_G.nodes)
        for i in range(len(node_list)):
            new_G.add_edge(node_list[i], node_list[(i + 1) % len(node_list)])
        
        return new_G
    
    def _scale_and_tile(self, G: "nx.Graph") -> "nx.Graph":
        """Tile the unit cell into a tiling × tiling array."""
        tiled = nx.Graph()
        s = self.side_length
        n = self.tiling
        
        for ti in range(n):
            for tj in range(n):
                for node in G.nodes:
                    pos = G.nodes[node]['pos']
                    new_name = f"{node}_{ti}_{tj}"
                    new_pos = (pos[0] + ti * s, pos[1] + tj * s)
                    tiled.add_node(new_name, pos=new_pos)
        
        # Internal edges
        for ti in range(n):
            for tj in range(n):
                for u, v in G.edges():
                    tiled.add_edge(f"{u}_{ti}_{tj}", f"{v}_{ti}_{tj}")
        
        return tiled
    
    def _scale_to_unit_square(self, G: "nx.Graph") -> "nx.Graph":
        """Scale all positions to fit in [0, 1] × [0, 1]."""
        pos = nx.get_node_attributes(G, 'pos')
        if not pos:
            return G
        
        coords = np.array(list(pos.values()))
        min_xy = coords.min(axis=0)
        max_xy = coords.max(axis=0)
        span = max(max_xy - min_xy)
        if span == 0:
            span = 1.0
        
        scale = 1.0 / span
        for node in G.nodes():
            p = np.array(pos[node])
            G.nodes[node]['pos'] = tuple((p - min_xy) * scale)
        
        return G
    

    def to_fiber_network(self, radius: float = 0.1, material=None):
        """Convert the nx.Graph to FiberNetwork using FiberGraph for node merging.
        
        Parameters
        ----------
        radius : float
            Fiber radius.
        material : Material, optional
            Fiber material.
            
        Returns
        -------
        FiberNetwork
            Connected fiber network.
        """
        from fibernet.gen._graph_builder import FiberGraph
        from fibernet.core.material import Material
        import numpy as np
        
        G = self.generate()
        
        if material is None:
            material = Material()
        
        # Use FiberGraph for proper node merging
        fg = FiberGraph(dimension=2, tolerance=1e-6)
        
        # Add nodes (FiberGraph automatically merges nodes at same position)
        node_map = {}
        for node in G.nodes():
            pos = np.array(G.nodes[node]['pos'])
            node_id = fg.add_node(pos, original_node=node)
            node_map[node] = node_id
        
        # Add edges
        for u, v in G.edges():
            fg.add_edge(node_map[u], node_map[v], radius=radius, material=material)
        
        # Convert to FiberNetwork
        net = fg.to_network()
        
        return net


    @staticmethod
    def random_perturbations(num_points: int = 3, 
                             rng: Optional[np.random.Generator] = None) -> List[Tuple[float, float]]:
        """Generate random perturbation offsets.
        
        Parameters
        ----------
        num_points : int
            Number of perturbation pairs to generate.
        rng : np.random.Generator, optional
            Random number generator.
        
        Returns
        -------
        list of (float, float)
            Random offsets in [-0.5, 0.5].
        """
        if rng is None:
            rng = np.random.default_rng()
        return [
            (round(rng.uniform(-0.5, 0.5), 2), round(rng.uniform(-0.5, 0.5), 2))
            for _ in range(num_points)
        ]
