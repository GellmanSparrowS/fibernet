"""
FiberNetwork - core data structure for a collection of fibers with connectivity.

Manages fiber collections, crosslinks, contacts, periodicity, and provides
methods for I/O, querying, and basic analysis.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, Union
import json
import warnings

from fibernet.core.fiber import Fiber, CrossSection
from fibernet.core.material import Material


@dataclass
class Crosslink:
    """Represents a crosslink (junction) between two fibers.
    
    Parameters
    ----------
    fiber_i : int
        Index of first fiber.
    fiber_j : int
        Index of second fiber.
    param_i : float
        Parametric position on fiber_i (0 to 1).
    param_j : float
        Parametric position on fiber_j (0 to 1).
    position : np.ndarray
        3D coordinates of the crosslink point.
    crosslink_type : str
        Type: 'welded', 'pinned', 'sliding', 'bonded'.
    strength : float
        Crosslink strength in N (optional).
    stiffness : float
        Crosslink stiffness in N/m (optional).
    """
    fiber_i: int
    fiber_j: int
    param_i: float
    param_j: float
    position: np.ndarray
    crosslink_type: str = "welded"
    strength: float = float('inf')
    stiffness: float = float('inf')


@dataclass
class FiberNetwork:
    """A network of fibers with crosslinks and contacts.

    Parameters
    ----------
    fibers : list of Fiber
        The fibers in the network.
    crosslinks : list of Crosslink
        Crosslinks between fibers.
    box_size : np.ndarray or None
        Bounding box dimensions [Lx, Ly, Lz] for periodic systems.
    periodic : bool
        Whether the system uses periodic boundary conditions.
    dimension : int
        2 for 2D networks, 3 for 3D networks.
    metadata : dict
        Arbitrary metadata (generator info, parameters, etc.).
    """
    fibers: List[Fiber] = field(default_factory=list)
    crosslinks: List[Crosslink] = field(default_factory=list)
    box_size: Optional[np.ndarray] = None
    periodic: bool = False
    dimension: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_fibers(self) -> int:
        return len(self.fibers)

    @property
    def num_crosslinks(self) -> int:
        return len(self.crosslinks)

    @property
    def total_length(self) -> float:
        """Total length of all fibers."""
        return sum(f.length for f in self.fibers)

    @property
    def total_volume(self) -> float:
        """Total volume of fiber material."""
        return sum(f.length * f.cross_section_area for f in self.fibers)

    @property
    def mean_fiber_length(self) -> float:
        if not self.fibers:
            return 0.0
        return self.total_length / self.num_fibers

    @property
    def mean_radius(self) -> float:
        if not self.fibers:
            return 0.0
        return np.mean([f.radius for f in self.fibers])

    def density(self) -> float:
        """Volume fraction (solid volume / bounding box volume)."""
        vol = self.total_volume
        if self.box_size is not None:
            box_vol = float(np.prod(self.box_size))
        else:
            bb_min, bb_max = self.bounding_box()
            bb_max = np.maximum(bb_max, bb_min + 1e-12)
            box_vol = float(np.prod(bb_max - bb_min))
        return vol / box_vol if box_vol > 0 else 0.0

    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Overall bounding box (min_corner, max_corner)."""
        if not self.fibers:
            return np.zeros(3), np.zeros(3)
        all_pts = np.vstack([f.centerline for f in self.fibers])
        return all_pts.min(axis=0), all_pts.max(axis=0)

    def add_fiber(self, fiber: Fiber) -> int:
        """Add a fiber to the network. Returns fiber index."""
        fiber.fiber_id = len(self.fibers)
        self.fibers.append(fiber)
        return fiber.fiber_id

    def add_crosslink(self, crosslink: Crosslink):
        """Add a crosslink to the network."""
        self.crosslinks.append(crosslink)

    def remove_fiber(self, index: int):
        """Remove fiber at given index and associated crosslinks."""
        if 0 <= index < len(self.fibers):
            self.fibers.pop(index)
            self.crosslinks = [
                cl for cl in self.crosslinks
                if cl.fiber_i != index and cl.fiber_j != index
            ]
            for i, f in enumerate(self.fibers):
                f.fiber_id = i

    def get_fiber(self, index: int) -> Fiber:
        return self.fibers[index]

    def fiber_orientations(self) -> np.ndarray:
        """Get orientation vectors (direction) for all fibers."""
        if not self.fibers:
            return np.array([])
        return np.array([f.direction for f in self.fibers])

    def fiber_lengths(self) -> np.ndarray:
        return np.array([f.length for f in self.fibers])

    def connectivity_matrix(self) -> np.ndarray:
        """Sparse connectivity matrix (NxN) where N is number of fibers."""
        n = self.num_fibers
        conn = np.zeros((n, n), dtype=int)
        for cl in self.crosslinks:
            conn[cl.fiber_i, cl.fiber_j] += 1
            conn[cl.fiber_j, cl.fiber_i] += 1
        return conn

    def degree_distribution(self) -> np.ndarray:
        """Number of crosslinks per fiber."""
        n = self.num_fibers
        degrees = np.zeros(n, dtype=int)
        for cl in self.crosslinks:
            degrees[cl.fiber_i] += 1
            degrees[cl.fiber_j] += 1
        return degrees

    def detect_contacts(self, threshold: float = None, tree_leafsize: int = 16) -> List[Crosslink]:
        """Detect potential contacts (near-intersections) between fibers.
        
        Parameters
        ----------
        threshold : float
            Distance threshold for contact detection. Defaults to 2 * mean_radius.
        """
        if threshold is None:
            threshold = 2.0 * self.mean_radius if self.fibers else 1.0
        
        if not self.fibers:
            return []
        
        all_points = []
        point_to_fiber = []
        point_to_param = []
        
        for i, f in enumerate(self.fibers):
            for j, pt in enumerate(f.centerline):
                all_points.append(pt)
                point_to_fiber.append(i)
                point_to_param.append(j / max(len(f.centerline) - 1, 1))
        
        if len(all_points) < 2:
            return []
        
        all_points = np.array(all_points, dtype=np.float64)
        
        # Validate points
        if not np.all(np.isfinite(all_points)):
            return []
        
        # Use cKDTree for large networks, brute force for small ones
        # (cKDTree can segfault on some platforms with small datasets)
        if len(all_points) > 5000:
            try:
                tree = cKDTree(all_points, leafsize=tree_leafsize)
                pairs = tree.query_pairs(threshold)
            except Exception:
                pairs = set()
        else:
            # Brute force O(n²) for small networks
            pairs = set()
            thresh_sq = threshold * threshold
            for pi in range(len(all_points)):
                for pj in range(pi + 1, len(all_points)):
                    dist_sq = np.sum((all_points[pi] - all_points[pj]) ** 2)
                    if dist_sq <= thresh_sq:
                        pairs.add((pi, pj))
        
        contacts = []
        seen = set()
        for (pi, pj) in pairs:
            fi, fj = point_to_fiber[pi], point_to_fiber[pj]
            if fi == fj:
                continue
            key = (min(fi, fj), max(fi, fj))
            if key in seen:
                continue
            seen.add(key)
            mid_pt = 0.5 * (all_points[pi] + all_points[pj])
            contacts.append(Crosslink(
                fiber_i=fi, fiber_j=fj,
                param_i=point_to_param[pi], param_j=point_to_param[pj],
                position=mid_pt,
                crosslink_type="contact",
            ))
        
        return contacts

    def auto_crosslink(self, threshold: float = None, crosslink_type: str = "welded"):
        """Automatically detect and add crosslinks based on proximity."""
        contacts = self.detect_contacts(threshold)
        for c in contacts:
            c.crosslink_type = crosslink_type
            self.add_crosslink(c)

    def to_networkx(self):
        """Convert to a NetworkX graph for topology analysis."""
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx is required. Install with: pip install networkx")
        G = nx.Graph()
        for i, f in enumerate(self.fibers):
            G.add_node(i, fiber=f, length=f.length, orientation=f.direction.tolist())
        for cl in self.crosslinks:
            G.add_edge(
                cl.fiber_i, cl.fiber_j,
                crosslink_type=cl.crosslink_type,
                position=cl.position.tolist(),
            )
        return G

    def to_dict(self) -> Dict[str, Any]:
        """Serialize network to dictionary."""
        return {
            "version": "0.1.0",
            "dimension": self.dimension,
            "periodic": self.periodic,
            "box_size": self.box_size.tolist() if self.box_size is not None else None,
            "metadata": self.metadata,
            "fibers": [f.to_dict() for f in self.fibers],
            "crosslinks": [
                {
                    "fiber_i": cl.fiber_i,
                    "fiber_j": cl.fiber_j,
                    "param_i": cl.param_i,
                    "param_j": cl.param_j,
                    "position": cl.position.tolist(),
                    "crosslink_type": cl.crosslink_type,
                    "strength": cl.strength,
                    "stiffness": cl.stiffness,
                }
                for cl in self.crosslinks
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FiberNetwork":
        """Create FiberNetwork from dictionary."""
        net = cls(
            dimension=data.get("dimension", 3),
            periodic=data.get("periodic", False),
            box_size=np.array(data["box_size"]) if data.get("box_size") else None,
            metadata=data.get("metadata", {}),
        )
        for fd in data.get("fibers", []):
            net.add_fiber(Fiber.from_dict(fd))
        for cd in data.get("crosslinks", []):
            net.add_crosslink(Crosslink(
                fiber_i=cd["fiber_i"], fiber_j=cd["fiber_j"],
                param_i=cd["param_i"], param_j=cd["param_j"],
                position=np.array(cd["position"]),
                crosslink_type=cd.get("crosslink_type", "welded"),
                strength=cd.get("strength", float('inf')),
                stiffness=cd.get("stiffness", float('inf')),
            ))
        return net

    def save_json(self, filepath: str):
        """Save network to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, filepath: str) -> "FiberNetwork":
        """Load network from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_hdf5(self, filepath: str):
        """Save network to HDF5 file for large-scale data."""
        import h5py
        with h5py.File(filepath, 'w') as hf:
            hf.attrs['version'] = '0.1.0'
            hf.attrs['dimension'] = self.dimension
            hf.attrs['periodic'] = self.periodic
            if self.box_size is not None:
                hf.attrs['box_size'] = self.box_size
            hf.attrs['metadata'] = json.dumps(self.metadata)
            
            fiber_grp = hf.create_group('fibers')
            for i, f in enumerate(self.fibers):
                fg = fiber_grp.create_group(f'fiber_{i}')
                fg.create_dataset('centerline', data=f.centerline)
                fg.attrs['radius'] = f.radius
                fg.attrs['fiber_id'] = f.fiber_id
                fg.attrs['cross_section'] = f.cross_section.value
                fg.attrs['segments'] = f.segments
                fg.attrs['material'] = json.dumps(f.material.to_dict())
            
            cl_grp = hf.create_group('crosslinks')
            for i, cl in enumerate(self.crosslinks):
                cg = cl_grp.create_group(f'cl_{i}')
                cg.attrs['fiber_i'] = cl.fiber_i
                cg.attrs['fiber_j'] = cl.fiber_j
                cg.attrs['param_i'] = cl.param_i
                cg.attrs['param_j'] = cl.param_j
                cg.create_dataset('position', data=cl.position)
                cg.attrs['crosslink_type'] = cl.crosslink_type

    def summary(self) -> str:
        """Print a summary of the network."""
        lines = [
            f"FiberNetwork Summary",
            f"  Fibers:      {self.num_fibers}",
            f"  Crosslinks:  {self.num_crosslinks}",
            f"  Dimension:   {self.dimension}D",
            f"  Total Length: {self.total_length:.3f}",
            f"  Mean Length:  {self.mean_fiber_length:.3f}",
            f"  Mean Radius:  {self.mean_radius:.4f}",
            f"  Volume Frac:  {self.density():.4f}",
            f"  Periodic:     {self.periodic}",
        ]
        if self.box_size is not None:
            lines.append(f"  Box Size:     {self.box_size}")
        bb_min, bb_max = self.bounding_box()
        lines.append(f"  BBox Min:     {bb_min}")
        lines.append(f"  BBox Max:     {bb_max}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"FiberNetwork(fibers={self.num_fibers}, crosslinks={self.num_crosslinks}, "
            f"dim={self.dimension}D, L_total={self.total_length:.2f})"
        )

    def describe(self) -> str:
        """Return a statistical summary of the network.

        Returns
        -------
        str
            Multi-line string with network statistics.
        """
        import numpy as np

        lines = [
            f"FiberNetwork Summary",
            f"{'='*40}",
            f"Fibers: {self.num_fibers}",
            f"Crosslinks: {self.num_crosslinks}",
            f"Dimension: {self.dimension}D",
        ]

        if self.num_fibers == 0:
            return '\n'.join(lines)

        lengths = np.array([f.length for f in self.fibers])
        radii = np.array([f.radius for f in self.fibers])
        tortuosities = np.array([f.tortuosity() for f in self.fibers])

        lines.extend([
            f"",
            f"Fiber Length:",
            f"  Mean: {np.mean(lengths):.3f}",
            f"  Std:  {np.std(lengths):.3f}",
            f"  Min:  {np.min(lengths):.3f}",
            f"  Max:  {np.max(lengths):.3f}",
            f"",
            f"Fiber Radius:",
            f"  Mean: {np.mean(radii):.4f}",
            f"  Std:  {np.std(radii):.4f}",
            f"",
            f"Tortuosity:",
            f"  Mean: {np.mean(tortuosities):.3f}",
            f"  Std:  {np.std(tortuosities):.3f}",
        ])

        # Connectivity
        if self.num_crosslinks > 0:
            lines.append(f"")
            lines.append(f"Connectivity:")
            lines.append(f"  Crosslinks per fiber: {2*self.num_crosslinks/self.num_fibers:.2f}")

        # Material
        mat_names = list(set(f.material.name for f in self.fibers))
        if len(mat_names) == 1:
            lines.append(f"Material: {mat_names[0]}")
        else:
            lines.append(f"Materials: {', '.join(mat_names)}")

        return '\n'.join(lines)

    def plot(self, **kwargs):
        """Quick visualization of the fiber network.

        For 2D networks, creates a 2D plot.
        For 3D networks, creates a 3D render.

        Parameters
        ----------
        **kwargs
            Passed to plot_network_2d or render_network_3d.
            
            2D options:
            - ax: matplotlib axes
            - color_by: 'uniform', 'stress', 'strain', 'material' (default: 'uniform')
            - colormap: matplotlib colormap name (default: 'viridis')
            - show_crosslinks: bool (default: True)
            - line_width: float (default: 1.0)
            - title: str
            - figsize: tuple (default: (8, 8))
            - save_path: str
            
            3D options:
            - color_by: 'uniform', 'stress', 'strain', 'material' (default: 'uniform')
            - colormap: str (default: 'viridis')
            - tube_radius: float
            - show_crosslinks: bool (default: True)
            - background: str (default: 'white')
            - window_size: tuple (default: (1024, 768))
            - save_path: str

        Returns
        -------
        fig : matplotlib.figure.Figure or None
            The matplotlib figure (2D only).

        Examples
        --------
        >>> net = gen.random_straight_2d(num_fibers=50, fiber_length=10, box_size=(30, 30))
        >>> net.plot(color_by='material', show_crosslinks=False)
        >>>
        >>> net_3d = gen.random_straight_3d(num_fibers=30, fiber_length=8, box_size=(20, 20, 20))
        >>> net_3d.plot(background='black')  # Opens 3D viewer
        """
        if self.dimension == 2:
            from fibernet.viz import plot_network_2d
            import inspect
            sig = inspect.signature(plot_network_2d)
            valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            return plot_network_2d(self, **valid_kwargs)
        else:
            from fibernet.viz import render_network_3d
            import inspect
            sig = inspect.signature(render_network_3d)
            valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            return render_network_3d(self, **valid_kwargs)

    def plot_statistics(self, figsize=(12, 4)):
        """Plot statistical distributions of network properties.

        Creates a figure with three subplots:
        1. Length distribution
        2. Orientation distribution (2D only)
        3. Tortuosity distribution

        Parameters
        ----------
        figsize : tuple, optional
            Figure size, by default (12, 4).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure.

        Examples
        --------
        >>> net = gen.random_straight_2d(num_fibers=100, fiber_length=10, box_size=(30, 30))
        >>> fig = net.plot_statistics()
        >>> fig.savefig('statistics.png', dpi=150, bbox_inches='tight')
        """
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Length distribution
        lengths = [f.length for f in self.fibers]
        length_range = np.ptp(lengths)
        min_range = np.mean(lengths) * 1e-10  # Relative threshold
        if length_range > min_range:
            axes[0].hist(lengths, bins=20, edgecolor='black', alpha=0.7)
        else:
            # All same length (within precision) - use single bar
            axes[0].bar([np.mean(lengths)], [len(lengths)], width=np.mean(lengths)*0.1, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Fiber Length')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Length Distribution')
        axes[0].axvline(np.mean(lengths), color='red', linestyle='--', label=f'Mean: {np.mean(lengths):.2f}')
        axes[0].legend()

        # Orientation distribution
        if self.dimension == 2:
            orientations = [f.direction for f in self.fibers]
            angles = np.degrees(orientations)
            axes[1].hist(angles, bins=36, range=(-180, 180), edgecolor='black', alpha=0.7)
            axes[1].set_xlabel('Orientation Angle (degrees)')
            axes[1].set_ylabel('Count')
            axes[1].set_title('Orientation Distribution')
        else:
            # For 3D, plot polar angles
            orientations = [f.direction for f in self.fibers]
            # Extract theta (polar angle)
            thetas = [np.degrees(np.arccos(ori[2] / np.linalg.norm(ori))) for ori in orientations]
            axes[1].hist(thetas, bins=20, edgecolor='black', alpha=0.7)
            axes[1].set_xlabel('Polar Angle (degrees)')
            axes[1].set_ylabel('Count')
            axes[1].set_title('Orientation Distribution')

        # Tortuosity distribution
        tortuosities = [f.tortuosity() for f in self.fibers]
        tort_range = np.ptp(tortuosities)
        min_tort_range = np.mean(tortuosities) * 1e-10
        if tort_range > min_tort_range:
            axes[2].hist(tortuosities, bins=20, edgecolor='black', alpha=0.7)
        else:
            axes[2].bar([np.mean(tortuosities)], [len(tortuosities)], width=0.05, edgecolor='black', alpha=0.7)
        axes[2].set_xlabel('Tortuosity')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Tortuosity Distribution')
        axes[2].axvline(np.mean(tortuosities), color='red', linestyle='--', label=f'Mean: {np.mean(tortuosities):.3f}')
        axes[2].legend()

        plt.tight_layout()
        return fig

    def validate(self) -> Dict[str, Any]:
        """Validate network integrity and return diagnostic information.

        Checks for:
        - Empty networks
        - Invalid fiber geometries (zero-length, NaN values)
        - Invalid crosslinks (out-of-bounds fiber IDs)
        - Disconnected components
        - Overlapping fibers (optional)

        Returns
        -------
        dict
            Validation results with keys:
            - 'valid': bool - Overall validity
            - 'errors': list of str - Critical issues
            - 'warnings': list of str - Non-critical issues
            - 'stats': dict - Network statistics

        Examples
        --------
        >>> net = gen.random_straight_2d(num_fibers=50, fiber_length=10, box_size=(30, 30))
        >>> result = net.validate()
        >>> if result['valid']:
        ...     print("Network is valid")
        ... else:
        ...     print(f"Errors: {result['errors']}")
        """
        import numpy as np
        
        errors = []
        warnings = []
        stats = {
            'num_fibers': self.num_fibers,
            'num_crosslinks': self.num_crosslinks,
            'dimension': self.dimension,
        }

        # Check for empty network
        if self.num_fibers == 0:
            errors.append("Network has no fibers")
            return {'valid': False, 'errors': errors, 'warnings': warnings, 'stats': stats}

        # Check each fiber
        zero_length_count = 0
        nan_count = 0
        for i, fiber in enumerate(self.fibers):
            # Check for NaN in centerline
            if np.any(np.isnan(fiber.centerline)):
                nan_count += 1
            
            # Check for zero length
            if fiber.length < 1e-12:
                zero_length_count += 1
            
            # Check radius
            if fiber.radius <= 0:
                warnings.append(f"Fiber {i} has non-positive radius: {fiber.radius}")

        if nan_count > 0:
            errors.append(f"{nan_count} fiber(s) have NaN values in centerline")
        
        if zero_length_count > 0:
            errors.append(f"{zero_length_count} fiber(s) have zero length")

        # Check crosslinks
        invalid_crosslinks = 0
        for cl in self.crosslinks:
            if cl.fiber_i < 0 or cl.fiber_i >= self.num_fibers:
                invalid_crosslinks += 1
            if cl.fiber_j < 0 or cl.fiber_j >= self.num_fibers:
                invalid_crosslinks += 1

        if invalid_crosslinks > 0:
            errors.append(f"{invalid_crosslinks} crosslink(s) reference invalid fiber IDs")

        # Check connectivity (if crosslinks exist)
        if self.num_crosslinks > 0:
            # Build adjacency
            from collections import defaultdict
            adj = defaultdict(set)
            for cl in self.crosslinks:
                adj[cl.fiber_i].add(cl.fiber_j)
                adj[cl.fiber_j].add(cl.fiber_i)
            
            # BFS to find connected components
            visited = set()
            components = 0
            for start in range(self.num_fibers):
                if start not in visited:
                    components += 1
                    queue = [start]
                    while queue:
                        node = queue.pop(0)
                        if node in visited:
                            continue
                        visited.add(node)
                        queue.extend(adj[node] - visited)
            
            stats['num_components'] = components
            if components > 1:
                warnings.append(f"Network has {components} disconnected components")
        else:
            warnings.append("No crosslinks - fibers may not be mechanically connected")

        # Add statistics
        lengths = [f.length for f in self.fibers]
        stats['mean_length'] = np.mean(lengths)
        stats['total_length'] = np.sum(lengths)
        stats['mean_radius'] = np.mean([f.radius for f in self.fibers])

        valid = len(errors) == 0
        return {'valid': valid, 'errors': errors, 'warnings': warnings, 'stats': stats}

    def to_networkx(self):
        """Convert fiber network to NetworkX graph.

        Creates a graph where:
        - Nodes represent fibers
        - Edges represent crosslinks between fibers
        - Node attributes: length, radius, material
        - Edge attributes: stiffness, strength

        Returns
        -------
        networkx.Graph
            Graph representation of the network.

        Examples
        --------
        >>> net = gen.random_straight_2d(num_fibers=50, fiber_length=10, box_size=(30, 30))
        >>> G = net.to_networkx()
        >>> print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        >>> import networkx as nx
        >>> print(f"Clustering coefficient: {nx.average_clustering(G):.3f}")
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx is required. Install with: pip install networkx")

        G = nx.Graph()

        # Add nodes (fibers)
        for fiber in self.fibers:
            G.add_node(fiber.fiber_id,
                      length=fiber.length,
                      radius=fiber.radius,
                      material=fiber.material.name,
                      youngs_modulus=fiber.material.youngs_modulus)

        # Add edges (crosslinks)
        for cl in self.crosslinks:
            G.add_edge(cl.fiber_i, cl.fiber_j,
                      stiffness=cl.stiffness,
                      strength=cl.strength,
                      crosslink_type=cl.crosslink_type)

        return G
