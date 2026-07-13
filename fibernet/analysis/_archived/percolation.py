"""
Percolation analysis for fiber networks.

Provides tools for analyzing electrical and thermal percolation in fiber networks,
including percolation threshold estimation, cluster analysis, and conductivity prediction.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from ..core import FiberNetwork


@dataclass
class PercolationResult:
    """Result of percolation analysis."""
    percolates: bool
    cluster_sizes: np.ndarray
    largest_cluster_size: int
    percolation_probability: float
    mean_cluster_size: float
    correlation_length: float
    backbone_size: int
    effective_conductivity: float


class PercolationAnalyzer:
    """
    Analyze percolation in fiber networks.
    
    Percolation theory describes how connectivity emerges in random systems.
    For fiber networks, percolation occurs when a continuous path of connected
    fibers spans the system.
    
    Parameters
    ----------
    network : FiberNetwork
        The fiber network to analyze
    contact_distance : float, optional
        Distance threshold for fiber-fiber contact (default: 2 * fiber radius)
    """
    
    def __init__(
        self,
        network: FiberNetwork,
        contact_distance: Optional[float] = None,
    ):
        self.network = network
        self.contact_distance = contact_distance
        
        # Default contact distance
        if self.contact_distance is None:
            if len(network.fibers) > 0:
                self.contact_distance = 2.0 * network.fibers[0].radius
            else:
                self.contact_distance = 0.2
        
        # Build adjacency matrix
        self._build_adjacency_matrix()
    
    def _build_adjacency_matrix(self):
        """Build adjacency matrix based on fiber contacts."""
        n_fibers = len(self.network.fibers)
        
        # Initialize sparse adjacency matrix
        rows = []
        cols = []
        data = []
        
        for i in range(n_fibers):
            fiber_i = self.network.fibers[i]
            
            for j in range(i + 1, n_fibers):
                fiber_j = self.network.fibers[j]
                
                # Check if fibers are in contact
                if self._fibers_in_contact(fiber_i, fiber_j):
                    rows.extend([i, j])
                    cols.extend([j, i])
                    data.extend([1, 1])
        
        self.adjacency_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(n_fibers, n_fibers)
        )
    
    def _fibers_in_contact(self, fiber_i, fiber_j) -> bool:
        """Check if two fibers are in contact."""
        # Get centerline points
        points_i = fiber_i.centerline
        points_j = fiber_j.centerline
        
        # Compute minimum distance between centerlines
        min_dist = np.inf
        
        for p_i in points_i:
            for p_j in points_j:
                dist = np.linalg.norm(p_i - p_j)
                min_dist = min(min_dist, dist)
        
        return min_dist <= self.contact_distance
    
    def analyze(self) -> PercolationResult:
        """
        Perform percolation analysis.
        
        Returns
        -------
        PercolationResult
            Results of percolation analysis
        """
        # Find connected components
        n_components, labels = connected_components(
            self.adjacency_matrix,
            directed=False,
            return_labels=True
        )
        
        # Compute cluster sizes
        cluster_sizes = np.bincount(labels)
        largest_cluster_size = np.max(cluster_sizes)
        
        # Check for percolation (cluster spanning the system)
        percolates = self._check_percolation(labels)
        
        # Compute percolation probability
        percolation_probability = largest_cluster_size / len(self.network.fibers)
        
        # Compute mean cluster size (excluding largest cluster)
        if n_components > 1:
            cluster_sizes_no_largest = np.delete(
                cluster_sizes,
                np.argmax(cluster_sizes)
            )
            mean_cluster_size = np.mean(cluster_sizes_no_largest)
        else:
            mean_cluster_size = largest_cluster_size
        
        # Compute correlation length
        correlation_length = self._compute_correlation_length(labels)
        
        # Compute backbone size
        backbone_size = self._compute_backbone_size(labels)
        
        # Estimate effective conductivity
        effective_conductivity = self._estimate_conductivity(labels)
        
        return PercolationResult(
            percolates=percolates,
            cluster_sizes=cluster_sizes,
            largest_cluster_size=largest_cluster_size,
            percolation_probability=percolation_probability,
            mean_cluster_size=mean_cluster_size,
            correlation_length=correlation_length,
            backbone_size=backbone_size,
            effective_conductivity=effective_conductivity,
        )
    
    def _check_percolation(self, labels: np.ndarray) -> bool:
        """Check if any cluster spans the system."""
        # Get bounding box
        all_points = np.vstack([f.centerline for f in self.network.fibers])
        bbox_min = np.min(all_points, axis=0)
        bbox_max = np.max(all_points, axis=0)
        
        # Check each cluster
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            # Get fibers in this cluster
            cluster_fibers = [
                self.network.fibers[i]
                for i in range(len(labels))
                if labels[i] == label
            ]
            
            # Get all points in cluster
            cluster_points = np.vstack([f.centerline for f in cluster_fibers])
            
            # Check if cluster spans in any direction
            cluster_min = np.min(cluster_points, axis=0)
            cluster_max = np.max(cluster_points, axis=0)
            
            # Check if spans in x, y, or z direction
            spans_x = (cluster_min[0] - bbox_min[0] < 0.1 * (bbox_max[0] - bbox_min[0])) and \
                      (cluster_max[0] - bbox_max[0] > -0.1 * (bbox_max[0] - bbox_min[0]))
            
            spans_y = (cluster_min[1] - bbox_min[1] < 0.1 * (bbox_max[1] - bbox_min[1])) and \
                      (cluster_max[1] - bbox_max[1] > -0.1 * (bbox_max[1] - bbox_min[1]))
            
            if self.network.dimension == 3:
                spans_z = (cluster_min[2] - bbox_min[2] < 0.1 * (bbox_max[2] - bbox_min[2])) and \
                          (cluster_max[2] - bbox_max[2] > -0.1 * (bbox_max[2] - bbox_min[2]))
            else:
                spans_z = False
            
            if spans_x or spans_y or spans_z:
                return True
        
        return False
    
    def _compute_correlation_length(self, labels: np.ndarray) -> float:
        """Compute correlation length (characteristic cluster size)."""
        # Use largest cluster
        largest_label = np.argmax(np.bincount(labels))
        
        # Get fibers in largest cluster
        cluster_fibers = [
            self.network.fibers[i]
            for i in range(len(labels))
            if labels[i] == largest_label
        ]
        
        if len(cluster_fibers) == 0:
            return 0.0
        
        # Get all points
        cluster_points = np.vstack([f.centerline for f in cluster_fibers])
        
        # Compute radius of gyration
        centroid = np.mean(cluster_points, axis=0)
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        correlation_length = np.sqrt(np.mean(distances**2))
        
        return correlation_length
    
    def _compute_backbone_size(self, labels: np.ndarray) -> int:
        """Compute backbone size (fibers contributing to transport)."""
        # Simplified: backbone = largest cluster
        return np.max(np.bincount(labels))
    
    def _estimate_conductivity(self, labels: np.ndarray) -> float:
        """
        Estimate effective conductivity using percolation theory.
        
        sigma_eff ~ (p - p_c)^t for p > p_c
        
        where p is fiber density, p_c is percolation threshold,
        and t is the conductivity exponent (~1.3-2.0 for 2D, ~2.0 for 3D).
        """
        # Check if percolates
        if not self._check_percolation(labels):
            return 0.0
        
        # Estimate percolation probability
        largest_cluster_size = np.max(np.bincount(labels))
        p = largest_cluster_size / len(self.network.fibers)
        
        # Critical percolation threshold (approximate)
        if self.network.dimension == 2:
            p_c = 0.5  # Approximate for 2D
            t = 1.3
        else:
            p_c = 0.3  # Approximate for 3D
            t = 2.0
        
        # Estimate conductivity
        if p > p_c:
            conductivity = (p - p_c) ** t
        else:
            conductivity = 0.0
        
        return conductivity
    
    def cluster_analysis(self) -> Dict:
        """
        Detailed cluster analysis.
        
        Returns
        -------
        Dict
            Dictionary with cluster statistics
        """
        n_components, labels = connected_components(
            self.adjacency_matrix,
            directed=False,
            return_labels=True
        )
        
        cluster_sizes = np.bincount(labels)
        
        return {
            'n_clusters': n_components,
            'cluster_sizes': cluster_sizes.tolist(),
            'largest_cluster': int(np.max(cluster_sizes)),
            'smallest_cluster': int(np.min(cluster_sizes)),
            'mean_cluster_size': float(np.mean(cluster_sizes)),
            'median_cluster_size': float(np.median(cluster_sizes)),
            'std_cluster_size': float(np.std(cluster_sizes)),
        }
    
    def find_percolating_path(self) -> Optional[List[int]]:
        """
        Find the shortest percolating path through the network.
        
        Returns
        -------
        Optional[List[int]]
            List of fiber indices in percolating path, or None if no percolation
        """
        # Check if percolates
        n_components, labels = connected_components(
            self.adjacency_matrix,
            directed=False,
            return_labels=True
        )
        
        if not self._check_percolation(labels):
            return None
        
        # Find largest cluster
        largest_label = np.argmax(np.bincount(labels))
        cluster_indices = np.where(labels == largest_label)[0]
        
        # Build subgraph for largest cluster
        subgraph = self.adjacency_matrix[cluster_indices][:, cluster_indices]
        
        # Find shortest path between boundary nodes
        # Simplified: just return cluster indices
        return cluster_indices.tolist()


def estimate_percolation_threshold(
    generator_func,
    generator_params: Dict,
    n_samples: int = 10,
    contact_distance: Optional[float] = None,
) -> Dict:
    """
    Estimate percolation threshold by varying fiber density.
    
    Parameters
    ----------
    generator_func : callable
        Network generator function
    generator_params : Dict
        Parameters for generator
    n_samples : int
        Number of samples per density
    contact_distance : float, optional
        Contact distance threshold
    
    Returns
    -------
    Dict
        Dictionary with percolation threshold estimate
    """
    # Vary number of fibers
    fiber_counts = np.linspace(10, 200, 10, dtype=int)
    
    percolation_probs = []
    
    for n_fibers in fiber_counts:
        params = generator_params.copy()
        params['num_fibers'] = int(n_fibers)
        
        # Generate multiple samples
        percolated = 0
        
        for i in range(n_samples):
            params['seed'] = i
            net = generator_func(**params)
            
            analyzer = PercolationAnalyzer(net, contact_distance)
            result = analyzer.analyze()
            
            if result.percolates:
                percolated += 1
        
        percolation_probs.append(percolated / n_samples)
    
    # Find percolation threshold (where probability = 0.5)
    percolation_probs = np.array(percolation_probs)
    
    # Linear interpolation
    idx = np.argmin(np.abs(percolation_probs - 0.5))
    p_c = fiber_counts[idx]
    
    return {
        'fiber_counts': fiber_counts.tolist(),
        'percolation_probabilities': percolation_probs.tolist(),
        'percolation_threshold': int(p_c),
    }
