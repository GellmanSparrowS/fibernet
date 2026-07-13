"""
Advanced spatial and structural statistics for fiber networks.

Provides:
- Spatial statistics (Ripley's K, pair correlation)
- Orientation distribution analysis
- Length distribution analysis
- Connectivity statistics
- Anisotropy measures (fabric tensor)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.spatial import cKDTree
import warnings

from fibernet.core.network import FiberNetwork


class SpatialStatistics:
    """Compute spatial statistics for fiber networks."""
    
    def __init__(self, network: FiberNetwork):
        self.network = network
        self._tree = None
    
    def _get_tree(self) -> cKDTree:
        """Build KD-tree for crosslink positions."""
        if self._tree is None:
            positions = np.array([cl.position for cl in self.network.crosslinks])
            if len(positions) == 0:
                # Use fiber midpoints if no crosslinks
                positions = np.array([
                    (f.centerline[0] + f.centerline[-1]) / 2 
                    for f in self.network.fibers
                ])
            if len(positions) == 0:
                positions = np.array([[0, 0, 0]])
            self._tree = cKDTree(positions)
        return self._tree
    
    def ripley_k(self, r: np.ndarray) -> np.ndarray:
        """
        Compute Ripley's K function.
        
        Parameters
        ----------
        r : np.ndarray
            Distance values at which to evaluate K(r).
        
        Returns
        -------
        K : np.ndarray
            Ripley's K function values.
        """
        tree = self._get_tree()
        n = len(tree.data)
        if n < 2:
            return np.zeros_like(r)
        
        K = np.zeros_like(r, dtype=float)
        for i, ri in enumerate(r):
            pairs = tree.query_pairs(ri)
            K[i] = len(pairs) * 2.0 / n
        
        # Normalize by density
        bbox = self.network.bounding_box()
        if bbox is not None:
            if self.network.dimension == 2:
                area = (bbox[1][0] - bbox[0][0]) * (bbox[1][1] - bbox[0][1])
                if area > 0:
                    density = n / area
                    K /= density
            else:
                volume = np.prod(bbox[1] - bbox[0])
                if volume > 0:
                    density = n / volume
                    K /= density
        
        return K
    
    def pair_correlation(self, r: np.ndarray, dr: Optional[float] = None) -> np.ndarray:
        """
        Compute pair correlation function g(r).
        
        Parameters
        ----------
        r : np.ndarray
            Distance values.
        dr : float, optional
            Bin width for histogram. If None, uses r[1] - r[0].
        
        Returns
        -------
        g : np.ndarray
            Pair correlation function values.
        """
        if dr is None and len(r) > 1:
            dr = r[1] - r[0]
        elif dr is None:
            dr = 1.0
        
        tree = self._get_tree()
        n = len(tree.data)
        if n < 2:
            return np.ones_like(r)
        
        max_r = r.max()
        pairs = tree.query_pairs(max_r)
        if not pairs:
            return np.ones_like(r)
        
        distances = np.array([np.linalg.norm(tree.data[i] - tree.data[j]) 
                             for i, j in pairs])
        
        bin_edges = np.append(r - dr/2, r[-1] + dr/2)
        hist, _ = np.histogram(distances, bins=bin_edges)
        
        # Normalize
        bbox = self.network.bounding_box()
        if bbox is not None:
            if self.network.dimension == 2:
                area = (bbox[1][0] - bbox[0][0]) * (bbox[1][1] - bbox[0][1])
                density = n / area if area > 0 else 1.0
                expected = 2 * np.pi * r * dr * density * n / 2
            else:
                volume = np.prod(bbox[1] - bbox[0])
                density = n / volume if volume > 0 else 1.0
                expected = 4 * np.pi * r**2 * dr * density * n / 2
        else:
            expected = np.ones_like(r)
        
        expected = np.maximum(expected, 1e-10)
        g = hist / expected
        
        return g
    
    def nearest_neighbor_distances(self) -> np.ndarray:
        """
        Compute nearest neighbor distances for all nodes.
        
        Returns
        -------
        distances : np.ndarray
            Distance to nearest neighbor for each node.
        """
        tree = self._get_tree()
        if len(tree.data) < 2:
            return np.array([])
        
        distances, _ = tree.query(tree.data, k=2)
        return distances[:, 1]


class OrientationAnalysis:
    """Analyze fiber orientation distributions."""
    
    def __init__(self, network: FiberNetwork):
        self.network = network
        self._orientations = None
    
    def _compute_orientations(self):
        """Compute fiber orientation vectors."""
        if self._orientations is None:
            orientations = []
            for fiber in self.network.fibers:
                start = fiber.centerline[0]
                end = fiber.centerline[-1]
                direction = end - start
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    orientations.append(direction / norm)
            self._orientations = np.array(orientations) if orientations else np.empty((0, 3))
    
    def get_orientations(self) -> np.ndarray:
        """Get fiber orientation vectors."""
        self._compute_orientations()
        return self._orientations
    
    def orientation_histogram(
        self, 
        n_bins: int = 36,
        dimension: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute orientation histogram.
        
        Parameters
        ----------
        n_bins : int
            Number of angular bins.
        dimension : str, optional
            Which dimension to analyze ('2d' or '3d').
        
        Returns
        -------
        counts : np.ndarray
            Histogram counts.
        bin_edges : np.ndarray
            Bin edges in radians.
        """
        self._compute_orientations()
        if len(self._orientations) == 0:
            return np.zeros(n_bins), np.linspace(0, np.pi, n_bins + 1)
        
        if dimension is None:
            dimension = '2d' if self.network.dimension == 2 else '3d'
        
        if dimension == '2d':
            angles = np.arctan2(self._orientations[:, 1], self._orientations[:, 0])
            angles = np.abs(angles) % np.pi
            counts, bin_edges = np.histogram(angles, bins=n_bins, range=(0, np.pi))
        else:
            z = self._orientations[:, 2]
            angles = np.arccos(np.clip(z, -1, 1))
            counts, bin_edges = np.histogram(angles, bins=n_bins, range=(0, np.pi))
        
        return counts, bin_edges
    
    def nematic_order_parameter(self) -> float:
        """
        Compute nematic order parameter S.
        
        S = <(3 cos²θ - 1) / 2>
        
        Returns
        -------
        S : float
            Nematic order parameter (0 = isotropic, 1 = perfectly aligned).
        """
        self._compute_orientations()
        if len(self._orientations) < 2:
            return 0.0
        
        Q = np.zeros((3, 3))
        for u in self._orientations:
            Q += np.outer(u, u)
        Q /= len(self._orientations)
        
        eigenvalues = np.linalg.eigvalsh(Q)
        S = (3 * eigenvalues[-1] - 1) / 2
        
        return float(np.clip(S, 0, 1))
    
    def mean_orientation(self) -> np.ndarray:
        """
        Compute mean orientation vector.
        
        Returns
        -------
        mean : np.ndarray
            Mean orientation vector.
        """
        self._compute_orientations()
        if len(self._orientations) == 0:
            return np.zeros(3)
        
        mean = np.mean(self._orientations, axis=0)
        norm = np.linalg.norm(mean)
        if norm > 1e-10:
            mean /= norm
        
        return mean


class LengthAnalysis:
    """Analyze fiber length distributions."""
    
    def __init__(self, network: FiberNetwork):
        self.network = network
    
    def get_lengths(self) -> np.ndarray:
        """Get all fiber lengths."""
        return np.array([f.length for f in self.network.fibers])
    
    def length_statistics(self) -> Dict[str, float]:
        """
        Compute length distribution statistics.
        
        Returns
        -------
        stats : dict
            Dictionary with mean, std, median, min, max, skewness, kurtosis.
        """
        lengths = self.get_lengths()
        if len(lengths) == 0:
            return {
                'mean': 0.0, 'std': 0.0, 'median': 0.0,
                'min': 0.0, 'max': 0.0, 'skewness': 0.0, 'kurtosis': 0.0
            }
        
        return {
            'mean': float(np.mean(lengths)),
            'std': float(np.std(lengths)),
            'median': float(np.median(lengths)),
            'min': float(np.min(lengths)),
            'max': float(np.max(lengths)),
            'skewness': float(stats.skew(lengths)) if len(lengths) > 2 else 0.0,
            'kurtosis': float(stats.kurtosis(lengths)) if len(lengths) > 3 else 0.0,
        }
    
    def length_histogram(
        self,
        n_bins: int = 20,
        density: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute length histogram.
        
        Parameters
        ----------
        n_bins : int
            Number of bins.
        density : bool
            If True, normalize to probability density.
        
        Returns
        -------
        counts : np.ndarray
            Histogram counts (or density).
        bin_edges : np.ndarray
            Bin edges.
        """
        lengths = self.get_lengths()
        if len(lengths) == 0:
            return np.zeros(n_bins), np.linspace(0, 1, n_bins + 1)
        
        if np.ptp(lengths) < 1e-10:
            # All lengths are identical, create bins around the single value
            center = lengths[0]
            bin_edges = np.linspace(center - 0.5, center + 0.5, n_bins + 1)
            counts = np.zeros(n_bins)
            counts[n_bins // 2] = len(lengths)
            if density:
                dr = bin_edges[1] - bin_edges[0]
                counts = counts / (len(lengths) * dr) if len(lengths) * dr > 0 else counts
            return counts, bin_edges
        
        counts, bin_edges = np.histogram(lengths, bins=n_bins, density=density)
        return counts, bin_edges
    
    def fit_distribution(self, dist_name: str = 'norm') -> Tuple:
        """
        Fit a distribution to the length data.
        
        Parameters
        ----------
        dist_name : str
            Distribution name (e.g., 'norm', 'lognorm', 'gamma', 'expon').
        
        Returns
        -------
        params : tuple
            Fitted distribution parameters.
        """
        lengths = self.get_lengths()
        if len(lengths) < 10:
            warnings.warn("Too few data points for reliable fit")
            return None
        
        dist = getattr(stats, dist_name)
        params = dist.fit(lengths)
        return params


class ConnectivityAnalysis:
    """Analyze network connectivity."""
    
    def __init__(self, network: FiberNetwork):
        self.network = network
    
    def degree_distribution(self) -> Dict[int, int]:
        """
        Compute degree distribution.
        
        Returns
        -------
        degrees : dict
            Dictionary mapping degree -> count.
        """
        degrees = {}
        for cl in self.network.crosslinks:
            deg = 2  # Each crosslink connects 2 fibers
            degrees[deg] = degrees.get(deg, 0) + 1
        return degrees
    
    def mean_connectivity(self) -> float:
        """
        Compute mean connectivity (average degree).
        
        Returns
        -------
        mean : float
            Average number of fibers per crosslink.
        """
        if not self.network.crosslinks:
            return 0.0
        
        total = 2 * len(self.network.crosslinks)
        return total / len(self.network.crosslinks)
    
    def connectivity_statistics(self) -> Dict[str, float]:
        """
        Compute connectivity statistics.
        
        Returns
        -------
        stats : dict
            Dictionary with mean, std, min, max.
        """
        if not self.network.crosslinks:
            return {'mean': 0.0, 'std': 0.0, 'min': 0, 'max': 0}
        
        degrees = [2] * len(self.network.crosslinks)
        return {
            'mean': float(np.mean(degrees)),
            'std': float(np.std(degrees)),
            'min': int(np.min(degrees)),
            'max': int(np.max(degrees)),
        }


class AnisotropyAnalysis:
    """Analyze network anisotropy."""
    
    def __init__(self, network: FiberNetwork):
        self.network = network
    
    def fabric_tensor(self) -> np.ndarray:
        """
        Compute fabric tensor (second-order orientation tensor).
        
        Returns
        -------
        A : np.ndarray
            3x3 fabric tensor.
        """
        orientations = OrientationAnalysis(self.network).get_orientations()
        if len(orientations) == 0:
            return np.eye(3) / 3
        
        A = np.zeros((3, 3))
        for u in orientations:
            A += np.outer(u, u)
        A /= len(orientations)
        
        return A
    
    def anisotropy_index(self) -> float:
        """
        Compute anisotropy index from fabric tensor eigenvalues.
        
        Returns
        -------
        AI : float
            Anisotropy index (0 = isotropic, 1 = perfectly anisotropic).
        """
        A = self.fabric_tensor()
        eigenvalues = np.linalg.eigvalsh(A)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        if eigenvalues[0] < 1e-10:
            return 0.0
        
        AI = (eigenvalues[0] - eigenvalues[-1]) / eigenvalues[0]
        
        return float(np.clip(AI, 0, 1))
    
    def principal_directions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute principal directions and eigenvalues.
        
        Returns
        -------
        eigenvalues : np.ndarray
            Eigenvalues (descending order).
        eigenvectors : np.ndarray
            Eigenvectors (columns).
        """
        A = self.fabric_tensor()
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return eigenvalues, eigenvectors


def compute_spatial_statistics(network: FiberNetwork) -> Dict:
    """
    Compute all available spatial and structural statistics.
    
    Parameters
    ----------
    network : FiberNetwork
        The network to analyze.
    
    Returns
    -------
    stats : dict
        Dictionary containing all statistics.
    """
    results = {}
    
    results['n_fibers'] = len(network.fibers)
    results['n_crosslinks'] = len(network.crosslinks)
    results['dimension'] = network.dimension
    
    length_analyzer = LengthAnalysis(network)
    results['length'] = length_analyzer.length_statistics()
    results['total_length'] = float(network.total_length)
    
    orient_analyzer = OrientationAnalysis(network)
    results['nematic_order'] = orient_analyzer.nematic_order_parameter()
    
    conn_analyzer = ConnectivityAnalysis(network)
    results['connectivity'] = conn_analyzer.connectivity_statistics()
    results['mean_connectivity'] = conn_analyzer.mean_connectivity()
    
    aniso_analyzer = AnisotropyAnalysis(network)
    results['anisotropy_index'] = aniso_analyzer.anisotropy_index()
    
    if results['n_crosslinks'] >= 10 or results['n_fibers'] >= 10:
        spatial = SpatialStatistics(network)
        nn_dist = spatial.nearest_neighbor_distances()
        if len(nn_dist) > 0:
            results['nearest_neighbor'] = {
                'mean': float(np.mean(nn_dist)),
                'std': float(np.std(nn_dist)),
                'min': float(np.min(nn_dist)),
                'max': float(np.max(nn_dist)),
            }
    
    return results
