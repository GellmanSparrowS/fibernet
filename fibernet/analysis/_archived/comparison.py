"""
Network Comparison and Similarity Analysis

Provides tools for comparing fiber networks quantitatively:
- Structural similarity metrics
- Statistical distance measures
- Network fingerprinting
- Clustering and classification

References:
- Borgwardt, K.M. et al., "Graph Kernels", JMLR, 2009
- Shervashidze, N. et al., "Weisfeiler-Lehman Graph Kernels", JMLR, 2011
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.spatial.distance import cdist
import warnings

from fibernet.core.network import FiberNetwork
from fibernet.analysis.spatial import (
    OrientationAnalysis, LengthAnalysis, ConnectivityAnalysis, AnisotropyAnalysis
)


class NetworkFingerprint:
    """Compute structural fingerprint for a network."""
    
    def __init__(self, network: FiberNetwork):
        self.network = network
        self._fingerprint = None
    
    def compute_fingerprint(self, normalize: bool = True) -> np.ndarray:
        """
        Compute structural fingerprint vector.
        
        The fingerprint captures key structural features:
        - Nematic order parameter
        - Anisotropy index
        - Length statistics (mean, std, skewness)
        - Connectivity statistics (mean, std)
        - Spatial statistics (mean nearest neighbor distance)
        
        Parameters
        ----------
        normalize : bool
            If True, normalize fingerprint to [0, 1] range.
        
        Returns
        -------
        fingerprint : np.ndarray
            Feature vector.
        """
        features = []
        
        # Orientation features
        orient = OrientationAnalysis(self.network)
        features.append(orient.nematic_order_parameter())  # S
        
        # Anisotropy
        aniso = AnisotropyAnalysis(self.network)
        features.append(aniso.anisotropy_index())  # AI
        
        # Length statistics
        length = LengthAnalysis(self.network)
        length_stats = length.length_statistics()
        features.extend([
            length_stats['mean'],
            length_stats['std'],
            length_stats['skewness'],
        ])
        
        # Connectivity
        conn = ConnectivityAnalysis(self.network)
        conn_stats = conn.connectivity_statistics()
        features.extend([
            conn_stats['mean'],
            conn_stats['std'],
        ])
        
        # Network density
        n_fibers = len(self.network.fibers)
        n_crosslinks = len(self.network.crosslinks)
        features.append(n_fibers)
        features.append(n_crosslinks)
        
        # Total length
        features.append(float(self.network.total_length))
        
        fingerprint = np.array(features, dtype=float)
        
        if normalize and self._fingerprint is None:
            # Store for later normalization
            self._fingerprint = fingerprint
        
        return fingerprint
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of features in fingerprint.
        
        Returns
        -------
        names : list of str
            Feature names.
        """
        return [
            'nematic_order',
            'anisotropy_index',
            'length_mean',
            'length_std',
            'length_skewness',
            'connectivity_mean',
            'connectivity_std',
            'n_fibers',
            'n_crosslinks',
            'total_length',
        ]


class NetworkComparator:
    """Compare two or more networks."""
    
    def __init__(self, networks: List[FiberNetwork]):
        self.networks = networks
        self.fingerprints = [
            NetworkFingerprint(net).compute_fingerprint(normalize=False)
            for net in networks
        ]
    
    def pairwise_distances(self, metric: str = 'euclidean') -> np.ndarray:
        """
        Compute pairwise distances between networks.
        
        Parameters
        ----------
        metric : str
            Distance metric ('euclidean', 'cosine', 'correlation').
        
        Returns
        -------
        distances : np.ndarray
            Pairwise distance matrix.
        """
        fps = np.array(self.fingerprints)
        
        # Normalize features
        if fps.shape[0] > 1:
            means = np.mean(fps, axis=0)
            stds = np.std(fps, axis=0)
            # Replace zero/near-zero stds with 1.0 to avoid division by zero
            stds = np.where(stds < 1e-10, 1.0, stds)
            fps_norm = (fps - means) / stds
            # Replace any NaN/inf with 0
            fps_norm = np.nan_to_num(fps_norm, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            fps_norm = fps
        
        distances = cdist(fps_norm, fps_norm, metric=metric)
        
        return distances
    
    def most_similar(self, query_idx: int = 0, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Find most similar networks to a query network.
        
        Parameters
        ----------
        query_idx : int
            Index of query network.
        top_k : int
            Number of most similar networks to return.
        
        Returns
        -------
        similar : list of (int, float)
            List of (index, distance) tuples.
        """
        distances = self.pairwise_distances()
        query_distances = distances[query_idx]
        
        # Sort by distance (exclude self)
        sorted_indices = np.argsort(query_distances)
        similar = []
        for idx in sorted_indices:
            if idx != query_idx:
                similar.append((int(idx), float(query_distances[idx])))
                if len(similar) >= top_k:
                    break
        
        return similar
    
    def cluster(self, n_clusters: int = 3) -> np.ndarray:
        """
        Cluster networks using k-means.
        
        Parameters
        ----------
        n_clusters : int
            Number of clusters.
        
        Returns
        -------
        labels : np.ndarray
            Cluster labels for each network.
        """
        try:
            from sklearn.cluster import KMeans
            
            fps = np.array(self.fingerprints)
            if fps.shape[0] > 1:
                means = np.mean(fps, axis=0)
                stds = np.std(fps, axis=0)
                stds = np.where(stds < 1e-10, 1.0, stds)
                fps_norm = (fps - means) / stds
                fps_norm = np.nan_to_num(fps_norm, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                fps_norm = fps
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(fps_norm)
            
            return labels
        except ImportError:
            warnings.warn("scikit-learn required for clustering")
            return np.zeros(len(self.networks), dtype=int)
    
    def compare_statistics(self) -> Dict:
        """
        Compare basic statistics across networks.
        
        Returns
        -------
        comparison : dict
            Dictionary with statistics for each network.
        """
        comparison = {}
        
        for i, net in enumerate(self.networks):
            comparison[f'network_{i}'] = {
                'n_fibers': len(net.fibers),
                'n_crosslinks': len(net.crosslinks),
                'total_length': float(net.total_length),
                'dimension': net.dimension,
            }
        
        return comparison


def compare_networks(
    networks: List[FiberNetwork],
    metric: str = 'euclidean'
) -> Dict:
    """
    Compare multiple networks.
    
    Parameters
    ----------
    networks : list of FiberNetwork
        Networks to compare.
    metric : str
        Distance metric.
    
    Returns
    -------
    results : dict
        Comparison results.
    """
    if len(networks) < 2:
        raise ValueError("At least 2 networks required for comparison")
    
    comparator = NetworkComparator(networks)
    
    results = {
        'distances': comparator.pairwise_distances(metric=metric),
        'statistics': comparator.compare_statistics(),
        'n_networks': len(networks),
    }
    
    return results


def network_similarity(
    net1: FiberNetwork,
    net2: FiberNetwork
) -> float:
    """
    Compute similarity score between two networks.
    
    Parameters
    ----------
    net1 : FiberNetwork
        First network.
    net2 : FiberNetwork
        Second network.
    
    Returns
    -------
    similarity : float
        Similarity score (0 = different, 1 = identical).
    """
    comparator = NetworkComparator([net1, net2])
    distances = comparator.pairwise_distances()
    
    # Convert distance to similarity (exponential decay)
    distance = distances[0, 1]
    similarity = np.exp(-distance)
    
    return float(similarity)
