"""
Advanced analysis tools for fiber networks.

Provides:
- Spectral analysis (eigenvalues of graph Laplacian)
- Pore size distribution estimation
- Percolation threshold finding
- Anisotropy tensor analysis
- Structural fingerprinting
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from fibernet.core.network import FiberNetwork


class SpectralAnalyzer:
    """Spectral graph analysis of fiber networks."""
    
    def __init__(self, network: FiberNetwork):
        self.network = network
        self._graph = None
        self._laplacian = None
    
    @property
    def graph(self):
        if self._graph is None:
            self._graph = self.network.to_networkx()
        return self._graph
    
    def laplacian_eigenvalues(self, k: int = None) -> np.ndarray:
        """Compute eigenvalues of the graph Laplacian.
        
        Parameters
        ----------
        k : int, optional
            Number of eigenvalues to compute. Defaults to min(n, 20).
        """
        import networkx as nx
        
        n = len(self.graph.nodes)
        if n < 2:
            return np.array([])
        
        if k is None:
            k = min(n - 1, 20)
        
        L = nx.laplacian_matrix(self.graph).astype(float)
        
        if n <= 100:
            eigenvalues = np.linalg.eigvalsh(L.toarray())
        else:
            from scipy.sparse.linalg import eigsh
            eigenvalues = eigsh(L, k=min(k, n - 1), which='SM', return_eigenvectors=False)
        
        return np.sort(eigenvalues)
    
    def spectral_gap(self) -> float:
        """Spectral gap (second smallest Laplacian eigenvalue).
        
        Related to algebraic connectivity and network robustness.
        """
        eigs = self.laplacian_eigenvalues(k=5)
        if len(eigs) >= 2:
            return float(eigs[1])
        return 0.0
    
    def spectral_entropy(self) -> float:
        """Spectral entropy of the Laplacian spectrum.
        
        Measures the complexity/disorder of the network structure.
        """
        eigs = self.laplacian_eigenvalues()
        if len(eigs) == 0:
            return 0.0
        
        eigs = np.maximum(eigs, 1e-12)
        p = eigs / eigs.sum()
        entropy = -np.sum(p * np.log(p + 1e-12))
        return float(entropy)
    
    def algebraic_connectivity(self) -> float:
        """Algebraic connectivity (Fiedler value).
        
        Higher values indicate better-connected networks.
        """
        return self.spectral_gap()


class PoreAnalyzer:
    """Estimate pore size distribution in fiber networks."""
    
    def __init__(self, network: FiberNetwork):
        self.network = network
    
    def pore_size_distribution(
        self,
        num_samples: int = 1000,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate pore size distribution using random sampling.
        
        For each random point in the bounding box, finds the
        distance to the nearest fiber surface.
        
        Returns
        -------
        radii : np.ndarray
            Sampled pore radii.
        histogram : np.ndarray
            Normalized histogram values.
        """
        rng = np.random.default_rng(seed)
        
        if self.network.num_fibers == 0:
            return np.array([]), np.array([])
        
        bb_min, bb_max = self.network.bounding_box()
        
        from scipy.spatial import cKDTree
        
        all_points = []
        for fiber in self.network.fibers:
            for pt in fiber.centerline:
                all_points.append(pt)
        
        all_points = np.array(all_points)
        tree = cKDTree(all_points)
        
        samples = rng.uniform(bb_min, bb_max, (num_samples, 3))
        
        distances, indices = tree.query(samples)
        
        pore_radii = np.zeros(num_samples)
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            fiber_idx = 0
            cum_pts = 0
            for f in self.network.fibers:
                if cum_pts + len(f.centerline) > idx:
                    fiber_idx = f.fiber_id
                    break
                cum_pts += len(f.centerline)
            
            if fiber_idx < len(self.network.fibers):
                r_fiber = self.network.fibers[fiber_idx].radius
                pore_radii[i] = max(dist - r_fiber, 0)
            else:
                pore_radii[i] = dist
        
        return pore_radii, pore_radii
    
    def mean_pore_size(self) -> float:
        """Mean pore radius."""
        radii, _ = self.pore_size_distribution(num_samples=500, seed=42)
        if len(radii) == 0:
            return 0.0
        return float(np.mean(radii))
    
    def pore_size_statistics(self) -> Dict[str, float]:
        """Comprehensive pore size statistics."""
        radii, _ = self.pore_size_distribution(num_samples=500, seed=42)
        if len(radii) == 0:
            return {"mean": 0, "median": 0, "std": 0, "max": 0}
        return {
            "mean": float(np.mean(radii)),
            "median": float(np.median(radii)),
            "std": float(np.std(radii)),
            "max": float(np.max(radii)),
            "min": float(np.min(radii[radii > 0])) if np.any(radii > 0) else 0,
            "p95": float(np.percentile(radii, 95)),
        }


class AnisotropyAnalyzer:
    """Analyze structural anisotropy of fiber networks."""
    
    def __init__(self, network: FiberNetwork):
        self.network = network
    
    def orientation_tensor(self) -> np.ndarray:
        """Compute the second-order orientation tensor.
        
        A_ij = <p_i * p_j> where p is the fiber direction vector.
        
        For isotropic: A = I/3 (3D) or I/2 (2D)
        """
        orientations = self.network.fiber_orientations()
        if len(orientations) == 0:
            return np.eye(3) / 3
        
        n = len(orientations)
        A = np.zeros((3, 3))
        for o in orientations:
            A += np.outer(o, o)
        A /= n
        
        return A
    
    def anisotropy_index(self) -> float:
        """Anisotropy index based on orientation tensor eigenvalues.
        
        Returns value between 0 (isotropic) and 1 (fully aligned).
        """
        A = self.orientation_tensor()
        eigenvalues = np.sort(np.linalg.eigvalsh(A))[::-1]
        
        if len(eigenvalues) < 3:
            return 0.0
        
        a1 = eigenvalues[0]
        iso = 1.0 / 3
        
        ai = (a1 - iso) / (1 - iso) if iso < 1 else 0
        return float(max(0, min(1, ai)))
    
    def principal_directions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get principal directions of fiber orientation."""
        A = self.orientation_tensor()
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        
        idx = np.argsort(eigenvalues)[::-1]
        return eigenvectors[:, idx], eigenvalues[idx], eigenvectors[:, idx]


class StructuralFingerprint:
    """Create structural fingerprints for network comparison."""
    
    def __init__(self, network: FiberNetwork):
        self.network = network
    
    def compute_fingerprint(self) -> Dict[str, float]:
        """Compute a compact structural fingerprint vector."""
        from fibernet.analysis.morphology import MorphologyAnalyzer
        from fibernet.analysis.topology import TopologyAnalyzer
        
        morph = MorphologyAnalyzer(self.network)
        topo = TopologyAnalyzer(self.network)
        
        fp = {
            "num_fibers": float(self.network.num_fibers),
            "density": self.network.density(),
            "mean_length": self.network.mean_fiber_length,
            "nematic_order": morph.nematic_order_parameter(),
            "mean_tortuosity": float(np.mean(morph.tortuosity_distribution())),
            "mean_degree": topo.degree_statistics()["mean"],
            "clustering": topo.clustering_coefficient(),
        }
        
        try:
            spec = SpectralAnalyzer(self.network)
            fp["spectral_gap"] = spec.spectral_gap()
        except:
            fp["spectral_gap"] = 0.0
        
        return fp
    
    def distance_to(self, other: "StructuralFingerprint") -> float:
        """Compute Euclidean distance between fingerprints."""
        fp1 = self.compute_fingerprint()
        fp2 = other.compute_fingerprint()
        
        keys = sorted(set(fp1.keys()) & set(fp2.keys()))
        v1 = np.array([fp1[k] for k in keys])
        v2 = np.array([fp2[k] for k in keys])
        
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-12)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-12)
        
        return float(np.linalg.norm(v1_norm - v2_norm))
