"""
Morphological analysis for fiber networks.

Provides:
- Orientation distribution function (ODF)
- Fiber length distribution
- Curvature distribution
- Tortuosity statistics
- Porosity and pore analysis
- Alignment metrics (nematic order parameter)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from fibernet.core.network import FiberNetwork


class MorphologyAnalyzer:
    """Analyze morphological properties of fiber networks."""
    
    def __init__(self, network: FiberNetwork):
        self.network = network
    
    def orientation_distribution(self, num_bins: int = 36) -> Tuple[np.ndarray, np.ndarray]:
        """Compute 2D orientation distribution (angle histogram).
        
        Returns
        -------
        angles : np.ndarray
            Bin centers (radians).
        counts : np.ndarray
            Normalized frequency per bin.
        """
        orientations = self.network.fiber_orientations()
        if len(orientations) == 0:
            return np.array([]), np.array([])
        
        angles = np.arctan2(orientations[:, 1], orientations[:, 0])
        angles = np.mod(angles, np.pi)
        
        bins = np.linspace(0, np.pi, num_bins + 1)
        counts, _ = np.histogram(angles, bins=bins, density=True)
        centers = 0.5 * (bins[:-1] + bins[1:])
        
        return centers, counts
    
    def orientation_3d(self, num_bins_theta: int = 18, num_bins_phi: int = 36) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute 3D orientation distribution (spherical histogram)."""
        orientations = self.network.fiber_orientations()
        if len(orientations) == 0:
            return np.array([]), np.array([]), np.array([])
        
        theta = np.arccos(np.clip(orientations[:, 2], -1, 1))
        phi = np.arctan2(orientations[:, 1], orientations[:, 0])
        
        theta_bins = np.linspace(0, np.pi, num_bins_theta + 1)
        phi_bins = np.linspace(-np.pi, np.pi, num_bins_phi + 1)
        
        H, _, _ = np.histogram2d(theta, phi, bins=[theta_bins, phi_bins], density=True)
        
        return 0.5 * (theta_bins[:-1] + theta_bins[1:]), 0.5 * (phi_bins[:-1] + phi_bins[1:]), H
    
    def nematic_order_parameter(self, preferred_direction: Optional[np.ndarray] = None) -> float:
        """Compute nematic order parameter S (2D or 3D).
        
        S = 1 means perfectly aligned, S = 0 means isotropic.
        
        Parameters
        ----------
        preferred_direction : array-like, optional
            If None, uses the principal orientation.
        """
        orientations = self.network.fiber_orientations()
        if len(orientations) == 0:
            return 0.0
        
        if preferred_direction is not None:
            d = np.asarray(preferred_direction)
            d = d / np.linalg.norm(d)
            cos_theta = np.abs(orientations @ d)
            if self.network.dimension == 2:
                S = 2 * np.mean(cos_theta**2) - 1
            else:
                S = 0.5 * (3 * np.mean(cos_theta**2) - 1)
        else:
            if self.network.dimension == 2:
                angles = np.arctan2(orientations[:, 1], orientations[:, 0])
                S = np.abs(np.mean(np.exp(2j * angles)))
            else:
                Q = np.zeros((3, 3))
                for o in orientations:
                    Q += np.outer(o, o) - np.eye(3) / 3
                Q /= len(orientations)
                eigenvalues = np.linalg.eigvalsh(Q)
                S = np.max(np.abs(eigenvalues)) * 1.5
        
        return float(S)
    
    def length_distribution(self, num_bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Fiber length distribution."""
        lengths = self.network.fiber_lengths()
        if len(lengths) == 0:
            return np.array([]), np.array([])
        
        bins = np.linspace(lengths.min(), lengths.max(), num_bins + 1)
        counts, edges = np.histogram(lengths, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        
        return centers, counts
    
    def curvature_distribution(self) -> np.ndarray:
        """Maximum curvature for each fiber."""
        curvatures = []
        for fiber in self.network.fibers:
            k = fiber.curvature()
            curvatures.append(np.max(k))
        return np.array(curvatures)
    
    def tortuosity_distribution(self) -> np.ndarray:
        """Tortuosity (L/L_ee) for each fiber."""
        return np.array([f.tortuosity() for f in self.network.fibers])
    
    def porosity(self) -> float:
        """Volume fraction of void space (1 - solid fraction)."""
        return 1.0 - self.network.density()
    
    def full_report(self) -> Dict[str, any]:
        """Comprehensive morphology report."""
        lengths = self.network.fiber_lengths()
        tort = self.tortuosity_distribution()
        
        report = {
            "num_fibers": self.network.num_fibers,
            "total_length": self.network.total_length,
            "mean_length": float(np.mean(lengths)) if len(lengths) > 0 else 0,
            "std_length": float(np.std(lengths)) if len(lengths) > 0 else 0,
            "mean_radius": self.network.mean_radius,
            "volume_fraction": self.network.density(),
            "porosity": self.porosity(),
            "nematic_order": self.nematic_order_parameter(),
            "mean_tortuosity": float(np.mean(tort)) if len(tort) > 0 else 0,
            "max_tortuosity": float(np.max(tort)) if len(tort) > 0 else 0,
        }
        
        return report
