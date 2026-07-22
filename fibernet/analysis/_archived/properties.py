"""
Property computation for fiber networks.

Provides effective property estimation:
- Effective elastic modulus
- Effective thermal conductivity
- Effective electrical conductivity
- Specific surface area
- Anisotropy metrics
"""

import numpy as np
from typing import Dict, Optional
from fibernet.core.network import FiberNetwork


class PropertyEstimator:
    """Estimate effective properties of fiber networks."""
    
    def __init__(self, network: FiberNetwork):
        self.network = network
    
    def rule_of_mixtures_modulus(self, axis: int = 0) -> float:
        """Estimate modulus using rule of mixtures (Voigt bound)."""
        if self.network.num_fibers == 0:
            return 0.0
        
        orientations = self.network.fiber_orientations()
        E_fibers = np.array([f.material.youngs_modulus for f in self.network.fibers])
        V_f = self.network.density()
        
        if len(orientations) == 0:
            return 0.0
        
        cos_theta = np.abs(orientations[:, axis])
        E_eff = V_f * np.mean(E_fibers * cos_theta**4)
        
        return float(E_eff)
    
    def reuss_bound_modulus(self, axis: int = 0) -> float:
        """Estimate modulus using Reuss (lower) bound."""
        if self.network.num_fibers == 0:
            return 0.0
        
        orientations = self.network.fiber_orientations()
        E_fibers = np.array([f.material.youngs_modulus for f in self.network.fibers])
        V_f = self.network.density()
        
        if len(orientations) == 0 or V_f < 1e-12:
            return 0.0
        
        cos_theta = np.abs(orientations[:, axis])
        inv_E = np.mean(cos_theta**2 / E_fibers)
        
        return float(V_f / inv_E) if inv_E > 1e-30 else 0.0
    
    def specific_surface_area(self) -> float:
        """Surface area per unit volume of the network."""
        if self.network.num_fibers == 0:
            return 0.0
        
        surface = 0.0
        for fiber in self.network.fibers:
            surface += 2 * np.pi * fiber.radius * fiber.length
        
        bb_min, bb_max = self.network.bounding_box()
        dims = bb_max - bb_min
        volume = float(np.prod(np.maximum(dims, 1e-12)))
        
        return surface / volume if volume > 1e-12 else 0.0
    
    def anisotropy_ratio(self) -> Dict[str, float]:
        """Compute mechanical anisotropy ratio (E_max/E_min) across axes."""
        moduli = []
        for axis in range(min(self.network.dimension, 3)):
            E = self.rule_of_mixtures_modulus(axis)
            moduli.append(E)
        
        if not moduli or min(moduli) < 1e-12:
            return {"ratio": 1.0, "moduli": moduli}
        
        return {
            "ratio": max(moduli) / min(moduli),
            "moduli": moduli,
            "E_x": moduli[0] if len(moduli) > 0 else 0,
            "E_y": moduli[1] if len(moduli) > 1 else 0,
            "E_z": moduli[2] if len(moduli) > 2 else 0,
        }
    
    def full_report(self) -> Dict[str, any]:
        """Comprehensive property report."""
        return {
            "Voigt_modulus_x": self.rule_of_mixtures_modulus(0),
            "Voigt_modulus_y": self.rule_of_mixtures_modulus(1),
            "Voigt_modulus_z": self.rule_of_mixtures_modulus(2) if self.network.dimension == 3 else None,
            "Reuss_modulus_x": self.reuss_bound_modulus(0),
            "specific_surface_area": self.specific_surface_area(),
            "anisotropy": self.anisotropy_ratio(),
            "volume_fraction": self.network.density(),
        }
