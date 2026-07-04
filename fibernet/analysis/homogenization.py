"""
Homogenization and Effective Property Computation

Provides tools for computing effective (homogenized) properties of fiber networks:
- Effective elastic modulus
- Effective thermal conductivity
- Effective electrical conductivity
- Effective permeability
- Representative Volume Element (RVE) analysis

References:
- Torquato, S., "Random Heterogeneous Materials", Springer, 2002
- Zohdi, T.I. & Wriggers, P., "Introduction to Computational Micromechanics", Springer, 2005
"""

import numpy as np
from typing import Dict, Optional, Tuple
from scipy import sparse
from scipy.sparse import linalg as spla
import warnings

from fibernet.core.network import FiberNetwork
from fibernet.core.material import Material


class EffectiveElasticProperties:
    """Compute effective elastic properties using homogenization."""
    
    def __init__(self, network: FiberNetwork):
        self.network = network
    
    def effective_modulus_2d(
        self,
        direction: str = 'x',
        strain: float = 0.01
    ) -> float:
        """
        Compute effective Young's modulus in 2D.
        
        Parameters
        ----------
        direction : str
            Loading direction ('x' or 'y').
        strain : float
            Applied strain magnitude.
        
        Returns
        -------
        E_eff : float
            Effective Young's modulus.
        """
        if self.network.dimension != 2:
            raise ValueError("This method is for 2D networks only")
        
        # Get fiber properties
        fibers = self.network.fibers
        if not fibers:
            return 0.0
        
        # Compute total cross-sectional area
        total_area = sum(np.pi * f.radius**2 for f in fibers)
        
        # Compute effective stiffness using rule of mixtures (simplified)
        # E_eff = (1/V) * sum(E_i * A_i * cos^2(theta_i))
        
        bbox = self.network.bounding_box()
        if bbox is None:
            return 0.0
        
        if direction == 'x':
            L = bbox[1][0] - bbox[0][0]
            W = bbox[1][1] - bbox[0][1]
        else:
            L = bbox[1][1] - bbox[0][1]
            W = bbox[1][0] - bbox[0][0]
        
        if L <= 0 or W <= 0:
            return 0.0
        
        volume = L * W
        E_sum = 0.0
        
        for fiber in fibers:
            # Fiber direction
            start = fiber.centerline[0]
            end = fiber.centerline[-1]
            direction_vec = end - start
            length = np.linalg.norm(direction_vec[:2])
            
            if length < 1e-10:
                continue
            
            direction_vec = direction_vec[:2] / length
            
            # Angle with loading direction
            if direction == 'x':
                cos_theta = abs(direction_vec[0])
            else:
                cos_theta = abs(direction_vec[1])
            
            # Contribution to stiffness
            A = np.pi * fiber.radius**2
            E = fiber.material.youngs_modulus
            E_sum += E * A * length * cos_theta**2
        
        E_eff = E_sum / (volume * strain)
        
        return float(E_eff)
    
    def effective_poisson_ratio(self) -> float:
        """
        Estimate effective Poisson's ratio.
        
        Returns
        -------
        nu_eff : float
            Effective Poisson's ratio.
        """
        # Simplified estimate based on fiber orientations
        # For random networks, nu ≈ 0.3-0.4
        # For aligned networks, nu ≈ fiber Poisson's ratio
        
        fibers = self.network.fibers
        if not fibers:
            return 0.3
        
        # Compute orientation distribution
        angles = []
        for fiber in fibers:
            start = fiber.centerline[0]
            end = fiber.centerline[-1]
            direction = end - start
            if np.linalg.norm(direction[:2]) > 1e-10:
                angle = np.arctan2(direction[1], direction[0])
                angles.append(angle)
        
        if not angles:
            return 0.3
        
        # Compute nematic order parameter
        angles = np.array(angles)
        S = np.mean(np.cos(2 * angles))
        
        # Interpolate between random (0.35) and aligned (fiber nu)
        nu_fiber = np.mean([f.material.poissons_ratio for f in fibers if f.material])
        nu_eff = 0.35 * (1 - abs(S)) + nu_fiber * abs(S)
        
        return float(np.clip(nu_eff, 0.0, 0.5))
    
    def effective_shear_modulus(self) -> float:
        """
        Compute effective shear modulus.
        
        Returns
        -------
        G_eff : float
            Effective shear modulus.
        """
        E = self.effective_modulus_2d(direction='x')
        nu = self.effective_poisson_ratio()
        G = E / (2 * (1 + nu))
        return float(G)
    
    def compute_all(self) -> Dict[str, float]:
        """
        Compute all effective elastic properties.
        
        Returns
        -------
        properties : dict
            Dictionary with E_x, E_y, nu, G.
        """
        if self.network.dimension == 2:
            E_x = self.effective_modulus_2d(direction='x')
            E_y = self.effective_modulus_2d(direction='y')
            nu = self.effective_poisson_ratio()
            G = self.effective_shear_modulus()
            
            return {
                'E_x': E_x,
                'E_y': E_y,
                'nu': nu,
                'G': G,
            }
        else:
            # 3D: use simplified estimate
            fibers = self.network.fibers
            if not fibers:
                return {'E': 0.0, 'nu': 0.3, 'G': 0.0}
            
            # Volume-weighted average
            total_volume = sum(np.pi * f.radius**2 * f.length for f in fibers)
            E_sum = sum(f.material.youngs_modulus * np.pi * f.radius**2 * f.length 
                       for f in fibers if f.material)
            
            E = E_sum / total_volume if total_volume > 0 else 0.0
            nu = np.mean([f.material.poissons_ratio for f in fibers if f.material])
            G = E / (2 * (1 + nu))
            
            return {
                'E': E,
                'nu': nu,
                'G': G,
            }


class EffectiveThermalProperties:
    """Compute effective thermal properties."""
    
    def __init__(self, network: FiberNetwork):
        self.network = network
    
    def effective_thermal_conductivity(self) -> float:
        """
        Compute effective thermal conductivity.
        
        Returns
        -------
        k_eff : float
            Effective thermal conductivity (W/(m·K)).
        """
        fibers = self.network.fibers
        if not fibers:
            return 0.0
        
        # Volume-weighted average (parallel model)
        total_volume = sum(np.pi * f.radius**2 * f.length for f in fibers)
        if total_volume == 0:
            return 0.0
        
        k_sum = 0.0
        for fiber in fibers:
            if fiber.material and fiber.material.thermal_conductivity is not None:
                V = np.pi * fiber.radius**2 * fiber.length
                k_sum += fiber.material.thermal_conductivity * V
        
        k_eff = k_sum / total_volume
        
        return float(k_eff)
    
    def effective_thermal_expansion(self) -> float:
        """
        Compute effective coefficient of thermal expansion.
        
        Returns
        -------
        alpha_eff : float
            Effective CTE (1/K).
        """
        fibers = self.network.fibers
        if not fibers:
            return 0.0
        
        # Simplified: volume-weighted average
        total_volume = sum(np.pi * f.radius**2 * f.length for f in fibers)
        if total_volume == 0:
            return 0.0
        
        alpha_sum = 0.0
        for fiber in fibers:
            if fiber.material and fiber.material.thermal_expansion is not None:
                V = np.pi * fiber.radius**2 * fiber.length
                alpha_sum += fiber.material.thermal_expansion * V
        
        alpha_eff = alpha_sum / total_volume
        
        return float(alpha_eff)


class EffectiveElectricalProperties:
    """Compute effective electrical properties."""
    
    def __init__(self, network: FiberNetwork):
        self.network = network
    
    def effective_electrical_conductivity(self) -> float:
        """
        Compute effective electrical conductivity.
        
        Returns
        -------
        sigma_eff : float
            Effective electrical conductivity (S/m).
        """
        fibers = self.network.fibers
        if not fibers:
            return 0.0
        
        # Check if any fibers are conductive
        conductive_fibers = [
            f for f in fibers 
            if f.material and f.material.electrical_conductivity is not None
        ]
        
        if not conductive_fibers:
            return 0.0
        
        # Volume-weighted average
        total_volume = sum(np.pi * f.radius**2 * f.length for f in conductive_fibers)
        if total_volume == 0:
            return 0.0
        
        sigma_sum = sum(
            f.material.electrical_conductivity * np.pi * f.radius**2 * f.length
            for f in conductive_fibers
        )
        
        sigma_eff = sigma_sum / total_volume
        
        return float(sigma_eff)


def compute_effective_properties(network: FiberNetwork) -> Dict:
    """
    Compute all effective properties for a network.
    
    Parameters
    ----------
    network : FiberNetwork
        The network to analyze.
    
    Returns
    -------
    properties : dict
        Dictionary containing all effective properties.
    """
    results = {}
    
    # Elastic properties
    elastic = EffectiveElasticProperties(network)
    results['elastic'] = elastic.compute_all()
    
    # Thermal properties
    thermal = EffectiveThermalProperties(network)
    results['thermal'] = {
        'conductivity': thermal.effective_thermal_conductivity(),
        'expansion': thermal.effective_thermal_expansion(),
    }
    
    # Electrical properties
    electrical = EffectiveElectricalProperties(network)
    results['electrical'] = {
        'conductivity': electrical.effective_electrical_conductivity(),
    }
    
    # Basic info
    results['n_fibers'] = len(network.fibers)
    results['n_crosslinks'] = len(network.crosslinks)
    results['total_length'] = float(network.total_length)
    
    return results
