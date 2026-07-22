"""
Coefficient of Thermal Expansion (CTE) Module

Provides CTE computation for fiber networks:
- Effective CTE tensor
- Thermal strain computation
- Rule of mixtures validation
- Anisotropy analysis

Essential for thermal-mechanical coupling in composites and biomaterials.
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
import warnings

from fibernet.core.network import FiberNetwork


@dataclass
class CTEResult:
    """Result of CTE analysis.
    
    Attributes
    ----------
    cte_tensor : np.ndarray
        3x3 CTE tensor (1/K).
    principal_ctes : np.ndarray
        Principal CTE values.
    effective_cte : float
        Isotropic effective CTE (average of principal values).
    anisotropy_ratio : float
        Ratio of max to min principal CTE.
    rule_of_mixtures : float
        CTE predicted by rule of mixtures.
    """
    cte_tensor: np.ndarray = None
    principal_ctes: np.ndarray = None
    effective_cte: float = 0.0
    anisotropy_ratio: float = 1.0
    rule_of_mixtures: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'cte_tensor': self.cte_tensor.tolist() if self.cte_tensor is not None else None,
            'principal_ctes': self.principal_ctes.tolist() if self.principal_ctes is not None else None,
            'effective_cte': self.effective_cte,
            'anisotropy_ratio': self.anisotropy_ratio,
            'rule_of_mixtures': self.rule_of_mixtures,
        }


class CTEAnalyzer:
    """Analyzer for coefficient of thermal expansion.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to analyze.
    
    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.sim.cte import CTEAnalyzer
    >>> net = gen.random_straight_2d(num_fibers=100, seed=42)
    >>> analyzer = CTEAnalyzer(net)
    >>> result = analyzer.compute_cte()
    >>> print(f"CTE = {result.effective_cte:.2e} 1/K")
    """
    
    def __init__(self, network: FiberNetwork):
        self.network = network
    
    def compute_cte(
        self,
        fiber_cte: float = 5e-6,
        matrix_cte: float = 50e-6,
        fiber_modulus: float = 200e9,
        matrix_modulus: float = 3e9,
    ) -> CTEResult:
        """Compute effective CTE tensor.
        
        Parameters
        ----------
        fiber_cte : float
            CTE of fibers (1/K).
        matrix_cte : float
            CTE of matrix/void (1/K).
        fiber_modulus : float
            Young's modulus of fibers (Pa).
        matrix_modulus : float
            Young's modulus of matrix (Pa).
        
        Returns
        -------
        result : CTEResult
            CTE analysis results.
        
        Notes
        -----
        Uses Schapery's model for anisotropic CTE:
        α_1 = (α_f E_f V_f + α_m E_m V_m) / (E_f V_f + E_m V_m)
        α_2 = (1 + ν_f) α_f V_f + (1 + ν_m) α_m V_m - α_1 (ν_f V_f + ν_m V_m)
        """
        # Compute fiber volume fraction
        V_f = self._compute_fiber_volume_fraction()
        V_m = 1.0 - V_f
        
        # Compute fiber orientation distribution
        orientation_tensor = self._compute_orientation_tensor()
        
        # Longitudinal CTE (along fiber direction)
        E_f = fiber_modulus
        E_m = matrix_modulus
        alpha_f = fiber_cte
        alpha_m = matrix_cte
        
        alpha_1 = (alpha_f * E_f * V_f + alpha_m * E_m * V_m) / (E_f * V_f + E_m * V_m)
        
        # Transverse CTE
        nu_f = 0.3  # Poisson's ratio
        nu_m = 0.35
        alpha_2 = ((1 + nu_f) * alpha_f * V_f + 
                   (1 + nu_m) * alpha_m * V_m - 
                   alpha_1 * (nu_f * V_f + nu_m * V_m))
        
        # Build CTE tensor from orientation
        # α_ij = α_1 * a_ij + α_2 * (δ_ij - a_ij)
        # where a_ij is orientation tensor
        cte_tensor = alpha_1 * orientation_tensor + alpha_2 * (np.eye(3) - orientation_tensor)
        
        # Principal CTEs
        eigenvalues = np.linalg.eigvalsh(cte_tensor)
        eigenvalues = np.sort(eigenvalues)
        
        # Effective (isotropic) CTE
        effective_cte = np.mean(eigenvalues)
        
        # Anisotropy ratio
        if abs(eigenvalues[0]) > 1e-15:
            anisotropy = abs(eigenvalues[-1] / eigenvalues[0])
        else:
            anisotropy = float('inf')
        
        # Rule of mixtures
        rom = alpha_f * V_f + alpha_m * V_m
        
        result = CTEResult(
            cte_tensor=cte_tensor,
            principal_ctes=eigenvalues,
            effective_cte=effective_cte,
            anisotropy_ratio=anisotropy,
            rule_of_mixtures=rom,
        )
        
        return result
    
    def thermal_strain(self, delta_T: float, cte_result: CTEResult = None) -> np.ndarray:
        """Compute thermal strain tensor.
        
        Parameters
        ----------
        delta_T : float
            Temperature change (K).
        cte_result : CTEResult, optional
            Pre-computed CTE result. If None, computes CTE.
        
        Returns
        -------
        strain : np.ndarray
            3x3 thermal strain tensor.
        """
        if cte_result is None:
            cte_result = self.compute_cte()
        
        strain = cte_result.cte_tensor * delta_T
        return strain
    
    def _compute_fiber_volume_fraction(self) -> float:
        """Compute fiber volume fraction."""
        coords = np.array([f.centerline for f in self.network.fibers])
        coords = coords.reshape(-1, 3)
        
        bbox_min = coords.min(axis=0)
        bbox_max = coords.max(axis=0)
        dims = bbox_max - bbox_min
        
        avg_radius = np.mean([f.radius for f in self.network.fibers])
        thickness = max(avg_radius * 10, 1e-6)
        
        for i in range(3):
            if dims[i] < 1e-10:
                dims[i] = thickness
        
        V_total = np.prod(dims)
        
        V_fiber = sum(np.pi * f.radius**2 * f.length for f in self.network.fibers)
        
        if V_total > 0:
            V_f = V_fiber / V_total
        else:
            V_f = 0.0
        
        return max(0.0, min(1.0, V_f))
    
    def _compute_orientation_tensor(self) -> np.ndarray:
        """Compute second-order orientation tensor.
        
        Returns
        -------
        a_ij : np.ndarray
            3x3 orientation tensor.
        
        Notes
        -----
        a_ij = <p_i p_j> where p is unit vector along fiber.
        """
        a = np.zeros((3, 3))
        total_weight = 0.0
        
        for fiber in self.network.fibers:
            # Fiber direction
            p1 = fiber.centerline[0]
            p2 = fiber.centerline[-1]
            direction = p2 - p1
            length = np.linalg.norm(direction)
            
            if length > 1e-10:
                direction = direction / length
                weight = fiber.length  # Weight by fiber length
                
                # Outer product
                a += weight * np.outer(direction, direction)
                total_weight += weight
        
        if total_weight > 0:
            a /= total_weight
        
        return a


def compute_cte(
    network: FiberNetwork,
    fiber_cte: float = 5e-6,
    matrix_cte: float = 50e-6,
    **kwargs,
) -> CTEResult:
    """Convenience function for CTE computation.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to analyze.
    fiber_cte : float
        CTE of fibers (1/K).
    matrix_cte : float
        CTE of matrix (1/K).
    **kwargs : dict
        Additional arguments passed to CTEAnalyzer.
    
    Returns
    -------
    result : CTEResult
        CTE analysis results.
    
    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.sim.cte import compute_cte
    >>> net = gen.random_straight_2d(num_fibers=100, seed=42)
    >>> result = compute_cte(net)
    >>> print(f"CTE = {result.effective_cte:.2e} 1/K")
    """
    analyzer = CTEAnalyzer(network)
    return analyzer.compute_cte(fiber_cte=fiber_cte, matrix_cte=matrix_cte, **kwargs)
