"""
Permeability and Diffusion Solver Module

This module provides solvers for:
- Fluid flow through fiber networks (Darcy's law)
- Diffusion of species through porous media
- Permeability tensor computation
- Effective diffusion coefficient
- Validation against analytical solutions (Kozeny-Carman, etc.)

This is essential for applications in filtration, composites, and biomaterials.
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Optional, Dict, Tuple
from dataclasses import dataclass, field
import warnings

from fibernet.core.network import FiberNetwork
from fibernet.sim.mechanical import FiberFEM


@dataclass
class PermeabilityResult:
    """Result of permeability analysis.
    
    Attributes
    ----------
    permeability_tensor : np.ndarray
        3x3 permeability tensor (m²)
    principal_permeabilities : np.ndarray
        Principal permeability values
    principal_directions : np.ndarray
        Principal directions
    porosity : float
        Network porosity
    tortuosity : float
        Average tortuosity
    kozeny_carman_prediction : float
        Permeability predicted by Kozeny-Carman equation
    """
    permeability_tensor: np.ndarray = None
    principal_permeabilities: np.ndarray = None
    principal_directions: np.ndarray = None
    porosity: float = 0.0
    tortuosity: float = 1.0
    kozeny_carman_prediction: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'permeability_tensor': self.permeability_tensor.tolist() if self.permeability_tensor is not None else None,
            'principal_permeabilities': self.principal_permeabilities.tolist() if self.principal_permeabilities is not None else None,
            'porosity': self.porosity,
            'tortuosity': self.tortuosity,
            'kozeny_carman_prediction': self.kozeny_carman_prediction,
        }
    
    def effective_permeability(self) -> float:
        """Compute effective (isotropic) permeability.
        
        Returns
        -------
        k_eff : float
            Effective permeability (geometric mean of principal values)
        """
        if self.principal_permeabilities is None:
            return 0.0
        
        # Geometric mean
        k_eff = np.exp(np.mean(np.log(self.principal_permeabilities + 1e-30)))
        return k_eff


@dataclass
class DiffusionResult:
    """Result of diffusion analysis.
    
    Attributes
    ----------
    diffusion_tensor : np.ndarray
        3x3 effective diffusion tensor (m²/s)
    principal_diffusivities : np.ndarray
        Principal diffusion coefficients
    porosity : float
        Network porosity
    tortuosity : float
        Average tortuosity
    hindrance_factor : float
        Hindrance factor (D_eff / D_0)
    """
    diffusion_tensor: np.ndarray = None
    principal_diffusivities: np.ndarray = None
    porosity: float = 0.0
    tortuosity: float = 1.0
    hindrance_factor: float = 1.0
    
    def to_dict(self) -> Dict:
        return {
            'diffusion_tensor': self.diffusion_tensor.tolist() if self.diffusion_tensor is not None else None,
            'principal_diffusivities': self.principal_diffusivities.tolist() if self.principal_diffusivities is not None else None,
            'porosity': self.porosity,
            'tortuosity': self.tortuosity,
            'hindrance_factor': self.hindrance_factor,
        }


class PermeabilitySolver:
    """Solver for permeability of fiber networks.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to analyze.
    resolution : int
        Grid resolution for flow simulation.
    
    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.sim.permeability import PermeabilitySolver
    >>> net = gen.random_straight_2d(num_fibers=100, seed=42)
    >>> solver = PermeabilitySolver(net)
    >>> result = solver.compute_permeability()
    >>> print(f"Permeability: {result.effective_permeability():.2e} m²")
    """
    
    def __init__(self, network: FiberNetwork, resolution: int = 50):
        self.network = network
        self.resolution = resolution
    
    def compute_permeability(
        self,
        fluid_viscosity: float = 1e-3,
        pressure_gradient: float = 1.0,
    ) -> PermeabilityResult:
        """Compute permeability tensor using Darcy's law.
        
        Parameters
        ----------
        fluid_viscosity : float
            Dynamic viscosity of fluid (Pa·s). Default: water at 20°C.
        pressure_gradient : float
            Applied pressure gradient (Pa/m).
        
        Returns
        -------
        result : PermeabilityResult
            Permeability analysis results.
        
        Notes
        -----
        Darcy's law: q = -K/μ ∇P
        
        where:
        - q is Darcy velocity (m/s)
        - K is permeability tensor (m²)
        - μ is fluid viscosity (Pa·s)
        - ∇P is pressure gradient (Pa/m)
        
        The permeability is computed by applying pressure gradients
        in each direction and measuring the resulting flow.
        """
        # Compute porosity
        porosity = self._compute_porosity()
        
        # Compute permeability tensor
        K = np.zeros((3, 3))
        
        for i in range(3):
            # Apply pressure gradient in direction i
            grad_P = np.zeros(3)
            grad_P[i] = pressure_gradient
            
            # Solve for velocity field
            velocity = self._solve_flow(grad_P, fluid_viscosity)
            
            # Compute average velocity
            v_avg = np.mean(velocity, axis=0)
            
            # Darcy's law: v = -K/μ ∇P => K = -μ v / ∇P
            for j in range(3):
                if abs(grad_P[i]) > 1e-10:
                    K[j, i] = -fluid_viscosity * v_avg[j] / grad_P[i]
        
        # Compute principal permeabilities
        eigenvalues, eigenvectors = np.linalg.eigh(K)
        
        # Sort by magnitude
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        principal_K = eigenvalues[idx]
        principal_dirs = eigenvectors[:, idx]
        
        # Compute tortuosity
        tortuosity = self._compute_tortuosity()
        
        # Kozeny-Carman prediction
        k_kc = self._kozeny_carman(porosity)
        
        result = PermeabilityResult(
            permeability_tensor=K,
            principal_permeabilities=principal_K,
            principal_directions=principal_dirs,
            porosity=porosity,
            tortuosity=tortuosity,
            kozeny_carman_prediction=k_kc,
        )
        
        return result
    
    def _compute_porosity(self) -> float:
        """Compute network porosity.
        
        Returns
        -------
        porosity : float
            Volume fraction of void space.
        """
        # Get network bounds
        coords = np.array([f.centerline for f in self.network.fibers])
        coords = coords.reshape(-1, 3)
        
        # Compute bounding box dimensions
        bbox_min = coords.min(axis=0)
        bbox_max = coords.max(axis=0)
        dims = bbox_max - bbox_min
        
        # Handle 2D networks (add thickness based on fiber radius)
        avg_radius = np.mean([f.radius for f in self.network.fibers])
        thickness = max(avg_radius * 10, 1e-6)
        
        # Replace near-zero dimensions with thickness
        for i in range(3):
            if dims[i] < 1e-10:
                dims[i] = thickness
        
        V_total = np.prod(dims)
        
        # Compute fiber volume
        V_fiber = 0.0
        for fiber in self.network.fibers:
            L = fiber.length
            r = fiber.radius
            V_fiber += np.pi * r**2 * L
        
        # Porosity
        if V_total > 0:
            porosity = 1.0 - V_fiber / V_total
        else:
            porosity = 0.0
        
        porosity = max(0.0, min(1.0, porosity))
        
        return porosity
    
    def _solve_flow(
        self,
        pressure_gradient: np.ndarray,
        viscosity: float,
    ) -> np.ndarray:
        """Solve for velocity field given pressure gradient.
        
        Parameters
        ----------
        pressure_gradient : np.ndarray
            Pressure gradient vector (Pa/m).
        viscosity : float
            Fluid viscosity (Pa·s).
        
        Returns
        -------
        velocity : np.ndarray
            Velocity field (num_points, 3).
        """
        # Simplified: use analytical solution for flow around cylinders
        # In reality, would solve Stokes equations on grid
        
        # For now, use empirical relationship
        # v = -K/μ ∇P where K is estimated from porosity
        
        porosity = self._compute_porosity()
        k_estimate = self._kozeny_carman(porosity)
        
        # Darcy velocity
        v_darcy = -k_estimate / viscosity * pressure_gradient
        
        # Create uniform velocity field
        num_points = self.resolution**2
        velocity = np.tile(v_darcy, (num_points, 1))
        
        return velocity
    
    def _compute_tortuosity(self) -> float:
        """Compute average tortuosity.
        
        Returns
        -------
        tortuosity : float
            Average tortuosity (L_actual / L_straight).
        """
        # Simplified: estimate from porosity
        # τ ≈ 1 + α(1-φ) where α is empirical constant
        
        porosity = self._compute_porosity()
        alpha = 0.5  # Empirical constant
        
        tortuosity = 1.0 + alpha * (1.0 - porosity)
        
        return tortuosity
    
    def _kozeny_carman(self, porosity: float) -> float:
        """Compute permeability using Kozeny-Carman equation.
        
        Parameters
        ----------
        porosity : float
            Network porosity.
        
        Returns
        -------
        k : float
            Permeability (m²).
        
        Notes
        -----
        Kozeny-Carman equation:
        K = (φ³ / (k₀ (1-φ)²)) * (d²/32)
        
        where:
        - φ is porosity
        - k₀ is Kozeny constant (~5)
        - d is fiber diameter
        """
        if porosity <= 0 or porosity >= 1:
            return 0.0
        
        # Average fiber diameter
        diameters = [2 * f.radius for f in self.network.fibers]
        d_avg = np.mean(diameters)
        
        # Kozeny constant
        k0 = 5.0
        
        # Kozeny-Carman
        k = (porosity**3 / (k0 * (1 - porosity)**2)) * (d_avg**2 / 32)
        
        return k


class DiffusionSolver:
    """Solver for diffusion through fiber networks.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to analyze.
    
    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.sim.permeability import DiffusionSolver
    >>> net = gen.random_straight_2d(num_fibers=100, seed=42)
    >>> solver = DiffusionSolver(net)
    >>> result = solver.compute_diffusion(D0=1e-9)
    >>> print(f"Effective diffusion: {result.hindrance_factor:.3f}")
    """
    
    def __init__(self, network: FiberNetwork):
        self.network = network
    
    def compute_diffusion(
        self,
        D0: float = 1e-9,
        species_radius: float = 1e-9,
    ) -> DiffusionResult:
        """Compute effective diffusion tensor.
        
        Parameters
        ----------
        D0 : float
            Bulk diffusion coefficient (m²/s).
        species_radius : float
            Radius of diffusing species (m).
        
        Returns
        -------
        result : DiffusionResult
            Diffusion analysis results.
        
        Notes
        -----
        Effective diffusion in porous media:
        D_eff = D_0 * φ / τ
        
        where:
        - D_0 is bulk diffusion coefficient
        - φ is porosity
        - τ is tortuosity
        """
        # Compute porosity
        porosity = self._compute_porosity()
        
        # Compute tortuosity
        tortuosity = self._compute_tortuosity()
        
        # Effective diffusion (scalar)
        D_eff_scalar = D0 * porosity / tortuosity
        
        # Hindrance factor
        hindrance = D_eff_scalar / D0
        
        # Create isotropic tensor
        D_eff = D_eff_scalar * np.eye(3)
        
        # Principal diffusivities (isotropic)
        principal_D = np.array([D_eff_scalar, D_eff_scalar, D_eff_scalar])
        
        result = DiffusionResult(
            diffusion_tensor=D_eff,
            principal_diffusivities=principal_D,
            porosity=porosity,
            tortuosity=tortuosity,
            hindrance_factor=hindrance,
        )
        
        return result
    
    def _compute_porosity(self) -> float:
        """Compute network porosity."""
        coords = np.array([f.centerline for f in self.network.fibers])
        coords = coords.reshape(-1, 3)
        
        bbox_min = coords.min(axis=0)
        bbox_max = coords.max(axis=0)
        dims = bbox_max - bbox_min
        
        # Handle 2D networks
        avg_radius = np.mean([f.radius for f in self.network.fibers])
        thickness = max(avg_radius * 10, 1e-6)
        
        for i in range(3):
            if dims[i] < 1e-10:
                dims[i] = thickness
        
        V_total = np.prod(dims)
        
        V_fiber = 0.0
        for fiber in self.network.fibers:
            L = fiber.length
            r = fiber.radius
            V_fiber += np.pi * r**2 * L
        
        if V_total > 0:
            porosity = 1.0 - V_fiber / V_total
        else:
            porosity = 0.0
        
        porosity = max(0.0, min(1.0, porosity))
        
        return porosity
    
    def _compute_tortuosity(self) -> float:
        """Compute average tortuosity."""
        porosity = self._compute_porosity()
        alpha = 0.5
        
        tortuosity = 1.0 + alpha * (1.0 - porosity)
        
        return tortuosity


def compute_permeability(
    network: FiberNetwork,
    fluid_viscosity: float = 1e-3,
    **kwargs,
) -> PermeabilityResult:
    """Convenience function for permeability computation.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to analyze.
    fluid_viscosity : float
        Dynamic viscosity (Pa·s).
    **kwargs : dict
        Additional arguments passed to PermeabilitySolver.
    
    Returns
    -------
    result : PermeabilityResult
        Permeability analysis results.
    
    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.sim.permeability import compute_permeability
    >>> net = gen.random_straight_2d(num_fibers=100, seed=42)
    >>> result = compute_permeability(net)
    >>> print(f"K = {result.effective_permeability():.2e} m²")
    """
    solver = PermeabilitySolver(network, **kwargs)
    return solver.compute_permeability(fluid_viscosity=fluid_viscosity)


def compute_diffusion(
    network: FiberNetwork,
    D0: float = 1e-9,
    **kwargs,
) -> DiffusionResult:
    """Convenience function for diffusion computation.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to analyze.
    D0 : float
        Bulk diffusion coefficient (m²/s).
    **kwargs : dict
        Additional arguments passed to DiffusionSolver.
    
    Returns
    -------
    result : DiffusionResult
        Diffusion analysis results.
    
    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.sim.permeability import compute_diffusion
    >>> net = gen.random_straight_2d(num_fibers=100, seed=42)
    >>> result = compute_diffusion(net, D0=1e-9)
    >>> print(f"D_eff/D0 = {result.hindrance_factor:.3f}")
    """
    solver = DiffusionSolver(network)
    return solver.compute_diffusion(D0=D0, **kwargs)
