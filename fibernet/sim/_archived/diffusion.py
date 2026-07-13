"""
Diffusion and Transport Module for Fiber Networks

Provides tools for mass transport analysis:
- Effective diffusion coefficient
- Tortuosity calculation
- Permeability analysis
- Concentration profile simulation
- Fiber network as filtration media

References:
- Carman, P.C., "Flow of Gases Through Porous Media", Academic Press, 1956
- Dullien, F.A.L., "Porous Media: Fluid Transport and Pore Structure", Academic Press, 1992
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy import sparse
from scipy.sparse.linalg import spsolve
import warnings

from fibernet.core.network import FiberNetwork


@dataclass
class DiffusionResult:
    """Result of diffusion analysis."""
    effective_diffusion_coefficient: float = 0.0  # m²/s
    tortuosity: float = 0.0
    porosity: float = 0.0
    concentration_profile: Optional[np.ndarray] = None
    time_points: Optional[np.ndarray] = None
    breakthrough_time: float = 0.0  # s
    
    def to_dict(self) -> Dict:
        return {
            'effective_diffusion_coefficient': self.effective_diffusion_coefficient,
            'tortuosity': self.tortuosity,
            'porosity': self.porosity,
            'breakthrough_time': self.breakthrough_time,
        }


@dataclass
class FiltrationResult:
    """Result of filtration analysis."""
    capture_efficiency: float = 0.0  # fraction
    pressure_drop: float = 0.0  # Pa
    filtration_velocity: float = 0.0  # m/s
    particle_trajectory: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        return {
            'capture_efficiency': self.capture_efficiency,
            'pressure_drop': self.pressure_drop,
            'filtration_velocity': self.filtration_velocity,
        }


class DiffusionAnalyzer:
    """Analyze diffusion and transport through fiber networks.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to analyze.
    molecular_diffusion : float, optional
        Molecular diffusion coefficient (m²/s).
        Default: 1e-9 (typical for small molecules in water).
    porosity : float, optional
        Network porosity (0-1).
        Default: computed from fiber volume fraction.
    
    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.sim.diffusion import DiffusionAnalyzer
    >>> net = gen.random_straight_2d(num_fibers=50, seed=42)
    >>> diff = DiffusionAnalyzer(net)
    >>> result = diff.compute_effective_diffusion()
    >>> print(f"Effective D: {result.effective_diffusion_coefficient:.2e} m²/s")
    """
    
    def __init__(
        self,
        network: FiberNetwork,
        molecular_diffusion: float = 1e-9,
        porosity: Optional[float] = None,
    ):
        self.network = network
        self.D_mol = molecular_diffusion
        
        # Compute porosity if not provided
        if porosity is None:
            self.porosity = self._compute_porosity()
        else:
            self.porosity = porosity
    
    def _compute_porosity(self) -> float:
        """Compute porosity from fiber volume fraction."""
        # Estimate fiber volume
        fiber_volume = 0.0
        for fiber in self.network.fibers:
            # Volume = pi * r^2 * L
            V_fiber = np.pi * fiber.radius**2 * fiber.length
            fiber_volume += V_fiber
        
        # Box volume
        bb_min, bb_max = self.network.bounding_box()
        box_dims = bb_max - bb_min
        box_volume = np.prod(box_dims[box_dims > 0])
        
        if box_volume > 0:
            fiber_fraction = fiber_volume / box_volume
            porosity = 1.0 - fiber_fraction
            return max(0.0, min(1.0, porosity))
        else:
            return 0.5  # Default
    
    def compute_effective_diffusion(
        self,
        direction: int = 0,
    ) -> DiffusionResult:
        """Compute effective diffusion coefficient.
        
        Parameters
        ----------
        direction : int
            Diffusion direction (0=x, 1=y, 2=z).
        
        Returns
        -------
        result : DiffusionResult
            Diffusion analysis results.
        """
        # Tortuosity model: D_eff = D_mol * porosity / tortuosity
        # Bruggeman correlation: tortuosity = porosity^(-0.5)
        tortuosity = self.porosity ** (-0.5) if self.porosity > 0 else 1.0
        
        # Effective diffusion
        D_eff = self.D_mol * self.porosity / tortuosity
        
        return DiffusionResult(
            effective_diffusion_coefficient=D_eff,
            tortuosity=tortuosity,
            porosity=self.porosity,
        )
    
    def simulate_concentration_profile(
        self,
        initial_concentration: float = 1.0,
        duration: float = 3600.0,
        num_time_steps: int = 100,
        num_space_steps: int = 50,
        direction: int = 0,
    ) -> DiffusionResult:
        """Simulate concentration profile evolution.
        
        Parameters
        ----------
        initial_concentration : float
            Initial concentration at x=0.
        duration : float
            Simulation duration (s).
        num_time_steps : int
            Number of time steps.
        num_space_steps : int
            Number of spatial steps.
        direction : int
            Diffusion direction.
        
        Returns
        -------
        result : DiffusionResult
            Concentration profile data.
        """
        # Compute effective diffusion
        diff_result = self.compute_effective_diffusion(direction)
        D_eff = diff_result.effective_diffusion_coefficient
        
        # Get domain size
        bb_min, bb_max = self.network.bounding_box()
        L = bb_max[direction] - bb_min[direction]
        
        # Spatial grid
        x = np.linspace(0, L, num_space_steps)
        dx = x[1] - x[0]
        
        # Time grid
        t = np.linspace(0, duration, num_time_steps)
        dt = t[1] - t[0]
        
        # Stability check
        if D_eff * dt / dx**2 > 0.5:
            warnings.warn(f"Diffusion may be unstable. Consider smaller dt.")
        
        # Initial condition: C(x,0) = C0 for x=0, 0 elsewhere
        C = np.zeros((num_time_steps, num_space_steps))
        C[0, 0] = initial_concentration
        
        # Finite difference: C[i,j+1] = C[i,j] + D*dt/dx^2 * (C[i+1,j] - 2*C[i,j] + C[i-1,j])
        alpha = D_eff * dt / dx**2
        
        for ti in range(num_time_steps - 1):
            for xi in range(1, num_space_steps - 1):
                C[ti+1, xi] = C[ti, xi] + alpha * (
                    C[ti, xi+1] - 2*C[ti, xi] + C[ti, xi-1]
                )
            
            # Boundary conditions
            C[ti+1, 0] = initial_concentration  # Fixed at x=0
            C[ti+1, -1] = C[ti+1, -2]  # Zero flux at x=L
        
        # Breakthrough time (when concentration at x=L reaches 10% of C0)
        breakthrough_threshold = 0.1 * initial_concentration
        breakthrough_indices = np.where(C[:, -1] >= breakthrough_threshold)[0]
        breakthrough_time = t[breakthrough_indices[0]] if len(breakthrough_indices) > 0 else duration
        
        return DiffusionResult(
            effective_diffusion_coefficient=D_eff,
            tortuosity=diff_result.tortuosity,
            porosity=diff_result.porosity,
            concentration_profile=C,
            time_points=t,
            breakthrough_time=breakthrough_time,
        )
    
    def compute_tortuosity(
        self,
        method: str = 'bruggeman',
    ) -> float:
        """Compute tortuosity factor.
        
        Parameters
        ----------
        method : str
            Tortuosity model: 'bruggeman', 'comiti', 'weissberg'.
        
        Returns
        -------
        tortuosity : float
            Tortuosity factor.
        """
        phi = self.porosity
        
        if method == 'bruggeman':
            # Bruggeman: tau = phi^(-0.5)
            tau = phi ** (-0.5) if phi > 0 else 1.0
        elif method == 'comiti':
            # Comiti & Chang: tau = 1 + 0.5*(1-phi)
            tau = 1.0 + 0.5 * (1.0 - phi)
        elif method == 'weissberg':
            # Weissberg: tau = 1 - 0.49*ln(phi)
            tau = 1.0 - 0.49 * np.log(phi) if phi > 0 else 1.0
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return tau
    
    def filtration_analysis(
        self,
        particle_size: float = 1e-6,
        flow_velocity: float = 0.01,
        fluid_viscosity: float = 1e-3,
        num_particles: int = 100,
    ) -> FiltrationResult:
        """Analyze filtration performance.
        
        Parameters
        ----------
        particle_size : float
            Particle diameter (m).
        flow_velocity : float
            Superficial flow velocity (m/s).
        fluid_viscosity : float
            Fluid dynamic viscosity (Pa·s).
        num_particles : int
            Number of particles to simulate.
        
        Returns
        -------
        result : FiltrationResult
            Filtration analysis results.
        """
        # Estimate fiber diameter
        fiber_diameters = [2 * f.radius for f in self.network.fibers]
        avg_fiber_diameter = np.mean(fiber_diameters)
        
        # Single fiber efficiency (interception)
        # eta_R = (1 + R)^(-1) - (1 + R)
        R = particle_size / avg_fiber_diameter
        eta_R = (1.0 + R)**(-1) - (1.0 + R)
        eta_R = max(0, eta_R)
        
        # Inertial impaction (Stokes number)
        # Stk = rho_p * d_p^2 * v / (18 * mu * d_f)
        rho_p = 1000.0  # Particle density (kg/m³)
        Stk = rho_p * particle_size**2 * flow_velocity / (18 * fluid_viscosity * avg_fiber_diameter)
        eta_I = Stk**2 / (Stk + 0.77)**2 if Stk > 0 else 0
        
        # Total single fiber efficiency
        eta_total = eta_R + eta_I
        
        # Filter efficiency (log penetration model)
        # E = 1 - exp(-4 * alpha * eta * L / (pi * d_f))
        alpha = 1.0 - self.porosity  # Solid volume fraction
        bb_min, bb_max = self.network.bounding_box()
        L = bb_max[0] - bb_min[0]  # Filter thickness
        
        E = 1.0 - np.exp(-4 * alpha * eta_total * L / (np.pi * avg_fiber_diameter))
        
        # Pressure drop (Darcy's law)
        # Delta P = mu * v * L / K
        # K = d_f^2 * phi^3 / (180 * (1-phi)^2) (Kozeny-Carman)
        K = avg_fiber_diameter**2 * self.porosity**3 / (180 * (1.0 - self.porosity)**2)
        delta_P = fluid_viscosity * flow_velocity * L / K if K > 0 else 0
        
        return FiltrationResult(
            capture_efficiency=E,
            pressure_drop=delta_P,
            filtration_velocity=flow_velocity,
        )


def analyze_diffusion(
    network: FiberNetwork,
    molecular_diffusion: float = 1e-9,
    **kwargs,
) -> DiffusionResult:
    """Convenience function for diffusion analysis.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to analyze.
    molecular_diffusion : float
        Molecular diffusion coefficient (m²/s).
    
    Returns
    -------
    result : DiffusionResult
        Diffusion analysis results.
    
    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.sim.diffusion import analyze_diffusion
    >>> net = gen.random_straight_2d(num_fibers=50, seed=42)
    >>> result = analyze_diffusion(net)
    >>> print(f"Tortuosity: {result.tortuosity:.2f}")
    """
    analyzer = DiffusionAnalyzer(network, molecular_diffusion=molecular_diffusion, **kwargs)
    return analyzer.compute_effective_diffusion()
