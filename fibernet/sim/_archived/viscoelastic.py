"""
Viscoelastic material models for time-dependent simulations.

Implements common viscoelastic models:
- Maxwell model (spring + dashpot in series)
- Kelvin-Voigt model (spring + dashpot in parallel)
- Standard Linear Solid (SLS) model
- Generalized Maxwell (Prony series)
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass, field


@dataclass
class ViscoelasticResult:
    """
    Container for viscoelastic simulation results.
    
    Attributes
    ----------
    time : np.ndarray
        Time values (s)
    strain : np.ndarray
        Strain values
    stress : np.ndarray
        Stress values (Pa)
    strain_rate : np.ndarray
        Strain rate (1/s)
    """
    time: np.ndarray
    strain: np.ndarray
    stress: np.ndarray
    strain_rate: np.ndarray
    
    def plot(self, ax=None, **kwargs):
        """Plot stress vs time."""
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(self.time, self.stress / 1e6, **kwargs)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Stress (MPa)')
        ax.set_title('Viscoelastic Response')
        ax.grid(True, alpha=0.3)
        
        return ax


class MaxwellModel:
    """
    Maxwell viscoelastic model (spring + dashpot in series).
    
    σ̇ = E·ε̇ - (E/η)·σ
    
    Good for stress relaxation.
    
    Parameters
    ----------
    E : float
        Young's modulus (Pa)
    eta : float
        Viscosity (Pa·s)
    """
    
    def __init__(self, E: float, eta: float):
        self.E = E
        self.eta = eta
        self.tau = eta / E  # Relaxation time
    
    def stress_relaxation(
        self,
        strain: float,
        time_range: Tuple[float, float] = (0, 10),
        num_steps: int = 100
    ) -> ViscoelasticResult:
        """
        Simulate stress relaxation at constant strain.
        
        Parameters
        ----------
        strain : float
            Applied constant strain
        time_range : tuple
            (t_start, t_end)
        num_steps : int
            Number of time steps
        
        Returns
        -------
        ViscoelasticResult
        """
        time = np.linspace(time_range[0], time_range[1], num_steps)
        
        # σ(t) = E·ε·exp(-t/τ)
        stress = self.E * strain * np.exp(-time / self.tau)
        strain_arr = np.full_like(time, strain)
        strain_rate = np.zeros_like(time)
        
        return ViscoelasticResult(
            time=time,
            strain=strain_arr,
            stress=stress,
            strain_rate=strain_rate
        )
    
    def creep(
        self,
        stress: float,
        time_range: Tuple[float, float] = (0, 10),
        num_steps: int = 100
    ) -> ViscoelasticResult:
        """
        Simulate creep at constant stress.
        
        Parameters
        ----------
        stress : float
            Applied constant stress (Pa)
        time_range : tuple
            (t_start, t_end)
        num_steps : int
            Number of time steps
        
        Returns
        -------
        ViscoelasticResult
        """
        time = np.linspace(time_range[0], time_range[1], num_steps)
        
        # ε(t) = σ/E + σ·t/η
        strain = stress / self.E + stress * time / self.eta
        stress_arr = np.full_like(time, stress)
        strain_rate = np.gradient(strain, time)
        
        return ViscoelasticResult(
            time=time,
            strain=strain,
            stress=stress_arr,
            strain_rate=strain_rate
        )


class KelvinVoigtModel:
    """
    Kelvin-Voigt viscoelastic model (spring + dashpot in parallel).
    
    σ = E·ε + η·ε̇
    
    Good for creep behavior.
    
    Parameters
    ----------
    E : float
        Young's modulus (Pa)
    eta : float
        Viscosity (Pa·s)
    """
    
    def __init__(self, E: float, eta: float):
        self.E = E
        self.eta = eta
        self.tau = eta / E  # Retardation time
    
    def creep(
        self,
        stress: float,
        time_range: Tuple[float, float] = (0, 10),
        num_steps: int = 100
    ) -> ViscoelasticResult:
        """
        Simulate creep at constant stress.
        
        Parameters
        ----------
        stress : float
            Applied constant stress (Pa)
        time_range : tuple
            (t_start, t_end)
        num_steps : int
            Number of time steps
        
        Returns
        -------
        ViscoelasticResult
        """
        time = np.linspace(time_range[0], time_range[1], num_steps)
        
        # ε(t) = (σ/E)·(1 - exp(-t/τ))
        strain = (stress / self.E) * (1 - np.exp(-time / self.tau))
        stress_arr = np.full_like(time, stress)
        strain_rate = np.gradient(strain, time)
        
        return ViscoelasticResult(
            time=time,
            strain=strain,
            stress=stress_arr,
            strain_rate=strain_rate
        )


class StandardLinearSolid:
    """
    Standard Linear Solid (SLS) model.
    
    Combines instantaneous elastic response with time-dependent relaxation.
    
    Parameters
    ----------
    E1 : float
        Instantaneous modulus (Pa)
    E2 : float
        Equilibrium modulus (Pa)
    eta : float
        Viscosity (Pa·s)
    """
    
    def __init__(self, E1: float, E2: float, eta: float):
        self.E1 = E1
        self.E2 = E2
        self.eta = eta
        self.tau = eta / E2  # Relaxation time
        self.E_instant = E1 + E2  # Instantaneous modulus
        self.E_equilibrium = E1  # Equilibrium modulus
    
    def stress_relaxation(
        self,
        strain: float,
        time_range: Tuple[float, float] = (0, 10),
        num_steps: int = 100
    ) -> ViscoelasticResult:
        """
        Simulate stress relaxation at constant strain.
        
        Parameters
        ----------
        strain : float
            Applied constant strain
        time_range : tuple
            (t_start, t_end)
        num_steps : int
            Number of time steps
        
        Returns
        -------
        ViscoelasticResult
        """
        time = np.linspace(time_range[0], time_range[1], num_steps)
        
        # σ(t) = E1·ε + E2·ε·exp(-t/τ)
        stress = self.E1 * strain + self.E2 * strain * np.exp(-time / self.tau)
        strain_arr = np.full_like(time, strain)
        strain_rate = np.zeros_like(time)
        
        return ViscoelasticResult(
            time=time,
            strain=strain_arr,
            stress=stress,
            strain_rate=strain_rate
        )


class GeneralizedMaxwell:
    """
    Generalized Maxwell model (Prony series).
    
    Multiple Maxwell elements in parallel for complex relaxation spectra.
    
    Parameters
    ----------
    E_inf : float
        Equilibrium modulus (Pa)
    E_i : list of float
        Moduli for each Maxwell element (Pa)
    tau_i : list of float
        Relaxation times for each element (s)
    """
    
    def __init__(self, E_inf: float, E_i: List[float], tau_i: List[float]):
        if len(E_i) != len(tau_i):
            raise ValueError("E_i and tau_i must have same length")
        
        self.E_inf = E_inf
        self.E_i = np.array(E_i)
        self.tau_i = np.array(tau_i)
        self.E_instant = E_inf + np.sum(E_i)
    
    def stress_relaxation(
        self,
        strain: float,
        time_range: Tuple[float, float] = (0, 10),
        num_steps: int = 100
    ) -> ViscoelasticResult:
        """
        Simulate stress relaxation at constant strain.
        
        Parameters
        ----------
        strain : float
            Applied constant strain
        time_range : tuple
            (t_start, t_end)
        num_steps : int
            Number of time steps
        
        Returns
        -------
        ViscoelasticResult
        """
        time = np.linspace(time_range[0], time_range[1], num_steps)
        
        # σ(t) = E_inf·ε + Σ E_i·ε·exp(-t/τ_i)
        stress = self.E_inf * strain
        for E_i, tau_i in zip(self.E_i, self.tau_i):
            stress += E_i * strain * np.exp(-time / tau_i)
        
        strain_arr = np.full_like(time, strain)
        strain_rate = np.zeros_like(time)
        
        return ViscoelasticResult(
            time=time,
            strain=strain_arr,
            stress=stress,
            strain_rate=strain_rate
        )
