"""
Creep Analysis Module for Fiber Networks

Provides tools for time-dependent deformation analysis:
- Creep compliance curves
- Creep-recovery simulations
- Time-temperature superposition
- Burger's model fitting
- Steady-state creep rate

References:
- Findley, W.N. et al., "Creep and Relaxation of Nonlinear Viscoelastic Materials", Dover, 1989
- Ward, I.M. & Sweeney, J., "Mechanical Properties of Solid Polymers", Wiley, 2012
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy.optimize import curve_fit
import warnings

from fibernet.core.network import FiberNetwork
from fibernet.sim import FiberFEM


@dataclass
class CreepResult:
    """Result of creep analysis."""
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    strain: np.ndarray = field(default_factory=lambda: np.array([]))
    creep_compliance: np.ndarray = field(default_factory=lambda: np.array([]))
    applied_stress: float = 0.0  # Pa
    instantaneous_strain: float = 0.0
    steady_state_rate: float = 0.0  # 1/s
    total_strain: float = 0.0
    recovery_strain: float = 0.0  # After unloading
    
    def to_dict(self) -> Dict:
        return {
            'num_time_points': len(self.time),
            'applied_stress': self.applied_stress,
            'instantaneous_strain': self.instantaneous_strain,
            'steady_state_rate': self.steady_state_rate,
            'total_strain': self.total_strain,
            'recovery_strain': self.recovery_strain,
        }


@dataclass
class CreepModelParameters:
    """Parameters for creep models."""
    # Burger's model parameters
    E1: float = 0.0  # Instantaneous modulus (Pa)
    E2: float = 0.0  # Delayed elastic modulus (Pa)
    eta1: float = 0.0  # Viscosity of dashpot 1 (Pa·s)
    eta2: float = 0.0  # Viscosity of dashpot 2 (Pa·s)
    
    def to_dict(self) -> Dict:
        return {
            'E1': self.E1,
            'E2': self.E2,
            'eta1': self.eta1,
            'eta2': self.eta2,
        }


class CreepAnalyzer:
    """Analyze creep behavior of fiber networks.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to analyze.
    instantaneous_modulus : float, optional
        Instantaneous elastic modulus (Pa).
    viscosity : float, optional
        Viscosity parameter (Pa·s).
    temperature : float, optional
        Temperature (K).
    
    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.sim.creep import CreepAnalyzer
    >>> net = gen.random_straight_2d(num_fibers=50, seed=42)
    >>> creep = CreepAnalyzer(net)
    >>> result = creep.creep_test(stress=1e6, duration=3600)
    >>> print(f"Steady-state rate: {result.steady_state_rate:.2e} 1/s")
    """
    
    def __init__(
        self,
        network: FiberNetwork,
        instantaneous_modulus: Optional[float] = None,
        viscosity: float = 1e10,
        temperature: float = 293.15,
    ):
        self.network = network
        self.fem = FiberFEM(network)
        
        # Compute instantaneous modulus if not provided
        if instantaneous_modulus is None:
            self.E0 = self.fem.effective_modulus()
        else:
            self.E0 = instantaneous_modulus
        
        self.eta = viscosity
        self.T = temperature
    
    def creep_test(
        self,
        stress: float,
        duration: float,
        num_points: int = 100,
    ) -> CreepResult:
        """Perform creep test under constant stress.
        
        Parameters
        ----------
        stress : float
            Applied stress (Pa).
        duration : float
            Test duration (s).
        num_points : int
            Number of time points.
        
        Returns
        -------
        result : CreepResult
            Creep test results.
        """
        # Time array
        t = np.linspace(0, duration, num_points)
        
        # Burger's model: J(t) = 1/E1 + t/eta1 + (1/E2)(1 - exp(-E2*t/eta2))
        # Simplified: J(t) = 1/E0 + t/eta (linear creep)
        compliance = 1.0 / self.E0 + t / self.eta
        
        # Strain = stress * compliance
        strain = stress * compliance
        
        # Compute steady-state creep rate
        # For linear creep: rate = stress / eta
        steady_state_rate = stress / self.eta
        
        return CreepResult(
            time=t,
            strain=strain,
            creep_compliance=compliance,
            applied_stress=stress,
            instantaneous_strain=strain[0],
            steady_state_rate=steady_state_rate,
            total_strain=strain[-1],
        )
    
    def creep_recovery(
        self,
        stress: float,
        creep_duration: float,
        recovery_duration: float,
        num_points: int = 100,
    ) -> Tuple[CreepResult, CreepResult]:
        """Perform creep-recovery test.
        
        Parameters
        ----------
        stress : float
            Applied stress (Pa).
        creep_duration : float
            Duration of creep phase (s).
        recovery_duration : float
            Duration of recovery phase (s).
        num_points : int
            Number of time points per phase.
        
        Returns
        -------
        creep_result : CreepResult
            Creep phase results.
        recovery_result : CreepResult
            Recovery phase results.
        """
        # Creep phase
        creep_result = self.creep_test(stress, creep_duration, num_points)
        
        # Recovery phase
        t_recovery = np.linspace(0, recovery_duration, num_points)
        
        # For linear creep: recovery = elastic + viscous (irreversible)
        # Elastic recovery happens immediately
        # Viscous strain remains
        elastic_strain = stress / self.E0
        viscous_strain = stress * creep_duration / self.eta
        
        # During recovery, only elastic part recovers
        recovery_strain = np.ones_like(t_recovery) * viscous_strain
        
        recovery_compliance = recovery_strain / stress if stress > 0 else np.zeros_like(t_recovery)
        
        recovery_result = CreepResult(
            time=t_recovery,
            strain=recovery_strain,
            creep_compliance=recovery_compliance,
            applied_stress=0.0,
            instantaneous_strain=creep_result.total_strain,
            steady_state_rate=0.0,
            total_strain=recovery_strain[-1],
            recovery_strain=elastic_strain,
        )
        
        return creep_result, recovery_result
    
    def fit_burgers_model(
        self,
        time: np.ndarray,
        strain: np.ndarray,
        stress: float,
    ) -> CreepModelParameters:
        """Fit Burger's model to creep data.
        
        Parameters
        ----------
        time : np.ndarray
            Time array (s).
        strain : np.ndarray
            Strain array.
        stress : float
            Applied stress (Pa).
        
        Returns
        -------
        params : CreepModelParameters
            Fitted model parameters.
        """
        # Burger's model: epsilon(t) = sigma * [1/E1 + t/eta1 + (1/E2)(1 - exp(-E2*t/eta2))]
        def burgers_model(t, E1, E2, eta1, eta2):
            compliance = 1.0/E1 + t/eta1 + (1.0/E2)*(1.0 - np.exp(-E2*t/eta2))
            return stress * compliance
        
        # Initial guesses
        E1_guess = self.E0
        E2_guess = self.E0 * 0.5
        eta1_guess = self.eta
        eta2_guess = self.eta * 0.5
        
        try:
            popt, _ = curve_fit(
                burgers_model,
                time,
                strain,
                p0=[E1_guess, E2_guess, eta1_guess, eta2_guess],
                bounds=(0, np.inf),
                maxfev=1000,
            )
            
            return CreepModelParameters(
                E1=popt[0],
                E2=popt[1],
                eta1=popt[2],
                eta2=popt[3],
            )
        except Exception as e:
            warnings.warn(f"Burger's model fitting failed: {e}")
            return CreepModelParameters(
                E1=self.E0,
                E2=self.E0 * 0.5,
                eta1=self.eta,
                eta2=self.eta * 0.5,
            )
    
    def time_temperature_superposition(
        self,
        reference_temperature: float,
        temperatures: List[float],
        shift_factor: str = 'wlf',
    ) -> Dict[str, np.ndarray]:
        """Compute time-temperature superposition shift factors.
        
        Parameters
        ----------
        reference_temperature : float
            Reference temperature (K).
        temperatures : list
            List of temperatures (K).
        shift_factor : str
            Shift factor model: 'wlf' (Williams-Landel-Ferry) or 'arrhenius'.
        
        Returns
        -------
        data : dict
            Shift factors and master curve data.
        """
        T = np.array(temperatures)
        T_ref = reference_temperature
        
        if shift_factor == 'wlf':
            # WLF equation: log(a_T) = -C1*(T-T_ref) / (C2 + T - T_ref)
            # Typical values: C1=17.44, C2=51.6 K
            C1 = 17.44
            C2 = 51.6
            log_aT = -C1 * (T - T_ref) / (C2 + T - T_ref)
            aT = 10 ** log_aT
        else:
            # Arrhenius: a_T = exp(Ea/R * (1/T - 1/T_ref))
            Ea = 50000  # Activation energy (J/mol)
            R = 8.314  # Gas constant
            aT = np.exp(Ea / R * (1.0/T - 1.0/T_ref))
        
        return {
            'temperatures': T,
            'reference_temperature': T_ref,
            'shift_factors': aT,
            'log_shift_factors': np.log10(aT),
        }
    
    def stress_relaxation(
        self,
        initial_strain: float,
        duration: float,
        num_points: int = 100,
    ) -> Dict[str, np.ndarray]:
        """Simulate stress relaxation under constant strain.
        
        Parameters
        ----------
        initial_strain : float
            Applied strain.
        duration : float
            Relaxation duration (s).
        num_points : int
            Number of time points.
        
        Returns
        -------
        data : dict
            Stress relaxation data.
        """
        t = np.linspace(0, duration, num_points)
        
        # Maxwell model: sigma(t) = E * epsilon * exp(-t/tau)
        # tau = eta / E
        tau = self.eta / self.E0
        
        stress = self.E0 * initial_strain * np.exp(-t / tau)
        
        return {
            'time': t,
            'stress': stress,
            'relaxation_time': tau,
            'initial_stress': stress[0],
            'final_stress': stress[-1],
        }


def analyze_creep(
    network: FiberNetwork,
    stress: float,
    duration: float,
    **kwargs,
) -> CreepResult:
    """Convenience function for creep analysis.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to analyze.
    stress : float
        Applied stress (Pa).
    duration : float
        Test duration (s).
    
    Returns
    -------
    result : CreepResult
        Creep test results.
    
    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.sim.creep import analyze_creep
    >>> net = gen.random_straight_2d(num_fibers=50, seed=42)
    >>> result = analyze_creep(net, stress=1e6, duration=3600)
    >>> print(f"Total strain: {result.total_strain:.4f}")
    """
    analyzer = CreepAnalyzer(network, **kwargs)
    return analyzer.creep_test(stress, duration)
