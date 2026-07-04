"""
Fatigue Analysis Module for Fiber Networks

Provides tools for fatigue life prediction under cyclic loading:
- S-N curve generation
- Cyclic loading simulation
- Fatigue life prediction (Basquin equation)
- Damage accumulation (Miner's rule)
- Goodman diagram for mean stress effects
- Stress ratio effects

References:
- Suresh, S. "Fatigue of Materials", Cambridge University Press, 1998
- Stephens et al., "Metal Fatigue in Engineering", Wiley, 2000
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from scipy.interpolate import interp1d
import warnings

from fibernet.core.network import FiberNetwork
from fibernet.sim import FiberFEM


@dataclass
class SNPoint:
    """Single point on S-N curve."""
    stress_amplitude: float  # Pa
    cycles_to_failure: int
    stress_ratio: float = -1.0  # R = sigma_min / sigma_max
    
    def to_dict(self) -> Dict:
        return {
            'stress_amplitude': self.stress_amplitude,
            'cycles_to_failure': self.cycles_to_failure,
            'stress_ratio': self.stress_ratio,
        }


@dataclass
class FatigueResult:
    """Result of fatigue analysis."""
    sn_curve: List[SNPoint] = field(default_factory=list)
    fatigue_strength_coefficient: float = 0.0  # sigma_f'
    fatigue_strength_exponent: float = 0.0  # b
    endurance_limit: float = 0.0  # Pa
    cycles_to_failure: int = 0
    damage_accumulation: float = 0.0  # Miner's rule
    is_failed: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'num_sn_points': len(self.sn_curve),
            'fatigue_strength_coefficient': self.fatigue_strength_coefficient,
            'fatigue_strength_exponent': self.fatigue_strength_exponent,
            'endurance_limit': self.endurance_limit,
            'cycles_to_failure': self.cycles_to_failure,
            'damage_accumulation': self.damage_accumulation,
            'is_failed': self.is_failed,
        }


@dataclass
class CyclicLoadResult:
    """Result of cyclic loading simulation."""
    num_cycles: int = 0
    max_stress: float = 0.0  # Pa
    min_stress: float = 0.0  # Pa
    stress_amplitude: float = 0.0  # Pa
    mean_stress: float = 0.0  # Pa
    stress_ratio: float = 0.0  # R
    damage_per_cycle: float = 0.0
    stiffness_degradation: List[float] = field(default_factory=list)
    residual_stiffness: float = 1.0  # fraction of initial
    
    def to_dict(self) -> Dict:
        return {
            'num_cycles': self.num_cycles,
            'stress_amplitude': self.stress_amplitude,
            'mean_stress': self.mean_stress,
            'stress_ratio': self.stress_ratio,
            'damage_per_cycle': self.damage_per_cycle,
            'residual_stiffness': self.residual_stiffness,
        }


class FatigueAnalyzer:
    """Analyze fatigue behavior of fiber networks.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to analyze.
    fatigue_strength_coefficient : float, optional
        Basquin fatigue strength coefficient (sigma_f').
        Default: 0.9 * UTS.
    fatigue_strength_exponent : float, optional
        Basquin fatigue strength exponent (b).
        Default: -0.12 (typical for polymers/composites).
    endurance_limit : float, optional
        Endurance limit stress (Pa).
        Default: 0.4 * UTS.
    ultimate_tensile_strength : float, optional
        Ultimate tensile strength (Pa).
        Default: computed from FEM.
    
    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.sim.fatigue import FatigueAnalyzer
    >>> net = gen.random_straight_2d(num_fibers=50, seed=42)
    >>> fatigue = FatigueAnalyzer(net)
    >>> result = fatigue.generate_sn_curve()
    >>> print(f"Endurance limit: {result.endurance_limit:.2e} Pa")
    """
    
    def __init__(
        self,
        network: FiberNetwork,
        fatigue_strength_coefficient: Optional[float] = None,
        fatigue_strength_exponent: float = -0.12,
        endurance_limit: Optional[float] = None,
        ultimate_tensile_strength: Optional[float] = None,
    ):
        self.network = network
        self.fem = FiberFEM(network)
        
        # Compute UTS if not provided
        if ultimate_tensile_strength is None:
            # Estimate from effective modulus and typical strain to failure
            E = self.fem.effective_modulus()
            strain_failure = 0.02  # 2% strain to failure (typical)
            self.uts = E * strain_failure
        else:
            self.uts = ultimate_tensile_strength
        
        # Set fatigue parameters
        self.sigma_f = fatigue_strength_coefficient or (0.9 * self.uts)
        self.b = fatigue_strength_exponent
        self.endurance = endurance_limit or (0.4 * self.uts)
    
    def generate_sn_curve(
        self,
        num_points: int = 10,
        stress_range: Optional[Tuple[float, float]] = None,
        stress_ratio: float = -1.0,
    ) -> FatigueResult:
        """Generate S-N curve (stress vs cycles to failure).
        
        Parameters
        ----------
        num_points : int
            Number of points on S-N curve.
        stress_range : tuple, optional
            (min_stress, max_stress) in Pa.
        stress_ratio : float
            Stress ratio R = sigma_min / sigma_max.
        
        Returns
        -------
        result : FatigueResult
            S-N curve data.
        """
        if stress_range is None:
            stress_range = (0.3 * self.uts, 0.8 * self.uts)
        
        # Generate stress amplitudes (logarithmic spacing)
        stress_amps = np.logspace(
            np.log10(stress_range[0]),
            np.log10(stress_range[1]),
            num_points,
        )
        
        sn_points = []
        for sigma_a in stress_amps:
            # Basquin equation: sigma_a = sigma_f' * (2*Nf)^b
            # Solve for Nf: Nf = 0.5 * (sigma_a / sigma_f')^(1/b)
            if sigma_a < self.endurance:
                # Below endurance limit - infinite life
                Nf = int(1e7)  # 10 million cycles
            else:
                Nf = int(0.5 * (sigma_a / self.sigma_f) ** (1.0 / self.b))
                Nf = max(Nf, 1)  # At least 1 cycle
            
            sn_points.append(SNPoint(
                stress_amplitude=float(sigma_a),
                cycles_to_failure=Nf,
                stress_ratio=stress_ratio,
            ))
        
        return FatigueResult(
            sn_curve=sn_points,
            fatigue_strength_coefficient=self.sigma_f,
            fatigue_strength_exponent=self.b,
            endurance_limit=self.endurance,
        )
    
    def predict_life(
        self,
        stress_amplitude: float,
        mean_stress: float = 0.0,
        stress_ratio: float = -1.0,
    ) -> int:
        """Predict fatigue life for given loading.
        
        Parameters
        ----------
        stress_amplitude : float
            Stress amplitude (Pa).
        mean_stress : float
            Mean stress (Pa).
        stress_ratio : float
            Stress ratio R.
        
        Returns
        -------
        Nf : int
            Cycles to failure.
        """
        # Apply Goodman correction for mean stress
        if mean_stress > 0:
            # Goodman: sigma_a / sigma_e + sigma_m / sigma_uts = 1
            # sigma_e = sigma_a / (1 - sigma_m / sigma_uts)
            sigma_e = stress_amplitude / (1.0 - mean_stress / self.uts)
        else:
            sigma_e = stress_amplitude
        
        # Check endurance limit
        if sigma_e < self.endurance:
            return int(1e7)  # Infinite life
        
        # Basquin equation
        Nf = int(0.5 * (sigma_e / self.sigma_f) ** (1.0 / self.b))
        return max(Nf, 1)
    
    def cyclic_loading(
        self,
        stress_amplitude: float,
        mean_stress: float = 0.0,
        num_cycles: int = 1000,
        damage_model: str = 'linear',
    ) -> CyclicLoadResult:
        """Simulate cyclic loading and track damage accumulation.
        
        Parameters
        ----------
        stress_amplitude : float
            Stress amplitude (Pa).
        mean_stress : float
            Mean stress (Pa).
        num_cycles : int
            Number of cycles to simulate.
        damage_model : str
            Damage accumulation model: 'linear' (Miner's rule) or 'nonlinear'.
        
        Returns
        -------
        result : CyclicLoadResult
            Cyclic loading results.
        """
        # Compute stress parameters
        max_stress = mean_stress + stress_amplitude
        min_stress = mean_stress - stress_amplitude
        R = min_stress / max_stress if max_stress > 0 else -1.0
        
        # Predict total life
        Nf = self.predict_life(stress_amplitude, mean_stress, R)
        
        # Damage per cycle
        damage_per_cycle = 1.0 / Nf if Nf > 0 else 1.0
        
        # Simulate cycles
        stiffness_degradation = []
        residual_stiffness = 1.0
        
        for cycle in range(num_cycles):
            if damage_model == 'linear':
                # Linear damage accumulation (Miner's rule)
                damage = (cycle + 1) * damage_per_cycle
            else:
                # Nonlinear damage (exponential degradation)
                damage = 1.0 - np.exp(-damage_per_cycle * (cycle + 1))
            
            # Stiffness degradation (proportional to damage)
            residual_stiffness = max(0.0, 1.0 - damage)
            stiffness_degradation.append(residual_stiffness)
            
            # Check for failure
            if residual_stiffness < 0.1:  # 90% stiffness loss
                break
        
        return CyclicLoadResult(
            num_cycles=len(stiffness_degradation),
            max_stress=max_stress,
            min_stress=min_stress,
            stress_amplitude=stress_amplitude,
            mean_stress=mean_stress,
            stress_ratio=R,
            damage_per_cycle=damage_per_cycle,
            stiffness_degradation=stiffness_degradation,
            residual_stiffness=residual_stiffness,
        )
    
    def variable_amplitude_loading(
        self,
        load_spectrum: List[Tuple[float, int]],
        mean_stress: float = 0.0,
    ) -> FatigueResult:
        """Analyze fatigue under variable amplitude loading.
        
        Parameters
        ----------
        load_spectrum : list
            List of (stress_amplitude, num_cycles) tuples.
        mean_stress : float
            Mean stress (Pa).
        
        Returns
        -------
        result : FatigueResult
            Fatigue analysis results.
        """
        # Miner's rule: sum(n_i / N_i) = 1 at failure
        total_damage = 0.0
        
        for sigma_a, n_cycles in load_spectrum:
            Nf = self.predict_life(sigma_a, mean_stress)
            damage = n_cycles / Nf if Nf > 0 else 1.0
            total_damage += damage
        
        # Predict remaining life
        if total_damage < 1.0:
            # Compute equivalent constant amplitude
            total_cycles = sum(n for _, n in load_spectrum)
            equivalent_cycles_to_failure = int(total_cycles / total_damage)
        else:
            equivalent_cycles_to_failure = 0
        
        return FatigueResult(
            damage_accumulation=total_damage,
            cycles_to_failure=equivalent_cycles_to_failure,
            is_failed=total_damage >= 1.0,
            fatigue_strength_coefficient=self.sigma_f,
            fatigue_strength_exponent=self.b,
            endurance_limit=self.endurance,
        )
    
    def goodman_diagram(
        self,
        stress_amplitudes: List[float],
        mean_stresses: List[float],
    ) -> Dict[str, np.ndarray]:
        """Generate Goodman diagram data.
        
        Parameters
        ----------
        stress_amplitudes : list
            Stress amplitudes (Pa).
        mean_stresses : list
            Mean stresses (Pa).
        
        Returns
        -------
        data : dict
            Goodman diagram data with keys:
            - 'alternating_stress': alternating stress array
            - 'mean_stress': mean stress array
            - 'safe_region': boolean array (True = safe)
        """
        sigma_a = np.array(stress_amplitudes)
        sigma_m = np.array(mean_stresses)
        
        # Goodman line: sigma_a / sigma_e + sigma_m / sigma_uts = 1
        # sigma_a = sigma_e * (1 - sigma_m / sigma_uts)
        sigma_e = self.endurance
        sigma_a_goodman = sigma_e * (1.0 - sigma_m / self.uts)
        
        # Check if points are safe
        safe = sigma_a <= sigma_a_goodman
        
        return {
            'alternating_stress': sigma_a,
            'mean_stress': sigma_m,
            'goodman_line': sigma_a_goodman,
            'safe_region': safe,
        }


def analyze_fatigue(
    network: FiberNetwork,
    stress_amplitude: Optional[float] = None,
    num_cycles: int = 1000,
    **kwargs,
) -> Union[FatigueResult, CyclicLoadResult]:
    """Convenience function for fatigue analysis.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to analyze.
    stress_amplitude : float, optional
        If provided, runs cyclic loading simulation.
        If None, generates S-N curve.
    num_cycles : int
        Number of cycles for cyclic loading.
    
    Returns
    -------
    result : FatigueResult or CyclicLoadResult
        Analysis results.
    
    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.sim.fatigue import analyze_fatigue
    >>> net = gen.random_straight_2d(num_fibers=50, seed=42)
    >>> # Generate S-N curve
    >>> sn_result = analyze_fatigue(net)
    >>> # Cyclic loading
    >>> cyclic_result = analyze_fatigue(net, stress_amplitude=1e7, num_cycles=500)
    """
    analyzer = FatigueAnalyzer(network, **kwargs)
    
    if stress_amplitude is None:
        return analyzer.generate_sn_curve()
    else:
        return analyzer.cyclic_loading(stress_amplitude, num_cycles=num_cycles)
