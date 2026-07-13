"""
Damage mechanics and fatigue module for fiber networks.

Provides tools for:
- Continuum damage mechanics (CDM) modeling
- Progressive fiber failure simulation
- Fatigue life prediction
- Damage accumulation tracking
- Residual strength estimation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from ..core import FiberNetwork


@dataclass
class DamageState:
    """Represents the damage state of a fiber network."""
    fiber_damage: np.ndarray  # Damage variable per fiber (0-1)
    crosslink_damage: np.ndarray  # Damage per crosslink (0-1)
    broken_fibers: List[int] = field(default_factory=list)
    broken_crosslinks: List[int] = field(default_factory=list)
    global_damage: float = 0.0  # Overall damage (0-1)
    load_history: List[float] = field(default_factory=list)
    damage_history: List[float] = field(default_factory=list)
    
    @property
    def fraction_broken_fibers(self) -> float:
        """Fraction of broken fibers."""
        if len(self.fiber_damage) == 0:
            return 0.0
        return len(self.broken_fibers) / len(self.fiber_damage)
    
    @property
    def fraction_broken_crosslinks(self) -> float:
        """Fraction of broken crosslinks."""
        if len(self.crosslink_damage) == 0:
            return 0.0
        return len(self.broken_crosslinks) / len(self.crosslink_damage)


@dataclass
class FatigueResult:
    """Result of a fatigue simulation."""
    cycles_to_failure: int
    sn_curve: np.ndarray  # (stress_amplitude, cycles) pairs
    damage_per_cycle: np.ndarray
    residual_stiffness: np.ndarray
    residual_strength: np.ndarray
    failure_mode: str


@dataclass
class ProgressiveFailureResult:
    """Result of progressive failure analysis."""
    load_displacement: np.ndarray
    damage_evolution: np.ndarray
    peak_load: float
    failure_strain: float
    energy_absorbed: float
    failure_sequence: List[int]


class DamageMechanicsSolver:
    """
    Solve damage mechanics problems for fiber networks.
    
    Implements continuum damage mechanics with:
    - Isotropic damage model
    - Fiber breakage criterion (maximum stress)
    - Crosslink failure criterion
    - Damage evolution laws
    
    Parameters
    ----------
    network : FiberNetwork
        The fiber network
    youngs_modulus : float
        Young's modulus (Pa)
    tensile_strength : float
        Fiber tensile strength (Pa)
    crosslink_strength : float
        Crosslink strength (Pa)
    damage_exponent : float
        Damage evolution exponent
    critical_damage : float
        Critical damage value for failure (0-1)
    """
    
    def __init__(
        self,
        network: FiberNetwork,
        youngs_modulus: float = 1e9,
        tensile_strength: float = 1e8,
        crosslink_strength: float = 5e7,
        damage_exponent: float = 2.0,
        critical_damage: float = 0.95,
    ):
        self.network = network
        self.E = youngs_modulus
        self.sigma_f = tensile_strength
        self.sigma_cl = crosslink_strength
        self.m = damage_exponent
        self.D_c = critical_damage
        
        # Initialize damage state
        self.state = DamageState(
            fiber_damage=np.zeros(network.num_fibers),
            crosslink_damage=np.zeros(network.num_crosslinks),
        )
    
    def compute_fiber_stress(
        self,
        strain: float,
        axis: int = 0,
    ) -> np.ndarray:
        """
        Compute stress in each fiber accounting for damage.
        
        sigma = (1 - D) * E * epsilon
        
        Parameters
        ----------
        strain : float
            Applied strain
        axis : int
            Loading axis
        
        Returns
        -------
        np.ndarray
            Stress in each fiber (Pa)
        """
        # Effective stiffness accounting for damage
        E_eff = self.E * (1.0 - self.state.fiber_damage)
        
        # Compute fiber orientation factor
        stresses = np.zeros(self.network.num_fibers)
        
        for i, fiber in enumerate(self.network.fibers):
            if i in self.state.broken_fibers:
                stresses[i] = 0.0
                continue
            
            # Orientation factor
            centerline = fiber.centerline
            direction = centerline[-1] - centerline[0]
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 0:
                direction = direction / direction_norm
                cos_theta = abs(direction[axis]) if axis < len(direction) else 0.0
            else:
                cos_theta = 0.0
            
            stresses[i] = E_eff[i] * strain * cos_theta**2
        
        return stresses
    
    def update_damage(
        self,
        stresses: np.ndarray,
    ) -> DamageState:
        """
        Update damage state based on current stresses.
        
        Uses power-law damage evolution:
        dD = (sigma / sigma_f)^m * dn
        
        Parameters
        ----------
        stresses : np.ndarray
            Current fiber stresses
        
        Returns
        -------
        DamageState
            Updated damage state
        """
        # Update fiber damage
        for i in range(len(self.state.fiber_damage)):
            if i in self.state.broken_fibers:
                continue
            
            stress_ratio = abs(stresses[i]) / self.sigma_f
            
            if stress_ratio >= 1.0:
                # Immediate failure
                self.state.fiber_damage[i] = 1.0
                self.state.broken_fibers.append(i)
            elif stress_ratio > 0.1:
                # Gradual damage accumulation
                dD = stress_ratio ** self.m * 0.01  # Small increment
                self.state.fiber_damage[i] = min(
                    self.state.fiber_damage[i] + dD,
                    1.0
                )
                
                # Check if fiber should break
                if self.state.fiber_damage[i] >= self.D_c:
                    self.state.fiber_damage[i] = 1.0
                    self.state.broken_fibers.append(i)
        
        # Update global damage
        if len(self.state.fiber_damage) > 0:
            self.state.global_damage = np.mean(self.state.fiber_damage)
        
        return self.state
    
    def progressive_failure(
        self,
        max_strain: float = 0.1,
        num_steps: int = 100,
        axis: int = 0,
    ) -> ProgressiveFailureResult:
        """
        Simulate progressive failure under monotonic loading.
        
        Parameters
        ----------
        max_strain : float
            Maximum strain to apply
        num_steps : int
            Number of loading steps
        axis : int
            Loading axis
        
        Returns
        -------
        ProgressiveFailureResult
            Results of progressive failure analysis
        """
        strains = np.linspace(0, max_strain, num_steps + 1)
        loads = []
        damages = []
        
        # Reset damage state
        self.state = DamageState(
            fiber_damage=np.zeros(self.network.num_fibers),
            crosslink_damage=np.zeros(self.network.num_crosslinks),
        )
        
        for strain in strains:
            # Compute stresses
            stresses = self.compute_fiber_stress(strain, axis)
            
            # Update damage
            self.update_damage(stresses)
            
            # Compute effective load (sum of stresses * area)
            # Assume unit cross-section area for simplicity
            total_load = np.sum(stresses * (1.0 - self.state.fiber_damage))
            loads.append(total_load)
            damages.append(self.state.global_damage)
        
        load_displacement = np.column_stack([strains, loads])
        damage_evolution = np.column_stack([strains, damages])
        
        # Find peak load
        peak_idx = np.argmax(loads)
        peak_load = loads[peak_idx]
        failure_strain = strains[peak_idx]
        
        # Compute energy absorbed (area under load-displacement curve)
        try:
            energy_absorbed = np.trapezoid(loads, strains)
        except AttributeError:
            energy_absorbed = np.trapz(loads, strains)
        
        return ProgressiveFailureResult(
            load_displacement=load_displacement,
            damage_evolution=damage_evolution,
            peak_load=peak_load,
            failure_strain=failure_strain,
            energy_absorbed=energy_absorbed,
            failure_sequence=self.state.broken_fibers.copy(),
        )
    
    def residual_stiffness(self) -> float:
        """
        Compute residual stiffness after damage.
        
        E_res = E_0 * (1 - D_global)
        
        Returns
        -------
        float
            Residual stiffness (Pa)
        """
        return self.E * (1.0 - self.state.global_damage)
    
    def residual_strength(self) -> float:
        """
        Compute residual strength after damage.
        
        sigma_res = sigma_f * (1 - D_global)
        
        Returns
        -------
        float
            Residual strength (Pa)
        """
        return self.sigma_f * (1.0 - self.state.global_damage)


class FatigueSolver:
    """
    Simulate fatigue behavior of fiber networks.
    
    Implements:
    - S-N curve generation
    - Miner's rule for damage accumulation
    - Stiffness degradation tracking
    - Fatigue life prediction
    
    Parameters
    ----------
    network : FiberNetwork
        The fiber network
    youngs_modulus : float
        Young's modulus (Pa)
    fatigue_limit : float
        Fatigue limit (endurance limit) (Pa)
    fatigue_exponent : float
        S-N curve exponent (Basquin exponent)
    """
    
    def __init__(
        self,
        network: FiberNetwork,
        youngs_modulus: float = 1e9,
        tensile_strength: float = 1e8,
        fatigue_limit: float = 3e7,
        fatigue_exponent: float = -0.1,
    ):
        self.network = network
        self.E = youngs_modulus
        self.sigma_u = tensile_strength
        self.sigma_e = fatigue_limit
        self.b = fatigue_exponent
        
        # Initialize damage solver
        self.damage_solver = DamageMechanicsSolver(
            network,
            youngs_modulus=youngs_modulus,
            tensile_strength=tensile_strength,
        )
    
    def compute_cycles_to_failure(
        self,
        stress_amplitude: float,
    ) -> int:
        """
        Compute number of cycles to failure at given stress amplitude.
        
        Uses Basquin's law: N_f = (sigma_a / sigma_f')^(1/b)
        
        Parameters
        ----------
        stress_amplitude : float
            Applied stress amplitude (Pa)
        
        Returns
        -------
        int
            Number of cycles to failure
        """
        if stress_amplitude <= self.sigma_e:
            return int(1e9)  # Infinite life
        
        # Basquin's law
        sigma_f_prime = self.sigma_u * 0.9  # Fatigue strength coefficient
        N_f = (stress_amplitude / sigma_f_prime) ** (1.0 / self.b)
        
        return int(min(N_f, 1e9))
    
    def generate_sn_curve(
        self,
        stress_range: Tuple[float, float] = (0.3, 0.9),
        num_points: int = 10,
    ) -> np.ndarray:
        """
        Generate S-N curve (stress vs. cycles to failure).
        
        Parameters
        ----------
        stress_range : Tuple[float, float]
            Range of stress ratios (sigma/sigma_u)
        num_points : int
            Number of points on S-N curve
        
        Returns
        -------
        np.ndarray
            S-N curve data (stress_amplitude, N_f)
        """
        stress_ratios = np.linspace(stress_range[0], stress_range[1], num_points)
        sn_data = []
        
        for ratio in stress_ratios:
            sigma_a = ratio * self.sigma_u
            N_f = self.compute_cycles_to_failure(sigma_a)
            sn_data.append([sigma_a, N_f])
        
        return np.array(sn_data)
    
    def simulate_fatigue(
        self,
        stress_amplitude: float,
        max_cycles: int = 1000000,
        check_interval: int = 100,
    ) -> FatigueResult:
        """
        Simulate fatigue loading cycle by cycle.
        
        Parameters
        ----------
        stress_amplitude : float
            Applied stress amplitude (Pa)
        max_cycles : int
            Maximum number of cycles to simulate
        check_interval : int
            Check for failure every N cycles
        
        Returns
        -------
        FatigueResult
            Fatigue simulation results
        """
        # Convert stress to strain
        strain_amplitude = stress_amplitude / self.E
        
        damage_per_cycle = []
        residual_stiffness = []
        residual_strength = []
        
        cycles_to_failure = max_cycles
        failure_mode = "no_failure"
        
        for cycle in range(0, max_cycles, check_interval):
            # Apply cyclic strain
            stresses = self.damage_solver.compute_fiber_stress(strain_amplitude)
            self.damage_solver.update_damage(stresses)
            
            # Record state
            damage_per_cycle.append(self.damage_solver.state.global_damage)
            residual_stiffness.append(self.damage_solver.residual_stiffness())
            residual_strength.append(self.damage_solver.residual_strength())
            
            # Check for failure
            if self.damage_solver.state.global_damage >= 0.95:
                cycles_to_failure = cycle
                failure_mode = "fiber_breakage"
                break
        
        # Generate S-N curve
        sn_curve = self.generate_sn_curve()
        
        return FatigueResult(
            cycles_to_failure=cycles_to_failure,
            sn_curve=sn_curve,
            damage_per_cycle=np.array(damage_per_cycle),
            residual_stiffness=np.array(residual_stiffness),
            residual_strength=np.array(residual_strength),
            failure_mode=failure_mode,
        )
    
    def miners_rule(
        self,
        load_history: List[Tuple[float, int]],
    ) -> float:
        """
        Apply Miner's rule for variable amplitude loading.
        
        D = sum(n_i / N_f,i)
        
        Parameters
        ----------
        load_history : List[Tuple[float, int]]
            List of (stress_amplitude, num_cycles) pairs
        
        Returns
        -------
        float
            Cumulative damage (failure when D >= 1.0)
        """
        total_damage = 0.0
        
        for stress_amp, num_cycles in load_history:
            N_f = self.compute_cycles_to_failure(stress_amp)
            total_damage += num_cycles / N_f
        
        return total_damage


def compute_damage_tolerance(
    network: FiberNetwork,
    initial_damage_fraction: float = 0.1,
    youngs_modulus: float = 1e9,
    tensile_strength: float = 1e8,
) -> Dict:
    """
    Compute damage tolerance of a fiber network.
    
    Parameters
    ----------
    network : FiberNetwork
        The fiber network
    initial_damage_fraction : float
        Fraction of fibers initially damaged
    youngs_modulus : float
        Young's modulus (Pa)
    tensile_strength : float
        Tensile strength (Pa)
    
    Returns
    -------
    Dict
        Damage tolerance metrics
    """
    solver = DamageMechanicsSolver(
        network,
        youngs_modulus=youngs_modulus,
        tensile_strength=tensile_strength,
    )
    
    # Apply initial damage
    num_to_damage = int(initial_damage_fraction * network.num_fibers)
    damaged_indices = np.random.choice(network.num_fibers, num_to_damage, replace=False)
    
    for idx in damaged_indices:
        solver.state.fiber_damage[idx] = 0.5
    
    solver.state.global_damage = np.mean(solver.state.fiber_damage)
    
    # Compute residual properties
    E_res = solver.residual_stiffness()
    sigma_res = solver.residual_strength()
    
    # Run progressive failure
    result = solver.progressive_failure(max_strain=0.05, num_steps=50)
    
    return {
        'initial_damage_fraction': initial_damage_fraction,
        'residual_stiffness': E_res,
        'residual_strength': sigma_res,
        'stiffness_retention': E_res / youngs_modulus,
        'strength_retention': sigma_res / tensile_strength,
        'peak_load_damaged': result.peak_load,
        'energy_absorbed': result.energy_absorbed,
        'num_broken_fibers': len(solver.state.broken_fibers),
    }
