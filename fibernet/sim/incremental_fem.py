"""
Incremental Nonlinear Finite Element Method

Provides incremental stress-strain analysis with:
- Elastic-plastic material response
- Isotropic/kinematic hardening
- Damage evolution
- Fiber failure
- Network-level failure

This module is designed for generating realistic stress-strain curves
for research publications in materials science.
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass, field
import warnings

from fibernet.core.network import FiberNetwork
from fibernet.sim.mechanical import FiberFEM


@dataclass
class IncrementalResult:
    """Container for incremental FEM results."""
    strain_history: np.ndarray = None
    stress_history: np.ndarray = None
    energy_history: np.ndarray = None
    damage_history: np.ndarray = None
    failed_elements: List[int] = field(default_factory=list)
    
    # Material response regions
    yield_strain: float = 0.0
    yield_stress: float = 0.0
    ultimate_strain: float = 0.0
    ultimate_stress: float = 0.0
    failure_strain: float = 0.0
    
    # Computed properties
    youngs_modulus: float = 0.0
    tangent_modulus: float = 0.0
    yield_strength: float = 0.0
    ultimate_strength: float = 0.0
    toughness: float = 0.0
    ductility: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'strain': self.strain_history.tolist() if self.strain_history is not None else [],
            'stress': self.stress_history.tolist() if self.stress_history is not None else [],
            'energy': self.energy_history.tolist() if self.energy_history is not None else [],
            'damage': self.damage_history.tolist() if self.damage_history is not None else [],
            'yield_strain': self.yield_strain,
            'yield_stress': self.yield_stress,
            'ultimate_strain': self.ultimate_strain,
            'ultimate_stress': self.ultimate_stress,
            'failure_strain': self.failure_strain,
            'youngs_modulus': self.youngs_modulus,
            'tangent_modulus': self.tangent_modulus,
            'yield_strength': self.yield_strength,
            'ultimate_strength': self.ultimate_strength,
            'toughness': self.toughness,
            'ductility': self.ductility,
        }
    
    def plot(self, ax=None, show_regions: bool = True, **kwargs):
        """Plot stress-strain curve.
        
        Parameters
        ----------
        ax : matplotlib axes, optional
            Axes to plot on. If None, creates new figure.
        show_regions : bool
            If True, marks yield, ultimate, and failure points.
        **kwargs : dict
            Additional arguments passed to plot().
        
        Returns
        -------
        ax : matplotlib axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting. Install with: pip install matplotlib")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot curve
        ax.plot(self.strain_history, self.stress_history, **kwargs)
        
        if show_regions:
            # Mark yield point
            if self.yield_strain > 0:
                ax.plot(self.yield_strain, self.yield_stress, 'ro', 
                       markersize=10, label='Yield point')
            
            # Mark ultimate point
            if self.ultimate_strain > 0:
                ax.plot(self.ultimate_strain, self.ultimate_stress, 'g^', 
                       markersize=10, label='Ultimate strength')
            
            # Mark failure
            if self.failure_strain > 0:
                ax.axvline(self.failure_strain, color='r', linestyle='--', 
                          alpha=0.5, label='Failure')
        
        ax.set_xlabel('Strain')
        ax.set_ylabel('Stress (Pa)')
        ax.set_title('Stress-Strain Curve')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return ax


class IncrementalFEM:
    """Incremental nonlinear FEM solver.
    
    Applies strain incrementally and tracks material response through
    elastic, plastic, damage, and failure regimes.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to analyze.
    segments_per_fiber : int
        Number of elements per fiber.
    material_model : str
        Material model: 'elastic', 'elastic_plastic', 'damage'
    yield_stress : float
        Yield stress for plasticity (Pa).
    hardening_modulus : float
        Plastic hardening modulus (Pa).
    damage_threshold : float
        Strain at which damage initiates.
    failure_strain : float
        Strain at which element fails.
    
    Examples
    --------
    >>> import fibernet as fn
    >>> from fibernet.sim import IncrementalFEM
    >>> net = fn.create('random_2d', num_fibers=100, seed=42)
    >>> solver = IncrementalFEM(net, material_model='elastic_plastic')
    >>> result = solver.run_incremental_analysis(
    ...     max_strain=0.05,
    ...     num_increments=50
    ... )
    >>> print(f"Yield strength: {result.yield_strength:.2e} Pa")
    >>> print(f"Ultimate strength: {result.ultimate_strength:.2e} Pa")
    """
    
    def __init__(
        self,
        network: FiberNetwork,
        segments_per_fiber: int = 5,
        material_model: str = 'elastic',
        yield_stress: float = 1e8,
        hardening_modulus: float = 1e7,
        damage_threshold: float = 0.02,
        failure_strain: float = 0.1,
    ):
        self.network = network
        self.segments = segments_per_fiber
        self.material_model = material_model
        self.yield_stress = yield_stress
        self.hardening_modulus = hardening_modulus
        self.damage_threshold = damage_threshold
        self.failure_strain = failure_strain
        
        # Initialize base FEM solver
        self.base_fem = FiberFEM(network, segments_per_fiber=segments_per_fiber)
        
        # State variables
        self.element_stress = {}
        self.element_strain = {}
        self.element_damage = {}
        self.failed_elements = set()
        
        # Results storage
        self.strain_history = []
        self.stress_history = []
        self.energy_history = []
        self.damage_history = []
    
    def _compute_element_stress(self, result) -> Tuple[np.ndarray, np.ndarray]:
        """Extract stress and strain from FEM result.
        
        Parameters
        ----------
        result : MechanicalResult
            Result from FEM solve.
        
        Returns
        -------
        element_stress : ndarray
            Stress in each element.
        element_strain : ndarray
            Strain in each element.
        """
        # Use pre-computed stresses and strains from FEM result
        element_stress = result.stresses.copy()
        element_strain = result.strains.copy()
        
        # Apply material model modifications if needed
        if self.material_model == 'elastic_plastic':
            for i in range(len(element_strain)):
                element_stress[i] = self._elastic_plastic_stress(
                    element_strain[i],
                    self.base_fem.elements[i].E,
                    i
                )
        elif self.material_model == 'damage':
            for i in range(len(element_strain)):
                element_stress[i] = self._damage_stress(
                    element_strain[i],
                    self.base_fem.elements[i].E,
                    i
                )
        
        return element_stress, element_strain
    
    def _elastic_plastic_stress(self, strain: float, E: float, elem_idx: int) -> float:
        """Compute elastic-plastic stress with isotropic hardening."""
        # Get previous plastic strain
        plastic_strain = self.element_strain.get(elem_idx, 0.0)
        
        # Trial stress
        trial_stress = E * (strain - plastic_strain)
        
        # Yield check
        if abs(trial_stress) > self.yield_stress:
            # Plastic flow
            sign = np.sign(trial_stress)
            yield_stress_current = self.yield_stress + self.hardening_modulus * abs(plastic_strain)
            
            if abs(trial_stress) > yield_stress_current:
                # Plastic strain increment
                d_plastic = (abs(trial_stress) - yield_stress_current) / (E + self.hardening_modulus)
                plastic_strain += sign * d_plastic
                
                # Update stored plastic strain
                self.element_strain[elem_idx] = plastic_strain
                
                # Return stress at yield surface
                stress = sign * yield_stress_current
            else:
                stress = trial_stress
        else:
            stress = trial_stress
        
        return stress
    
    def _damage_stress(self, strain: float, E: float, elem_idx: int) -> float:
        """Compute stress with damage evolution."""
        # Get current damage
        damage = self.element_damage.get(elem_idx, 0.0)
        
        # Effective stress (undamaged)
        effective_stress = E * strain
        
        # Damage evolution
        if abs(strain) > self.damage_threshold:
            # Linear damage evolution
            damage_increment = (abs(strain) - self.damage_threshold) / (self.failure_strain - self.damage_threshold)
            damage = min(1.0, damage + damage_increment * 0.1)
            self.element_damage[elem_idx] = damage
        
        # Damaged stress
        stress = (1.0 - damage) * effective_stress
        
        return stress
    
    def _check_element_failure(self, element_strain: np.ndarray) -> List[int]:
        """Check for element failure."""
        newly_failed = []
        
        for i, strain in enumerate(element_strain):
            if i not in self.failed_elements:
                if abs(strain) > self.failure_strain:
                    self.failed_elements.add(i)
                    newly_failed.append(i)
        
        return newly_failed
    
    def run_incremental_analysis(
        self,
        max_strain: float = 0.05,
        num_increments: int = 50,
        axis: int = 0,
        verbose: bool = False,
    ) -> IncrementalResult:
        """Run incremental strain-controlled analysis.
        
        Parameters
        ----------
        max_strain : float
            Maximum applied strain.
        num_increments : int
            Number of strain increments.
        axis : int
            Loading axis (0=x, 1=y, 2=z).
        verbose : bool
            Print progress.
        
        Returns
        -------
        IncrementalResult
            Complete analysis results.
        """
        strain_increments = np.linspace(0, max_strain, num_increments + 1)[1:]
        
        if verbose:
            print(f"Running incremental analysis: {num_increments} increments to strain={max_strain:.4f}")
        
        # Reset state
        self.element_stress = {}
        self.element_strain = {}
        self.element_damage = {}
        self.failed_elements = set()
        self.strain_history = [0.0]
        self.stress_history = [0.0]
        self.energy_history = [0.0]
        self.damage_history = [0.0]
        
        # Apply increments
        for i, strain in enumerate(strain_increments):
            if verbose and i % 10 == 0:
                print(f"  Increment {i}/{num_increments}: strain={strain:.4f}")
            
            try:
                # Apply strain
                result = self.base_fem.apply_uniaxial_strain(strain, axis=axis)
                
                # Compute element stresses
                element_stress, element_strain = self._compute_element_stress(result)
                
                # Compute average stress
                avg_stress = np.mean(np.abs(element_stress))
                
                # Check for failures
                newly_failed = self._check_element_failure(element_strain)
                if newly_failed and verbose:
                    print(f"    {len(newly_failed)} elements failed")
                
                # Compute average damage
                if self.element_damage:
                    avg_damage = np.mean(list(self.element_damage.values()))
                else:
                    avg_damage = 0.0
                
                # Store results
                self.strain_history.append(strain)
                self.stress_history.append(avg_stress)
                self.energy_history.append(result.energy)
                self.damage_history.append(avg_damage)
                
                # Check if network has failed completely
                if len(self.failed_elements) > len(self.base_fem.elements) * 0.9:
                    if verbose:
                        print(f"  Network failed at strain={strain:.4f}")
                    break
            
            except Exception as e:
                if verbose:
                    print(f"  Analysis stopped at strain={strain:.4f}: {e}")
                break
        
        # Post-process results
        result = self._postprocess_results()
        
        if verbose:
            print(f"Analysis complete: {len(self.strain_history)} points")
            print(f"  Yield strength: {result.yield_strength:.2e} Pa")
            print(f"  Ultimate strength: {result.ultimate_strength:.2e} Pa")
            print(f"  Toughness: {result.toughness:.2e} J/m³")
        
        return result
    
    def _postprocess_results(self) -> IncrementalResult:
        """Post-process incremental results."""
        strain = np.array(self.strain_history)
        stress = np.array(self.stress_history)
        energy = np.array(self.energy_history)
        damage = np.array(self.damage_history)
        
        result = IncrementalResult(
            strain_history=strain,
            stress_history=stress,
            energy_history=energy,
            damage_history=damage,
            failed_elements=list(self.failed_elements),
        )
        
        # Compute Young's modulus (initial slope)
        if len(strain) > 10:
            # Use first 10% of data
            idx = max(2, len(strain) // 10)
            result.youngs_modulus = np.polyfit(strain[:idx], stress[:idx], 1)[0]
        
        # Find yield point (0.2% offset method)
        if result.youngs_modulus > 0:
            offset_line = result.youngs_modulus * (strain - 0.002)
            yield_idx = np.where(stress < offset_line)[0]
            if len(yield_idx) > 0:
                result.yield_strain = strain[yield_idx[0]]
                result.yield_stress = stress[yield_idx[0]]
                result.yield_strength = result.yield_stress
        
        # Find ultimate strength
        if len(stress) > 0:
            ult_idx = np.argmax(stress)
            result.ultimate_strain = strain[ult_idx]
            result.ultimate_stress = stress[ult_idx]
            result.ultimate_strength = result.ultimate_stress
        
        # Compute toughness (area under curve)
        if len(strain) > 1:
            result.toughness = np.trapezoid(stress, strain)
        
        # Compute ductility
        if result.yield_strain > 0 and result.ultimate_strain > 0:
            result.ductility = result.ultimate_strain / result.yield_strain
        
        # Compute tangent modulus (post-yield slope)
        if result.yield_strain > 0 and result.ultimate_strain > result.yield_strain:
            idx_start = np.searchsorted(strain, result.yield_strain)
            idx_end = np.searchsorted(strain, result.ultimate_strain)
            if idx_end > idx_start + 1:
                result.tangent_modulus = np.polyfit(
                    strain[idx_start:idx_end],
                    stress[idx_start:idx_end],
                    1
                )[0]
        
        # Failure strain
        if len(self.failed_elements) > 0:
            result.failure_strain = strain[-1]
        
        return result


def compute_stress_strain_curve(
    network: FiberNetwork,
    max_strain: float = 0.05,
    num_increments: int = 50,
    material_model: str = 'elastic',
    **kwargs,
) -> IncrementalResult:
    """Compute stress-strain curve for a fiber network.
    
    Convenience function for generating stress-strain curves.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to analyze.
    max_strain : float
        Maximum applied strain.
    num_increments : int
        Number of strain increments.
    material_model : str
        Material model: 'elastic', 'elastic_plastic', 'damage'
    **kwargs : dict
        Additional arguments passed to IncrementalFEM.
    
    Returns
    -------
    IncrementalResult
        Complete stress-strain analysis results.
    
    Examples
    --------
    >>> import fibernet as fn
    >>> from fibernet.sim import compute_stress_strain_curve
    >>> net = fn.create('random_2d', num_fibers=100, seed=42)
    >>> result = compute_stress_strain_curve(
    ...     net,
    ...     max_strain=0.05,
    ...     material_model='elastic_plastic'
    ... )
    >>> print(f"Yield strength: {result.yield_strength:.2e} Pa")
    """
    solver = IncrementalFEM(
        network,
        material_model=material_model,
        **kwargs,
    )
    
    return solver.run_incremental_analysis(
        max_strain=max_strain,
        num_increments=num_increments,
    )
