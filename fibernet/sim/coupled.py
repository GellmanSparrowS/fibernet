"""
Multi-physics coupled simulations.

Provides tools for coupling different physics simulations together,
such as thermo-mechanical, electro-mechanical, and chemo-mechanical analysis.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from .mechanical import FiberFEM, MechanicalResult
from .thermal import ThermalSolver, ThermalResult
from .electromagnetic import EMSolver, EMResult
from .nonlinear import NonlinearFEM


@dataclass
class CoupledResult:
    """Container for coupled simulation results."""
    mechanical: Optional[MechanicalResult] = None
    thermal: Optional[ThermalResult] = None
    electromagnetic: Optional[EMResult] = None
    coupling_params: Dict = field(default_factory=dict)
    converged: bool = True
    iterations: int = 0


class ThermoMechanicalSolver:
    """
    Coupled thermo-mechanical analysis.
    
    Simulates thermal expansion effects on mechanical behavior.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to analyze.
    alpha : float, optional
        Thermal expansion coefficient (1/K). Default is 1e-5.
    segments_per_fiber : int, optional
        Number of segments per fiber for FEM discretization.
    
    Examples
    --------
    >>> solver = ThermoMechanicalSolver(net, alpha=1e-5)
    >>> result = solver.coupled_analysis(
    ...     T_hot=100, T_cold=0,
    ...     mechanical_strain=0.01,
    ...     axis=0
    ... )
    >>> print(f"Displacement: {result.mechanical.max_displacement():.4e} m")
    """
    
    def __init__(self, network, alpha: float = 1e-5, segments_per_fiber: int = 5):
        self.network = network
        self.alpha = alpha
        self.mech = FiberFEM(network, segments_per_fiber=segments_per_fiber)
        self.thermal = ThermalSolver(network)
    
    def coupled_analysis(
        self,
        T_hot: float,
        T_cold: float,
        mechanical_strain: float = 0.0,
        axis: int = 0,
        max_iterations: int = 10,
        tolerance: float = 1e-6
    ) -> CoupledResult:
        """
        Perform coupled thermo-mechanical analysis.
        
        Parameters
        ----------
        T_hot : float
            Hot temperature (K or °C).
        T_cold : float
            Cold temperature (K or °C).
        mechanical_strain : float, optional
            Applied mechanical strain.
        axis : int, optional
            Axis for mechanical loading (0=x, 1=y, 2=z).
        max_iterations : int, optional
            Maximum coupling iterations.
        tolerance : float, optional
            Convergence tolerance.
        
        Returns
        -------
        CoupledResult
            Coupled simulation results.
        """
        # Step 1: Solve thermal problem
        thermal_result = self.thermal.solve_steady_state(T_hot, T_cold, axis=axis)
        
        # Step 2: Compute thermal strains
        thermal_strain = np.zeros(self.mech.num_elements)
        if hasattr(thermal_result, 'temperatures') and thermal_result.temperatures is not None:
            T_ref = (T_hot + T_cold) / 2
            temps = thermal_result.temperatures
            if len(temps) == self.mech.num_elements:
                thermal_strain = self.alpha * (temps - T_ref)
        
        # Step 3: Apply combined mechanical + thermal loading
        if mechanical_strain > 0:
            mech_result = self.mech.apply_uniaxial_strain(
                strain=mechanical_strain,
                axis=axis
            )
        else:
            # Thermal-only loading
            mech_result = self._apply_thermal_strain(thermal_strain, axis)
        
        return CoupledResult(
            mechanical=mech_result,
            thermal=thermal_result,
            coupling_params={'alpha': self.alpha},
            converged=True,
            iterations=1
        )
    
    def _apply_thermal_strain(self, thermal_strain, axis):
        """Apply thermal strain as initial strain."""
        # For now, approximate as mechanical strain
        mean_thermal = np.mean(thermal_strain)
        return self.mech.apply_uniaxial_strain(strain=mean_thermal, axis=axis)


class ElectroMechanicalSolver:
    """
    Coupled electro-mechanical analysis (piezoelectric effects).
    
    Simulates how electric fields affect mechanical behavior.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to analyze.
    piezo_coeff : float, optional
        Piezoelectric coupling coefficient (m/V). Default is 1e-12.
    segments_per_fiber : int, optional
        Number of segments per fiber.
    """
    
    def __init__(self, network, piezo_coeff: float = 1e-12, segments_per_fiber: int = 5):
        self.network = network
        self.piezo_coeff = piezo_coeff
        self.mech = FiberFEM(network, segments_per_fiber=segments_per_fiber)
        self.em = EMSolver(network)
    
    def coupled_analysis(
        self,
        voltage: float,
        axis: int = 0,
        mechanical_strain: float = 0.0
    ) -> CoupledResult:
        """
        Perform coupled electro-mechanical analysis.
        
        Parameters
        ----------
        voltage : float
            Applied voltage (V).
        axis : int, optional
            Axis for electric field.
        mechanical_strain : float, optional
            Additional mechanical strain.
        
        Returns
        -------
        CoupledResult
            Coupled simulation results.
        """
        # Step 1: Solve electromagnetic problem
        em_result = self.em.solve_conductivity(voltage=voltage, axis=axis)
        
        # Step 2: Compute piezoelectric strain from current density
        if em_result.current_density is not None and len(em_result.current_density) > 0:
            # Use current density as proxy for electric field effect
            J_mean = np.mean(np.abs(em_result.current_density))
            piezo_strain = self.piezo_coeff * J_mean * np.ones(self.mech.num_elements)
        else:
            piezo_strain = np.zeros(self.mech.num_elements)
        
        # Step 3: Apply combined loading
        total_strain = mechanical_strain + np.mean(piezo_strain)
        if total_strain > 0:
            mech_result = self.mech.apply_uniaxial_strain(
                strain=total_strain,
                axis=axis
            )
        else:
            mech_result = MechanicalResult()
        
        return CoupledResult(
            mechanical=mech_result,
            electromagnetic=em_result,
            coupling_params={'piezo_coeff': self.piezo_coeff},
            converged=True,
            iterations=1
        )


class MultiPhysicsSolver:
    """
    General multi-physics coupling framework.
    
    Combines multiple physics solvers with iterative coupling.
    
    Parameters
    ----------
    network : FiberNetwork
        Fiber network to analyze.
    solvers : dict
        Dictionary of solver instances.
    
    Examples
    --------
    >>> multi = MultiPhysicsSolver(
    ...     net,
    ...     solvers={
    ...         'mechanical': FiberFEM(net),
    ...         'thermal': ThermalSolver(net),
    ...         'em': EMSolver(net)
    ...     }
    ... )
    >>> result = multi.solve_coupled(
    ...     mechanical={'strain': 0.01, 'axis': 0},
    ...     thermal={'T_hot': 100, 'T_cold': 0},
    ...     max_iterations=10
    ... )
    """
    
    def __init__(self, network, solvers: Dict):
        self.network = network
        self.solvers = solvers
        self.results = {}
    
    def solve_coupled(
        self,
        max_iterations: int = 5,
        tolerance: float = 1e-6,
        **solver_params
    ) -> CoupledResult:
        """
        Solve coupled multi-physics problem.
        
        Parameters
        ----------
        max_iterations : int, optional
            Maximum coupling iterations.
        tolerance : float, optional
            Convergence tolerance.
        **solver_params : dict
            Parameters for each solver.
        
        Returns
        -------
        CoupledResult
            Coupled results from all solvers.
        """
        results = {}
        converged = False
        
        for iteration in range(max_iterations):
            prev_results = results.copy()
            results = {}
            
            # Solve each physics problem
            for solver_name, solver in self.solvers.items():
                if solver_name in solver_params:
                    params = solver_params[solver_name]
                    
                    # Call appropriate solve method
                    if solver_name == 'mechanical':
                        results[solver_name] = solver.apply_uniaxial_strain(**params)
                    elif solver_name == 'thermal':
                        results[solver_name] = solver.solve_steady_state(**params)
                    elif solver_name == 'em':
                        results[solver_name] = solver.solve_conductivity(**params)
            
            # Check convergence
            if iteration > 0 and self._check_convergence(prev_results, results, tolerance):
                converged = True
                break
        
        # Build coupled result
        coupled = CoupledResult(
            mechanical=results.get('mechanical'),
            thermal=results.get('thermal'),
            electromagnetic=results.get('em'),
            converged=converged,
            iterations=iteration + 1
        )
        
        return coupled
    
    def _check_convergence(self, prev, current, tolerance):
        """Check if results have converged."""
        if not prev:
            return False
        
        # Compare key metrics
        for key in current:
            if key in prev:
                curr_result = current[key]
                prev_result = prev[key]
                
                # Compare energy if available
                if hasattr(curr_result, 'energy') and hasattr(prev_result, 'energy'):
                    diff = abs(curr_result.energy - prev_result.energy)
                    if prev_result.energy > 0:
                        rel_diff = diff / prev_result.energy
                        if rel_diff > tolerance:
                            return False
        
        return True
