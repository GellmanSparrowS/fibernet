"""
SciPy-based Optimization for Fiber Networks

Provides optimization tools using SciPy:
- Energy minimization (geometry optimization)
- Parameter optimization (design optimization)
- Constraint satisfaction

References:
- SciPy optimization: https://docs.scipy.org/doc/scipy/reference/optimize.html
- Nocedal & Wright, "Numerical Optimization", Springer, 2006
"""

import numpy as np
from typing import Callable, Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from scipy import optimize
from scipy.optimize import minimize, differential_evolution, basinhopping

from fibernet.core.network import FiberNetwork


@dataclass
class OptimizationResult:
    """Container for optimization results."""
    success: bool
    message: str
    num_iterations: int
    num_function_evals: int
    final_value: float
    final_params: np.ndarray
    history: List[float]


class EnergyMinimizer:
    """
    Minimize energy of fiber network using SciPy optimizers.
    
    Methods available:
    - 'L-BFGS-B': Limited-memory BFGS with bounds
    - 'CG': Conjugate gradient
    - 'BFGS': Broyden-Fletcher-Goldfarb-Shanno
    - 'Nelder-Mead': Simplex method (derivative-free)
    - 'Powell': Powell's method (derivative-free)
    
    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.sim.optimization import EnergyMinimizer
    >>> 
    >>> net = gen.random_straight_3d(num_fibers=20, box_size=(50, 50, 50))
    >>> minimizer = EnergyMinimizer(net)
    >>> result = minimizer.minimize(method='L-BFGS-B')
    >>> print(f"Energy reduced by {result.final_value:.2e}")
    """
    
    def __init__(self, network: FiberNetwork):
        """
        Initialize energy minimizer.
        
        Parameters
        ----------
        network : FiberNetwork
            Network to minimize
        """
        self.network = network
        self._build_variables()
    
    def _build_variables(self):
        """Build optimization variables from network."""
        # Flatten all fiber positions into a single vector
        positions = []
        self._fiber_offsets = []
        self._fiber_lengths = []
        
        offset = 0
        for fiber in self.network.fibers:
            pts = fiber.centerline
            positions.extend(pts.flatten())
            self._fiber_offsets.append(offset)
            self._fiber_lengths.append(len(pts))
            offset += len(pts) * 3
        
        self.x0 = np.array(positions)
        self.num_vars = len(self.x0)
    
    def _positions_to_x(self, positions: List[np.ndarray]) -> np.ndarray:
        """Convert list of position arrays to flat vector."""
        return np.concatenate([p.flatten() for p in positions])
    
    def _x_to_positions(self, x: np.ndarray) -> List[np.ndarray]:
        """Convert flat vector to list of position arrays."""
        positions = []
        for i, (offset, length) in enumerate(zip(self._fiber_offsets, self._fiber_lengths)):
            pts = x[offset:offset + length * 3].reshape(length, 3)
            positions.append(pts)
        return positions
    
    def _energy_function(self, x: np.ndarray) -> float:
        """
        Compute total energy of network.
        
        Energy includes:
        - Stretching energy (axial deformation)
        - Bending energy (curvature)
        """
        positions = self._x_to_positions(x)
        energy = 0.0
        
        for i, fiber in enumerate(self.network.fibers):
            pts = positions[i]
            ref_pts = fiber.centerline
            
            E = fiber.material.youngs_modulus
            A = fiber.cross_section_area
            
            # Stretching energy
            for j in range(len(pts) - 1):
                L0 = np.linalg.norm(ref_pts[j + 1] - ref_pts[j])
                L = np.linalg.norm(pts[j + 1] - pts[j])
                
                if L0 > 1e-12:
                    strain = (L - L0) / L0
                    energy += 0.5 * E * A * L0 * strain**2
            
            # Bending energy
            if len(pts) > 2:
                EI = E * np.pi * fiber.radius**4 / 4.0
                for j in range(1, len(pts) - 1):
                    # Discrete curvature
                    kappa = pts[j - 1] - 2 * pts[j] + pts[j + 1]
                    kappa_ref = ref_pts[j - 1] - 2 * ref_pts[j] + ref_pts[j + 1]
                    
                    ds = np.linalg.norm(ref_pts[j] - ref_pts[j - 1])
                    if ds > 1e-12:
                        delta_kappa = kappa - kappa_ref
                        energy += 0.5 * EI * np.sum(delta_kappa**2) / ds**3
        
        return energy
    
    def minimize(
        self,
        method: str = 'L-BFGS-B',
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
        fixed_nodes: Optional[List[Tuple[int, int]]] = None,
    ) -> OptimizationResult:
        """
        Minimize energy.
        
        Parameters
        ----------
        method : str
            Optimization method
        max_iterations : int
            Maximum iterations
        tolerance : float
            Convergence tolerance
        fixed_nodes : list of tuples, optional
            List of (fiber_idx, node_idx) to keep fixed
        
        Returns
        -------
        result : OptimizationResult
            Optimization result
        """
        # Build bounds and constraints for fixed nodes
        bounds = None
        if fixed_nodes:
            bounds = []
            for i in range(self.num_vars):
                bounds.append((None, None))
            
            # Fix specific nodes
            for fiber_idx, node_idx in fixed_nodes:
                offset = self._fiber_offsets[fiber_idx] + node_idx * 3
                for k in range(3):
                    bounds[offset + k] = (self.x0[offset + k], self.x0[offset + k])
        
        # Track history
        history = []
        
        def callback(x):
            energy = self._energy_function(x)
            history.append(energy)
        
        # Run optimization
        result = minimize(
            self._energy_function,
            self.x0,
            method=method,
            bounds=bounds,
            options={'maxiter': max_iterations},
            callback=callback
        )
        
        return OptimizationResult(
            success=result.success,
            message=result.message,
            num_iterations=result.nit,
            num_function_evals=result.nfev,
            final_value=result.fun,
            final_params=result.x,
            history=history
        )
    
    def update_network(self, result: OptimizationResult):
        """
        Update network with optimized positions.
        
        Parameters
        ----------
        result : OptimizationResult
            Optimization result containing final positions
        """
        positions = self._x_to_positions(result.final_params)
        
        for i, fiber in enumerate(self.network.fibers):
            fiber.centerline = positions[i]


class ParameterOptimizer:
    """
    Optimize network parameters for target properties.
    
    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.sim.optimization import ParameterOptimizer
    >>> 
    >>> # Define objective function
    >>> def objective(params):
    ...     num_fibers = int(params[0])
    ...     fiber_length = params[1]
    ...     net = gen.random_straight_2d(num_fibers=num_fibers, fiber_length=fiber_length)
    ...     # Compute some property
    ...     return -len(net.fibers)  # Maximize fiber count (negative for minimization)
    >>> 
    >>> # Optimize
    >>> optimizer = ParameterOptimizer(objective)
    >>> result = optimizer.optimize(
    ...     bounds=[(10, 100), (5.0, 20.0)]
    ... )
    """
    
    def __init__(self, objective: Callable[[np.ndarray], float]):
        """
        Initialize parameter optimizer.
        
        Parameters
        ----------
        objective : callable
            Objective function to minimize: f(params) -> scalar
        """
        self.objective = objective
    
    def optimize(
        self,
        bounds: List[Tuple[float, float]],
        method: str = 'L-BFGS-B',
        x0: Optional[np.ndarray] = None,
        max_iterations: int = 1000,
    ) -> OptimizationResult:
        """
        Optimize parameters.
        
        Parameters
        ----------
        bounds : list of tuples
            Parameter bounds [(min, max), ...]
        method : str
            Optimization method
        x0 : np.ndarray, optional
            Initial guess (midpoint of bounds if None)
        max_iterations : int
            Maximum iterations
        
        Returns
        -------
        result : OptimizationResult
            Optimization result
        """
        # Default initial guess
        if x0 is None:
            x0 = np.array([(b[0] + b[1]) / 2 for b in bounds])
        
        # Track history
        history = []
        
        def callback(x):
            val = self.objective(x)
            history.append(val)
        
        # Run optimization
        result = minimize(
            self.objective,
            x0,
            method=method,
            bounds=bounds,
            options={'maxiter': max_iterations},
            callback=callback
        )
        
        return OptimizationResult(
            success=result.success,
            message=result.message,
            num_iterations=result.nit,
            num_function_evals=result.nfev,
            final_value=result.fun,
            final_params=result.x,
            history=history
        )
    
    def optimize_global(
        self,
        bounds: List[Tuple[float, float]],
        max_iterations: int = 1000,
        seed: Optional[int] = None,
    ) -> OptimizationResult:
        """
        Global optimization using differential evolution.
        
        Parameters
        ----------
        bounds : list of tuples
            Parameter bounds
        max_iterations : int
            Maximum iterations
        seed : int, optional
            Random seed
        
        Returns
        -------
        result : OptimizationResult
            Optimization result
        """
        history = []
        
        def callback(x, convergence):
            val = self.objective(x)
            history.append(val)
        
        result = differential_evolution(
            self.objective,
            bounds,
            maxiter=max_iterations,
            seed=seed,
            callback=callback
        )
        
        return OptimizationResult(
            success=result.success,
            message=result.message,
            num_iterations=result.nit,
            num_function_evals=result.nfev,
            final_value=result.fun,
            final_params=result.x,
            history=history
        )


