"""
Design of Experiments (DOE) Module for Fiber Networks

Provides tools for systematic parameter studies:
- Grid search (full factorial)
- Latin hypercube sampling
- Random sampling
- Parameter sweep workflows
- Result aggregation and analysis

References:
- Montgomery, D.C., "Design and Analysis of Experiments", Wiley, 2017
- Santner et al., "The Design and Analysis of Computer Experiments", Springer, 2018
"""

import numpy as np
import inspect
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from itertools import product
import warnings

from fibernet.core.network import FiberNetwork


@dataclass
class ExperimentDesign:
    """Design of experiments specification."""
    parameters: Dict[str, List[Any]] = field(default_factory=dict)
    parameter_names: List[str] = field(default_factory=list)
    design_points: np.ndarray = field(default_factory=lambda: np.array([]))
    num_points: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'parameters': self.parameters,
            'parameter_names': self.parameter_names,
            'num_points': self.num_points,
        }


@dataclass
class ExperimentResult:
    """Result from a single experiment."""
    parameters: Dict[str, Any] = field(default_factory=dict)
    network: Optional[FiberNetwork] = None
    outputs: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'parameters': self.parameters,
            'outputs': self.outputs,
        }


@dataclass
class SweepResult:
    """Result from parameter sweep."""
    design: Optional[ExperimentDesign] = None
    results: List[ExperimentResult] = field(default_factory=list)
    
    def to_dataframe(self):
        """Convert results to pandas DataFrame."""
        try:
            import pandas as pd
            
            data = []
            for result in self.results:
                row = {}
                row.update(result.parameters)
                row.update(result.outputs)
                data.append(row)
            
            return pd.DataFrame(data)
        except ImportError:
            warnings.warn("Pandas required for DataFrame export")
            return None
    
    def to_dict(self) -> Dict:
        return {
            'design': self.design.to_dict() if self.design else None,
            'num_results': len(self.results),
            'results': [r.to_dict() for r in self.results],
        }




def _convert_params(generator_func: Callable, params: Dict[str, Any]) -> Dict[str, Any]:
    """Auto-convert parameter types based on function signature."""
    try:
        sig = inspect.signature(generator_func)
        converted = {}
        for name, value in params.items():
            if name in sig.parameters:
                param = sig.parameters[name]
                if param.annotation == int and isinstance(value, (float, np.floating)):
                    converted[name] = int(round(value))
                else:
                    converted[name] = value
            else:
                converted[name] = value
        return converted
    except Exception:
        return params


class DesignOfExperiments:
    """Design and execute parameter studies for fiber networks.
    
    Parameters
    ----------
    generator_func : callable
        Network generator function.
    generator_params : dict
        Base parameters for generator.
    output_func : callable, optional
        Function to compute outputs from network.
        Signature: output_func(network) -> dict
    
    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.doe import DesignOfExperiments
    >>> from fibernet.sim import FiberFEM
    >>> 
    >>> # Define parameter ranges
    >>> params = {
    ...     'num_fibers': [50, 100, 150],
    ...     'fiber_length': [5.0, 10.0, 15.0],
    ... }
    >>> 
    >>> # Define output function
    >>> def compute_properties(net):
    ...     fem = FiberFEM(net)
    ...     return {'modulus': fem.effective_modulus()}
    >>> 
    >>> # Create and run experiment
    >>> doe = DesignOfExperiments(gen.random_straight_2d, {}, compute_properties)
    >>> result = doe.grid_search(params)
    >>> print(f"Ran {len(result.results)} experiments")
    """
    
    def __init__(
        self,
        generator_func: Callable,
        generator_params: Dict[str, Any],
        output_func: Optional[Callable[[FiberNetwork], Dict]] = None,
    ):
        self.generator_func = generator_func
        self.generator_params = generator_params
        self.output_func = output_func
    
    def grid_search(
        self,
        parameter_ranges: Dict[str, List[Any]],
        seed: Optional[int] = None,
    ) -> SweepResult:
        """Full factorial grid search.
        
        Parameters
        ----------
        parameter_ranges : dict
            Dictionary mapping parameter names to lists of values.
        seed : int, optional
            Random seed for reproducibility.
        
        Returns
        -------
        result : SweepResult
            Results from all parameter combinations.
        """
        # Create experiment design
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        # Generate all combinations
        combinations = list(product(*param_values))
        
        design = ExperimentDesign(
            parameters=parameter_ranges,
            parameter_names=param_names,
            num_points=len(combinations),
        )
        
        # Run experiments
        results = []
        for combo in combinations:
            params = dict(zip(param_names, combo))
            
            # Merge with base parameters
            full_params = {**self.generator_params, **params}
            
            if seed is not None:
                full_params['seed'] = seed
            
            # Generate network
            try:
                full_params_converted = _convert_params(self.generator_func, full_params)
                network = self.generator_func(**full_params_converted)
                
                # Compute outputs
                outputs = {}
                if self.output_func:
                    outputs = self.output_func(network)
                
                result = ExperimentResult(
                    parameters=params,
                    network=network,
                    outputs=outputs,
                )
                results.append(result)
                
            except Exception as e:
                warnings.warn(f"Experiment failed for {params}: {e}")
                continue
        
        return SweepResult(design=design, results=results)
    
    def latin_hypercube(
        self,
        parameter_ranges: Dict[str, Tuple[float, float]],
        num_samples: int = 100,
        seed: Optional[int] = None,
    ) -> SweepResult:
        """Latin hypercube sampling.
        
        Parameters
        ----------
        parameter_ranges : dict
            Dictionary mapping parameter names to (min, max) tuples.
        num_samples : int
            Number of samples.
        seed : int, optional
            Random seed.
        
        Returns
        -------
        result : SweepResult
            Results from LHS samples.
        """
        if seed is not None:
            np.random.seed(seed)
        
        param_names = list(parameter_ranges.keys())
        param_ranges = list(parameter_ranges.values())
        num_params = len(param_names)
        
        # Generate Latin hypercube samples
        samples = np.zeros((num_samples, num_params))
        
        for i in range(num_params):
            # Divide [0, 1] into num_samples intervals
            intervals = np.linspace(0, 1, num_samples + 1)
            # Sample uniformly from each interval
            points = np.random.uniform(intervals[:-1], intervals[1:])
            # Shuffle
            np.random.shuffle(points)
            samples[:, i] = points
        
        # Scale to parameter ranges
        for i, (pmin, pmax) in enumerate(param_ranges):
            samples[:, i] = pmin + samples[:, i] * (pmax - pmin)
        
        # Create design
        design = ExperimentDesign(
            parameters={name: list(samples[:, i]) for i, name in enumerate(param_names)},
            parameter_names=param_names,
            num_points=num_samples,
            design_points=samples,
        )
        
        # Run experiments
        results = []
        for i in range(num_samples):
            params = {name: samples[i, j] for j, name in enumerate(param_names)}
            
            full_params = {**self.generator_params, **params}
            
            if seed is not None:
                full_params['seed'] = seed + i
            
            try:
                full_params_converted = _convert_params(self.generator_func, full_params)
                network = self.generator_func(**full_params_converted)
                
                outputs = {}
                if self.output_func:
                    outputs = self.output_func(network)
                
                result = ExperimentResult(
                    parameters=params,
                    network=network,
                    outputs=outputs,
                )
                results.append(result)
                
            except Exception as e:
                warnings.warn(f"Experiment failed for {params}: {e}")
                continue
        
        return SweepResult(design=design, results=results)
    
    def random_search(
        self,
        parameter_ranges: Dict[str, Tuple[float, float]],
        num_samples: int = 100,
        seed: Optional[int] = None,
    ) -> SweepResult:
        """Random sampling from parameter ranges.
        
        Parameters
        ----------
        parameter_ranges : dict
            Dictionary mapping parameter names to (min, max) tuples.
        num_samples : int
            Number of samples.
        seed : int, optional
            Random seed.
        
        Returns
        -------
        result : SweepResult
            Results from random samples.
        """
        if seed is not None:
            np.random.seed(seed)
        
        param_names = list(parameter_ranges.keys())
        param_ranges = list(parameter_ranges.values())
        
        # Generate random samples
        samples = []
        for pmin, pmax in param_ranges:
            samples.append(np.random.uniform(pmin, pmax, num_samples))
        samples = np.array(samples).T
        
        # Create design
        design = ExperimentDesign(
            parameters={name: list(samples[:, i]) for i, name in enumerate(param_names)},
            parameter_names=param_names,
            num_points=num_samples,
            design_points=samples,
        )
        
        # Run experiments
        results = []
        for i in range(num_samples):
            params = {name: samples[i, j] for j, name in enumerate(param_names)}
            
            full_params = {**self.generator_params, **params}
            
            if seed is not None:
                full_params['seed'] = seed + i
            
            try:
                full_params_converted = _convert_params(self.generator_func, full_params)
                network = self.generator_func(**full_params_converted)
                
                outputs = {}
                if self.output_func:
                    outputs = self.output_func(network)
                
                result = ExperimentResult(
                    parameters=params,
                    network=network,
                    outputs=outputs,
                )
                results.append(result)
                
            except Exception as e:
                warnings.warn(f"Experiment failed for {params}: {e}")
                continue
        
        return SweepResult(design=design, results=results)
    
    def sensitivity_analysis(
        self,
        sweep_result: SweepResult,
        output_key: str,
    ) -> Dict[str, float]:
        """Compute parameter sensitivity (correlation with output).
        
        Parameters
        ----------
        sweep_result : SweepResult
            Results from parameter sweep.
        output_key : str
            Key of output to analyze.
        
        Returns
        -------
        sensitivities : dict
            Correlation coefficients for each parameter.
        """
        if not sweep_result.results:
            return {}
        
        # Extract parameter values and outputs
        param_names = sweep_result.design.parameter_names
        outputs = np.array([r.outputs.get(output_key, 0) for r in sweep_result.results])
        
        sensitivities = {}
        for param in param_names:
            values = np.array([r.parameters.get(param, 0) for r in sweep_result.results])
            
            # Compute correlation
            if len(values) > 1 and np.std(values) > 0 and np.std(outputs) > 0:
                corr = np.corrcoef(values, outputs)[0, 1]
                sensitivities[param] = abs(corr)
            else:
                sensitivities[param] = 0.0
        
        return sensitivities


def run_parameter_sweep(
    generator_func: Callable,
    parameter_ranges: Dict[str, List[Any]],
    output_func: Optional[Callable] = None,
    method: str = 'grid',
    **kwargs,
) -> SweepResult:
    """Convenience function for parameter sweeps.
    
    Parameters
    ----------
    generator_func : callable
        Network generator function.
    parameter_ranges : dict
        Parameter ranges to sweep.
    output_func : callable, optional
        Function to compute outputs.
    method : str
        Sampling method: 'grid', 'lhs', 'random'.
    **kwargs
        Additional arguments.
    
    Returns
    -------
    result : SweepResult
        Sweep results.
    
    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.doe import run_parameter_sweep
    >>> 
    >>> params = {
    ...     'num_fibers': [50, 100, 150],
    ...     'fiber_length': [5.0, 10.0],
    ... }
    >>> 
    >>> result = run_parameter_sweep(
    ...     gen.random_straight_2d,
    ...     params,
    ...     method='grid'
    ... )
    >>> 
    >>> print(f"Ran {len(result.results)} experiments")
    """
    doe = DesignOfExperiments(generator_func, {}, output_func)
    
    if method == 'grid':
        return doe.grid_search(parameter_ranges, **kwargs)
    elif method in ['lhs', 'latin_hypercube']:
        return doe.latin_hypercube(parameter_ranges, **kwargs)
    elif method == 'random':
        return doe.random_search(parameter_ranges, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
