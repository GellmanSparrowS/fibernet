"""
Batch simulation utilities.

Tools for running multiple simulations in parallel or sequentially,
with automatic result collection and error handling.
"""

import numpy as np
from typing import Callable, Dict, List, Any, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import traceback


@dataclass
class BatchResult:
    """Container for batch simulation results."""
    results: List[Any] = field(default_factory=list)
    errors: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    @property
    def success_count(self) -> int:
        return len(self.results)
    
    @property
    def error_count(self) -> int:
        return len(self.errors)
    
    @property
    def total_count(self) -> int:
        return self.success_count + self.error_count
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for easy inspection."""
        return {
            'success_count': self.success_count,
            'error_count': self.error_count,
            'total_count': self.total_count,
            'errors': self.errors,
            'metadata': self.metadata,
        }


def batch_simulate(
    networks: List,
    simulation_func: Callable,
    parallel: bool = False,
    max_workers: Optional[int] = None,
    show_progress: bool = True,
    raise_on_error: bool = False,
) -> BatchResult:
    """
    Run simulations on multiple networks.

    Parameters
    ----------
    networks : list
        List of FiberNetwork objects.
    simulation_func : callable
        Function that takes a network and returns results.
    parallel : bool, optional
        Whether to run in parallel (uses ThreadPoolExecutor).
        Default is False (sequential).
    max_workers : int, optional
        Maximum number of workers for parallel execution.
    show_progress : bool, optional
        Whether to show progress (requires tqdm).
    raise_on_error : bool, optional
        Whether to raise exceptions immediately. Default False (collect errors).

    Returns
    -------
    BatchResult
        Container with results and errors.

    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.sim import FiberFEM
    >>> 
    >>> # Generate multiple networks
    >>> networks = [gen.random_straight_2d(num_fibers=n, fiber_length=10, box_size=(30, 30), seed=i)
    ...             for i, n in enumerate([30, 50, 70, 90])]
    >>> 
    >>> # Define simulation
    >>> def simulate(net):
    ...     fem = FiberFEM(net, segments_per_fiber=5)
    ...     result = fem.apply_uniaxial_strain(strain=0.01, axis=0)
    ...     return {'energy': result.energy, 'max_displacement': result.max_displacement()}
    >>> 
    >>> # Run batch simulation
    >>> batch_result = batch_simulate(networks, simulate, parallel=True)
    >>> print(f"Success: {batch_result.success_count}/{batch_result.total_count}")
    """
    results = []
    errors = []
    
    if parallel:
        # Use ThreadPoolExecutor for I/O-bound tasks
        # Note: ProcessPoolExecutor has issues with pickling custom objects
        executor = ThreadPoolExecutor(max_workers=max_workers)
        
        futures = {executor.submit(simulation_func, net): i for i, net in enumerate(networks)}
        
        if show_progress:
            try:
                from tqdm import tqdm
                future_iter = tqdm(as_completed(futures), total=len(futures), desc="Simulating")
            except ImportError:
                future_iter = as_completed(futures)
        else:
            future_iter = as_completed(futures)
        
        for future in future_iter:
            idx = futures[future]
            try:
                result = future.result()
                results.append((idx, result))
            except Exception as e:
                if raise_on_error:
                    raise
                errors.append({
                    'index': idx,
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                })
        
        executor.shutdown()
    else:
        # Sequential execution
        iterator = enumerate(networks)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(list(iterator), desc="Simulating")
            except ImportError:
                pass
        
        for idx, net in iterator:
            try:
                result = simulation_func(net)
                results.append((idx, result))
            except Exception as e:
                if raise_on_error:
                    raise
                errors.append({
                    'index': idx,
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                })
    
    # Sort results by index
    results.sort(key=lambda x: x[0])
    results = [r for _, r in results]
    
    return BatchResult(
        results=results,
        errors=errors,
        metadata={'parallel': parallel, 'max_workers': max_workers}
    )


def parameter_study(
    param_grid: Dict[str, List[Any]],
    generator_func: Callable,
    simulation_func: Callable,
    parallel: bool = False,
    max_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run parameter study with automatic grid generation.

    Parameters
    ----------
    param_grid : dict
        Dictionary of parameter names to value lists.
    generator_func : callable
        Function that generates a network given parameters.
    simulation_func : callable
        Function that simulates a network.
    parallel : bool, optional
        Whether to run in parallel.
    max_workers : int, optional
        Maximum workers for parallel execution.

    Returns
    -------
    dict
        Dictionary with 'params', 'results', 'errors'.

    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.sim import FiberFEM
    >>> 
    >>> param_grid = {
    ...     'num_fibers': [30, 50, 70],
    ...     'fiber_length': [8, 10, 12],
    ... }
    >>> 
    >>> def generate(**kw):
    ...     return gen.random_straight_2d(**kw, box_size=(30, 30), seed=42)
    >>> 
    >>> def simulate(net):
    ...     fem = FiberFEM(net, segments_per_fiber=5)
    ...     result = fem.apply_uniaxial_strain(strain=0.01, axis=0)
    ...     return {'energy': result.energy}
    >>> 
    >>> study = parameter_study(param_grid, generate, simulate)
    >>> print(f"Completed {len(study['results'])} simulations")
    """
    import itertools
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))
    
    # Generate networks
    networks = []
    params_list = []
    for combo in combinations:
        params = dict(zip(param_names, combo))
        try:
            net = generator_func(**params)
            networks.append(net)
            params_list.append(params)
        except Exception as e:
            # Skip failed generations
            continue
    
    # Run simulations
    batch_result = batch_simulate(
        networks,
        simulation_func,
        parallel=parallel,
        max_workers=max_workers,
        show_progress=True,
    )
    
    return {
        'params': params_list,
        'results': batch_result.results,
        'errors': batch_result.errors,
    }
