"""
Parametric study and sensitivity analysis utilities.

Tools for systematically varying parameters and analyzing their effects
on fiber network properties.
"""

import numpy as np
import itertools
from typing import Callable, Dict, List, Tuple, Any
from tqdm import tqdm


def parametric_sweep(
    param_ranges: Dict[str, List[Any]],
    generator_func: Callable,
    analysis_func: Callable,
    base_params: Dict[str, Any] = None,
    show_progress: bool = True
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Perform a parametric sweep over parameter combinations.
    
    Parameters
    ----------
    param_ranges : dict
        Dictionary mapping parameter names to lists of values.
        Example: {'num_fibers': [50, 100, 200], 'fiber_length': [5, 10, 15]}
    generator_func : callable
        Function that generates a FiberNetwork given parameters.
    analysis_func : callable
        Function that analyzes a network and returns metrics dict.
    base_params : dict, optional
        Base parameters to use for all combinations.
    show_progress : bool, optional
        Whether to show a progress bar (requires tqdm).
    
    Returns
    -------
    param_values : dict
        Dictionary of parameter value arrays.
    metrics : dict
        Dictionary of metric value arrays.
    
    Examples
    --------
    >>> from fibernet.gen import random_straight_2d
    >>> from fibernet.analysis import MorphologyAnalyzer
    >>> 
    >>> def analyze(net):
    ...     morph = MorphologyAnalyzer(net)
    ...     return {'nematic': morph.nematic_order_parameter(),
    ...             'porosity': morph.porosity()}
    >>> 
    >>> params, metrics = parametric_sweep(
    ...     {'num_fibers': [50, 100, 200], 'fiber_length': [5, 10, 15]},
    ...     lambda **kw: random_straight_2d(**kw, box_size=(50, 50), seed=42),
    ...     analyze
    ... )
    """
    if base_params is None:
        base_params = {}
    
    # Generate all combinations
    param_names = list(param_ranges.keys())
    param_lists = list(param_ranges.values())
    combinations = list(itertools.product(*param_lists))
    
    # Initialize results
    results = {name: [] for name in param_names}
    metric_results = {}
    
    # Run combinations
    iterator = tqdm(combinations) if show_progress else combinations
    for combo in iterator:
        # Build parameters
        params = base_params.copy()
        for name, value in zip(param_names, combo):
            params[name] = value
        
        # Generate network
        try:
            net = generator_func(**params)
            
            # Analyze
            metrics = analysis_func(net)
            
            # Store results
            for name, value in zip(param_names, combo):
                results[name].append(value)
            
            for metric_name, metric_value in metrics.items():
                if metric_name not in metric_results:
                    metric_results[metric_name] = []
                metric_results[metric_name].append(metric_value)
        
        except Exception as e:
            if show_progress:
                tqdm.write(f"Warning: Failed for params {params}: {e}")
            continue
    
    # Convert to arrays
    param_arrays = {name: np.array(vals) for name, vals in results.items()}
    metric_arrays = {name: np.array(vals) for name, vals in metric_results.items()}
    
    return param_arrays, metric_arrays


def sensitivity_analysis(
    param_name: str,
    param_values: List[Any],
    generator_func: Callable,
    analysis_func: Callable,
    base_params: Dict[str, Any] = None,
    num_samples: int = 1,
    show_progress: bool = True
) -> Dict[str, np.ndarray]:
    """
    Analyze sensitivity of metrics to a single parameter.
    
    Parameters
    ----------
    param_name : str
        Name of the parameter to vary.
    param_values : list
        List of parameter values to test.
    generator_func : callable
        Function that generates a FiberNetwork given parameters.
    analysis_func : callable
        Function that analyzes a network and returns metrics dict.
    base_params : dict, optional
        Base parameters for all other parameters.
    num_samples : int, optional
        Number of random samples per parameter value (for stochastic analysis).
    show_progress : bool, optional
        Whether to show progress bar.
    
    Returns
    -------
    results : dict
        Dictionary with 'param_values', 'metrics_mean', 'metrics_std'.
    
    Examples
    --------
    >>> results = sensitivity_analysis(
    ...     'num_fibers',
    ...     [50, 100, 150, 200],
    ...     lambda **kw: random_straight_2d(**kw, box_size=(50, 50)),
    ...     analyze,
    ...     num_samples=5
    ... )
    >>> print(results['metrics_mean']['nematic'])  # Mean nematic order per param value
    """
    if base_params is None:
        base_params = {}
    
    metric_values = {value: [] for value in param_values}
    
    iterator = tqdm(param_values) if show_progress else param_values
    for value in iterator:
        params = base_params.copy()
        params[param_name] = value
        
        for _ in range(num_samples):
            try:
                net = generator_func(**params)
                metrics = analysis_func(net)
                metric_values[value].append(metrics)
            except Exception as e:
                if show_progress:
                    tqdm.write(f"Warning: Failed for {param_name}={value}: {e}")
    
    # Compute statistics
    metric_names = list(metric_values[param_values[0]][0].keys()) if metric_values[param_values[0]] else []
    
    results = {
        'param_values': np.array(param_values),
        'metrics_mean': {},
        'metrics_std': {}
    }
    
    for metric_name in metric_names:
        means = []
        stds = []
        
        for value in param_values:
            if metric_values[value]:
                values = [m[metric_name] for m in metric_values[value]]
                means.append(np.mean(values))
                stds.append(np.std(values))
            else:
                means.append(np.nan)
                stds.append(np.nan)
        
        results['metrics_mean'][metric_name] = np.array(means)
        results['metrics_std'][metric_name] = np.array(stds)
    
    return results


def monte_carlo_analysis(
    generator_func: Callable,
    analysis_func: Callable,
    base_params: Dict[str, Any] = None,
    num_samples: int = 100,
    show_progress: bool = True
) -> Dict[str, np.ndarray]:
    """
    Perform Monte Carlo analysis with random parameter variations.
    
    Parameters
    ----------
    generator_func : callable
        Function that generates a FiberNetwork. Should accept a 'seed' parameter.
    analysis_func : callable
        Function that analyzes a network and returns metrics dict.
    base_params : dict, optional
        Base parameters for generation.
    num_samples : int, optional
        Number of random samples.
    show_progress : bool, optional
        Whether to show progress bar.
    
    Returns
    -------
    results : dict
        Dictionary with metric arrays and statistics.
    
    Examples
    --------
    >>> results = monte_carlo_analysis(
    ...     lambda seed: random_straight_2d(num_fibers=100, fiber_length=10,
    ...                                     box_size=(50, 50), seed=seed),
    ...     analyze,
    ...     num_samples=50
    ... )
    >>> print(f"Nematic order: {results['mean']['nematic']:.3f} ± {results['std']['nematic']:.3f}")
    """
    if base_params is None:
        base_params = {}
    
    all_metrics = []
    
    iterator = tqdm(range(num_samples)) if show_progress else range(num_samples)
    for i in iterator:
        params = base_params.copy()
        if 'seed' not in params:
            params['seed'] = i
        
        try:
            net = generator_func(**params)
            metrics = analysis_func(net)
            all_metrics.append(metrics)
        except Exception as e:
            if show_progress:
                tqdm.write(f"Warning: Sample {i} failed: {e}")
            continue
    
    if not all_metrics:
        return {'mean': {}, 'std': {}, 'samples': []}
    
    # Compute statistics
    metric_names = list(all_metrics[0].keys())
    results = {
        'mean': {},
        'std': {},
        'samples': all_metrics
    }
    
    for metric_name in metric_names:
        values = np.array([m[metric_name] for m in all_metrics])
        results['mean'][metric_name] = np.mean(values)
        results['std'][metric_name] = np.std(values)
    
    return results


def correlation_matrix(
    param_arrays: Dict[str, np.ndarray],
    metric_arrays: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute correlation matrix between parameters and metrics.
    
    Parameters
    ----------
    param_arrays : dict
        Dictionary of parameter value arrays.
    metric_arrays : dict
        Dictionary of metric value arrays.
    
    Returns
    -------
    correlations : ndarray
        Correlation matrix (params x metrics).
    p_values : ndarray
        P-values for correlations.
    """
    from scipy import stats
    
    param_names = list(param_arrays.keys())
    metric_names = list(metric_arrays.keys())
    
    n_params = len(param_names)
    n_metrics = len(metric_names)
    
    correlations = np.zeros((n_params, n_metrics))
    p_values = np.zeros((n_params, n_metrics))
    
    for i, pname in enumerate(param_names):
        for j, mname in enumerate(metric_names):
            r, p = stats.pearsonr(param_arrays[pname], metric_arrays[mname])
            correlations[i, j] = r
            p_values[i, j] = p
    
    return correlations, p_values
