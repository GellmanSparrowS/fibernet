"""
Uncertainty Quantification Module

Provides Monte Carlo ensemble methods for:
- Statistical property estimation
- Convergence analysis
- Confidence intervals
- Sensitivity analysis

This is essential for research publications to quantify uncertainty
in simulation results.
"""

import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
from dataclasses import dataclass, field
import warnings


@dataclass
class EnsembleResult:
    """Result of ensemble Monte Carlo analysis.
    
    Attributes
    ----------
    values : np.ndarray
        Array of computed property values.
    mean : float
        Mean value.
    std : float
        Standard deviation.
    min_val : float
        Minimum value.
    max_val : float
        Maximum value.
    confidence_interval : tuple
        95% confidence interval (lower, upper).
    cv : float
        Coefficient of variation (std/mean).
    """
    values: np.ndarray = None
    mean: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    cv: float = 0.0
    num_samples: int = 0
    converged: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'values': self.values.tolist() if self.values is not None else [],
            'mean': self.mean,
            'std': self.std,
            'min': self.min_val,
            'max': self.max_val,
            'ci_lower': self.confidence_interval[0],
            'ci_upper': self.confidence_interval[1],
            'cv': self.cv,
            'num_samples': self.num_samples,
            'converged': self.converged,
        }


def monte_carlo_ensemble(
    network_generator: Callable,
    property_function: Callable,
    num_samples: int = 100,
    generator_kwargs: Dict = None,
    property_kwargs: Dict = None,
    convergence_check: bool = True,
    convergence_threshold: float = 0.05,
    verbose: bool = False,
) -> EnsembleResult:
    """Run Monte Carlo ensemble analysis.
    
    Parameters
    ----------
    network_generator : callable
        Function that generates a FiberNetwork. Should accept a 'seed' parameter.
    property_function : callable
        Function that computes a property from a network.
    num_samples : int
        Number of Monte Carlo samples.
    generator_kwargs : dict
        Keyword arguments for network generator.
    property_kwargs : dict
        Keyword arguments for property function.
    convergence_check : bool
        If True, check for convergence.
    convergence_threshold : float
        Relative change threshold for convergence.
    verbose : bool
        Print progress.
    
    Returns
    -------
    result : EnsembleResult
        Ensemble analysis results.
    
    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.sim.uncertainty import monte_carlo_ensemble
    >>> from fibernet.analysis import MorphologyAnalyzer
    >>> 
    >>> def compute_order(net):
    ...     return MorphologyAnalyzer(net).nematic_order_parameter()
    >>> 
    >>> result = monte_carlo_ensemble(
    ...     gen.random_straight_2d,
    ...     compute_order,
    ...     num_samples=50,
    ...     generator_kwargs={'num_fibers': 100}
    ... )
    >>> print(f"Order = {result.mean:.3f} ± {result.std:.3f}")
    """
    if generator_kwargs is None:
        generator_kwargs = {}
    if property_kwargs is None:
        property_kwargs = {}
    
    values = []
    
    for i in range(num_samples):
        # Generate network with different seed
        net = network_generator(seed=i, **generator_kwargs)
        
        # Compute property
        try:
            value = property_function(net, **property_kwargs)
            values.append(float(value))
        except Exception as e:
            if verbose:
                print(f"Sample {i} failed: {e}")
            continue
        
        if verbose and (i + 1) % 10 == 0:
            current_mean = np.mean(values)
            current_std = np.std(values)
            print(f"Sample {i+1}/{num_samples}: mean={current_mean:.4e}, std={current_std:.4e}")
    
    values = np.array(values)
    
    if len(values) == 0:
        warnings.warn("No successful samples")
        return EnsembleResult(num_samples=0)
    
    # Statistics
    mean = np.mean(values)
    std = np.std(values)
    min_val = np.min(values)
    max_val = np.max(values)
    
    # 95% confidence interval
    ci_lower = mean - 1.96 * std / np.sqrt(len(values))
    ci_upper = mean + 1.96 * std / np.sqrt(len(values))
    
    # Coefficient of variation
    cv = std / abs(mean) if abs(mean) > 1e-10 else float('inf')
    
    # Convergence check
    converged = False
    if convergence_check and len(values) > 10:
        # Check if last 10% of samples give similar mean
        split = int(len(values) * 0.9)
        early_mean = np.mean(values[:split])
        late_mean = np.mean(values[split:])
        
        if abs(early_mean) > 1e-10:
            rel_change = abs(late_mean - early_mean) / abs(early_mean)
            converged = rel_change < convergence_threshold
    
    result = EnsembleResult(
        values=values,
        mean=mean,
        std=std,
        min_val=min_val,
        max_val=max_val,
        confidence_interval=(ci_lower, ci_upper),
        cv=cv,
        num_samples=len(values),
        converged=converged,
    )
    
    if verbose:
        print(f"\nEnsemble results:")
        print(f"  Mean: {mean:.4e}")
        print(f"  Std:  {std:.4e}")
        print(f"  CV:   {cv:.4f}")
        print(f"  95% CI: [{ci_lower:.4e}, {ci_upper:.4e}]")
        print(f"  Converged: {converged}")
    
    return result


def sensitivity_analysis(
    network_generator: Callable,
    property_function: Callable,
    parameter_name: str,
    parameter_values: List,
    num_samples_per_value: int = 20,
    generator_kwargs: Dict = None,
    property_kwargs: Dict = None,
) -> Dict:
    """Perform sensitivity analysis on a parameter.
    
    Parameters
    ----------
    network_generator : callable
        Function that generates a FiberNetwork.
    property_function : callable
        Function that computes a property.
    parameter_name : str
        Name of parameter to vary.
    parameter_values : list
        Values to test.
    num_samples_per_value : int
        Number of MC samples per parameter value.
    generator_kwargs : dict
        Base kwargs for generator.
    property_kwargs : dict
        Kwargs for property function.
    
    Returns
    -------
    results : dict
        Dictionary with 'values', 'means', 'stds' lists.
    
    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.sim.uncertainty import sensitivity_analysis
    >>> from fibernet.analysis import MorphologyAnalyzer
    >>> 
    >>> def compute_order(net):
    ...     return MorphologyAnalyzer(net).nematic_order_parameter()
    >>> 
    >>> results = sensitivity_analysis(
    ...     gen.random_straight_2d,
    ...     compute_order,
    ...     'num_fibers',
    ...     [50, 100, 200],
    ...     num_samples_per_value=10
    ... )
    """
    if generator_kwargs is None:
        generator_kwargs = {}
    if property_kwargs is None:
        property_kwargs = {}
    
    all_values = []
    all_means = []
    all_stds = []
    
    for param_val in parameter_values:
        values = []
        
        for i in range(num_samples_per_value):
            kwargs = generator_kwargs.copy()
            kwargs[parameter_name] = param_val
            kwargs['seed'] = i
            
            try:
                net = network_generator(**kwargs)
                value = property_function(net, **property_kwargs)
                values.append(float(value))
            except Exception:
                continue
        
        if len(values) > 0:
            all_values.append(param_val)
            all_means.append(np.mean(values))
            all_stds.append(np.std(values))
    
    return {
        'values': all_values,
        'means': all_means,
        'stds': all_stds,
    }


def convergence_study(
    network_generator: Callable,
    property_function: Callable,
    sample_sizes: List[int] = None,
    num_repeats: int = 3,
    generator_kwargs: Dict = None,
    property_kwargs: Dict = None,
) -> Dict:
    """Perform convergence study with increasing sample sizes.
    
    Parameters
    ----------
    network_generator : callable
        Function that generates a FiberNetwork.
    property_function : callable
        Function that computes a property.
    sample_sizes : list of int
        Sample sizes to test.
    num_repeats : int
        Number of repeats per sample size.
    generator_kwargs : dict
        Kwargs for generator.
    property_kwargs : dict
        Kwargs for property function.
    
    Returns
    -------
    results : dict
        Dictionary with convergence data.
    """
    if sample_sizes is None:
        sample_sizes = [10, 20, 50, 100, 200]
    
    results = {
        'sample_sizes': [],
        'means': [],
        'stds': [],
        'ci_widths': [],
    }
    
    for n in sample_sizes:
        means = []
        
        for r in range(num_repeats):
            result = monte_carlo_ensemble(
                network_generator,
                property_function,
                num_samples=n,
                generator_kwargs=generator_kwargs,
                property_kwargs=property_kwargs,
                convergence_check=False,
            )
            means.append(result.mean)
        
        results['sample_sizes'].append(n)
        results['means'].append(np.mean(means))
        results['stds'].append(np.std(means))
        
        # CI width
        ci_width = 2 * 1.96 * np.std(means) / np.sqrt(len(means))
        results['ci_widths'].append(ci_width)
    
    return results
