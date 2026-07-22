"""
Statistical ensemble generation for fiber networks.

Provides tools for:
- Generating multiple network realizations with same parameters
- Computing ensemble averages and statistics
- Parallel generation for large ensembles
- Reproducible random seed management

This is essential for research where statistical properties need to be
averaged over multiple realizations to reduce noise.
"""

import numpy as np
from typing import Callable, Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import warnings


@dataclass
class EnsembleResult:
    """
    Container for ensemble generation results.
    
    Attributes
    ----------
    networks : list
        List of generated FiberNetwork objects
    seeds : list
        Random seeds used for each network
    statistics : dict
        Computed statistics over the ensemble
    metadata : dict
        Generation parameters and metadata
    """
    networks: List[Any] = field(default_factory=list)
    seeds: List[int] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_networks(self) -> int:
        """Number of networks in ensemble."""
        return len(self.networks)
    
    def compute_statistics(self, metrics: Optional[Dict[str, Callable]] = None) -> Dict[str, Any]:
        """
        Compute statistics over the ensemble.
        
        Parameters
        ----------
        metrics : dict, optional
            Dictionary of metric_name -> function(network) -> value.
            If None, uses default metrics.
        
        Returns
        -------
        dict
            Statistics with mean, std, min, max for each metric
        """
        if metrics is None:
            # Default metrics
            metrics = {
                'num_fibers': lambda net: net.num_fibers,
                'num_crosslinks': lambda net: net.num_crosslinks,
                'mean_length': lambda net: np.mean([f.length for f in net.fibers]),
                'total_length': lambda net: sum(f.length for f in net.fibers),
                'mean_radius': lambda net: np.mean([f.radius for f in net.fibers]),
            }
        
        stats = {}
        for metric_name, metric_func in metrics.items():
            values = []
            for net in self.networks:
                try:
                    val = metric_func(net)
                    values.append(val)
                except Exception as e:
                    warnings.warn(f"Failed to compute {metric_name}: {e}")
            
            if values:
                values = np.array(values)
                stats[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values,
                }
        
        self.statistics = stats
        return stats
    
    def get_network(self, index: int):
        """Get network by index."""
        return self.networks[index]
    
    def __getitem__(self, index):
        """Get network by index."""
        return self.networks[index]
    
    def __len__(self):
        """Number of networks."""
        return len(self.networks)
    
    def __iter__(self):
        """Iterate over networks."""
        return iter(self.networks)
    
    def summary(self) -> str:
        """Generate summary string."""
        if not self.statistics:
            self.compute_statistics()
        
        lines = [
            f"Ensemble: {self.num_networks} networks",
            f"Seeds: {self.seeds[:5]}{'...' if len(self.seeds) > 5 else ''}",
            "",
            "Statistics:",
        ]
        
        for metric, stats in self.statistics.items():
            lines.append(
                f"  {metric}: {stats['mean']:.3f} ± {stats['std']:.3f} "
                f"[{stats['min']:.3f}, {stats['max']:.3f}]"
            )
        
        return '\n'.join(lines)


def generate_ensemble(
    generator_func: Callable,
    num_networks: int = 10,
    base_seed: int = 42,
    parallel: bool = False,
    max_workers: Optional[int] = None,
    show_progress: bool = True,
    **generator_kwargs
) -> EnsembleResult:
    """
    Generate ensemble of fiber networks with different random seeds.
    
    Parameters
    ----------
    generator_func : callable
        Network generator function that accepts 'seed' parameter
    num_networks : int
        Number of networks to generate
    base_seed : int
        Base seed for reproducibility (seeds will be base_seed, base_seed+1, ...)
    parallel : bool
        Whether to generate in parallel
    max_workers : int, optional
        Maximum number of workers for parallel generation
    show_progress : bool
        Whether to show progress bar
    **generator_kwargs
        Additional arguments passed to generator_func
    
    Returns
    -------
    EnsembleResult
        Container with generated networks and statistics
    
    Examples
    --------
    >>> from fibernet import gen
    >>> ensemble = generate_ensemble(
    ...     gen.random_straight_2d,
    ...     num_networks=20,
    ...     num_fibers=100,
    ...     fiber_length=10,
    ...     box_size=(50, 50)
    ... )
    >>> print(ensemble.summary())
    >>> print(f"Mean num_crosslinks: {ensemble.statistics['num_crosslinks']['mean']:.1f}")
    """
    networks = []
    seeds = [base_seed + i for i in range(num_networks)]
    
    if parallel:
        # Use ThreadPoolExecutor (ProcessPoolExecutor has pickling issues)
        executor = ThreadPoolExecutor(max_workers=max_workers)
        
        futures = {}
        for i, seed in enumerate(seeds):
            kwargs = generator_kwargs.copy()
            kwargs['seed'] = seed
            futures[executor.submit(generator_func, **kwargs)] = i
        
        if show_progress:
            try:
                from tqdm import tqdm
                future_iter = tqdm(as_completed(futures), total=len(futures), desc="Generating")
            except ImportError:
                future_iter = as_completed(futures)
        else:
            future_iter = as_completed(futures)
        
        results = [None] * num_networks
        for future in future_iter:
            idx = futures[future]
            try:
                net = future.result()
                results[idx] = net
            except Exception as e:
                warnings.warn(f"Failed to generate network {idx}: {e}")
                results[idx] = None
        
        executor.shutdown()
        
        # Filter out None values
        networks = [net for net in results if net is not None]
    else:
        # Sequential generation
        iterator = enumerate(seeds)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(list(iterator), desc="Generating")
            except ImportError:
                pass
        
        for i, seed in iterator:
            try:
                kwargs = generator_kwargs.copy()
                kwargs['seed'] = seed
                net = generator_func(**kwargs)
                networks.append(net)
            except Exception as e:
                warnings.warn(f"Failed to generate network {i}: {e}")
    
    result = EnsembleResult(
        networks=networks,
        seeds=seeds[:len(networks)],
        metadata={
            'generator': generator_func.__name__,
            'num_requested': num_networks,
            'num_generated': len(networks),
            'base_seed': base_seed,
            'parallel': parallel,
            'generator_kwargs': generator_kwargs,
        }
    )
    
    # Compute default statistics
    result.compute_statistics()
    
    return result


def ensemble_analysis(
    ensemble: EnsembleResult,
    analysis_func: Callable,
    parallel: bool = False,
    max_workers: Optional[int] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Run analysis on each network in ensemble and compute statistics.
    
    Parameters
    ----------
    ensemble : EnsembleResult
        Ensemble of networks
    analysis_func : callable
        Function that takes a network and returns dict of metrics
    parallel : bool
        Whether to run in parallel
    max_workers : int, optional
        Maximum number of workers
    show_progress : bool
        Whether to show progress bar
    
    Returns
    -------
    dict
        Statistics for each metric: {metric_name: {mean, std, min, max, values}}
    
    Examples
    --------
    >>> from fibernet.analysis import MorphologyAnalyzer
    >>> 
    >>> def analyze(net):
    ...     morph = MorphologyAnalyzer(net)
    ...     return {
    ...         'nematic_order': morph.nematic_order_parameter(),
    ...         'porosity': morph.porosity(),
    ...     }
    >>> 
    >>> stats = ensemble_analysis(ensemble, analyze)
    >>> print(f"Nematic order: {stats['nematic_order']['mean']:.3f} ± {stats['nematic_order']['std']:.3f}")
    """
    results = []
    
    if parallel:
        executor = ThreadPoolExecutor(max_workers=max_workers)
        futures = {executor.submit(analysis_func, net): i for i, net in enumerate(ensemble.networks)}
        
        if show_progress:
            try:
                from tqdm import tqdm
                future_iter = tqdm(as_completed(futures), total=len(futures), desc="Analyzing")
            except ImportError:
                future_iter = as_completed(futures)
        else:
            future_iter = as_completed(futures)
        
        for future in future_iter:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                warnings.warn(f"Analysis failed: {e}")
        
        executor.shutdown()
    else:
        iterator = enumerate(ensemble.networks)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(list(iterator), desc="Analyzing")
            except ImportError:
                pass
        
        for i, net in iterator:
            try:
                result = analysis_func(net)
                results.append(result)
            except Exception as e:
                warnings.warn(f"Analysis failed for network {i}: {e}")
    
    # Aggregate results
    if not results:
        return {}
    
    # Get all metric names
    metric_names = set()
    for result in results:
        if isinstance(result, dict):
            metric_names.update(result.keys())
    
    # Collect values for each metric
    stats = {}
    for metric_name in metric_names:
        values = []
        for result in results:
            if isinstance(result, dict) and metric_name in result:
                values.append(result[metric_name])
        
        if values:
            values = np.array(values)
            stats[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values,
            }
    
    return stats


def convergence_study(
    generator_func: Callable,
    analysis_func: Callable,
    ensemble_sizes: List[int] = [5, 10, 20, 50, 100],
    base_seed: int = 42,
    metric_name: str = 'mean_length',
    **generator_kwargs
) -> Dict[str, Any]:
    """
    Study convergence of ensemble statistics with increasing ensemble size.
    
    Parameters
    ----------
    generator_func : callable
        Network generator function
    analysis_func : callable
        Analysis function returning dict of metrics
    ensemble_sizes : list of int
        Ensemble sizes to test
    base_seed : int
        Base seed for reproducibility
    metric_name : str
        Metric to track for convergence
    **generator_kwargs
        Arguments for generator
    
    Returns
    -------
    dict
        Convergence data with ensemble_size, mean, std, error
    
    Examples
    --------
    >>> from fibernet import gen
    >>> from fibernet.analysis import MorphologyAnalyzer
    >>> 
    >>> def analyze(net):
    ...     morph = MorphologyAnalyzer(net)
    ...     return {'mean_length': morph.mean_fiber_length()}
    >>> 
    >>> convergence = convergence_study(
    ...     gen.random_straight_2d,
    ...     analyze,
    ...     ensemble_sizes=[5, 10, 20, 50],
    ...     num_fibers=100,
    ...     fiber_length=10,
    ...     box_size=(50, 50)
    ... )
    >>> for size, data in convergence.items():
    ...     print(f"N={size}: {data['mean']:.3f} ± {data['std']:.3f}")
    """
    convergence_data = {}
    
    for N in ensemble_sizes:
        print(f"Generating ensemble of size {N}...")
        ensemble = generate_ensemble(
            generator_func,
            num_networks=N,
            base_seed=base_seed,
            show_progress=False,
            **generator_kwargs
        )
        
        # Run analysis
        stats = ensemble_analysis(ensemble, analysis_func, show_progress=False)
        
        if metric_name in stats:
            metric_stats = stats[metric_name]
            convergence_data[N] = {
                'mean': metric_stats['mean'],
                'std': metric_stats['std'],
                'error': metric_stats['std'] / np.sqrt(N),  # Standard error
                'num_networks': N,
            }
            print(f"  {metric_name}: {metric_stats['mean']:.3f} ± {metric_stats['std']:.3f}")
    
    return convergence_data
