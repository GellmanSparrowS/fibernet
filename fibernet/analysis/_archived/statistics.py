"""
Statistical Analysis Tools for Fiber Networks

Provides:
- Bootstrap confidence intervals
- Statistical hypothesis testing
- Correlation analysis
- Distribution fitting
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from scipy import stats
from fibernet.core.network import FiberNetwork


class StatisticalAnalyzer:
    """Advanced statistical analysis for fiber networks.
    
    Parameters
    ----------
    networks : List[FiberNetwork]
        List of fiber networks for ensemble statistics.
    """
    
    def __init__(self, networks: List[FiberNetwork]):
        self.networks = networks
        self._compute_ensemble_stats()
    
    def _compute_ensemble_stats(self):
        """Compute ensemble statistics."""
        self.num_samples = len(self.networks)
        
        # Collect properties
        self.num_fibers = [n.num_fibers for n in self.networks]
        self.num_crosslinks = [n.num_crosslinks for n in self.networks]
        
        from fibernet.analysis import MorphologyAnalyzer
        
        self.lengths = []
        self.tortuosities = []
        self.nematic_orders = []
        
        for net in self.networks:
            morph = MorphologyAnalyzer(net)
            report = morph.full_report()
            
            self.lengths.append(report.get('mean_length', 0))
            self.tortuosities.append(report.get('mean_tortuosity', 1.0))
            self.nematic_orders.append(report.get('nematic_order', 0))
    
    def bootstrap_confidence_interval(
        self,
        property_name: str = 'mean_length',
        confidence: float = 0.95,
        num_bootstrap: int = 1000,
    ) -> Tuple[float, float, float]:
        """Compute bootstrap confidence interval for a property.
        
        Parameters
        ----------
        property_name : str
            Property to analyze: 'mean_length', 'num_fibers', 'num_crosslinks',
            'mean_tortuosity', 'nematic_order'.
        confidence : float
            Confidence level (0-1).
        num_bootstrap : int
            Number of bootstrap samples.
        
        Returns
        -------
        Tuple[float, float, float]
            (mean, lower_bound, upper_bound)
        """
        # Get property values
        if property_name == 'mean_length':
            values = np.array(self.lengths)
        elif property_name == 'num_fibers':
            values = np.array(self.num_fibers, dtype=float)
        elif property_name == 'num_crosslinks':
            values = np.array(self.num_crosslinks, dtype=float)
        elif property_name == 'mean_tortuosity':
            values = np.array(self.tortuosities)
        elif property_name == 'nematic_order':
            values = np.array(self.nematic_orders)
        else:
            raise ValueError(f"Unknown property: {property_name}")
        
        if len(values) < 2:
            return np.mean(values), np.mean(values), np.mean(values)
        
        # Bootstrap resampling
        np.random.seed(42)
        bootstrap_means = []
        
        for _ in range(num_bootstrap):
            # Resample with replacement
            sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        # Compute confidence interval
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        mean = np.mean(values)
        
        return mean, lower, upper
    
    def compare_groups(
        self,
        group1: List[FiberNetwork],
        group2: List[FiberNetwork],
        property_name: str = 'mean_length',
        test: str = 'ttest',
    ) -> Dict[str, float]:
        """Compare two groups of networks using statistical tests.
        
        Parameters
        ----------
        group1, group2 : List[FiberNetwork]
            Two groups to compare.
        property_name : str
            Property to compare.
        test : str
            Statistical test: 'ttest', 'mannwhitney', 'ks'.
        
        Returns
        -------
        Dict[str, float]
            Test results with 'statistic' and 'p_value'.
        """
        def get_values(networks):
            values = []
            for net in networks:
                morph = MorphologyAnalyzer(net)
                report = morph.full_report()
                
                if property_name == 'mean_length':
                    values.append(report.get('mean_length', 0))
                elif property_name == 'num_fibers':
                    values.append(net.num_fibers)
                elif property_name == 'num_crosslinks':
                    values.append(net.num_crosslinks)
                elif property_name == 'mean_tortuosity':
                    values.append(report.get('mean_tortuosity', 1.0))
                elif property_name == 'nematic_order':
                    values.append(report.get('nematic_order', 0))
            
            return np.array(values)
        
        values1 = get_values(group1)
        values2 = get_values(group2)
        
        if len(values1) < 2 or len(values2) < 2:
            return {'statistic': 0.0, 'p_value': 1.0}
        
        if test == 'ttest':
            stat, pval = stats.ttest_ind(values1, values2)
        elif test == 'mannwhitney':
            stat, pval = stats.mannwhitneyu(values1, values2, alternative='two-sided')
        elif test == 'ks':
            stat, pval = stats.ks_2samp(values1, values2)
        else:
            raise ValueError(f"Unknown test: {test}")
        
        return {
            'statistic': float(stat),
            'p_value': float(pval),
            'mean1': float(np.mean(values1)),
            'mean2': float(np.mean(values2)),
            'std1': float(np.std(values1)),
            'std2': float(np.std(values2)),
        }
    
    def correlation_analysis(
        self,
        properties: List[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Compute correlations between network properties.
        
        Parameters
        ----------
        properties : List[str], optional
            Properties to correlate. Defaults to all.
        
        Returns
        -------
        Dict[str, Dict[str, float]]
            Correlation matrix with p-values.
        """
        if properties is None:
            properties = ['mean_length', 'num_fibers', 'num_crosslinks',
                         'mean_tortuosity', 'nematic_order']
        
        # Collect values
        values = {}
        for prop in properties:
            if prop == 'mean_length':
                values[prop] = self.lengths
            elif prop == 'num_fibers':
                values[prop] = self.num_fibers
            elif prop == 'num_crosslinks':
                values[prop] = self.num_crosslinks
            elif prop == 'mean_tortuosity':
                values[prop] = self.tortuosities
            elif prop == 'nematic_order':
                values[prop] = self.nematic_orders
        
        # Compute correlations
        correlations = {}
        
        for prop1 in properties:
            correlations[prop1] = {}
            for prop2 in properties:
                if prop1 == prop2:
                    correlations[prop1][prop2] = {'r': 1.0, 'p': 0.0}
                else:
                    r, p = stats.pearsonr(values[prop1], values[prop2])
                    correlations[prop1][prop2] = {'r': float(r), 'p': float(p)}
        
        return correlations
    
    def fit_distribution(
        self,
        property_name: str = 'mean_length',
        distributions: List[str] = None,
    ) -> Dict[str, Dict]:
        """Fit statistical distributions to property data.
        
        Parameters
        ----------
        property_name : str
            Property to fit.
        distributions : List[str], optional
            Distributions to try. Defaults to common ones.
        
        Returns
        -------
        Dict[str, Dict]
            Fit results for each distribution.
        """
        if distributions is None:
            distributions = ['norm', 'lognorm', 'expon', 'gamma', 'weibull_min']
        
        # Get values
        if property_name == 'mean_length':
            values = np.array(self.lengths)
        elif property_name == 'num_fibers':
            values = np.array(self.num_fibers, dtype=float)
        elif property_name == 'num_crosslinks':
            values = np.array(self.num_crosslinks, dtype=float)
        elif property_name == 'mean_tortuosity':
            values = np.array(self.tortuosities)
        elif property_name == 'nematic_order':
            values = np.array(self.nematic_orders)
        else:
            raise ValueError(f"Unknown property: {property_name}")
        
        if len(values) < 3:
            return {}
        
        # Fit each distribution
        results = {}
        
        for dist_name in distributions:
            try:
                dist = getattr(stats, dist_name)
                
                # Fit parameters
                params = dist.fit(values)
                
                # Compute goodness of fit
                ks_stat, ks_p = stats.kstest(values, dist_name, args=params)
                
                # AIC and BIC
                log_likelihood = np.sum(dist.logpdf(values, *params))
                k = len(params)
                n = len(values)
                aic = 2 * k - 2 * log_likelihood
                bic = k * np.log(n) - 2 * log_likelihood
                
                results[dist_name] = {
                    'params': params,
                    'ks_statistic': float(ks_stat),
                    'ks_p_value': float(ks_p),
                    'aic': float(aic),
                    'bic': float(bic),
                    'log_likelihood': float(log_likelihood),
                }
                
            except Exception:
                continue
        
        return results
    
    def summary_report(self) -> Dict:
        """Generate comprehensive statistical summary.
        
        Returns
        -------
        Dict
            Summary statistics for all properties.
        """
        report = {
            'num_samples': self.num_samples,
            'properties': {}
        }
        
        properties = ['mean_length', 'num_fibers', 'num_crosslinks',
                     'mean_tortuosity', 'nematic_order']
        
        for prop in properties:
            if prop == 'mean_length':
                values = np.array(self.lengths)
            elif prop == 'num_fibers':
                values = np.array(self.num_fibers, dtype=float)
            elif prop == 'num_crosslinks':
                values = np.array(self.num_crosslinks, dtype=float)
            elif prop == 'mean_tortuosity':
                values = np.array(self.tortuosities)
            elif prop == 'nematic_order':
                values = np.array(self.nematic_orders)
            
            if len(values) > 0:
                report['properties'][prop] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                    'q25': float(np.percentile(values, 25)),
                    'q75': float(np.percentile(values, 75)),
                }
        
        return report
