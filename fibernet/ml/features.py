"""
Feature Extraction for Fiber Networks

Converts fiber network structures into feature vectors suitable for ML models.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from fibernet.core.network import FiberNetwork
from fibernet.analysis import MorphologyAnalyzer, TopologyAnalyzer
from fibernet.analysis.advanced import SpectralAnalyzer, PoreAnalyzer, AnisotropyAnalyzer


class FeatureExtractor:
    """Extract comprehensive feature vectors from fiber networks.
    
    Parameters
    ----------
    include_morphology : bool
        Include morphological features (length, orientation, tortuosity)
    include_topology : bool
        Include topological features (degree, connectivity, components)
    include_spectral : bool
        Include spectral features (eigenvalues, spectral gap)
    include_pore : bool
        Include pore structure features (pore size distribution)
    include_anisotropy : bool
        Include anisotropy features (orientation tensors)
    """
    
    def __init__(
        self,
        include_morphology: bool = True,
        include_topology: bool = True,
        include_spectral: bool = True,
        include_pore: bool = True,
        include_anisotropy: bool = True,
    ):
        self.include_morphology = include_morphology
        self.include_topology = include_topology
        self.include_spectral = include_spectral
        self.include_pore = include_pore
        self.include_anisotropy = include_anisotropy
    
    def extract(self, network: FiberNetwork) -> Dict[str, float]:
        """Extract all features from a fiber network.
        
        Parameters
        ----------
        network : FiberNetwork
            Input fiber network
            
        Returns
        -------
        Dict[str, float]
            Feature dictionary with string keys and numeric values
        """
        features = {}
        
        # Basic network properties
        features['num_fibers'] = network.num_fibers
        features['num_crosslinks'] = network.num_crosslinks
        features['dimension'] = network.dimension
        
        # Morphological features
        if self.include_morphology:
            try:
                morph = MorphologyAnalyzer(network)
                report = morph.full_report()
                features['mean_length'] = report.get('mean_length', 0.0)
                features['std_length'] = report.get('std_length', 0.0)
                features['mean_radius'] = report.get('mean_radius', 0.0)
                features['std_radius'] = report.get('std_radius', 0.0)
                features['nematic_order'] = report.get('nematic_order', 0.0)
                features['mean_tortuosity'] = report.get('mean_tortuosity', 1.0)
                features['total_length'] = report.get('total_length', 0.0)
                features['volume_fraction'] = report.get('volume_fraction', 0.0)
            except Exception:
                # Fill with defaults if analysis fails
                features.update({
                    'mean_length': 0.0,
                    'std_length': 0.0,
                    'mean_radius': 0.0,
                    'std_radius': 0.0,
                    'nematic_order': 0.0,
                    'mean_tortuosity': 1.0,
                    'total_length': 0.0,
                    'volume_fraction': 0.0,
                })
        
        # Topological features
        if self.include_topology:
            try:
                topo = TopologyAnalyzer(network)
                report = topo.full_report()
                features['num_nodes'] = report.get('num_nodes', 0)
                features['num_edges'] = report.get('num_edges', 0)
                features['mean_degree'] = report.get('degree_stats', {}).get('mean', 0.0)
                features['std_degree'] = report.get('degree_stats', {}).get('std', 0.0)
                features['max_degree'] = report.get('degree_stats', {}).get('max', 0)
                features['min_degree'] = report.get('degree_stats', {}).get('min', 0)
                features['is_connected'] = 1.0 if report.get('is_connected', False) else 0.0
                features['num_components'] = report.get('num_components', 1)
            except Exception:
                features.update({
                    'num_nodes': 0,
                    'num_edges': 0,
                    'mean_degree': 0.0,
                    'std_degree': 0.0,
                    'max_degree': 0,
                    'min_degree': 0,
                    'is_connected': 0.0,
                    'num_components': 1,
                })
        
        # Spectral features
        if self.include_spectral:
            try:
                spectral = SpectralAnalyzer(network)
                features['spectral_gap'] = spectral.spectral_gap()
                features['spectral_entropy'] = spectral.spectral_entropy()
                # Add first few eigenvalues
                eigenvalues = spectral.eigenvalues()
                for i, eig in enumerate(eigenvalues[:5]):
                    features[f'eigenvalue_{i}'] = eig
            except Exception:
                features.update({
                    'spectral_gap': 0.0,
                    'spectral_entropy': 0.0,
                    'eigenvalue_0': 0.0,
                    'eigenvalue_1': 0.0,
                    'eigenvalue_2': 0.0,
                    'eigenvalue_3': 0.0,
                    'eigenvalue_4': 0.0,
                })
        
        # Pore structure features
        if self.include_pore:
            try:
                pore = PoreAnalyzer(network)
                stats = pore.pore_size_statistics()
                features['mean_pore_size'] = stats.get('mean', 0.0)
                features['std_pore_size'] = stats.get('std', 0.0)
                features['median_pore_size'] = stats.get('median', 0.0)
                features['p95_pore_size'] = stats.get('p95', 0.0)
                features['pore_size_range'] = stats.get('max', 0.0) - stats.get('min', 0.0)
            except Exception:
                features.update({
                    'mean_pore_size': 0.0,
                    'std_pore_size': 0.0,
                    'median_pore_size': 0.0,
                    'p95_pore_size': 0.0,
                    'pore_size_range': 0.0,
                })
        
        # Anisotropy features
        if self.include_anisotropy:
            try:
                aniso = AnisotropyAnalyzer(network)
                features['anisotropy_index'] = aniso.anisotropy_index()
                # Add orientation tensor components
                tensor = aniso.orientation_tensor()
                features['orientation_xx'] = tensor[0, 0]
                features['orientation_yy'] = tensor[1, 1]
                features['orientation_zz'] = tensor[2, 2]
                features['orientation_xy'] = tensor[0, 1]
                features['orientation_xz'] = tensor[0, 2]
                features['orientation_yz'] = tensor[1, 2]
            except Exception:
                features.update({
                    'anisotropy_index': 0.0,
                    'orientation_xx': 0.0,
                    'orientation_yy': 0.0,
                    'orientation_zz': 0.0,
                    'orientation_xy': 0.0,
                    'orientation_xz': 0.0,
                    'orientation_yz': 0.0,
                })
        
        return features
    
    def extract_to_array(self, network: FiberNetwork) -> np.ndarray:
        """Extract features as a numpy array.
        
        Parameters
        ----------
        network : FiberNetwork
            Input fiber network
            
        Returns
        -------
        np.ndarray
            1D array of feature values
        """
        features = self.extract(network)
        return np.array(list(features.values()))
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names.
        
        Returns
        -------
        List[str]
            Feature names in consistent order
        """
        # Create a dummy network to get feature keys
        from fibernet import gen
        dummy = gen.random_straight_2d(num_fibers=5, fiber_length=5, box_size=(20, 20), seed=42)
        features = self.extract(dummy)
        return list(features.keys())


def extract_features(
    network: FiberNetwork,
    as_array: bool = False,
) -> Any:
    """Convenience function to extract features from a network.
    
    Parameters
    ----------
    network : FiberNetwork
        Input fiber network
    as_array : bool
        If True, return numpy array; otherwise return dict
        
    Returns
    -------
    Dict[str, float] or np.ndarray
        Extracted features
    """
    extractor = FeatureExtractor()
    if as_array:
        return extractor.extract_to_array(network)
    else:
        return extractor.extract(network)
