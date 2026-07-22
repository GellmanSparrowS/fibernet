"""
Dataset Generation and Management for ML Training

Provides utilities for generating training datasets and managing fiber network datasets.
"""

import numpy as np
from typing import List, Dict, Optional, Callable, Tuple
from pathlib import Path
import json
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

from fibernet.core.network import FiberNetwork
from fibernet import gen
from fibernet.ml.features import FeatureExtractor


def generate_dataset(
    num_samples: int = 100,
    generator_func: Optional[Callable] = None,
    property_func: Optional[Callable] = None,
    param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    seed: int = 42,
) -> Tuple[List[FiberNetwork], np.ndarray, Dict[str, np.ndarray]]:
    """Generate a dataset of fiber networks with properties.
    
    Parameters
    ----------
    num_samples : int
        Number of samples to generate
    generator_func : Callable, optional
        Function that generates a network. If None, uses random_straight_2d
    property_func : Callable, optional
        Function that computes property from network. If None, uses random values
    param_ranges : Dict[str, Tuple[float, float]], optional
        Parameter ranges for random sampling
    seed : int
        Random seed
        
    Returns
    -------
    Tuple[List[FiberNetwork], np.ndarray, Dict[str, np.ndarray]]
        (networks, properties, parameters)
    """
    np.random.seed(seed)
    
    networks = []
    properties = []
    parameters = {
        'num_fibers': [],
        'fiber_length': [],
        'radius': [],
    }
    
    if generator_func is None:
        def generator_func(**kwargs):
            return gen.random_straight_2d(
                num_fibers=kwargs.get('num_fibers', 50),
                fiber_length=kwargs.get('fiber_length', 10.0),
                box_size=(50, 50),
                radius=kwargs.get('radius', 0.1),
                seed=kwargs.get('seed', None),
            )
    
    if param_ranges is None:
        param_ranges = {
            'num_fibers': (10, 100),
            'fiber_length': (5.0, 20.0),
            'radius': (0.05, 0.3),
        }
    
    print(f"Generating {num_samples} fiber networks...")
    
    for i in tqdm(range(num_samples), desc="Generating"):
        # Sample parameters
        params = {}
        for key, (low, high) in param_ranges.items():
            if key == 'num_fibers':
                params[key] = int(np.random.uniform(low, high))
            else:
                params[key] = np.random.uniform(low, high)
        
        params['seed'] = seed + i
        
        # Generate network
        try:
            net = generator_func(**params)
            networks.append(net)
            
            # Store parameters
            for key in parameters:
                parameters[key].append(params.get(key, 0))
            
            # Compute property
            if property_func is not None:
                prop = property_func(net)
            else:
                # Default: use volume fraction as property
                from fibernet.analysis import MorphologyAnalyzer
                morph = MorphologyAnalyzer(net)
                report = morph.full_report()
                prop = report.get('volume_fraction', 0.0)
            
            properties.append(prop)
            
        except Exception as e:
            print(f"  Warning: Sample {i} failed: {e}")
            continue
    
    properties = np.array(properties)
    parameters = {k: np.array(v) for k, v in parameters.items()}
    
    print(f"Generated {len(networks)} valid samples")
    
    return networks, properties, parameters


def load_dataset(
    filepath: str,
) -> Tuple[List[FiberNetwork], np.ndarray, Dict[str, np.ndarray]]:
    """Load a dataset from JSON files.
    
    Parameters
    ----------
    filepath : str
        Path to dataset directory
        
    Returns
    -------
    Tuple[List[FiberNetwork], np.ndarray, Dict[str, np.ndarray]]
        (networks, properties, parameters)
    """
    path = Path(filepath)
    
    # Load metadata
    with open(path / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    networks = []
    properties = []
    parameters = {}
    
    print(f"Loading {metadata['num_samples']} samples...")
    
    for i in tqdm(range(metadata['num_samples']), desc="Loading"):
        # Load network
        net_path = path / f'network_{i:04d}.json'
        if net_path.exists():
            net = FiberNetwork.load_json(str(net_path))
            networks.append(net)
            
            # Load property
            prop_path = path / f'property_{i:04d}.json'
            if prop_path.exists():
                with open(prop_path, 'r') as f:
                    prop_data = json.load(f)
                    properties.append(prop_data['value'])
            
            # Load parameters
            param_path = path / f'params_{i:04d}.json'
            if param_path.exists():
                with open(param_path, 'r') as f:
                    param_data = json.load(f)
                    for key, val in param_data.items():
                        if key not in parameters:
                            parameters[key] = []
                        parameters[key].append(val)
    
    properties = np.array(properties)
    parameters = {k: np.array(v) for k, v in parameters.items()}
    
    print(f"Loaded {len(networks)} samples")
    
    return networks, properties, parameters


def save_dataset(
    networks: List[FiberNetwork],
    properties: np.ndarray,
    parameters: Dict[str, np.ndarray],
    filepath: str,
):
    """Save a dataset to JSON files.
    
    Parameters
    ----------
    networks : List[FiberNetwork]
        List of fiber networks
    properties : np.ndarray
        Property values
    parameters : Dict[str, np.ndarray]
        Parameter arrays
    filepath : str
        Output directory path
    """
    path = Path(filepath)
    path.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata = {
        'num_samples': len(networks),
        'property_name': 'modulus',
    }
    with open(path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saving {len(networks)} samples...")
    
    for i in tqdm(range(len(networks)), desc="Saving"):
        # Save network
        net_path = path / f'network_{i:04d}.json'
        networks[i].save_json(str(net_path))
        
        # Save property
        prop_path = path / f'property_{i:04d}.json'
        with open(prop_path, 'w') as f:
            json.dump({'value': float(properties[i])}, f)
        
        # Save parameters
        param_path = path / f'params_{i:04d}.json'
        param_data = {k: float(v[i]) for k, v in parameters.items()}
        with open(param_path, 'w') as f:
            json.dump(param_data, f)
    
    print(f"Saved {len(networks)} samples to {filepath}")


def extract_features_dataset(
    networks: List[FiberNetwork],
    properties: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract features from a dataset of networks.
    
    Parameters
    ----------
    networks : List[FiberNetwork]
        List of fiber networks
    properties : np.ndarray
        Target property values
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[str]]
        (X, y, feature_names)
    """
    extractor = FeatureExtractor()
    
    print(f"Extracting features from {len(networks)} networks...")
    
    X = []
    for i, net in enumerate(tqdm(networks, desc="Extracting")):
        features = extractor.extract_to_array(net)
        X.append(features)
    
    X = np.array(X)
    y = np.array(properties)
    feature_names = extractor.get_feature_names()
    
    return X, y, feature_names
