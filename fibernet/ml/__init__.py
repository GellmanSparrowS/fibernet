"""
Machine Learning Integration Module

Provides:
- Feature extraction from fiber networks
- Property prediction with sklearn models
- Inverse design capabilities
- Dataset generation for ML training
"""

from fibernet.ml.features import extract_features, FeatureExtractor
from fibernet.ml.predictor import PropertyPredictor, train_predictor
from fibernet.ml.dataset import generate_dataset, load_dataset

__all__ = [
    'extract_features',
    'FeatureExtractor',
    'PropertyPredictor',
    'train_predictor',
    'generate_dataset',
    'load_dataset',
]

# GNN feature extraction
from .features import GNNFeatureExtractor

# Graph Neural Network
from .graph_neural_network import (
    GraphData, NetworkGraphConverter, GNNPropertyPredictor,
    predict_property
)
__all__.extend([
    "GraphData", "NetworkGraphConverter", "GNNPropertyPredictor",
    "predict_property"
])
