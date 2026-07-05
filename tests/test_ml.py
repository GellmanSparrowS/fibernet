"""Tests for ML module."""

import numpy as np
import pytest
from fibernet import gen
from fibernet.ml.features import FeatureExtractor, extract_features
from fibernet.ml.dataset import generate_dataset

pytest.importorskip("sklearn")



class TestFeatureExtraction:
    def test_extract_basic(self):
        net = gen.random_straight_2d(num_fibers=20, fiber_length=10, box_size=(30, 30), seed=42)
        features = extract_features(net)
        
        assert isinstance(features, dict)
        assert 'num_fibers' in features
        assert 'num_crosslinks' in features
        assert features['num_fibers'] == 20
    
    def test_extract_as_array(self):
        net = gen.random_straight_2d(num_fibers=20, fiber_length=10, box_size=(30, 30), seed=42)
        features = extract_features(net, as_array=True)
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 10
    
    def test_feature_extractor(self):
        net = gen.random_straight_2d(num_fibers=20, fiber_length=10, box_size=(30, 30), seed=42)
        extractor = FeatureExtractor()
        
        features = extractor.extract(net)
        assert 'mean_length' in features
        assert 'nematic_order' in features
    
    def test_feature_names(self):
        extractor = FeatureExtractor()
        names = extractor.get_feature_names()
        
        assert isinstance(names, list)
        assert len(names) > 10
        assert 'num_fibers' in names


class TestDatasetGeneration:
    def test_generate_dataset(self):
        networks, properties, parameters = generate_dataset(
            num_samples=5,
            seed=42,
        )
        
        assert len(networks) == 5
        assert len(properties) == 5
        assert 'num_fibers' in parameters
        assert len(parameters['num_fibers']) == 5
