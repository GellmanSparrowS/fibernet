"""
Property Prediction with Machine Learning

Provides ML models for predicting fiber network properties from structure.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import pickle

from fibernet.core.network import FiberNetwork
from fibernet.ml.features import FeatureExtractor


class PropertyPredictor:
    """ML-based predictor for fiber network properties.
    
    Parameters
    ----------
    model_type : str
        Type of sklearn model ('random_forest', 'gradient_boosting', 'neural_net')
    property_name : str
        Name of the property being predicted
    """
    
    def __init__(
        self,
        model_type: str = 'random_forest',
        property_name: str = 'modulus',
    ):
        self.model_type = model_type
        self.property_name = property_name
        self.model = None
        self.feature_extractor = FeatureExtractor()
        self.feature_names = self.feature_extractor.get_feature_names()
        self.scaler = None
    
    def _create_model(self):
        """Create sklearn model based on model_type."""
        try:
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.neural_network import MLPRegressor
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError(
                "scikit-learn required for ML features. "
                "Install with: pip install scikit-learn"
            )
        
        if self.model_type == 'random_forest':
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif self.model_type == 'neural_net':
            return MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def fit(
        self,
        networks: List[FiberNetwork],
        properties: np.ndarray,
    ) -> 'PropertyPredictor':
        """Train the predictor on a dataset.
        
        Parameters
        ----------
        networks : List[FiberNetwork]
            List of fiber networks
        properties : np.ndarray
            Target property values for each network
            
        Returns
        -------
        PropertyPredictor
            Self for method chaining
        """
        from sklearn.preprocessing import StandardScaler
        
        # Extract features
        print(f"Extracting features from {len(networks)} networks...")
        X = []
        for i, net in enumerate(networks):
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(networks)}")
            features = self.feature_extractor.extract_to_array(net)
            X.append(features)
        
        X = np.array(X)
        y = np.array(properties)
        
        # Scale features
        print("Scaling features...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        print(f"Training {self.model_type} model...")
        self.model = self._create_model()
        self.model.fit(X_scaled, y)
        
        # Compute training score
        train_score = self.model.score(X_scaled, y)
        print(f"Training R² score: {train_score:.4f}")
        
        return self
    
    def predict(self, network: FiberNetwork) -> float:
        """Predict property for a single network.
        
        Parameters
        ----------
        network : FiberNetwork
            Input fiber network
            
        Returns
        -------
        float
            Predicted property value
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        features = self.feature_extractor.extract_to_array(network)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        return float(self.model.predict(features_scaled)[0])
    
    def predict_batch(self, networks: List[FiberNetwork]) -> np.ndarray:
        """Predict properties for multiple networks.
        
        Parameters
        ----------
        networks : List[FiberNetwork]
            List of fiber networks
            
        Returns
        -------
        np.ndarray
            Predicted property values
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        X = []
        for net in networks:
            features = self.feature_extractor.extract_to_array(net)
            X.append(features)
        
        X = np.array(X)
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)
    
    def feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores.
        
        Returns
        -------
        Dict[str, float]
            Feature names mapped to importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_)
        else:
            return {}
        
        return dict(zip(self.feature_names, importances))
    
    def save(self, filepath: str):
        """Save predictor to file.
        
        Parameters
        ----------
        filepath : str
            Output file path (.pkl)
        """
        data = {
            'model_type': self.model_type,
            'property_name': self.property_name,
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'PropertyPredictor':
        """Load predictor from file.
        
        Parameters
        ----------
        filepath : str
            Input file path (.pkl)
            
        Returns
        -------
        PropertyPredictor
            Loaded predictor
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        predictor = cls(
            model_type=data['model_type'],
            property_name=data['property_name'],
        )
        predictor.model = data['model']
        predictor.scaler = data['scaler']
        predictor.feature_names = data['feature_names']
        
        return predictor


def train_predictor(
    networks: List[FiberNetwork],
    properties: np.ndarray,
    property_name: str = 'modulus',
    model_type: str = 'random_forest',
) -> PropertyPredictor:
    """Convenience function to train a property predictor.
    
    Parameters
    ----------
    networks : List[FiberNetwork]
        Training networks
    properties : np.ndarray
        Target property values
    property_name : str
        Name of property being predicted
    model_type : str
        Type of ML model
        
    Returns
    -------
    PropertyPredictor
        Trained predictor
    """
    predictor = PropertyPredictor(
        model_type=model_type,
        property_name=property_name,
    )
    predictor.fit(networks, properties)
    return predictor
