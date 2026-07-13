"""
Machine learning tools for FiberNet.

- features: Feature extraction (FiberNetwork-based)
- utils: One-line ML workflows (train, CV, visualize)
"""

from fibernet.ml.utils import (
    train_predictor,
    cross_validate,
    compare_models,
    predict_from_csv,
    plot_predictions,
    plot_feature_importance,
    plot_residuals,
    plot_learning_curve,
)

__all__ = [
    "train_predictor",
    "cross_validate",
    "compare_models",
    "predict_from_csv",
    "plot_predictions",
    "plot_feature_importance",
    "plot_residuals",
    "plot_learning_curve",
]
