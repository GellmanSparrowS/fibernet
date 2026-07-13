"""
ML Utilities for FiberNet — One-line ML workflows.

Provides convenience functions for training, evaluating, and visualizing
ML models on fiber network data.

Examples
--------
>>> from fibernet.ml.utils import train_predictor, cross_validate, plot_predictions
>>> model, metrics = train_predictor(X, y, model_type="rf")
>>> cv_results = cross_validate(X, y, model_type="rf", cv=5)
>>> plot_predictions(y_test, y_pred, save_path="pred.png")
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)


# ======================================================================
# Model Training
# ======================================================================

def train_predictor(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model_type: str = "rf",
    test_size: float = 0.2,
    random_state: int = 42,
    feature_names: Optional[List[str]] = None,
    **model_kwargs,
) -> Tuple[Any, Dict[str, float]]:
    """Train a regression model and return (model, metrics).

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix.
    y : array-like, shape (n_samples,)
        Target values.
    model_type : str
        Model type: "rf" (RandomForest), "ridge", "gb" (GradientBoosting),
        "svm" (SVR), "mlp" (MLPRegressor).
    test_size : float
        Fraction for test set.
    random_state : int
        Random seed.
    feature_names : list of str, optional
        Feature names for interpretability.
    **model_kwargs
        Extra keyword arguments passed to the model constructor.

    Returns
    -------
    model : sklearn estimator
        Trained model.
    metrics : dict
        Keys: r2, rmse, mae, test_size.

    Examples
    --------
    >>> model, metrics = train_predictor(X, y, model_type="rf")
    >>> print(f"R² = {metrics['r2']:.3f}")
    """
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
    except ImportError:
        raise ImportError("scikit-learn required: pip install scikit-learn")

    X = np.asarray(X)
    y = np.asarray(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = _build_model(model_type, **model_kwargs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = mean_absolute_error(y_test, y_pred)

    metrics = {
        "r2": r2,
        "rmse": rmse,
        "mae": float(mae),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    return model, metrics


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model_type: str = "rf",
    cv: int = 5,
    random_state: int = 42,
    **model_kwargs,
) -> Dict[str, Any]:
    """K-fold cross-validation with detailed metrics.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Target values.
    model_type : str
        Model type (same options as train_predictor).
    cv : int
        Number of folds.
    random_state : int
        Random seed.

    Returns
    -------
    dict
        Keys: mean_r2, std_r2, mean_rmse, std_rmse, fold_r2s, fold_rmses.
    """
    try:
        from sklearn.model_selection import KFold
        from sklearn.metrics import r2_score, mean_squared_error
    except ImportError:
        raise ImportError("scikit-learn required: pip install scikit-learn")

    X = np.asarray(X)
    y = np.asarray(y)

    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    fold_r2s = []
    fold_rmses = []

    for train_idx, test_idx in kf.split(X):
        model = _build_model(model_type, **model_kwargs)
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        fold_r2s.append(r2_score(y[test_idx], y_pred))
        fold_rmses.append(float(np.sqrt(mean_squared_error(y[test_idx], y_pred))))

    return {
        "mean_r2": float(np.mean(fold_r2s)),
        "std_r2": float(np.std(fold_r2s)),
        "mean_rmse": float(np.mean(fold_rmses)),
        "std_rmse": float(np.std(fold_rmses)),
        "fold_r2s": fold_r2s,
        "fold_rmses": fold_rmses,
    }


def compare_models(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model_types: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Compare multiple models on the same train/test split.

    Parameters
    ----------
    model_types : list of str, optional
        Default: ["rf", "ridge", "gb", "svm"].

    Returns
    -------
    dict
        {model_name: {r2, rmse, mae}}.
    """
    if model_types is None:
        model_types = ["rf", "ridge", "gb"]

    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        raise ImportError("scikit-learn required")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    results = {}
    for mt in model_types:
        try:
            model = _build_model(mt)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            results[mt] = {
                "r2": r2_score(y_test, y_pred),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "mae": float(mean_absolute_error(y_test, y_pred)),
            }
        except Exception as e:
            results[mt] = {"error": str(e)}

    return results


# ======================================================================
# Visualization
# ======================================================================

def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str = "Predictions vs Actual",
    save_path: Optional[str] = None,
    show: bool = False,
) -> Any:
    """Scatter plot of predictions vs actual values.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted values.
    title : str
        Plot title.
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display interactively.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#0a0a0f")

    ax.scatter(y_true, y_pred, c="#b388ff", s=30, alpha=0.7, edgecolors="none")

    # Perfect prediction line
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    margin = (lims[1] - lims[0]) * 0.05
    lims = [lims[0] - margin, lims[1] + margin]
    ax.plot(lims, lims, "--", color="#666", linewidth=1, alpha=0.5)

    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    ax.set_xlabel("Actual", color="#aaa", fontsize=12)
    ax.set_ylabel("Predicted", color="#aaa", fontsize=12)
    ax.set_title(f"{title}\nR² = {r2:.3f}", color="#ddd", fontsize=14)
    ax.tick_params(colors="#888")
    ax.spines["bottom"].set_color("#333")
    ax.spines["left"].set_color("#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    if show:
        plt.show()
    return fig


def plot_feature_importance(
    model: Any,
    feature_names: Optional[List[str]] = None,
    *,
    top_k: int = 20,
    title: str = "Feature Importance",
    save_path: Optional[str] = None,
    show: bool = False,
) -> Any:
    """Bar chart of feature importance from tree-based or linear models.

    Parameters
    ----------
    model : sklearn estimator
        Must have feature_importances_ or coef_ attribute.
    feature_names : list of str, optional
        Feature names.
    top_k : int
        Show top K features.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Extract importance
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()
        if hasattr(model, "named_steps"):
            # Pipeline — try to get the actual model
            for name, step in model.named_steps.items():
                if hasattr(step, "feature_importances_"):
                    importances = step.feature_importances_
                    break
                elif hasattr(step, "coef_"):
                    importances = np.abs(step.coef_).flatten()
                    break
    else:
        raise ValueError("Model has no feature_importances_ or coef_ attribute")

    n_features = len(importances)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]

    # Sort and take top K
    indices = np.argsort(importances)[::-1][:top_k]
    names = [feature_names[i] for i in indices]
    values = [importances[i] for i in indices]

    fig, ax = plt.subplots(figsize=(10, max(4, top_k * 0.35)))
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#0a0a0f")

    bars = ax.barh(range(len(names)), values, color="#b388ff", alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9, color="#aaa")
    ax.invert_yaxis()
    ax.set_xlabel("Importance", color="#aaa")
    ax.set_title(title, color="#ddd", fontsize=14)
    ax.tick_params(colors="#888")
    ax.spines["bottom"].set_color("#333")
    ax.spines["left"].set_color("#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    if show:
        plt.show()
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str = "Residual Analysis",
    save_path: Optional[str] = None,
    show: bool = False,
) -> Any:
    """Residual plot: residuals vs predicted values.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0a0a0f")

    for ax in [ax1, ax2]:
        ax.set_facecolor("#0a0a0f")
        ax.tick_params(colors="#888")
        ax.spines["bottom"].set_color("#333")
        ax.spines["left"].set_color("#333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Residuals vs predicted
    ax1.scatter(y_pred, residuals, c="#b388ff", s=20, alpha=0.6, edgecolors="none")
    ax1.axhline(0, color="#666", linestyle="--", linewidth=1)
    ax1.set_xlabel("Predicted", color="#aaa")
    ax1.set_ylabel("Residual", color="#aaa")
    ax1.set_title("Residuals vs Predicted", color="#ddd")

    # Residual distribution
    ax2.hist(residuals, bins=30, color="#b388ff", alpha=0.7, edgecolor="#333")
    ax2.axvline(0, color="#666", linestyle="--", linewidth=1)
    ax2.set_xlabel("Residual", color="#aaa")
    ax2.set_ylabel("Count", color="#aaa")
    ax2.set_title("Residual Distribution", color="#ddd")

    fig.suptitle(title, color="#ddd", fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    if show:
        plt.show()
    return fig


def plot_learning_curve(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model_type: str = "rf",
    train_sizes: Optional[np.ndarray] = None,
    cv: int = 5,
    title: str = "Learning Curve",
    save_path: Optional[str] = None,
    show: bool = False,
) -> Any:
    """Plot learning curve (train/test score vs training set size).

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from sklearn.model_selection import learning_curve
    except ImportError:
        raise ImportError("scikit-learn required")

    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 8)

    model = _build_model(model_type)
    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X, y, train_sizes=train_sizes, cv=cv, scoring="r2",
        n_jobs=-1
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#0a0a0f")

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color="#b388ff")
    ax.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std,
                    alpha=0.15, color="#82b1ff")
    ax.plot(train_sizes_abs, train_mean, "o-", color="#b388ff", label="Train")
    ax.plot(train_sizes_abs, test_mean, "o-", color="#82b1ff", label="Test")

    ax.set_xlabel("Training Set Size", color="#aaa")
    ax.set_ylabel("R² Score", color="#aaa")
    ax.set_title(title, color="#ddd", fontsize=14)
    ax.legend(loc="lower right", facecolor="#1a1a2e", edgecolor="#333",
              labelcolor="#aaa")
    ax.tick_params(colors="#888")
    ax.spines["bottom"].set_color("#333")
    ax.spines["left"].set_color("#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    if show:
        plt.show()
    return fig


# ======================================================================
# Pipeline: CSV → ML → Results
# ======================================================================

def predict_from_csv(
    csv_path: str,
    target: str = "max_force",
    *,
    model_type: str = "rf",
    feature_prefix: str = "feat_",
    test_size: float = 0.2,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """One-line ML: load CSV, train model, save results.

    Parameters
    ----------
    csv_path : str
        Path to CSV (output of batch_simulate or pipeline).
    target : str
        Target column name.
    model_type : str
        Model type.
    feature_prefix : str
        Columns starting with this are features.
    output_dir : str, optional
        Directory to save results. If None, returns without saving.

    Returns
    -------
    dict
        Keys: model, metrics, feature_names, cv_results.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required: pip install pandas")

    df = pd.read_csv(csv_path)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[target])

    feature_cols = [c for c in df.columns if c.startswith(feature_prefix)]
    feature_cols = [c for c in feature_cols if df[c].std() > 1e-12]

    X = df[feature_cols].fillna(0).values
    y = df[target].values

    model, metrics = train_predictor(X, y, model_type=model_type,
                                     test_size=test_size,
                                     feature_names=feature_cols)
    cv_results = cross_validate(X, y, model_type=model_type, cv=5)

    print(f"✓ {model_type.upper()}: R²={metrics['r2']:.3f}, "
          f"RMSE={metrics['rmse']:.2e}, CV-R²={cv_results['mean_r2']:.3f}±{cv_results['std_r2']:.3f}")

    result = {
        "model": model,
        "metrics": metrics,
        "cv_results": cv_results,
        "feature_names": feature_cols,
        "target": target,
        "n_samples": len(df),
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Save metrics
        import json
        with open(out / "metrics.json", "w") as f:
            json.dump({
                "model_type": model_type,
                "target": target,
                "metrics": metrics,
                "cv_results": {k: v for k, v in cv_results.items()
                               if not isinstance(v, list)},
                "n_samples": len(df),
                "n_features": len(feature_cols),
            }, f, indent=2)

        # Save model
        import pickle
        with open(out / "model.pkl", "wb") as f:
            pickle.dump({"model": model, "feature_names": feature_cols}, f)

        # Save plots
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        y_pred = model.predict(X_test)
        plot_predictions(y_test, y_pred, save_path=str(out / "predictions.png"))
        plot_feature_importance(model, feature_cols, save_path=str(out / "importance.png"))
        plot_residuals(y_test, y_pred, save_path=str(out / "residuals.png"))

        print(f"✓ Results saved to {out}")

    return result


# ======================================================================
# Internal helpers
# ======================================================================

def _build_model(model_type: str, **kwargs):
    """Build a sklearn model by name."""
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    defaults = {
        "rf": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
        "gb": {"n_estimators": 100, "random_state": 42},
    }

    if model_type == "rf":
        params = {**defaults["rf"], **kwargs}
        return RandomForestRegressor(**params)
    elif model_type == "ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=kwargs.get("alpha", 1.0)))
        ])
    elif model_type == "gb":
        params = {**defaults["gb"], **kwargs}
        return GradientBoostingRegressor(**params)
    elif model_type == "svm":
        from sklearn.svm import SVR
        return Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVR(kernel=kwargs.get("kernel", "rbf"),
                        C=kwargs.get("C", 1.0)))
        ])
    elif model_type == "mlp":
        from sklearn.neural_network import MLPRegressor
        return Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=kwargs.get("hidden", (64, 32)),
                max_iter=kwargs.get("max_iter", 500),
                random_state=42
            ))
        ])
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. "
                         f"Options: rf, ridge, gb, svm, mlp")
