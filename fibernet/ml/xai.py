"""
Explainable AI (XAI) for FiberNet ML Models.

Provides model interpretation tools:
- Permutation feature importance
- SHAP values (KernelExplainer and TreeExplainer)
- Partial dependence plots
- Feature interaction detection
- Model comparison explainability

Examples
--------
>>> from fibernet.ml.xai import explain_model, permutation_importance
>>> # sklearn model
>>> importance = permutation_importance(model, X_test, y_test)
>>> shap_values = explain_model(model, X_test, method="shap")

>>> # PyTorch model
>>> from fibernet.ml.xai import explain_torch_model
>>> importance = explain_torch_model(model, X_test, method="gradient")
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ======================================================================
# Permutation Importance (works with any model)
# ======================================================================

def permutation_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_repeats: int = 10,
    random_state: int = 42,
    scoring: Optional[Callable] = None,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute permutation feature importance.

    Works with any model that has a predict() method.

    Parameters
    ----------
    model : object
        Fitted model with predict().
    X : (n_samples, n_features)
        Test features.
    y : (n_samples,)
        True targets.
    n_repeats : int
        Number of permutations per feature.
    random_state : int
        Random seed.
    scoring : callable, optional
        Scoring function(y_true, y_pred) -> float. Default: R².
    feature_names : list of str, optional
        Feature names.

    Returns
    -------
    dict
        Keys: importances_mean, importances_std, importances (per repeat),
        feature_names, sorted_indices.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.RandomState(random_state)

    if scoring is None:
        def scoring(yt, yp):
            ss_res = np.sum((yt - yp) ** 2)
            ss_tot = np.sum((yt - yt.mean()) ** 2)
            return 1 - ss_res / max(ss_tot, 1e-12)

    # Baseline score
    y_pred = model.predict(X) if hasattr(model, "predict") else model(X)
    if hasattr(y_pred, "detach"):
        y_pred = y_pred.detach().numpy()
    baseline_score = scoring(y, y_pred.flatten())

    n_features = X.shape[1]
    importances = np.zeros((n_features, n_repeats))

    for f in range(n_features):
        scores = []
        for r in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, f] = rng.permutation(X_perm[:, f])
            y_pred_perm = model.predict(X_perm) if hasattr(model, "predict") else model(X_perm)
            if hasattr(y_pred_perm, "detach"):
                y_pred_perm = y_pred_perm.detach().numpy()
            score = scoring(y, y_pred_perm.flatten())
            scores.append(baseline_score - score)
        importances[f] = scores

    mean_imp = importances.mean(axis=1)
    std_imp = importances.std(axis=1)
    sorted_idx = np.argsort(mean_imp)[::-1]

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    return {
        "importances_mean": mean_imp,
        "importances_std": std_imp,
        "importances": importances,
        "feature_names": feature_names,
        "sorted_indices": sorted_idx,
        "baseline_score": baseline_score,
    }


# ======================================================================
# SHAP Values
# ======================================================================

def shap_explanation(
    model: Any,
    X: np.ndarray,
    *,
    method: str = "auto",
    n_background: int = 50,
    feature_names: Optional[List[str]] = None,
    max_samples: int = 100,
) -> Dict[str, Any]:
    """Compute SHAP values for model explanation.

    Parameters
    ----------
    model : object
        Fitted model (sklearn or PyTorch).
    X : (n_samples, n_features)
        Data to explain.
    method : str
        "auto", "tree" (for tree models), "kernel" (model-agnostic),
        "linear" (for linear models), "deep" (for PyTorch).
    n_background : int
        Number of background samples for KernelExplainer.
    feature_names : list of str, optional
    max_samples : int
        Max samples to explain (for speed).

    Returns
    -------
    dict
        Keys: shap_values, feature_names, base_value, mean_abs_shap,
        sorted_features.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP required: pip install shap")

    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(X.shape[0], max_samples, replace=False)
        X = X[idx]

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # Auto-detect method
    model_class = type(model).__name__
    if method == "auto":
        if any(name in model_class for name in ["Forest", "Tree", "Boost", "Gradient"]):
            method = "tree"
        elif any(name in model_class for name in ["Ridge", "Lasso", "Linear", "Pipeline"]):
            method = "linear"
        else:
            method = "kernel"

    # Background data
    if X.shape[0] > n_background:
        rng = np.random.RandomState(42)
        bg_idx = rng.choice(X.shape[0], n_background, replace=False)
        background = X[bg_idx]
    else:
        background = X

    try:
        if method == "tree":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            base_value = float(np.mean(explainer.expected_value))

        elif method == "kernel":
            predict_fn = model.predict if hasattr(model, "predict") else model
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X, nsamples=min(100, X.shape[0]))
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            base_value = float(np.mean(explainer.expected_value))

        elif method == "linear":
            explainer = shap.LinearExplainer(model, background)
            shap_values = explainer.shap_values(X)
            base_value = float(np.mean(explainer.expected_value))

        else:
            predict_fn = model.predict if hasattr(model, "predict") else model
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X, nsamples=min(100, X.shape[0]))
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            base_value = float(np.mean(explainer.expected_value))

    except Exception as e:
        # Fallback to permutation importance
        return {
            "error": str(e),
            "fallback": "permutation",
            "shap_values": None,
        }

    shap_values = np.asarray(shap_values)
    mean_abs = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1]

    return {
        "shap_values": shap_values,
        "feature_names": feature_names,
        "base_value": base_value,
        "mean_abs_shap": mean_abs,
        "sorted_features": [feature_names[i] for i in sorted_idx],
        "sorted_indices": sorted_idx,
    }


# ======================================================================
# PyTorch Model Explanation
# ======================================================================

def explain_torch_model(
    model,
    X: np.ndarray,
    *,
    method: str = "gradient",
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Explain a PyTorch model using gradient-based methods.

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model.
    X : (n_samples, n_features)
        Data to explain.
    method : str
        "gradient": Input × Gradient
        "integrated_gradients": Integrated gradients (approximation)
        "saliency": Raw gradient saliency
    feature_names : list of str, optional

    Returns
    -------
    dict
        Keys: importance, feature_names, method
    """
    import torch

    X_t = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    model.eval()

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    if method == "gradient":
        output = model(X_t)
        if isinstance(output, dict):
            output = list(output.values())[0]
        if output.dim() > 1:
            output = output.sum(dim=-1)
        output.sum().backward()
        importance = (X_t * X_t.grad).abs().mean(dim=0).detach().numpy()

    elif method == "saliency":
        output = model(X_t)
        if isinstance(output, dict):
            output = list(output.values())[0]
        if output.dim() > 1:
            output = output.sum(dim=-1)
        output.sum().backward()
        importance = X_t.grad.abs().mean(dim=0).detach().numpy()

    elif method == "integrated_gradients":
        # Approximate integrated gradients with 10 steps
        n_steps = 10
        baseline = torch.zeros_like(X_t)
        total_grad = torch.zeros_like(X_t)

        for alpha in np.linspace(0, 1, n_steps):
            interp = baseline + alpha * (X_t.detach() - baseline)
            interp.requires_grad_(True)
            out = model(interp)
            if isinstance(out, dict):
                out = list(out.values())[0]
            if out.dim() > 1:
                out = out.sum(dim=-1)
            out.sum().backward()
            total_grad += interp.grad.detach()

        total_grad /= n_steps
        importance = ((X_t.detach() - baseline) * total_grad).abs().mean(dim=0).numpy()

    else:
        raise ValueError(f"Unknown method: {method}. Use 'gradient', 'saliency', or 'integrated_gradients'")

    sorted_idx = np.argsort(importance)[::-1]

    return {
        "importance": importance,
        "feature_names": feature_names,
        "sorted_features": [feature_names[i] for i in sorted_idx],
        "sorted_indices": sorted_idx,
        "method": method,
    }


# ======================================================================
# Partial Dependence
# ======================================================================

def partial_dependence(
    model: Any,
    X: np.ndarray,
    feature_idx: int,
    *,
    n_grid: int = 20,
    feature_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, np.ndarray]:
    """Compute partial dependence for a single feature.

    Parameters
    ----------
    model : object
        Fitted model with predict().
    X : (n_samples, n_features)
    feature_idx : int
        Feature index to compute PD for.
    n_grid : int
        Number of grid points.
    feature_range : (min, max), optional
        Range for feature values.

    Returns
    -------
    dict
        Keys: grid_values, pd_values
    """
    X = np.asarray(X)

    if feature_range is None:
        feature_range = (X[:, feature_idx].min(), X[:, feature_idx].max())

    grid = np.linspace(feature_range[0], feature_range[1], n_grid)
    pd_values = []

    for val in grid:
        X_mod = X.copy()
        X_mod[:, feature_idx] = val
        preds = model.predict(X_mod) if hasattr(model, "predict") else model(X_mod)
        if hasattr(preds, "detach"):
            preds = preds.detach().numpy()
        pd_values.append(float(np.mean(preds)))

    return {
        "grid_values": grid,
        "pd_values": np.array(pd_values),
    }


# ======================================================================
# Feature Interaction Detection
# ======================================================================

def detect_interactions(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    top_k: int = 10,
    feature_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Detect feature interactions via H-statistic approximation.

    Uses a simplified approach: measures how much pairwise feature
    combinations improve prediction beyond individual contributions.

    Parameters
    ----------
    model : object
        Fitted model.
    X, y : data
    top_k : int
        Return top K interactions.
    feature_names : list of str, optional

    Returns
    -------
    list of dict
        Each: {"feature_i", "feature_j", "interaction_strength"}
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n_features = X.shape[1]

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    predict = model.predict if hasattr(model, "predict") else model

    # Baseline prediction
    y_pred = predict(X)
    if hasattr(y_pred, "detach"):
        y_pred = y_pred.detach().numpy()
    y_pred = y_pred.flatten()

    # Individual feature contributions (via permutation)
    single_effects = {}
    rng = np.random.RandomState(42)
    for f in range(n_features):
        X_perm = X.copy()
        X_perm[:, f] = rng.permutation(X_perm[:, f])
        y_perm = predict(X_perm)
        if hasattr(y_perm, "detach"):
            y_perm = y_perm.detach().numpy()
        single_effects[f] = float(np.mean((y_pred - y_perm.flatten()) ** 2))

    # Pairwise interactions
    interactions = []
    for i in range(min(n_features, 15)):
        for j in range(i + 1, min(n_features, 15)):
            X_perm = X.copy()
            X_perm[:, i] = rng.permutation(X_perm[:, i])
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            y_perm = predict(X_perm)
            if hasattr(y_perm, "detach"):
                y_perm = y_perm.detach().numpy()

            joint_effect = float(np.mean((y_pred - y_perm.flatten()) ** 2))
            expected = single_effects.get(i, 0) + single_effects.get(j, 0)
            interaction = max(0, joint_effect - expected)

            if interaction > 1e-8:
                interactions.append({
                    "feature_i": feature_names[i],
                    "feature_j": feature_names[j],
                    "interaction_strength": interaction,
                    "joint_effect": joint_effect,
                })

    interactions.sort(key=lambda x: x["interaction_strength"], reverse=True)
    return interactions[:top_k]


# ======================================================================
# Unified Explanation API
# ======================================================================

def explain_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    methods: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None,
    n_background: int = 50,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run multiple explanation methods on a model.

    Parameters
    ----------
    model : fitted model (sklearn or PyTorch)
    X, y : test data
    methods : list of str, optional
        Default: ["permutation", "shap"]. Options: "permutation", "shap",
        "gradient", "saliency", "interactions", "partial_dependence"
    feature_names : list of str
    n_background : int
    verbose : bool

    Returns
    -------
    dict
        Keys per method used.
    """
    if methods is None:
        methods = ["permutation", "shap"]

    results = {}

    for method in methods:
        if verbose:
            print(f"Running {method}...")

        try:
            if method == "permutation":
                results["permutation"] = permutation_importance(
                    model, X, y, feature_names=feature_names,
                )
            elif method == "shap":
                results["shap"] = shap_explanation(
                    model, X, feature_names=feature_names,
                    n_background=n_background,
                )
            elif method in ("gradient", "saliency"):
                results[method] = explain_torch_model(
                    model, X, method=method, feature_names=feature_names,
                )
            elif method == "interactions":
                results["interactions"] = detect_interactions(
                    model, X, y, feature_names=feature_names,
                )
            elif method == "partial_dependence":
                # Run PD for top 5 features
                perm = permutation_importance(model, X, y, feature_names=feature_names)
                pd_results = {}
                for idx in perm["sorted_indices"][:5]:
                    pd_results[feature_names[idx]] = partial_dependence(
                        model, X, idx, feature_names=feature_names,
                    )
                results["partial_dependence"] = pd_results
        except Exception as e:
            results[method] = {"error": str(e)}
            if verbose:
                print(f"  Warning: {method} failed: {e}")

    return results
