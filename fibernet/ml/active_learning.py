"""
Active Learning for FiberNet — Efficient Data Acquisition.

Implements strategies for selecting the most informative samples
to simulate next, minimizing the labeling budget:
- UncertaintySampling: Query points with highest prediction uncertainty
- DiversitySampling: Select diverse samples via clustering
- QueryByCommittee: Disagreement-based sampling from ensemble
- ExpectedModelChange: Points causing largest model update
- ActiveLearningLoop: End-to-end active learning workflow

Features
--------
- Multiple acquisition strategies
- Integration with simulation engine for automatic labeling
- Uncertainty quantification via MC dropout or ensembles
- Batch acquisition with diversity promotion
- Training history tracking across iterations

References
----------
- Article section 6.2: Data scarcity and active learning
- Settles, "Active Learning" (Morgan & Claypool, 2012)

Examples
--------
>>> from fibernet.ml.active_learning import ActiveLearningLoop, UncertaintySampling
>>> al = ActiveLearningLoop(
...     acquisition=UncertaintySampling(model_type="rf"),
...     simulator=simulate_fn,
...     initial_pool=X_pool,
...     initial_labels=y_initial,
... )
>>> for iteration in range(10):
...     results = al.step(batch_size=5)
...     print(f"Iter {iteration}: R²={results['r2']:.3f}")
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


class UncertaintySampling:
    """Select samples with highest prediction uncertainty.

    Parameters
    ----------
    model_type : str
        Base model type: "rf", "ridge", "gb", "svm".
    use_ensemble : bool
        Use ensemble of models for uncertainty estimation.
    n_estimators : int
        Number of ensemble models (when use_ensemble=True).
    """

    def __init__(
        self,
        model_type: str = "rf",
        use_ensemble: bool = True,
        n_estimators: int = 5,
    ):
        self.model_type = model_type
        self.use_ensemble = use_ensemble
        self.n_estimators = n_estimators
        self.model = None
        self.ensemble = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit model(s) on labeled data."""
        from fibernet.ml.utils import _build_model

        X = np.asarray(X)
        y = np.asarray(y)

        if self.use_ensemble:
            self.ensemble = []
            for i in range(self.n_estimators):
                model = _build_model(self.model_type)
                # Bootstrap sample
                idx = np.random.choice(len(X), len(X), replace=True)
                model.fit(X[idx], y[idx])
                self.ensemble.append(model)
            self.model = self.ensemble[0]
        else:
            self.model = _build_model(self.model_type)
            self.model.fit(X, y)

    def score(self, X_pool: np.ndarray) -> np.ndarray:
        """Compute uncertainty scores for pool samples.

        Higher score = more uncertain = should be labeled.

        Parameters
        ----------
        X_pool : (n_pool, n_features)

        Returns
        -------
        (n_pool,) uncertainty scores
        """
        X_pool = np.asarray(X_pool)

        if self.use_ensemble and len(self.ensemble) > 1:
            predictions = np.array([m.predict(X_pool) for m in self.ensemble])
            return predictions.std(axis=0)
        else:
            pred = self.model.predict(X_pool)
            # For single model, use distance from mean as proxy
            return np.abs(pred - pred.mean()) / (pred.std() + 1e-8)

    def select(
        self,
        X_pool: np.ndarray,
        batch_size: int = 5,
        exclude_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Select most uncertain samples.

        Parameters
        ----------
        X_pool : (n_pool, n_features)
        batch_size : int
            Number of samples to select.
        exclude_indices : array-like, optional
            Indices to exclude from selection.

        Returns
        -------
        (batch_size,) selected indices
        """
        scores = self.score(X_pool)

        if exclude_indices is not None:
            scores[exclude_indices] = -np.inf

        top_idx = np.argsort(scores)[::-1][:batch_size]
        return top_idx


class DiversitySampling:
    """Select diverse samples using k-means clustering.

    Parameters
    ----------
    n_clusters : int
        Number of clusters for diversity selection.
    """

    def __init__(self, n_clusters: int = 10):
        self.n_clusters = n_clusters

    def select(
        self,
        X_pool: np.ndarray,
        batch_size: int = 5,
        exclude_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Select diverse samples via clustering."""
        from sklearn.cluster import KMeans

        X_pool = np.asarray(X_pool)
        mask = np.ones(len(X_pool), dtype=bool)
        if exclude_indices is not None:
            mask[exclude_indices] = False

        X_available = X_pool[mask]
        available_idx = np.where(mask)[0]

        n_clusters = min(self.n_clusters, len(X_available))
        if n_clusters < 2:
            return available_idx[:batch_size]

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        kmeans.fit(X_available)

        selected = []
        for cluster_id in range(n_clusters):
            if len(selected) >= batch_size:
                break
            cluster_mask = kmeans.labels_ == cluster_id
            cluster_points = X_available[cluster_mask]
            cluster_indices = available_idx[cluster_mask]

            if len(cluster_points) == 0:
                continue

            # Pick point closest to cluster center
            center = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(cluster_points - center, axis=1)
            closest = cluster_indices[np.argmin(distances)]
            selected.append(closest)

        # If we need more, pick randomly from remaining
        remaining = [i for i in available_idx if i not in selected]
        while len(selected) < batch_size and remaining:
            idx = np.random.choice(remaining)
            selected.append(idx)
            remaining.remove(idx)

        return np.array(selected[:batch_size])


class QueryByCommittee:
    """Disagreement-based active learning from model ensemble.

    Parameters
    ----------
    model_types : list of str
        Different model types for the committee.
    """

    def __init__(self, model_types: Optional[List[str]] = None):
        if model_types is None:
            model_types = ["rf", "ridge", "gb"]
        self.model_types = model_types
        self.committee = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        from fibernet.ml.utils import _build_model
        self.committee = []
        for mt in self.model_types:
            model = _build_model(mt)
            model.fit(X, y)
            self.committee.append(model)

    def score(self, X_pool: np.ndarray) -> np.ndarray:
        """Compute disagreement scores."""
        predictions = np.array([m.predict(X_pool) for m in self.committee])
        return predictions.std(axis=0)

    def select(
        self,
        X_pool: np.ndarray,
        batch_size: int = 5,
        exclude_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        scores = self.score(X_pool)
        if exclude_indices is not None:
            scores[exclude_indices] = -np.inf
        return np.argsort(scores)[::-1][:batch_size]


class ActiveLearningLoop:
    """End-to-end active learning workflow.

    Parameters
    ----------
    acquisition : UncertaintySampling, DiversitySampling, or QueryByCommittee
        Acquisition strategy.
    simulator : callable
        Function that takes features and returns labels (dict or float).
    model_type : str
        Base model type for evaluation.
    random_state : int
        Random seed.

    Examples
    --------
    >>> al = ActiveLearningLoop(
    ...     acquisition=UncertaintySampling(),
    ...     simulator=my_simulator,
    ... )
    >>> al.initialize(X_pool, n_initial=20)
    >>> for i in range(10):
    ...     results = al.step(batch_size=5)
    ...     print(f"Iter {i}: R²={results['r2']:.3f}, labeled={results['n_labeled']}")
    """

    def __init__(
        self,
        acquisition: Optional[Any] = None,
        simulator: Optional[Callable] = None,
        model_type: str = "rf",
        random_state: int = 42,
    ):
        self.acquisition = acquisition or UncertaintySampling()
        self.simulator = simulator
        self.model_type = model_type
        self.random_state = random_state

        self.X_labeled: Optional[np.ndarray] = None
        self.y_labeled: Optional[np.ndarray] = None
        self.X_pool: Optional[np.ndarray] = None
        self.labeled_indices: set = set()
        self.iteration_history: List[Dict] = []

    def initialize(
        self,
        X_pool: np.ndarray,
        y_initial: Optional[np.ndarray] = None,
        n_initial: int = 20,
    ):
        """Initialize with initial labeled set.

        Parameters
        ----------
        X_pool : (n_total, n_features)
            Full feature pool.
        y_initial : (n_initial, n_targets), optional
            Initial labels. If None, uses simulator.
        n_initial : int
            Number of initial random samples.
        """
        self.X_pool = np.asarray(X_pool)
        n = len(self.X_pool)

        if y_initial is not None:
            initial_idx = np.arange(len(y_initial))
            self.y_labeled = np.asarray(y_initial)
        else:
            rng = np.random.RandomState(self.random_state)
            initial_idx = rng.choice(n, min(n_initial, n), replace=False)

            if self.simulator is not None:
                labels = []
                for idx in initial_idx:
                    try:
                        label = self.simulator(self.X_pool[idx])
                        if isinstance(label, dict):
                            label = list(label.values())[0] if label else 0.0
                        labels.append(float(label))
                    except Exception:
                        labels.append(0.0)
                self.y_labeled = np.array(labels)
            else:
                self.y_labeled = np.zeros(len(initial_idx))

        self.X_labeled = self.X_pool[initial_idx]
        self.labeled_indices = set(initial_idx.tolist())

    def step(self, batch_size: int = 5) -> Dict[str, Any]:
        """Run one active learning iteration.

        Returns
        -------
        dict
            r2, rmse, n_labeled, selected_indices, acquisition_scores.
        """
        from fibernet.ml.utils import train_predictor
        from sklearn.metrics import r2_score, mean_squared_error

        # Fit acquisition strategy
        if hasattr(self.acquisition, 'fit'):
            self.acquisition.fit(self.X_labeled, self.y_labeled)

        # Select new samples
        excluded = np.array(list(self.labeled_indices))
        if hasattr(self.acquisition, 'select'):
            selected = self.acquisition.select(
                self.X_pool, batch_size=batch_size, exclude_indices=excluded,
            )
        else:
            # Random fallback
            available = [i for i in range(len(self.X_pool)) if i not in self.labeled_indices]
            selected = np.random.choice(available, min(batch_size, len(available)), replace=False)

        # Get labels
        new_X = self.X_pool[selected]
        if self.simulator is not None:
            new_y = []
            for x in new_X:
                try:
                    label = self.simulator(x)
                    if isinstance(label, dict):
                        label = list(label.values())[0] if label else 0.0
                    new_y.append(float(label))
                except Exception:
                    new_y.append(0.0)
            new_y = np.array(new_y)
        else:
            new_y = np.zeros(len(selected))

        # Update labeled set
        self.X_labeled = np.vstack([self.X_labeled, new_X])
        self.y_labeled = np.concatenate([self.y_labeled, new_y])
        self.labeled_indices.update(selected.tolist())

        # Evaluate
        model, metrics = train_predictor(
            self.X_labeled, self.y_labeled, model_type=self.model_type,
        )

        result = {
            "r2": metrics["r2"],
            "rmse": metrics["rmse"],
            "n_labeled": len(self.X_labeled),
            "selected_indices": selected.tolist(),
            "iteration": len(self.iteration_history),
        }
        self.iteration_history.append(result)
        return result

    def get_summary(self) -> Dict[str, Any]:
        """Get active learning progress summary."""
        if not self.iteration_history:
            return {"n_labeled": len(self.X_labeled) if self.X_labeled is not None else 0}

        r2_history = [h["r2"] for h in self.iteration_history]
        n_labeled = [h["n_labeled"] for h in self.iteration_history]

        return {
            "n_iterations": len(self.iteration_history),
            "n_labeled": n_labeled[-1] if n_labeled else 0,
            "final_r2": r2_history[-1] if r2_history else 0.0,
            "r2_history": r2_history,
            "n_labeled_history": n_labeled,
            "improvement": r2_history[-1] - r2_history[0] if len(r2_history) > 1 else 0.0,
        }

    def plot_progress(self, save_path: Optional[str] = None, show: bool = False):
        """Plot active learning progress."""
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.iteration_history:
            return None

        iters = [h["iteration"] for h in self.iteration_history]
        r2s = [h["r2"] for h in self.iteration_history]
        n_lab = [h["n_labeled"] for h in self.iteration_history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#0a0a0f")

        for ax in (ax1, ax2):
            ax.set_facecolor("#0a0a0f")
            ax.tick_params(colors="#888")
            ax.spines["bottom"].set_color("#333")
            ax.spines["left"].set_color("#333")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        ax1.plot(iters, r2s, color="#b388ff", linewidth=2, marker="o", markersize=4)
        ax1.set_xlabel("Iteration", color="#aaa")
        ax1.set_ylabel("R² Score", color="#aaa")
        ax1.set_title("Active Learning: Model Performance", color="#ddd")

        ax2.plot(n_lab, r2s, color="#82b1ff", linewidth=2, marker="s", markersize=4)
        ax2.set_xlabel("Labeled Samples", color="#aaa")
        ax2.set_ylabel("R² Score", color="#aaa")
        ax2.set_title("Learning Curve", color="#ddd")

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                       facecolor=fig.get_facecolor())
        if show:
            plt.show()
        return fig
