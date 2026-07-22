"""
Symbolic Regression for FiberNet — Discovering Interpretable Relationships.

Wraps PySR for discovering analytical expressions relating
structure features to mechanical properties:
- SymbolicRegressor: Wrapper around PySR for fiber network data
- ExpressionAnalyzer: Analyze and visualize discovered expressions
- FeatureSelection: Pre-processing for symbolic regression

Features
--------
- Automatic feature selection for interpretable models
- Expression complexity vs. accuracy trade-off (Pareto front)
- Export of discovered expressions as Python/LaTeX
- Integration with fibernet ML pipeline

References
----------
- Cranmer, "Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl"
- Article section 4.2: Scalar and functional predictions

Examples
--------
>>> from fibernet.ml.symbolic import SymbolicRegressor
>>> sr = SymbolicRegressor(n_features_to_select=5)
>>> sr.fit(X, y, feature_names=["n_nodes", "mean_degree", ...])
>>> print(sr.best_expression())
>>> print(sr.hall_of_fame())  # Pareto front of expressions
>>> latex = sr.to_latex()  # LaTeX formula
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


class FeatureSelection:
    """Pre-processing feature selection for symbolic regression.

    Reduces feature space to most relevant features using
    multiple importance metrics.

    Parameters
    ----------
    n_select : int
        Number of features to select.
    method : str
        "mutual_info", "correlation", "permutation", "combined".
    """

    def __init__(self, n_select: int = 5, method: str = "combined"):
        self.n_select = n_select
        self.method = method
        self.selected_indices: Optional[np.ndarray] = None
        self.selected_names: Optional[List[str]] = None
        self.scores: Optional[Dict[str, np.ndarray]] = None

    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Select most relevant features.

        Parameters
        ----------
        X : (n_samples, n_features)
        y : (n_samples,)
        feature_names : list of str, optional

        Returns
        -------
        X_selected : (n_samples, n_select)
        names : list of selected feature names
        """
        X = np.asarray(X)
        y = np.asarray(y).flatten()
        n_features = X.shape[1]

        if feature_names is None:
            feature_names = [f"f{i}" for i in range(n_features)]

        n_select = min(self.n_select, n_features)

        if self.method == "correlation":
            scores = np.array([abs(np.corrcoef(X[:, i], y)[0, 1])
                              for i in range(n_features)])
            scores = np.nan_to_num(scores)
        elif self.method == "mutual_info":
            from sklearn.feature_selection import mutual_info_regression
            scores = mutual_info_regression(X, y, random_state=42)
        elif self.method == "permutation":
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.inspection import permutation_importance
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X, y)
            result = permutation_importance(rf, X, y, n_repeats=10, random_state=42)
            scores = result.importances_mean
        else:
            # Combined: average of correlation and MI
            corr = np.array([abs(np.corrcoef(X[:, i], y)[0, 1])
                            for i in range(n_features)])
            corr = np.nan_to_num(corr)
            try:
                from sklearn.feature_selection import mutual_info_regression
                mi = mutual_info_regression(X, y, random_state=42)
            except ImportError:
                mi = corr
            scores = (corr / (corr.max() + 1e-8) + mi / (mi.max() + 1e-8)) / 2

        top_idx = np.argsort(scores)[::-1][:n_select]
        self.selected_indices = top_idx
        self.selected_names = [feature_names[i] for i in top_idx]
        self.scores = {"scores": scores, "feature_names": feature_names}

        return X[:, top_idx], self.selected_names


class SymbolicRegressor:
    """Symbolic regression wrapper for discovering interpretable formulas.

    Parameters
    ----------
    n_features_to_select : int
        Number of features to use (pre-selection).
    max_iterations : int
        Maximum PySR iterations.
    populations : int
        Number of populations for evolutionary search.
    complexity_penalty : float
        Penalty for expression complexity.
    binary_operators : list of str
        Operators to use: ["+", "*", "-", "/", "^"].
    unary_operators : list of str
        Unary operators: ["sin", "cos", "exp", "log", "sqrt", "abs"].

    Examples
    --------
    >>> sr = SymbolicRegressor(n_features_to_select=5)
    >>> sr.fit(X, y, feature_names=["density", "mean_degree", "n_edges"])
    >>> print(sr.best_expression())
    >>> print(sr.predict(X_test))
    """

    def __init__(
        self,
        n_features_to_select: int = 5,
        max_iterations: int = 40,
        populations: int = 15,
        complexity_penalty: float = 0.01,
        binary_operators: Optional[List[str]] = None,
        unary_operators: Optional[List[str]] = None,
    ):
        self.n_features_to_select = n_features_to_select
        self.max_iterations = max_iterations
        self.populations = populations
        self.complexity_penalty = complexity_penalty

        self.binary_operators = binary_operators or ["+", "*", "-", "/"]
        self.unary_operators = unary_operators or ["sin", "cos", "sqrt", "abs"]

        self.feature_selector = FeatureSelection(n_features_to_select, "combined")
        self.model = None
        self.feature_names_selected: Optional[List[str]] = None
        self.X_selected: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        feature_names: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> "SymbolicRegressor":
        """Fit symbolic regression model.

        Parameters
        ----------
        X : (n_samples, n_features)
        y : (n_samples,)
        feature_names : list of str, optional
        verbose : bool

        Returns
        -------
        self
        """
        try:
            from pysr import PySRRegressor
        except ImportError:
            raise ImportError("PySR required: pip install pysr")

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).flatten()

        # Feature selection
        X_sel, names = self.feature_selector.select(X, y, feature_names)
        self.X_selected = X_sel
        self.feature_names_selected = names

        if verbose:
            print(f"Selected features: {names}")

        # Fit PySR
        self.model = PySRRegressor(
            niterations=self.max_iterations,
            populations=self.populations,
            binary_operators=self.binary_operators,
            unary_operators=self.unary_operators,
            complexity_of_constants=self.complexity_penalty,
            verbosity=1 if verbose else 0,
            random_state=42,
        )
        self.model.fit(X_sel, y, variable_names=names)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using best expression."""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        if self.feature_selector.selected_indices is not None:
            X = X[:, self.feature_selector.selected_indices]
        return self.model.predict(X)

    def best_expression(self) -> str:
        """Get the best expression (highest scoring)."""
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        return str(self.model.get_best())

    def hall_of_fame(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get Pareto front of expressions (complexity vs accuracy).

        Returns
        -------
        list of dict
            Each with expression, complexity, score, mse.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted.")

        equations = self.model.equations_
        if equations is None or len(equations) == 0:
            return []

        results = []
        for idx, row in equations.iterrows():
            results.append({
                "expression": str(row["equation"]),
                "complexity": int(row["complexity"]),
                "loss": float(row["loss"]),
                "score": float(row.get("score", 0)),
            })

        return sorted(results, key=lambda x: x["loss"])[:n]

    def to_latex(self) -> str:
        """Get LaTeX representation of best expression."""
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        try:
            return self.model.latex()
        except Exception:
            return self.best_expression()

    def to_sympy(self):
        """Get SymPy expression for further analysis."""
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        try:
            return self.model.sympy()
        except Exception:
            return None

    def get_equation_df(self):
        """Get all discovered equations as DataFrame."""
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        return self.model.equations_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """R² score on test data."""
        from sklearn.metrics import r2_score
        pred = self.predict(X)
        return r2_score(y, pred)

    def save(self, path: str):
        """Save model to disk."""
        import pickle
        bundle = {
            "model": self.model,
            "feature_selector": self.feature_selector,
            "feature_names": self.feature_names_selected,
        }
        with open(path, "wb") as f:
            pickle.dump(bundle, f)

    def load(self, path: str):
        """Load model from disk."""
        import pickle
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        self.model = bundle["model"]
        self.feature_selector = bundle["feature_selector"]
        self.feature_names_selected = bundle.get("feature_names")


class ExpressionAnalyzer:
    """Analyze discovered symbolic expressions.

    Parameters
    ----------
    expressions : list of dict
        From SymbolicRegressor.hall_of_fame().
    """

    def __init__(self, expressions: List[Dict[str, Any]]):
        self.expressions = expressions

    def complexity_accuracy_tradeoff(self) -> Dict[str, Any]:
        """Analyze Pareto front of complexity vs accuracy."""
        complexities = [e["complexity"] for e in self.expressions]
        losses = [e["loss"] for e in self.expressions]
        return {
            "complexities": complexities,
            "losses": losses,
            "min_complexity": min(complexities) if complexities else 0,
            "min_loss": min(losses) if losses else float("inf"),
            "best_tradeoff_idx": self._find_elbow(complexities, losses),
        }

    def _find_elbow(self, x: List[int], y: List[float]) -> int:
        """Find elbow point in Pareto front."""
        if len(x) < 3:
            return 0
        x_arr = np.array(x, dtype=float)
        y_arr = np.array(y, dtype=float)
        # Normalize
        x_norm = (x_arr - x_arr.min()) / (x_arr.max() - x_arr.min() + 1e-8)
        y_norm = (y_arr - y_arr.min()) / (y_arr.max() - y_arr.min() + 1e-8)
        distances = x_norm + y_norm
        return int(np.argmin(distances))

    def plot_pareto(self, save_path: Optional[str] = None, show: bool = False):
        """Plot complexity vs loss Pareto front."""
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        complexities = [e["complexity"] for e in self.expressions]
        losses = [e["loss"] for e in self.expressions]

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#0a0a0f")
        ax.set_facecolor("#0a0a0f")

        ax.plot(complexities, losses, "o-", color="#b388ff", linewidth=2, markersize=6)

        for i, (c, l, expr) in enumerate(zip(complexities, losses, self.expressions)):
            ax.annotate(
                expr["expression"][:30],
                (c, l), fontsize=7, color="#aaa",
                textcoords="offset points", xytext=(5, 5),
            )

        ax.set_xlabel("Complexity", color="#aaa")
        ax.set_ylabel("Loss (MSE)", color="#aaa")
        ax.set_title("Symbolic Regression: Pareto Front", color="#ddd")
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
