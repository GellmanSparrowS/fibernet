"""
Multi-Objective Optimization for FiberNet Structure Design.

Implements:
- MultiObjectiveOptimizer: Optuna-based multi-objective optimization
- NSGA2Optimizer: NSGA-II via pymoo (optional)
- ParetoAnalysis: Analyze and visualize Pareto fronts
- StructureSweeper: Systematic parameter sweeps with checkpoint

Examples
--------
>>> from fibernet.ml.optimization import MultiObjectiveOptimizer, ParetoAnalysis
>>> optimizer = MultiObjectiveOptimizer(
...     objective_fn=lambda params: (-params["max_force"], params["mean_stretch"]),
...     param_space={"grid_x": (2, 6), "n_internal": (0, 5), "damping": (0.1, 0.9)},
...     n_objectives=2,
... )
>>> result = optimizer.optimize(n_trials=50)
>>> pareto = ParetoAnalysis(result)
>>> pareto.front()  # Pareto-optimal solutions
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ======================================================================
# Multi-Objective Optimizer (Optuna)
# ======================================================================

class MultiObjectiveOptimizer:
    """Optuna-based multi-objective optimization for structure design.

    Parameters
    ----------
    objective_fn : callable
        Function(params: dict) -> tuple of objectives to minimize.
        For maximization, return negative values.
    param_space : dict
        Parameter space. Keys: names, Values: (low, high) for continuous/int,
        or list for categorical.
    n_objectives : int
        Number of objectives.
    study_name : str, optional
        Study name for persistence.
    storage : str, optional
        Optuna storage URL.
    seed : int
        Random seed.

    Examples
    --------
    >>> opt = MultiObjectiveOptimizer(
    ...     objective_fn=lambda p: (p["x"]**2, (p["x"]-2)**2),
    ...     param_space={"x": (0, 3)},
    ...     n_objectives=2,
    ... )
    >>> result = opt.optimize(n_trials=30)
    >>> print(f"Pareto front: {len(result['pareto_front'])} solutions")
    """

    def __init__(
        self,
        objective_fn: Callable,
        param_space: Dict[str, Any],
        n_objectives: int = 2,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        seed: int = 42,
    ):
        self.objective_fn = objective_fn
        self.param_space = param_space
        self.n_objectives = n_objectives
        self.study_name = study_name or "fibernet_moo"
        self.storage = storage
        self.seed = seed

    def optimize(
        self,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run multi-objective optimization.

        Parameters
        ----------
        n_trials : int
            Number of optimization trials.
        timeout : int, optional
            Timeout in seconds.
        verbose : bool
            Print progress.

        Returns
        -------
        dict
            Keys: pareto_front, all_trials, study, n_trials.
        """
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna required: pip install optuna")

        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            directions=["minimize"] * self.n_objectives,
            sampler=optuna.samplers.NSGAIISampler(seed=self.seed),
            load_if_exists=True,
        )

        def objective(trial):
            params = self._suggest_params(trial)
            try:
                values = self.objective_fn(params)
                if not isinstance(values, (list, tuple)):
                    values = [values]
                return list(values)
            except Exception as e:
                if verbose:
                    print(f"  Trial {trial.number} failed: {e}")
                return [float("inf")] * self.n_objectives

        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        # Extract Pareto front
        pareto_trials = study.best_trials
        pareto_front = []
        for t in pareto_trials:
            pareto_front.append({
                "params": t.params,
                "values": t.values,
                "number": t.number,
            })

        all_trials = []
        for t in study.trials:
            if t.values is not None:
                all_trials.append({
                    "params": t.params,
                    "values": t.values,
                    "number": t.number,
                })

        if verbose:
            print(f"\nPareto front: {len(pareto_front)} solutions from {len(all_trials)} trials")
            for i, sol in enumerate(pareto_front[:5]):
                print(f"  #{i+1}: objectives={sol['values']}, params={sol['params']}")

        return {
            "pareto_front": pareto_front,
            "all_trials": all_trials,
            "study": study,
            "n_trials": len(all_trials),
        }

    def _suggest_params(self, trial) -> Dict[str, Any]:
        """Suggest parameters from space definition."""
        params = {}
        for name, spec in self.param_space.items():
            if isinstance(spec, (list, tuple)) and len(spec) == 2:
                low, high = spec
                if isinstance(low, int) and isinstance(high, int):
                    params[name] = trial.suggest_int(name, low, high)
                elif isinstance(low, float) or isinstance(high, float):
                    params[name] = trial.suggest_float(name, float(low), float(high))
                else:
                    params[name] = trial.suggest_float(name, float(low), float(high))
            elif isinstance(spec, list):
                params[name] = trial.suggest_categorical(name, spec)
            elif isinstance(spec, dict):
                # Advanced spec
                t = spec.get("type", "float")
                if t == "float":
                    params[name] = trial.suggest_float(
                        name, spec["low"], spec["high"], log=spec.get("log", False),
                    )
                elif t == "int":
                    params[name] = trial.suggest_int(name, spec["low"], spec["high"])
                elif t == "categorical":
                    params[name] = trial.suggest_categorical(name, spec["choices"])
            else:
                raise ValueError(f"Invalid spec for {name}: {spec}")
        return params


# ======================================================================
# NSGA-II (pymoo, optional)
# ======================================================================

class NSGA2Optimizer:
    """NSGA-II optimizer using pymoo (optional dependency).

    Parameters
    ----------
    objective_fn : callable
        Function(x: np.ndarray) -> np.ndarray of objectives.
    n_var : int
        Number of decision variables.
    n_obj : int
        Number of objectives.
    xl : np.ndarray
        Lower bounds.
    xu : np.ndarray
        Upper bounds.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        objective_fn: Callable,
        n_var: int,
        n_obj: int = 2,
        xl: Optional[np.ndarray] = None,
        xu: Optional[np.ndarray] = None,
        seed: int = 42,
    ):
        self.objective_fn = objective_fn
        self.n_var = n_var
        self.n_obj = n_obj
        self.xl = xl if xl is not None else np.zeros(n_var)
        self.xu = xu if xu is not None else np.ones(n_var)
        self.seed = seed

    def optimize(
        self,
        pop_size: int = 50,
        n_gen: int = 100,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run NSGA-II optimization.

        Parameters
        ----------
        pop_size : int
            Population size.
        n_gen : int
            Number of generations.

        Returns
        -------
        dict with pareto_front, all_solutions.
        """
        try:
            from pymoo.algorithms.moo.nsga2 import NSGA2
            from pymoo.core.problem import Problem
            from pymoo.optimize import minimize
            from pymoo.termination import get_termination
        except ImportError:
            raise ImportError("pymoo required: pip install pymoo")

        problem = Problem(
            n_var=self.n_var,
            n_obj=self.n_obj,
            xl=self.xl,
            xu=self.xu,
        )
        # Override evaluate
        def evaluate(x, out, *args, **kwargs):
            F = np.array([self.objective_fn(xi) for xi in x])
            out["F"] = F
        problem.evaluate = evaluate

        algorithm = NSGA2(pop_size=pop_size)
        termination = get_termination("n_gen", n_gen)

        result = minimize(
            problem, algorithm, termination,
            seed=self.seed, verbose=verbose,
        )

        if result.F is not None:
            pareto_front = []
            for i in range(len(result.X)):
                x = result.X[i] if result.X.ndim > 1 else result.X
                pareto_front.append({
                    "x": x.tolist() if hasattr(x, "tolist") else x,
                    "objectives": result.F[i].tolist(),
                })

            return {
                "pareto_front": pareto_front,
                "n_solutions": len(pareto_front),
                "result": result,
            }
        else:
            return {"pareto_front": [], "n_solutions": 0, "result": result}


# ======================================================================
# Pareto Analysis
# ======================================================================

class ParetoAnalysis:
    """Analyze and manipulate Pareto-optimal solutions.

    Parameters
    ----------
    optimization_result : dict
        Result from MultiObjectiveOptimizer.optimize().
    """

    def __init__(self, optimization_result: Dict[str, Any]):
        self.pareto_front = optimization_result.get("pareto_front", [])
        self.all_trials = optimization_result.get("all_trials", [])

    def front(self) -> List[Dict]:
        """Get Pareto-optimal solutions."""
        return self.pareto_front

    def front_values(self) -> np.ndarray:
        """Get Pareto front objective values as array."""
        if not self.pareto_front:
            return np.array([])
        return np.array([s["values"] for s in self.pareto_front])

    def select_best(
        self,
        weights: Optional[List[float]] = None,
        method: str = "weighted_sum",
    ) -> Dict[str, Any]:
        """Select best solution from Pareto front.

        Parameters
        ----------
        weights : list of float, optional
            Objective weights (for weighted_sum). Default: uniform.
        method : str
            "weighted_sum": weighted sum of objectives
            "knee_point": solution with max curvature
            "closest_to_ideal": closest to ideal point (0, 0, ...)

        Returns
        -------
        dict
            Selected solution.
        """
        if not self.pareto_front:
            return {}

        values = self.front_values()
        n_obj = values.shape[1]

        if weights is None:
            weights = [1.0 / n_obj] * n_obj

        if method == "weighted_sum":
            scores = values @ np.array(weights)
            best_idx = np.argmin(scores)
        elif method == "knee_point":
            # Normalize to [0, 1]
            v_min = values.min(axis=0)
            v_max = values.max(axis=0)
            v_range = v_max - v_min
            v_range[v_range == 0] = 1.0
            norm_values = (values - v_min) / v_range

            # Distance to line connecting extreme points
            if len(norm_values) > 2 and n_obj == 2:
                p1 = norm_values[0]
                p2 = norm_values[-1]
                line_vec = p2 - p1
                line_len = np.linalg.norm(line_vec)
                if line_len > 0:
                    distances = []
                    for p in norm_values:
                        d = abs(np.cross(line_vec, p1 - p)) / line_len
                        distances.append(d)
                    best_idx = np.argmax(distances)
                else:
                    best_idx = 0
            else:
                best_idx = len(self.pareto_front) // 2

        elif method == "closest_to_ideal":
            ideal = values.min(axis=0)
            distances = np.linalg.norm(values - ideal, axis=1)
            best_idx = np.argmin(distances)
        else:
            raise ValueError(f"Unknown method: {method}")

        return self.pareto_front[best_idx]

    def hypervolume(self, reference_point: Optional[List[float]] = None) -> float:
        """Compute hypervolume indicator.

        Parameters
        ----------
        reference_point : list of float, optional
            Reference point. Default: max of each objective + margin.

        Returns
        -------
        float
            Hypervolume value.
        """
        values = self.front_values()
        if len(values) == 0:
            return 0.0

        if reference_point is None:
            reference_point = values.max(axis=0) * 1.1

        ref = np.array(reference_point)

        # Simple 2D hypervolume
        if values.shape[1] == 2:
            sorted_vals = values[values[:, 0].argsort()]
            hv = 0.0
            prev_y = ref[1]
            for v in sorted_vals:
                hv += (ref[0] - v[0]) * (prev_y - v[1])
                prev_y = v[1]
            return hv

        # For higher dimensions, use approximation
        # Monte Carlo estimation
        rng = np.random.RandomState(42)
        n_samples = 10000
        n_obj = values.shape[1]

        # Sample random points in the bounding box
        samples = np.zeros((n_samples, n_obj))
        for d in range(n_obj):
            samples[:, d] = rng.uniform(values[:, d].min(), ref[d], n_samples)

        # Count dominated points
        dominated = 0
        for s in samples:
            for v in values:
                if np.all(v <= s):
                    dominated += 1
                    break

        # Volume of bounding box
        box_vol = np.prod(ref - values.min(axis=0))
        return box_vol * dominated / n_samples

    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required: pip install pandas")

        rows = []
        for sol in self.pareto_front:
            row = dict(sol.get("params", {}))
            for i, v in enumerate(sol.get("values", [])):
                row[f"objective_{i}"] = v
            rows.append(row)
        return pd.DataFrame(rows)

    def save(self, path: str):
        """Save Pareto front to JSON."""
        data = {
            "pareto_front": self.pareto_front,
            "all_trials": self.all_trials,
        }
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "ParetoAnalysis":
        """Load from JSON."""
        with open(path) as fh:
            data = json.load(fh)
        return cls(data)


# ======================================================================
# Structure Parameter Sweeper
# ======================================================================

class StructureSweeper:
    """Systematic parameter sweep with checkpoint/resume.

    Parameters
    ----------
    objective_fn : callable
        Function(params) -> dict of metrics.
    param_grid : dict
        Parameter grid. Keys: names, Values: list of values.
    checkpoint_path : str, optional
        Checkpoint file for resume.
    """

    def __init__(
        self,
        objective_fn: Callable,
        param_grid: Dict[str, List],
        checkpoint_path: Optional[str] = None,
    ):
        self.objective_fn = objective_fn
        self.param_grid = param_grid
        self.checkpoint_path = checkpoint_path
        self.results: List[Dict] = []
        self._completed_keys = set()

        # Load checkpoint
        if checkpoint_path and Path(checkpoint_path).exists():
            with open(checkpoint_path) as fh:
                ckpt = json.load(fh)
            self.results = ckpt.get("results", [])
            self._completed_keys = {r.get("_key", "") for r in self.results}

    def run(
        self,
        verbose: bool = True,
        checkpoint_every: int = 10,
    ) -> List[Dict]:
        """Run the parameter sweep.

        Returns
        -------
        list of dict
            All results with parameters and metrics.
        """
        import itertools

        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combos = list(itertools.product(*values))

        total = len(combos)
        done = 0

        for i, combo in enumerate(combos):
            params = {k: v for k, v in zip(keys, combo)}
            key = json.dumps(params, sort_keys=True)

            if key in self._completed_keys:
                done += 1
                continue

            try:
                metrics = self.objective_fn(params)
                result = {**params, **metrics, "_key": key}
                self.results.append(result)
                self._completed_keys.add(key)
                done += 1

                if verbose:
                    print(f"  [{done}/{total}] {params} → {metrics}")

            except Exception as e:
                result = {**params, "_error": str(e), "_key": key}
                self.results.append(result)
                done += 1
                if verbose:
                    print(f"  [{done}/{total}] FAILED: {e}")

            if self.checkpoint_path and done % checkpoint_every == 0:
                self._save_checkpoint()

        if self.checkpoint_path:
            self._save_checkpoint()

        if verbose:
            print(f"\nSweep complete: {len(self.results)} results")

        return self.results

    def _save_checkpoint(self):
        if self.checkpoint_path:
            Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            tmp = self.checkpoint_path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump({"results": self.results}, fh, indent=2, default=str)
            import os
            os.replace(tmp, self.checkpoint_path)

    def to_dataframe(self):
        """Convert results to pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required")
        return pd.DataFrame(self.results).drop(columns=["_key"], errors="ignore")

    def pareto_front(
        self,
        objective_cols: List[str],
        minimize: Optional[List[bool]] = None,
    ) -> "ParetoAnalysis":
        """Extract Pareto front from sweep results.

        Parameters
        ----------
        objective_cols : list of str
            Column names to use as objectives.
        minimize : list of bool, optional
            Whether to minimize each objective. Default: all minimize.

        Returns
        -------
        ParetoAnalysis
        """
        valid = [r for r in self.results if "_error" not in r]
        if minimize is None:
            minimize = [True] * len(objective_cols)

        # Convert to optimization format
        pareto_front = []
        all_trials = []
        for r in valid:
            values = []
            for j, col in enumerate(objective_cols):
                v = float(r.get(col, 0))
                if not minimize[j]:
                    v = -v
                values.append(v)

            params = {k: v for k, v in r.items() if k not in objective_cols and not k.startswith("_")}
            trial = {"params": params, "values": values}
            all_trials.append(trial)

        # Compute Pareto front
        values_arr = np.array([t["values"] for t in all_trials])
        is_pareto = np.ones(len(values_arr), dtype=bool)
        for i in range(len(values_arr)):
            if not is_pareto[i]:
                continue
            for j in range(len(values_arr)):
                if i == j or not is_pareto[j]:
                    continue
                if np.all(values_arr[j] <= values_arr[i]) and np.any(values_arr[j] < values_arr[i]):
                    is_pareto[i] = False
                    break

        pareto_front = [all_trials[i] for i in range(len(all_trials)) if is_pareto[i]]

        return ParetoAnalysis({
            "pareto_front": pareto_front,
            "all_trials": all_trials,
        })
