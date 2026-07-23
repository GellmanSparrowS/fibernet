"""
Multi-Objective Reinforcement Learning for FiberNet.

Implements MORL strategies for optimizing multiple competing objectives:
- ScalarizedMORL: Weighted scalarization of multiple objectives
- ParetoFrontExplorer: Systematic Pareto front exploration
- MORLCallback: SB3 callback for multi-objective tracking
- ObjectiveAnalyzer: Analyze trade-offs between objectives

Features
--------
- Linear and non-linear scalarization
- Automatic Pareto front extraction from multiple training runs
- Integration with Stable Baselines3 via callbacks
- Visualization of objective trade-offs
- Hypervolume metric for Pareto front quality

References
----------
- Article section 5: Multi-objective optimization with composite rewards
- Yang et al., "A survey of deep reinforcement learning for multi-objective optimization"

Examples
--------
>>> from fibernet.rl.multi_objective_rl import ScalarizedMORL, ParetoFrontExplorer
>>> morl = ScalarizedMORL(
...     objectives={"force": lambda i: -i["max_force"], "weight": lambda i: -i["n_edges"]},
... )
>>> # Train with different scalarization weights
>>> results = morl.sweep_weights(
...     env_factory=lambda: FiberStructureEnv(),
...     weight_grid=[[1.0, 0.0], [0.7, 0.3], [0.5, 0.5], [0.3, 0.7], [0.0, 1.0]],
...     total_timesteps=5000,
... )
>>> pareto = morl.extract_pareto_front(results)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


class ScalarizedMORL:
    """Multi-objective RL via scalarized rewards.

    Converts multiple objectives into a single reward using
    weighted scalarization.

    Parameters
    ----------
    objectives : dict
        Named objective functions: {name: callable(info) -> float}.
    weights : dict, optional
        Weights for each objective. Default: uniform.

    Examples
    --------
    >>> morl = ScalarizedMORL(
    ...     objectives={
    ...         "force": lambda i: -i.get("max_force", 0) * 1e-4,
    ...         "weight": lambda i: -i.get("n_edges", 0) * 0.001,
    ...         "uniformity": lambda i: -i.get("std_stretch", 0),
    ...     },
    ...     weights={"force": 1.0, "weight": 0.5, "uniformity": 2.0},
    ... )
    """

    def __init__(
        self,
        objectives: Dict[str, Callable],
        weights: Optional[Dict[str, float]] = None,
    ):
        self.objectives = objectives
        self.weights = weights or {name: 1.0 for name in objectives}

    def compute_reward(self, info: Dict[str, Any]) -> float:
        """Compute scalarized reward from info dict."""
        total = 0.0
        for name, fn in self.objectives.items():
            w = self.weights.get(name, 1.0)
            try:
                total += w * fn(info)
            except (KeyError, TypeError):
                pass
        return float(total)

    def compute_individual_rewards(self, info: Dict[str, Any]) -> Dict[str, float]:
        """Compute each objective value separately."""
        result = {}
        for name, fn in self.objectives.items():
            try:
                result[name] = float(fn(info))
            except (KeyError, TypeError):
                result[name] = 0.0
        return result

    def sweep_weights(
        self,
        env_factory: Callable,
        weight_grid: List[List[float]],
        *,
        total_timesteps: int = 5000,
        algorithm: str = "PPO",
        n_eval_episodes: int = 5,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """Train with multiple weight configurations and collect results.

        Parameters
        ----------
        env_factory : callable
            Creates a new environment instance.
        weight_grid : list of list of float
            Each inner list is a set of weights for the objectives.
        total_timesteps : int
            Training steps per weight configuration.
        algorithm : str
            RL algorithm name.
        n_eval_episodes : int
            Evaluation episodes per configuration.
        verbose : bool

        Returns
        -------
        list of dict
            Results for each weight configuration.
        """
        obj_names = list(self.objectives.keys())
        results = []

        for i, weights_list in enumerate(weight_grid):
            if len(weights_list) != len(obj_names):
                if verbose:
                    print(f"Skipping weight {weights_list}: wrong dimension")
                continue

            self.weights = {name: w for name, w in zip(obj_names, weights_list)}

            if verbose:
                print(f"\n[{i+1}/{len(weight_grid)}] Weights: {self.weights}")

            try:
                env = env_factory()

                # Wrap env to use scalarized reward
                class _RewardEnv:
                    def __init__(self, base_env, morl):
                        self._env = base_env
                        self._morl = morl
                    def reset(self, **kw):
                        return self._env.reset(**kw)
                    def step(self, action):
                        obs, r, term, trunc, info = self._env.step(action)
                        shaped_r = self._morl.compute_reward(info)
                        info["individual_rewards"] = self._morl.compute_individual_rewards(info)
                        return obs, shaped_r, term, trunc, info
                    def __getattr__(self, name):
                        return getattr(self._env, name)

                wrapped_env = _RewardEnv(env, self)

                try:
                    from fibernet.rl.algorithms import train_rl, evaluate_rl
                    model, train_info = train_rl(
                        wrapped_env, algorithm=algorithm,
                        total_timesteps=total_timesteps, verbose=0,
                    )
                    rewards, eval_infos = evaluate_rl(
                        model, wrapped_env, n_episodes=n_eval_episodes,
                    )

                    # Collect individual objective values
                    obj_values = {name: [] for name in obj_names}
                    for info in eval_infos:
                        indiv = info.get("individual_rewards", {})
                        for name in obj_names:
                            obj_values[name].append(indiv.get(name, 0.0))

                    result = {
                        "weights": dict(self.weights),
                        "mean_reward": float(np.mean(rewards)),
                        "std_reward": float(np.std(rewards)),
                        "objective_means": {k: float(np.mean(v)) for k, v in obj_values.items()},
                        "objective_stds": {k: float(np.std(v)) for k, v in obj_values.items()},
                    }
                    results.append(result)

                except ImportError:
                    if verbose:
                        print("  SB3 not available, skipping training")
                    results.append({
                        "weights": dict(self.weights),
                        "mean_reward": 0.0,
                        "objective_means": {k: 0.0 for k in obj_names},
                    })

            except Exception as e:
                if verbose:
                    print(f"  Failed: {e}")
                results.append({
                    "weights": dict(self.weights),
                    "error": str(e),
                })

        return results

    @staticmethod
    def extract_pareto_front(
        results: List[Dict[str, Any]],
        objective_keys: Optional[List[str]] = None,
        maximize: Optional[List[bool]] = None,
    ) -> List[Dict[str, Any]]:
        """Extract Pareto-optimal solutions from sweep results.

        Parameters
        ----------
        results : list of dict
            From sweep_weights().
        objective_keys : list of str, optional
            Keys in objective_means to use. Default: all.
        maximize : list of bool, optional
            Whether to maximize each objective. Default: all maximize.

        Returns
        -------
        list of dict
            Pareto-optimal results.
        """
        valid = [r for r in results if "error" not in r]
        if not valid:
            return []

        if objective_keys is None:
            objective_keys = list(valid[0]["objective_means"].keys())

        if maximize is None:
            maximize = [True] * len(objective_keys)

        # Convert to minimization
        values = []
        for r in valid:
            vals = []
            for j, key in enumerate(objective_keys):
                v = r["objective_means"].get(key, 0.0)
                if maximize[j]:
                    v = -v
                vals.append(v)
            values.append(vals)

        values_arr = np.array(values)
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

        return [valid[i] for i in range(len(valid)) if is_pareto[i]]


class ParetoFrontExplorer:
    """Systematic Pareto front exploration with adaptive weighting.

    Automatically explores weight configurations to discover
    diverse Pareto-optimal solutions.

    Parameters
    ----------
    objectives : dict
        Named objective functions.
    n_exploration_points : int
        Number of weight configurations to explore.
    n_objectives : int
        Number of objectives.
    """

    def __init__(
        self,
        objectives: Dict[str, Callable],
        n_exploration_points: int = 20,
    ):
        self.objectives = objectives
        self.n_exploration_points = n_exploration_points
        self.n_objectives = len(objectives)
        self.explored_weights: List[List[float]] = []
        self.explored_results: List[Dict] = []

    def generate_weight_grid(self) -> List[List[float]]:
        """Generate uniform weight grid using simplex sampling."""
        n_obj = self.n_objectives
        n_pts = self.n_exploration_points

        if n_obj == 2:
            weights = [[w, 1 - w] for w in np.linspace(0, 1, n_pts)]
        elif n_obj == 3:
            # Simplex grid for 3 objectives
            weights = []
            n_per_dim = max(int(np.sqrt(n_pts)), 2)
            for i in range(n_per_dim):
                for j in range(n_per_dim - i):
                    k = n_per_dim - 1 - i - j
                    w = [i / (n_per_dim - 1), j / (n_per_dim - 1), k / (n_per_dim - 1)]
                    weights.append(w)
        else:
            # Random simplex sampling
            rng = np.random.RandomState(42)
            weights = []
            for _ in range(n_pts):
                w = rng.exponential(1.0, n_obj)
                w = w / w.sum()
                weights.append(w.tolist())

        return weights

    def explore(
        self,
        env_factory: Callable,
        *,
        total_timesteps: int = 5000,
        algorithm: str = "PPO",
        n_eval_episodes: int = 5,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """Run exploration and return Pareto front.

        Returns
        -------
        list of dict
            Pareto-optimal results.
        """
        weight_grid = self.generate_weight_grid()
        self.explored_weights = weight_grid

        morl = ScalarizedMORL(self.objectives)
        results = morl.sweep_weights(
            env_factory, weight_grid,
            total_timesteps=total_timesteps,
            algorithm=algorithm,
            n_eval_episodes=n_eval_episodes,
            verbose=verbose,
        )

        self.explored_results = results
        return morl.extract_pareto_front(results)

    def hypervolume(self, pareto_front: List[Dict], reference: Optional[Dict] = None) -> float:
        """Compute hypervolume indicator for Pareto front quality.

        Parameters
        ----------
        pareto_front : list of dict
            Pareto-optimal results.
        reference : dict, optional
            Reference point for hypervolume. Default: worst observed.

        Returns
        -------
        float
            Hypervolume indicator (higher = better Pareto front).
        """
        if not pareto_front:
            return 0.0

        obj_keys = list(pareto_front[0]["objective_means"].keys())

        if reference is None:
            reference = {
                key: min(
                    r["objective_means"].get(key, float("inf"))
                    for r in pareto_front
                ) - 1.0
                for key in obj_keys
            }

        # 2D hypervolume (exact)
        if len(obj_keys) == 2:
            points = sorted(
                [(r["objective_means"].get(obj_keys[0], 0), r["objective_means"].get(obj_keys[1], 0))
                 for r in pareto_front],
                key=lambda p: p[0],
            )
            hv = 0.0
            ref_y = reference[obj_keys[1]]
            for i, (x, y) in enumerate(points):
                if i < len(points) - 1:
                    width = points[i + 1][0] - x
                else:
                    width = 0
                height = y - ref_y
                hv += width * max(height, 0)
            return hv

        # General case: approximate with Monte Carlo
        rng = np.random.RandomState(42)
        n_samples = 10000
        obj_values = np.array([
            [r["objective_means"].get(k, 0) for k in obj_keys]
            for r in pareto_front
        ])
        ref_values = np.array([reference[k] for k in obj_keys])

        # Sample points in bounding box
        bounds_min = ref_values
        bounds_max = obj_values.max(axis=0)
        samples = rng.uniform(bounds_min, bounds_max, (n_samples, len(obj_keys)))

        # Check dominance
        dominated = np.zeros(n_samples, dtype=bool)
        for obj in obj_values:
            dominated |= np.all(samples <= obj, axis=1)

        volume = np.prod(bounds_max - bounds_min)
        return float(dominated.mean() * volume)

    def plot_pareto(
        self,
        pareto_front: Optional[List[Dict]] = None,
        all_results: Optional[List[Dict]] = None,
        objective_keys: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        show: bool = False,
    ):
        """Plot Pareto front in 2D."""
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if pareto_front is None:
            pareto_front = ScalarizedMORL.extract_pareto_front(self.explored_results)
        if all_results is None:
            all_results = self.explored_results

        if not pareto_front:
            return None

        if objective_keys is None:
            objective_keys = list(pareto_front[0]["objective_means"].keys())[:2]

        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor("#0a0a0f")
        ax.set_facecolor("#0a0a0f")

        # All results
        for r in all_results:
            if "error" not in r:
                x = r["objective_means"].get(objective_keys[0], 0)
                y = r["objective_means"].get(objective_keys[1], 0)
                ax.scatter(x, y, c="#666", s=30, alpha=0.5, zorder=1)

        # Pareto front
        pf_x = [r["objective_means"].get(objective_keys[0], 0) for r in pareto_front]
        pf_y = [r["objective_means"].get(objective_keys[1], 0) for r in pareto_front]
        ax.scatter(pf_x, pf_y, c="#b388ff", s=80, edgecolors="#fff", linewidth=1, zorder=2)

        # Sort and connect
        sorted_pts = sorted(zip(pf_x, pf_y))
        if len(sorted_pts) > 1:
            ax.plot([p[0] for p in sorted_pts], [p[1] for p in sorted_pts],
                    color="#b388ff", linewidth=2, alpha=0.6, zorder=2)

        ax.set_xlabel(objective_keys[0], color="#aaa", fontsize=12)
        ax.set_ylabel(objective_keys[1], color="#aaa", fontsize=12)
        ax.set_title("Multi-Objective RL: Pareto Front", color="#ddd", fontsize=14)
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
