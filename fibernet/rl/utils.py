"""
RL Utilities for FiberNet — One-line RL workflows.

Provides convenience functions for training, evaluating, and visualizing
RL agents for fiber network optimization.

Examples
--------
>>> from fibernet.rl.utils import plot_reward_curve, run_bayesian_optimization
>>> plot_reward_curve(rewards, window=20, save_path="reward.png")
>>> best = run_bayesian_optimization(objective_fn, param_space, n_iter=50)
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ======================================================================
# Visualization
# ======================================================================

def plot_reward_curve(
    rewards: Sequence[float],
    *,
    window: int = 20,
    title: str = "Training Progress",
    save_path: Optional[str] = None,
    show: bool = False,
) -> Any:
    """Plot reward curve with moving average.

    Parameters
    ----------
    rewards : list of float
        Per-episode rewards.
    window : int
        Smoothing window size.
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

    rewards = np.asarray(rewards)
    episodes = np.arange(1, len(rewards) + 1)

    # Moving average
    if len(rewards) >= window:
        kernel = np.ones(window) / window
        smoothed = np.convolve(rewards, kernel, mode="valid")
        smooth_x = episodes[window - 1:]
    else:
        smoothed = rewards
        smooth_x = episodes

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#0a0a0f")

    # Raw rewards (faint)
    ax.scatter(episodes, rewards, c="#b388ff", s=8, alpha=0.3, edgecolors="none")
    # Smoothed
    ax.plot(smooth_x, smoothed, color="#b388ff", linewidth=2,
            label=f"MA-{window}")

    # Best so far
    best_so_far = np.maximum.accumulate(rewards)
    ax.plot(episodes, best_so_far, color="#82b1ff", linewidth=1.5,
            alpha=0.6, linestyle="--", label="Best so far")

    ax.set_xlabel("Episode", color="#aaa", fontsize=12)
    ax.set_ylabel("Reward", color="#aaa", fontsize=12)
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


def plot_convergence(
    values: Sequence[float],
    *,
    minimize: bool = True,
    title: str = "Optimization Convergence",
    save_path: Optional[str] = None,
    show: bool = False,
) -> Any:
    """Plot convergence curve for Bayesian/grid optimization.

    Parameters
    ----------
    values : list of float
        Objective values per iteration.
    minimize : bool
        If True, show best-so-far minimum. If False, show best-so-far maximum.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(values)
    iters = np.arange(1, len(values) + 1)

    if minimize:
        best_so_far = np.minimum.accumulate(values)
        best_label = "Best (min)"
    else:
        best_so_far = np.maximum.accumulate(values)
        best_label = "Best (max)"

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#0a0a0f")

    ax.scatter(iters, values, c="#666", s=15, alpha=0.5, label="All trials")
    ax.plot(iters, best_so_far, color="#b388ff", linewidth=2, label=best_label)
    ax.axhline(best_so_far[-1], color="#82b1ff", linestyle="--",
               linewidth=1, alpha=0.5, label=f"Final: {best_so_far[-1]:.2e}")

    ax.set_xlabel("Iteration", color="#aaa")
    ax.set_ylabel("Objective", color="#aaa")
    ax.set_title(title, color="#ddd", fontsize=14)
    ax.legend(loc="best", facecolor="#1a1a2e", edgecolor="#333",
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


def plot_action_distribution(
    actions: Sequence[Any],
    *,
    action_names: Optional[List[str]] = None,
    title: str = "Action Distribution",
    save_path: Optional[str] = None,
    show: bool = False,
) -> Any:
    """Plot distribution of RL actions taken.

    Parameters
    ----------
    actions : list
        Actions taken (each can be a dict, array, or scalar).
    action_names : list of str, optional
        Names for each action dimension.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Convert to numpy
    if isinstance(actions[0], dict):
        keys = list(actions[0].keys())
        n_dim = len(keys)
        data = {k: np.array([a[k] for a in actions]) for k in keys}
        if action_names is None:
            action_names = keys
    elif isinstance(actions[0], (list, np.ndarray)):
        arr = np.array(actions)
        n_dim = arr.shape[1] if arr.ndim > 1 else 1
        if n_dim == 1:
            data = {"action": arr.flatten()}
        else:
            data = {f"dim_{i}": arr[:, i] for i in range(n_dim)}
        if action_names is None:
            action_names = list(data.keys())
    else:
        data = {"action": np.array(actions)}
        n_dim = 1
        if action_names is None:
            action_names = ["action"]

    ncols = min(3, n_dim)
    nrows = (n_dim + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.patch.set_facecolor("#0a0a0f")

    keys = list(data.keys())
    for idx, key in enumerate(keys):
        ax = axes[idx // ncols][idx % ncols]
        ax.set_facecolor("#0a0a0f")
        vals = data[key].flatten()
        ax.hist(vals, bins=30, color="#b388ff", alpha=0.7, edgecolor="#333")
        ax.set_title(action_names[idx] if idx < len(action_names) else key,
                     color="#ddd", fontsize=10)
        ax.tick_params(colors="#888")
        ax.spines["bottom"].set_color("#333")
        ax.spines["left"].set_color("#333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide empty axes
    for idx in range(len(keys), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(title, color="#ddd", fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    if show:
        plt.show()
    return fig


# ======================================================================
# Agent Evaluation
# ======================================================================

def evaluate_agent(
    agent: Any,
    env: Any,
    *,
    n_episodes: int = 50,
    render: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate a trained agent over multiple episodes.

    Parameters
    ----------
    agent : object
        Must have a ``predict(obs)`` or ``act(obs)`` method,
        or be a dict with a ``model`` key (e.g. from save_agent).
    env : gymnasium-compatible environment
        The RL environment.
    n_episodes : int
        Number of evaluation episodes.
    render : bool
        Whether to render each episode.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Keys: mean_reward, std_reward, max_reward, min_reward,
        rewards (list), episode_infos (list of info dicts).
    """
    rewards = []
    episode_infos = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        done = False
        steps = 0

        while not done:
            # Get action
            if hasattr(agent, "predict"):
                action = agent.predict(obs, deterministic=True)
                if isinstance(action, tuple):
                    action = action[0]
            elif hasattr(agent, "act"):
                action = agent.act(obs)
            elif isinstance(agent, dict) and "model" in agent:
                # Loaded agent
                model = agent["model"]
                action = model.predict(obs, deterministic=True)
                if isinstance(action, tuple):
                    action = action[0]
            else:
                raise ValueError("Agent must have predict() or act() method")

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

            if render:
                env.render()

        rewards.append(total_reward)
        episode_infos.append({
            "episode": ep,
            "reward": total_reward,
            "steps": steps,
            "info": info,
        })

        if verbose and (ep + 1) % max(1, n_episodes // 5) == 0:
            avg = np.mean(rewards[-10:])
            print(f"  Eval {ep+1}/{n_episodes}: avg_reward={avg:.3f}")

    result = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "max_reward": float(np.max(rewards)),
        "min_reward": float(np.min(rewards)),
        "rewards": rewards,
        "episode_infos": episode_infos,
    }

    if verbose:
        print(f"✓ Evaluation: mean={result['mean_reward']:.3f}, "
              f"std={result['std_reward']:.3f}, "
              f"max={result['max_reward']:.3f}")

    return result


# ======================================================================
# Serialization
# ======================================================================

def save_agent(
    agent: Any,
    path: str,
    *,
    metadata: Optional[Dict] = None,
):
    """Save an RL agent to disk.

    Supports both sklearn-style agents and stable-baselines3 agents.

    Parameters
    ----------
    agent : object
        The agent to save.
    path : str
        File path (.pkl or .zip for SB3).
    metadata : dict, optional
        Extra metadata to store alongside the agent.
    """
    path = str(path)

    # Stable-baselines3 agents have their own save method
    if hasattr(agent, "save") and path.endswith(".zip"):
        agent.save(path)
        return

    # Generic pickle
    bundle = {
        "agent": agent,
        "metadata": metadata or {},
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def load_agent(path: str) -> Any:
    """Load an RL agent from disk.

    Parameters
    ----------
    path : str
        File path (.pkl or .zip).

    Returns
    -------
    agent : object
    """
    path = str(path)

    if path.endswith(".zip"):
        # SB3 format
        try:
            from stable_baselines3 import A2C, PPO, SAC, TD3
            # Try each algorithm
            for cls in [A2C, PPO, SAC, TD3]:
                try:
                    return cls.load(path)
                except Exception:
                    continue
            raise ValueError(f"Could not load SB3 agent from {path}")
        except ImportError:
            raise ImportError("stable-baselines3 required for .zip agents")

    with open(path, "rb") as f:
        bundle = pickle.load(f)

    if isinstance(bundle, dict) and "agent" in bundle:
        return bundle
    return bundle


# ======================================================================
# Bayesian Optimization
# ======================================================================

def run_bayesian_optimization(
    objective_fn: Callable,
    param_space: Dict[str, Tuple],
    *,
    n_iter: int = 50,
    n_initial: int = 10,
    random_state: int = 42,
    verbose: bool = True,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run Bayesian optimization on a parametric structure design problem.

    Parameters
    ----------
    objective_fn : callable
        Function that takes a dict of parameters and returns a scalar
        (to minimize). For maximization, return negative value.
    param_space : dict
        Parameter space definition.
        Keys: parameter names.
        Values: (low, high) tuples for continuous, or list for discrete.
    n_iter : int
        Number of optimization iterations.
    n_initial : int
        Number of random initial points.
    random_state : int
        Random seed.
    verbose : bool
        Print progress.
    save_path : str, optional
        Path to save results JSON.

    Returns
    -------
    dict
        Keys: best_params, best_value, all_params, all_values, n_iter.

    Examples
    --------
    >>> def objective(params):
    ...     g = pattern_2d("voronoi", grid=(int(params["grid_x"]), int(params["grid_y"])))
    ...     r = simulate(g, mode="stretch", strain=1.5)
    ...     return r.max_force  # minimize
    >>> space = {"grid_x": (2, 5), "grid_y": (2, 5), "n_internal": (5, 25)}
    >>> result = run_bayesian_optimization(objective, space, n_iter=30)
    >>> print(f"Best: {result['best_params']}, force={result['best_value']:.0f}")
    """
    try:
        from skopt import gp_minimize
        from skopt.space import Real, Integer
    except ImportError:
        raise ImportError("scikit-optimize required: pip install scikit-optimize")

    # Build skopt dimensions
    dimensions = []
    dim_names = []
    for name, spec in param_space.items():
        dim_names.append(name)
        if isinstance(spec, (list, tuple)) and len(spec) == 2:
            low, high = spec
            if isinstance(low, int) and isinstance(high, int):
                dimensions.append(Integer(low, high, name=name))
            else:
                dimensions.append(Real(float(low), float(high), name=name))
        elif isinstance(spec, list):
            # Categorical
            from skopt.space import Categorical
            dimensions.append(Categorical(spec, name=name))
        else:
            raise ValueError(f"Invalid spec for {name}: {spec}")

    def _objective(x):
        params = {dim_names[i]: x[i] for i in range(len(x))}
        try:
            return float(objective_fn(params))
        except Exception as e:
            if verbose:
                print(f"  Warning: objective failed: {e}")
            return 1e10  # Large penalty

    if verbose:
        print(f"Bayesian Optimization: {n_iter} iterations, "
              f"{len(dimensions)} params")

    result = gp_minimize(
        _objective,
        dimensions,
        n_calls=n_iter,
        n_initial_points=n_initial,
        random_state=random_state,
        verbose=verbose,
    )

    best_params = {dim_names[i]: result.x[i] for i in range(len(result.x))}
    all_params = [
        {dim_names[i]: x[i] for i in range(len(x))}
        for x in result.x_iters
    ]

    output = {
        "best_params": best_params,
        "best_value": float(result.fun),
        "all_params": all_params,
        "all_values": [float(v) for v in result.func_vals],
        "n_iter": n_iter,
    }

    if verbose:
        print(f"\n✓ Best: {best_params}")
        print(f"  Value: {output['best_value']:.4e}")

    if save_path:
        # JSON-safe output
        safe_output = {
            "best_params": {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
                           for k, v in best_params.items()},
            "best_value": output["best_value"],
            "all_values": output["all_values"],
            "n_iter": n_iter,
        }
        with open(save_path, "w") as f:
            json.dump(safe_output, f, indent=2)
        print(f"✓ Saved to {save_path}")

    return output
