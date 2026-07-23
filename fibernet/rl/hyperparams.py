"""
Hyperparameter Optimization for FiberNet RL with Optuna.

Features
--------
- tune_rl_hyperparams: Optuna-based HP search for RL algorithms
- tune_env_params: Optimize environment parameters
- Multi-objective optimization (reward + efficiency)
- Checkpoint/resume for long-running studies
- Study export and visualization helpers

Examples
--------
>>> from fibernet.rl.hyperparams import tune_rl_hyperparams
>>> from fibernet.rl.env import FiberStructureEnv
>>> best_params, study = tune_rl_hyperparams(
...     env_factory=lambda: FiberStructureEnv(unit="honeycomb"),
...     algorithm="PPO",
...     n_trials=20,
...     eval_timesteps=5000,
... )
>>> print(f"Best: reward={study.best_value:.2f}, params={best_params}")
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


def tune_rl_hyperparams(
    env_factory: Callable,
    *,
    algorithm: str = "PPO",
    n_trials: int = 20,
    eval_timesteps: int = 5000,
    eval_episodes: int = 3,
    timeout: Optional[int] = None,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Any]:
    """Tune RL hyperparameters using Optuna.

    Parameters
    ----------
    env_factory : callable
        Function that creates a fresh environment.
    algorithm : str
        SB3 algorithm: "PPO", "SAC", "TD3", "A2C", "DDPG".
    n_trials : int
        Number of Optuna trials.
    eval_timesteps : int
        Training timesteps per trial.
    eval_episodes : int
        Evaluation episodes per trial.
    timeout : int, optional
        Total timeout in seconds.
    study_name : str, optional
        Optuna study name (for persistence).
    storage : str, optional
        Optuna storage URL (e.g., "sqlite:///study.db").
    seed : int
        Random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    best_params : dict
        Best hyperparameters found.
    study : optuna.Study
        Full study object.
    """
    try:
        import optuna
    except ImportError:
        raise ImportError("Optuna required: pip install optuna")

    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=study_name or f"fibernet_{algorithm.lower()}_tuning",
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    def objective(trial):
        from fibernet.rl.algorithms import train_rl, evaluate_rl

        env = env_factory()

        # Algorithm-specific HP search
        kwargs = _suggest_hyperparams(trial, algorithm)

        try:
            model, _ = train_rl(
                env,
                algorithm=algorithm,
                total_timesteps=eval_timesteps,
                verbose=0,
                seed=seed + trial.number,
                **kwargs,
            )

            rewards, _ = evaluate_rl(model, env, n_episodes=eval_episodes)
            return float(np.mean(rewards))

        except Exception as e:
            if verbose:
                print(f"  Trial {trial.number} failed: {e}")
            return float("-inf")

    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best_params = study.best_params
    if verbose:
        print(f"\nBest trial: #{study.best_trial.number}")
        print(f"  Reward: {study.best_value:.4f}")
        print(f"  Params: {best_params}")

    return best_params, study


def tune_env_params(
    env_class: type,
    objective_fn: Callable,
    *,
    param_space: Optional[Dict[str, Any]] = None,
    n_trials: int = 30,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Any]:
    """Tune environment parameters for optimal structure design.

    Parameters
    ----------
    env_class : type
        Environment class (e.g., FiberStructureEnv).
    objective_fn : callable
        Function(env) -> float to maximize.
    param_space : dict, optional
        Parameter search space. Keys: param names.
        Values: {"type": "float"/"int"/"categorical", "low": ..., "high": ..., "choices": ...}
    n_trials : int
    verbose : bool

    Returns
    -------
    best_params, study
    """
    try:
        import optuna
    except ImportError:
        raise ImportError("Optuna required: pip install optuna")

    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    if param_space is None:
        param_space = {
            "target_stretch": {"type": "float", "low": 1.1, "high": 3.0},
            "stiffness": {"type": "float", "low": 1e4, "high": 1e6, "log": True},
            "damping": {"type": "float", "low": 0.1, "high": 0.9},
            "n_pts_per_side": {"type": "int", "low": 2, "high": 8},
        }

    study = optuna.create_study(
        study_name=study_name or "fibernet_env_tuning",
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    def objective(trial):
        env_kwargs = {}
        for name, spec in param_space.items():
            if spec["type"] == "float":
                log = spec.get("log", False)
                env_kwargs[name] = trial.suggest_float(name, spec["low"], spec["high"], log=log)
            elif spec["type"] == "int":
                env_kwargs[name] = trial.suggest_int(name, spec["low"], spec["high"])
            elif spec["type"] == "categorical":
                env_kwargs[name] = trial.suggest_categorical(name, spec["choices"])

        try:
            env = env_class(**env_kwargs)
            score = objective_fn(env)
            return float(score)
        except Exception as e:
            if verbose:
                print(f"  Trial {trial.number} failed: {e}")
            return float("-inf")

    study.optimize(objective, n_trials=n_trials)

    if verbose:
        print(f"\nBest env params: {study.best_params}")
        print(f"  Score: {study.best_value:.4f}")

    return study.best_params, study


def _suggest_hyperparams(trial, algorithm: str) -> Dict[str, Any]:
    """Suggest algorithm-specific hyperparameters."""
    kwargs = {}

    # Common
    kwargs["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    algo = algorithm.upper()

    if algo == "PPO":
        kwargs["n_steps"] = trial.suggest_categorical("n_steps", [128, 256, 512, 1024, 2048])
        kwargs["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        kwargs["n_epochs"] = trial.suggest_int("n_epochs", 3, 20)
        kwargs["gamma"] = trial.suggest_float("gamma", 0.9, 0.9999)
        kwargs["clip_range"] = trial.suggest_float("clip_range", 0.1, 0.4)
        kwargs["ent_coef"] = trial.suggest_float("ent_coef", 1e-4, 0.1, log=True)

    elif algo == "SAC":
        kwargs["buffer_size"] = trial.suggest_int("buffer_size", 10000, 200000)
        kwargs["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        kwargs["tau"] = trial.suggest_float("tau", 0.001, 0.05)
        kwargs["gamma"] = trial.suggest_float("gamma", 0.9, 0.9999)

    elif algo == "TD3":
        kwargs["buffer_size"] = trial.suggest_int("buffer_size", 10000, 200000)
        kwargs["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256])
        kwargs["tau"] = trial.suggest_float("tau", 0.001, 0.05)
        kwargs["gamma"] = trial.suggest_float("gamma", 0.9, 0.9999)
        kwargs["policy_delay"] = trial.suggest_int("policy_delay", 1, 3)

    elif algo == "A2C":
        kwargs["n_steps"] = trial.suggest_categorical("n_steps", [3, 5, 10, 20])
        kwargs["gamma"] = trial.suggest_float("gamma", 0.9, 0.9999)
        kwargs["ent_coef"] = trial.suggest_float("ent_coef", 1e-4, 0.1, log=True)

    elif algo == "DDPG":
        kwargs["buffer_size"] = trial.suggest_int("buffer_size", 10000, 200000)
        kwargs["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256])
        kwargs["tau"] = trial.suggest_float("tau", 0.001, 0.05)
        kwargs["gamma"] = trial.suggest_float("gamma", 0.9, 0.9999)

    return kwargs


def study_summary(study) -> Dict[str, Any]:
    """Summarize an Optuna study.

    Parameters
    ----------
    study : optuna.Study

    Returns
    -------
    dict with statistics.
    """
    df = study.trials_dataframe()
    return {
        "n_trials": len(study.trials),
        "n_completed": len([t for t in study.trials if t.state.name == "COMPLETE"]),
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "mean_value": float(df["value"].mean()),
        "std_value": float(df["value"].std()),
    }
