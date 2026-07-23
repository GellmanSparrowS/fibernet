"""
RL Algorithm Wrappers for FiberNet.

Provides easy-to-use wrappers for Stable Baselines3 algorithms:
- train_rl: One-line training with auto-algorithm selection
- evaluate_rl: Evaluation with statistics
- replay_best: Replay best episode and collect structure data
- Algorithm registry with recommended hyperparameters

Supported algorithms: PPO, SAC, TD3, A2C, DDPG

Examples
--------
>>> from fibernet.rl.algorithms import train_rl, evaluate_rl
>>> from fibernet.rl.env import FiberStructureEnv
>>> env = FiberStructureEnv(unit="honeycomb", grid=(3,3))
>>> model, history = train_rl(env, algorithm="PPO", total_timesteps=10000)
>>> rewards, infos = evaluate_rl(model, env, n_episodes=10)
>>> print(f"Mean reward: {np.mean(rewards):.2f}")
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


# Algorithm registry with recommended defaults
ALGORITHM_REGISTRY = {
    "PPO": {
        "class_name": "PPO",
        "default_kwargs": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "clip_range": 0.2,
            "ent_coef": 0.01,
        },
        "description": "Proximal Policy Optimization — good default for most environments",
    },
    "SAC": {
        "class_name": "SAC",
        "default_kwargs": {
            "learning_rate": 3e-4,
            "buffer_size": 100000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
        },
        "description": "Soft Actor-Critic — best for continuous action spaces",
    },
    "TD3": {
        "class_name": "TD3",
        "default_kwargs": {
            "learning_rate": 1e-3,
            "buffer_size": 100000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "policy_delay": 2,
        },
        "description": "Twin Delayed DDPG — robust continuous control",
    },
    "A2C": {
        "class_name": "A2C",
        "default_kwargs": {
            "learning_rate": 7e-4,
            "n_steps": 5,
            "gamma": 0.99,
            "ent_coef": 0.01,
        },
        "description": "Advantage Actor-Critic — fast, simple",
    },
    "DDPG": {
        "class_name": "DDPG",
        "default_kwargs": {
            "learning_rate": 1e-3,
            "buffer_size": 100000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
        },
        "description": "Deep Deterministic Policy Gradient",
    },
}


def train_rl(
    env,
    *,
    algorithm: str = "PPO",
    total_timesteps: int = 10000,
    learning_rate: Optional[float] = None,
    policy: str = "MlpPolicy",
    verbose: int = 1,
    log_interval: int = 10,
    callback: Optional[Any] = None,
    save_path: Optional[str] = None,
    save_freq: Optional[int] = None,
    seed: int = 42,
    **kwargs,
) -> Tuple[Any, Dict[str, Any]]:
    """Train an RL agent on a FiberNet environment.

    Parameters
    ----------
    env : gym.Env
        FiberNet environment (from fibernet.rl.env).
    algorithm : str
        Algorithm name: "PPO", "SAC", "TD3", "A2C", "DDPG".
    total_timesteps : int
        Total training steps.
    learning_rate : float, optional
        Override default learning rate.
    policy : str
        Policy architecture: "MlpPolicy", "CnnPolicy".
    verbose : int
        0=silent, 1=info, 2=debug.
    save_path : str, optional
        Path to save best model (.zip).
    save_freq : int, optional
        Save checkpoint every N steps.
    seed : int
        Random seed.
    **kwargs
        Extra algorithm-specific parameters.

    Returns
    -------
    model : SB3 model
        Trained model.
    info : dict
        Training metadata.
    """
    try:
        import stable_baselines3 as sb3
        from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    except ImportError:
        raise ImportError("stable-baselines3 required: pip install stable-baselines3")

    algo_info = ALGORITHM_REGISTRY.get(algorithm.upper())
    if algo_info is None:
        raise ValueError(f"Unknown algorithm: {algorithm}. Options: {list(ALGORITHM_REGISTRY.keys())}")

    algo_cls = getattr(sb3, algo_info["class_name"])
    algo_kwargs = {**algo_info["default_kwargs"]}

    if learning_rate is not None:
        algo_kwargs["learning_rate"] = learning_rate
    algo_kwargs.update(kwargs)

    # Adjust n_steps for short training
    if "n_steps" in algo_kwargs and total_timesteps < algo_kwargs["n_steps"] * 2:
        algo_kwargs["n_steps"] = max(min(total_timesteps // 4, algo_kwargs["n_steps"]), 32)

    # Adjust buffer_size for short training
    if "buffer_size" in algo_kwargs and total_timesteps < algo_kwargs["buffer_size"]:
        algo_kwargs["buffer_size"] = max(total_timesteps, 1000)

    model = algo_cls(
        policy=policy,
        env=env,
        seed=seed,
        verbose=verbose,
        **algo_kwargs,
    )

    # Callbacks
    callbacks = []
    if save_path and save_freq:
        callbacks.append(CheckpointCallback(
            save_freq=save_freq, save_path=save_path, name_prefix="rl_model",
        ))
    if save_path:
        callbacks.append(sb3.common.callbacks.EvalCallback(
            env, best_model_save_path=save_path,
            log_path=save_path, eval_freq=max(total_timesteps // 10, 100),
            n_eval_episodes=3, deterministic=True,
        ))

    if callback:
        callbacks.append(callback)

    cb = sb3.common.callbacks.CallbackList(callbacks) if callbacks else None

    model.learn(
        total_timesteps=total_timesteps,
        log_interval=log_interval,
        callback=cb,
    )

    # Save final
    if save_path:
        from pathlib import Path
        Path(save_path).mkdir(parents=True, exist_ok=True)
        model.save(f"{save_path}/final_model")

    info = {
        "algorithm": algorithm,
        "total_timesteps": total_timesteps,
        "seed": seed,
        "kwargs": algo_kwargs,
    }

    return model, info


def evaluate_rl(
    model: Any,
    env,
    *,
    n_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
) -> Tuple[List[float], List[Dict]]:
    """Evaluate a trained RL agent.

    Parameters
    ----------
    model : SB3 model
        Trained model.
    env : gym.Env
        Environment to evaluate in.
    n_episodes : int
        Number of evaluation episodes.
    deterministic : bool
        Use deterministic policy.
    render : bool
        Render episodes.

    Returns
    -------
    rewards : list of float
        Per-episode total rewards.
    infos : list of dict
        Per-episode info dicts.
    """
    episode_rewards = []
    episode_infos = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        ep_infos = [info]
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            ep_infos.append(info)
            done = terminated or truncated

            if render:
                env.render()

        episode_rewards.append(episode_reward)
        episode_infos.append(ep_infos[-1])

    mean_r = np.mean(episode_rewards)
    std_r = np.std(episode_rewards)
    print(f"Evaluation: {n_episodes} episodes, "
          f"reward={mean_r:.2f} ± {std_r:.2f}, "
          f"min={np.min(episode_rewards):.2f}, max={np.max(episode_rewards):.2f}")

    return episode_rewards, episode_infos


def replay_best(
    model: Any,
    env,
    *,
    n_steps: int = 1,
) -> List[Dict[str, Any]]:
    """Replay the agent's best policy and collect trajectory data.

    Parameters
    ----------
    model : SB3 model
    env : gym.Env
    n_steps : int
        Number of episodes to replay.

    Returns
    -------
    list of dict
        Trajectory data per step, including graph and simulation results.
    """
    trajectories = []

    for _ in range(n_steps):
        obs, info = env.reset()
        done = False
        step_data = {"initial_info": info, "steps": []}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            step_data["steps"].append({
                "action": action.tolist() if hasattr(action, "tolist") else action,
                "reward": float(reward),
                "info": {k: v for k, v in info.items() if k != "graph"},
            })

            if "graph" in info:
                step_data["final_graph"] = info["graph"]

        trajectories.append(step_data)

    return trajectories


def get_algorithm_info(algorithm: str = "PPO") -> Dict[str, Any]:
    """Get information about an algorithm.

    Parameters
    ----------
    algorithm : str

    Returns
    -------
    dict with class_name, default_kwargs, description.
    """
    algo = algorithm.upper()
    if algo not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown: {algorithm}. Options: {list(ALGORITHM_REGISTRY.keys())}")
    return ALGORITHM_REGISTRY[algo]
