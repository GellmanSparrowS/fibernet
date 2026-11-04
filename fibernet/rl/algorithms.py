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

import warnings
from inspect import getattr_static
from types import GetSetDescriptorType
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


# SB3 2.0/2.8/current documented action families, not a training guarantee.
# Keep names here so importing this module does not import optional backends.
_SB3_ACTION_SPACE_NAMES = {
    "PPO": ("Box", "Discrete", "MultiDiscrete", "MultiBinary"),
    "A2C": ("Box", "Discrete", "MultiDiscrete", "MultiBinary"),
    "SAC": ("Box",),
    "TD3": ("Box",),
    "DDPG": ("Box",),
}


def _check_sb3_compatibility(env, algorithm: str, *, role: str) -> str:
    """Check nominal env types and stored action spaces, without user getters.

    Return ``checked`` for a supported stored action family or ``deferred``
    for a training string ID or dynamic/descriptor/missing space metadata.
    Neither result certifies observations, runtime protocol or successful fit.
    """
    if issubclass(type(env), str):
        if role == "env":
            # SB3 makes registered training environments itself. Do not create
            # one merely to inspect its spaces; EvalCallback has no such route.
            return "deferred"
        raise ValueError(
            "eval_env must be an independent environment object, not a string ID."
        )

    from gymnasium import Env, spaces

    if not issubclass(type(env), Env):
        from stable_baselines3.common.vec_env import VecEnv

        if not issubclass(type(env), VecEnv):
            try:
                import gym
            except ImportError:
                gym = None
            if gym is None or not issubclass(type(env), gym.Env):
                raise ValueError(
                    f"{role} must be a Gymnasium Env, an SB3 VecEnv, or an "
                    "OpenAI Gym Env supported by SB3's Shimmy compatibility. "
                    "FiberNet's legacy ParametricStructureEnv returns "
                    "(graph, result, reward, info), not Gym transitions; "
                    "its protocol is not converted by train_rl."
                )
            # Check the original family, but do not construct a compatibility
            # wrapper or convert any space. SB3 retains that responsibility.
            spaces = gym.spaces

    missing = object()
    action_space = getattr_static(env, "action_space", missing)
    if action_space is missing or getattr_static(
        type(action_space), "__get__", missing
    ) is not missing:
        # Includes standard Gym Wrapper.action_space delegation. Do not invoke
        # properties, unwrap environments, inspect workers or guess metadata.
        return "deferred"

    names = _SB3_ACTION_SPACE_NAMES[algorithm]
    allowed = tuple(getattr(spaces, name) for name in names)
    if not issubclass(type(action_space), allowed):
        raise ValueError(
            f"{algorithm} does not support {role}.action_space of type "
            f"{type(action_space).__name__}; supported action families: "
            f"{', '.join(names)}. Mixed Dict actions require an explicit, "
            "separately validated action adapter; train_rl does not flatten "
            "or remap actions."
        )
    return "checked"


def _check_eval_env_isolation(env, eval_env) -> None:
    """Reject known local aliases, without stepping or probing remote workers."""
    message = "eval_env must not share an environment object with the training env."
    if env is eval_env:
        raise ValueError(message)

    # Optional dependencies are loaded only for explicitly requested evaluation.
    from gymnasium import Wrapper
    from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper

    def object_ids(root):
        pending = [root]
        seen = set()
        references = 1
        while pending:
            current = pending.pop()
            identity = id(current)
            if identity in seen:
                continue
            seen.add(identity)
            current_type = type(current)
            if issubclass(current_type, Wrapper):
                link = "env"
            elif issubclass(current_type, VecEnvWrapper):
                link = "venv"
            elif issubclass(current_type, DummyVecEnv):
                link = "envs"
            else:
                continue

            # Read known stored links, never unwrapped/get_attr/env_method or
            # arbitrary properties. Custom delegation and external resources
            # still require caller-managed isolation.
            for cls in type.__getattribute__(current_type, "__mro__"):
                namespace = type.__getattribute__(cls, "__dict__")
                if "__dict__" in namespace:
                    descriptor = namespace["__dict__"]
                    if type(descriptor) is not GetSetDescriptorType:
                        raise ValueError(
                            "Cannot inspect a known environment wrapper with "
                            "custom __dict__ storage."
                        )
                    state = descriptor.__get__(current, current_type)
                    break
            else:
                raise ValueError("Cannot inspect known environment wrapper storage.")
            if type(state) is not dict:
                raise ValueError(
                    "Cannot inspect a known environment wrapper with "
                    "custom __dict__ storage."
                )
            child = state.get(link)
            if child is None or (
                link == "envs" and (
                    type(child) not in (list, tuple) or not child
                )
            ):
                raise ValueError(
                    f"Cannot inspect known environment wrapper link {link!r}."
                )
            children = child if link == "envs" else (child,)
            references += len(children)
            if references > 1024:
                raise ValueError(
                    "Known environment graph exceeds the inspection limit "
                    "of 1024 object references."
                )
            if any(item is None for item in children):
                raise ValueError(
                    f"Cannot inspect known environment wrapper link {link!r}."
                )
            pending.extend(children)
        return seen

    if object_ids(env) & object_ids(eval_env):
        raise ValueError(message)


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
    eval_env: Optional[Any] = None,
    seed: int = 42,
    **kwargs,
) -> Tuple[Any, Dict[str, Any]]:
    """Train an RL agent on a FiberNet environment.

    Parameters
    ----------
    env : Gymnasium Env, SB3 VecEnv, OpenAI Gym Env, or str
        Training environment or registered training environment ID. Legacy Gym
        compatibility remains SB3/Shimmy's responsibility. FiberNet's legacy
        ParametricStructureEnv protocol is not accepted or converted.
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
        Output directory for the final model and optional checkpoints.
        Best-model and evaluation logs require an independent eval_env.
    save_freq : int, optional
        Save checkpoint every N callback calls when save_path is supplied.
    eval_env : gym.Env or SB3 VecEnv, optional
        Independent evaluation environment. Enables automatic evaluation even
        without save_path; then no automatic best-model/evaluation-file output
        is requested. User callbacks or SB3 loggers may still write elsewhere.
        Direct aliases and shared known in-process wrapper/member objects are
        rejected before model construction. Remote resources and custom wrapper
        delegation cannot be proven independent; isolation is the caller's duty.
        This function does not clone or close either environment.
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

    Notes
    -----
    A save_path without eval_env warns and disables automatic evaluation, while
    retaining checkpoint/final saves. It no longer evaluates on the training
    environment. Known Gymnasium Wrapper, SB3 VecEnvWrapper and DummyVecEnv
    links are inspected with a limit of 1024 object references per environment.
    See docs/architecture/rl-evaluation-isolation.md for migration and limits.
    A passive preflight checks nominal environment types and stored action-space
    families before construction. PPO/A2C support Box, Discrete, MultiDiscrete
    and MultiBinary; SAC/TD3/DDPG support Box. Mixed Dict actions (including
    FiberDesignEnv's) are not directly supported. String training IDs and
    descriptor/delegated spaces defer space validation to SB3 without creating
    environments or invoking getters. This is not complete validation of
    observations, policies, shapes, bounds, runtime protocols or dependencies.
    See docs/architecture/rl-capability-preflight.md for the precise boundary.
    """
    try:
        import stable_baselines3 as sb3
        from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    except ImportError:
        raise ImportError("stable-baselines3 required: pip install stable-baselines3")

    if eval_env is not None:
        _check_eval_env_isolation(env, eval_env)

    algo_info = ALGORITHM_REGISTRY.get(algorithm.upper())
    if algo_info is None:
        raise ValueError(f"Unknown algorithm: {algorithm}. Options: {list(ALGORITHM_REGISTRY.keys())}")

    _check_sb3_compatibility(env, algorithm.upper(), role="env")
    if eval_env is not None:
        _check_sb3_compatibility(eval_env, algorithm.upper(), role="eval_env")

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

    if save_path and eval_env is None:
        warnings.warn(
            "Automatic evaluation and best-model saving are disabled because "
            "eval_env was not supplied. Pass an independent eval_env to enable "
            "them; checkpoint and final-model saving remain enabled.",
            UserWarning,
            stacklevel=2,
        )

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
    if eval_env is not None:
        eval_path = save_path if save_path else None
        callbacks.append(EvalCallback(
            eval_env, best_model_save_path=eval_path,
            log_path=eval_path, eval_freq=max(total_timesteps // 10, 100),
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
