"""
Reinforcement Learning Tools for FiberNet — Advanced RL Pipeline.

Modules
-------
- env: Gymnasium-compatible environments (FiberStructureEnv, FiberDesignEnv)
- algorithms: SB3 wrappers (PPO, SAC, TD3, A2C, DDPG)
- hyperparams: Optuna-based hyperparameter tuning
- parametric: Parametric structure environment
- curriculum: Curriculum learning with adaptive difficulty
- reward_shaping: Composite rewards, PBRS, normalization
- multi_objective_rl: Multi-objective RL with Pareto exploration
- utils: Visualization, serialization, Bayesian optimization

Quick Start
-----------
>>> import gymnasium as gym
>>> from fibernet.rl import train_policy
>>> env = gym.make("CartPole-v1")
>>> try:
...     model, info = train_policy(env, algorithm="PPO", total_timesteps=10000)
... finally:
...     env.close()

>>> # Reward shaping
>>> from fibernet.rl.reward_shaping import CompositeReward, create_default_reward
>>> reward_fn = create_default_reward("balanced")

>>> # Curriculum learning
>>> from fibernet.rl.curriculum import AdaptiveCurriculum, CurriculumWrapper
>>> curriculum = AdaptiveCurriculum("grid_x", start_value=2, max_value=6)

>>> # Multi-objective
>>> from fibernet.rl.multi_objective_rl import ScalarizedMORL
>>> morl = ScalarizedMORL(objectives={"force": ..., "weight": ...})
"""

from fibernet.rl.utils import (
    plot_reward_curve,
    plot_convergence,
    plot_action_distribution,
    evaluate_agent,
    save_agent,
    load_agent,
    run_bayesian_optimization,
)

# A utility import must not probe Gym or native simulation providers.
# Drop cached aliases on reload; explicit access resolves their real owner.
for _name in ("ParametricStructureEnv", "create_rl_environment"):
    globals().pop(_name, None)
del _name

__all__ = [
    "train_policy",
    # Utils
    "plot_reward_curve",
    "plot_convergence",
    "plot_action_distribution",
    "evaluate_agent",
    "save_agent",
    "load_agent",
    "run_bayesian_optimization",
    # Parametric
    "ParametricStructureEnv",
    "create_rl_environment",
]

# Lazy imports for heavy submodules
def __getattr__(name):
    _submodules = {
        # Environments
        "ParametricStructureEnv": "fibernet.rl.parametric",
        "create_rl_environment": "fibernet.rl.parametric",
        "FiberStructureEnv": "fibernet.rl.env",
        "FiberDesignEnv": "fibernet.rl.env",
        "make_env": "fibernet.rl.env",
        # Algorithms
        "train_policy": "fibernet.rl.algorithms",
        "train_rl": "fibernet.rl.algorithms",
        "evaluate_rl": "fibernet.rl.algorithms",
        "replay_best": "fibernet.rl.algorithms",
        "get_algorithm_info": "fibernet.rl.algorithms",
        # Hyperparameters
        "tune_rl_hyperparams": "fibernet.rl.hyperparams",
        "tune_env_params": "fibernet.rl.hyperparams",
        "study_summary": "fibernet.rl.hyperparams",
        
        # Curriculum
        "CurriculumScheduler": "fibernet.rl.curriculum",
        "LinearCurriculum": "fibernet.rl.curriculum",
        "AdaptiveCurriculum": "fibernet.rl.curriculum",
        "MultiStageCurriculum": "fibernet.rl.curriculum",
        "CurriculumWrapper": "fibernet.rl.curriculum",
        # Reward shaping
        "CompositeReward": "fibernet.rl.reward_shaping",
        "DistanceReward": "fibernet.rl.reward_shaping",
        "PotentialBasedShaping": "fibernet.rl.reward_shaping",
        "SparseReward": "fibernet.rl.reward_shaping",
        "RewardNormalizer": "fibernet.rl.reward_shaping",
        "RewardWrapper": "fibernet.rl.reward_shaping",
        "create_default_reward": "fibernet.rl.reward_shaping",
        # Multi-objective
        "ScalarizedMORL": "fibernet.rl.multi_objective_rl",
        "ParetoFrontExplorer": "fibernet.rl.multi_objective_rl",
    }

    if name in _submodules:
        import importlib
        mod = importlib.import_module(_submodules[name])
        return getattr(mod, "train_rl" if name == "train_policy" else name)

    raise AttributeError(f"module 'fibernet.rl' has no attribute '{name}'")


def __dir__():
    return sorted(set(globals()) | {"ParametricStructureEnv", "create_rl_environment"})
