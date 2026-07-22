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
>>> from fibernet.rl import train_rl, evaluate_rl, plot_reward_curve
>>> from fibernet.rl.env import FiberStructureEnv
>>> env = FiberStructureEnv(unit="honeycomb", grid=(3,3))
>>> model, info = train_rl(env, algorithm="PPO", total_timesteps=10000)
>>> rewards, infos = evaluate_rl(model, env, n_episodes=10)

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

from fibernet.rl.parametric import (
    ParametricStructureEnv,
    create_rl_environment,
)

__all__ = [
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
        "FiberStructureEnv": "fibernet.rl.env",
        "FiberDesignEnv": "fibernet.rl.env",
        "make_env": "fibernet.rl.env",
        # Algorithms
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
        return getattr(mod, name)

    raise AttributeError(f"module 'fibernet.rl' has no attribute '{name}'")
