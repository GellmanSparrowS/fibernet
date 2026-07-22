# Reinforcement Learning

`fibernet.rl` provides parametric structure environments and optimization methods for structure design.

## Core Concept

Structure design as continuous optimization: agent outputs displacement parameters → applied to structure edges → simulate → evaluate → reward.

The action space is `(dx, dy)` displacement pairs for internal points on each unit edge. For square with `n_pts_per_side=5`: 40 continuous parameters (20 pairs).

## Parametric Environment

`ParametricStructureEnv` implements the Gymnasium interface. Wraps structure generation, simulation, and reward computation into a single step.

- **Action**: continuous vector in `[-0.3, 0.3]^(2n)`
- **Observation**: structure features or flattened state
- **Reward**: configurable (default: negative max force)
- **Compatible with**: stable-baselines3, custom agents

## Optimization Methods

| Method | Use Case |
|--------|----------|
| **CEM** (Cross-Entropy Method) | Population-based, no gradient needed. Built-in `CEMOptimizer`. |
| **Bayesian Optimization** | Black-box optimization over mixed parameter spaces. `run_bayesian_optimization()`. |
| **Stable-Baselines3** | Standard RL algorithms (PPO, SAC, A2C, etc.) via Gymnasium env. |

## Visualization

`plot_reward_curve`, `plot_convergence`, `plot_action_distribution` — standard RL analysis plots.

## Agent Persistence

`save_agent()` / `load_agent()` for serialization.

## Reward Design

The reward function is the primary customization point. Common patterns:
- **Force minimization**: reward = -max_force
- **Multi-objective**: weighted combination of force, energy, stretch
- **Constrained**: penalty terms for deformation limits

Custom rewards are passed via the environment constructor.
