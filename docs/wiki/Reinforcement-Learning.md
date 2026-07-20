# Reinforcement Learning

The `fibernet.rl` module provides parametric structure environments and optimization methods for structure design via reinforcement learning and black-box optimization.

## Core Concept

FiberNet's RL interface treats structure design as a continuous optimization problem:

```
Agent outputs action ∈ [-0.3, 0.3]^(2n)
    → displacement parameters applied to structure edges
    → simulate → evaluate → reward
```

The action space is the set of `(dx, dy)` displacement pairs for internal points on each unit edge, providing fine-grained geometric control.

## Parametric Environment

```python
from fibernet.rl import ParametricStructureEnv

env = ParametricStructureEnv(
    unit="square",
    grid=(3, 3),
    n_pts_per_side=5,       # 4 edges x 5 pts = 20 pairs = 40-dim action
    target_stretch=1.5,
)

# Gymnasium interface
obs = env.reset()
action = env.action_space.sample()  # shape: (40,)
obs, reward, terminated, truncated, info = env.step(action)
```

### Action Space

| Parameter | Value |
|-----------|-------|
| Type | `Box(-0.3, 0.3)` |
| Dimension | `2 × n_edges × n_pts_per_side` |
| Interpretation | `(dx, dy)` displacement per internal point |

### Reward Design

Default reward: negative maximum edge force (force minimization objective). Custom reward functions can be provided via the environment constructor.

## Optimization Methods

### CEM (Cross-Entropy Method)

```python
from fibernet.rl import CEMOptimizer

optimizer = CEMOptimizer(
    env,
    pop_size=20,         # population per generation
    elite_frac=0.3,      # top 30% selected
    n_generations=50,
)
result = optimizer.optimize()
```

### Bayesian Optimization

```python
from fibernet.rl import run_bayesian_optimization

result = run_bayesian_optimization(
    objective_fn,
    param_space={
        "grid_x": (2, 5),
        "grid_y": (2, 5),
        "stiffness": (1e4, 1e6),
    },
    n_iter=50,
)
```

### Stable-Baselines3 Integration

```python
from stable_baselines3 import PPO

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
```

## Visualization

```python
from fibernet.rl import (
    plot_reward_curve,
    plot_convergence,
    plot_action_distribution,
)

plot_reward_curve(rewards, window=20, save_path="reward.png")
plot_convergence(objectives, minimize=True, save_path="convergence.png")
plot_action_distribution(actions, save_path="actions.png")
```

## Agent Persistence

```python
from fibernet.rl import save_agent, load_agent

save_agent(agent, "agent.pkl")
agent = load_agent("agent.pkl")
```

## Design Patterns

### Force Minimization

```python
def force_reward_fn(env, result):
    return -result.max_force  # minimize force → maximize reward
```

### Multi-Objective

Combine force, energy, and stretch into composite reward:

```python
def multi_obj_reward(env, result):
    w1, w2, w3 = 0.5, 0.3, 0.2
    return -(w1 * result.max_force + w2 * result.energy + w3 * result.max_stretch)
```

### Constrained Optimization

Penalize structures that exceed deformation limits:

```python
def constrained_reward(env, result):
    reward = -result.max_force
    if result.max_stretch > 2.0:
        reward -= 1e4  # penalty for excessive deformation
    return reward
```
