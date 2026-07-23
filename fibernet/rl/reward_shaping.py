"""
Advanced Reward Shaping for FiberNet RL.

Provides reward functions and wrappers for guiding RL agents:
- CompositeReward: Weighted combination of multiple objectives
- DistanceReward: Reward based on distance to target properties
- PotentialBasedShaping: Potential-based reward shaping (PBRS)
- SparseReward: Sparse rewards with bonus for milestones
- RewardNormalizer: Running statistics normalization
- RewardWrapper: Apply reward shaping to gymnasium environments

Features
--------
- Composable reward functions
- Potential-based shaping for theoretical guarantees
- Reward normalization for training stability
- Milestone-based sparse rewards for long horizons
- Penalty functions for constraint violations

References
----------
- Ng et al., "Policy Invariance Under Reward Transformations" (ICML 1999)
- Article section 5: Reward function design for structure optimization

Examples
--------
>>> from fibernet.rl.reward_shaping import CompositeReward, RewardWrapper
>>> reward_fn = CompositeReward({
...     "force_minimization": lambda info: -info.get("max_force", 0) * 1e-4,
...     "stretch_uniformity": lambda info: -info.get("std_stretch", 0),
...     "compactness": lambda info: info.get("n_edges", 0) * 0.001,
... }, weights={"force_minimization": 1.0, "stretch_uniformity": 2.0, "compactness": 0.5})
>>> reward = reward_fn(info)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


class RewardFunction:
    """Base class for reward functions."""

    def __call__(self, info: Dict[str, Any]) -> float:
        raise NotImplementedError

    def reset(self):
        """Reset internal state (for stateful reward functions)."""
        pass


class CompositeReward(RewardFunction):
    """Weighted combination of multiple reward components.

    Parameters
    ----------
    components : dict
        Named reward functions: {name: callable(info) -> float}.
    weights : dict, optional
        Weights for each component. Default: uniform (1.0).

    Examples
    --------
    >>> reward = CompositeReward({
    ...     "force": lambda i: -i.get("max_force", 0) * 1e-4,
    ...     "stretch": lambda i: -i.get("std_stretch", 0),
    ... })
    >>> r = reward({"max_force": 500, "std_stretch": 0.1})
    """

    def __init__(
        self,
        components: Dict[str, Callable],
        weights: Optional[Dict[str, float]] = None,
    ):
        self.components = components
        self.weights = weights or {name: 1.0 for name in components}

    def __call__(self, info: Dict[str, Any]) -> float:
        total = 0.0
        for name, fn in self.components.items():
            w = self.weights.get(name, 1.0)
            try:
                total += w * fn(info)
            except Exception:
                pass
        return float(total)

    def breakdown(self, info: Dict[str, Any]) -> Dict[str, float]:
        """Get reward breakdown by component."""
        result = {}
        for name, fn in self.components.items():
            w = self.weights.get(name, 1.0)
            try:
                result[name] = w * fn(info)
            except Exception:
                result[name] = 0.0
        result["total"] = sum(result.values())
        return result


class DistanceReward(RewardFunction):
    """Reward based on distance to target properties.

    Parameters
    ----------
    targets : dict
        Target property values: {name: value}.
    info_keys : dict, optional
        Mapping from target names to info dict keys.
    norm : str
        Distance norm: "l1", "l2", "relative".
    scale : float
        Scale factor for reward.
    """

    def __init__(
        self,
        targets: Dict[str, float],
        info_keys: Optional[Dict[str, str]] = None,
        norm: str = "relative",
        scale: float = 1.0,
    ):
        self.targets = targets
        self.info_keys = info_keys or {}
        self.norm = norm
        self.scale = scale

    def __call__(self, info: Dict[str, Any]) -> float:
        total_error = 0.0
        count = 0

        for name, target in self.targets.items():
            key = self.info_keys.get(name, name)
            actual = info.get(key, None)
            if actual is None:
                continue

            if self.norm == "relative":
                error = abs(actual - target) / max(abs(target), 1.0)
            elif self.norm == "l1":
                error = abs(actual - target)
            elif self.norm == "l2":
                error = (actual - target) ** 2
            else:
                error = abs(actual - target)

            total_error += error
            count += 1

        if count == 0:
            return 0.0

        return -self.scale * total_error / count


class PotentialBasedShaping(RewardFunction):
    """Potential-based reward shaping (PBRS).

    Adds F(s, s') = γΦ(s') - Φ(s) to rewards, which preserves
    optimal policies while providing denser reward signal.

    Parameters
    ----------
    potential_fn : callable
        Function(info) -> float computing state potential.
    gamma : float
        Discount factor.
    """

    def __init__(
        self,
        potential_fn: Callable,
        gamma: float = 0.99,
    ):
        self.potential_fn = potential_fn
        self.gamma = gamma
        self._prev_potential: Optional[float] = None

    def __call__(self, info: Dict[str, Any]) -> float:
        current_potential = self.potential_fn(info)

        if self._prev_potential is None:
            shaping = 0.0
        else:
            shaping = self.gamma * current_potential - self._prev_potential

        self._prev_potential = current_potential
        return float(shaping)

    def reset(self):
        self._prev_potential = None


class SparseReward(RewardFunction):
    """Sparse reward with milestone bonuses.

    Parameters
    ----------
    milestones : list of dict
        Each: {"condition": callable(info) -> bool, "reward": float, "name": str}.
    default_reward : float
        Reward when no milestone is reached.
    """

    def __init__(
        self,
        milestones: List[Dict[str, Any]],
        default_reward: float = -0.01,
    ):
        self.milestones = milestones
        self.default_reward = default_reward
        self.reached: set = set()

    def __call__(self, info: Dict[str, Any]) -> float:
        for ms in self.milestones:
            name = ms.get("name", str(id(ms)))
            if name in self.reached:
                continue
            try:
                if ms["condition"](info):
                    self.reached.add(name)
                    return float(ms.get("reward", 1.0))
            except Exception:
                pass
        return self.default_reward

    def reset(self):
        self.reached.clear()


class RewardNormalizer:
    """Running statistics reward normalization.

    Parameters
    ----------
    epsilon : float
        Small constant for numerical stability.
    clip_range : float
        Clip normalized rewards to [-clip_range, clip_range].
    """

    def __init__(self, epsilon: float = 1e-8, clip_range: float = 10.0):
        self.epsilon = epsilon
        self.clip_range = clip_range
        self.count = 0
        self.mean = 0.0
        self.var = 1.0

    def __call__(self, reward: float) -> float:
        """Normalize reward using running statistics."""
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.var += (delta * delta2 - self.var) / max(self.count, 1)

        normalized = (reward - self.mean) / (np.sqrt(self.var) + self.epsilon)
        return float(np.clip(normalized, -self.clip_range, self.clip_range))

    def reset(self):
        self.count = 0
        self.mean = 0.0
        self.var = 1.0


class RewardWrapper:
    """Apply reward shaping to a gymnasium environment.

    Parameters
    ----------
    env : gym.Env
        Base environment.
    reward_fn : RewardFunction
        Custom reward function.
    base_reward_fn : callable, optional
        Function to extract base reward from env info dict.
    normalize : bool
        Apply running normalization.

    Examples
    --------
    >>> env = FiberStructureEnv(unit="honeycomb", grid=(3,3))
    >>> reward_fn = CompositeReward({
    ...     "force": lambda i: -i.get("max_force", 0) * 1e-4,
    ... })
    >>> wrapped = RewardWrapper(env, reward_fn)
    """

    def __init__(
        self,
        env: Any,
        reward_fn: RewardFunction,
        base_reward_fn: Optional[Callable] = None,
        normalize: bool = True,
    ):
        self.env = env
        self.reward_fn = reward_fn
        self.base_reward_fn = base_reward_fn
        self.normalizer = RewardNormalizer() if normalize else None

    def reset(self, **kwargs):
        self.reward_fn.reset()
        if self.normalizer:
            self.normalizer.reset()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Combine base and shaped rewards
        shaped = self.reward_fn(info)
        if self.base_reward_fn:
            base = self.base_reward_fn(info)
        else:
            base = reward

        total_reward = base + shaped

        if self.normalizer:
            total_reward = self.normalizer(total_reward)

        info["original_reward"] = reward
        info["shaped_reward"] = shaped

        return obs, total_reward, terminated, truncated, info

    def __getattr__(self, name):
        return getattr(self.env, name)


# ======================================================================
# Pre-built Reward Functions for Common Tasks
# ======================================================================

def minimize_force_reward(info: Dict) -> float:
    """Minimize maximum force during deformation."""
    return -info.get("max_force", 0) * 1e-4

def maximize_stretch_reward(info: Dict) -> float:
    """Maximize mean stretch ratio."""
    return info.get("mean_stretch", 0)

def uniform_deformation_reward(info: Dict) -> float:
    """Minimize deformation non-uniformity (std of stretch)."""
    return -info.get("std_stretch", 0)

def energy_absorption_reward(info: Dict) -> float:
    """Maximize strain energy absorption."""
    return info.get("strain_energy", 0) * 1e-6

def lightweight_reward(info: Dict) -> float:
    """Reward fewer edges (lighter structure)."""
    n_edges = info.get("n_edges", 1)
    return -n_edges * 0.001

def create_default_reward(mode: str = "minimize_force") -> CompositeReward:
    """Create a default reward function for common objectives.

    Parameters
    ----------
    mode : str
        "minimize_force", "maximize_stretch", "uniform",
        "energy_absorption", "lightweight", "balanced".

    Returns
    -------
    CompositeReward
    """
    if mode == "minimize_force":
        return CompositeReward(
            {"force": minimize_force_reward},
            weights={"force": 1.0},
        )
    elif mode == "maximize_stretch":
        return CompositeReward(
            {"stretch": maximize_stretch_reward},
            weights={"stretch": 1.0},
        )
    elif mode == "uniform":
        return CompositeReward(
            {"uniformity": uniform_deformation_reward},
            weights={"uniformity": 1.0},
        )
    elif mode == "energy_absorption":
        return CompositeReward(
            {"energy": energy_absorption_reward},
            weights={"energy": 1.0},
        )
    elif mode == "lightweight":
        return CompositeReward(
            {"weight": lightweight_reward},
            weights={"weight": 1.0},
        )
    elif mode == "balanced":
        return CompositeReward(
            {
                "force": minimize_force_reward,
                "uniformity": uniform_deformation_reward,
                "weight": lightweight_reward,
            },
            weights={"force": 1.0, "uniformity": 2.0, "weight": 0.5},
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
