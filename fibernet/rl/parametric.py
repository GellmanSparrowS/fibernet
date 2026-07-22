"""
Parametric Structure Environment for RL.

Wraps FiberNet's pattern_2d with point_displacements for reinforcement learning.
Supports continuous action spaces where each action dimension controls the
(dx, dy) displacement of an internal point on the structure edges.

Examples
--------
>>> from fibernet.rl.parametric import ParametricStructureEnv
>>> env = ParametricStructureEnv(unit="square", grid=(3,3), n_pts_per_side=5)
>>> obs = env.reset()
>>> action = env.action_space.sample()  # continuous displacement vector
>>> graph, sim_result, reward, info = env.step(action)
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, Optional, Tuple

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    try:
        import gym
        from gym import spaces
        HAS_GYM = True
    except ImportError:
        HAS_GYM = False

from fibernet.gen.pattern import pattern_2d
from fibernet.sim.accelerated import TaichiEngine, SimResult
from fibernet.core.structure_graph import StructureGraph


class ParametricStructureEnv:
    """RL environment for parametric fiber structure optimization.

    The action space is a continuous vector of (dx, dy) displacements
    for each internal point on the structure edges. This directly controls
    beam geometry, enabling fine-grained structural optimization.

    Parameters
    ----------
    unit : str
        Unit type: "square", "triangle", "hexagon", "voronoi", etc.
    box : tuple of float
        Cell dimensions (w, h).
    grid : tuple of int
        Tiling grid (nx, ny).
    n_pts_per_side : int
        Internal points per edge (controls action space dimensionality).
    target_stretch : float
        Stretch ratio for evaluation.
    stiffness : float
        Spring stiffness.
    damping : float
        Damping ratio.
    num_steps : int
        Simulation steps.
    displacement_range : float
        Max absolute displacement per axis (default: 0.3).
    reward_mode : str
        "minimize_force": reward = -max_force
        "maximize_stretch": reward = mean_stretch
        "uniform_stretch": reward = -std_stretch (uniform deformation)
    unit_kwargs : dict, optional
        Extra kwargs for pattern_2d (e.g., n_seeds for voronoi).

    Examples
    --------
    >>> env = ParametricStructureEnv(unit="square", grid=(3,3), n_pts_per_side=5)
    >>> print(f"Action space: {env.action_space}")  # Box(40,) = 20 points × 2 (dx,dy)
    >>> obs = env.reset()
    >>> action = np.random.uniform(-0.3, 0.3, env.n_actions)
    >>> graph, result, reward, info = env.step(action)
    >>> print(f"Reward: {reward:.2f}, max_force: {info['max_force']:.0f}")
    """

    def __init__(
        self,
        unit: str = "square",
        box: Tuple[float, float] = (10.0, 10.0),
        grid: Tuple[int, int] = (3, 3),
        n_pts_per_side: int = 5,
        target_stretch: float = 1.5,
        stiffness: float = 1e5,
        damping: float = 0.3,
        num_steps: int = 1000,
        displacement_range: float = 0.3,
        reward_mode: str = "minimize_force",
        unit_kwargs: Optional[Dict] = None,
    ):
        self.unit = unit
        self.box = box
        self.grid = grid
        self.n_pts_per_side = n_pts_per_side
        self.target_stretch = target_stretch
        self.stiffness = stiffness
        self.damping = damping
        self.num_steps = num_steps
        self.displacement_range = displacement_range
        self.reward_mode = reward_mode
        self.unit_kwargs = unit_kwargs or {}

        self.engine = TaichiEngine()

        # Calculate action space dimensionality
        self.n_sides = self._get_n_sides()
        self.n_displacement_pairs = self.n_sides * n_pts_per_side
        self.n_actions = self.n_displacement_pairs * 2  # dx, dy for each point

        if HAS_GYM:
            self.action_space = spaces.Box(
                low=-displacement_range,
                high=displacement_range,
                shape=(self.n_actions,),
                dtype=np.float32,
            )
            self.observation_space = spaces.Box(
                low=-1e6, high=1e6, shape=(10,), dtype=np.float32,
            )

        self._current_graph = None
        self._step_count = 0

    def _get_n_sides(self) -> int:
        """Get number of sides for the unit polygon."""
        sides_map = {
            "square": 4, "triangle": 3, "hexagon": 6,
            "diamond": 4, "star": 8, "cross": 12,
        }
        if self.unit in sides_map:
            return sides_map[self.unit]
        try:
            g = pattern_2d(unit=self.unit, box=self.box, grid=(1, 1),
                          n_pts_per_side=0, seed=0, **self.unit_kwargs)
            return g.num_edges
        except Exception:
            return 4

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment. Returns initial observation."""
        self._step_count = 0
        self._current_graph = None
        return np.zeros(10, dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[StructureGraph, SimResult, float, Dict]:
        """Apply displacement action and evaluate structure.

        Parameters
        ----------
        action : np.ndarray
            Continuous displacement vector of shape (n_actions,).
            Layout: [dx0, dy0, dx1, dy1, ..., dxN, dyN]

        Returns
        -------
        graph : StructureGraph
        result : SimResult
        reward : float
        info : dict
        """
        action = np.asarray(action, dtype=float).flatten()
        assert len(action) == self.n_actions, (
            f"Expected action of length {self.n_actions}, got {len(action)}"
        )

        # Convert action to displacement pairs
        displacements = []
        for i in range(self.n_displacement_pairs):
            dx = float(np.clip(action[2 * i], -self.displacement_range, self.displacement_range))
            dy = float(np.clip(action[2 * i + 1], -self.displacement_range, self.displacement_range))
            displacements.append((dx, dy))

        try:
            g = pattern_2d(
                unit=self.unit, box=self.box, grid=self.grid,
                n_pts_per_side=self.n_pts_per_side,
                point_displacements=displacements,
                seed=self._step_count, **self.unit_kwargs,
            )
        except Exception as e:
            return None, None, -1e6, {"error": str(e), "max_force": 1e10}

        try:
            r = self.engine.stretch_test(
                g, target_stretch=self.target_stretch,
                stiffness=self.stiffness, damping=self.damping,
                num_steps=self.num_steps, save_interval=self.num_steps,
                auto_steps=False,
            )
        except Exception as e:
            return g, None, -1e6, {"error": str(e), "max_force": 1e10}

        if self.reward_mode == "minimize_force":
            reward = -float(r.max_force)
        elif self.reward_mode == "maximize_stretch":
            reward = float(r.mean_stretch)
        elif self.reward_mode == "uniform_stretch":
            reward = -float(r.std_stretch)
        else:
            reward = -float(r.max_force)

        info = {
            "max_force": float(r.max_force),
            "max_stretch": float(r.max_stretch),
            "mean_stretch": float(r.mean_stretch),
            "std_stretch": float(r.std_stretch),
            "n_nodes": g.num_nodes,
            "n_edges": g.num_edges,
            "displacements": displacements,
        }

        self._current_graph = g
        self._step_count += 1

        return g, r, reward, info


def create_rl_environment(
    unit: str = "square",
    grid: Tuple[int, int] = (3, 3),
    n_pts_per_side: int = 5,
    reward_mode: str = "minimize_force",
    **kwargs,
) -> ParametricStructureEnv:
    """Convenience function to create a parametric RL environment.

    Examples
    --------
    >>> env = create_rl_environment(unit="voronoi", grid=(3,3), n_pts_per_side=3)
    >>> action = np.random.uniform(-0.3, 0.3, env.n_actions)
    >>> g, r, reward, info = env.step(action)
    """
    return ParametricStructureEnv(
        unit=unit, grid=grid, n_pts_per_side=n_pts_per_side,
        reward_mode=reward_mode, **kwargs,
    )
