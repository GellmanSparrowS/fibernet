"""
Gymnasium-Compatible RL Environments for FiberNet.

Provides properly structured environments following the Gymnasium API:
- FiberStructureEnv: Continuous action space for displacement control
- FiberDesignEnv: Discrete + continuous action space for structure design
- FiberTopologyEnv: Multi-step iterative refinement environment

All environments:
- Follow gymnasium.spaces conventions
- Return 5-tuple from step(): (obs, reward, terminated, truncated, info)
- Support seeding and deterministic resets
- Include comprehensive info dicts

Examples
--------
>>> from fibernet.rl.env import FiberStructureEnv, FiberDesignEnv
>>> import gymnasium as gym
>>> env = FiberStructureEnv(unit="honeycomb", grid=(3,3))
>>> obs, info = env.reset(seed=42)
>>> action = env.action_space.sample()
>>> obs, reward, terminated, truncated, info = env.step(action)
>>> print(f"Reward: {reward:.2f}, force: {info['max_force']:.0f}")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

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


def _require_gym():
    if not HAS_GYM:
        raise ImportError("gymnasium required: pip install gymnasium")


if HAS_GYM:
    from fibernet.gen.pattern import pattern_2d, list_units
    from fibernet.sim.accelerated import TaichiEngine

    class FiberStructureEnv(gym.Env):
        """RL environment for parametric fiber structure optimization.

        Continuous action space controlling node displacements on structure edges.
        Properly implements Gymnasium API.

        Parameters
        ----------
        unit : str
            Unit type: "square", "honeycomb", "triangle", "voronoi", etc.
        box : (w, h)
            Cell dimensions.
        grid : (nx, ny)
            Tiling grid.
        n_pts_per_side : int
            Internal points per edge.
        target_stretch : float
            Stretch ratio for evaluation.
        stiffness : float
            Spring stiffness.
        damping : float
            Damping ratio.
        num_steps : int
            Simulation steps.
        displacement_range : float
            Max displacement per axis.
        reward_mode : str
            "minimize_force", "maximize_stretch", "uniform_stretch",
            "target_force" (requires target_force parameter)
        target_force : float, optional
            Target force for "target_force" reward mode.
        reward_scale : float
            Scale factor for rewards.
        unit_kwargs : dict, optional
            Extra kwargs for pattern_2d.
        """

        metadata = {"render_modes": ["human"], "render_fps": 4}

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
            target_force: float = 100.0,
            reward_scale: float = 1e-4,
            unit_kwargs: Optional[Dict] = None,
            render_mode: Optional[str] = None,
        ):
            _require_gym()
            super().__init__()

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
            self.target_force = target_force
            self.reward_scale = reward_scale
            self.unit_kwargs = unit_kwargs or {}
            self.render_mode = render_mode

            self.engine = TaichiEngine()

            # Calculate dimensions
            self.n_sides = self._get_n_sides()
            self.n_disp_points = self.n_sides * n_pts_per_side
            self.n_actions = self.n_disp_points * 2

            # Action space: continuous displacements
            self.action_space = spaces.Box(
                low=-displacement_range,
                high=displacement_range,
                shape=(self.n_actions,),
                dtype=np.float32,
            )

            # Observation space: structure summary
            obs_dim = 12
            self.observation_space = spaces.Box(
                low=-1e6, high=1e6, shape=(obs_dim,), dtype=np.float32,
            )

            self._current_graph = None
            self._step_count = 0
            self._last_result = None

        def _get_n_sides(self) -> int:
            sides_map = {
                "square": 4, "triangle": 3, "hexagon": 6,
                "diamond": 4, "star": 8, "cross": 12,
            }
            if self.unit in sides_map:
                return sides_map[self.unit]
            return 4

        def _get_obs(self) -> np.ndarray:
            """Build observation vector."""
            if self._current_graph is None:
                return np.zeros(12, dtype=np.float32)

            g = self._current_graph
            obs = [
                g.num_nodes, g.num_edges,
                g.num_edges / max(g.num_nodes, 1),
            ]

            # Degree stats
            degrees = [g.degree(n) for n in g.nodes]
            obs.extend([
                np.mean(degrees) if degrees else 0,
                np.std(degrees) if degrees else 0,
                np.max(degrees) if degrees else 0,
            ])

            # Last simulation result
            if self._last_result is not None:
                obs.extend([
                    float(self._last_result.max_force),
                    float(self._last_result.max_stretch),
                    float(self._last_result.mean_stretch),
                    float(self._last_result.std_stretch),
                ])
            else:
                obs.extend([0, 0, 0, 0])

            # Step count (normalized)
            obs.append(float(self._step_count) / 100.0)

            # Displacement magnitude
            obs.append(0.0)

            return np.array(obs, dtype=np.float32)

        def _compute_reward(self, result) -> float:
            """Compute reward from simulation result."""
            if self.reward_mode == "minimize_force":
                return -float(result.max_force) * self.reward_scale
            elif self.reward_mode == "maximize_stretch":
                return float(result.mean_stretch) * self.reward_scale
            elif self.reward_mode == "uniform_stretch":
                return -float(result.std_stretch) * self.reward_scale
            elif self.reward_mode == "target_force":
                error = abs(float(result.max_force) - self.target_force)
                return -error * self.reward_scale
            else:
                return -float(result.max_force) * self.reward_scale

        def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
        ) -> Tuple[np.ndarray, dict]:
            super().reset(seed=seed)
            self._current_graph = None
            self._step_count = 0
            self._last_result = None
            return self._get_obs(), {"step": 0}

        def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
            action = np.asarray(action, dtype=np.float32).flatten()
            assert len(action) == self.n_actions

            # Convert to displacement pairs
            displacements = []
            for i in range(self.n_disp_points):
                dx = float(np.clip(action[2*i], -self.displacement_range, self.displacement_range))
                dy = float(np.clip(action[2*i+1], -self.displacement_range, self.displacement_range))
                displacements.append((dx, dy))

            try:
                g = pattern_2d(
                    unit=self.unit, box=self.box, grid=self.grid,
                    n_pts_per_side=self.n_pts_per_side,
                    point_displacements=displacements,
                    seed=self._step_count + self.np_random.integers(0, 10000),
                    **self.unit_kwargs,
                )

                r = self.engine.stretch_test(
                    g, target_stretch=self.target_stretch,
                    stiffness=self.stiffness, damping=self.damping,
                    num_steps=self.num_steps, save_interval=self.num_steps,
                    auto_steps=False,
                )

                reward = self._compute_reward(r)
                self._current_graph = g
                self._last_result = r

                info = {
                    "max_force": float(r.max_force),
                    "max_stretch": float(r.max_stretch),
                    "mean_stretch": float(r.mean_stretch),
                    "std_stretch": float(r.std_stretch),
                    "n_nodes": g.num_nodes,
                    "n_edges": g.num_edges,
                    "graph": g,
                    "step": self._step_count,
                }

            except Exception as e:
                reward = -10.0
                info = {"error": str(e), "step": self._step_count}

            self._step_count += 1
            terminated = True  # Single-step episode by default
            truncated = False

            return self._get_obs(), reward, terminated, truncated, info

        def render(self):
            if self.render_mode == "human" and self._current_graph is not None:
                from fibernet.viz.render import render_graph
                return render_graph(self._current_graph, theme="dark")

        def close(self):
            pass


    class FiberDesignEnv(gym.Env):
        """RL environment for discrete + continuous structure design.

        Agent chooses unit type, grid size, and beam parameters.
        Multi-step episodes allow iterative refinement.

        Parameters
        ----------
        available_units : list of str
            Unit types the agent can choose from.
        max_grid : int
            Maximum grid size.
        box : (w, h)
            Cell dimensions.
        target_properties : dict
            Target mechanical properties {"max_force": float, ...}
        max_steps : int
            Maximum steps per episode.
        """

        metadata = {"render_modes": ["human"]}

        def __init__(
            self,
            available_units: Optional[List[str]] = None,
            max_grid: int = 8,
            box: Tuple[float, float] = (10.0, 10.0),
            target_properties: Optional[Dict[str, float]] = None,
            max_steps: int = 5,
            reward_scale: float = 1e-4,
            render_mode: Optional[str] = None,
        ):
            _require_gym()
            super().__init__()

            self.available_units = available_units or ["honeycomb", "square", "triangle", "kagome", "reentrant"]
            self.max_grid = max_grid
            self.box = box
            self.target_properties = target_properties or {"max_force": 100.0}
            self.max_steps = max_steps
            self.reward_scale = reward_scale
            self.render_mode = render_mode

            self.engine = TaichiEngine()

            # Action space: Dict
            n_units = len(self.available_units)
            self.action_space = spaces.Dict({
                "unit_idx": spaces.Discrete(n_units),
                "grid_x": spaces.Discrete(max_grid - 1),
                "grid_y": spaces.Discrete(max_grid - 1),
                "radius": spaces.Box(low=0.02, high=0.5, shape=(1,), dtype=np.float32),
                "n_internal": spaces.Discrete(6),
            })

            # Observation
            obs_dim = 15
            self.observation_space = spaces.Box(
                low=-1e6, high=1e6, shape=(obs_dim,), dtype=np.float32,
            )

            self._step_count = 0
            self._current_graph = None
            self._history: List[Dict] = []

        def _get_obs(self) -> np.ndarray:
            obs = np.zeros(15, dtype=np.float32)
            obs[0] = self._step_count / self.max_steps

            if self._history:
                last = self._history[-1]
                obs[1] = last.get("max_force", 0)
                obs[2] = last.get("max_stretch", 0)
                obs[3] = last.get("mean_stretch", 0)
                obs[4] = last.get("std_stretch", 0)
                obs[5] = last.get("n_nodes", 0)
                obs[6] = last.get("n_edges", 0)

            # Target properties (normalized)
            obs[7] = self.target_properties.get("max_force", 0) * self.reward_scale
            obs[8] = self.target_properties.get("max_stretch", 0)
            obs[9] = self.target_properties.get("mean_stretch", 0)

            # Remaining for future use
            obs[10:] = 0
            return obs

        def _compute_reward(self, info: Dict) -> float:
            reward = 0.0
            for key, target in self.target_properties.items():
                actual = info.get(key, 0)
                error = abs(actual - target) / max(abs(target), 1.0)
                reward -= error
            return float(reward)

        def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
        ) -> Tuple[np.ndarray, dict]:
            super().reset(seed=seed)
            self._step_count = 0
            self._current_graph = None
            self._history = []
            return self._get_obs(), {"step": 0}

        def step(self, action: dict) -> Tuple[np.ndarray, float, bool, bool, dict]:
            self._step_count += 1

            unit_name = self.available_units[action["unit_idx"]]
            grid_x = int(action["grid_x"]) + 2
            grid_y = int(action["grid_y"]) + 2
            radius = float(action["radius"][0])
            n_internal = int(action["n_internal"])

            info = {"step": self._step_count}

            try:
                g = pattern_2d(
                    unit=unit_name, box=self.box, grid=(grid_x, grid_y),
                    n_internal=n_internal,
                    seed=self.np_random.integers(0, 10000),
                )

                r = self.engine.stretch_test(
                    g, target_stretch=1.5, stiffness=1e5, damping=0.3,
                    num_steps=1000, save_interval=1000, auto_steps=False,
                )

                info.update({
                    "max_force": float(r.max_force),
                    "max_stretch": float(r.max_stretch),
                    "mean_stretch": float(r.mean_stretch),
                    "std_stretch": float(r.std_stretch),
                    "n_nodes": g.num_nodes,
                    "n_edges": g.num_edges,
                    "unit": unit_name,
                    "grid": (grid_x, grid_y),
                    "graph": g,
                })

                self._current_graph = g
                self._history.append(info)

                reward = self._compute_reward(info)

            except Exception as e:
                reward = -10.0
                info["error"] = str(e)
                self._history.append(info)

            terminated = self._step_count >= self.max_steps
            truncated = False

            return self._get_obs(), reward, terminated, truncated, info

        def render(self):
            if self.render_mode == "human" and self._current_graph is not None:
                from fibernet.viz.render import render_graph
                return render_graph(self._current_graph, theme="dark")

        def close(self):
            pass


    def make_env(
        env_type: str = "structure",
        **kwargs,
    ) -> gym.Env:
        """Factory function for creating FiberNet RL environments.

        Parameters
        ----------
        env_type : str
            "structure" (continuous), "design" (mixed), "parametric" (legacy)

        Returns
        -------
        gym.Env
        """
        if env_type == "structure":
            return FiberStructureEnv(**kwargs)
        elif env_type == "design":
            return FiberDesignEnv(**kwargs)
        elif env_type == "parametric":
            from fibernet.rl.parametric import ParametricStructureEnv
            return ParametricStructureEnv(**kwargs)
        else:
            raise ValueError(f"Unknown env_type: {env_type}. Options: structure, design, parametric")

else:
    class FiberStructureEnv:
        def __init__(self, *a, **kw):
            _require_gym()

    class FiberDesignEnv:
        def __init__(self, *a, **kw):
            _require_gym()

    def make_env(*a, **kw):
        _require_gym()
