"""
Reinforcement learning environment for fiber network optimization.

Provides a Gymnasium-compatible environment where an agent designs
fiber structures to achieve target mechanical properties.

Design
------
- **Action space**: Choose unit type, grid size, beam radius, reentrant angle
- **Observation**: Graph features + FEM properties
- **Reward**: Negative distance to target properties (E*, ν*)
- **Episode**: Single step (generate + evaluate) or multi-step (iterative refinement)

Examples
--------
>>> from fibernet.sim.rl_env import FiberNetworkEnv
>>> env = FiberNetworkEnv(target_E=1e6, target_nu=-0.3)
>>> obs, info = env.reset()
>>> action = env.action_space.sample()
>>> obs, reward, terminated, truncated, info = env.step(action)
>>> print(f"Reward: {reward:.3f}, E*={info['E_star']:.2e}, nu*={info['nu_star']:.3f}")
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

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

from fibernet.gen.pattern import pattern_2d, list_units
from fibernet.sim.accelerated import TaichiFEMSolver, SimResult
from fibernet.ml.dataset_v2 import extract_features


class FiberNetworkEnv:
    """RL environment for fiber network structure optimization.

    Parameters
    ----------
    target_E : float
        Target effective Young's modulus (Pa).
    target_nu : float
        Target effective Poisson's ratio.
    box : (w, h)
        Unit cell dimensions.
    max_grid : int
        Maximum grid size allowed.
    radius_range : (min, max)
        Allowed beam radius range.
    available_units : list of str, optional
        Unit types the agent can choose from.
    multi_step : bool
        If True, agent can iteratively refine (max 10 steps).
        If False, single-step (choose structure → evaluate).
    """

    def __init__(
        self,
        target_E: float = 1e6,
        target_nu: float = 0.0,
        box: Tuple[float, float] = (10.0, 10.0),
        max_grid: int = 8,
        radius_range: Tuple[float, float] = (0.02, 0.5),
        available_units: Optional[list] = None,
        multi_step: bool = False,
        default_E: float = 1e9,
        applied_strain: float = 0.01,
    ):
        if not HAS_GYM:
            raise ImportError("gymnasium required: pip install gymnasium")

        self.target_E = target_E
        self.target_nu = target_nu
        self.box = box
        self.max_grid = max_grid
        self.radius_range = radius_range
        self.available_units = available_units or list_units()
        self.multi_step = multi_step
        self.default_E = default_E
        self.applied_strain = applied_strain

        # Action space:
        # [unit_idx (discrete), grid_x (2-max), grid_y (2-max), radius (continuous)]
        n_units = len(self.available_units)
        self.action_space = spaces.Dict({
            "unit_idx": spaces.Discrete(n_units),
            "grid_x": spaces.Discrete(max_grid - 1),  # 2 to max_grid
            "grid_y": spaces.Discrete(max_grid - 1),
            "radius": spaces.Box(low=radius_range[0], high=radius_range[1], shape=(1,)),
        })

        # Observation space: feature vector
        self._feature_names = [
            "n_nodes", "n_edges", "density", "mean_degree", "max_degree",
            "total_length", "mean_edge_length", "bbox_width", "bbox_height",
            "length_density", "mean_radius", "E_star", "nu_star",
        ]
        self.observation_space = spaces.Box(
            low=-1e12, high=1e12, shape=(len(self._feature_names),),
        )

        self._current_graph = None
        self._step_count = 0
        self._max_steps = 10 if multi_step else 1

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment."""
        self._step_count = 0
        self._current_graph = None

        # Return initial observation (empty structure features)
        obs = np.zeros(len(self._feature_names))
        info = {"target_E": self.target_E, "target_nu": self.target_nu}
        return obs, info

    def step(self, action: dict) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step.

        Parameters
        ----------
        action : dict
            Keys: unit_idx, grid_x, grid_y, radius.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        self._step_count += 1

        # Parse action
        unit_name = self.available_units[action["unit_idx"]]
        grid_x = int(action["grid_x"]) + 2
        grid_y = int(action["grid_y"]) + 2
        radius = float(action["radius"][0])

        # Generate structure
        try:
            g = pattern_2d(
                unit=unit_name,
                box=self.box,
                grid=(grid_x, grid_y),
                radius=radius,
                n_internal=4,
            )

            # Run FEM
            fem = TaichiFEMSolver()
            result = fem.uniaxial_tension(strain=self.applied_strain)

            # Extract features
            feat = extract_features(g)
            E_star = result.effective_youngs_modulus
            nu_star = result.effective_poissons_ratio

            # Build observation
            obs = np.array([
                feat.get("n_nodes", 0),
                feat.get("n_edges", 0),
                feat.get("density", 0),
                feat.get("mean_degree", 0),
                feat.get("max_degree", 0),
                feat.get("total_length", 0),
                feat.get("mean_edge_length", 0),
                feat.get("bbox_width", 0),
                feat.get("bbox_height", 0),
                feat.get("length_density", 0),
                feat.get("mean_radius", 0),
                E_star,
                nu_star,
            ])

            # Reward: negative relative distance to targets
            E_err = (E_star - self.target_E) / max(abs(self.target_E), 1.0)
            nu_err = (nu_star - self.target_nu) / max(abs(self.target_nu) + 0.1, 0.1)
            reward = -(E_err**2 + nu_err**2)

            # Bonus for connectivity
            if g.is_connected():
                reward += 0.1

            info = {
                "unit": unit_name,
                "grid": (grid_x, grid_y),
                "radius": radius,
                "E_star": E_star,
                "nu_star": nu_star,
                "strain_energy": result.strain_energy,
                "n_nodes": g.num_nodes,
                "n_edges": g.num_edges,
                "graph": g,
                "deformed_graph": result.deformed_graph,
            }

            self._current_graph = g

        except Exception as exc:
            obs = np.zeros(len(self._feature_names))
            reward = -10.0
            info = {"error": str(exc)}

        terminated = not self.multi_step or self._step_count >= self._max_steps
        truncated = False

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the current structure (returns figure, doesn't display)."""
        if self._current_graph is None:
            return None
        from fibernet.viz.render import render_graph
        return render_graph(self._current_graph, theme="dark", color_by="orientation")

    def close(self):
        """Cleanup."""
        pass
