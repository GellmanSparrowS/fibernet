"""
Reinforcement Learning environments for fiber network inverse design.

Provides gymnasium-compatible environments for:
- Metamaterial property optimization
- Structure inverse design
- Multi-objective optimization

Supports both custom RL agents and stable-baselines3 algorithms.

Based on the original zigzag_RL.ipynb implementation, adapted for FiberNet.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass, field
import warnings

from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork
from fibernet.core.material import Material
from fibernet.sim.mechanical import FiberFEM


@dataclass
class RLEnvConfig:
    """Configuration for RL environment."""
    # Target properties
    target_modulus: float = 5e7  # Pa
    target_poisson: float = -0.3
    
    # Generator type
    generator_type: str = "reentrant"  # "reentrant", "random", "honeycomb"
    
    # Parameter bounds
    param_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'angle': (100, 170),
        'grid_x': (3, 6),
        'grid_y': (3, 6),
        'radius': (0.1, 0.35),
    })
    
    # Environment settings
    max_steps: int = 50
    reward_mode: str = "multi_objective"  # "modulus", "poisson", "multi_objective", "custom"
    
    # FEM settings
    segments_per_fiber: int = 3
    strain: float = 0.001
    
    seed: int = 42


class FiberNetworkEnv:
    """
    Reinforcement learning environment for fiber network inverse design.
    
    The environment wraps FiberNet's structure generators and FEM solver,
    allowing RL agents to propose structural parameters and receive
    mechanical property feedback.
    
    Compatible with stable-baselines3 algorithms (PPO, SAC, TD3).
    
    Parameters
    ----------
    config : RLEnvConfig
        Environment configuration.
    material : Material, optional
        Fiber material.
    ml_surrogate : object, optional
        Trained ML model for fast reward estimation.
        Must have predict(X) method returning [log_E, nu].
    
    Examples
    --------
    >>> env = FiberNetworkEnv(RLEnvConfig(target_modulus=1e7, target_poisson=-0.3))
    >>> obs = env.reset()
    >>> action = env.action_space.sample()
    >>> obs, reward, done, info = env.step(action)
    """
    
    def __init__(self, config: Optional[RLEnvConfig] = None,
                 material: Optional[Material] = None,
                 ml_surrogate: Optional[Any] = None):
        self.config = config or RLEnvConfig()
        self.material = material or Material(youngs_modulus=1e9, poissons_ratio=0.3)
        self.ml_surrogate = ml_surrogate
        
        self.rng = np.random.RandomState(self.config.seed)
        
        # Define observation and action spaces
        self.param_keys = list(self.config.param_bounds.keys())
        self.n_params = len(self.param_keys)
        
        # Normalize parameter bounds to [0, 1]
        self._bounds_low = np.array([self.config.param_bounds[k][0] for k in self.param_keys])
        self._bounds_high = np.array([self.config.param_bounds[k][1] for k in self.param_keys])
        self._bounds_range = self._bounds_high - self._bounds_low
        
        # Action space: continuous adjustments to parameters (normalized)
        self.action_low = -0.2 * np.ones(self.n_params)
        self.action_high = 0.2 * np.ones(self.n_params)
        
        # State
        self.current_params = None
        self.current_step = 0
        self.history = []
        
        # Try to import gymnasium for proper spaces
        self._has_gym = False
        try:
            import gymnasium as gym
            from gymnasium import spaces
            self._has_gym = True
            
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(self.n_params,), dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self.n_params,), dtype=np.float32
            )
        except ImportError:
            pass
    
    def _normalize_params(self, params: np.ndarray) -> np.ndarray:
        """Normalize parameters to [0, 1]."""
        return (params - self._bounds_low) / self._bounds_range
    
    def _denormalize_params(self, norm_params: np.ndarray) -> np.ndarray:
        """Denormalize from [0, 1] to actual parameter values."""
        return norm_params * self._bounds_range + self._bounds_low
    
    def _get_obs(self) -> np.ndarray:
        """Get current observation (normalized parameters)."""
        return self._normalize_params(self.current_params).astype(np.float32)
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to random initial state.
        
        Returns
        -------
        observation : np.ndarray
            Normalized parameter vector.
        info : dict
            Additional information.
        """
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        
        # Random initial parameters
        self.current_params = self.rng.uniform(self._bounds_low, self._bounds_high)
        self.current_step = 0
        self.history = []
        
        return self._get_obs(), {'params': self.current_params.copy()}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Apply action and return next state.
        
        Parameters
        ----------
        action : np.ndarray
            Parameter adjustments (normalized, clipped to [-1, 1]).
        
        Returns
        -------
        observation : np.ndarray
        reward : float
        terminated : bool
        truncated : bool
        info : dict
        """
        action = np.clip(action, -1.0, 1.0)
        
        # Scale action to parameter ranges
        scaled_action = action * 0.1 * self._bounds_range
        
        # Update parameters
        new_params = self.current_params + scaled_action
        new_params = np.clip(new_params, self._bounds_low, self._bounds_high)
        self.current_params = new_params
        
        self.current_step += 1
        
        # Evaluate properties
        params_dict = {k: v for k, v in zip(self.param_keys, self.current_params)}
        E_eff, nu_eff = self._evaluate_properties(params_dict)
        
        # Compute reward
        reward = self._compute_reward(E_eff, nu_eff)
        
        # Check termination
        terminated = False
        truncated = self.current_step >= self.config.max_steps
        
        info = {
            'params': params_dict,
            'modulus': E_eff,
            'poisson': nu_eff,
            'step': self.current_step,
        }
        
        self.history.append(info)
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def _evaluate_properties(self, params: Dict) -> Tuple[float, float]:
        """
        Evaluate mechanical properties for given parameters.
        
        Uses ML surrogate if available, otherwise runs FEM.
        """
        if self.ml_surrogate is not None:
            return self._evaluate_with_ml(params)
        else:
            return self._evaluate_with_fem(params)
    
    def _evaluate_with_ml(self, params: Dict) -> Tuple[float, float]:
        """Fast evaluation using ML surrogate."""
        try:
            # Build feature vector
            feat = self._params_to_features(params)
            pred = self.ml_surrogate.predict(feat.reshape(1, -1))[0]
            
            # Assume predict returns [log_E, nu]
            if hasattr(self.ml_surrogate, 'predict_proba'):
                E = 10 ** pred[0]
                nu = pred[1]
            else:
                E = 10 ** pred[0] if len(pred) > 0 else 0
                nu = pred[1] if len(pred) > 1 else 0
            
            return E, nu
        except Exception:
            return self._evaluate_with_fem(params)
    
    def _evaluate_with_fem(self, params: Dict) -> Tuple[float, float]:
        """Full FEM evaluation."""
        try:
            net = self._generate_network(params)
            if net is None or len(net.fibers) == 0:
                return 0.0, 0.0
            
            fem = FiberFEM(net, segments_per_fiber=self.config.segments_per_fiber)
            
            E = fem.effective_modulus(strain=self.config.strain, axis=0)
            
            # Try to compute Poisson's ratio
            try:
                result = fem.apply_uniaxial_strain(self.config.strain, axis=0)
                if result.displacements is not None:
                    # Compute transverse strain
                    u = result.displacements.reshape(-1, 6)
                    pos = fem.node_positions
                    
                    # Average transverse displacement
                    mask_top = pos[:, 0] > np.max(pos[:, 0]) * 0.9
                    mask_bot = pos[:, 0] < np.max(pos[:, 0]) * 0.1
                    
                    if np.any(mask_top) and np.any(mask_bot):
                        trans_strain = (np.mean(u[mask_top, 1]) - np.mean(u[mask_bot, 1])) / \
                                       (np.mean(pos[mask_top, 1]) - np.mean(pos[mask_bot, 1]) + 1e-10)
                        nu = -trans_strain / self.config.strain if self.config.strain > 0 else 0
                    else:
                        nu = 0.0
                else:
                    nu = 0.0
            except Exception:
                nu = 0.0
            
            return E, nu
            
        except Exception:
            return 0.0, 0.0
    
    def _generate_network(self, params: Dict) -> Optional[FiberNetwork]:
        """Generate network from parameters."""
        gen_type = self.config.generator_type
        
        try:
            if gen_type == "reentrant":
                from fibernet.gen.metamaterials import reentrant_honeycomb_2d
                grid_x = int(params.get('grid_x', 4))
                grid_y = int(params.get('grid_y', 4))
                return reentrant_honeycomb_2d(
                    reentrant_angle=params.get('angle', 120),
                    grid_size=(grid_x, grid_y),
                    radius=params.get('radius', 0.2),
                )
            
            elif gen_type == "random":
                from fibernet.gen.disordered import random_straight_2d
                return random_straight_2d(
                    num_fibers=int(params.get('num_fibers', 50)),
                    fiber_length=params.get('fiber_length', 5.0),
                    box_size=(10, 10),
                    radius=params.get("radius", 0.05),
                    material=self.material,
                    seed=self.rng.randint(0, 10000),
                )
            
            elif gen_type == "honeycomb":
                from fibernet.gen.metamaterials import reentrant_honeycomb_2d
                grid_x = int(params.get('grid_x', 4))
                return reentrant_honeycomb_2d(
                    reentrant_angle=120,  # regular honeycomb
                    grid_size=(grid_x, grid_x),
                    radius=params.get('radius', 0.2),
                )
            
            elif gen_type == "field_guided":
                from fibernet.gen.field_guided import field_guided_network, FieldGuidedConfig
                config = FieldGuidedConfig(
                    fiber_count=int(params.get('fiber_count', 100)),
                    field_strength=params.get('field_strength', 0.5),
                    radius=params.get("radius", 0.05),
                    seed=self.rng.randint(0, 10000),
                )
                return field_guided_network(config=config, material=self.material)
            
            else:
                warnings.warn(f"Unknown generator type: {gen_type}")
                return None
                
        except Exception as e:
            warnings.warn(f"Network generation failed: {e}")
            return None
    
    def _params_to_features(self, params: Dict) -> np.ndarray:
        """Convert parameters to feature vector for ML."""
        features = []
        for key in self.param_keys:
            val = params.get(key, 0)
            features.append(val)
            # Add derived features
            if key == 'angle':
                features.extend([np.radians(val), np.cos(np.radians(val)), np.sin(np.radians(val))])
        return np.array(features)
    
    def _compute_reward(self, E_eff: float, nu_eff: float) -> float:
        """
        Compute reward based on current properties and targets.
        
        Supports multiple reward modes:
        - "modulus": minimize |E - E_target|
        - "poisson": minimize |nu - nu_target|
        - "multi_objective": weighted combination
        """
        mode = self.config.reward_mode
        
        if mode == "modulus":
            if E_eff <= 0:
                return -10.0
            return -abs(E_eff - self.config.target_modulus) / self.config.target_modulus
        
        elif mode == "poisson":
            return -abs(nu_eff - self.config.target_poisson)
        
        elif mode == "multi_objective":
            if E_eff <= 0:
                return -10.0
            
            # Stiffness reward
            r_E = -abs(np.log10(E_eff + 1e-10) - np.log10(self.config.target_modulus))
            
            # Auxetic reward
            r_nu = -abs(nu_eff - self.config.target_poisson)
            
            return 0.5 * r_E + 0.5 * r_nu
        
        elif mode == "custom":
            # Default: maximize stiffness and auxetic behavior
            if E_eff <= 0:
                return -10.0
            return 0.5 * np.log10(E_eff) + 0.5 * (-nu_eff)
        
        return 0.0
    
    def render(self):
        """Render current state (text output)."""
        if self.current_params is not None:
            params_dict = {k: v for k, v in zip(self.param_keys, self.current_params)}
            print(f"Step {self.current_step}: {params_dict}")
    
    def get_history(self) -> List[Dict]:
        """Get episode history."""
        return self.history.copy()


def run_rl_optimization(
    env: FiberNetworkEnv,
    n_episodes: int = 100,
    n_steps: int = 20,
    learning_rate: float = 0.1,
    exploration: float = 0.3,
    verbose: bool = True,
) -> Tuple[Dict, List[Dict]]:
    """
    Simple RL optimization with epsilon-greedy exploration.
    
    This is a basic built-in optimizer. For production use,
    consider using stable-baselines3 (PPO, SAC, TD3).
    
    Parameters
    ----------
    env : FiberNetworkEnv
        The RL environment.
    n_episodes : int
        Number of training episodes.
    n_steps : int
        Steps per episode.
    learning_rate : float
        Learning rate for Q-value updates.
    exploration : float
        Initial exploration rate (decays over episodes).
    verbose : bool
        Print progress.
    
    Returns
    -------
    best_params : dict
        Best parameters found.
    history : list of dict
        Training history.
    """
    n_params = env.n_params
    n_actions = 5
    
    # Discrete action space
    from itertools import product
    action_steps = np.linspace(-1, 1, n_actions)
    action_space = list(product(*[action_steps] * n_params))
    q_values = {i: 0.0 for i in range(len(action_space))}
    
    best_reward = -np.inf
    best_params = None
    history = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0
        
        for step in range(n_steps):
            # Epsilon-greedy
            if np.random.random() < exploration:
                action_idx = np.random.randint(len(action_space))
            else:
                action_idx = max(q_values, key=q_values.get)
            
            action = np.array(action_space[action_idx])
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Q-value update
            q_values[action_idx] += learning_rate * (reward - q_values[action_idx])
            
            ep_reward += reward
            
            if reward > best_reward:
                best_reward = reward
                best_params = info['params'].copy()
            
            history.append({
                'episode': ep, 'step': step,
                'reward': reward, 'best_reward': best_reward,
                'modulus': info.get('modulus', 0),
                'poisson': info.get('poisson', 0),
                **info.get('params', {}),
            })
            
            if terminated or truncated:
                break
        
        # Decay exploration
        exploration = max(0.05, exploration * 0.98)
        
        if verbose and (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{n_episodes}: "
                  f"avg_reward={ep_reward/n_steps:.3f}, "
                  f"best={best_reward:.3f}")
    
    return best_params, history
