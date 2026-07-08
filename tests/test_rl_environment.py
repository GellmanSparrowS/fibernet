"""Tests for RL environment."""

import numpy as np
import pytest

from fibernet.sim.rl_environment import (
    RLEnvConfig,
    FiberNetworkEnv,
    run_rl_optimization,
)
from fibernet.core.material import Material


class TestRLEnvConfig:
    def test_default_config(self):
        config = RLEnvConfig()
        assert config.target_modulus == 5e7
        assert config.generator_type == "reentrant"
    
    def test_custom_config(self):
        config = RLEnvConfig(target_modulus=1e8, target_poisson=-0.5)
        assert config.target_modulus == 1e8
        assert config.target_poisson == -0.5


class TestFiberNetworkEnv:
    def test_reset(self):
        env = FiberNetworkEnv(RLEnvConfig(seed=42))
        obs, info = env.reset()
        assert obs.shape == (4,)
        assert 'params' in info
    
    def test_step(self):
        env = FiberNetworkEnv(RLEnvConfig(seed=42))
        obs, info = env.reset()
        action = np.array([0.1, 0.0, 0.0, 0.0])
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (4,)
        assert isinstance(reward, float)
        assert 'modulus' in info
    
    def test_multiple_steps(self):
        env = FiberNetworkEnv(RLEnvConfig(seed=42, max_steps=10))
        obs, info = env.reset()
        for _ in range(10):
            action = np.random.uniform(-1, 1, size=4)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    
    def test_different_reward_modes(self):
        for mode in ["modulus", "poisson", "multi_objective", "custom"]:
            env = FiberNetworkEnv(RLEnvConfig(seed=42, reward_mode=mode))
            obs, info = env.reset()
            action = np.array([0.1, 0.0, 0.0, 0.0])
            obs, reward, terminated, truncated, info = env.step(action)
            assert isinstance(reward, float)
    
    def test_random_generator(self):
        env = FiberNetworkEnv(RLEnvConfig(
            seed=42, generator_type="random",
            param_bounds={
                'num_fibers': (20, 100),
                'fiber_length': (2.0, 8.0),
                'radius': (0.02, 0.1),
            }
        ))
        obs, info = env.reset()
        action = np.array([0.1, 0.1, 0.1])
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, float)


class TestRLOptimization:
    def test_basic_optimization(self):
        env = FiberNetworkEnv(RLEnvConfig(seed=42, max_steps=5))
        best_params, history = run_rl_optimization(
            env, n_episodes=5, n_steps=5, verbose=False
        )
        assert best_params is not None
        assert len(history) > 0
    
    def test_history_structure(self):
        env = FiberNetworkEnv(RLEnvConfig(seed=42, max_steps=3))
        best_params, history = run_rl_optimization(
            env, n_episodes=3, n_steps=3, verbose=False
        )
        assert len(history) > 0
        assert 'episode' in history[0]
        assert 'reward' in history[0]
        assert 'modulus' in history[0]
