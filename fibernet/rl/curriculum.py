"""
Curriculum Learning for FiberNet RL — Progressive Difficulty Training.

Implements curriculum strategies for RL training:
- CurriculumScheduler: Gradually increase environment difficulty
- LinearCurriculum: Linear difficulty progression
- AdaptiveCurriculum: Adjust difficulty based on agent performance
- MultiStageCurriculum: Discrete training stages

Features
--------
- Automatic difficulty scheduling based on agent performance
- Integration with Gymnasium environments via wrappers
- Performance-based adaptation (reward threshold triggers)
- Multi-stage training with different objective focuses
- Logging and visualization of curriculum progress

References
----------
- Narvekar et al., "Curriculum Learning for Reinforcement Learning Domains" (JMLR 2020)
- Article section 5: Progressive optimization strategies

Examples
--------
>>> from fibernet.rl.curriculum import AdaptiveCurriculum, CurriculumWrapper
>>> curriculum = AdaptiveCurriculum(
...     difficulty_param="grid",
...     start_value=2, max_value=6,
...     reward_threshold=0.8, min_episodes=50,
... )
>>> env = CurriculumWrapper(base_env, curriculum)
>>> model, info = train_rl(env, algorithm="PPO", total_timesteps=100000)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


class CurriculumScheduler:
    """Base class for curriculum difficulty scheduling.

    Parameters
    ----------
    difficulty_param : str
        Name of the environment parameter controlling difficulty.
    start_value : float
        Starting difficulty value.
    max_value : float
        Maximum difficulty value.
    """

    def __init__(
        self,
        difficulty_param: str = "grid_x",
        start_value: float = 2.0,
        max_value: float = 6.0,
    ):
        self.difficulty_param = difficulty_param
        self.start_value = start_value
        self.max_value = max_value
        self.current_value = start_value
        self.episode_count = 0
        self.history: List[Dict[str, Any]] = []

    def get_env_params(self) -> Dict[str, Any]:
        """Get current environment parameters."""
        return {self.difficulty_param: self.current_value}

    def should_advance(self, episode_reward: float) -> bool:
        """Check if difficulty should increase. Override in subclasses."""
        return False

    def advance(self):
        """Advance to next difficulty level. Override in subclasses."""
        pass

    def step(self, episode_reward: float) -> Dict[str, Any]:
        """Update curriculum based on episode outcome.

        Parameters
        ----------
        episode_reward : float
            Total reward from the episode.

        Returns
        -------
        dict
            Current curriculum state.
        """
        self.episode_count += 1

        if self.should_advance(episode_reward):
            self.advance()

        state = {
            "episode": self.episode_count,
            "difficulty": self.current_value,
            "reward": episode_reward,
        }
        self.history.append(state)
        return state

    def get_progress(self) -> float:
        """Get progress as fraction [0, 1]."""
        range_val = self.max_value - self.start_value
        if range_val <= 0:
            return 1.0
        return min(1.0, (self.current_value - self.start_value) / range_val)


class LinearCurriculum(CurriculumScheduler):
    """Linear difficulty progression based on episode count.

    Parameters
    ----------
    difficulty_param : str
        Environment parameter name.
    start_value : float
        Starting value.
    max_value : float
        Maximum value.
    n_episodes : int
        Total episodes over which to progress.
    """

    def __init__(
        self,
        difficulty_param: str = "grid_x",
        start_value: float = 2.0,
        max_value: float = 6.0,
        n_episodes: int = 1000,
    ):
        super().__init__(difficulty_param, start_value, max_value)
        self.n_episodes = n_episodes
        self.step_size = (max_value - start_value) / max(n_episodes, 1)

    def should_advance(self, episode_reward: float) -> bool:
        return True

    def advance(self):
        self.current_value = min(
            self.current_value + self.step_size, self.max_value
        )


class AdaptiveCurriculum(CurriculumScheduler):
    """Performance-based adaptive difficulty adjustment.

    Increases difficulty when reward exceeds threshold for
    a minimum number of episodes.

    Parameters
    ----------
    difficulty_param : str
    start_value : float
    max_value : float
    step_size : float
        Amount to increase difficulty per advancement.
    reward_threshold : float
        Minimum average reward to trigger advancement.
    min_episodes : int
        Minimum episodes at current level before advancement.
    window_size : int
        Reward averaging window.
    """

    def __init__(
        self,
        difficulty_param: str = "grid_x",
        start_value: float = 2.0,
        max_value: float = 6.0,
        step_size: float = 1.0,
        reward_threshold: float = 0.8,
        min_episodes: int = 50,
        window_size: int = 20,
    ):
        super().__init__(difficulty_param, start_value, max_value)
        self.step_size = step_size
        self.reward_threshold = reward_threshold
        self.min_episodes = min_episodes
        self.window_size = window_size
        self.recent_rewards: List[float] = []
        self.episodes_at_level = 0

    def should_advance(self, episode_reward: float) -> bool:
        self.recent_rewards.append(episode_reward)
        if len(self.recent_rewards) > self.window_size:
            self.recent_rewards.pop(0)
        self.episodes_at_level += 1

        if self.episodes_at_level < self.min_episodes:
            return False
        if self.current_value >= self.max_value:
            return False
        if len(self.recent_rewards) < self.window_size:
            return False

        avg_reward = np.mean(self.recent_rewards[-self.window_size:])
        return avg_reward >= self.reward_threshold

    def advance(self):
        self.current_value = min(self.current_value + self.step_size, self.max_value)
        self.episodes_at_level = 0
        self.recent_rewards.clear()


class MultiStageCurriculum:
    """Multi-stage curriculum with different training objectives.

    Each stage can have different difficulty parameters, reward weights,
    and training durations.

    Parameters
    ----------
    stages : list of dict
        Each stage: {
            "name": str,
            "env_params": dict,
            "reward_weights": dict (optional),
            "n_episodes": int,
            "success_threshold": float (optional),
        }

    Examples
    --------
    >>> curriculum = MultiStageCurriculum([
    ...     {"name": "simple", "env_params": {"grid_x": 2, "grid_y": 2},
    ...      "n_episodes": 500, "success_threshold": 0.5},
    ...     {"name": "medium", "env_params": {"grid_x": 4, "grid_y": 4},
    ...      "n_episodes": 500, "success_threshold": 0.6},
    ...     {"name": "hard", "env_params": {"grid_x": 6, "grid_y": 6},
    ...      "n_episodes": 1000},
    ... ])
    """

    def __init__(self, stages: List[Dict[str, Any]]):
        self.stages = stages
        self.current_stage = 0
        self.episode_count = 0
        self.recent_rewards: List[float] = []
        self.history: List[Dict] = []

    def get_env_params(self) -> Dict[str, Any]:
        return self.stages[self.current_stage].get("env_params", {})

    def get_reward_weights(self) -> Dict[str, float]:
        return self.stages[self.current_stage].get("reward_weights", {})

    def step(self, episode_reward: float) -> Dict[str, Any]:
        self.episode_count += 1
        self.recent_rewards.append(episode_reward)
        if len(self.recent_rewards) > 50:
            self.recent_rewards.pop(0)

        stage = self.stages[self.current_stage]
        max_eps = stage.get("n_episodes", 1000)
        threshold = stage.get("success_threshold", None)

        should_advance = self.episode_count >= max_eps

        if threshold is not None and len(self.recent_rewards) >= 20:
            avg = np.mean(self.recent_rewards[-20:])
            if avg >= threshold and self.episode_count >= max_eps // 2:
                should_advance = True

        result = {
            "stage": self.current_stage,
            "stage_name": stage.get("name", f"stage_{self.current_stage}"),
            "episode": self.episode_count,
            "env_params": self.get_env_params(),
            "should_advance": should_advance,
        }
        self.history.append(result)

        if should_advance and self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            self.episode_count = 0
            self.recent_rewards.clear()

        return result

    @property
    def is_complete(self) -> bool:
        return self.current_stage >= len(self.stages) - 1 and self.episode_count >= self.stages[-1].get("n_episodes", 1000)


class CurriculumWrapper:
    """Wrapper that applies curriculum to a Gymnasium environment.

    Parameters
    ----------
    env_factory : callable
        Function that creates an environment given kwargs.
    curriculum : CurriculumScheduler or MultiStageCurriculum
        Curriculum strategy.

    Examples
    --------
    >>> wrapper = CurriculumWrapper(
    ...     lambda **kw: FiberStructureEnv(unit="honeycomb", **kw),
    ...     AdaptiveCurriculum("grid_x", start_value=2, max_value=6),
    ... )
    >>> env = wrapper.create_env()
    """

    def __init__(
        self,
        env_factory: Callable,
        curriculum: Union[CurriculumScheduler, MultiStageCurriculum],
    ):
        self.env_factory = env_factory
        self.curriculum = curriculum
        self._env = None

    def create_env(self):
        """Create environment with current curriculum parameters."""
        params = self.curriculum.get_env_params()
        self._env = self.env_factory(**params)
        return self._env

    def update_env(self, episode_reward: float):
        """Update curriculum and recreate env if difficulty changed."""
        result = self.curriculum.step(episode_reward)
        new_params = self.curriculum.get_env_params()

        if self._env is not None:
            old_params = getattr(self._env, '_curriculum_params', {})
            if new_params != old_params:
                self._env = self.env_factory(**new_params)

        if self._env is not None:
            self._env._curriculum_params = new_params

        return result

    @property
    def env(self):
        if self._env is None:
            return self.create_env()
        return self._env
