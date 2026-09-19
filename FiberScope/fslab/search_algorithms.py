"""Search registry and real Stable-Baselines3 adapters, imported only by workers."""
import json
import os
from pathlib import Path
import numpy as np
from .storage import atomic_replace

_COMMON = {'learning_rate': (.0003, .000001, .1, '学习率', 'Learning rate'),
           'gamma': (.99, .1, .9999, '折扣因子', 'Discount'),
           'step_size': (.65, .01, 1., '变形步长', 'Deformation step')}
SEARCH_SPECS = {'cem': ('当前搜索 · CEM', 'Current search · CEM', {})}
for _key in ('PPO', 'A2C', 'DQN', 'SAC', 'TD3', 'DDPG'):
    _params = dict(_COMMON)
    if _key in ('PPO', 'A2C'):
        _params['n_steps'] = (16, 4, 256, '每轮采样步数', 'Rollout steps')
    else:
        _params['learning_starts'] = (8, 1, 500, '预热步数', 'Warm-up steps')
        _params['batch_size'] = (16, 4, 256, '训练批量', 'Batch size')
    SEARCH_SPECS[_key] = (_key + ' 强化学习', _key + ' reinforcement learning', _params)
SEARCH_PLUGINS = {}


def register_search(key, names, parameters, runner):
    if key in SEARCH_SPECS or not callable(runner):
        raise ValueError('search key must be new and runner callable')
    SEARCH_SPECS[key] = (*names, parameters)
    SEARCH_PLUGINS[key] = runner


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    atomic_replace(tmp, path)


def run_rl(algorithm, objective, initial, budget=40, seed=0, parameters=None,
           checkpoint=None, stop_cb=None):
    """Bounded deformation MDP; objective returns a scalar to minimize.

    Checkpoint restores policy and best known state; stochastic trajectories
    after restart need not be bit-identical to an uninterrupted run.
    """
    if algorithm in SEARCH_PLUGINS:
        return SEARCH_PLUGINS[algorithm](objective, initial, budget, seed, parameters, checkpoint, stop_cb)
    import gymnasium as gym
    import stable_baselines3 as sb3
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    torch.set_num_threads(1)
    if algorithm not in SEARCH_SPECS or algorithm == 'cem':
        raise ValueError('unknown RL algorithm')
    params = {k: v[0] for k, v in SEARCH_SPECS[algorithm][2].items()}
    params.update(parameters or {})
    for k, v in params.items():
        if k in SEARCH_SPECS[algorithm][2] and isinstance(SEARCH_SPECS[algorithm][2][k][0], int) and (isinstance(v, bool) or not isinstance(v, int)):
            raise ValueError('parameter must be an integer: ' + k)
        if k not in SEARCH_SPECS[algorithm][2] or not SEARCH_SPECS[algorithm][2][k][1] <= v <= SEARCH_SPECS[algorithm][2][k][2]:
            raise ValueError('invalid search parameter: ' + k)
    initial = np.clip(np.asarray(initial, np.float32), -1, 1)
    if int(budget) < 1 or not 1 <= len(initial) <= 32:
        raise ValueError('search budget/dimension exceeds limits')
    state = dict(best_x=initial.tolist(), best_value=None, evaluations=0)
    folder = Path(checkpoint) if checkpoint else None
    if folder:
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / 'search_state.json').exists():
            state = json.loads((folder / 'search_state.json').read_text(encoding='utf-8'))
    if state['best_value'] is None:
        state.update(best_value=float(objective(initial)), evaluations=1)

    class DeformationEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(-1., 1., initial.shape, dtype=np.float32)
            self.action_space = (gym.spaces.Discrete(len(initial) * 2) if algorithm == 'DQN'
                                 else gym.spaces.Box(-1., 1., initial.shape, dtype=np.float32))
            self.x = initial.copy()
            self.steps = 0

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.x = np.asarray(state['best_x'], np.float32)
            self.steps = 0
            return self.x.copy(), {}

        def step(self, action):
            if state['evaluations'] >= budget or (stop_cb and stop_cb()):
                return self.x.copy(), 0., False, True, {}
            if algorithm == 'DQN':
                change = np.zeros_like(self.x)
                change[int(action) // 2] = 1. if int(action) % 2 else -1.
            else:
                change = np.asarray(action, np.float32)
            self.x = np.clip(self.x + params['step_size'] * change, -1, 1)
            value = float(objective(self.x))
            if not np.isfinite(value):
                raise ValueError('non-finite objective')
            previous = state['best_value']
            state['evaluations'] += 1
            if value < previous:
                state.update(best_x=self.x.tolist(), best_value=value)
            self.steps += 1
            reward = float(np.tanh((previous - value) / max(abs(previous), .001)))
            if folder:
                atomic_json(folder / 'search_state.json', state)
            return self.x.copy(), reward, False, self.steps >= 16, {}

    env = DeformationEnv()
    cls = getattr(sb3, algorithm)
    kwargs = dict(learning_rate=params['learning_rate'], gamma=params['gamma'],
                  seed=int(seed), device='cpu', verbose=0, policy_kwargs=dict(net_arch=[32, 32]))
    if algorithm in ('PPO', 'A2C'):
        kwargs['n_steps'] = params['n_steps']
        if algorithm == 'PPO':
            kwargs.update(batch_size=params['n_steps'], n_epochs=4)
    else:
        kwargs.update(buffer_size=10000, learning_starts=params['learning_starts'],
                      batch_size=params['batch_size'], train_freq=1, gradient_steps=1)
    policy = folder / 'policy.zip' if folder else None
    model = cls.load(str(policy), env=env, device='cpu') if policy and policy.exists() else cls('MlpPolicy', env, **kwargs)
    if folder and hasattr(model, 'load_replay_buffer') and (folder / 'replay.pkl').exists():
        # Only load the application's own local checkpoint, never downloaded files.
        model.load_replay_buffer(str(folder / 'replay.pkl'))

    def save():
        if folder:
            model.save(str(folder / 'policy_next.zip'))
            atomic_replace(folder / 'policy_next.zip', policy)
            if hasattr(model, 'save_replay_buffer'):
                model.save_replay_buffer(str(folder / 'replay_next.pkl'))
                atomic_replace(folder / 'replay_next.pkl', folder / 'replay.pkl')
            atomic_json(folder / 'search_state.json', state)

    class Checkpoint(BaseCallback):
        def _on_step(self):
            if self.n_calls % 16 == 0:
                save()
            return state['evaluations'] < budget and not (stop_cb and stop_cb())

    if state['evaluations'] < budget and not (stop_cb and stop_cb()):
        model.learn(total_timesteps=budget-state['evaluations'], callback=Checkpoint(), reset_num_timesteps=False)
    save()
    return state
