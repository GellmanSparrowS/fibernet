"""Inert RL boundary regressions, not a physics scene or policy training.

The real SimResult class supplies the result contract. Optional providers are
blocked; explicit environment doubles exercise the random-rollout wrapper.
Run with --noconftest and plugin autoload disabled to avoid native test setup.
"""
import importlib
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def modules(monkeypatch):
    for name in ('taichi', 'lammps', 'exudyn', 'stable_baselines3'):
        monkeypatch.setitem(sys.modules, name, None)
    gym = ModuleType('gymnasium')
    gym.spaces = SimpleNamespace(
        Dict=lambda *a, **k: object(), Discrete=lambda *a, **k: object(),
        Box=lambda *a, **k: object())
    monkeypatch.setitem(sys.modules, 'gymnasium', gym)
    dataset = ModuleType('fibernet.ml.dataset')
    dataset._extract_graph_features = lambda graph: {}
    monkeypatch.setitem(sys.modules, 'fibernet.ml.dataset', dataset)
    easy = importlib.import_module('fibernet.easy')
    env = importlib.import_module('fibernet.sim.rl_env')
    accelerated = importlib.import_module('fibernet.sim.accelerated')
    return SimpleNamespace(easy=easy, env=env, result=accelerated.SimResult)


def test_actual_result_contract_has_no_effective_property_reward(modules):
    result = modules.result(energy=2.0, deformed_positions=np.zeros((2, 3)))
    assert result.energy == 2.0
    for missing in ('effective_youngs_modulus', 'effective_poissons_ratio',
                    'strain_energy', 'deformed_graph'):
        assert not hasattr(result, missing)


@pytest.mark.parametrize('config', [{}, {'target_E': 10.0}, {'target_nu': -0.3},
                                    {'default_E': 123, 'applied_strain': 0.02}])
def test_unsupported_environment_rejected_before_any_provider(modules, monkeypatch, config):
    calls = []
    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError('No graph generation or native provider is permitted')
    monkeypatch.setattr(modules.env, 'TaichiEngine', forbidden, raising=False)
    monkeypatch.setattr(modules.env, 'pattern_2d', forbidden, raising=False)
    with pytest.raises(NotImplementedError, match='effective.*Young|E\\*'):
        modules.env.FiberNetworkEnv(**config)
    assert calls == []


def test_root_default_rejects_without_statistics(modules, tmp_path, capsys):
    root_api = importlib.import_module('fibernet')
    assert root_api.train_rl is modules.easy.train_rl
    output = tmp_path / 'statistics.json'
    with pytest.raises(NotImplementedError, match='effective.*Young|E\\*'):
        root_api.train_rl(n_episodes=1, save_path=output)
    assert not output.exists()
    assert 'OK' not in capsys.readouterr().out


class RolloutDouble:
    """Explicit inert double. Rewards are not evaluated mechanical properties."""
    def __init__(self, *, reset_error=None, step_error=None, close_error=None,
                 reset_info=None, step_info=None, reward=2.0, fail_episode=1):
        self.reset_error, self.step_error, self.close_error = reset_error, step_error, close_error
        self.reset_info, self.step_info = reset_info or {}, step_info or {}
        self.reward, self.fail_episode = reward, fail_episode
        self.events = []
        self.episode = 0
        self.action_space = SimpleNamespace(sample=self.sample)

    def reset(self):
        self.events.append('reset')
        self.episode += 1
        if self.reset_error:
            raise self.reset_error
        return np.zeros(1), self.reset_info if self.episode == self.fail_episode else {}

    def sample(self):
        self.events.append('sample')
        return {'inert': True}

    def step(self, action):
        self.events.append('step')
        assert action == {'inert': True}
        if self.episode == self.fail_episode:
            if self.step_error:
                raise self.step_error
            info = self.step_info
        else:
            info = {}
        return np.zeros(1), self.reward, True, False, info

    def close(self):
        self.events.append('close')
        if self.close_error:
            raise self.close_error


def use_double(modules, monkeypatch, env):
    monkeypatch.setattr(modules.env, 'FiberNetworkEnv', lambda **config: env)


@pytest.mark.parametrize('value', [True, False, np.bool_(True), 0, -1, 1.5, '2', None])
def test_invalid_episode_count_rejected_before_environment(modules, monkeypatch, value):
    calls = []
    monkeypatch.setattr(modules.env, 'FiberNetworkEnv', lambda **config: calls.append(config))
    with pytest.raises(ValueError, match='n_episodes.*positive integer'):
        modules.easy.train_rl(n_episodes=value)
    assert calls == []


@pytest.mark.parametrize('phase', ['reset', 'step'])
@pytest.mark.parametrize('close_fails', [False, True])
def test_failures_preserve_original_exception_and_close(modules, monkeypatch, tmp_path, capsys,
                                                       phase, close_fails):
    original = RuntimeError('deliberate ' + phase + ' failure')
    cause = ValueError('original cause')
    original.__cause__ = cause
    env = RolloutDouble(**{phase + '_error': original},
                        close_error=RuntimeError('cleanup failed') if close_fails else None)
    use_double(modules, monkeypatch, env)
    output = tmp_path / 'statistics.json'
    output.write_text('preserve previous output')
    with pytest.raises(RuntimeError) as caught:
        modules.easy.train_rl(n_episodes=2, save_path=output)
    assert caught.value is original
    assert caught.value.__cause__ is cause
    assert env.events.count('close') == 1
    assert output.read_text() == 'preserve previous output'
    assert 'OK' not in capsys.readouterr().out


@pytest.mark.parametrize('phase', ['reset', 'step'])
@pytest.mark.parametrize('failed_episode', [1, 2])
def test_error_info_never_becomes_valid_episode_or_saved_statistics(
        modules, monkeypatch, tmp_path, capsys, phase, failed_episode):
    env = RolloutDouble(**{phase + '_info': {'error': 'effective_youngs_modulus unavailable'}},
                        reward=-10.0, fail_episode=failed_episode)
    use_double(modules, monkeypatch, env)
    output = tmp_path / 'statistics.json'
    with pytest.raises(RuntimeError, match='effective_youngs_modulus unavailable'):
        modules.easy.train_rl(n_episodes=3, save_path=output)
    assert env.events.count('close') == 1
    assert not output.exists()
    assert 'OK' not in capsys.readouterr().out
    assert env.events.count('reset') <= failed_episode


@pytest.mark.parametrize('reward', [float('nan'), float('inf'), float('-inf'), True])
def test_invalid_reward_is_not_success(modules, monkeypatch, tmp_path, capsys, reward):
    env = RolloutDouble(reward=reward)
    use_double(modules, monkeypatch, env)
    output = tmp_path / 'statistics.json'
    with pytest.raises(ValueError, match='reward.*finite'):
        modules.easy.train_rl(n_episodes=1, save_path=output)
    assert env.events.count('close') == 1
    assert not output.exists()
    assert 'OK' not in capsys.readouterr().out


def test_close_failure_prevents_success_and_save(modules, monkeypatch, tmp_path, capsys):
    original = RuntimeError('cleanup failed')
    env = RolloutDouble(close_error=original)
    use_double(modules, monkeypatch, env)
    output = tmp_path / 'statistics.json'
    with pytest.raises(RuntimeError) as caught:
        modules.easy.train_rl(n_episodes=1, save_path=output)
    assert caught.value is original
    assert env.events.count('close') == 1
    assert not output.exists()
    assert 'OK' not in capsys.readouterr().out


def test_save_failure_still_has_closed_environment(modules, monkeypatch, tmp_path, capsys):
    env = RolloutDouble()
    use_double(modules, monkeypatch, env)
    original = OSError('deliberate write failure')
    def fail_save(*args, **kwargs):
        raise original
    monkeypatch.setattr(modules.easy.json, 'dump', fail_save)
    with pytest.raises(OSError) as caught:
        modules.easy.train_rl(n_episodes=1, save_path=tmp_path / 'statistics.json')
    assert caught.value is original
    assert env.events.count('close') == 1
    assert 'OK' not in capsys.readouterr().out


@pytest.mark.parametrize('count', [2, np.int64(2)])
def test_successful_inert_rollout_retains_statistics_schema(modules, monkeypatch, tmp_path, count):
    env = RolloutDouble(reward=2.0)
    use_double(modules, monkeypatch, env)
    output = tmp_path / 'statistics.json'
    stats = modules.easy.train_rl(n_episodes=count, save_path=output)
    assert stats == {'n_episodes': 2, 'final_reward': 2.0, 'mean_reward': 2.0, 'max_reward': 2.0}
    assert json.loads(output.read_text()) == stats
    assert env.events == ['reset', 'sample', 'step', 'reset', 'sample', 'step', 'close']


def test_unconstructed_step_cannot_hide_actual_result_contract_error(modules, monkeypatch):
    # Exercise the former step contract with the real result class. A provider
    # double is prepared, but the repaired boundary must reject before using it.
    calls = []
    result = modules.result(energy=2.0, deformed_positions=np.zeros((2, 3)))
    class ProviderDouble:
        def __init__(self):
            calls.append('construct')
        def stretch_test(self, *args, **kwargs):
            calls.append('stretch_test')
            return result
    monkeypatch.setattr(modules.env, 'TaichiEngine', ProviderDouble, raising=False)
    monkeypatch.setattr(modules.env, 'pattern_2d', lambda **kwargs: object(), raising=False)
    monkeypatch.setattr(modules.env, 'extract_features', lambda graph: {}, raising=False)
    env = object.__new__(modules.env.FiberNetworkEnv)
    env.available_units, env.box = ['square'], (10.0, 10.0)
    env._step_count, env._max_steps, env.multi_step = 0, 1, False
    env._feature_names, env.target_E, env.target_nu = ['E_star', 'nu_star'], 1e6, 0.0
    with pytest.raises(NotImplementedError, match='effective.*Young|E\\*'):
        env.step({'unit_idx': 0, 'grid_x': 0, 'grid_y': 0, 'radius': [0.1]})
    assert calls == []


def test_sample_failure_closes_without_success(modules, monkeypatch, tmp_path, capsys):
    env = RolloutDouble()
    original = RuntimeError('action sampling failed')
    def fail_sample():
        raise original
    env.action_space.sample = fail_sample
    use_double(modules, monkeypatch, env)
    output = tmp_path / 'statistics.json'
    with pytest.raises(RuntimeError) as caught:
        modules.easy.train_rl(n_episodes=1, save_path=output)
    assert caught.value is original
    assert env.events.count('close') == 1
    assert not output.exists()
    assert 'OK' not in capsys.readouterr().out


def test_reset_interrupt_still_closes(modules, monkeypatch):
    original = KeyboardInterrupt('inert interruption')
    env = RolloutDouble(reset_error=original)
    use_double(modules, monkeypatch, env)
    with pytest.raises(KeyboardInterrupt) as caught:
        modules.easy.train_rl(n_episodes=1)
    assert caught.value is original
    assert env.events == ['reset', 'close']


def test_failed_construction_is_preserved(modules, monkeypatch, tmp_path, capsys):
    original = RuntimeError('environment construction failed')
    def fail_constructor(**config):
        raise original
    monkeypatch.setattr(modules.env, 'FiberNetworkEnv', fail_constructor)
    output = tmp_path / 'statistics.json'
    with pytest.raises(RuntimeError) as caught:
        modules.easy.train_rl(n_episodes=1, save_path=output)
    assert caught.value is original
    assert not output.exists()
    assert 'OK' not in capsys.readouterr().out


@pytest.mark.parametrize('existing', [False, True])
@pytest.mark.parametrize('phase', ['serialize', 'fdopen', 'write', 'close', 'replace'])
def test_atomic_statistics_failure_preserves_destination(
        modules, monkeypatch, tmp_path, capsys, existing, phase):
    import os

    env = RolloutDouble()
    use_double(modules, monkeypatch, env)
    destination = tmp_path / 'statistics.json'
    previous = b'{"previous": "keep byte-for-byte"}\n'
    if existing:
        destination.write_bytes(previous)
    original = OSError('deliberate ' + phase + ' failure')
    real_fdopen = os.fdopen

    class StageStream:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self
        def write(self, data):
            if phase == 'write':
                self.stream.write(data[:1])
                raise original
            return self.stream.write(data)
        def __exit__(self, *args):
            self.stream.close()
            if phase == 'close':
                raise original

    def fdopen(fd, *args, **kwargs):
        if phase == 'fdopen':
            raise original
        return StageStream(real_fdopen(fd, *args, **kwargs))

    monkeypatch.setattr(os, 'fdopen', fdopen)
    if phase == 'serialize':
        def fail_dump(data, stream, **kwargs):
            assert kwargs.get('allow_nan') is False
            stream.write('{"incomplete":')
            raise original
        monkeypatch.setattr(modules.easy.json, 'dump', fail_dump)
    elif phase == 'replace':
        def fail_replace(*args, **kwargs):
            raise original
        monkeypatch.setattr(os, 'replace', fail_replace)

    with pytest.raises(OSError) as caught:
        modules.easy.train_rl(n_episodes=1, save_path=destination)
    assert caught.value is original
    assert env.events.count('close') == 1
    assert 'OK' not in capsys.readouterr().out
    if existing:
        assert destination.read_bytes() == previous
        assert list(tmp_path.iterdir()) == [destination]
    else:
        assert not destination.exists()
        assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('existing', [False, True])
def test_atomic_statistics_success_replaces_only_after_stage_close(
        modules, monkeypatch, tmp_path, existing):
    import os
    import stat

    env = RolloutDouble()
    use_double(modules, monkeypatch, env)
    destination = tmp_path / 'statistics.json'
    previous = b'{"previous": true}\n'
    if existing:
        destination.write_bytes(previous)
        destination.chmod(0o640)
    expected_chmod = stat.S_IMODE(destination.stat().st_mode) if existing else 0o600
    streams, replacements, chmods = [], [], []
    real_fdopen, real_replace = os.fdopen, os.replace
    real_chmod = os.chmod
    def fdopen(*args, **kwargs):
        stream = real_fdopen(*args, **kwargs)
        streams.append(stream)
        return stream
    def replace(stage, target):
        from pathlib import Path
        assert streams and streams[-1].closed
        assert env.events[-1] == 'close'
        assert Path(stage).parent == destination.parent
        assert Path(stage) != destination and Path(target) == destination
        if existing:
            assert destination.read_bytes() == previous
        else:
            assert not destination.exists()
        assert json.loads(Path(stage).read_text()) == {
            'n_episodes': 1, 'final_reward': 2.0, 'mean_reward': 2.0, 'max_reward': 2.0}
        replacements.append((stage, target))
        return real_replace(stage, target)
    def chmod(stage, mode):
        from pathlib import Path
        assert streams and streams[-1].closed
        assert env.events[-1] == 'close'
        assert Path(stage).parent == destination.parent and Path(stage) != destination
        assert mode == expected_chmod
        chmods.append((stage, mode))
        return real_chmod(stage, mode)
    monkeypatch.setattr(os, 'fdopen', fdopen)
    monkeypatch.setattr(os, 'replace', replace)
    monkeypatch.setattr(os, 'chmod', chmod)
    result = modules.easy.train_rl(n_episodes=1, save_path=destination)
    assert len(replacements) == 1
    assert chmods == [(replacements[0][0], expected_chmod)]
    assert json.loads(destination.read_text()) == result
    assert list(tmp_path.iterdir()) == [destination]
    if os.name == 'nt':
        # Windows chmod controls only the read-only flag, not POSIX rwx bits.
        assert stat.S_IMODE(destination.stat().st_mode) == 0o666
    else:
        assert stat.S_IMODE(destination.stat().st_mode) == (0o640 if existing else 0o600)
