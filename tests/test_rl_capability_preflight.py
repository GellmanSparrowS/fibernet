"""Inert nominal-family/routing checks, not genuine Gym or SB3 execution.

No model is constructed: allowed train_rl routes stop at a sentinel callable.
No reset/step, training, prediction, physics provider or optional package import.
Use --noconftest and PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 for isolated execution.
"""

import ast
from pathlib import Path
import sys
import typing
from types import ModuleType, SimpleNamespace

import pytest

from test_rl_evaluation_isolation import (
    Box, Dict, Discrete, Env, MultiBinary, MultiDiscrete, Space, Tuple,
    UntouchedEnv, VecEnv, fake_sb3, load_algorithms,
)


class ConstructionBoundary(Exception):
    pass


@pytest.fixture
def subject(fake_sb3, monkeypatch):
    module = load_algorithms()
    attempted = []

    def stop_before_construction(**kwargs):
        attempted.append(kwargs)
        raise ConstructionBoundary

    for name in module.ALGORITHM_REGISTRY:
        monkeypatch.setattr(sys.modules['stable_baselines3'], name, stop_before_construction)
    return SimpleNamespace(module=module, attempted=attempted, events=fake_sb3.events)


def poisoned_env(space):
    env = UntouchedEnv()
    env.action_space = space
    return env


def assert_no_effects(subject, path):
    assert subject.attempted == []
    assert subject.events == []
    assert not path.exists()


@pytest.mark.parametrize('algorithm', ['PPO', 'A2C', 'SAC', 'TD3', 'DDPG'])
@pytest.mark.parametrize('family', [Box, Discrete, MultiDiscrete, MultiBinary, Dict, Tuple])
@pytest.mark.parametrize('custom_subclass', [False, True])
def test_all_action_matrix_cells_preserve_subclasses(subject, tmp_path, algorithm, family, custom_subclass):
    cls = type('CustomAction', (family,), {}) if custom_subclass else family
    env = poisoned_env(cls())
    path = tmp_path / 'must_not_exist'
    allowed = family is Box or (algorithm in ('PPO', 'A2C') and family in (Discrete, MultiDiscrete, MultiBinary))
    if allowed:
        with pytest.raises(ConstructionBoundary):
            subject.module.train_rl(env, algorithm=algorithm, eval_env=poisoned_env(cls()), save_path=str(path))
        assert subject.attempted[0]['env'] is env
        assert subject.events == []
        assert not path.exists()
    else:
        with pytest.raises(ValueError, match=rf'{algorithm} does not support env.action_space'):
            subject.module.train_rl(env, algorithm=algorithm, save_path=str(path), save_freq=1)
        assert_no_effects(subject, path)


def test_name_lookalike_is_not_a_space_family(subject, tmp_path):
    lookalike = type('Box', (), {})()
    with pytest.raises(ValueError, match='does not support env.action_space'):
        subject.module.train_rl(poisoned_env(lookalike), save_path=str(tmp_path / 'unused'))
    assert_no_effects(subject, tmp_path / 'unused')


@pytest.mark.parametrize('algorithm,space_class', [('PPO', Dict), ('SAC', Discrete)])
def test_invalid_evaluation_space_precedes_model_and_callbacks(subject, tmp_path, algorithm, space_class):
    path = tmp_path / 'unused'
    with pytest.raises(ValueError, match='does not support eval_env.action_space'):
        subject.module.train_rl(poisoned_env(Box()), algorithm=algorithm,
                                eval_env=poisoned_env(space_class()), save_path=str(path), save_freq=1)
    assert_no_effects(subject, path)


def test_alias_guard_still_precedes_space_validation(subject, tmp_path):
    env = poisoned_env(Dict())
    path = tmp_path / 'unused'
    with pytest.raises(ValueError, match='must not share'):
        subject.module.train_rl(env, eval_env=env, save_path=str(path))
    assert_no_effects(subject, path)


@pytest.mark.parametrize('policy', ['MlpPolicy', 'CnnPolicy', 'MultiInputPolicy', type('CustomPolicy', (), {})])
def test_policy_and_observation_are_not_restricted_or_probed(subject, policy):
    class CustomEnv(UntouchedEnv):
        @property
        def observation_space(self):
            raise AssertionError('This preflight does not inspect observations')
    env = CustomEnv()
    with pytest.raises(ConstructionBoundary):
        subject.module.train_rl(env, policy=policy)
    assert subject.attempted[0]['policy'] is policy
    assert subject.attempted[0]['env'] is env


def test_string_training_id_is_deferred_without_creation(subject):
    env = 'CallerRegistered-v1'
    assert subject.module._check_sb3_compatibility(env, 'PPO', role='env') == 'deferred'
    with pytest.raises(ConstructionBoundary):
        subject.module.train_rl(env)
    assert subject.attempted[0]['env'] is env
    assert subject.events == []


def test_string_evaluation_id_is_rejected(subject, tmp_path):
    path = tmp_path / 'unused'
    with pytest.raises(ValueError, match='independent environment object'):
        subject.module.train_rl(poisoned_env(Box()), eval_env='CallerRegistered-v1', save_path=str(path))
    assert_no_effects(subject, path)


@pytest.mark.parametrize('metadata', ['property', 'missing', 'custom_getattr', 'slot'])
def test_dynamic_spaces_are_deferred_without_getter_execution(subject, metadata):
    def forbidden(*args):
        raise AssertionError('Dynamic space metadata must not be invoked')
    namespace = {'__getattr__': forbidden}
    if metadata == 'property':
        namespace['action_space'] = property(forbidden)
    elif metadata == 'slot':
        namespace['__slots__'] = ('action_space',)
    cls = type('DynamicEnv', (Env,), namespace)
    env = cls()
    assert subject.module._check_sb3_compatibility(env, 'PPO', role='env') == 'deferred'
    with pytest.raises(ConstructionBoundary):
        subject.module.train_rl(env)
    assert subject.events == []


def test_wrapper_property_deferral_is_not_claimed_checked(subject, fake_sb3):
    class DelegatingWrapper(fake_sb3.Wrapper):
        @property
        def action_space(self):
            raise AssertionError('Even known wrapper property is not invoked')
    env = DelegatingWrapper(poisoned_env(Dict()))
    assert subject.module._check_sb3_compatibility(env, 'PPO', role='env') == 'deferred'


@pytest.mark.parametrize('family,algorithm,allowed', [(Box, 'SAC', True), (Discrete, 'PPO', True),
                                                      (MultiDiscrete, 'A2C', True), (MultiBinary, 'PPO', True),
                                                      (Dict, 'PPO', False), (Discrete, 'TD3', False)])
def test_nominal_vecenv_outer_spaces_no_worker_probes(subject, family, algorithm, allowed):
    class CustomVec(VecEnv):
        def get_attr(self, *args):
            raise AssertionError('No worker attributes')
        def env_method(self, *args):
            raise AssertionError('No worker methods')
        def __getattr__(self, name):
            raise AssertionError(f'No vector introspection: {name}')
        @property
        def __class__(self):
            raise AssertionError('Use actual vector type')
    env = CustomVec()
    env.action_space = family()
    if allowed:
        assert subject.module._check_sb3_compatibility(env, algorithm, role='env') == 'checked'
    else:
        with pytest.raises(ValueError, match='does not support env.action_space'):
            subject.module._check_sb3_compatibility(env, algorithm, role='env')
    assert subject.attempted == subject.events == []


@pytest.fixture
def legacy_gym(monkeypatch):
    legacy = ModuleType('gym')
    legacy.Env = type('LegacyEnv', (), {})
    legacy.Space = type('LegacySpace', (), {})
    legacy.spaces = SimpleNamespace(**{
        name: type(name, (legacy.Space,), {})
        for name in ('Box', 'Discrete', 'MultiDiscrete', 'MultiBinary', 'Dict', 'Tuple')
    })
    monkeypatch.setitem(sys.modules, 'gym', legacy)
    return legacy


@pytest.mark.parametrize('algorithm', ['PPO', 'A2C', 'SAC', 'TD3', 'DDPG'])
@pytest.mark.parametrize('name', ['Box', 'Discrete', 'MultiDiscrete', 'MultiBinary', 'Dict', 'Tuple'])
def test_raw_gym_subclasses_use_original_families_without_conversion(subject, legacy_gym, algorithm, name):
    class CustomLegacy(legacy_gym.Env):
        @property
        def unwrapped(self):
            raise AssertionError('Shimmy protocol inspection must not run')
        def reset(self, *args, **kwargs):
            raise AssertionError('No reset')
        def step(self, *args, **kwargs):
            raise AssertionError('No step')
    env = CustomLegacy()
    env.action_space = type('CustomLegacySpace', (getattr(legacy_gym.spaces, name),), {})()
    original = env.action_space
    allowed = name == 'Box' or (algorithm in ('PPO', 'A2C') and name in ('Discrete', 'MultiDiscrete', 'MultiBinary'))
    if allowed:
        assert subject.module._check_sb3_compatibility(env, algorithm, role='env') == 'checked'
    else:
        with pytest.raises(ValueError, match='does not support env.action_space'):
            subject.module._check_sb3_compatibility(env, algorithm, role='env')
    assert env.action_space is original
    assert subject.attempted == subject.events == []


def test_non_gym_legacy_protocol_rejected_before_getters(subject, legacy_gym, tmp_path):
    class LegacyParametricProtocol:
        @property
        def action_space(self):
            raise AssertionError('Non-Gym legacy metadata must not be probed')
        @property
        def __class__(self):
            raise AssertionError('No masquerading classes')
    path = tmp_path / 'unused'
    with pytest.raises(ValueError, match='legacy ParametricStructureEnv'):
        subject.module.train_rl(LegacyParametricProtocol(), save_path=str(path))
    assert_no_effects(subject, path)


def test_factory_signature_and_legacy_branch_remain_source_only():
    source = Path(__file__).resolve().parents[1] / 'fibernet' / 'rl' / 'env.py'
    tree = ast.parse(source.read_text())
    factory = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'make_env')
    assert ast.unparse(factory.returns) == 'Any'
    assert 'step ->' in ast.get_docstring(factory)
    calls = [n for n in ast.walk(factory) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == 'ParametricStructureEnv']
    assert len(calls) == 1
    assert [k.arg for k in calls[0].keywords] == ['geometry_policy', None]
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr in ('reset', 'step') for n in ast.walk(factory))


def test_factory_return_hints_resolve_without_legacy_import_or_factory_call():
    source = Path(__file__).resolve().parents[1] / 'fibernet' / 'rl' / 'env.py'
    tree = ast.parse(source.read_text())
    factory = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'make_env')
    future = next(n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == '__future__')
    # Execute only the actual factory definition, never its body or the module's
    # provider imports. Use the source's postponed annotations and harmless Gym.
    harmless_gym = SimpleNamespace(Env=type('HarmlessGymEnv', (), {}))
    namespace = {'Any': typing.Any, 'gym': harmless_gym}
    definition = ast.Module(body=[future, factory], type_ignores=[])
    exec(compile(definition, str(source), 'exec'), namespace)
    assert typing.get_type_hints(namespace['make_env']) == {
        'env_type': str, 'geometry_policy': str, 'return': typing.Any,
    }
    assert 'ParametricStructureEnv' not in namespace
