"""Inert train_rl API tests: no real SB3 import, learning or simulation.

For an isolated run that also avoids the repository's Taichi conftest:
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest --noconftest -o addopts='' \
    tests/test_rl_evaluation_isolation.py
"""

import ast
import builtins
import importlib.util
import inspect
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import warnings

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "fibernet" / "rl" / "algorithms.py"


def load_algorithms():
    """Load this leaf directly, avoiding FiberNet package initialization."""
    spec = importlib.util.spec_from_file_location("_rl_isolation_subject", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class UntouchedEnv:
    def reset(self, *args, **kwargs):
        raise AssertionError("Inert regression must not reset an environment")

    def step(self, *args, **kwargs):
        raise AssertionError("Inert regression must not step an environment")

    def render(self, *args, **kwargs):
        raise AssertionError("Inert regression must not render an environment")

    def close(self, *args, **kwargs):
        raise AssertionError("Caller-owned environments must not be closed")

    @property
    def unwrapped(self):
        raise AssertionError("Preflight must not invoke unwrapped")

    @property
    def __class__(self):
        raise AssertionError("Preflight must inspect actual types, not __class__")

    def __getattr__(self, name):
        raise AssertionError(f"Preflight must not probe unknown attribute {name}")

    def __bool__(self):
        raise AssertionError("Environment presence must use identity, not truth")


@pytest.fixture
def fake_sb3(monkeypatch):
    """Only the module/class surface used here; learn/save record, never run."""
    events = []

    class Wrapper(UntouchedEnv):
        def __init__(self, env):
            self.env = env

    class VecEnvWrapper(UntouchedEnv):
        def __init__(self, venv):
            self.venv = venv

    class DummyVecEnv(UntouchedEnv):
        def __init__(self, envs):
            self.envs = envs

    class Model:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.learn_kwargs = None
            self.saved = []
            events.append(("model", self))

        def learn(self, **kwargs):
            self.learn_kwargs = kwargs
            events.append(("learn", kwargs))
            return self

        def save(self, path):
            self.saved.append(path)
            events.append(("save", path))

    class EvalCallback:
        def __init__(self, env, **kwargs):
            self.env = env
            self.kwargs = kwargs
            events.append(("evaluation", self))

    class CheckpointCallback:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            events.append(("checkpoint", self))

    class CallbackList:
        def __init__(self, callbacks):
            self.callbacks = callbacks
            events.append(("callbacks", self))

    modules = {
        name: ModuleType(name)
        for name in (
            "gymnasium",
            "stable_baselines3",
            "stable_baselines3.common",
            "stable_baselines3.common.callbacks",
            "stable_baselines3.common.vec_env",
        )
    }
    modules["gymnasium"].Wrapper = Wrapper
    sb3 = modules["stable_baselines3"]
    for name in ("PPO", "SAC", "TD3", "A2C", "DDPG"):
        setattr(sb3, name, Model)
    sb3.common = modules["stable_baselines3.common"]
    sb3.common.callbacks = modules["stable_baselines3.common.callbacks"]
    sb3.common.vec_env = modules["stable_baselines3.common.vec_env"]
    sb3.common.callbacks.EvalCallback = EvalCallback
    sb3.common.callbacks.CheckpointCallback = CheckpointCallback
    sb3.common.callbacks.CallbackList = CallbackList
    sb3.common.vec_env.VecEnvWrapper = VecEnvWrapper
    sb3.common.vec_env.DummyVecEnv = DummyVecEnv
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(
        events=events,
        Wrapper=Wrapper,
        VecEnvWrapper=VecEnvWrapper,
        DummyVecEnv=DummyVecEnv,
        EvalCallback=EvalCallback,
        CheckpointCallback=CheckpointCallback,
    )


@pytest.fixture
def algorithms(fake_sb3):
    return load_algorithms()


def assert_rejected(algorithms, fake_sb3, train, evaluation, path, match="share"):
    with pytest.raises(ValueError, match=match):
        algorithms.train_rl(
            train, eval_env=evaluation, save_path=str(path), save_freq=4
        )
    assert fake_sb3.events == []
    assert not path.exists()


def test_eval_env_signature(algorithms):
    parameter = inspect.signature(algorithms.train_rl).parameters["eval_env"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


@pytest.mark.parametrize("with_save", [False, True])
def test_direct_alias_is_rejected_before_side_effects(
    algorithms, fake_sb3, tmp_path, with_save
):
    env = UntouchedEnv()
    path = tmp_path / "unused"
    with pytest.raises(ValueError, match="must not share"):
        algorithms.train_rl(
            env, eval_env=env, save_path=str(path) if with_save else None
        )
    assert fake_sb3.events == []
    assert not path.exists()


@pytest.mark.parametrize(
    "shape",
    [
        "raw_wrapped",
        "wrapped_raw",
        "separate_wrappers",
        "nested",
        "vec_wrappers",
        "shared_vector_member",
        "mixed_graph",
    ],
)
def test_known_wrapper_aliases_fail_before_side_effects(
    algorithms, fake_sb3, tmp_path, shape
):
    shared = UntouchedEnv()
    W, V, D = fake_sb3.Wrapper, fake_sb3.VecEnvWrapper, fake_sb3.DummyVecEnv
    if shape == "raw_wrapped":
        train, evaluation = shared, W(shared)
    elif shape == "wrapped_raw":
        train, evaluation = W(shared), shared
    elif shape == "separate_wrappers":
        train, evaluation = W(shared), W(shared)
    elif shape == "nested":
        train, evaluation = W(W(shared)), W(W(shared))
    elif shape == "vec_wrappers":
        vec = D([shared])
        train, evaluation = V(vec), V(vec)
    elif shape == "shared_vector_member":
        train = D([UntouchedEnv(), shared])
        evaluation = D([shared, UntouchedEnv()])
    else:
        train = V(D([UntouchedEnv(), W(W(shared))]))
        evaluation = V(V(D([W(shared)])))
    assert_rejected(algorithms, fake_sb3, train, evaluation, tmp_path / "unused")


@pytest.mark.parametrize("with_save", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
def test_disjoint_environments_route_separately(
    algorithms, fake_sb3, tmp_path, with_save, wrapped
):
    train, evaluation = UntouchedEnv(), UntouchedEnv()
    if wrapped:
        W, V, D = fake_sb3.Wrapper, fake_sb3.VecEnvWrapper, fake_sb3.DummyVecEnv
        train = V(D([W(train), UntouchedEnv()]))
        evaluation = V(D([UntouchedEnv(), W(evaluation)]))
    path = tmp_path / "models"
    custom = object()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model, info = algorithms.train_rl(
            train,
            eval_env=evaluation,
            save_path=str(path) if with_save else None,
            save_freq=9,
            total_timesteps=2500,
            callback=custom,
        )
    assert caught == []
    assert model.kwargs["env"] is train
    assert "eval_env" not in model.kwargs
    callbacks = model.learn_kwargs["callback"].callbacks
    if with_save:
        assert isinstance(callbacks[0], fake_sb3.CheckpointCallback)
        assert callbacks[0].kwargs == {
            "save_freq": 9, "save_path": str(path), "name_prefix": "rl_model"
        }
    evaluation_callback = callbacks[-2]
    assert isinstance(evaluation_callback, fake_sb3.EvalCallback)
    assert evaluation_callback.env is evaluation
    assert evaluation_callback.kwargs == {
        "best_model_save_path": str(path) if with_save else None,
        "log_path": str(path) if with_save else None,
        "eval_freq": 250,
        "n_eval_episodes": 3,
        "deterministic": True,
    }
    assert callbacks[-1] is custom
    assert len(callbacks) == (3 if with_save else 2)
    assert model.saved == ([f"{path}/final_model"] if with_save else [])
    assert path.exists() is with_save
    assert set(info) == {"algorithm", "total_timesteps", "seed", "kwargs"}


@pytest.mark.parametrize("save_freq", [None, 7])
def test_save_only_warns_once_and_retains_persistence_and_custom_callback(
    algorithms, fake_sb3, tmp_path, save_freq
):
    path = tmp_path / "models"
    custom = object()
    for _ in range(2):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model, _ = algorithms.train_rl(
                UntouchedEnv(),
                save_path=str(path),
                save_freq=save_freq,
                callback=custom,
            )
        assert len(caught) == 1
        assert caught[0].category is UserWarning
        assert "eval_env was not supplied" in str(caught[0].message)
        assert "checkpoint and final-model saving remain enabled" in str(caught[0].message)
        assert caught[0].filename == __file__
        callbacks = model.learn_kwargs["callback"].callbacks
        assert callbacks[-1] is custom
        assert not any(isinstance(cb, fake_sb3.EvalCallback) for cb in callbacks)
        assert len(callbacks) == (1 if save_freq is None else 2)
        if save_freq is not None:
            assert callbacks[0].kwargs["save_freq"] == save_freq
        assert model.saved == [f"{path}/final_model"]
        assert path.is_dir()


@pytest.mark.parametrize("save_path", [None, ""])
@pytest.mark.parametrize("with_custom", [False, True])
def test_default_no_evaluation_or_save_is_unchanged(
    algorithms, fake_sb3, save_path, with_custom
):
    custom = object() if with_custom else None
    train = UntouchedEnv()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model, info = algorithms.train_rl(
            train, save_path=save_path, save_freq=5, callback=custom
        )
    assert caught == []
    assert model.saved == []
    assert not any(event[0] in ("evaluation", "checkpoint") for event in fake_sb3.events)
    callback = model.learn_kwargs["callback"]
    assert (callback.callbacks == [custom]) if with_custom else (callback is None)
    assert model.learn_kwargs["total_timesteps"] == 10000
    assert model.learn_kwargs["log_interval"] == 10
    assert info == {
        "algorithm": "PPO",
        "total_timesteps": 10000,
        "seed": 42,
        "kwargs": dict(algorithms.ALGORITHM_REGISTRY["PPO"]["default_kwargs"]),
    }


def test_empty_save_path_with_evaluation_has_none_paths(algorithms, fake_sb3):
    model, _ = algorithms.train_rl(
        UntouchedEnv(), eval_env=UntouchedEnv(), save_path=""
    )
    evaluation = model.learn_kwargs["callback"].callbacks[0]
    assert evaluation.kwargs["best_model_save_path"] is None
    assert evaluation.kwargs["log_path"] is None
    assert evaluation.kwargs["eval_freq"] == 1000
    assert model.saved == []


def test_small_run_keeps_minimum_evaluation_frequency(algorithms):
    model, _ = algorithms.train_rl(
        UntouchedEnv(), eval_env=UntouchedEnv(), total_timesteps=32
    )
    assert model.learn_kwargs["callback"].callbacks[0].kwargs["eval_freq"] == 100


def test_cycles_and_repeated_members_terminate(algorithms, fake_sb3):
    train = fake_sb3.Wrapper(UntouchedEnv())
    train.env = train
    repeated = UntouchedEnv()
    evaluation = fake_sb3.DummyVecEnv([repeated, repeated])
    model, _ = algorithms.train_rl(train, eval_env=evaluation)
    assert model.kwargs["env"] is train


@pytest.mark.parametrize("count", [1024, 1025])
def test_exact_inspection_bound(algorithms, fake_sb3, tmp_path, count):
    train = UntouchedEnv()
    for _ in range(count - 1):
        train = fake_sb3.Wrapper(train)
    if count == 1024:
        model, _ = algorithms.train_rl(train, eval_env=UntouchedEnv())
        assert model.kwargs["env"] is train
    else:
        assert_rejected(
            algorithms, fake_sb3, train, UntouchedEnv(), tmp_path / "unused",
            match="inspection limit of 1024",
        )


def test_vector_width_also_counts_towards_bound(algorithms, fake_sb3, tmp_path):
    repeated = UntouchedEnv()
    train = fake_sb3.DummyVecEnv([repeated] * 1024)
    assert_rejected(
        algorithms, fake_sb3, train, UntouchedEnv(), tmp_path / "unused",
        match="inspection limit of 1024",
    )


@pytest.mark.parametrize("member_state", [None, [], {}, [None]])
def test_malformed_vector_members_fail_closed(
    algorithms, fake_sb3, tmp_path, member_state
):
    assert_rejected(
        algorithms, fake_sb3, fake_sb3.DummyVecEnv(member_state), UntouchedEnv(),
        tmp_path / "unused", match="Cannot inspect known environment wrapper link",
    )


@pytest.mark.parametrize("kind,link", [("Wrapper", "env"), ("VecEnvWrapper", "venv"),
                                      ("DummyVecEnv", "envs")])
def test_property_only_known_links_fail_without_invoking_property(
    algorithms, fake_sb3, tmp_path, kind, link
):
    def forbidden_property(self):
        raise AssertionError("Wrapper property must not run")

    cls = type("PropertyOnly", (getattr(fake_sb3, kind),), {
        link: property(forbidden_property),
    })
    env = object.__new__(cls)
    assert_rejected(
        algorithms, fake_sb3, env, UntouchedEnv(), tmp_path / "unused",
        match="Cannot inspect known environment wrapper link",
    )


def test_unknown_environment_attributes_are_not_probed(algorithms):
    class UnknownWrapper(UntouchedEnv):
        @property
        def env(self):
            raise AssertionError("Unknown wrappers are explicitly outside the check")

        @property
        def envs(self):
            raise AssertionError("Unknown vector wrappers must not be probed")

    model, _ = algorithms.train_rl(UnknownWrapper(), eval_env=UntouchedEnv())
    assert model.saved == []


def test_custom_dict_descriptor_fails_without_invoking_it(
    algorithms, fake_sb3, tmp_path
):
    class CustomStorage(fake_sb3.Wrapper):
        @property
        def __dict__(self):
            raise AssertionError("Custom __dict__ descriptor must not be invoked")

    assert_rejected(
        algorithms, fake_sb3, CustomStorage(UntouchedEnv()), UntouchedEnv(),
        tmp_path / "unused", match="custom __dict__ storage",
    )


def test_custom_vector_container_fails_without_invoking_it(
    algorithms, fake_sb3, tmp_path
):
    class CustomMembers(list):
        def __len__(self):
            raise AssertionError("Custom member container must not be evaluated")

    assert_rejected(
        algorithms, fake_sb3, fake_sb3.DummyVecEnv(CustomMembers([UntouchedEnv()])),
        UntouchedEnv(), tmp_path / "unused",
        match="Cannot inspect known environment wrapper link",
    )


def test_custom_instance_dict_fails_without_invoking_get(
    algorithms, fake_sb3, tmp_path
):
    class CustomState(dict):
        def get(self, *args, **kwargs):
            raise AssertionError("Custom instance dictionary get must not run")

    env = fake_sb3.Wrapper(UntouchedEnv())
    object.__setattr__(env, "__dict__", CustomState(env=UntouchedEnv()))
    assert_rejected(
        algorithms, fake_sb3, env, UntouchedEnv(), tmp_path / "unused",
        match="custom __dict__ storage",
    )


@pytest.mark.parametrize(
    "algorithm,total_timesteps,expected_key,expected_value",
    [("ppo", 128, "n_steps", 32), ("SAC", 500, "buffer_size", 1000)],
)
def test_algorithm_defaults_and_overrides_are_unchanged(
    algorithms, algorithm, total_timesteps, expected_key, expected_value
):
    model, info = algorithms.train_rl(
        UntouchedEnv(), algorithm=algorithm, total_timesteps=total_timesteps,
        learning_rate=0.012, gamma=0.8, seed=12, verbose=0,
    )
    assert model.kwargs[expected_key] == expected_value
    assert model.kwargs["learning_rate"] == 0.012
    assert model.kwargs["gamma"] == 0.8
    assert model.kwargs["seed"] == 12
    assert model.kwargs["verbose"] == 0
    assert info["algorithm"] == algorithm
    assert info["kwargs"]["learning_rate"] == 0.012
    assert algorithms.ALGORITHM_REGISTRY[algorithm.upper()]["default_kwargs"]["gamma"] == 0.99


def test_invalid_algorithm_still_fails_early(algorithms, fake_sb3, tmp_path):
    path = tmp_path / "unused"
    with pytest.raises(ValueError, match="Unknown algorithm"):
        algorithms.train_rl(UntouchedEnv(), algorithm="missing", save_path=str(path))
    assert fake_sb3.events == []
    assert not path.exists()


def test_missing_sb3_error_is_actionable_and_leaf_import_is_optional(monkeypatch):
    original_import = builtins.__import__
    attempted = []

    def without_optional(name, *args, **kwargs):
        if name.split(".")[0] in ("stable_baselines3", "gymnasium", "torch", "fibernet"):
            attempted.append(name)
            raise ImportError(f"Deliberately unavailable: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_optional)
    module = load_algorithms()
    assert attempted == []
    with pytest.raises(ImportError, match="pip install stable-baselines3"):
        module.train_rl(UntouchedEnv())
    assert attempted == ["stable_baselines3"]


def test_no_eval_does_not_import_wrapper_dependencies(algorithms, fake_sb3, monkeypatch):
    original_import = builtins.__import__

    def without_wrappers(name, *args, **kwargs):
        if name == "gymnasium" or name.startswith("stable_baselines3.common.vec_env"):
            raise AssertionError("No eval_env must not import wrapper dependencies")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_wrappers)
    model, _ = algorithms.train_rl(UntouchedEnv())
    assert model.saved == []


def _assert_distinct_lazy_train_rl_routes(root_api, rl_api, fail_owner=None):
    """Exercise only facade resolvers with inert, distinct owning objects."""
    random_rollout, policy_trainer = object(), object()
    owners = {
        "fibernet.easy": SimpleNamespace(train_rl=random_rollout, evaluate_rl=object()),
        "fibernet.rl.algorithms": SimpleNamespace(train_rl=policy_trainer, evaluate_rl=object()),
    }
    imported = []

    def load_owner(name):
        assert name in owners, "Unexpected train_rl owner"
        imported.append(name)
        if fail_owner is not None and name == fail_owner[0]:
            raise fail_owner[1]
        return owners[name]

    def isolated_import(name, globals=None, locals=None, fromlist=(), level=0):
        assert name == "importlib" and level == 0, "Unexpected resolver import"
        return SimpleNamespace(import_module=load_owner)

    def resolver(tree, root=False):
        # A lazy declaration must not conceal an eager train_rl rebound.
        for node in ast.walk(tree):
            assert not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
                        and node.id == "train_rl"), "Eager train_rl assignment"
            assert not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                        and node.name == "train_rl"), "Eager train_rl definition"
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                assert all((alias.asname or alias.name.split(".")[0]) != "train_rl"
                           for alias in node.names), "Eager train_rl import"
        namespace = {"__builtins__": {"dict": dict, "getattr": getattr,
                     "globals": globals, "AttributeError": AttributeError,
                     "__import__": isolated_import}}
        declarations = []
        if root:
            for name in ("_LAZY_EXPORTS", "_LAZY_MODULES", "_RENDER_EXPORTS"):
                matches = [node for node in tree.body if isinstance(node, ast.Assign)
                           and len(node.targets) == 1
                           and isinstance(node.targets[0], ast.Name)
                           and node.targets[0].id == name]
                assert len(matches) == 1, "Ambiguous lazy declaration"
                declarations.extend(matches)
            exec(compile(ast.Module(body=declarations, type_ignores=[]),
                         "<isolated-facade-declarations>", "exec"), namespace)
            assert namespace["_LAZY_EXPORTS"].get("train_rl") == "fibernet.easy"
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "__getattr__"]
        assert len(functions) == 1
        # Extraction must not discard later map mutations or resolver rebinding.
        # These are the only read/reload uses outside the extracted declarations.
        allowed_root_uses = (
            "for _name in (*_LAZY_EXPORTS, *_LAZY_MODULES, '_HAS_RL'):\n"
            "    globals().pop(_name, None)",
            "def __dir__():\n"
            "    return sorted(set(globals()) | set(_RENDER_EXPORTS) | "
            "set(_LAZY_EXPORTS) | set(_LAZY_MODULES) | {'_HAS_RL'})",
        )
        allowed = {ast.dump(ast.parse(source).body[0]) for source in allowed_root_uses}
        dependencies = {"_LAZY_EXPORTS", "_LAZY_MODULES", "_RENDER_EXPORTS",
                        "__getattr__", "__dir__", "train_rl"}
        for statement in tree.body:
            if statement in declarations or statement in functions:
                continue
            if root and ast.dump(statement) in allowed:
                continue
            for node in ast.walk(statement):
                assert not (isinstance(node, ast.Name) and node.id in dependencies), \
                    "Out-of-slice route reference"
                assert not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                            and node.name in dependencies), "Out-of-slice route definition"
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    assert all((alias.asname or alias.name.split(".")[0]) not in dependencies
                               for alias in node.names), "Out-of-slice route import"
                assert not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                            and node.func.id in {"globals", "locals", "vars", "exec", "eval"}), \
                    "Unsupported dynamic module binding"
        exec(compile(ast.Module(body=functions, type_ignores=[]),
                     "<isolated-facade-resolver>", "exec"), namespace)
        return namespace

    root = resolver(root_api, root=True)
    rl = resolver(rl_api)
    assert root["__getattr__"]("train_rl") is random_rollout
    assert root["train_rl"] is random_rollout  # root facade caches its owner object
    assert rl["__getattr__"]("train_rl") is policy_trainer
    assert random_rollout is not policy_trainer
    assert imported == ["fibernet.easy", "fibernet.rl.algorithms"]


def test_root_random_rollout_and_rl_trainer_routes_are_documented_not_rebound():
    root = SOURCE.parents[1]
    easy = ast.parse((root / "easy.py").read_text(encoding="utf-8"))
    train = next(
        node for node in easy.body
        if isinstance(node, ast.FunctionDef) and node.name == "train_rl"
    )
    assert "随机策略 rollout 基线" in ast.get_docstring(easy)
    documentation = ast.get_docstring(train)
    for phrase in (
        "不进行策略学习", "不训练或返回策略模型", "from fibernet.rl import train_rl",
        "Gymnasium/SB3", "JSON 文件路径", "最后一次随机回合",
    ):
        assert phrase in documentation
    assert [arg.arg for arg in train.args.args] == ["env_config", "n_episodes", "save_path"]
    assert train.args.kwonlyargs == []
    calls = [
        ast.unparse(node.func) for node in ast.walk(train) if isinstance(node, ast.Call)
    ]
    assert "env.action_space.sample" in calls
    assert "json.dump" in calls
    assert not any(call.endswith(".learn") for call in calls)

    root_api = ast.parse((root / "__init__.py").read_text(encoding="utf-8"))
    rl_api = ast.parse((root / "rl" / "__init__.py").read_text(encoding="utf-8"))
    _assert_distinct_lazy_train_rl_routes(root_api, rl_api)


def test_lazy_train_rl_routes_reject_adverse_rebindings():
    root = SOURCE.parents[1]
    root_source = (root / "__init__.py").read_text(encoding="utf-8")
    rl_source = (root / "rl" / "__init__.py").read_text(encoding="utf-8")
    for mutation in (
        "root_owner", "root_missing", "root_return", "root_symbol",
        "root_import", "root_assignment", "root_duplicate_map", "root_decoy",
        "rl_owner", "rl_return", "rl_symbol", "root_subscript", "root_update",
        "root_getter_rebind", "rl_getter_rebind", "root_alias", "root_dynamic",
    ):
        root_api, rl_api = ast.parse(root_source), ast.parse(rl_source)
        target = rl_api if mutation.startswith("rl_") else root_api
        if mutation in ("root_subscript", "root_update", "root_getter_rebind",
                        "rl_getter_rebind", "root_alias", "root_dynamic"):
            statement = {
                "root_subscript": '_LAZY_EXPORTS["train_rl"] = "fibernet.rl.algorithms"',
                "root_update": '_LAZY_EXPORTS.update({"train_rl": "fibernet.rl.algorithms"})',
                "root_getter_rebind": '__getattr__ = lambda name: None',
                "rl_getter_rebind": '__getattr__ = lambda name: None',
                "root_alias": 'routes = _LAZY_EXPORTS; routes["train_rl"] = "fibernet.rl.algorithms"',
                "root_dynamic": 'globals()["train_rl"] = None',
            }[mutation]
            target.body.extend(ast.parse(statement).body)
        elif mutation in ("root_owner", "root_decoy"):
            owner = next(node for node in ast.walk(root_api)
                         if isinstance(node, ast.Constant) and node.value == "fibernet.easy")
            owner.value = "fibernet.rl.algorithms"
            if mutation == "root_decoy":
                root_api.body.extend(ast.parse('_UNUSED = {"train_rl": "fibernet.easy"}').body)
        elif mutation == "root_missing":
            exports = next(node.value for node in root_api.body
                           if isinstance(node, ast.Assign) and any(
                               isinstance(target, ast.Name) and target.id == "_LAZY_EXPORTS"
                               for target in node.targets))
            name = next(node for node in ast.walk(exports)
                        if isinstance(node, ast.Constant) and node.value == "train_rl")
            name.value = "not_train_rl"
        elif mutation.endswith("_return"):
            function = next(node for node in target.body
                            if isinstance(node, ast.FunctionDef) and node.name == "__getattr__")
            for node in ast.walk(function):
                if isinstance(node, ast.Return):
                    node.value = ast.Constant(value=None)
        elif mutation.endswith("_symbol"):
            function = next(node for node in target.body
                            if isinstance(node, ast.FunctionDef) and node.name == "__getattr__")
            call = next(node for node in ast.walk(function)
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == "getattr")
            call.args[1] = ast.Constant(value="evaluate_rl")
        elif mutation == "root_import":
            root_api.body.extend(ast.parse('from fibernet.rl.algorithms import train_rl').body)
        elif mutation == "root_assignment":
            root_api.body.extend(ast.parse('train_rl = None').body)
        elif mutation == "root_duplicate_map":
            root_api.body.extend(ast.parse('_LAZY_EXPORTS = {"train_rl": "fibernet.rl.algorithms"}').body)
        elif mutation == "rl_owner":
            routes = next(node for node in ast.walk(rl_api)
                          if isinstance(node, ast.Dict) and any(
                              isinstance(key, ast.Constant) and key.value == "train_rl"
                              for key in node.keys))
            index = next(index for index, key in enumerate(routes.keys)
                         if isinstance(key, ast.Constant) and key.value == "train_rl")
            routes.values[index] = ast.Constant(value="fibernet.easy")
        ast.fix_missing_locations(root_api)
        ast.fix_missing_locations(rl_api)
        try:
            _assert_distinct_lazy_train_rl_routes(root_api, rl_api)
        except AssertionError:
            continue
        raise AssertionError("Route mutation was accepted: " + mutation)


def test_lazy_train_rl_routes_preserve_owner_failures():
    root = SOURCE.parents[1]
    root_api = ast.parse((root / "__init__.py").read_text(encoding="utf-8"))
    rl_api = ast.parse((root / "rl" / "__init__.py").read_text(encoding="utf-8"))
    for owner in ("fibernet.easy", "fibernet.rl.algorithms"):
        for kind in (ImportError, RuntimeError, AttributeError):
            failure = kind("deliberate inert owner failure")
            try:
                _assert_distinct_lazy_train_rl_routes(root_api, rl_api, (owner, failure))
            except kind as caught:
                assert caught is failure
            else:
                raise AssertionError("Owner failure was hidden: " + owner)


def test_batch_damping_documentation_preserves_resume_caveat():
    easy = ast.parse((SOURCE.parents[1] / "easy.py").read_text(encoding="utf-8"))
    batch = next(
        node for node in easy.body
        if isinstance(node, ast.FunctionDef) and node.name == "batch_simulate_from_json"
    )
    documentation = ast.get_docstring(batch)
    assert "Legacy metadata-only value; does not change the simulation." in documentation
    assert "default 0.3 explicitly" in documentation
    assert "use the direct dynamics interface to control dashpot/drag" in documentation
    assert "implementation" in documentation and "fresh output directory" in documentation
