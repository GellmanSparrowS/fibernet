# Separate training and evaluation environments

`fibernet.rl.algorithms.train_rl` is the optional Stable-Baselines3 (SB3)
training adapter. Its keyword-only `eval_env=None` keeps evaluation separate
from persistence. It does not change `fibernet.easy.train_rl`, create another
environment abstraction, or adapt the non-Gymnasium `ParametricStructureEnv`.
Use Gymnasium/SB3-compatible environments and action spaces for this adapter.

## Two existing names, different contracts

`from fibernet import train_rl` and `fibernet.easy.train_rl` refer to the
historical random-action rollout helper, **not policy learning**. It builds
its own `FiberNetworkEnv`, samples an action independently at every step,
closes that environment and returns `n_episodes`, `final_reward`, `mean_reward`
and `max_reward`. Here `final_reward` is the last random episode's reward.
Its `save_path` is a JSON statistics **file**, not a model output directory.

`from fibernet.rl import train_rl` lazily resolves to
`fibernet.rl.algorithms.train_rl`, the SB3 learner covered below. It takes a
caller-owned compatible environment and returns `(model, info)`. These two
routes are not interchangeable; the root helper is not silently rebound or
renamed in this change, and its runtime is unchanged. Its environment is not
automatically an SB3-compatible replacement.

## Usage and migration

Create and own independent training and evaluation environments:

```python
from fibernet.rl.algorithms import train_rl
from fibernet.rl.env import FiberStructureEnv

train_env = FiberStructureEnv(unit="honeycomb", grid=(3, 3))
eval_env = FiberStructureEnv(unit="honeycomb", grid=(3, 3))

model, info = train_rl(
    train_env,
    eval_env=eval_env,
    algorithm="PPO",
    total_timesteps=10_000,
    save_path="runs/design",
    save_freq=1_000,
)
```

This is a usage example, not a reported training experiment or held-out
benchmark. Two fresh instances prevent sharing those Python environment
objects, but identical geometry still is not an independent test scenario.
The caller owns environment construction and cleanup; `train_rl` neither
copies nor closes either environment.

- `eval_env` plus `save_path`: automatic evaluation on `eval_env`; selected
  best-model and evaluation files, final-model saving, and checkpoints if
  `save_freq` is supplied.
- `eval_env` without `save_path`: automatic evaluation with no requested
  best-model/evaluation files, final save or checkpoints. SB3's normal logger
  behavior remains; user-configured loggers or callbacks may write elsewhere.
- `save_path` without `eval_env`: one `UserWarning` is issued per invocation
  (normal Python warning filters apply), automatic evaluation is disabled,
  and checkpoint/final saving remains enabled. No new `best_model` or automatic
  evaluation log is produced. Existing files in that directory are not removed.
- Neither: unchanged training with no automatic evaluation or saving.
  An empty-string `save_path` is treated as no save directory.

Previously, `save_path` silently attached `EvalCallback` to the training
environment itself. Evaluation can reset/step that same live state. Removing
that behavior is an intentional compatibility change: existing save-only
calls still run and save the final model, but must provide a separate
`eval_env` to regain automatic best-model selection. Do not pass the training
environment through a new wrapper as a workaround.

Existing algorithm defaults, overrides and `(model, info)` return shape remain.
The automatic callback uses three deterministic evaluation episodes, with
`eval_freq=max(total_timesteps // 10, 100)`. As in SB3, this frequency and
`save_freq` count callback calls, not aggregate transitions across vector
members. Very short runs may never trigger evaluation. Custom callbacks are
still appended after automatic checkpoint/evaluation callbacks; callers remain
responsible for any environment mutation or persistence in those callbacks.

## Bounded alias check, not an independence certificate

Before constructing the algorithm or callbacks, or creating output directories,
the adapter rejects direct object identity and any shared object reachable
through these known in-process links:

- Gymnasium `Wrapper.env`
- SB3 `VecEnvWrapper.venv`
- SB3 `DummyVecEnv.envs`, including wrappers around its members

The check reads stored links, without calling `unwrapped`, `get_attr`,
`env_method`, `reset`, `step`, `render` or `close`. Repeated objects/cycles are
traversed once. Each root is limited to 1,024 stored object references,
including the root and repeated outgoing references; exceeding the limit or
having missing/malformed known links raises `ValueError` before construction.
Property-only versions of those known links are not evaluated and fail closed.
Custom instance-dictionary storage and non-list/tuple member containers are
also rejected without evaluating their descriptors or iterators.
Wrapper dependency imports occur only when `eval_env` is explicitly supplied.

This does not inspect arbitrary custom wrappers or legacy Gym/Shimmy links,
custom subclasses that override delegation, subprocess/remote worker internals,
shared simulator services, files, global state or other external resources.
Distinct objects, including distinct subprocess vector environments, are not
proof of independent resources. Supply genuinely isolated environments and
compatible wrapper stacks; SB3 can impose additional requirements, including
normalization compatibility. No remote probing or implicit simulator creation
is performed.

The accompanying `easy.py` corrections are documentation-only, including
clarifying that batch `damping` is legacy metadata and that its forwarded
default can emit the existing compatibility warning. Nevertheless, batch
resume provenance hashes the complete `easy.py` source. A previously saved
run binding can therefore reject after this source change: use a fresh
output directory rather than editing the binding or bypassing the guard.

Environment-state isolation is also different from scientific holdout design.
Best-model selection uses validation scenarios, not a final test set. Declare
independent geometry/source groups, material/loading regimes, seed families
and reward version, then reserve separate final-test scenarios. Different
integer seeds alone do not demonstrate different source geometry.

## Evidence and scope

`tests/test_rl_evaluation_isolation.py` uses inert fake SB3 classes and known
wrapper stand-ins. Fake `learn` and `save` only record arguments; sentinel
environments raise on lifecycle operations. The tests check API wiring,
bounded alias rejection, warning and save routing, callback order, legacy
defaults and missing optional dependencies. They do not train a model, run a
simulation or establish real-SB3 runtime/performance compatibility.
AST-only checks also distinguish the root random-rollout API and the lazy
SB3 route without importing or executing the root helper.

The upstream [EvalCallback implementation](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/callbacks.html)
accepts `None` for both persistence paths, so evaluation need not require
saving. Its environment wrapping is not cloning. No package is installed by
this adapter, and SB3 remains optional until `train_rl` is called.
