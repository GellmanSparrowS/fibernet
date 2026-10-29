# RL composition preflight

`fibernet.rl.train_rl` wraps Stable-Baselines3 (SB3); it does not translate arbitrary environment protocols or action meanings. The root `fibernet.train_rl`/`easy.train_rl` random-rollout route is separate and unchanged.

Before model/callback construction, learning or automatic output creation, the trainer checks known evaluation aliases, nominal environment classes and **statically available action-space families**:

| Algorithm | Action families |
|---|---|
| PPO, A2C | Box, Discrete, MultiDiscrete, MultiBinary |
| SAC, TD3, DDPG | Box |

Actual subclasses of the supported Gymnasium space classes remain eligible. Raw OpenAI Gym environments are checked against their original Gym space families without conversion; SB3 remains responsible for its version-dependent Shimmy requirement and adaptation. Existing SB3 VecEnv objects are checked through their declared outer space, without worker access. Objects that merely masquerade as a class through a custom `__class__` property are not nominal subclasses.

A stored unsupported space raises a `ValueError` naming the algorithm and whether `env` or `eval_env` failed. This includes FiberDesignEnv's mixed Dict actions. Its action semantics are **not** automatically flattened, discretized or remapped. An explicit adapter needs separate design and validation; preflight does not establish that such an adapter is scientifically appropriate.

## Inspection and delegation

The passive check inspects nominal environment types and stored action-space families without invoking getters. Properties, slot descriptors, missing/dynamic metadata, and delegated spaces are **deferred** to SB3. This includes an ordinary Gymnasium Wrapper's action-space property, even when the underlying environment stores an unsupported space. Wrappers remain intact rather than being traversed by preflight.

Registered string training IDs remain supported by SB3's creation route, but their spaces are deferred: preflight does not create an environment just to inspect it. Evaluation requires an independent environment object, not an ID string.

The internal `checked`/`deferred` status records this inspection boundary and leaves the public training-result fields unchanged. SB3 retains responsibility for deferred spaces, observation/policy compatibility, shapes, bounds, start indices, dependencies and runtime protocol validation. MultiInputPolicy and custom policy classes pass through unchanged. Reward correctness, learning outcomes and performance require their own evaluation. The guard leaves reset, step, close and protocol conversion to the existing environment/backend workflow.

## Legacy parametric route

`make_env('parametric')` intentionally returns ParametricStructureEnv. Its reset returns an observation and its step returns `(graph, result, reward, info)`, not a Gym transition. This is not the old Gym `(observation, reward, done, info)` protocol and must not be passed to Shimmy as if it were. `train_rl` rejects the non-Gym object; the factory and legacy behavior remain unchanged. The factory uses a broad `Any` return annotation for runtime introspection, while its return documentation identifies the protocols selected by `env_type`.

The action matrix was checked against official [SB3 2.0](https://stable-baselines3.readthedocs.io/en/v2.0.0/guide/algos.html) and [2.8](https://stable-baselines3.readthedocs.io/en/v2.8.0/guide/algos.html) documentation. The dependency minimum is not a claim that every future SB3 version has been executed. Inert contract tests establish dispatch and rejection ordering only, not genuine backend integration or policy quality.
