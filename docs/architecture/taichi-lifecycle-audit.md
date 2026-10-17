# Taichi runtime lifecycle contract

Supported runtime: **Taichi 1.7.4 only**. The implementation uses version-verified private runtime APIs; other versions reject before initialization.

## Implemented lifecycle contract

- The first FiberNet runtime initialization establishes one owning Python
  Thread object. It may be a worker. The runtime, Program and Thread are held
  by strong references, so recycled integer IDs cannot impersonate them
- Sequential compatible engines on that owner share the runtime without init
  and can alternate different graph sizes. Every force/dynamics call checks
  generation before field access. Dynamics/stretch wrappers reject a wrong
  owner before significant graph work; cache clearing also rejects wrong owners
- Cross-owner calls fail early, including warmed-cache calls and constructors.
  Ownership remains even if the original worker exits. A later QThread is not
  an automatic replacement owner; use a persistent owner worker or a separate
  process. No runtime-transfer or adoption API is provided
- A detected external reset invalidates both caches without dereferencing old
  entries. With no active Program, only the established owner may reinitialize
- An externally initialized active runtime, whether initial or a new generation,
  has unverifiable ownership and is rejected after stale-cache discard. The
  diagnostic instructs reset and FiberNet initialization on the intended owner.
  This deliberately does not adopt an external `ti.init`, even on the same
  apparent calling thread. The user must call `ti.reset()` first on that owner
- External reset/init **during a solve is unsupported**. This contract controls
  FiberNet entry points; it cannot serialize arbitrary external Taichi calls
- Requested CPU/GPU must match the active actual backend. A mismatch fails
  without resetting it. Unknown architecture strings and `TI_ARCH` overrides
  at initial setup fail clearly. New GPU initialization disables CPU fallback
- First initialization determines CPU thread count. Later differing constructor
  requests preserve that count and emit a RuntimeWarning; they do not reset
- A short lifecycle lock protects constructor/reset bookkeeping. Correctness
  does not rely on that lock making native cross-thread allocation safe

GPU execution, unsupported Taichi versions and concurrent external Taichi
mutation have not been validated. This is a runtime safety repair, not a
numerical-method change.

## Regression and numerical checks

`tests/test_taichi_runtime_lifecycle_isolated.py` uses opt-in environment variable
`FIBERNET_RUN_LIFECYCLE_ACCEPTANCE=1`, a separate 60-second subprocess per case,
fault handlers, writable caches, and `TI_SKIP_VERSION_CHECK=ON`. It covers both
caches, same/different-size A → B → A, reset recovery, safe external-init rejection
and subsequent reset recovery, CPU/GPU mismatch rejection without needing a GPU,
wrong-owner rejection, worker-owned success and replacement-worker rejection,
unknown preinitialized runtime rejection, and first-initializer thread settings.
Default runs skip the native acceptance cases unless explicitly enabled.
CPU validation does not establish GPU or cross-platform support.

Run the isolated CPU acceptance cases from the repository root in a suitable
optional-dependency environment:

```sh
FIBERNET_RUN_LIFECYCLE_ACCEPTANCE=1 python -m pytest -q tests/test_taichi_runtime_lifecycle_isolated.py
```
