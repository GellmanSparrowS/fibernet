# Optional completed-state observer

TaichiEngine.dynamics(..., state_observer=None) adds one keyword-only hook.
None preserves existing sampling, exports, kernels and result serialization.
Enabled synchronous callbacks receive immutable owned float64 position/velocity
snapshots at t0 and each completed step, after fixed projection, with stable IDs
and declared units. The engine retains no events. Consumers must bound retention.
Exceptions propagate as the original object and stop at that completed step.
Ledger recording and nonempty displacement schedules are initially unsupported.

Callbacks are trusted Python, not sandboxed. Keep them read-only/non-reentrant.
DTO equality excludes array fields: compare dtype, shape and bytes explicitly.
Snapshots are completed states only, not per-stage drift/projection accounting,
and do not provide a restart or complete training checkpoint contract.

Verify N+1 snapshots, immutable ownership, callback exception behavior and
subsequent-call state reset for the supported runtime. Source/mock checks alone
do not establish native parity for this release, new physical models or long
campaigns.

accelerated.py remains an existing generation and batch source dependency.
Its changed hash intentionally rejects old generation/batch resume directories;
use a fresh directory or retain the frozen old source. No automatic relabeling.

See [the API guide](TAICHI_API_DISPATCH.md) for explicit inputs and actual result
quantities. The observer does not add bending, contact or a complete physical
work balance, and consumers must bound the snapshots they retain.
