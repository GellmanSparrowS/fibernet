# Native axial dashpot staging

`TaichiEngine.dynamics` evaluates all edge dashpot forces using a common
old-step velocity snapshot. The snapshot is refreshed in a completed node loop
before any edge velocity kick. This applies to the existing default `dashpot=10`.

Previously, the edge loop read and updated the same velocity array. A later edge
could therefore use a partly updated velocity. Reordering identical physical
members changed the computed trajectory, including on a one-thread CPU runtime.
Existing nonzero-dashpot results should be treated as results of that older update;
the correction intentionally changes them. It does not migrate saved results.

The softened direction `d / (norm(d) + 1e-6)` and rest-length multiplier remain
unchanged. Elastic forces, external kick, exponential drag, position drift,
fixed-node projection, and subsequent scheduled-node projection retain their
existing order. `damping` remains a legacy metadata parameter; it does not set
`dashpot` or `drag`. Unit nodal mass remains the default; explicit masses are described in the
[mass/state contract](NATIVE_AXIAL_MASS_STATE.md). No density or viscosity is
inferred from graph material records.

A physical parallel member contributes its own dashpot. Splitting one spring
into two half-stiffness springs while keeping the same global dashpot coefficient
changes total damping; that transformation is not an equivalence test.

## Validation and limits

Verification should cover simultaneous old-velocity recurrence, chains,
shared-node stars, parallel members, edge order/orientation, zero and nonzero
dashpot, same-shape reuse, drag and constrained updates. Compare floating-point
atomic sums with explicit tolerances; bit-identical results are not promised
under arbitrary edge order or thread count. Source/mock checks alone do not
establish native parity or large-network performance for an installation.

This is a staging correction, not a new integrator. It does not guarantee stable
explicit integration at arbitrary time steps or monotonic total energy. Optional
state and ledger diagnostics do not provide a contact law or GPU validation.

Kernel fields are reused by shape. The old-velocity snapshot is overwritten on
every substep, and cache entries without that field are not reused by this method.
Source upgrades require a fresh process under the existing runtime contract;
this layout check is not a general hot-reload guarantee. Batch/generation source
bindings change when this module changes. Consumers pinning the full native
source manifest must explicitly refresh and validate their binding.
