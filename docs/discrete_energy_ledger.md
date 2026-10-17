# Discrete algorithmic energy ledger

`TaichiEngine.dynamics(..., record_state=True, record_energy_ledger=True)` collects
`result.algorithmic_energy_ledger`, schema `native_axial_discrete_ledger/1`.
The default is off. `native_axial_state/1` and its unavailable physical work record
are unchanged. Existing calls retain their kernel and field layout. The ledger
uses a separate cache/kernel variant with diagnostic snapshots only.

The opt-in flag must be a boolean and requires `record_state=True`. It supports
explicit positive nodal masses or unit masses, finite nonnegative spring, dashpot
and drag coefficients, and the existing fixed-row and displacement schedules.
`num_steps=0` is accepted only in ledger mode and returns one initial/final state.
No initial velocity, new force law, contact model, or integrator is introduced.

## What is measured

Actual stages are A old state; B after the existing combined edge kick; C after
external force; D after drag; E after drift; F after fixed projection; and G after
scheduled projection. Fixed projection precedes scheduled projection. Scheduled
velocity is the existing zero reset, not the derivative of a position schedule.

Each sample stores six cumulative signed changes:

- `edge_kick_delta_K = K(B)-K(A)`
- `external_kick_delta_K = K(C)-K(B)`
- `drag_delta_K = K(D)-K(C)`
- `drift_delta_U = U(E)-U(A)`
- `fixed_projection_delta_H = K(F)-K(D)+U(F)-U(E)`
- `schedule_projection_delta_H = K(G)-K(F)+U(G)-U(F)`

Projection changes are algorithmic energy changes, not measured support or
actuator work. Negative values are preserved. `-drag_delta_K` is the algorithmic
kinetic decrease due to drag; it is not independently measured dissipated heat.
For nonzero dashpot the edge kick remains combined. The elastic/dashpot cross term
is not allocated arbitrarily: separate physical work and an elastic-only defect
remain unavailable with explicit reasons. At zero dashpot, `edge_drift_delta`
is an elastic integration-defect indicator affected by external force and drag.
It is not a pure local truncation-error estimate.

`geometric_external_work.value` is the sum of stored external force dotted with
preprojection drift displacement. Subtractions are correctly rounded binary64,
products and accumulation exact rational. Exactly zero force components are skipped.
`subtraction_bound` bounds coordinate-subtraction error, and `bound` adds output
rounding. `minus_external_kick` is the exact rational difference before one output
rounding; `difference_bound` includes coordinate-subtraction error and rounding of
this difference and the exported external-kick channel. It is excluded from closure.

## Arithmetic contract and limits

Each shared K/U endpoint is evaluated once and reused. Existing checked
mass-weighted K and the existing ideal spring U expression are retained. Six
channel differences and totals are exact rational arithmetic on those binary64
numbers. Output channels round once per saved sample. Exact zero channels use +0.
Nonfinite or unrepresentable nonzero diagnostics are rejected, including overflow
and rounding a nonzero output to zero.

The exported-number residual is evaluated exactly as the sum of six exported
channels minus the difference of the stored initial/current H values. Its bound
is the sum of six conservative half-ULP output bounds plus two H-addition bounds.
Subnormal half-ULPs remain exact rational; reported residual magnitudes and bounds
round upward. The validator recomputes the residual and bound, never trusting a
claimed residual of zero. No arbitrary tolerance floor is used.

`worst_step` is the largest exact endpoint-addition rounding ratio, with earliest
tie, across all steps. It stores initial/final K/U/H triples and audits exact
`(K_G+U_G-K_A-U_A)-(H_G-H_A)` against `q(H_A)+q(H_G)`. It does not round unsaved step
channels or form E/F H. This witness is not worst physical energy error.

This telescoping identity is arithmetic bookkeeping, not independent evidence of
correct forces, the integrator, physical work, or continuum energy conservation.
Correlated changes preserving a channel sum cannot be detected from a summary.
Independent recurrence and physical energy checks are also necessary.
Near-cutoff means nominal length <= 1e-12 at any evaluated A/E/F/G configuration;
it does not inspect paths within drift or certify matching of force cutoff and U.

Integer IEEE-754 conversion/rounding and ledger-only JSON decimal parsing protect
the scalar arithmetic from host flush-to-zero/denormals-are-zero modes that native
libraries may enable. No global floating-point mode is changed. This does not
repair or certify existing physical K/U evaluation for subnormal trajectories.

## Saving, provenance and cost

For a ledger-bearing result, the save writer derives decimal numeric tokens from
stored binary64 bits, preserving subnormal ledger bounds under flushing arithmetic.
Nonfinite values anywhere in that saved document are rejected before the owned
temporary replaces its destination. Results without a ledger retain the existing
JSON writer. This guarantee concerns the ledger-bearing save writer and exact
ledger reader, not earlier coercions, arbitrary external JSON calls, float-valued
dictionary keys or restoration of the caller's floating-point environment.

`result.save(path, detailed=True)` includes cumulative rows on exactly the saved
state step/time axis. Summary output includes final channels, initial/final K/U/H,
provenance, interpretation and witness, without inventing intermediate samples.
Both representations strictly validate fieldsets, schema, finite numbers, state
agreement, units, row IDs, masses and availability. Declared units are copied;
energy units are not inferred when units are unknown. Unknown versions reject.

`algorithmic_energy_ledger` is now a reserved result/metadata name. An old custom
metadata value with that name must be manually renamed before migration. Metadata
cannot shadow the reserved result field; old files without the field remain absent.
An old custom payload stored under the new name is rejected rather than guessed.

The implementation retains O(N) stage buffers and O(saved samples) scalar records,
not every-step trajectories. Host copies, energy evaluation and exact accumulation
occur every step and can be expensive. Benchmark a small fixed-step warmup and
forecast before a large run; no claim of negligible overhead is made.
