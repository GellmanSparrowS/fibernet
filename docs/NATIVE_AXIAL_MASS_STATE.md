# Optional nodal masses and sampled native axial state

`TaichiEngine.dynamics` retains its existing unit-mass default. New options are
keyword-only, after existing parameters:

- `nodal_masses`: finite strictly positive float64-convertible array of exact
  shape (N,), following graph node insertion order
- `nodal_masses_by_id`: complete mapping from strict integer stable graph node
  IDs to positive masses; mutually exclusive with the row array
- `record_state=False`: opt into aligned physical-step/time/state samples
- `state_units=None`: optional caller declaration with nonempty string keys
  `length`, `time`, `mass`, `frame`; all four are required when provided

No scalar mass broadcast, material density, radius-based inertia or automatic SI
assumption is supplied. Masked/nonfinite/nonpositive masses, fractional/bool IDs,
missing/extra ID keys, and nonzero values that cast to float64 zero are rejected.
Negative stable IDs are preserved; fixed_nodes and schedule keys remain array-row indices under the existing APIs. Inputs are detached. A Python dict that has already collapsed duplicate-equal
keys cannot reveal overwritten entries; visible keys must be strict and complete.

Explicit mass mode divides each existing endpoint impulse and external-force
impulse by that node's mass. The simultaneous old-velocity dashpot snapshot,
elastic law, softened dashpot direction, external kick, exponential drag, drift,
fixed projection and subsequent schedule projection stay in the same order.
A schedule continues to override a fixed clamp at an overlapping node. No
coefficient is rescaled to maintain acceleration or damping ratio as mass changes.
The no-mass path retains its previous arithmetic, input acceptance and sampling
unless state recording is explicitly requested. Explicit all-ones agrees with
legacy unit masses in deterministic controls. Explicit input provenance is kept
under metadata `native_axial_inputs`, even when state recording is off. A units-only declaration records resolved unit masses as provenance without changing the unit-mass arithmetic.

Explicit mass, state-recording or units-declaration mode requires finite positive dt, finite exact
(N,3) external forces when supplied, and finite nonnegative dashpot/drag before
per-graph Taichi field allocation. In these opt-in modes fixed_nodes must be unmasked strict integer/nonbool row indices in [0,N); negative stable IDs do not make negative row indices valid. Legacy no-mass/no-record validation is retained.
`damping` remains metadata-only. Mass input does not add initial velocity, bending,
contact, prestress, tension-only behavior or a new integrator.

## Units and numerical limits

Declared M,L,T imply axial coefficient k and external force in M L/T², dashpot
in M/(L T) because its force contains L0 times relative velocity, and drag in 1/T.
Caller declarations are assertions, not proof of dimensional consistency. The
existing softened denominator adds 1e-6 in coordinate units; changing labels does
not rescale it. This is not a calibrated material viscosity. Unit conversion,
mass density estimation and automatic stable timestep selection are not provided.

Reference lengths retain the existing 1e-12 floor. The elastic small-distance
cutoff and ideal axial potential are not a globally smooth matched model at
collapse. `near_cutoff` reports sampled lengths at/below nominal 1e-12 and
`cutoff_check='sampled_only'` explicitly does not certify unsampled steps. Existing
Taichi threshold literal precision is unchanged. Explicit stepping is not
unconditionally stable; nonfinite recorded states/energies/times fail rather than
being replaced by zeros. Positive subnormal masses are divided directly, without
precomputing an overflowing reciprocal.

## State and persistence contract

`SimResult.dynamics_state` defaults to None. With recording, schema
`native_axial_state/1` and representation `sampled` contain resolved IDs/masses,
dt, steps, times, positions, velocities, sampled kinetic/potential/total energies,
near-cutoff flags and an exact final sample. Steps include zero, interval multiples
and num_steps once; times are steps*dt. State arrays and positions_trajectory share
this axis, including an opt-in initial history entry. Scheduled-node velocities are the actual zero values after projection,
not a numerical derivative of the imposed path. `time_seconds` remains wall runtime.

K uses exponent-scaled mass-weighted velocity products to avoid premature overflow/underflow; truly unrepresentable nonzero kinetic energy is rejected. K is one-half sum(m v²); U is the existing ideal axial spring energy; H=K+U.
The work ledger always has status `unavailable`, value null, reason `not_collected`,
including conservative and damped runs. Neither sparse samples nor this state
record establishes external/support/damping work, an energy balance, or conservation.

Detailed JSON includes the full sampled record. Summary JSON uses representation
`summary`, availability `summary_only`, units/dt/status and final scalar diagnostics;
it does not contain or reconstruct state arrays. Reloading a summary preserves
that limitation even when later exported with detailed=True. Explicit input-mass
provenance remains separately in metadata. Absent old JSON fields stay None and
absent on export. `dynamics_state` is reserved against metadata shadowing.

Compatibility limitation: older flat metadata could use an arbitrary custom
`dynamics_state` key. Loading such a historical JSON now interprets that key as
the new state record and can reject it; exporting an in-memory metadata entry
with that name omits it in favor of the reserved result field. Before migrating,
retain the original file and rename any unrelated custom metadata key to an
application-specific name. There is no automatic collision migration.

New records reject unknown versions/representations, missing/extra fields, invalid
IDs/shapes/status, nonfinite values, inconsistent steps/time, mass provenance,
kinetic/total energies, final samples and available top-level trajectory/final
position/energy disagreement. The potential cannot be independently reconstructed
from a result lacking its original graph. Atomic save/file ownership behavior is
unchanged.

## Compatibility and validation

The field-cache layout uses a versioned mass-state key. Mass values and mode reset
on every call, including A(m1), B(m2), then None; old velocities refresh every step.
Source changes update batch/generation provenance. Consumers that bind results
to exact source versions must validate updated source identities. Source upgrades
require a fresh process.

Verification must cover input rejection/detachment, unequal-mass response,
clamping/schedules, sparse IDs/parallel members, sampling, summary/detail
roundtrips and same-shape cache reuse. Source/mock checks alone do not establish
native numerical parity or stable behavior for a new problem size or timestep.
Use an owned process-local writable TI_OFFLINE_CACHE_FILE_PATH with Taichi 1.7.4
CPU in restricted environments; no global permission change is required.

See [the API guide](TAICHI_API_DISPATCH.md) for explicit inputs, actual result
quantities and process-isolation limits. This endpoint axial model does not
provide bending, contact or a complete physical work balance.
