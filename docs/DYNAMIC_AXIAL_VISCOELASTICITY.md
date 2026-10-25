# Dynamic small-strain axial viscoelasticity

The public `simulate(graph, backend="taichi", mode="dynamics", material=...)`
path reuses `StructureGraph`, the existing Taichi kick/drag/drift/projection
integrator, and `LinearViscoelasticMaterial.trial`. The host updates the existing
Maxwell branch history exactly once after each completed projected interval;
the committed signed `A0 * stress` drives the next native kick. This is not a
second node solver or the legacy numerical dashpot.

## Explicit inputs and example

`examples/dynamic_viscoelastic_network.py` constructs a small noncollinear graph.
`examples/dynamic_viscoelastic_105.py` loads the frozen representative network:

```python
from examples.dynamic_viscoelastic_105 import prepare
from fibernet import simulate

graph, arguments = prepare("fine")  # Only prepares inputs; no solver execution.
result = simulate(graph, **arguments)  # Explicitly run 4096 intervals.
result.save("network.json", detailed=True)
```

All nodes require explicit positive `nodal_masses_by_id` in kg; every physical
edge requires positive `reference_areas_by_id` in m² and `rest_lengths_by_id`
exactly equal to `math.dist` of the initial endpoints in metres. The material is
one uniform `LinearViscoelasticMaterial`. No density, radius, prestress, or
edge-Material conversion is inferred. `state_units` must declare m, s, kg and
`world_xyz`. Fixed nodes and external-force rows follow actual graph insertion
order, not stable IDs. Initial velocities use a sparse stable-node-ID mapping.

`max_abs_strain` is a required positive bound on engineering strain, checked
at every interval, not a stability estimate. `dashpot=0` must be explicit;
optional nonnegative `drag` is numerical drag, not material viscosity.
Explicit `stiffness`/`spring_k`, material restart, displacement schedules,
parallel viscosity, and the old elastic energy ledger are unsupported.
The new rest/initial-velocity keywords are restricted to material mode so the
legacy elastic meaning and result contract remain unchanged.

## Result, observation and persistence

`result.metadata["native_axial_viscoelastic"]` retains original geometry,
stable identities, physical inputs, material parameters, sampled full branch
state and final positions/velocities, physical loss, backward-Euler loss and
separately named signed dynamic bookkeeping residuals. `result.energy` is only
committed axial stored energy; `result.edge_forces` is signed `A0 * stress`.
`record_state=True` adds aligned sampled x/v. An optional `state_observer`
receives immutable completed x/v and committed material state at the same time.
Callbacks, serialization and diagnostics never advance material history.
Exceptions abort rather than return a shortened successful trajectory.

Both summary and detailed `SimResult.save`/`load` retain and validate the full
material record. Loading is not a restart certificate: this slice cannot resume
material dynamics. An elastic stiffness override cannot overwrite material U.
For every noninitial saved state, validation also checks its last physical-loss
increment against the committed Maxwell branch stresses. It uses the producer
order `q = E * (strain - viscous_strain)`, `S = sum(q**2 / E / tau)`,
`d = dt * S`, then `d * (A0 * L0)`. The interval is one native `dt`, not the
gap between saved samples. This is an algebraic read, never a material trial.

The local binary64 allowance is
`8 * ((b+1)*dt*V*ulp(S) + V*ulp(d) + ulp(max(expected, saved)))`, where `b`
is the branch count and `V=A0*L0`. It covers different positive-sum reduction
orders and the two subsequent products under round-to-nearest arithmetic with
gradual underflow, assuming `b*2**-53 <= 1/16`. The budget and comparison use
bit-decoded exact ratios of the finite binary64 values to avoid intermediate budget
underflow or overflow. An empty or zero rate sum requires exactly zero loss;
otherwise `ulp(0)` is the smallest subnormal, not a fixed Joule floor.
Producer-permitted subnormal and underflow-to-zero results are retained;
nonfinite producer intermediates and nonpositive/nonfinite reference volumes
are rejected. This does not extend the separate stress/stored-energy check's
numerical domain or change the producer's operation order.
Subnormal operands or potentially underflowed intermediates require gradual
underflow; if unavailable, validation reports an arithmetic-environment error
rather than a record inconsistency or a silently accepted zero. Ordinary
normal-valued arithmetic does not require this environment probe. Validation
does not change the process floating-point mode. Because this schema records
no producer floating-point mode, this is not a universal compatibility claim
for historical extreme-valued records across hosts.

The endpoint check does not certify cumulative physical loss, cumulative BE
loss, cumulative endpoint work, or cumulative dynamic bookkeeping. Sparse
saved endpoints cannot reconstruct the omitted intervals, and a branch with
zero stress at every saved endpoint cannot authenticate its relaxation time
through this check. No full-history authenticity or cumulative closure claim
is implied.

The backward-Euler identity separates physical material dissipation from
constitutive discretization loss; signed integration/projection residuals are
not mislabeled as either. This implementation uses host NumPy updates and
per-interval native uploads; no GPU or throughput claim is made.

## Numerical validation

The frozen benchmark inputs and compact numerical summary are in
`examples/fixtures/dynamic_viscoelastic_105*.json`.

The connected in-plane synthetic network has 105 nodes, 273 physical edges,
39 fixed boundary nodes, and cycle rank 169. Explicit nodal masses are 0.001 kg
(not density-lumped fibers). Its equilibrium modulus is 1e5 Pa, with Maxwell
branches (2e5 Pa, 0.02 s) and (1e5 Pa, 0.1 s). Initial strain/history and constant
loads are zero; prescribed small initial velocities excite relaxation.
Three timestep levels span 0.5 s with 1024, 2048 and 4096 intervals and retain
129 common sampled states. All 15 fixed self-convergence gates passed:

- Finest maximum engineering strain: 9.376700133227117e-5.
- Finest physical-energy defect: 0.138705% of initial kinetic energy.
- Final physical dissipation: 99.247384% of initial kinetic energy.
- Final backward-Euler loss: 0.306241% of initial kinetic energy.
- Displacement, velocity, strain and stress refinement ratios: 0.499–0.503.

The network workflow includes public result save/load and a stable-ID
manufacturing mapping: one closed 274-row route covers all 273 edges once,
with metre source geometry explicitly converted to millimetres. This is a
geometry/identity mapping, not G-code or a printability certificate. Single-member
generalized-Maxwell summary data also exercises public save/load without
advancing material history.

The network evidence establishes discrete identities and three-grid
self-convergence, not an experimental calibration or continuous-reference
error bound. No contact, adhesion, full-web, finite-strain polymer, bending,
torsion, plasticity, or large-rotation accuracy is claimed or exposed here.

## Inspecting saved cumulative energy residuals

```python
from fibernet.sim import SimResult

result = SimResult.load('saved-result.json')
report = result.viscoelastic_energy_diagnostics()
steps = report['steps']
constitutive = report['constitutive_cumulative_residual_J_by_edge']
dynamic = report['dynamic_bookkeeping_cumulative_residual_J']
```

This read-only method works with existing v1 summary and detailed files. It
returns detached plain dictionaries/lists; it does not run a simulation,
advance material history, or add fields to the saved result. Rows follow saved
steps/times and per-edge columns follow `edge_ids`. All residuals are in joules.

The constitutive sign follows the producer's interval diagnostic:
`C_e = (U_e-U_e0) + P_e + B_e - W_e`. Its aggregate is the sum over edges.
The separate dynamic bookkeeping curve is
`D = (K-K0) + sum(W_e) - (E+G+Q-H+R+S)`, retaining the initial kinetic
energy and the negative projection-force-work term. The combined curve is
`D + sum(C_e)`. Here E is external force work, G numerical drag energy change,
Q projection kinetic energy change, H projection force work, R kick-drift
residual, and S old-stress endpoint residual. These signed terms are not one
physical dissipation channel. Endpoint W is constitutive, not external work.

Residuals are evaluated exactly from the stored binary64 operands and rounded
once for display; aggregate and combined values need not equal sums of their
already-rounded displayed components. Unrepresentable residuals raise
`ValueError` rather than becoming infinity. Existing record validation runs on
each call, including on a result modified after loading; missing material or
unsupported records are refused.

`coverage['cumulative_consistency']` is always `not_certified` for this v1 API,
even for an exactly zero residual. No cumulative numerical-error budget or
pass/fail tolerance is inferred. The curves can reveal inconsistencies but do
not authenticate history or prove solver accuracy. Saved `dynamic_interval`
residuals remain single-last-interval observations, not these cumulative
curves; sparse sample intervals must not be summed or interpolated to recover
omitted updates. Constitutive and dynamic channels are checked separately so
one does not obscure the other, and per-edge values expose cross-edge
cancellation.
