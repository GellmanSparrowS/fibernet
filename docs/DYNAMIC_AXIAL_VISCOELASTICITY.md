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
