# API layers and supported simulation routes

[Feature availability and checked dataset selections](FEATURE_AVAILABILITY.md)

## Choose a phenomenon, then an explicit supported route

FiberNet's conceptual order is **graph → physical model and material → analysis,
loading and time controls → backend → typed result**. These are distinct choices,
not an arbitrary Cartesian product of selector strings. The existing `simulate`
facade retains historical names: `stretch` is a loading protocol, `history` is a
quasistatic material history, and `dynamics` is inertial integration. Likewise,
`spring` names a model while `taichi` names its execution engine; they are aliases,
not two different solvers. CPU/GPU is a Taichi device choice, not a physical mode.

- **Getting started:** use `StructureGraph`, choose an explicit backend and a
  small deformation, and read that route's unit and result contract below.
- **Modeling and research:** choose material law, boundary conditions, loads,
  time grid, sampling and result family explicitly. Numerical damping is not
  material viscosity. Do not infer one solver's meaning from another's keywords.
- **Advanced use:** use `TaichiEngine`, `BeamFrameFEM`, typed transport/thermal
  functions, or LAMMPS preparation/task records directly. These expose distinct
  contracts; they are not interchangeable implementations of a universal solver.

`StructureGraph` is the authoritative structure and stable identity source.
`FiberNetwork` and its generators are legacy/interoperability APIs. Conversion
from a legacy network is approximate, not lossless preservation of curved fibers
or every crosslink. NetworkX, learning features and manufacturing routes/solids
are derived representations, not alternative authoritative graphs.

`graph.to_fiber_network()` returns a legacy endpoint-chord projection with
independently owned box and per-fiber material data, including nested material
records and tensors. Editing these returned values does not edit the source
graph or another converted fiber, even when source edges share a material.
Coordinates and graph metadata also remain detached. Custom materials follow
their `deepcopy` implementation, as in the graph's per-edge copy policy.
This does not make the conversion lossless: source internal points, stable IDs,
per-node/per-edge metadata and isolated nodes are not preserved as graph records;
endpoint chords shorter than `1e-12` in the supplied coordinate units remain
omitted. Fiber IDs are dense rows, and the existing discretization and junction
crosslink construction remain unchanged. Geometry coordinates, radii and box
values are copied without unit conversion or unit inference. Keep the original
`StructureGraph` or its native serialization for authoritative geometry and IDs.
Legacy OBJ export still writes endpoint cylinders, not the stored source curves.

### Edit geometry before preparing solver controls

Use `graph.displace_node(node_id, [dx, dy])`,
`graph.set_node_position(node_id, [x, y, z])`, or
`graph.set_node_positions({node_id: [x, y]})` with stable node IDs. These
coordinates use the graph's existing units; no conversion is performed.
Rejected edits leave geometry, merge lookup and ID counters unchanged. A batch
with a rejected ID or coordinate is rejected before any node is changed.
Valid edits retain IDs, boundary labels and internal edge points; they do not
recompute boundary membership or deform a curved edge's internal geometry.
Prepare supports and loads after editing. Existing prepared controls must be
prepared again when the graph changes, as described in
[stable-ID nodal controls](STABLE_NODAL_CONTROLS.md).

## Mechanical facade capability matrix

Selectors are case-sensitive. Default `simulate` is `backend="spring"`,
`mode="stretch"`; it requires Taichi, with no silent NumPy fallback.
Prefer the keyword-only `stretch_ratio` for new stretch calls. Legacy
`strain` means the same **stretch ratio λ**, not engineering strain ε: λ=1.01 means
ε=0.01 (1% extension). The effective default stretch ratio is 2.0, not 2%.
The facade default `strain=None` means unset; omission or `None` resolves to
λ=2.0 for stretch. Every non-`None` `strain` is rejected outside stretch,
including explicit `2.0` and NaN. Other modes take explicit controls. There is
no facade `target_stretch` alias; direct solver methods use that parameter.
`stretch_ratio` must be a finite positive real scalar, not a boolean, and is
rejected outside stretch. Do not supply both non-`None` names, even with equal
values. `None` means unset for either name; the omitted default remains 2.0.
Legacy positional calls retain their argument order and behavior. A positive
ratio does not establish physical applicability: choose a small deformation
for the small-strain FEM model.

| Backend | Mode | Model/material | Controls | Return |
|---|---|---|---|---|
| `spring` or `taichi` | `stretch` | Axial springs, explicit global force-like stiffness; no graph `Material` inference | Automatic left/right grips, target ratio, ramp and step/sampling options | `SimResult` |
| `spring` or `taichi` | `dynamics` | Axial springs; scalar/per-edge `spring_k` overrides `stiffness` | Fixed rows, optional displacement schedule, force rows, `dt`, `num_steps`, `save_interval`, explicit masses/units when needed | `SimResult` |
| `spring` or `taichi` | `dynamics` with `material` | Exact `LinearViscoelasticMaterial`, small-strain axial generalized Maxwell/SLS; `parallel_viscosity=0` | Explicit masses, areas, rest lengths, strain bound, SI state units; fixed supports, optional initial velocities; `dashpot=0` | `SimResult` with material record |
| `fem` | `stretch` | Beam-frame elasticity; explicit `E` and `nu`, not graph moduli | Automatic grips; `dim`, `pct`, `nonlinear`; target ratio | `SimResult` converted from beam-frame result |
| `fem` | `history` | 2-D small-strain viscoelastic beam frame, axial and bending | `material`, `times`, `constraints`, optional `loads`, explicit `length_unit`, optional `initial_state` | `ViscoelasticResult` |
| `lammps` | `dynamics` | Explicit 3-D harmonic beads, or `CGRequest` harmonic bonds/cosine angles/optional projectile pairs | Typed model/run records, pinned `runtime`, new owned `workdir`, `limits`; optional `record_state` | `SimResult`; task API returns `LAMMPSRunResult` |

No other facade combinations are supported. In particular, FEM `history` is not
inertial dynamics, and there is no `electrical`, `thermal`, `joule`, `finite`,
`reduced_beam`, or `cg` value of `simulate(mode=...)` or backend alias.
`batch_simulate` runs FEM stretch cases; `batch_simulate_from_json` supports
spring/stretch only. Neither is a general backend router.

### Mechanical controls and restrictions

- Taichi `arch` is `"cpu"` (default) or `"gpu"`; `num_threads` defaults to 4 and
  must be a positive integer, not a boolean. Importability is not runtime
  qualification. GPU does not silently fall back to CPU.
- Ordinary Taichi dynamics defaults include `stiffness=1e5`, `dt=1e-4`,
  `num_steps=5000`, `save_interval=500`, `dashpot=10`, and `drag=1`.
  `damping` is a legacy metadata-only parameter, not an active damping knob.
  Omission/`None` preserves metadata value 0.3; explicit use emits a compatibility
  warning without changing the physical update.
  Use the explicit dynamics `dashpot`/`drag` controls for numerical dissipation;
  neither substitutes for a viscoelastic material. Stretch internally uses
  dynamics with `dt=1e-5`; its automatic step count is a heuristic, not a
  convergence certificate.
- Material dynamics additionally requires full stable-ID maps
  `nodal_masses_by_id`, `reference_areas_by_id`, `rest_lengths_by_id`, positive
  `max_abs_strain`, and `state_units` declaring `m`, `s`, `kg`, `world_xyz`.
  Initial rest lengths must equal initial geometric lengths. Do not supply
  `stiffness`, `spring_k`, `nodal_masses`, a displacement schedule or the elastic
  `record_energy_ledger`. Initial velocities are a sparse stable-ID map.
  See [dynamic axial viscoelasticity](DYNAMIC_AXIAL_VISCOELASTICITY.md).
- An optional synchronous `state_observer` receives completed immutable snapshots;
  it cannot be combined with a nonempty displacement schedule or energy ledger.
  The elastic ledger requires `record_state=True`; only that path permits zero
  dynamics steps. Observation does not advance constitutive state.
- FEM facade stretch infers `dim` from `graph.dimension` unless explicitly 2 or 3.
  Direct `BeamFrameFEM.stretch_test` instead defaults to 2. Its 2-D
  `nonlinear=None` selects updated-geometry increments for |λ−1|>0.3; this is not
  a nonlinear material law. Every selected 2-D incremental call warns, including
  explicit `nonlinear=True` at small stretch: final nonlinear equilibrium is
  unverified and final stiffness, reactions and nodal moments are unavailable.
  With 2-D `nonlinear=False`, |λ−1|>0.3 warns about the small-deformation linear
  model instead. The 3-D solver is linear only: do not request `nonlinear=True`
  (rejected before solving). In 3-D, `None` preserves the linear solve;
  |λ−1|>0.3 emits an applicability warning. This unchanged dispatch/warning
  threshold is not a physical validity bound, and silence is not a qualification
  of any particular network or loading.
  `BeamFrameFEM(E=1e9, nu=0.3)` resides in
  `fibernet.ml.beam_frame_fem` for historical reasons. Its raw `solve_2d` and
  `solve_3d` expose DOFs, constraints, loads, reactions and beam resultants, not
  the facade's automatic grip protocol.
- `ReducedBeamSolver`, `ReducedBeamConfig`, and `ReducedBeamResult` in
  `fibernet.sim.reduced_beam` are aliases for the separate NumPy Engine2 reduced
  model (degree-2 bending surrogate and nodal contact), not beam-frame FEM,
  LAMMPS, or a supported facade backend.

## Typed electrical, thermal and one-way coupling routes

| Entry point | Explicit physical inputs and controls | Return |
|---|---|---|
| `fibernet.sim.linear_transport.solve_linear_transport` | Edge `conductance_S`, node `dirichlet_voltage_V`, declared `length_unit` | `LinearTransportResult` |
| `fibernet.sim.transient_thermal.solve_transient_thermal` | Node `capacity_J_per_K`, edge `conductance_W_per_K`, `initial_temperature_K`, `times_s`, optional `source_W` and `fixed_temperature_K`, declared `length_unit` | `TransientThermalResult` |
| `fibernet.sim.joule_heating.assemble_joule_heating` | Already solved electrical result, explicit `endpoint_weights`, optional external source, declared `length_unit` | `JouleHeatingLoad` |
| `fibernet.sim.joule_heating.run_joule_heating` | Same electrical-to-thermal load mapping plus explicit thermal inputs and time grid | `JouleHeatingRun` containing the load and thermal result |

Transport is linear DC conduction; thermal uses backward Euler on a lumped
network. Joule heating is **one-way** electrical dissipation → thermal source.
It does not recalculate electrical conductances with temperature, solve thermal
expansion or provide fully coupled multiphysics. See the executable
[one-way Joule example](../examples/one_way_joule_heating.py) and
[Joule contract](ONE_WAY_JOULE_HEATING.md).

## IDs, rows, units and coefficient dimensions

| Input family | Identity/time convention | Unit behavior |
|---|---|---|
| Taichi `fixed_nodes`, `displacement_schedule`, `external_force` | Insertion-order node **rows**, not stable IDs; schedule waypoints use **steps**, not seconds | Ordinary `state_units` are labels, not conversion or SI validation |
| Taichi `nodal_masses_by_id`, `rest_lengths_by_id`, `reference_areas_by_id`, `initial_velocities_by_id` | Stable node or edge IDs as named | Material mode requires exact SI declaration and explicit SI numerical inputs |
| FEM raw solves | DOF/row arrays and constraints defined by the raw solver | Graph coordinates and radii used as supplied; choose a consistent `E` unit system |
| FEM `history` | Stable node-ID constraints and load maps; `times` in seconds | `length_unit` is `m`, `mm`, or `um`; geometry/radii converted to SI. ux/uy in m, rz in rad; Fx/Fy in N, Mz in N·m |
| Electrical/thermal/Joule | Stable node/edge IDs; thermal `times_s` in seconds | `length_unit`: `m`, `cm`, `mm`, `um`, `nm`; geometry declaration does not infer physical coefficients |
| Bounded LAMMPS | Stable node/edge/angle IDs; `RunSpec.timestep_s` and integer steps | Declared graph length converted to SI for native work; supplied masses/forces/velocities/bond parameters already SI |

Build row mappings from the relevant input/result identity metadata; do not
assume sorting or identical row orders across solvers. FEM history uses constant
constraint masks; `None` denotes a free DOF. See the complete
[viscoelastic frame example](VISCOELASTIC_NETWORK.md).

Dimensional distinctions are part of the API:

- Taichi `spring_k` is force-like **EA in N** under consistent SI units:
  F=k·ε, U=½k·ε²·L₀. A conventional extension stiffness is k/L₀ in N/m.
- LAMMPS `HarmonicBond.stiffness_N_per_m` is **N/m**, with
  F=k·(L−L₀). The compiler uses LAMMPS harmonic coefficient k/2.
- Electrical edge conductance **S** is not material conductivity **S/m**.
- Thermal edge conductance **W/K** is not conductivity **W/(m·K)**.
- Nodal thermal capacity **J/K** is not specific heat **J/(kg·K)**.

Graph radius, density or `Material` attributes do not silently supply these
quantities. A graph unit label alone does not convert graph geometry or make
arbitrary numerical inputs SI. Use each solver's explicit conversion contract.

## Bounded LAMMPS selectors and legacy separation

The explicit route is `prepare_harmonic_beads` or `prepare_cg` → inspect the
`PreparedRun` contract → `LAMMPSTask` → parsed result. The facade is a synchronous
wrapper around that task, not another launcher. `RuntimeSpec.mode` accepts only
`"cli"` or `"cli_serial_static"`; it is separate from `simulate(mode="dynamics")`.
Runtime identity, working-directory ownership and resource limits are explicit;
there is no automatic engine installation, discovery or fallback.

Direct harmonic facade calls require `length_unit`, `maximum_bond_length_m`,
`masses_kg`, `bonds`, `run`, `runtime`, `workdir`, `limits`. Optional controls are
`fixed_node_ids`, `forces_N`, `velocities_m_s`, `record_state`, `profile`, and
`output_stride`. With `cg_request`, only `runtime`, `workdir`, `limits`, and
optional `record_state` accompany it; do not mix direct harmonic options in.
`RunSpec` requires `timestep_s`, `steps`, `sample_every`, `box_m` explicitly.
The graph must be 3-D, nonperiodic and straight-edged; graph materials are
provenance, not force-field generation.

| Request family | Accepted profiles | Important boundary |
|---|---|---|
| Harmonic | `None`, `harmonic_large_v1`, `harmonic_sampled_v1` | Large uses 32 steps and every-step samples; sampled uses 1–1024 steps and ≤129 output frames, with native every-step recording. `output_stride` is for sampled profile only. |
| CG with projectile | `None`, `cg_network_sampled_v1`, `cg_network_mixed_sampled_v1` | Network profiles require `pair_encoding="static-groups-zero-cutoff/1"`, explicit `thermo_every`, and aligned geometry/thermo samples. Pair roles are `attractive_site` or `repulsive_site`. |
| Pair-free pure-fiber CG | `None` only | `projectile_node_id=None`, `projectile_pairs={}`, `pair_encoding=None`; 2–64 nodes, ≤128 bonds, ≤128 angles, 0–1000 steps, every-step samples. |

The optional CG `neighbor_policy` is `None` or the strict `cached-nsq-si/1`
policy record, not a free-form native command. Profiles are bounded input/output
contracts, not claims of arbitrary-network accuracy or performance. No 144-edge
pure-fiber sampled profile or finite-axial coupled-material route is exposed by
these public entry points. See [harmonic CLI](integrations/LAMMPS_HARMONIC_CLI.md),
[CG network](integrations/LAMMPS_CG_CROSSLINKED_NETWORK.md), and
[pure-fiber CG](integrations/LAMMPS_CG_FIBER_ONLY.md).

**The public legacy `LAMMPSBridge`, `LAMMPSResult`, and `run_lammps_md` remain
available.** They operate on `FiberNetwork`; `LAMMPSBridge.run_simulation`
directly instantiates the Python `lammps()` interface. They do not delegate to
`RuntimeSpec`/`LAMMPSTask`. This legacy Python-library route is unvalidated and
outside the current bounded CLI qualification scope. An import-availability
flag is not native-runtime validation. Consult the preserved
[LAMMPS license/provider notice](../fibernet/integrations/LAMMPS_NOTICE.md);
optionality or CLI separation does not establish a blanket licensing exemption.

## Results are typed, not physically interchangeable

| Result source | `energy` | Force meaning |
|---|---|---|
| Ordinary Taichi springs | Axial stored potential energy | `edge_forces`: signed axial force; `max_force`: largest absolute axial force |
| Taichi material dynamics | Committed axial stored energy; separate loss/bookkeeping records | Signed A₀·stress |
| FEM facade | `axial_stress_proxy`, not complete beam bending energy | First axial component of beam resultants; full resultants remain in metadata |
| LAMMPS facade | Final native potential energy in J, including CG bond/angle/pair terms when present | Edge forces: harmonic bond contribution only. State nodal forces: native net forces after fixes, not edge force sums |

Do not compare identically named fields without checking their model, units and
availability metadata. Unavailable FEM reactions/stiffness are not measured
zeros. `SimResult.mode` alone does not preserve the entire protocol: Taichi
stretch can return `mode="dynamics"`, and FEM returns `"fem_beam_frame"`.
LAMMPS geometry-facing arrays return to graph units while `dynamics_state` stays
SI. `record_state=False` does not retain its native trajectory.

`simulate(save_path=...)` calls the result family's default save. For
`SimResult` this is a **summary**, not complete geometry/trajectory export;
use `result.save(path, detailed=True)` for detailed arrays. Some specialized
records, including material dynamics metadata, are retained even in summaries.
`ViscoelasticResult.save` instead writes its own full history schema. Use the
matching loader; `SimResult.load` is not a universal loader for electrical,
thermal or viscoelastic results. Neither summary nor detailed export is a
general restart checkpoint or a certificate that material dynamics can resume.

### Saved LAMMPS displacement/reference consistency

New `record_state=True` results retain `dynamics_state.displacement_reference`
(`lammps_original_si_reference/1`): the exact original graph coordinates in SI,
stable node-row IDs, declared arithmetic and existing prepared/graph source IDs.
Detailed export/load checks numeric binary64 `(final_SI - original_SI) / scale`
in that operation order; signed zero is equivalent. Sample 0 is only
tolerance-close to the original graph and is **not** the displacement reference.

`result.lammps_displacement_relationship()` reports `consistent` only when that
saved relationship is checked. This is internal consistency, **not** graph
authenticity, native accuracy or physics qualification. Malformed declared
references and inconsistent displacements raise, including before summary save.
Summary export projects `dynamics_state.displacement_reference` without its
coordinate and node-ID arrays, retaining an `unavailable / omitted_from_summary`
descriptor. Unrelated historical user metadata keeps its existing behavior. Historical reference-absent
results remain readable and reference-absent (`unavailable / not_recorded`);
loading them does not retroactively certify or invalidate their physics.
Removing an optional descriptor likewise loses checkability and reports
`unavailable`, never `consistent`; the format is not tamper-authenticating.
`record_state=False` likewise has no exact saved-reference check. Neither export
is a restart. Storage forecasts include the new reference and conversion copies;
resource caps and native behavior are unchanged.

## Minimal explicit example

```python
from fibernet import pattern_2d, simulate

# SI units: coordinates and radius in m; E in Pa; forces in N.
graph = pattern_2d(unit="honeycomb", box=(10, 10), grid=(4, 4), radius=0.05)
result = simulate(
    graph, backend="fem", mode="stretch", stretch_ratio=1.01,
    dim=2, nonlinear=False, E=1e9, nu=0.3,
)
result.save("frame_detailed.json", detailed=True)
```

This is a small-strain beam-frame example, not a material calibration or a unit
conversion. For advanced controls use the linked complete examples instead of
adding unsupported mode/material/backend combinations to this call.


## Prescribed-temperature mechanics

`fibernet.sim.thermoelastic` exposes the existing experimental planar sparse
thermoelastic owner and its detached snapshot type. The network entry point
keeps its 192-node/256-edge defaults; an explicit `ThermoelasticSizeLimits`
record can select different node and physical-edge ceilings; the conservative
assembly-contribution ceiling is derived read-only metadata. The saved-temperature adapter accepts the same record and delegates
to that owner. There is no new graph class, backend registry entry,
`simulate(...)` dispatch or task supervisor. Size limits do not guarantee
factorization memory/time or confer a large-task MCP capability. See
[THERMOELASTIC_NETWORK.md](THERMOELASTIC_NETWORK.md) for contracts and units.
