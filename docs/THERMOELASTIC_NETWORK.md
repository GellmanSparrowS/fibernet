# Prescribed-temperature planar thermoelastic networks

This experimental Python API provides a static, small-deformation, welded
Euler–Bernoulli frame snapshot on the authoritative `StructureGraph`. It uses
the existing SciPy sparse CPU assembly, constrained solve and recovery path.
It is not another graph representation, backend or `simulate(...)` mode.

## Direct temperature input

```python
from fibernet.sim.thermoelastic import (
    ThermoelasticSizeLimits, solve_thermoelastic_network_snapshot,
)

# Caller-selected size ceilings, not a prediction of solver cost.
limits = ThermoelasticSizeLimits(
    max_nodes=5000, max_edges=6000,
)
snapshot = solve_thermoelastic_network_snapshot(
    graph,
    length_unit="m",
    E_Pa=youngs_modulus,
    expansion_alpha_per_K=alpha_by_edge_id,
    stress_free_temperature_K=reference_temperature,
    temperature_K=temperature_by_node_id,
    constraints=prescribed_components_by_node_id,
    loads=nodal_loads_by_node_id,
    size_limits=limits,
)
result = snapshot.to_dict()
```

All parameter values above must be supplied for the intended model. A synthetic
material or support scenario must be identified as synthetic, not calibrated.
Equivalent imports are available from `fibernet.sim`.

Omitting `size_limits` retains the existing 2–192-node / 1–256-physical-edge
network profile, including its errors and result schema. Opt-in limits have no
default values: both positive integer ceilings are required (at least two
nodes and one edge). Booleans, floats and mappings are not
limit records. The direct network entry point checks counts before its mechanics
geometry preparation and sparse assembly. Saved-temperature binding separately
validates the complete source and graph; it is not a size-bounded reader.
No graph identity, particular size or special support plane is
privileged. An explicit record can also impose stricter ceilings.

`limits.max_assembly_entries` is derived read-only metadata, equal to
`36 * limits.max_edges`; it is not a third constructor argument. It bounds the
conservative **36 contributions per physical
6-DOF element before duplicate sparse entries are coalesced**. Therefore it also
bounds assembled CSR nonzeros, but is not the measured CSR count. Parallel
physical members contribute and count separately. Callers choose only node and physical-edge ceilings; no matrix-entry multiplier
is needed to construct the opt-in record. The derived ceiling may exceed the
actual coalesced nonzero count.

These size checks do **not** bound sparse factorization fill, workspace, peak
RSS, elapsed/CPU time, serialization cost or output bytes. They do not set thread
counts or isolate jobs. Callers must arrange separate resource enforcement when
required, especially for untrusted inputs. Increasing size limits is not a
numerical or physical qualification of an arbitrary graph. This Python API does
not establish an MCP task-isolation or large-job capability.

## Same saved-temperature adapter

```python
from fibernet.sim.saved_temperature import prepare_saved_temperature

frame = prepare_saved_temperature(
    graph, saved_thermal_result, frame_index=selected_frame, length_unit="m",
)
result = frame.solve(
    graph, profile="network", size_limits=limits,
    E_Pa=youngs_modulus,
    expansion_alpha_per_K=alpha_by_edge_id,
    stress_free_temperature_K=reference_temperature,
    constraints=prescribed_components_by_node_id,
    loads=nodal_loads_by_node_id,
)
```

Preparation selects and detaches one existing temperature row without solving.
It neither interpolates nor reruns upstream physics. Exact stable-ID coverage,
oriented endpoints, coordinates and declared length units are checked; graph
mutation after preparation is rejected. Incomplete accepted prefixes require
explicit `allow_partial=True` and remain labeled incomplete. See
[SAVED_THERMAL_MECHANICS.md](SAVED_THERMAL_MECHANICS.md) for transfer provenance
and reader limits. `size_limits` is rejected for `profile="elementary"`.

## Model, inputs and rejection conditions

- A connected planar XY graph with no isolated nodes, straight members and
  positive finite circular radii is required. Collinear internal points must be
  ordered along their edge. Minimum member length is 1e-12 m. Noncontiguous
  nonnegative integer IDs and physical multiedges are retained; geometry is not
  merged, trimmed, rescaled or subdivided automatically.
- Declare coordinate/radius units as `m`, `mm` or `um`. Use a common positive
  finite `E_Pa`, a finite expansion coefficient in 1/K for **every physical edge**,
  positive temperature in K for **every node**, and a positive stress-free
  temperature in K. Material fields or density on the graph do not supply these
  parameters. Density is unused in this static model.
- Constraints map stable node IDs to `(ux, uy, theta)` in m, m, rad, with `None`
  for each free component. Prescribed values may be nonzero. They must restrain
  all three rigid modes. Loads map stable node IDs to `(Fx, Fy, Mz)` in N, N,
  N·m; omitted loads are zero. Supports and loads are never inferred from
  periodic flags, node labels, electrical electrodes or bounding boxes.
- Each edge uses the **mean of its endpoint temperatures** for constant axial
  eigenstrain. Recovered axial stress follows that mean projection. It is not
  pointwise stress from a linearly varying thermal strain; no extra temperature
  variance energy term or through-section thermal bending is introduced.
- Existing numerical applicability checks remain: absolute endpoint thermal
  strain ≤1e-5; mean total axial strain and total/elastic extreme-fiber strain
  ≤1e-4; nodal rotations and Hermite slopes ≤1e-3 rad. Finite values, support
  rank, recovery, free residual and global force/moment balance are checked.
  Singular, nonfinite or out-of-range states raise; there is no regularization,
  automatic retry, alternative solver fallback or partial-success result.

These checks are not material calibration, junction validation, slender-beam
validity, buckling/nonlinear stability, shear correction or mesh convergence.
In particular, passing the absolute length check does not establish a suitable
length/diameter ratio. There is no inertia, new thermal solve, thermal/electrical
feedback, coupled energy conservation claim or mechanical history integration.

## Output compatibility

`ThermoelasticSnapshot.to_dict()` returns detached JSON-compatible data in stable
ID order: geometry and explicit inputs, displacement, strains, stresses,
moments, edge forces, reactions, residual/balance and applicability diagnostics,
units and input identity. The default network profile is unchanged. Explicit
limits use profile `thermoelastic_2d_network_bounded/1` and add `size_admission`,
which records the requested node/edge ceilings in `limits`, the read-only
contribution ceiling in `derived_limits`, and actual counts. Its admission
contract is `thermoelastic_network_size_limits/2`. This metadata is part
of the canonical input hash; all existing mechanical result fields remain.
The saved adapter keeps its existing envelope and embeds this mechanical result.
A changed profile/hash is expected even if a small graph yields identical
numerical fields under the two admission routes.
