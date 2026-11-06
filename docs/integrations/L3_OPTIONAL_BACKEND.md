# Prepare a StructureGraph for L3

`fibernet.sim.l3` converts an explicitly annotated `StructureGraph` into an
immutable, SI-normalized request for the external sealed planar L3 provider.
It preserves graph identities, physical-segment strengths, grip assignments and
provenance. This API prepares inputs only: it does not run the provider or
return simulated states, and `fibernet.simulate` has no L3 backend selection.

## Complete example

Save this as `prepare_l3_example.py`, then supply absolute paths to your external
provider directory and interpreter:

```text
python prepare_l3_example.py --provider PROVIDER_DIRECTORY --interpreter PYTHON_EXECUTABLE
```

The provider must contain the unchanged `L3_strengthCV05_portable_1.0.0` release
and its `SOURCE_IDENTITY.json`. Paths must exist and contain no symlink components.
The example declares each straight square edge to be one physical segment; it
does not infer a fracture discretization from a meshing request.

```python
import argparse
import math

import fibernet as fn
from fibernet.sim.l3 import (
    L3ProviderSpec, L3Resources, bind_l3, prepare_l3,
)

parser = argparse.ArgumentParser()
parser.add_argument("--provider", required=True)
parser.add_argument("--interpreter", required=True)
args = parser.parse_args()

material = fn.Material(
    name="explicit-example", density=1000.0,
    youngs_modulus=2.3e9, poissons_ratio=0.35,
)
graph = fn.StructureGraph(dimension=2)
for position in ((0, 0), (10, 0), (0, 10), (10, 10)):
    graph.add_node(position, merge=False)
for i, j in ((0, 1), (0, 2), (1, 3), (2, 3)):
    graph.add_edge(i, j, radius=0.5, material=material, segments=1)
graph.metadata.update(coordinate_units="mm", radius_units="mm")

edge_data = {}
for edge_id, edge in graph.edges.items():
    edge_data[edge_id] = {
        "failure_strain": 0.02,       # explicit synthetic assumption, not calibration
        "physical_span_id": edge_id,
        "source_edge_id": edge_id,
        "initially_active": True,
        "reference_length_m": math.dist(
            graph.nodes[edge.node_i].position,
            graph.nodes[edge.node_j].position,
        ) * 1e-3,
    }
model = dict(
    joint="welded", section="circular",
    constitutive="isotropic_elastic_brittle",
    loading="clamped_positive_y", contact=False,
)
binding = bind_l3(
    graph, coordinate_unit="mm", radius_unit="mm", edge_data=edge_data,
    fixed_node_ids=[0, 1], loaded_node_ids=[2, 3], model=model,
)
settings = dict(
    young_Pa=2.3e9, poisson=0.35, radius_m=0.0005, shear_correction=6 / 7,
    load_increment_m=1e-5, minimum_increment_m=1e-8,
    event_displacement_tolerance_m=1e-9, max_displacement_m=0.001,
    residual_tolerance=2e-7, timeout_s=30, post_rupture_opening_m=0.0,
    event_bisection_cap=20, frame_cap=100, notched=True,
    material_source={"note": "Explicit synthetic example material"},
    failure_parameter_status="Uncalibrated synthetic tensile-strain threshold",
    memory_budget_mb=256, reserve_memory_mb=512,
)
plan = prepare_l3(
    graph, binding, case="square-input-example", settings=settings,
    acceptance={"initial_inactive_segment_ids": []},
    resources=L3Resources(
        max_nodes=20, max_edges=30, timeout_s=30,
        total_rss_bytes=512 * 1024**2, output_bytes=32 * 1024**2, cpu_count=1,
    ),
    provider=L3ProviderSpec(args.provider, args.interpreter),
)
print(plan.status)
print("Plan SHA256:", plan.sha256)
print("Native request SHA256:", plan.native_request_sha256)
```

## Supported inputs

- **Graph:** dimension 2, finite XY positions and z exactly zero. Each edge must
  already represent one straight physical segment (`segments=1`), with no
  nonempty `internal_points`. Nodes, stable IDs and parallel edges are retained;
  nothing is projected, welded, split, deduplicated or remeshed by the adapter.
- **Material and section:** common positive E and circular radius, isotropic
  `-1 < nu < 0.5`, and G consistent with E/[2(1+nu)]. E is in Pa. Anisotropic
  stiffness tensors and nonuniform E/nu/radius reject. Material names never
  substitute values; density, stress strengths, toughness, thermal and electric
  fields are retained but are not inputs to this mechanical law.
- **Failure field:** `edge_data` exactly covers stable edge IDs. Required fields
  are `failure_strain`, `physical_span_id`, `source_edge_id`, `initially_active`
  and `reference_length_m`. Failure strain is positive and dimensionless;
  thresholds are not derived from stress or resampled. Equal span IDs require
  equal thresholds; integer `1` and text `"1"` remain distinct identities.
- **Grips:** explicit, nonempty, disjoint stable node-ID lists. Both groups
  clamp ux, uy and in-plane rotation; the loaded group moves only in +Y.
  `SNode.boundary` flags describe periodic geometric faces, not mechanical
  DOFs, and are preserved with that meaning. Contact, hinges and other load
  controls are unsupported.
- **Units:** explicitly declare `coordinate_unit` and `radius_unit`, each one
  of `m`, `mm` or `um`. Positions and radii are converted to SI; associated
  reference lengths are already in m and must match the graph. Conflicting
  coordinate/radius unit metadata rejects. The original units remain recorded.

Optional edge fields `native_lineage`, `parent_fiber_id`,
`source_named_path_id` and `provenance` remain in the binding. Preserve native
lineage when importing it; do not invent native fields for synthetic graphs or
replace a source segment identity with its parent identity. Initial inactive
edges remain present, and `acceptance.initial_inactive_segment_ids` must match
those bound flags. `notched=False` cannot reactivate them.

Recognized metadata declarations `joint`, `joint_type`, `section`,
`section_type`, `contact`, `loading`, `load_axis` and `constitutive` must agree
with the explicit model. Other metadata is retained provenance, not an implicit
mechanical control. Graph edits require explicit rebinding.

## Settings and numeric conventions

The example supplies all required settings. Unknown settings reject. Optional
`strength_field_mean` and `strength_field_cv` are checked against the initially
active, reference-length-weighted field—not an edge-count average.

Positive finite settings are `young_Pa`, `radius_m`, `shear_correction`,
`load_increment_m`, `minimum_increment_m`, `event_displacement_tolerance_m`,
`max_displacement_m`, `residual_tolerance` and `timeout_s`.
`post_rupture_opening_m` is nonnegative; `poisson` must match the graph.
`event_bisection_cap`, `frame_cap` and both memory budgets are positive integers;
`notched` is a literal bool. Material/source descriptions are explicit JSON.
E, nu and radius settings cannot override the graph properties.

`memory_budget_MiB` and `reserve_MiB` explicitly alias `memory_budget_mb` and
`reserve_memory_mb`; conflicting values reject. All denote binary MiB in this
provider. Missing budgets do not use provider defaults. `L3Resources` records
parent-process ceilings; preparation enforces input counts and timeout
compatibility, not runtime memory/time/disk limits. In particular, the provider's
incremental-RSS guard is not total-process memory enforcement.

JSON settings and associated numbers accept finite literal Python int/float,
not booleans or NumPy scalar wrappers. Graph-native E/nu/G/radius additionally
accept finite real NumPy scalars: integers convert exactly, and floating values
must round-trip through Python float without precision loss. Wider longdouble
values that cannot do so reject, as do bool, complex, datetime/timedelta,
masked and object values. IDs and caps are literal nonnegative signed-int64
integers. Graph arrays must be ordinary real numeric arrays.

## Returned records and errors

`L3Binding` and `L3Plan` contain detached immutable canonical JSON `bytes` and
SHA256 digests. `plan.status` is `prepared_not_executable`. Provider source
verification checks all six sealed files; it does not certify interpreter
compatibility, numerical results or runtime resource enforcement.

Canonical encoding is UTF-8 JSON with sorted keys, compact separators,
`allow_nan=False` and Python's default ASCII escaping. Rows are sorted by stable
integer IDs. The native request digest covers network/boundary/settings/
acceptance. The stronger plan digest also binds case, graph, associated data,
units, original and normalized settings, resource ceilings and provider spec.
An integrity hash is not external authentication.

`L3InputError` reports unsupported, stale or inconsistent inputs.
`L3ProviderIdentityError` reports invalid paths or absent/changed provider
identity. Wrong graph/config object types raise `TypeError`.

Plans may contain private provenance and local provider paths. Review them
before sharing; the external provider and its source assets are not bundled.
