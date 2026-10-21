# Prepare straight-chain models for Exudyn

FiberNet validates a straight-chain `StructureGraph` and compiles deterministic
SI section, degree-of-freedom and contact-connectivity records. Physical IDs,
edge orientation and material provenance are preserved throughout.

Create a graph, declare the SI request, then call `prepare_exudyn_contact()`.
The result provides a source snapshot, compiled mapping and content identities
that can be inspected and revalidated:

```python
from fibernet import Material, StructureGraph
from fibernet.sim._exudyn_contract import (
    ChordContactSpec, ExudynContactRequest, SphereSpec, StepPhase,
    prepare_exudyn_contact, validate_prepared,
)

graph = StructureGraph(dimension=3)
material = Material(
    name="example", youngs_modulus=1e6, poissons_ratio=0.3, density=1200.0,
)
left = graph.add_node([-0.001, 0.0, 0.0], merge=False)
right = graph.add_node([0.001, 0.0, 0.0], merge=False)
graph.add_edge(left, right, radius=50e-6, material=material, segments=1)

request = ExudynContactRequest(
    profile="straight-shared-chain/1", length_unit="m",
    clamped_node_ids=(left, right), shear_correction=(1.0, 1.0),
    sphere=SphereSpec(
        radius_m=350e-6, mass_kg=1.8e-7,
        center_m=(0.0, 0.00062, 0.0), velocity_m_s=(0.0, -0.1, 0.0),
    ),
    contact=ChordContactSpec(k_N_per_m2=100.0, w_N=2e-6, range_m=2e-4),
    phases=(StepPhase(start_s=0.0, end_s=1e-4, steps=20),),
    geometry_energy_budget_J=1e-11,
    gravity_m_s2=(0.0, 0.0, 0.0), external_forces_N=(),
    initial_fiber_state="reference_rest",
)
prepared = prepare_exudyn_contact(graph, request)
assert validate_prepared(prepared)
assert prepared.manifest()["execution_available"] is False
mapping = prepared.mapping()
assert mapping["counts"]["ODE2"] == 15  # 6 per graph node, 3 for sphere
```

## Supported inputs

- A single simple path with 2–257 nodes and 1–256 edges, exactly on the x-axis,
  with distinct positions and `|x| <= 1 m`. Every element is 1 micrometre to
  10 millimetres long, has `segments=1` and `internal_points=None`. Curves,
  empty internal-point arrays, branches, parallel edges, refinement and
  welded/expanded networks raise validation errors.
- Explicit metre coordinates, positive circular radii and isotropic SI
  material values: Young's modulus and shear modulus in Pa, density in kg/m³,
  and a consistent Poisson ratio in `(-1, 0.5)`. Both shear-correction factors
  are positive.
- The request's two endpoint clamps determine the constraints; existing node
  boundary flags remain source provenance. Gravity and external forces must
  be zero/empty; the fiber starts at reference rest.
- A positive sphere radius/mass, finite centre/velocity, positive contact
  stiffness/range, nonnegative adhesion parameter and geometry-energy budget,
  all in finite SI values.
- One to sixteen contiguous phases starting at zero, with at most one million
  positive integer steps total. The declared schedule determines payload-size
  forecasts.

Validation failures raise `ExudynContractError`, with a machine-readable `code`
and an optional physical `source_id`; incompatible object types raise
`TypeError`.

## Prepared records

`StructureGraph` remains authoritative and unchanged. The result stores
detached canonical JSON as immutable bytes, plus SHA-256 identities for the
source, request and compiled mapping.
Accessors return fresh dictionaries. `validate_prepared()` revalidates the
request and recompiles the mapping to check semantic consistency independently
of the stored hashes. Hashes identify content; they do not authenticate its
origin.

The mapping retains physical IDs, edge orientation and material provenance;
compilation ordinals are separate from source IDs. Shared nodes receive six
planned coordinates and the sphere receives three. Native object/node indices
are `None`; zero contact-mass entries describe the planned layout. Payload
forecasts cover named numeric arrays, excluding headers, contact data,
diagnostics, traces and metadata; total memory and disk use require a separate
estimate.

Signed in-memory physical IDs are preserved. The core graph JSON reader does
not accept negative IDs, so consult
`source_graph_json_roundtrip_supported` before attempting that round trip.

## Scope

The underscore-prefixed modules provide experimental, preparation-only direct
imports. Native execution, contact-force evaluation, restart, task objects and
`easy.simulate(backend="exudyn")` dispatch are outside this API.
`source_capabilities()` reports the supported profile without loading or
probing a provider. Prepared records describe model inputs and layout rather
than solver results or runtime compatibility.

## Dependencies and license

Preparation uses FiberNet's existing dependencies and works without Exudyn
installed. The package includes no Exudyn extra, provider source or binaries.
FiberNet retains its MIT license. When installed separately, Exudyn is licensed
under `LicenseRef-EXUDYN-General-License`; see its
[license and third-party notices](https://github.com/jgerstmayr/EXUDYN/blob/f172eb2c4b8cc1ccb19d722749145d185b9000c6/LICENSE.txt).
