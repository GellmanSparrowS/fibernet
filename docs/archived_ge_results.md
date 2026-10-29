# Read archived expanded-endpoint GE results

`fibernet.sim.import_archived_ge_state` reads one previously validated, saved n1 geometrically exact (GE) beam-network state. It maps translations to the existing `StructureGraph` and preserves full native state in a separate immutable sidecar. It does not invoke Exudyn or any other provider.

This is a format-specific saved-result utility, not a live GE backend, arbitrary native-state validator, material compiler or restart loader. The existing straight-chain preparation profile and `easy.simulate` capabilities are unchanged.

## Use

```python
import json
from pathlib import Path
from fibernet.core.structure_graph import StructureGraph
from fibernet.sim import import_archived_ge_state

source_bytes = Path('source_graph.json').read_bytes()
header_bytes = Path('accepted_header.json').read_bytes()
state_bytes = Path('accepted_state.json').read_bytes()
trusted_hashes = json.loads(Path('trusted_hashes.json').read_text())
section_override = json.loads(Path('section_override.json').read_text())

graph = StructureGraph.from_dict(json.loads(source_bytes))
result = import_archived_ge_state(
    graph,
    source_graph_bytes=source_bytes,
    header_bytes=header_bytes,
    state_bytes=state_bytes,
    expected_sha256=trusted_hashes,
    section_override=section_override,
    units={'length': 'm', 'time': 's', 'mass': 'kg',
           'frame': 'global Cartesian'},
    occurrence_tolerance_m=1e-12,
)
translations = result.translations()
native_state = result.sidecar()
```

Obtain the expected `graph`, `header` and `state` SHA256 values from trusted saved evidence. Recomputing hashes from arbitrary supplied bytes does not independently authenticate them. Byte identity does not establish that a native assembly, section or simulation is physically correct.

The source `StructureGraph` is not modified. It must match the archived source after the existing reader's schema migration, including stable IDs, materials, geometry and metadata. Original source bytes remain separately preserved; migrated serialization is not claimed byte-identical.

## Supported archive contract

The reader admits only `fibernet.new-ge-reference-lobatto-contact/1` with `subdivisions=1`, no added subdivision nodes, straight initial beam endpoints and no initial endpoint twist. Source graph schemas3.0/3.1 are read by the existing `StructureGraph` reader. Parallel physical edges, sparse/reordered IDs and reversed native incidence are supported; identity is never inferred from an endpoint-pair dictionary.

Headers use either `fibernet.new-nodal-contact-state-trace/1` or `fibernet.linked-nodal-contact-state-trace/1`. States use the corresponding `fibernet.complete-accepted-state/1` or `fibernet.linked-complete-accepted-state/1`. The header carries:

- `ode2_size`, `ae_size`, `runtime_identity`, `static_metadata`;
- `binding.reference_frames`: native endpoint IDs to `RigidBodyRxyz` six-entry reference coordinates;
- `binding.layout.node_ltg_ode2`: six distinct global ODE2 indices per endpoint, disjoint across distinct native nodes;
- `binding.assembly.node_map`, native beam object/physical-edge/segment rows, retained-object map and `reference_plan`;
- `reference_plan.source_graph`, model/subdivision fields, beams, contact stations, retained markers and objects.

Each beam contains its physical `parent_edge_id`, segment0 of1, two native endpoint IDs, reference length/radius, section ID and `sectionData`. Native section records have finite6×6 stiffness/damping matrices, finite3×3 inertia and positive mass per length. Shape checks do not certify material stability, units inferred from values, or mechanical correctness. Endpoint frames/sections/LTG must come from previously validated native assembly evidence.

Each contact station preserves physical edge ID, native endpoint ID, station0/1 and positive reference-length weight. Active coordinate clamps must map to known endpoint occurrences; graph boundary flags remain periodic metadata and are not converted into clamps.

An accepted state contains global `step`, finite nonnegative `time_s`, header hash, full finite `q/v/a/ae`, and diagnostics containing complete current algorithmic acceleration, phase-qualified preceding algorithmic acceleration, empty ODE1/data arrays and native acceptance flags for advanced states. Linked states additionally carry `global_step`, nonnegative `local_step`, exact boolean `anchor` and parent-state hash. Header initial step/time and finite positive fixed step size define the absolute clock; tolerance must be between0 and1e−15s. A linked anchor is exactly local0 at the initial global step. No positive non-anchor may bypass native acceptance/cache requirements. Parent-link tokens are preserved and checked locally; ancestor files are not traversed.

Limits: each input byte record≤20MiB; graph nodes/edges≤10,000 each; native beams≤10,000; ODE2/AE dimensions≤200,000; retained markers/objects≤100,000 each; contact stations≤20,000. Unknown schemas, malformed records, duplicate JSON keys, nonfinite values, colliding mappings and unsupported geometry raise `ValueError`. A non-`StructureGraph` argument raises `TypeError`. No failure starts a provider.

## Explicit native-section override

The graph's `Material` may intentionally be an unassigned placeholder while the native run used explicit SI section data. The reader never silently derives or substitutes constitutive parameters. `section_override` must contain exactly:

- `policy`: `archived-native-sectionData-overrides-source-Material`;
- nonempty `reason` describing the provenance choice;
- `units`: the explicit SI/global-frame dictionary above;
- `source_graph_sha256`: original graph byte hash;
- `binding_sha256`: SHA256 of canonical JSON for the complete binding;
- `native_sections_sha256`: SHA256 of canonical JSON for beam-order section records containing `parent_edge_id`, `segment_index`, `section_id`, `radius_m`, `sectionData`.

Canonical JSON here is UTF-8 `json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)`. The override acknowledges archived provenance; it is not new calibration. The source graph Material remains unchanged.

## Output semantics

`ArchivedGEProjection` stores immutable JSON/original-byte fields. `translations()` and `sidecar()` return detached decoded copies.

Translation rows are sorted by stable physical node ID. Every endpoint occurrence of a welded junction must agree in translation within the explicitly declared0..1e−9m bound; only then is the lowest-ID occurrence selected. No translation or Rxyz rotation averaging is performed. The result preserves physical edge→native object/endpoint/segment maps and station weights, including parallel members.

The separate GE sidecar retains native q/v/a/AE, current/preceding algorithmic caches, reference frames, units, mappings, sections, recorded diagnostics and provenance. Original input bytes are accessible unchanged. The sidecar is **not** compatible with `SimResult.dynamics_state`, and it is not a restart-validity claim. Recorded provenance may contain producer-local strings: inspect before external sharing.

No GE resultant is recast as signed axial `SimResult.edge_forces`; no spring energy is calculated. Do not use `compute_detailed(stiffness=...)` to reinterpret these GE results. Energy/force observations stay in their original labelled native diagnostics.

This utility enables portable inspection of existing results, not proof of convergence, capture, continuous-fiber contact clearance or future trajectory behavior. A synthetic two-node/parallel-edge fixture exercises the public API without simulation or optional providers.
