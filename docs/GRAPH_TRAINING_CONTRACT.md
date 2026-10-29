# Graph experiment input contracts

Bind graph inputs to explicit group-isolated splits, then compute degree statistics from training graphs only. The API uses FiberNet's existing `StructureGraph` and detached `GraphBatch`; its outputs are immutable metadata records, not another graph representation.

Import the functions from `fibernet.ml.graph_training_contract`. A source checkout or distribution containing this module is required. The module itself uses the standard library, NumPy, and the existing graph-data module. Ordinary FiberNet imports also execute the package initializers and retain the package's base dependency requirements. The metadata workflow needs no Torch, PyTorch Geometric, neural model, solver, or ML extra. It does not return model-ready tensors or run training.

## Complete example

The example creates three synthetic graphs, each with two distinct parallel fibers and an isolated node. It binds explicit geometry, conductivity, and electrical conditions; assigns one independent mother group to each role; and computes statistics from the training sample. A real experiment needs enough independently generated mother groups for meaningful evaluation.

Run the [complete source example](../examples/graph_training_contract.py) from the repository root:

```bash
python -m examples.graph_training_contract
```

```python
"""Bind graph inputs, an explicit group split, and training degree statistics.

The three synthetic specimens demonstrate the metadata API, not a learning
experiment. Run from a source checkout containing graph_training_contract:
    python -m examples.graph_training_contract
"""
import json

from fibernet.core.structure_graph import StructureGraph
from fibernet.ml.graph_data import graph_batch
from fibernet.ml.graph_training_contract import (
    FrozenRecord,
    bind_graph_inputs,
    bind_explicit_group_split,
    training_degree_histogram,
    validate_pna_degree_admission,
)


def make_source(graph_id, length_m):
    graph = StructureGraph(dimension=2)
    left = graph.add_node([0.0, 0.0, 0.0])
    right = graph.add_node([length_m, 0.0, 0.0])
    graph.add_node([0.0, length_m, 0.0])  # Keep an isolated node.
    graph.add_edge(left, right, radius=1.0e-5)
    graph.add_edge(left, right, radius=2.0e-5, allow_parallel=True)

    # Each sample has its own one-graph snapshot. Physical edge IDs are retained.
    batch = graph_batch(
        [graph], graph_ids=[graph_id], length_unit="m",
        coordinate_frame="specimen_xy",
        edge_features=("radius", "chord_length"),
    )
    return {
        "batch": batch,
        "external": {
            "units": {"conductivity_S_per_m": "S/m"},
            "rows": {
                int(edge_id): {
                    "conductivity_S_per_m": {
                        "value": 1.0e4, "available": True,
                    }
                }
                for edge_id in batch.edge_ids
            },
        },
        "conditions": {
            "schema": "fibernet.electrical_conditions.v1",
            "task_rule_id": "two_terminal_1V.v1",
            "conditioning": "metadata_only_fixed_rule",
            # Ordered role slots: reference terminal, driven terminal.
            "boundary_voltages": [
                {"node_id": left, "value": 0.0, "unit": "V"},
                {"node_id": right, "value": 1.0, "unit": "V"},
            ],
            "nodal_currents": [],
        },
        # These are opaque labels, not paths that the API opens or verifies.
        "source_refs": {
            "graph": "synthetic:" + graph_id + ":geometry",
            "external": "synthetic:" + graph_id + ":conductivity",
            "conditions": "synthetic:" + graph_id + ":conditions",
        },
    }


def build_records():
    sources = {
        "train-a": make_source("train-a", 1.0e-3),
        "validation-b": make_source("validation-b", 1.2e-3),
        "final-c": make_source("final-c", 0.8e-3),
    }
    feature_spec = {
        "schema": "fibernet.graph_feature_spec.v1",
        "length_unit": "m",
        "coordinate_frame": "specimen_xy",
        "dtype": "float32",
        # Columns are x, y, z, dimension. Length scales are in m.
        "node_offset": [0.0, 0.0, 0.0, 0.0],
        "node_scale": [1.0e-3, 1.0e-3, 1.0e-3, 2.0],
        "features": [
            {
                "name": "radius", "source": {"snapshot": "radius"},
                "unit": "m", "offset": 0.0, "scale": 1.0e-5,
                "required": True, "parity": "even",
            },
            {
                "name": "conductivity",
                "source": {"external": "conductivity_S_per_m"},
                "unit": "S/m", "offset": 0.0, "scale": 1.0e4,
                "required": True, "parity": "even",
            },
        ],
    }
    inputs = bind_graph_inputs(
        [item["batch"] for item in sources.values()],
        source_groups={
            "train-a": "mother-a",
            "validation-b": "mother-b",
            "final-c": "mother-c",
        },
        feature_spec=feature_spec,
        external={key: item["external"] for key, item in sources.items()},
        conditions={key: item["conditions"] for key, item in sources.items()},
        source_refs={key: item["source_refs"] for key, item in sources.items()},
    )
    split = bind_explicit_group_split(
        inputs, train_ids=["train-a"], validation_ids=["validation-b"],
        final_ids=["final-c"], split_seed=42,
    )
    degree = training_degree_histogram(
        inputs, split, training_sources={"train-a": sources["train-a"]},
    )
    admission = validate_pna_degree_admission(
        degree, scalers=["identity", "amplification", "attenuation"],
    )

    # A transport envelope must preserve the exact payload text and digest.
    envelope = {
        "kind": degree.kind,
        "payload_json": degree.payload_json,
        "digest": degree.digest,
    }
    restored_degree = FrozenRecord(**json.loads(json.dumps(envelope)))
    restored_degree.as_dict()  # Verify the payload against this envelope's digest.
    assert validate_pna_degree_admission(
        restored_degree, scalers=["identity", "amplification", "attenuation"],
    ) == admission
    return inputs, split, degree, admission


def main():
    inputs, split, degree, admission = build_records()
    summary = degree.as_dict()["summary"]
    assert summary["histogram"] == [1, 0, 2]
    assert summary["node_population"] == 3
    assert summary["incoming_messages"] == 4
    print(json.dumps({
        "input_binding": inputs.digest,
        "split": split.digest,
        "training_degree": degree.digest,
        "histogram": summary["histogram"],
        "node_population": summary["node_population"],
        "incoming_messages": summary["incoming_messages"],
        "pna_admission": admission.as_dict()["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
```

The intended histogram is `[1, 0, 2]`: one node has incoming degree zero, no node has degree one, and two nodes have degree two. The two parallel physical fibers contribute four directed messages. Validation and final graphs contribute nothing to this histogram. The admission status is `denominator_checks_only_not_backend_validation`.

## API reference

All five operations below return a `FrozenRecord` with `kind`, `payload_json`, and `digest`. `as_dict()` verifies the record digest and returns a fresh detached dictionary. It is not a replacement for operation-specific schema validation. Do not construct scientific records by editing a payload and recalculating its digest; rebind the original sources through the API.

### `bind_graph_inputs`

Signature: `bind_graph_inputs(batches, *, source_groups, feature_spec, external, conditions, source_refs, derived_views=None)`.

- `batches` is a list or tuple of exact `GraphBatch` instances, with one graph per sample and unique nonempty graph IDs. Both `fibernet.graph_batch.v1` and `.v2` are accepted. Use one call to `graph_batch([graph], ...)` for each sample, not a multi-graph batch.
- The four required mappings have exactly those graph IDs as keys. `source_groups` maps each ID to its mother-group ID. Descendants, variants, or augmentations of the same source must share a group.
- `feature_spec` declares ordered selectors, units, offsets, scales, masks, and scalar reversal parity independently of a model. The example contains its complete closed schema. `dtype` is `float32`; there are four node offsets/scales and 1–64 edge feature declarations.
- Snapshot selectors are `radius`, `chord_length`, and `polyline_length`. The selected column must be present in the batch. External columns are explicitly supplied per physical edge, not per directed message.
- Each external table is `{"units": {column: unit}, "rows": {edge_id: {column: {"value": number_or_None, "available": bool}}}}`. Integer physical-edge keys and every selected external column must match exactly. A missing value is `None` with `available=False`, not a fabricated zero. If there are no external selectors, that sample's `external` value must be `None`.
- `source_refs` has exactly `graph`, `external`, and `conditions`, each a nonempty string. These are provenance annotations; the API does not read a named file or authenticate its contents.
- Optional `derived_views` maps every graph ID to `node_features`, `edge_features`, `edge_index`, and `graph_index` NumPy arrays. The function recomputes each array from raw inputs and requires exact typed canonical equality. A prepared neural-model object is not raw authority.

The v1 metadata-only feature profile returns a `fibernet.graph_inputs.v2` payload; the [explicit Dirichlet profile](#explicit-dirichlet-voltagemask-profile) returns `.v3`. Each sample records its group, source/projection/input/topology hashes, counts, condition-profile metadata, and source references. Binding returns metadata; use `project_graph_inputs` for detached, revalidated NumPy arrays.

### `bind_explicit_group_split`

Signature: `bind_explicit_group_split(inputs, *, train_ids, validation_ids, final_ids, split_seed, algorithm="explicit_replay.v1")`.

Provide three nonempty lists or tuples of graph IDs. They must partition every bound sample exactly once, and no mother group may cross roles. The API preserves order and generates no random split. `split_seed` is an integer in `[0, 2**31 - 1]`, recorded for provenance only; it does not reproduce or certify the supplied membership. Return kind: `split`; schema: `fibernet.graph_split.v1`.

### `training_degree_histogram`

Signature: `training_degree_histogram(inputs, split, *, training_sources)`.

Pass exactly the training IDs, each mapped to a dictionary with `batch`, `external`, `conditions`, and `source_refs`, as in the example. Validation/final raw sources are not accepted in this mapping. Every training source is rebound and compared with the input binding before counting, so changed geometry, edge maps, values, masks, or conditions reject rather than silently refit.

The `degree` record has schema `fibernet.training_degree.v1`, the split digest, ordered training identities, physical-edge count, and a summary containing:

- `histogram[k]`: number of training node instances with incoming directed degree `k`.
- `node_population`: all training nodes, including isolates.
- `incoming_messages`: twice the number of retained physical edges.
- `mean_degree` and `mean_log_degree`: node-weighted means of degree and `log(1 + degree)`.

It fits no feature/target normalizer. Caller-chosen normalizers must be fixed independently or estimated from training data only, then frozen before binding. The API cannot determine how a caller obtained those numbers.

### `validate_pna_degree_admission`

Signature: `validate_pna_degree_admission(statistics, *, scalers)`.

Supported unique scaler names are `identity`, `amplification`, `attenuation`, `linear`, and `inverse_linear`. This validates the statistic record and denominator conditions corresponding to PyG 2.7.0. It returns a `pna_admission` record with schema `fibernet.pna_denominator_admission.v1`; it neither imports PyG nor constructs PNA.

Nonempty all-isolated data remain valid graph input. `identity`, `attenuation`, and `inverse_linear` pass these denominator checks; `amplification` and `linear` reject because their fitted denominator is zero. Admission alone does not establish meaningful predictions, finite values throughout a backend implementation, or model compatibility. Recompute statistics with `training_degree_histogram` when checking their relationship to raw inputs and a split; denominator admission alone cannot authenticate those sources.

### `make_lineage_sidecar`

Signature: `make_lineage_sidecar(artifact_bytes, *, result_bytes, reconstructed_inputs=None, raw_sources=None)`.

This optional metadata utility accepts original immutable JSON bytes for `fibernet.gine_artifact.v1`, kind `inference_best_validation`, and its results. It records their SHA256 hashes and original configuration, runtime, target, and development membership without modifying the bytes. Original train/validation indices must be typed integers, nonempty, disjoint, in range, and cover the original development IDs; mother groups must be disjoint.

Omit both reconstruction arguments to obtain `blocked_missing_raw_authority` with `authority_profile="legacy_prepared_only"`. Supply both arguments for a complete reconstruction: `raw_sources` must cover all IDs in `reconstructed_inputs` with the same source dictionaries used above. Accepted reconstruction yields `new_contract_lineage_only`, not old/new tensor or weight equivalence. Supplying only one reconstruction argument also leaves reconstruction blocked. There is no weight loading, state migration, or bypass of an older loader's runtime checks.

## Conditions, units, and common errors

For the v1 metadata-only feature profile, each sample needs a complete `fibernet.electrical_conditions.v1` dictionary. The [explicit Dirichlet profile](#explicit-dirichlet-voltagemask-profile) instead uses conditions `.v2` with its source declaration. Both `boundary_voltages` and `nodal_currents` are explicit lists, even when empty. Each entry has an existing stable node ID, finite numeric value, and unit `V` or `A`. IDs must be unique within each list. These records describe electrical conditions; stored geometric boundary-face flags do not impose mechanical constraints.

`conditioning="metadata_only_fixed_rule"` means every sample uses the same ordered numerical role slots: rule ID, slot counts, values, numeric types, units, and order must agree. Graph-local node IDs may differ. For example, two-terminal roles `[0.0 V, 1.0 V]` can use different node IDs in each graph, but `[0.0 V, 2.0 V]`, reversed slots, or a different number of boundary nodes is unsupported. The API does not infer physical roles from geometry and does not verify that the declared roles are physically equivalent. Boundary values are bound metadata, not additional node-feature columns.

No unit conversion occurs. Geometry and selected length columns use the declared length unit and coordinate frame. External conductivity in `S/m`, conductance in `S`, and a target expressed in `mS` are different quantities. Targets and solved fields do not belong in the input selectors. Normalization offsets and scales use the corresponding feature's units; the dimension column is dimensionless.

Invalid contract inputs raise `ValueError`. Useful categories and remedies are:

- `exact graph-ID ... required`: provide complete mappings with no missing or extra samples.
- `one graph per experiment sample required`: create separate one-graph snapshots.
- `external unit mismatch` or `geometry unit/frame mismatch`: align explicit declarations; convert source data yourself if needed.
- `required feature missing`: provide a measured/declared value, or explicitly choose a non-required selector. Do not fill missing data merely to suppress the error.
- `mixed/unencoded numerical condition rules unsupported by this profile`: keep one fixed ordered condition rule, or explicitly select the [Dirichlet profile](#explicit-dirichlet-voltagemask-profile) for supported variable boundary conditions.
- `split must exactly partition samples` or `mother groups overlap`: correct membership at the source-group level.
- `training source mutated or coalesced relative to bound source`: restore the original source or create a new binding/split; do not relabel the old record.
- `projection exceeds finite float32 range` or `raw integer loses precision before projection`: use meaningful units/scales or a supported representation without silently dropping nonzero information.
- `record digest mismatch`: restore the exact saved envelope, or recompute from raw authority.

Messages aid diagnosis but are not stable error codes. The underlying `graph_batch` factory can additionally raise `TypeError` for a non-`StructureGraph` input. Avoid concurrent mutation while creating or validating snapshots.

## Methods and interpretation

### Graph and feature profile

Samples are nonempty, with sorted nonnegative int64 node IDs, unique physical-edge IDs, dimension 2 or 3, no self-loops, and exact forward/reverse message pairs. Parallel physical fibers retain distinct identities and each contributes both directions. All-isolated samples are admissible. An internally consistent graph with coalesced edges is a different input; only comparison with a prior binding can identify that change.

The v1 metadata-only profile projects normalized `(x, y, z, dimension)`; the explicit Dirichlet profile appends fixed-scale voltage and its mask. Edge columns are selected scalar values followed by availability masks, then repeated according to physical-edge parent rows for both message directions. Only even reversal parity is supported. Arithmetic uses float64 before explicit float32 narrowing; nonfinite values and lost nonzero range reject. Lengths are recomputed from stored coordinates, with polyline length following stored internal points. Radius is read, never inferred.

GraphBatch v2 can bind stored material/boundary/segment provenance, but material names are not numerical predictors. This profile does not prepare `MaterialFeatureView` as a model input, infer electrical conductivity from a material selector, or support custom material extras or anisotropic stiffness tensors. The example therefore supplies conductivity as an explicit external column. Missing stored provenance is not replaced with a physically inferred value.

### Identity and reuse

The hash dependencies proceed from raw source and an independent feature specification to projection/input identity, then explicit split, then training statistics. Model architecture, targets, labels, optimizer state, and weights are outside this dependency chain.

Arbitrary graph/node/edge metadata are checked for bounded serialization but excluded from scientific source/input/split hashes. Put relevant pre-solve quantities in the supported raw/condition schemas. The top-level `source_refs` annotations do not change scientific input/split identity; changing one changes the outer input record digest. References inside the raw `current_declaration`, by contrast, participate in source identity. Material identities and exact raw numerical values remain in source identity even when their projections happen to match.

Changing held-out raw data requires rebinding the split and statistics link. It must not change histogram values or fitted moments for unchanged training samples. A checksum proves internal identity, not the truth, origin, or authenticity of a caller's physical assertions. Mother-group and condition-role declarations are also caller responsibilities.

Canonical encoding preserves exact numeric types, signed zero, ordered sequences, and little-endian C-order array values. Thus `1`, `1.0`, and `True` are distinct; the last is not accepted as a numerical count or ID. Dictionaries sort UTF-8 keys without Unicode normalization. Only finite values and the closed versioned schemas are accepted. No pickle or arbitrary object serialization is used. Each record digest uses SHA256 with its own domain tag.

Input envelopes are `fibernet.graph_inputs.v2` for the v1 metadata-only feature profile and `.v3` for the explicit Dirichlet profile; the unsupported input envelope `.v1` is not silently upgraded. Other public envelope schemas are listed with their corresponding operations above. Size limits are 8 MiB per bounded canonical aggregate/JSON record or array, 300,000 canonical items, depth 32, 4,096 UTF-8 bytes per string, 10,000 samples, 100,000 nodes and 200,000 physical edges per sample, and 64 edge selectors. These are format limits, not a demonstrated memory/performance envelope.

### Scientific scope

These operations support reproducible input identity, group-isolated membership, and training-only graph statistics. They do not validate a constitutive law, generate a physical label, solve a boundary-value problem, or establish a predictor's accuracy. A three-way split declaration alone does not make a final set untouched: previous inspection, selection, or tuning must be accounted for in an experiment's methods.

Historical artifact metadata can document a specific runtime without certifying it for a new package or model. No backend compatibility, forward/backward execution, training/inference quality, speedup, GPU performance, or unified ML/RL completion follows from these metadata operations. A neural adapter and its lifecycle require their own compatibility and scientific evaluation.

## Licensing and optional dependencies

FiberNet is MIT-licensed; retain the copyright, permission notice, and disclaimer in the repository's [LICENSE](../LICENSE) when redistributing copies or substantial portions. The PNA denominator profile references PyG 2.7.0, also MIT-licensed (copyright 2023 PyG Team). This module and example do not vendor PyG source, weights, or binaries. A distribution that copies substantial PyG code must carry PyG's full notice as well; a reference is not a substitute for that notice.

Do not infer that all optional dependencies share the MIT license. Installing a model backend is a separate choice governed by that backend's package and license terms. No optional dependency version pin or trainer API is added by this metadata interface.

## Explicit Dirichlet voltage/mask profile

Represent variable electrode placement, cardinality and voltage in a six-column
node projection of the existing `GraphBatch`. The explicit
`dirichlet_voltage_mask.v1` profile keeps geometry, physical multiedges and raw
condition provenance aligned through binding, projection and training statistics.

### Opt in and project

Starting with the feature specification and raw-source mapping in the example
above, select the profile and provide conditions for each sample:

```python
feature_spec = {
    **feature_spec,
    "schema": "fibernet.graph_feature_spec.v2",
    "condition_profile": "dirichlet_voltage_mask.v1",
}
sources[sample_id]["conditions"] = {
    "schema": "fibernet.electrical_conditions.v2",
    "task_rule_id": "declared_two_terminal_problem.v1",
    "conditioning": "dirichlet_voltage_mask.v1",
    "boundary_voltages": [
        {"node_id": left_id, "value": 1.0, "unit": "V"},
        {"node_id": right_id, "value": 0.0, "unit": "V"},
    ],
    "nodal_currents": [],
    "current_declaration": {
        "kind": "no_nodal_current_injections.v1",
        "source_ref": authoritative_declaration_reference,
        "source_sha256": authoritative_declaration_sha256,
    },
}
```

Bind the samples with the existing `bind_graph_inputs` call, then obtain their
arrays through the new helper. Each source retains the four fields `batch`,
`external`, `conditions` and `source_refs`:

```python
from fibernet.ml.graph_training_contract import project_graph_inputs
views = project_graph_inputs(inputs, sources=sources)
node_features = views[sample_id]["node_features"]  # (nodes, 6), float32
```

The helper recomputes each projection and checks its raw source, including source
references, against the frozen binding. Its returned `node_features`,
`edge_features`, `edge_index` and `graph_index` arrays own their memory, so callers
can modify a view independently of raw inputs, metadata and later projections.
The optional `derived_views` argument to binding checks supplied arrays against
the same authoritative recomputation.

### Feature columns and stable-ID alignment

Columns 0–3 are normalized x/y/z/dimension using the existing four offsets and
scales. Column 4 is voltage divided by the fixed scale **1 V**; column 5 is the
**Dirichlet mask**. A free node has `(+0, 0)` in the last two columns, while a
fixed 0 V node has `(+0, 1)`. Raw signed zeros and full numerical precision remain
source-bound; projected voltage zeros are canonical +0 and other values use
float32 rounding. The source, projection and input hashes retain these distinct
levels of identity. All normalization constants are fixed by the specification
or profile.

Rows follow `batch.node_ids`. The projection's `edge_index` is the directed
`batch.message_index`, with `2E` columns and `2E` edge-feature rows. The raw
`batch.edge_index` contains `E` physical edges. Both orientations of each physical
multiedge retain their alignment:

```python
batch = sources[sample_id]["batch"]  # The same successfully revalidated source.
node_ids_for_feature_rows = batch.node_ids
physical_ids_for_message_rows = batch.message_parent_ids
physical_rows_for_message_rows = batch.message_parent_rows
orientations_for_message_rows = batch.message_orientation
# physical_ids_for_message_rows == batch.edge_ids[physical_rows_for_message_rows]
```

These IDs and maps are provenance, bound by `topology_hash`, rather than predictor
columns. PNA degree statistics count incoming directed messages for every selected
training node, including isolates and parallel-edge instances.

### Current declaration and source identity

This profile describes Dirichlet electrodes with **no prescribed nodal current
injections**; electrode reaction currents may still be nonzero. Supply a
source-backed declaration and its SHA256 alongside the empty current list.
Recover that declaration from the experiment authority: absent fields, solver
defaults, geometric flags, fitted tolerances, trial potentials and solved fields
alone cannot establish it. The API binds the supplied declaration and provenance;
checking the referenced source and its scientific truth remains the caller's
responsibility.

The binding uses `fibernet.graph_inputs.v3` in the existing `FrozenRecord`.
`source_hash` covers raw geometry, material provenance, external features and
conditions; `projection_hash` covers the actual arrays and feature specification;
`input_hash` links both. Samples share a task-rule identity while their encoded
boundary assignments may vary.

### Validation and compatibility

- Supply a nonempty list of distinct known stable node IDs with finite voltages
  in V. Unlisted nodes are free. Duplicate or overlapping assignments, unknown
  IDs, non-V units, and values that overflow or underflow finite float32
  projection reject. Electrode selection and unit conversion are caller tasks.
- An explicit current declaration is required. Unknown policies and any
  prescribed-current row, including a zero-valued row, reject. Unsupported
  versions and mixed old/new profile combinations also reject.
- The v1 metadata-only profile keeps its original v2 binding identities and
  fixed numerical-slot/cardinality checks. Selecting this profile is explicit;
  four-column model weights require a separately designed compatibility path.
- Group splits and training-degree statistics support both profiles. Statistics
  accept exactly the training sources and remain independent of labels and
  held-out arrays. Raw-source mutation and supplied-array tampering reject.

The contract establishes input identity and preparation. Boundary-value
solvability, adequate electrode coverage, scientific suitability and model
performance belong to the downstream experiment's validation.
