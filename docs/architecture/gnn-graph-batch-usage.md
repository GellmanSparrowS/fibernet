# GraphBatch input for the existing CPU GCN

`fibernet.ml.gnn.graphs_from_batch(batch, *, source_groups)` converts a GraphBatch
v1 into the list of dictionaries accepted by `FiberGNN`. This optional path uses
PyTorch, with no PyG requirement. Base FiberNet and GraphBatch imports remain
backend-free. The integration environment is Linux, Python 3.12, Torch 2.7.1+cpu.
Other backend versions and GPU execution are not established by this CPU check.

```python
from fibernet.ml.graph_data import graph_batch
from fibernet.ml.gnn import graphs_from_batch, FiberGNN

batch = graph_batch(graphs, graph_ids=sample_ids,
                    length_unit='mm', coordinate_frame='specimen',
                    edge_features=('radius', 'chord_length', 'polyline_length'))
inputs = graphs_from_batch(batch, source_groups=specimen_ids)
model = FiberGNN(node_dim=4, hidden=8, n_layers=1, n_outputs=1,
                 layer_type='gcn', dropout=0)
model.eval()
# Use torch.no_grad() for inference.
predictions = model(inputs)
```

## Input meaning and retained identity

The explicit new schema is `fibernet.fibergnn.xyz_dimension.v1`, with node columns
`x, y, z, dimension`. The existing `graph_from_structure` five-column schema and
existing models/trainers are unchanged. A legacy model is not interchangeable
with a model trained on this new schema.

Coordinates use the caller's declared units and frame. There is no learned
normalization or inferred unit conversion. Raw XYZ features are not invariant to
translation, rotation or scale. Model tensors use CPU float32: finite rounding
is possible; overflow and nonzero-to-zero underflow raise ValueError rather than
clipping. Explicitly rescale a source representation and declare its units before
adapting if needed. Full original float64 geometry is retained in `snapshot`.

Each dictionary retains `graph_id`, `source_group`, `node_ids`,
`physical_edge_ids`, `physical_edge_index`, `message_parent_rows`,
`message_parent_ids`, `message_orientation`, `dimension`, `length_unit`, and
`coordinate_frame`. Physical endpoints and message endpoints are graph-local
dense rows; stable IDs are graph-local original IDs. Parent rows index the
physical edge table within that graph. Forward/reverse orientation is +1/-1
relative to each stored physical edge's node_i→node_j, even for reversed parallel
members. `n_edges` counts physical edges, not messages.

`edge_features` and boolean `edge_feature_available` are message-aligned: each
physical row is repeated for its two messages. The physical feature table,
internal polylines, original masks, metadata and global pointers remain in the
referenced `snapshot`; `snapshot_graph_row` identifies the graph there. No
material, boundary condition, target or unavailable feature is invented.

GCN consumes only node features and normalized adjacency. It does not consume
edge feature values, masks, internal polylines or physical IDs as learned inputs.
Their retention supports attribution and later consumers; it does not establish
an edge-conditioned or physics-informed model. Parallel messages each contribute
to adjacency and degree normalization. Duplicate COO entries sum numerically;
coalesced adjacency cannot reconstruct individual physical IDs. Computational
self-loops exist only in adjacency and have no physical parent mapping. The
existing edge-free GCN helper returns zero adjacency without loops; this behavior
is preserved. Zero-node graphs and zero-graph batches are rejected before pooling.

Returned tensors copy numerical storage. Mutating them does not alter the
snapshot. `snapshot` itself is a reference, not a fresh copy: do not mutate it
while using adapted inputs. Its read-only NumPy convention is not protection
against deliberate mutation. Structural checks reject forged pointer/index
arrays, boolean or fractional indices, malformed parent mappings and masked arrays
before conversion. The retained snapshot is not a new serialization format.

## Source-group holdout and training

`source_groups` must explicitly name every graph. These are caller assertions of
scientific origin; distinct graph IDs do not prove independence. Keep augmented,
remeshed, load-step and multi-fidelity variants from one source in one group.

For the small integration fixture, `GroupShuffleSplit(n_splits=1, test_size=.25,
random_state=42)` selects graph rows before optimization, using the same splitting
rule as native Dataset training. Record train/held-out graph IDs and source groups
and assert disjointness. Pass only train dictionaries and aligned train labels
to optimization. This adapter adds no fitted preprocessing. Any future fitted
scaler must fit only train groups and have its configuration/state recorded.

The existing `train_gnn` helper internally uses a row split and is not a
source-group-safe training entry point. The integration test therefore uses an
explicit two-step optimizer loop after group splitting; it does not introduce a
new training framework or change existing helpers. Synthetic fixtures demonstrate
execution, identity, leakage prevention and reproducibility, not predictive
superiority or measured physical validity.

## Same-runtime checkpoint example

Existing `save_model_bundle` and `load_model_bundle` remain unchanged. Save the
four feature names, target names, model constructor configuration and input schema
using their existing metadata fields, together with units/frame and split
provenance. Store JSON-native metadata only, not the snapshot sidecar. Instantiate
the same model configuration when loading. Check stored schema and feature names
against the intended input contract before inference; the existing loader does
not enforce these checks automatically.

Only load your own trusted bundle in the tested runtime. The existing loader uses
pickle-capable `torch.load`; do not load untrusted files. No cross-version durable
format is promised. Existing `torch.save` writes are not atomic and do not
promise to preserve a prior file on failure. This path does not inherit the
separate easy-model writer's safeguards.

The regression gate includes exact same-runtime save/load predictions and
held-out-target perturbation invariance. Graph/node/edge order comparisons use
rtol=1e-5 and atol=1e-6 because floating-point reduction order can change. Missing
Torch raises ImportError; no alternate model is selected silently.

Backend semantics: [Torch 2.7 sparse tensors](https://docs.pytorch.org/docs/2.7/sparse.html).
A future PyG adapter also needs explicit identity-field batching rules:
[PyG batching](https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html).

## Existing GAT and SAGE modes

The same four-column adapter also has CPU float32 integration coverage for
`FiberGNN(node_dim=4, hidden=8, n_heads=4, n_layers=2, n_outputs=1,
dropout=0, layer_type='gat')` and the same configuration with `layer_type='sage'`.
These are existing implementations, with unchanged model and artifact semantics.
They consume message endpoints directly; they do not use GCN adjacency or its
computational self-loops. Edge features, availability masks and physical IDs
remain attribution data rather than learned inputs in these modes too.

Parallel message occurrences remain present. SAGE averages incoming occurrences
before its L2 normalization; GAT normalizes attention across incoming occurrences.
Duplicating one source message can change its relative influence. Duplicating
all incoming messages equally can leave the result unchanged. Preserving physical
IDs does not make either model distinguish identical members by ID.

Two existing GAT configuration limits matter:

- With two or more layers, use a positive head count dividing the hidden width.
  For example hidden=7, heads=4 constructs but fails later at LayerNorm. This
  test extension documents that behavior rather than changing the constructor.
- `FiberGNN(dropout=0)` does not propagate that setting into the attention layers;
  their effective training attention dropout remains 0.1. `eval()` disables it.
  Reproducible training requires resetting seeds before the full run. Do not
  interpret the configuration as disabling every dropout during training.

Nonempty edge-free graphs execute; zero-node graphs remain rejected by the
adapter. GPU/arbitrary dtype support is not established by these CPU checks.
The regression fixtures exercise finite forward/gradients, relabeling and order
consistency, analytic duplicate-message aggregation, and edge-feature
nonconsumption. Each mode also has a bounded two-step train-only source-group
fixture, held-out-label perturbation invariance and an own-bundle same-runtime
roundtrip. They add no new trainer, accuracy claim or physics validation. Existing
checkpoint trust, schema-checking and non-atomic-write caveats above still apply.
