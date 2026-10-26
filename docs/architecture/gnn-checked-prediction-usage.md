# Checked graph prediction with the existing FiberGNN

`fibernet.ml.gnn.predict_gnn(model, batch, *, source_groups, input_contract,
target_name)` is an additive, checked prediction seam for the existing CPU
float32 scalar GCN/GAT/SAGE models. `from fibernet.ml import predict_gnn` is also
available lazily. Low-level `model(graphs)` and existing training/bundle behavior
are unchanged.

## Existing training and own-bundle workflow

Use the model and result from an existing group-training call:

```python
from fibernet.ml.gnn import predict_gnn

prediction = predict_gnn(
    model, batch, source_groups=source_groups,
    input_contract=training_result['training_data'],
    target_name='chosen_property',
)
```

Here `training_result` is the actual result returned by `train_gnn(...,
split='group', groups=...)`; `batch` is the GraphBatch v1 to predict. The target
name is chosen explicitly, not recovered or interpreted from the training record.
After loading an own trusted bundle, pass `info['metadata']['training_data']`
instead, as in the [group-training example](gnn-group-training-usage.md).
The helper does not deserialize any file. Existing pickle trust and same-runtime
limitations remain in force.

## Fresh untrained interface example

This example illustrates input compatibility, not predictive quality. Its
explicit declaration is not a training record.

```python
import numpy as np
import torch
from fibernet.core.structure_graph import StructureGraph, SNode, SEdge
from fibernet.ml.graph_data import graph_batch
from fibernet.ml.gnn import FiberGNN, predict_gnn

graph = StructureGraph(3)
for node_id, point in [(4, [0., 0., 0.]), (19, [1., 0., 1.]),
                       (73, [2., 1., 2.]), (200, [4., 4., 2.])]:
    graph.nodes[node_id] = SNode(np.array(point), node_id=node_id)
graph.edges[90] = SEdge(4, 19, edge_id=90, radius=None,
                        internal_points=np.array([[.2, .5, .6]]))
graph.edges[3] = SEdge(19, 4, edge_id=3, radius=.1)
graph.edges[21] = SEdge(19, 73, edge_id=21, radius=.1)
batch = graph_batch([graph], graph_ids=['example-a'],
                    length_unit='mm', coordinate_frame='specimen',
                    edge_features=('radius', 'chord_length', 'polyline_length'))
torch.manual_seed(42)
model = FiberGNN(node_dim=4, hidden=8, n_layers=1, n_outputs=1,
                 layer_type='gcn', pooling='mean', dropout=0)
contract = {
    'input_schema': 'fibernet.fibergnn.xyz_dimension.v1',
    'feature_names': ['x', 'y', 'z', 'dimension'],
    'length_unit': 'mm', 'coordinate_frame': 'specimen',
    'model': {'class': 'FiberGNN', 'node_dim': 4, 'hidden': 8,
              'n_outputs': 1, 'n_layers': 1, 'layer_type': 'gcn',
              'pooling': 'mean', 'dropout': 0., 'head_dropout': 0.,
              'layers': [{'class': 'GraphConvLayer'}]},
}
result = predict_gnn(model, batch, source_groups=['example-origin'],
                     input_contract=contract, target_name='example_property')
```

The model has fixed initial weights and has not learned the named property.
This API neither checks that training occurred nor supplies a useful pretrained
model. One graph and one source group are valid. Prediction IDs/groups may
also overlap an earlier training collection: this function does not claim to
create or certify a final-test set.

## Input and output meaning

Only five compatibility fields are required: input_schema, feature_names,
length_unit, coordinate_frame and model. Fresh declarations need no top-level
record schema. If schema is supplied, only `fibernet.gnn_group_training.v1` is
accepted. Existing graph_ids, split, training and model_selection bookkeeping
is ignored, not validated, interpreted or copied. Other top-level fields reject;
normalization or preprocessing declarations are not silently consumed.

The model field must match the existing recorded configuration keys and values,
including pooling, head dropout and per-layer settings such as GAT's effective
attention dropout. This compares recorded fields, not every architectural detail:
for example, that record does not encode LayerNorm epsilon, arbitrary internal
activation changes or hooks. It is not a model-state digest or an authenticated
binding between weights and a declaration.


Input must be an existing GraphBatch v1. It is adapted through
`graphs_from_batch`, preserving graph/node/physical-edge identity, parallel
members, curve metadata and availability masks. Mutable preadapted dictionaries,
GraphBatch v2, GINE profiles and legacy five-column features are not accepted by
this helper. Their existing separate APIs are unchanged.

The only supported input profile is raw ordered x, y, z, dimension, with exact
explicit geometry units and coordinate frame. There is no automatic unit
conversion, coordinate-frame conversion, learned scaler, feature reordering or
target inverse transformation. Declaring an equivalent physical geometry in a
different unit still rejects. Coordinates are not automatically invariant to
translation, rotation or scaling.

GCN/GAT/SAGE consume these raw node columns and topology, not the retained edge,
material or boundary metadata as learned physical inputs. No fitted input
preprocessing is applied by the helper; this does not remove the models' own
GCN adjacency normalization, LayerNorm or SAGE embedding normalization.

Internal output must be a finite CPU float32 tensor of shape `(n_graphs, 1)`.
The returned `predictions` is a flat JSON-ready list of scalar values, aligned
with `graph_ids` and `source_groups` in input order. Values are detached from
model tensor storage. `input_contract` contains an independent copy of only the
five checked compatibility fields.

The result explicitly reports:

- `validation='declared_input_contract_only'`;
- `prediction_space='model_output'`;
- `target_name_source='caller_declared'` and the exact supplied target_name;
- null target_unit/target_transform, each with a separate status of `unknown`.

Null target_transform does not mean identity. A caller's target name does not
establish a physical unit or prove that the values have been inverse-transformed.
There are no R²/RMSE scores or automatic final-test claims.

## Limits and execution state

Existing training records do not bind their metadata to actual training tensor
contents or record external fitted preprocessing. If someone normalized,
reordered or otherwise modified training tensors while keeping old tags, that
history cannot be reconstructed or certified here. Do not infer a historical
model's raw-input compatibility from absent normalization fields. Pair the model
with an appropriate declaration; this helper checks declarations, not their truth.

Prediction runs under eval/no_grad with CPU autocast disabled, then restores
every module's original training flag, including mixed per-module modes, on
success or failure. Ambient autocast and gradient-enabled contexts are restored.
No optimizer, backward, fitting or solver work occurs. Arbitrary caller hooks
may have their own side effects and are not made pure by this helper. The
existing adapter's other ambient default dtype/device limitations remain;
no global default dtype/device is changed.

Tests use nontrivial synthetic networks with cycles, branching, nonconsecutive
IDs, reversed parallel members, a curve, a missing radius and an isolated node.
Fixed untrained GCN/GAT/SAGE forward comparisons establish interface behavior,
not model quality, mechanics, electrical accuracy, historical state migration or
compatibility outside the tested CPU runtime.
