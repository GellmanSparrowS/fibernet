# Optional scalar edge-aware GINE adapter

`fibernet.ml.edge_gine` is an explicit optional module. It provides a two-layer GINE architecture with a closed input profile. Ordinary FiberNet imports and the public `FiberGNN` GCN/GAT/SAGE implementation are unchanged. `train_gnn` does not support `GINERegressor`; no training, checkpoint migration or optimizer-resume API is added here.

The module requires compatible optional PyTorch and PyTorch Geometric when preparing tensors or constructing a model. Missing dependencies produce an actionable `ImportError`; there is no topology-only or fake-GINE fallback. Configuration and pure input-contract tests do not prove runtime operator compatibility. The reference optional versions are Torch 2.7.1+cpu and PyG 2.7.0; runtime compatibility requires separate qualification.

## Existing scalar profile

`make_config(features=..., length_unit=..., coordinate_frame=..., node_offset=..., node_scale=..., pooling=...)` retains `fibernet.edge_gine.v1`. `prepare_gine(batch, config=..., source_groups=..., external=...)` accepts canonical GraphBatch v1 and exact external graph-ID/physical-edge-ID keyed scalar records. Existing scalar-even feature descriptors have exactly `name`, `source`, `unit`, `offset`, `scale`, `required`, `parity`. A source is `{'snapshot': 'radius'}`, `chord_length`, `polyline_length`, or a named `{'external': ...}` scalar. Normalization is `(value-offset)/scale`; scales are explicit and never fitted. Exact external units and row/field coverage are required. Optional missing values are `None` with availability false; true zero remains available.

## Material-view profile

Explicitly pass `material_profile=...` to `make_config` and the matching `material_view=...` to `prepare_gine`. The configuration version becomes `fibernet.edge_gine.material_view.v1`. There is no automatic promotion of v1 input or implicit field selection.

First construct the authoritative `StructureGraph`, then its existing snapshot and numeric view:

```python
from fibernet.ml.graph_data import graph_batch, material_features_from_batch
from fibernet.ml.edge_gine import make_config, prepare_gine, GINERegressor

batch = graph_batch(
    graphs, graph_ids=graph_ids, length_unit='m', coordinate_frame='specimen',
    edge_features=('radius', 'chord_length', 'polyline_length'),
    include_provenance=True,
)
fields = ('youngs_modulus', 'shear_modulus', 'density', 'poissons_ratio')
units = dict(zip(fields, ('Pa', 'Pa', 'kg/m^3', '1')))
# These are example reference scales, not fitted statistics or material estimates.
scales = dict(zip(fields, (1e9, 1e9, 1000., 1.)))
view = material_features_from_batch(
    batch, fields=fields, units=units, scales=scales,
    material_assumption='isotropic_scalar',
    origin_policy='retain_stored_origin_unknown',
)
profile = dict(
    view_schema='fibernet.material_feature_view.v1',
    batch_schema='fibernet.graph_batch.v2',
    provenance_schema='fibernet.graph_batch.provenance.v1',
    fields=fields, units=units, scales=scales,
    material_assumption='isotropic_scalar',
    origin_policy='retain_stored_origin_unknown',
)
features = [dict(name=name, source={'snapshot': name}, unit='m',
                 offset=0., scale=1., required=True, parity='even')
            for name in ('radius', 'chord_length', 'polyline_length')]
features += [dict(name=name, source={'material_view': name}, unit=units[name],
                  offset=0., scale=scales[name], required=False, parity='even')
             for name in fields]
config = make_config(
    features=features, length_unit='m', coordinate_frame='specimen',
    node_offset=[0., 0., 0., 0.], node_scale=[1., 1., 1., 1.],
    pooling='mean', material_profile=profile,
)
# source_groups explicitly identifies independent sources, not merely unique IDs.
prepared = prepare_gine(batch, config=config, source_groups=source_groups,
                        material_view=view)
model = GINERegressor(config)
# model(prepared) is a CPU float32 graph-scalar prediction from current weights.
# Constructor-initialized weights have no accuracy or physical interpretation.
```

The example assumes the caller has already supplied geometry in metres and Material properties in the declared SI units. Declaring `'m'` or `'Pa'` is not a unit conversion. Unknown fixture units, mixed declarations and external scalar records reject in this first material profile. Required missing selected features reject; optional missing fields retain masks. Zero-node graphs and self loops are unsupported; nonempty edge-free graphs retain the GINE root contribution.

### One numeric transformation, explicit consistency checks

A MaterialFeatureView already contains `stored_property/declared_scale`. Its descriptor keeps the original source unit and scale as provenance; the new profile marks that scaling as already applied. The bridge validates the stored record/scale reference by binary64 bits, then casts the existing view value to float32 without dividing it again. This consistency check is a recomputation from the raw record, not a second feature normalization. All selected view fields must appear exactly once and in their declared relative order, with matching units/scales and zero material offsets.

Geometry columns and XYZ/dimension use their own `(value-offset)/scale` exactly once. A nonzero raw value equal to its offset may legitimately center to zero. Unequal stored operands losing their difference to zero reject; subsequent division and float32 conversion reject newly lost finite range, nonzero or sign. Genuine signed zeros are retained where the specified arithmetic preserves them. No FP-mode reset, clipping, imputation, G derivation or E/G/nu reconciliation occurs. The current host arithmetic may cause a precise fail-closed refusal; the bridge cannot recover losses that occurred before snapshotting or certify general FTZ/DAZ accuracy.

### What the model consumes

Nodes contain XYZ and dimension. All selected physical-edge scalar columns come first, followed by availability-mask columns in the same feature order; the whole table is expanded through `batch.message_parent_rows`. Both directed orientations of an even scalar carry identical values/masks. Parallel physical members and repeated local IDs in different graphs remain distinct. There is no edge coalescing, inferred topology or automatic PyG batch relabelling.

Both numerical GINE layers receive the complete values-plus-masks tensor. Architecture is fixed: Linear(4,16), two upstream GINEConv layers with two-linear-layer ReLU MLPs, additive aggregation, eps=0, explicit sum/mean pooling, Linear(16,1). No dropout or BatchNorm. Forward checks the closed configuration, dimensions, dtype/device, finite tensors/parameters, identity/message parentage, even reversal and mask consistency. Same-width reordered fields, changed units/scales or another profile do not become compatible merely because tensor shapes agree.

`PreparedGINE` owns mutable CPU float32 tensors and immutable serialized config/provenance. It is not another graph owner. Tensor mutation and gradients are possible; source records are not aliased. Forward checks consistency, not cryptographic provenance or physical validity. Arbitrary available tensor-value changes are not authenticated back to source records; tensor availability and topology remain bound to the sidecar. The input boundary assumes factory-produced or equivalently trusted canonical snapshots/views; read-only flags are ordinary accidental-write protection, not hostile-object isolation.

### What remains provenance only

Material names, stable IDs and source-group strings are never numeric features. Missing material, missing property, null property and stored-origin-unknown remain distinct in provenance. The first model profile numerically consumes only availability, so missing and null share an unavailable placeholder while remaining auditable. Supplied/defaulted/derived origins cannot be reconstructed.

Cell-face flags describe geometric boundary faces, not mechanical constraints, loads or electrodes. Segments are meshing provenance. Their records are retained but are not model inputs. No mechanical-condition encoding is supplied.

Raw internal points are retained as provenance. Only selected polyline length conveys curve-derived numeric information. If no internal geometry was supplied, canonical polyline length is the endpoint path, with `internal_available=False` still retained. Equal scalar lengths do not encode equal curve shape; no tangent, curvature or curved-beam response is claimed. Raw XYZ is frame-dependent.

Non-null constitutive tensors, custom material records and unsupported fields reject through the numeric material profile. `isotropic_scalar` is a caller declaration, not physical isotropy/stability certification. No material class or property is inferred from a name.

## Scope, artifacts and qualification

Checkpoints from another feature, target, source or runtime contract are incompatible even when tensor widths agree. Do not weaken source checks, relabel features or treat weights learned for an electrical target as a mechanically qualified model. Previously observed benchmark sources cannot be counted as fresh holdout evidence.

This adapter does not provide target generation, source-independent data splitting, trainer integration or a new checkpoint format. Those require separate explicit contracts. A successful feature-consumption check would establish a software path, not accuracy, conservation, global equilibrium, objectivity, or generalization. Two local layers and graph pooling may miss global bottlenecks.

The reused FiberNet source is covered by the repository MIT license, copyright 2026 FiberNet Contributors. The operator is imported from upstream PyTorch Geometric, not vendored here. Its retained upstream source/license references are [GINEConv 2.7.0](https://pytorch-geometric.readthedocs.io/en/2.7.0/_modules/torch_geometric/nn/conv/gin_conv.html) and [PyG MIT license](https://raw.githubusercontent.com/pyg-team/pytorch_geometric/2.7.0/LICENSE). Preserve applicable notices when redistributing dependencies.
