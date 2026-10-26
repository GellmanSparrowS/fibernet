# Legacy ML feature boundary

`FeatureExtractor` and `GraphFeatureExtractor` are different APIs. The former
accepts `FiberNetwork`; it is **not an alias for the 94-feature extractor**.
Changing extractors changes the feature contract and requires explicit model
validation, not padding or renaming columns.

## Supported: inventory counts

```python
from fibernet import FiberNetwork
from fibernet.ml import FeatureExtractor

extractor = FeatureExtractor(
    include_morphology=False,
    include_topology=False,
    include_spectral=False,
    include_pore=False,
    include_anisotropy=False,
)
values = extractor.extract(FiberNetwork())
# {'num_fibers': 0, 'num_crosslinks': 0, 'dimension': 3}
names = extractor.get_feature_names()
vector = extractor.extract_to_array(FiberNetwork())
```

The dict and array use that three-field order. Names are static: obtaining them
does not generate or analyze a network. Counts describe stored inventory, not
physical validity or connectivity.

All five flags still default to `True`. Any enabled group raises
`NotImplementedError` with its missing-contract reason and counts-only guidance.
Default `FeatureExtractor().extract(...)`, default `get_feature_names()`, and
`fibernet.ml.features.extract_features(...)` therefore do **not** return a full
legacy feature vector. Failed or undefined quantities are not replaced by zeros.
There is no guaranteed complete historical 38-value schema.

| Legacy group | Unresolved contract |
|---|---|
| Morphology | Historical reports do not define the requested `std_radius`. |
| Topology | Report layout and approximate legacy graph projection need explicit migration. |
| Spectral | The historical method call and missing-eigenmode layout need migration. |
| Pore | The sampled-centerline-vertex approximation needs an explicit definition. |
| Anisotropy | The 3D tensor and empty-input conventions need an explicit definition. |

## Supported: bounded legacy GNN projection

Import this class from its defining module:

```python
from fibernet.ml.features import GNNFeatureExtractor
```

`extract_graph(network)` accepts `FiberNetwork` and returns `node_features`,
`edge_index`, `edge_features`, `num_nodes`, `num_edges`, and the NetworkX `graph`.
Fibers are nodes indexed by their position in the fiber list; contacts are
undirected simple edges, listed once. This is **not** a geometric-junction graph
or lossless physical serialization. Distinct co-located fibers remain distinct
nodes. Repeated unordered contacts, self-contacts, invalid indices and malformed
geometry are rejected instead of silently merged.

Default node columns are endpoint midpoint x/y/z, degree, arc length and radius
(`float32`, shape `(N, 6)`). Default edge columns are midpoint distance and the
historical regularized chord-angle feature (`float32`, shape `(E, 2)`). Edge
indices are `int64`, shape `(2, E)`. These widths also apply to empty inputs.
Feature lists retain membership-based fixed column order; reordering a list does
not reorder columns, and empty lists retain the historical defaults.

Internal centerline points contribute to arc length but are not otherwise
encoded; using midpoint/chord geometry with internal points emits a warning.
Angle normalization retains `norm + 1e-10`, so it is not an exact geometric angle
at small scales. Selecting angle rejects chord norms at or below `1e-12`, even
for isolated fibers. Optional `weight` remains the constant `1.0` and warns that
it is not contact stiffness or strength. Material laws and full crosslink data
are not encoded. Optional centrality is computed once; its errors propagate.

Nonfinite derived geometry and float32 overflow are rejected. Feature values must
be signed zero or lie within the float32 normal range: every nonzero magnitude
below `2**-126` is rejected before casting, even if its subnormal bits could be
stored on a particular platform. Cast output is also checked for nonfinite,
subnormal or nonzero-to-zero results. The normal-or-zero range provides consistent input bounds across floating-point
runtimes; `1e-40` is not an accepted feature value. No process floating-point mode is changed.
These checks do not claim that all preceding geometry arithmetic is immune to
platform-dependent subnormal handling. Coordinates retain the `Fiber` constructor's float64 storage contract.
No automatic unit conversion, clipping or rescaling is performed. The caller
must choose any rescaling and maintain its units and model-input contract.

No implicit `StructureGraph` conversion occurs. `api.analyze` and the public
94-feature implementation are unchanged by this repair. Archived predictors,
dataset builders and PyG/DGL workflows are not declared restored by this boundary.
