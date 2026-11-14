# 3D graph k-core features

`GraphFeatureExtractor3D.extract` returns a dictionary; `extract_vector` returns
its 60 named features in `get_feature_names()` order. The k-core correction
keeps those names, their positions and the public call signatures unchanged.

```python
import networkx as nx
from fibernet.analysis.graph_features_3d import GraphFeatureExtractor3D

# This topology-only example uses the extractor's default zero positions.
extractor = GraphFeatureExtractor3D()
features = extractor.extract(nx.complete_graph(4))
assert features["max_k_core"] == 3.0
assert features["kcore_frac"] == 1.0
vector = extractor.extract_vector(nx.complete_graph(4))
assert vector[extractor.get_feature_names().index("max_k_core")] == 3.0
```

## Definition

For a simple undirected, loop-free graph, `max_k_core` is the largest NetworkX
unweighted core number. The existing `kcore_frac` formula divides that maximum
by the unweighted mean degree across **all** nodes, including isolates. Both
values are dimensionless. Edge weights, lengths and radii do not enter them.

Despite its name, `kcore_frac` is not the fraction of nodes in a core and is not
restricted to [0, 1]. K4 gives (3, 1); a triangle gives (2, 1); a four-node path
gives (1, 2/3). K4 plus one isolated node gives (3, 1.25), because its mean
degree is 12/5. Edgeless, singleton and empty graphs return (0, 0).

Supported inputs and their existing conversion rules are unchanged. NetworkX
multigraphs and directed graphs, and StructureGraph inputs with parallel
physical edges, remain rejected. FiberNetwork uses its existing centerline
conversion. This correction does not extend support to self-loops or repair
other geometric, spectral or topological feature definitions.

## Existing features and models

Earlier implementations caught a missing local `mean_degree` key and silently
replaced both k-core values with zero, even for K4 and a triangle. The corrected
values intentionally differ. The 60-column layout is unchanged, but this does
**not** guarantee numerical compatibility with existing datasets or models.

Keep historical features and their producing revision for reproducibility.
Before using corrected features with a model, regenerate the affected training
features, retrain and validate, and record the producing FiberNet revision or
an explicit application-level feature-schema version. Do not mix old zero-valued
features with corrected inference inputs. The extractor does not automatically
version, migrate or validate saved model inputs. These topology descriptors are
not a claim of mechanical strength or physical validation.
