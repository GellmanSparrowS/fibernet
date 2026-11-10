3D edge-betweenness concentration
=================================

``GraphFeatureExtractor3D`` reports ``edge_betweenness_max`` and
``edge_betweenness_gini`` at zero-based columns 20 and 21 of its unchanged
60-feature vector. The Gini coefficient describes how unevenly unweighted
shortest paths pass through the represented edges. It is dimensionless;
larger values mean greater concentration, not greater mechanical strength.

Definition and example
----------------------

For edge centralities ``b`` and ``m`` edges, the population definition is
``sum(abs(b_i - b_j)) / (2 * m * sum(b))``, summed over all ordered pairs.
The implementation uses the equivalent ascending-rank formula, without
allocating an edge-by-edge matrix. It does not apply a finite-sample correction.
For nonnegative centralities its mathematical range is zero to ``(m - 1) / m``.
Equal centralities, a single edge and an edgeless graph have value zero.

This runnable example uses the public API and independently evaluates the
pairwise definition::

   import math
   import networkx as nx
   import fibernet as fn

   graph = nx.path_graph(4)
   for node in graph:
       graph.nodes[node]["pos"] = (float(node), 0.0, 0.0)
   extractor = fn.GraphFeatureExtractor3D()
   features = extractor.extract(graph)
   b = [0.5, 2.0 / 3.0, 0.5]
   expected = sum(abs(x - y) for x in b for y in b) / (2 * len(b) * sum(b))
   assert math.isclose(expected, 1.0 / 15.0)
   assert math.isclose(features["edge_betweenness_gini"], expected)
   assert math.isclose(extractor.extract_vector(graph)[21], expected)
   assert extractor.batch_extract_matrix([graph]).shape == (1, 60)

Centralities retain NetworkX's normalized, unweighted default. Edge weights,
lengths, radii and material properties do not enter this statistic. Disconnected
components and isolates remain in the input; a common normalization factor
cancels in the Gini ratio. A disconnected four-node path plus a separate edge
has Gini ``9 / 44``, rather than the connected path's ``1 / 15``.

Supported graph inputs and conversions are unchanged: simple undirected
NetworkX graphs, StructureGraph without parallel physical edges, and the
existing FiberNetwork centerline conversion. This correction does not add
self-loop support or change other geometric or spectral features.

Availability and existing artifacts
-----------------------------------

The existing implementation only computes edge betweenness below 500 nodes.
At 500 or more nodes, or after a caught centrality failure, both betweenness
features retain their zero fallback. A zero therefore does not by itself
establish equal path participation. This correction introduces no availability
flag, computation-budget change or new error behavior.

Earlier 3D implementations sorted centralities in descending order while using
the ascending-rank formula, producing negative concentration for unequal values.
The correction intentionally changes numerical values while preserving names,
order and call signatures. The 2D extractor already used ascending ordering
and is unchanged.

Keep historical features and their producing FiberNet revision. Regenerate
affected training features, retrain and validate before feeding corrected
features into an existing model. Keep resumed validation checkpoints from the
old implementation separate from new results. Record the producing revision
or an application-level feature-schema version; the extractor does not
automatically migrate saved features, models or checkpoints. Correcting this
statistic is not validation of the other 59 features or a physical load-path
model.
