# FiberNet 4.4.3

This patch covers the changes since the published 4.4.2 core package through
the main-branch merge of the 3MF batching work ([#234](https://github.com/GellmanSparrowS/fibernet/pull/234)).
It runs on Python 3.9 and later with NumPy, SciPy, NetworkX and Matplotlib.

## Start with the installed package

The wheel bundles a small graph, FEM and result-round-trip workflow.

```bash
python -m fibernet.examples.basic_usage --output-dir fibernet_demo
```

The [native Dataset guide](architecture/native-dataset-training-usage.md)
shows training on disjoint graph origins and fitting feature scaling only on
training rows. The electrical edge-deletion and notched-fracture workflows
run as examples from a source checkout.

## Changes since 4.4.2

- Correct the sign and length scaling of recovered 2D sparse-beam endpoint
  moments. Linear beam FEM also records full strain energy separately from the
  historical axial-stress proxy. Recheck moment-based stresses or comparisons
  calculated with the prior sparse solver.
- Sum each fiber's nominal area in 2D density calculations, preserve planar
  concave OBJ boundaries, and correct planar pore and 3D graph measures.
  Mixed-radius density, pore and betweenness values can change.
- Scale explicit rectangular, elliptical and hollow sections along with the
  fiber geometry. A shared fiber or crosslink object transforms once, while
  distinct objects retain independent transformed coordinates.
- Center electrical solves per conducting component to resolve small voltage
  differences on a nonzero common-mode voltage.
- Admit explicitly sampled welded fiber paths through a dedicated
  [importer](SAMPLED_WELDED_FIBER_IMPORT.md). Preserve stored curves, indexed
  component attachments, stable fiber identities and simulation fields in
  network analysis and interchange.
- Offer sparse fiber adjacency via
  `FiberNetwork.connectivity_matrix(sparse=True)`. Opt into HDF5 format 0.2.0
  to preserve cross-section parameters and crosslink strength and stiffness.
  Existing callers keep dense adjacency and the HDF5 0.1.0 layout.
- Correct interchange for PDB atom and residue indexing, Gmsh physical-line
  groups, VTP simulation fields and small-coordinate OBJ meshes. PDB bond
  records follow the atom records, and files that exceed fixed record widths
  are rejected before export.
- Preserve observed DataFrame point rows and reference IDs when grouping
  fiber samples. Report missing group or point identifiers on affected
  paths. Reject ambiguous graph identities and unobserved history inputs
  before numerical work.
- Reduce the cost of planar random intersections and 3D proximity candidate
  handling. Batch 3MF XML vertex and face writes while retaining the same
  model records.

## Migration

- Recompute stored scientific outputs if they depend on recovered sparse 2D
  beam moments, planar density, pore boundaries, 3D betweenness or sampled
  connectivity. The [beam contract](METHODS_BEAM_FEM_CONTRACT.md) explains the
  separate force, moment and energy fields.
- Treat the component-connection return value as a successful join count.
  Coincident endpoints receive a direct weld without inserting a zero-length
  fiber. Use the change in fiber count when counting newly inserted fibers.
- Preserve the `fibernet_fiber_index` column when moving exported noncanonical
  fiber labels through DataFrame tools. Use the explicit stable ID map
  alongside exported graph or simulation fields.
- The exact-source MCP manifest accompanies the core revision. The adapter
  and FiberScope remain separate distributions. Existing source-bound
  records may require revalidation with this revision.
