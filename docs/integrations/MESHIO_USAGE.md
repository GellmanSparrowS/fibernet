# Optional VTU line export

The first bounded adapter is implemented in `fibernet/io/meshio_adapter.py`.
It exports straight line cells, not solid meshes or complete beam models.

Install meshio separately in your chosen environment:

```sh
python -m pip install meshio==5.3.5
```

No change to FiberNet's base dependency set or automatic installation is made.
Importing the adapter does not import meshio until export is requested.

```python
from fibernet import StructureGraph, Material
from fibernet.io.meshio_adapter import write_vtu_bundle

graph = StructureGraph(dimension=3)
a = graph.add_node([0, 0, 0])
b = graph.add_node([10, 0, 0])
graph.add_edge(a, b, radius=0.1,
               material=Material(youngs_modulus=2e9))
write_vtu_bundle('line_export', graph, length_unit='mm',
                 provenance={'source': 'straight-rod example'})
```

This creates a **new directory** with `network.vtu` and `manifest.json`.
An existing destination is rejected, including an empty directory. Stage data
in a new sibling directory and rename only after both files have been written.
Concurrent writers to the same destination are unsupported; this is not a
transactional database or a crash-durability guarantee.

## Ordering and fields

- Point rows follow sorted stable node IDs
- Line cells follow the insertion order of `graph.edges`
- Sparse IDs are exported as integer point/cell data, separately from dense connectivity
- Parallel fibers retain separate cells, IDs, radii and material indices
- Isolated nodes remain present
- Numeric extra fields must be `(N,)` or `(N,3)` for points and `(E,)` or `(E,3)` for cells, already in the documented order
- Every extra field requires `units` and `meaning` strings under `field_metadata['point'][name]` or `field_metadata['cell'][name]`
- Names `node_id`, `edge_id`, `radius`, `material_index` are reserved

For example, pass `cell_data={'force': forces}` and
`field_metadata={'cell': {'force': {'units': 'N', 'meaning': 'axial force'}}}`.
The adapter does not calculate forces or validate the physical interpretation
of caller-provided arrays.

## Units, material and provenance

`length_unit` is mandatory: m, cm, mm, um or nm. Coordinates and radii are
copied without scaling. Material parameters use the SI conventions documented
by `Material`, independent of the geometry unit. User-defined material `extra`
fields retain their data but need their own documented units.

The manifest carries material tables, source dimension, ID ordering, declared
units, boundary flags, metadata, box size, segment counts, optional provenance,
meshio version and the SHA-256 hash of `network.vtu`. This is an interchange
bundle, not a replacement for native project persistence. Box size alone does
not establish periodic element/image connectivity. Stiffness/strength values
are retained as material information, not translated into a solver deck.

Nonfinite coordinates/radii/fields, nonexistent endpoints, zero-length edges,
nonpositive radii, unsupported field shapes and non-JSON metadata are rejected.
Edges with internal points are rejected even if those points are collinear;
explicit subdivision and interpolation are a later feature. Empty graphs or
graphs without edges are rejected. Arbitrary metadata keys must be strings.

## Verified scope

Tested with meshio 5.3.5 using actual write/read of VTU line cells: sparse IDs,
parallel fibers, nontrivial edge ordering, material tables, 3D coordinates,
scalar/vector arrays, no implicit unit conversion, detached snapshots and
failure cleanup. Also tested planar 2D storage and missing dependency errors.
No independent VTK/ParaView reader or large-network performance check was run.

## Complete fields keyed by stable ID

`to_meshio` and `write_vtu_bundle` also accept `point_data_by_id` and
`cell_data_by_id`. Each maps a field name to a mapping from stable entity ID to
a numeric scalar or three-vector. The adapter aligns values to its export rows;
field-map insertion order has no meaning. For example, with a
`solve_linear_transport` result for the same graph:

```python
write_vtu_bundle(
    'electrical_export', graph, length_unit='mm',
    point_data_by_id={'voltage_V': dict(zip(result.node_ids, result.voltage_V))},
    cell_data_by_id={'current_A': dict(zip(result.edge_ids, result.current_A))},
    field_metadata={
        'point': {'voltage_V': {'units': 'V', 'meaning': 'electric potential'}},
        'cell': {'current_A': {'units': 'A', 'meaning': 'signed current node_i to node_j'}},
    },
    provenance={'electrical_result': result.to_dict()},
)
```

Every field must cover **exactly all** graph nodes or edges at its location,
including isolated nodes and separate parallel edges. Partial solver results,
missing/foreign IDs, null/unavailable values and nonfinite values are rejected;
there is no zero filling. ID keys must be nonnegative integers within int64 range
(Python and NumPy integers are accepted). Booleans, strings and floating-point
keys, including integral-valued floats, are rejected rather than converted.
Scalar/vector shape, reserved-name and units/meaning checks are the same as for
row arrays. Input graphs, fields and metadata are not modified.

At each location, row arrays and ID maps are mutually exclusive, even when an
explicitly supplied container is empty or the field names differ. For example,
`point_data={}` with `point_data_by_id={}` is an error. Point and cell locations
may independently choose either form; omitted arguments and `None` mean absent.
The original row-array API and its serialized bundle output are unchanged.

Mappings are required, not sequences of pairs. Observable invalid or repeated
ID keys from a custom mapping are rejected. Python dictionaries may already
have collapsed equal keys before the adapter sees them (for example `7`, `7.0`
and `True`/`1`); overwritten entries cannot be recovered or diagnosed. Validate
source pairs before constructing a dictionary when that distinction matters.

The dedicated ID-field tests actually reload VTU for sparse/reordered IDs,
isolated nodes, parallel edges and scalar/vector fields. They also exercise a
generated square through native JSON reload, explicit-conductance transport and
ID-keyed VTU export. All voltages in that journey are prescribed: it validates
interchange, not a new free-node solver case. Equivalent array and ID-map
exports are checked byte-for-byte for both VTU and manifest.
