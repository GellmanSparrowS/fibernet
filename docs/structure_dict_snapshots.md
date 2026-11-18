# Editing a serialized design without changing the original

`StructureGraph.to_dict()` now deep-copies graph, node and edge metadata. Editing
those standard mutable containers in the exported record no longer changes the
source graph, and later source edits do not rewrite an earlier exported record.
This fixes a real persistence failure: editing a draft record previously changed
the original graph too, so saving the supposedly unchanged original saved the
draft's metadata.

```python
from fibernet import StructureGraph

original = StructureGraph(dimension=2)
a = original.add_node([0, 0])
b = original.add_node([1, 0])
original.add_edge(a, b, radius=0.1)
original.metadata['design'] = {'tags': ['original']}

draft_record = original.to_dict()
draft_record['metadata']['design']['tags'].append('option-B')
variant = StructureGraph.from_dict(draft_record)

assert original.metadata['design']['tags'] == ['original']
assert variant.metadata['design']['tags'] == ['original', 'option-B']
original.save_json('original.json')
variant.save_json('variant.json')
```

## Compatibility

- Schema remains 3.1. No field names, stable IDs, counters, ordering, material
  serialization, coordinates, stored curves or default values change.
- `graph.metadata`, `graph.nodes[id].metadata` and `graph.edges[id].metadata` remain
  live mutable accessors. Use them when the intention is to edit the graph.
- Code relying on mutation of `to_dict()` output to edit the original must switch
  to those accessors or explicitly create/replace a graph with `from_dict()`.
- Metadata follows Python `copy.deepcopy`, as existing `copy()` and `from_dict()`
  already do. Ordinary nested containers are detached without a JSON round trip.
  NumPy arrays and tuples retain their types in the dictionary. This does not
  make non-JSON objects serializable by `save_json()`.
- Custom objects control their own deep-copy behavior. Non-copyable values can
  fail during export; arbitrary custom objects are not promised an independent
  immutable snapshot. Serialization still completes before a destination file
  is opened, preserving it when serialization fails. This is not a guarantee of
  atomic saving under arbitrary disk or operating-system failures.
- This is an independent data copy, not a concurrent-access lock, a universal
  metadata validator, a simulation cache key, or a unit conversion.

## Validation scope

`tests/test_structure_dict_snapshot.py` covers detached graph, node and edge
metadata, reverse mutation, independent exports and 2-D/3-D persistence. Related
public JSON I/O, preparation-only and design-composition checks also exercise
the snapshot contract. These data-contract checks do not establish physical
validity or replace the repository CI and release gates.
