# Import multiple exact welded junctions

`convert_exact_welded_crosslinks` is an opt-in topology/geometry conversion for
legacy `FiberNetwork` objects. It does not replace the existing approximate
`StructureGraph.from_fiber_network` endpoint projection or the separate
`convert_single_welded_crosslink` contract.

```python
import numpy as np
from fibernet.core.fiber import Fiber
from fibernet.core.network import Crosslink, FiberNetwork
from fibernet.core.single_junction_import import convert_exact_welded_crosslinks

net = FiberNetwork(dimension=2)
net.fibers = [Fiber(np.array([[-2., 0., 0.], [2., 0., 0.]])),
              Fiber(np.array([[-1., -1., 0.], [-1., 1., 0.]])),
              Fiber(np.array([[1., -1., 0.], [1., 1., 0.]]))]
net.crosslinks = [Crosslink(0, 1, .25, .5, np.array([-1., 0., 0.])),
                  Crosslink(0, 2, .75, .5, np.array([1., 0., 0.]))]
graph, report = convert_exact_welded_crosslinks(net)
assert len(report['fibers'][0]['edge_ids']) == 3
assert graph.num_edges == 7
```

## Accepted connections and geometry

Each crosslink references two distinct **fiber list indices** as Python `int`
values, not the stored `fiber_id`. Parameters are strictly inside `(0, 1)`
and interpolate between successive centerline samples by index. Both interpolated points and the
stored position must coincide exactly in float64. Coordinates and parameters
must be exactly representable; near points are rejected, never snapped. A
repeated `fiber_id` does not merge fibers because mapping uses list indices. A
fiber may have multiple welds at different parameters to different fibers.
Only one record per unordered pair is allowed: duplicate/reversed records and
multiple records for a pair are rejected. Equal coordinates on one fiber must
use the same parameter.

At the same exact coordinate, weld records linked through a common fiber form
one shared graph node. Disjoint groups at identical coordinates remain
separate nodes unless an explicit weld chain joins them. Unrecorded crossings
are not inferred. Coincident endpoints and any endpoint coinciding with a weld
are rejected, including when they belong to unconnected fibers. Endpoint welds
are unsupported.

Inputs are limited to 1,024 fibers, 4,096 links, 10,000 samples per fiber and
100,000 samples in total. `network.periodic` must be exactly `False`. They
must be circular fibers without section parameters, with positive finite radii and ordinary `Material`
instances. Centerlines must be index-uniform straight lines; 2D z coordinates
must be zero. Only welded crosslinks with infinite strength and stiffness are
accepted. Masks, complex or inexact coordinates, invalid references, other
link types, finite weld properties, and empty link lists are rejected. A box,
if present, must have two or three finite exactly representable coordinates;
its physical validity is not certified.
These limits bound geometry counts, not the size of copied metadata or
Material contents; they are not a total-memory guarantee.

## Output and scope

For each source fiber list index, `report['fibers'][index]` records its source
`fiber_id`, ascending split parameters, matching junction node IDs, and edge
IDs in source start-to-end order. `report['crosslink_junction_node_ids']` maps
each original crosslink list index to the corresponding node. The same node
may appear for multiple explicitly connected records. New node and edge IDs
are allocated deterministically for the same input order; they are not source
IDs. Radius, detached material copies, network metadata and box are copied.
As in `StructureGraph`, a two-coordinate box is stored as `[x, y, 0]`.
Each output edge has four segments. Source discretization and weld strength or
stiffness are not represented.

This conversion does not infer units, boundary conditions, material behavior,
physical contact, or mechanical equivalence. Nor does it guarantee that
arbitrary copied metadata or material fields satisfy JSON or native-v1
serialization requirements. Source information lost before this call cannot
be recovered.

## Serialization boundary

The report describes the objects passed to this function. Legacy
`FiberNetwork.load_json` already renumbers fiber IDs to their list indices;
this converter cannot restore labels lost during that earlier load. For direct
in-memory conversion, repeated or sparse source IDs remain report labels, not
mapping keys. Default legacy weld properties serialize as Python JSON's
`Infinity`, whereas native JSON compatibility depends on the copied fields.
Use the native graph for editing after an admitted conversion; do not claim
arbitrary legacy or metadata serialization is lossless.

## Regression checks

`tests/test_multi_junction_import.py` covers the explicit-weld admission and
mapping contract. `tests/test_multi_junction_saved_workflow.py` adds a real
legacy JSON save/load, conversion, and native JSON save/load workflow. Its
assertions bind each source list index and split parameter to the corresponding
junction, oriented edge, radius and material, including an unwelded fiber.
It also checks nested material, network metadata, coordinate and box ownership
in both directions. These are geometry/data regressions, not mechanical tests.

Both modules use `unittest` and are discoverable by the repository's test suite.
With the repository dependencies available, their focused commands are:

```bash
python -m unittest discover -s tests -p 'test_multi_junction_import.py' -v
python -m unittest discover -s tests -p 'test_multi_junction_saved_workflow.py' -v
```

The saved workflow explicitly expects legacy loading to renumber fiber labels.
It does not claim arbitrary legacy JSON or material metadata is lossless.
