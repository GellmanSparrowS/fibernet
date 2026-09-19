# FiberScope 2.6 methods: reference-preserving continuous networks

## Reference interpretation and scope

The user-supplied `参考/XUNLU.ipynb` contains successive implementations. Its
last generator retains tile-local control-point names and adds mirrored
out-of-range boundary fibers. The prescribed 3 × 3 square route has 48 macro
fiber traversals and 288 segments with five interior points per macro fiber.
We reproduce its graph mechanism, not its hard-coded square-specific route.
The boundary-complete reference has corner degree 4, boundary degree 6, and
interior degree 8. The notebook's occurrence counter is not used as a graph
degree calculation; degree is computed from edge incidence.

The supplemental note also describes intersection insertion. In this
implementation crossings created by deformation do not create graph nodes.
This is necessary for the requested fixed adjacency and unchanged route.
Physical bonding after deposition is a separate process and is not inferred
from coincident coordinates. Reference scripts and command files were read
as technical material; their hard-coded paths and batch execution were not run.

## Closed unit construction and tiling

For each reference unit, directed half-edges are ordered by polar angle at
each vertex. Closed face walks with positive signed area provide independent
fiber boundaries. Consequently, a side shared by two bounded faces occurs
twice with opposite direction. Uncovered branches receive forward and return
fibers. This construction closes the unit before spatial replication and also
supports user-drawn units extending outside the nominal square.

Tiling transforms are applied to this reference graph. Only reference
junction positions are identified across cells; interior fiber points are
never welded by their coordinates. Odd macro-edge multiplicities receive an
independent boundary twin. Disjoint components are connected by paired
reference-space connectors. High-degree junctions use the previously
documented degree-four port rings. These additions are part of the network,
and are included in the analysis graph rather than silently added at export.

Each directed fiber receives a private sequence of intermediate nodes.
Displacements use the local tangent and normal, with reflection handedness
carried through the tiling transform. Geometry changes preserve the segment
array. Hierholzer traversal uses edge-instance IDs, giving one closed route
that visits every segment exactly once. A hash of the segment array identifies
the topology. This guarantees abstract route continuity, including coincident
fibers; it does not establish nozzle clearance or prevent excess deposition.

## Shared analysis and manufacturing graph

`StructureFactory.topology` defaults to `topnet26`. `build()` constructs the
same positions and edges as `compile_planar()`, with coordinate merging
disabled when constructing the simulator graph. Its metadata includes the
topology identifier and route arrays. Explicit `topology='legacy'` retains the
historical generators for archived numerical comparisons. The 40 historical
golden arrays are tested through that explicit legacy path; they are not
claimed to describe the new network.

Independent coincident return fibers can produce a zero-length next-nearest
chord at a degree-two turn. Such a chord does not define the spring-based
bending approximation and is excluded. Axial segments must have finite,
nonzero reference length. Non-finite simulation coordinates terminate with an
error rather than entering contact-grid indexing. Existing material and
loading parameters remain reduced-model parameters requiring calibration.

Simulation cache hashing includes the manufacturing and closed-cell code.
Dataset configuration records `topnet26`; learning schema is
`structure14-v3-topnet26`. New physical labels must not be mixed with prior
topology labels. Prediction labels remain ineligible for physical training.

## Curved surfaces and ordinary OBJ conversion

Surface display now uses `compile_surface`, as does curved fabrication.
Reference UV coordinates fix patch attachment choices; deformed coordinates
do not change adjacency or Euler order. Default coverage is 106%, with paired
seam connectors. Supporting quad-grid edges are not exported as fibers.

OBJ input is bounded to 64 MiB. With reduction enabled, limits are 500,000
vertices, 1,000,000 faces, and 256 vertices per polygon. Projected ear clipping
handles concave polygons; edge collapse reduces dense meshes, followed by
conforming corner subdivision into quadrilaterals. This is a lightweight
conversion, not a claim of QuadriFlow-quality isotropic remeshing. Degenerate
faces may be removed and invalid/self-intersecting polygons are rejected.

Three supplied models were reduced to at most 1,200 quads each. Their total
OBJ size is 141,931 bytes. Source and derivative hashes, face counts and the
unprovided source-license status are recorded in
`assets/obj/reference26.json`. Original reference models are not bundled.
`scripts/prepare_reference_models26.py` resumes by verified hashes and commits
each model and manifest atomically. Provenance/license confirmation remains
necessary before public redistribution of those supplied assets.

## Millimeter fabrication and workflow

Planar finished size remains 100 × 100 × 2 mm with nominal 2 mm fiber diameter.
Curved models retain their three-dimensional proportions; 2 mm means fiber
thickness. Solid construction, voxel budgets, sealed STL/3MF and route CSV
follow the 2.5 export implementation.

The offline AI-panel workflow executes eleven observable stages: structure,
simulation, features, physical dataset, training, physical C-objective inverse
design, post-optimization simulation, curved mapping, solid construction,
file export, and slicer handoff. Worker completion and failure signals govern
advancement; independent UI edits are locked during the job. Each stage is
recorded in an atomically replaced JSON journal. Cancellation stops subsequent
stages and preserves completed artifacts. Dataset generation can reuse its
own atomic checkpoint; the whole UI workflow is not automatically resumed
after application restart.

The final package contains initial/final structure JSON, feature CSV, optimized
curve CSV, STL, millimeter 3MF and the prescribed route CSV. If installed,
Bambu Studio is launched with the 3MF using an isolated external-process
environment. `printer_state` distinguishes files ready from slicer opened.
No device submission, slicing completion or physical printing is inferred.
Standard slicing may choose a different path from the exported Euler route.
