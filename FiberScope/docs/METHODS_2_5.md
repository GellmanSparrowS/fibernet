# FiberScope 2.5 continuous fabrication methods

## Scope and separation from mechanical analysis

The manufacturing workspace compiles the current unit, periodic rule and deformation parameters into a connected Eulerian multigraph and a millimetre-scale solid. It is available for every built-in unit and saved custom cell, and for quad-mapped surfaces. The existing structural modelling, simulation, feature and learned-model paths are preserved. Added return fibers, component connectors and junction rings belong to the explicitly labelled fabrication model; existing mechanical predictions must not be presented as physical validation of those additions.

This separation prevents silently changing validated simulation data while making the fabrication conversion inspectable. The fabrication workspace reports added return/connector edges and the graph identity, and exposes distinct 2D and 3D views. A new mechanical analysis of the optimized fabrication graph is a separate future integration task, not implied by this release.

## Connection to TOPNet

The supplied article, *A manufacturability-informed topology framework for AI-guided design of fibrous network materials*, describes TOPNet on page 3 (DOI 10.1038/s41467-026-76045-x). Adjacent cells retain separate identities for overlapping boundary fibers. Deformation changes geometry without changing the abstract connectivity or the canonical traversal. The article also separates topological continuity from process constraints such as degree, angles and segment length.

The original square mechanism is documented in `fslab/structure.py` and implemented through the oriented closed polygon in `StructureFactory.build`, with `_add_edge_with_intermediates` in the bundled fibernet pattern generator. The new generalized compiler is `fslab/manufacturing.py`; it does not alter that external generator or rewrite the original assets.

## Canonical planar compiler

1. Construct the undeformed single-cell corner graph with zero interior control points. Seed, cell type and custom-cell geometry define this template. Displacement spectrum, jitter and manual offsets are excluded from template connectivity.
2. Replicate canonical corners with the selected periodic transforms. Junctions are identified only using reference coordinates rounded to eight decimal places. Edge instances remain distinct, including coincident shared boundaries. For two adjacent square cells, the result has eight edge instances and seven distinct geometric endpoint pairs.
3. Connect disconnected components by adding paired opposite-direction connectors. A sparse nearest-neighbor candidate graph supplies short connections; a deterministic representative chain connects remaining separated components. This is a bounded deterministic construction, not a minimum-length global optimizer.
4. Correct odd degrees using a spanning-tree T-join: visit the tree in reverse traversal order and duplicate the parent edge whenever the subtree has odd parity. Duplicated edges are oriented in the opposite sense. This guarantees all-even degree on the connected graph, but does not claim a minimum-weight Chinese-postman solution.
5. Replace even junctions of degree greater than eight with small reference-space rings. Each ring port receives two external edges and two ring edges, yielding degree four. Ring radius is 2.5% of the shortest incident reference edge, bounded below numerically. Ring features require subsequent process-resolution review; topology repair alone does not certify their printability at arbitrary scales.
6. Subdivide each directed macro edge into the chosen number of interior points. Every interior node retains an independent identity, even when it coincides geometrically with another fiber. Apply the common relative displacement spectrum in the edge tangent/normal frame. Apply seeded jitter and manual anchor offsets to coordinates only. Shared endpoint identities are retained.

The compiler verifies one connected component, zero odd-degree vertices and a closed traversal containing every edge ID exactly once. It uses Hierholzer traversal on the canonical adjacency representation and caches up to four edge sequences. Spectrum and overlap changes preserve the edge array and traversal. Changing cell geometry, grid, periodic rule, seed of stochastic units or control-point count creates a different template and is not an isomorphic deformation claim.

## Surface compilation

One compiled Eulerian cell is mapped into each quadrilateral using the existing bilinear coordinate map. Cell coordinates are normalized by the reference cell bounds, not the deformed bounds. The default coverage is 106%. Both reference and deformed coordinates use the same face ordering and independent fiber identities.

Shared patch edges obtain fixed midpoint anchors. The attachment vertex is chosen once from the reference cell for each of its four sides. Paired connectors attach each incident cell to that anchor. No deformed-position welding or edge deduplication occurs. Disconnected mesh components receive explicit paired connectors. High-degree junctions are resolved in reference space with corresponding coordinate updates. Thus arbitrary changes in geometric overlap do not change the abstract route. Connectors can span separated components and must be inspected in the fabrication preview; they are not claimed to lie on the original surface everywhere.

The compiler bounds output to 180,000 points and 240,000 edges. Complex source meshes use the existing bounded coarsening routine before compilation. Coarsening changes the template and is determined by the reference cell size, independently of the deformation spectrum.

## Physical dimensions and solid meshing

The planar default outer envelope is 100 × 100 × 2 mm, with nominal fiber diameter 2 mm. Width/depth scale the centerline into the requested envelope minus one diameter. The center plane is at half the requested height. For nondefault unequal diameter/height, the cross-section is elliptical. Curved models retain 3D proportions and fit inside the requested width/depth envelope; 2 mm denotes fiber diameter, not total object height. The solid is moved onto z=0 for slicing.

Each fiber segment is sampled with spacing at most 0.4 radius. The union of radius-distance neighborhoods of these samples approximates a continuous swept tube. Coincident samples are deduplicated for material geometry only; the manufacturing graph and route retain separate fiber identities. A cKDTree evaluates the distance field one x-slab at a time. Lewiner marching cubes extracts the zero isosurface. The initial voxel spacing is radius/4 and increases only to satisfy the configured voxel budget, up to 0.8 radius; exceeding that resolution requirement aborts with a clear error. The sampling approximation introduces a maximum axial midpoint radial deficit of about 2.02% of radius before voxel discretization. Reported voxel spacing is an additional geometric resolution limit, not a printer accuracy claim.

Planar mesh bounds and corresponding path coordinates are jointly affine-calibrated to the requested exact outer dimensions. This small correction also scales the nominal tube width; the diameter is not an independently metrologically calibrated quantity. Curved models receive a common translation and retain aspect ratio.

Resource limits are 1.5 million centerline samples, default eight million voxels (configurable maximum sixteen million), and 1.5 million mesh faces. A cancellation check occurs between field slabs. The worker runs outside the Qt GUI thread, and application/window closure waits for cooperative termination. The preview caches projected fiber strokes and appends traversal segments incrementally; it is a dimensional fiber preview, while STL/3MF contain the verified closed triangle surface.

Every exported mesh must have exactly two incident triangles at each mesh edge, finite vertices and nonzero signed volume. Negative global orientation is reversed. This validates closed mesh incidence and volume, not all possible nozzle, support or material constraints.

## Files and Bambu Studio

- Binary STL stores the closed triangle mesh with millimetre coordinates. STL itself has no unit field.
- Standard 3MF explicitly declares `unit="millimeter"`, contains one mesh object and build item, and carries no assumed printer/material profile.
- CSV stores the closed route in millimetres, with step index, incoming edge identity and node identity. Repeated geometric fibers are distinguishable by edge identity.
- All files are written to a temporary sibling and atomically replaced after completion.

The Bambu button searches the environment override `FIBERSCOPE_BAMBU_STUDIO`, conventional installation locations and Windows App Paths, with a manual executable picker fallback. It saves a uniquely named local 3MF and launches `[executable, absolute_model_path]` without a shell. It neither emits printer-specific G-code nor transmits a printer job. The user selects the printer, material and slicing settings in Bambu Studio. Standard slicing may replace the exported graph traversal with another deposition order; route preservation by the slicer is not claimed. See the [official Bambu Studio command-line usage](https://github.com/bambulab/BambuStudio/wiki/Command-Line-Usage).

The mesh extraction follows the [scikit-image marching-cubes API](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.marching_cubes). The installed, bundled version is pinned in requirements and its licence is recorded with the distribution.

## Verification and limits

The topology suite tests all built-in units plus a boundary-crossing custom cell against all seven expansion modes and three deformation amplitudes (252 comparisons), shared-square multiplicity, high-degree custom junctions, exact all-edge traversal, surface deformation identity, dimensional envelope, solid closure, both mesh formats, cancellation and the Bambu launch contract. The real-asset runner separately checks the original three assets × eleven units and builds a closed solid for each asset; its atomic report is reusable only when related source hashes match.

An Eulerian multigraph remains traversable regardless of geometric overlap, but arbitrary overlap can still cause repeated deposition, excessive thickness, inaccessible paths or process violations. The article's angular, length, cycle-density and material/process limits are not universally certified by this conversion. No physical printing experiment is claimed. Bambu Studio was not installed on the validation machine, so actual GUI import, slicing and printer output remain unverified there.

### Frozen external-program launch

Before opening Bambu Studio, the Windows frozen process temporarily clears its inherited DLL search directory and restores it after process creation. Bundle-relative PATH and Qt plugin entries are removed from the child environment, and the child working directory is the slicer installation directory. This follows the [PyInstaller external-program guidance](https://pyinstaller.org/en/latest/common-issues-and-pitfalls.html#launching-external-programs-from-the-frozen-application). The regression executes this branch with a captured process-launch call; it does not claim an installed-slicer test.
