# Thresholded tensile recruitment: computational definition

## Input and reproducible entry point

The public Python API is `fibernet.analysis.analyze_tensile_recruitment`; its inputs are a finite F × E matrix of engineering edge strains, an E × 2 array of zero-based node indices, the left and right grip node sets, and the number of nodes. The APP adapter accepts a trajectory carrying the same arrays and delegates to this implementation. From the repository root, the library-only example is `python -m examples.tensile_recruitment_quickstart`. The output contains framewise recruited masks, spanning-component masks, node and edge fractions, normalized topological depth, the first spanning frame, and the threshold trajectory. It does not require Qt or FiberScope.

## Definition

At each frame, the activation threshold is the larger of a fixed positive floor and a fraction α of that frame's largest positive edge strain. Optionally, a quantile of only positive strains replaces the fractional maximum. An edge is newly recruited only if its positive strain **strictly exceeds** this threshold. An edge recruited in the preceding frame remains recruited while its strain exceeds a fraction h of the current threshold, where h is the hysteresis multiplier. The default values are α = 0.05, h = 0.6, and minimum threshold = 10⁻⁴. The method evaluates each frame in order; changing the sampling interval can therefore change the retained masks.

When grip filtering is enabled, connected components without a node in either grip set are removed from the recruited mask. A spanning component has recruited paths to both grip sets. The spanning node fraction is the number of nodes in such components divided by all nodes; the backbone edge fraction is the number of edges in such components divided by all edges. A multi-source breadth-first search from each grip computes the minimum graph distance to either side for nodes in spanning components; edge depth is normalized to the maximum depth of the spanning subgraph. If no spanning component exists, fractions and depths are zero and the first spanning frame is −1.

## Scope and limitations

This is a **positive axial-strain threshold graph**, not a force measurement or mechanical percolation criterion. Compression, bending moments, contact traction, out-of-plane behavior, and edge failure can contribute mechanically while their edges appear inactive here. The analysis depends on loading direction, threshold, floor, hysteresis, frame resolution, and grip definition. Claims about negligible mechanical function require independent perturbation and recalculation, preferably with edge-force and energy information. The current APP stretch solver provides the trajectory used for its animation; the Python API also accepts arrays from any solver with compatible node/edge indexing. APP and API equivalence is checked using the same function, while numerical solver credibility requires separate benchmarks.

## Software verification

The library test covers a three-edge graph with positive, negative, and hysteresis-retained strains, first spanning time, APP adapter equivalence, and rejection of malformed inputs. The APP's real-trajectory selftest and numerical golden regression test the same calculation path after migration. A frozen executable has not yet been rebuilt; the build recipe vendors the lightweight analysis module, but packaged runtime verification remains a release gate.
