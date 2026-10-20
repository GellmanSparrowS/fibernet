# Explicit geometry-export examples

For small export illustrations, load an unchanged bundled OBJ with
`load_obj(path, target_faces=48)`, then use `compile_surface` and `build_solid`
with these explicit settings:

- Square grid 3 by 3, seed 7.
- `PrintSettings(width=250, depth=250, curved=True)`: default diameter 2 mm,
  `max_height=250` mm and `up_axis='z'`. The planar `height=2` field is not the
  curved height limit.
- Shirt, vest and Paris: one sample per segment, perturbation 0.
- Sleeve: two samples per segment, perturbation 0.35.

The four configurations passed actual STL-byte validation and exact stored
binary64-coordinate and face-index 3MF readback locally. The requested count is
not exact: the loader produced 48 quads for shirt and 45 for the other three.
These settings are opt-in illustrations; no UI default or source asset is changed.

## Coarsening changes the design

A successful export is not a repair of the original dense geometry. Reducing
mesh density changes the visible surface and the resulting fiber network. The
sleeve recipe additionally changes its fiber pattern. The neutral sleeve at the
same target of 48 still fails STL validation and is not a positive example.

The coarse shirt spans were 96.873 by 74.788 by 28.854 source units, versus
115.941 by 77.253 by 34.126 originally. Coarse Paris spans were 0.918 by 6.614
by 0.840, versus 2.683 by 6.614 by 2.587 originally: substantial lateral detail
is lost. This Paris illustration is not a contour-preserving replacement or a
repair for its dense configuration. The regression reports each coarse/original
span ratio and checks noncollapse, not shape fidelity; choose density based on
the intended design and inspect its preview.

## Original configuration boundaries

The original fixture checks remain separate, with their original extent guards:

- `demo_sleeve.obj` (360 quads), `demo_vest.obj` (272) and
  `reference_shirt.obj` (248), using the neutral one-sample square configuration,
  produce accepted binary64 indexed surfaces but invalid welded binary32 STL
  edge incidence. The test requires that defect, actual STL refusal without
  creating/replacing a destination, and stored-coordinate/index 3MF readback.
- `reference_paris.obj` (576), with the same neutral configuration, is refused
  during construction with `ambiguous zero-thickness solid contact`. The current
  contact-normal guard has no reliable separation direction for at least one
  detected contact. The test requires this exact refusal and unchanged network
  arrays, rather than treating arbitrary errors as expected.

These cases use the same 250 mm limits, 2 mm diameter, grid 3 by 3, seed 7 and
z-up coordinates listed above. Original names and line-ending-normalized source
hashes are pinned so changed fixtures require renewed investigation.

## Validation scope

Local evidence used Linux Python 3.12, NumPy 2.3.5, SciPy 1.17.0,
manifold3d 3.5.1 and fast-simplification 0.1.13. Windows CI and other runtimes
must execute their own checks. Any different configuration remains subject to
the production writer's unchanged fail-closed guards.

These are geometry and file-format checks, not manufacturing qualification.
Neither format certifies self-intersection freedom, material connectedness,
sliceability, printer settings or a physical part. 3MF preservation refers to
the stored binary64 mesh through a binary64 XML parser, not every consumer's
precision.
