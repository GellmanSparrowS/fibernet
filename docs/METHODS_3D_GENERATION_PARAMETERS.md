# 3D generation parameter boundaries

`fibernet.pattern_3d` returns a `StructureGraph`. Intermediate graph nodes
(`n_pts_per_side`) and stored edge interior points (`n_internal`) are distinct.
This note describes parameter acceptance, not mechanical validity or a new
geometry capability.

## Unit-cell dimensions

`pattern_3d` requires `box` to be a sequence or one-dimensional NumPy array
containing exactly three finite positive real dimensions `(w, h, d)`.
Zero, negative, nonfinite, boolean, string, complex and missing values raise
`ValueError` before a built-in or registered factory runs. This also applies
when a grid axis is zero. A valid box with a zero grid axis still produces an
empty tiling after unit construction.

Previously a zero-width cubic cell could collapse to a planar graph, negative
widths could yield negative nominal extents, and complex NumPy coordinates could
lose their imaginary parts. Such inputs now fail explicitly. Choose positive
real cell dimensions rather than relying on collapse, reflection or coercion.

```python
from fibernet import pattern_3d

graph = pattern_3d(unit="cubic", box=(6.0, 8.0, 10.0), grid=(2, 1, 1))
assert (graph.num_nodes, graph.num_edges) == (12, 20)
```

For this 3D entry point, top-level arguments override `unit_kwargs` entries.
In particular, `unit_kwargs={"box": ...}` neither overrides nor repairs the
top-level box. The separate 2D built-in factory override policy is unchanged.
Valid input containers and scalar types are passed through without normalization,
so existing provider arithmetic, geometry, IDs, metadata and material forwarding
are retained. An unmasked NumPy masked array remains valid; missing coordinates
do not. Custom factories now share this public entry-point dimension contract.

This check does not change direct tiling or private unit-factory APIs, introduce
length units, or guarantee that extreme dimensions, grid products or requested
meshing allocations are representable. Existing downstream checks still apply.

## TPMS units

The built-in `gyroid`, `schwarz_p`, `schwarz_d`, `iwp`, `neovius` and `lidinoid`
units do not implement these subdivision or intermediate-node displacement
controls. They require:

- `n_internal=0`;
- `n_pts_per_side=0`;
- `point_displacements=None` or an empty sequence.

A nondefault request now raises `ValueError` before the geometry builder or its
optional meshing dependency is invoked. Previously these parameters were
silently ignored, and requested `n_pts_per_side` could appear in metadata even
though no subdivision occurred. This is an explicit compatibility change from
ignored options to an unsupported-option error, **not new TPMS subdivision
support**. Existing default geometry, radius/material forwarding, seed handling
and connectivity-repair behavior are unchanged. Historical subdivision metadata
alone is not evidence that subdivision was performed.

None, empty lists/tuples and an empty `(0, 3)` array remain accepted no-ops.
No meshing dependency is required merely to reject an unsupported request.

## Explicit displacements for non-TPMS built-in 3D units

For the built-in edge-based units, a nonempty `point_displacements` sequence
must contain exactly one entry per requested intermediate graph node: the
unit's enumerated edge count times `n_pts_per_side`. For example, a cubic cell
has twelve enumerated edges, so `n_pts_per_side=1` requires twelve entries.
A nonempty sequence is invalid when `n_pts_per_side=0`.

Mismatches raise `ValueError` with the expected and supplied counts before the
edge-construction helper mutates its graph or creates a random generator.
Previously a shorter nonempty sequence was discarded for seeded automatic
displacements, and a longer sequence was truncated. Callers must supply the
exact count rather than rely on either behavior.

Compatibility is retained for:

- None or an empty sequence: seeded automatic displacements for a positive
  intermediate-node count, and a no-op at zero;
- exact-length explicit sequences in single-group units: unchanged positions,
  stable IDs and incidence (the octet ordering correction is described below);
- the existing automatic random draw sequence, seed default and raw-coordinate
  displacement amplitude.

This repair does not introduce broader count/type/shape/radius validation, unit
conversion or new random models. Registration and public dispatch are unchanged;
custom factories that reuse these built-ins/helpers inherit their corrected
parameter contract.
It does not change the 2D custom-points `fit_to_box` restriction.

## Verification scope

The box-contract tests exercise all eight built-in edge-based units, native
serialization roundtrips and all six TPMS families with real scikit-image
meshing when that optional dependency is installed. TPMS provider tests skip
when scikit-image is absent. Invalid-box sentinel tests verify rejection before
factory, tiling, connectivity repair or random-number generation.

The earlier subdivision-contract tests use the real cubic and octet generators and a deliberately substituted
2-node/1-edge TPMS builder to test public parameter routing. Sentinel tests do
not validate actual TPMS meshes. They assert unsupported requests reject before
that builder, while default/no-op routing and existing graph identities remain
unchanged. No solver, trained model or rendering capability is established by
these checks.

### Octet's two edge groups

The historical `unit='octet'` generator uses twelve outer cube edges followed
by eight corner-to-body-center edges. Its nonempty explicit displacement vector
now covers **all twenty edges**, with `n_pts_per_side` consecutive entries for
each edge. The first `12 * n_pts_per_side` entries belong to the outer group;
the remaining `8 * n_pts_per_side` entries belong to the center group.

Corner indices are `(0,0,0)`, `(w,0,0)`, `(w,h,0)`, `(0,h,0)`, `(0,0,d)`,
`(w,0,d)`, `(w,h,d)`, `(0,h,d)`, respectively 0 through 7. Outer-edge order is
`(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5),
(2,6), (3,7)`. Center edges follow in corner order 0 through 7, directed from
each corner to `(w/2,h/2,d/2)`.

The total cardinality is validated before constructing the octet graph. Earlier
code reused the same supplied prefix separately in both groups; partial vectors
were either truncated or replaced by automatic values. Such nonempty partial
vectors now raise. Even a formerly supplied total-length vector has corrected
semantics: its center-group tail is used instead of reusing the outer prefix.
This is an intentional correction, not a claim of backward compatibility for
ambiguous prefix reuse.

None/empty automatic generation retains **two separate existing seeded helper
calls**, including their seed restart, edge order, center-node creation after
outer edges, and the center-edge radius multiplier 0.7. The zero-count/default
geometry and stable-ID construction order are not changed. This historical
`octet` interface remains distinct from `octet_fcc_v1`.
