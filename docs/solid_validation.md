# Solid triangle validation

`FiberSolid.validate()` now rejects the first triangle whose stored coordinates
are exactly collinear, reporting its zero-based face row. This check precedes
wrapper-array assignment, orientation correction and export staging. Distinct
vertex indices alone do not establish positive triangle area. A failed check
leaves the original wrapper arrays and existing export destination untouched.

The check decodes coordinate bytes and uses exact integer arithmetic. It has no
area tolerance and does not discard, weld or repair triangles. Positive areas,
including extremely small ones, remain positive for this predicate. The existing
closed-edge incidence checks, floating signed-volume calculation and absolute
volume floor of `1e-9` are unchanged; the new predicate does not certify their
accuracy at extreme scales. It concerns stored values, not information already
lost during construction or rounding.

## Coordinate dtype boundary

This release tightens the accepted coordinate dtype boundary. Signed and unsigned
integer storage through 64 bits and representation-qualified IEEE binary16,
binary32 and binary64 arrays are supported without narrowing. Native/non-native
byte order and strided arrays are supported. Genuine wider or unqualified float
formats, complex, boolean, object and other coordinate kinds raise an explicit
`ValueError`. A platform `longdouble` alias is accepted only if its actual storage
is the supported binary64 representation. No implicit conversion is performed to
make unsupported values pass; the earlier implementation did not enforce this
explicit whitelist.

The exact predicate uses constant auxiliary storage per triangle and
representation-bounded integer operations. It does not add an all-mesh rational
coordinate cache. Later closure and volume calculations retain their existing
allocations. This is geometric input validation, not a guarantee against
self-intersection, printer infeasibility or mesher defects. If a generated solid
fails, retain the geometry for diagnosis rather than deleting faces to bypass the
check: deletion can open the surface.

## Surface extraction precision

`tube_mesh` requires the optional Manifold provider's `Manifold.to_mesh64` API,
available in the verified 3.5.1 provider. An unavailable or non-callable API raises
a clear error before primitive construction; there is no float32 fallback or
automatic dependency installation. The provider remains a lazy optional import.

The mesher extracts the initially simplified native solid directly in binary64.
It no longer rounds surface coordinates to four decimal places or reconstructs
that rounded surface twice. Positions are copied into owned, writable float64
storage, and triangle indices are checked before conversion to owned int32
storage. This preserves the return types and allows `build_solid` to scale and
translate its returned positions without mutating provider-owned arrays.
Coordinates, vertex/triangle counts, ordering and serialized bytes can change
from earlier releases; they are not a stable byte-level compatibility contract.

Initial graph-point deduplication, native simplification and the existing
touching-contact displacement policy remain. Binary64 extraction does not make
those operations exact or certify the generated geometry. Keep the exact-zero
check and other validation enabled. Closed edge incidence and signed volume do
not replace independent vertex-link, connectivity, self-intersection or
manufacturing-process checks.

An accepted in-memory solid does not establish serialized precision. Binary STL
writes float32 coordinates; export validates those exact coordinates after
welding equal positions and refuses invalid serialization before replacing the
destination. 3MF writes 17 significant digits so stored binary64 coordinates
roundtrip through a binary64 XML parser; downstream readers may use lower
precision. Validate the coordinates and topology actually read back from the
chosen format before using it for a precision-sensitive workflow. Neither format
certifies connectivity, self-intersection freedom or manufacturing suitability.
This does not repair previously generated files or discard degenerate triangles.
