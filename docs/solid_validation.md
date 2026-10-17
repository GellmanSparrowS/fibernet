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
