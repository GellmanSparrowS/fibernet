# Portable generation and optional acceleration CI

## Generation contracts

The original 13 hashes and full pre-optimization JSON records are retained in
`tests/test_voronoi_owned_bridge.py` and
`tests/fixtures/voronoi_pre_owned_baselines.json`. The JSON is copied unchanged
from the original pre-optimization evidence, not regenerated from the candidate.
The independent copy-based bridge in `tests/fixtures/voronoi_pre_owned_bridge.py`
is extracted verbatim from that original source.

Every CI platform runs all 13 cases, comparing the optimized path with that
original copy-based bridge in the **same environment**. Every parsed field and
canonical JSON byte must match exactly, including seeds, coordinates, material
values, metadata, node/edge IDs and insertion order. The existing alias and
mutation-isolation tests remain. No production generation code is changed.

Against the historical full JSON, keys, shape, types, non-floating values,
IDs, connectivity and list order remain exact. Only floating-point leaves allow
1e-12 relative/absolute roundoff across numerical builds. This is an additional
cross-platform check, not a substitute for exact original-versus-owned equality.
Diagnostic output reports changed float fields and maximum absolute delta.
Changes in topology, ordering, or larger numerical values fail explicitly.

The historical SHA is additionally enforced on the verified replay environment:
Linux x86_64, CPython 3.12.14, NumPy 2.3.5, SciPy 1.17.0. All 13 original hashes
were reproduced there. The non-Voronoi points fixture retains its hash everywhere.
A fixed seed guarantees repeatability within the numerical environment; it does
not promise cross-platform QHull/NumPy binary-identical arithmetic.

### What the failing run establishes

macOS arm64 Python 3.12 job 116845491150 installed NumPy 2.5.4 and SciPy 1.18.1,
and failed 12 Voronoi hashes; the non-Voronoi case passed. Canonical JSON already
uses `sort_keys=True`, so dictionary serialization order cannot explain it.
The generator `_unit_voronoi` and `_clip_segment_to_box` have identical ASTs
to the pre-optimization source. The old log contains hashes only: it does **not** establish whether the change
is QHull geometry/order or floating-point roundoff. The new diagnostic CI step
supplies parsed-field evidence and independent old-path comparison. Until that
run passes, macOS equivalence and the specific numerical cause remain unverified.
Do not describe the hash mismatch alone as a proven harmless rounding change.

## Taichi availability is separate from core support

Official release files checked 2026-10-16:
https://pypi.org/project/taichi/1.7.4/#files

The release contains 14 wheels and no source distribution:

- CPython 3.9–3.13: Linux x86_64 (glibc 2.27+) and Windows AMD64
- CPython 3.10–3.13: macOS arm64 (macOS 11+)
- No macOS CPython 3.9, macOS x86_64, or Linux aarch64 wheel

These are artifact availability conditions, not a claim of tested native-runtime
ownership on every listed combination. Platform markers cannot express glibc
or macOS minimum OS versions; pip still validates wheel compatibility.

The `dev` aggregate includes pinned Taichi only for those interpreter/platform
families. Explicit `accel` and `full` requests retain the unconditional 1.7.4 pin:
on unsupported systems they fail visibly rather than pretending acceleration
was installed. There is no fallback to 1.7.3. The runtime ownership guard remains
unchanged. Core support is still Python 3.9–3.12 on all three CI operating systems.

The full 12-job matrix is retained. macOS Python 3.9 asserts Taichi is absent,
runs the explicit missing-backend API test, and then runs the full ordinary test
suite (existing optional-native tests retain their dependency guards). Other
matrix jobs continue to install 1.7.4 when wheels are available. The install
marker itself is tested against representative supported and unsupported tags.
