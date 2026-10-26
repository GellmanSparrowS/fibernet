# Saved electrothermal fixtures

These gzip/base64 containers retain exact strict-JSON envelopes from synthetic
native electrothermal results.
Tests and the replay command reopen them without recalculating their trajectories.
They are not raw native-result dataclass dumps.

- `axial_chain.json.gz.b64`: 3 nodes, 2 physical edges, 11 accepted frames; uniform
  temperature and analytic axial-expansion/blocking comparisons.
- `planar_network.json.gz.b64`: 25 nodes, 56 physical edges, 11 accepted frames;
  nonuniform temperature, thermal flux, transverse displacement and bending.

`VALIDATION.json` records byte and canonical identities plus measured comparison
errors from the original numerical calculations. Replaying these files verifies
serialization and transfer, not a new numerical reproduction or performance run.
See [the method and limits](../../../docs/SAVED_ELECTROTHERMAL_MECHANICS.md).

The stdlib-only `benchmarks.saved_electrothermal_fixtures` loader verifies fixed
encoded/compressed/decoded size limits, canonical base64, a single complete gzip
member and the exact original envelope SHA256. It rejects trailing data, extra
members and oversized expansion. Storage uses empty gzip filename, mtime=0 and
level=9; compressed bytes may differ across zlib versions. The raw byte identities
are authoritative. This is bundled fixture storage, not an additional native
input format. Use `--fixture planar_network.json` with the replay command; normal
user-supplied JSON files retain the original strict-reader route.
