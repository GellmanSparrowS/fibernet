# API contract revision 2 source snapshots

These four exact-source snapshots record the reviewed API-contract revision,
not a native solver qualification. The original
`../public_source_contract` snapshots remain immutable and are still used for
unaffected modules and independent native-kernel AST comparison.

Only `fibernet/__init__.py`, `fibernet/easy.py`, `fibernet/api.py`, and
`fibernet/sim/accelerated.py` route to this revision. All four are checked byte for
byte, including comments and line endings. `manifest.json` records exact SHA-256
hashes of both revisions and the original public commit provenance. It is not an
allowlist of interchangeable versions.

The source tests still reject changed writers, added unrelated statements,
mutated LAMMPS guards, and arbitrary source changes. The only argument adjustment
in the historical observer comparison first asserts the exact reviewed damping
annotation/default transition, then compares every other positional argument and
default unchanged. Extracted tests execute the actual new damping helper and
constant with prescribed inert fields, never a replacement physics model.
