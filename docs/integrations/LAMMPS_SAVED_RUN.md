# Reopen a saved LAMMPS task without running an engine

`read_saved_lammps_run` restores a `PreparedRun` from the existing task files and
passes captured output text to the same strict parser used by the bounded CLI
backend. It is an offline input-byte and output-protocol check, not a restart,
native execution receipt, output authentication service or physical validation.

```python
from fibernet.integrations import read_saved_lammps_run

# Retain this digest independently when recording the original input manifest.
# Computing it from an unknown bundle now does not establish that bundle's origin.
saved = read_saved_lammps_run(
    "saved-task", expected_manifest_sha256=retained_manifest_sha256,
)
prepared = saved.prepared
raw = saved.result
print(saved.receipt["status"])
```

The caller must supply a lowercase SHA256 for the original `manifest.json`.
An optional `expected_banner` checks the log's exact first line against a
caller-supplied nonempty string. A digest or banner is only as trustworthy as its
source. Neither establishes permission, provenance, native engine identity or
that LAMMPS ran.

## Saved files and returned values

The reader opens exactly six fixed names in the caller-selected directory:

- `manifest.json`: the full existing task manifest, including runtime and limits
- `graph.json`, `model.data`, `in.model`: exact stored preparation bytes
- `trajectory.dump`, `log.lammps`: output protocol texts

Other files, including process logs and scratch entries, are ignored and are not
validated. The files are never modified. Runtime paths, environment and hash pins
are retained as inert, unverified JSON. They are never opened, admitted, installed
or applied. `in.model` is checked as text, never executed.

The immutable `SavedLAMMPSRead` contains:

- `prepared`: the existing `PreparedRun`, with original StructureGraph geometry,
  stable IDs and explicit model contract
- `result`: the existing `LAMMPSRunResult`; positions are m, velocities m/s,
  forces N, energies J, and global pressure Pa. Its `workdir` is empty.
- `receipt_json` and detached `receipt`: byte digests, IDs, frame counts, original
  graph length unit and explicit nonverification flags
- `unverified_metadata_json`: saved runtime and limits records, with no implied
  runtime readiness or execution authenticity

The graph remains authoritative; topology is not inferred from native atom rows.
Atom/type IDs stay distinct from stable node/edge/angle IDs. No resampling,
material calibration, stress/reaction recovery or solver fallback is introduced.
No `SimResult` is manufactured because the saved input manifest cannot establish
the runtime and elapsed wall-time fields that conversion would otherwise present.

## Identity and compatibility

Manifest schema, preparation/graph/input hashes and all redundant ID/type maps
must match using type-sensitive comparisons. Boolean IDs cannot alias integers.
The exact stored preparation must also regenerate under the current compiler.
The output parser retains its existing schedule, SI, identity, initial/held-state,
domain, energy-arithmetic and completion-marker checks.

Current manifests do not record a compiler revision. A mismatch raises
`LAMMPSCompilerCompatibilityError`, which can mean modified bytes or a different
source/Python runtime. No byte normalization, historical compiler substitution or
automatic migration occurs. In particular, a documented one-ULP derived norm
difference on Python 3.9 may make older CG preparation bytes incompatible.

Catch the common `LAMMPSContractError` for any contract failure. Reader failures
use its `LAMMPSSavedReadError` subclass; unchanged parser errors can propagate
directly. Missing or invalid input never produces a successful empty result.

**The input manifest does not authenticate the output files.** Altered finite
forces or other protocol-valid output can pass the same input manifest anchor.
The new receipt records the captured output digests but does not prove they are
the original engine output. A separately trusted original-output anchor or other
evidence is needed for that claim. The saved input manifest also does not establish
the original outer task's terminal success. Completion markers are copyable text.

## Platform, paths and finite limits

This reader requires POSIX no-follow directory descriptors. Windows and systems
without the required support fail explicitly before file access. Importing the
public API remains portable. This is separate from native CLI platform admission
and does not change the existing `parse_output_files` path behavior.

Use canonical, quiescent local regular-file paths. Symlink roots/ancestors and
symlink or nonregular leaves are refused. Reads are descriptor-relative, chunked
and bounded. Pre/post file signatures detect ordinary replacement or mutation;
they are not an atomic snapshot against a privileged or same-user adversary.

`LAMMPSSavedReadLimits()` defaults and maximum supported bounds are: total raw
input 20 MiB; prepared files together 2 MiB; output pair 16 MiB; each JSON 1 MiB;
JSON depth 32 and punctuation/container budget 100,000; 256 nodes, 1,024 edges
and 2,048 steps. Callers may lower these limits. The selected existing compiler
or parser profile can impose stricter bounds. These reader ceilings do not expand
any native simulation profile.

The default cooperative deadline is 10 seconds, configurable up to 60 seconds.
Checks occur during reads and between compiler/parser phases. This is not a hard
CPU/RSS limit or a timeout for blocking network-filesystem I/O. Parsed objects and
text copies consume memory beyond raw byte sizes. Concurrent mutation and hostile
filesystems are outside the supported model.

## Verification and distribution scope

Tests use synthetic harmonic/CG output and existing public retained preparation
fixtures. They exercise exact restoration, maps/units, missing and malformed
files, resealed contradictions, compiler drift, finite budgets and no-execution
boundaries. They are not new historical-output replay, native physics, performance
or cross-platform execution evidence. Existing seven-bead and 48-bead native
claims, including the latter's resource-acceptance failure, remain unchanged.

This reader adds original adapter/test/documentation code and no engine code,
binary, wheel, MPI dependency, private research input or raw native result.
It adds no MCP tool or remote filesystem-reading route.
FiberNet's MIT license and the separate [LAMMPS notice](LAMMPS_LICENSE_NOTICE.md)
remain unchanged; it introduces no linked Python-engine route or licensing waiver.
