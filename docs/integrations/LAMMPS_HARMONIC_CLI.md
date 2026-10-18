# Bounded LAMMPS CLI backend

FiberNet can compile an authoritative `StructureGraph` into an explicit SI site-and-harmonic-bond model and execute it through a bounded, isolated CLI task. The engine is optional: importing the adapter, preparing inputs and parsing saved results do not import, install or start LAMMPS.

This interface is for synthetic/coarse-grained mechanical sites. A graph node is not automatically an atom, and a fiber material record is not automatically a molecular force field. Geometry and material provenance stay in the graph. Chemistry, calibrated coarse-graining, bending, torsion, contact and fracture require separate model contracts.

## Prepare without an engine

```python
from fibernet.core.structure_graph import StructureGraph
from fibernet.integrations import HarmonicBond, RunSpec, prepare_harmonic_beads

graph = StructureGraph(dimension=3)
left = graph.add_node([0.0, 0.0, 0.0])
right = graph.add_node([0.0011, 0.0, 0.0])
edge = graph.add_edge(left, right, n_internal=0)

prepared = prepare_harmonic_beads(
    graph,
    length_unit="m",
    maximum_bond_length_m=0.0015,
    masses_kg={left: 1e-9, right: 2e-9},
    bonds={edge: HarmonicBond(0.3, 0.001)},
    run=RunSpec(1e-7, 100, 1, ((-0.002, 0.003),) * 3),
)
print(prepared.manifest())
```

Every site mass must be finite and positive in kg. Every physical bond requires a finite positive spring stiffness k in N/m and equilibrium length r0 in m. The physical energy is 0.5*k*(r-r0)^2; the compiler writes the LAMMPS harmonic coefficient K=k/2, following the [official harmonic convention](https://docs.lammps.org/bond_harmonic.html). Rest lengths must be supplied independently of a deformed current graph.

The compiler currently requires 3D, straight nonzero edges, a fixed orthogonal nonperiodic box, 2–256 sites, 1–1024 physical bonds, at most 100,000 steps, and every step retained. Curves/internal points and parallel physical edges are explicitly rejected rather than flattened or deduplicated. Shared graph junctions become one site. Crossing edges do not create new welded junctions or contact.

Mass types group only exactly equal normalized binary64 kg values, without tolerance or rounding. Native atom IDs, graph node/edge IDs and their maps remain separate from type IDs. The parser requires each atom's expected type, not merely a type in the valid range. Material metadata, boundaries, loads and stable IDs do not become type-grouping keys.

[`pair_style none`](https://docs.lammps.org/pair_none.html) avoids an unused pair-force neighbor list. An explicit `maximum_bond_length_m` supplies communication reach; every saved integration step must remain within it and strictly inside the declared box. This is a one-rank qualification, not a guarantee for arbitrary deformation or MPI. Pair-free execution alone does not imply scalability; type tables, bonds, parsing and every-step trajectory output still consume resources.

## Explicit runtime and lifecycle

The publication candidate supports `RuntimeSpec(mode="cli", ...)` only. Linked Python-engine mode and automatic provider discovery fail closed. Native execution is restricted to Linux x86_64 and the qualified wheel-style layout: native `lammps/lmp`, `liblammps.so`, `liblammps.so.0`, bundled `lammps.libs` libraries and an explicit existing MPICH `libmpi.so.12` directory. It is not a generic adapter for every system installation.

The tested external provider is the unofficial PyPI wheel `lammps==2025.7.22.4.0`, based on official upstream `stable_22Jul2025_update4`, with MPICH 5.0.2. [Official current documentation](https://docs.lammps.org/Intro_opensource.html) identifies release 30Sep2026; that does not upgrade or qualify the tested runtime. The adapter does not install either provider and does not add either engine to FiberNet's mandatory, `dev` or `full` dependencies.

Supply absolute paths and SHA256 pins from a separately trusted runtime inventory. Required fields are `mode`, `python_executable`, `lammps_executable`, `package_dir`, `expected_version`, `expected_banner`, `file_sha256` (path/hash pairs) and `environment` (key/value pairs). Required binary/library pins are checked before launch and again in the worker. The trusted inventory should also pin bundled MPICH libfabric/UCX dependencies. RuntimeSpec enforces a minimum layout plus every supplied pin; it does not discover or certify a complete dependency closure. Host loader, system ABI libraries and the worker interpreter standard library remain ambient unless separately inventoried. Hashes establish byte identity, not provenance, and do not secure mutable files against an adversarial same-user race. The runtime inventory must include its provider/source/license provenance outside this dataclass.

Do not use the wheel's `bin/lmp` Python wrapper: it can discover installed entry-point plugins. The CLI task invokes the exact native executable through its own stdlib worker using `-I -S`, with a fresh exclusive working directory and task-local TMPDIR. The child receives a small environment; only explicit LD_LIBRARY_PATH and UCX_TLS additions are accepted, and thread counts are one. There is no LAMMPS plugin-path inheritance or persistent service. No global MPI, network or security settings are changed.

```python
from fibernet.integrations import LAMMPSTask, ResourceLimits, TaskState

# runtime is a separately reviewed RuntimeSpec for an existing native engine.
task = LAMMPSTask(prepared, runtime, new_workdir, ResourceLimits(
    cpu_seconds=3, wall_seconds=5, memory_bytes=1024**3,
    output_bytes=4*1024**2, file_size_bytes=8*1024**2,
))
try:
    task.start()
    state = task.wait(timeout=7)
finally:
    if task.state == TaskState.RUNNING:
        task.cancel()
        task.wait(timeout=2)
if task.state != TaskState.SUCCEEDED:
    raise RuntimeError(task.error)
result = task.result
```

Construction performs no native run. Tasks are one-shot and expose created/running/succeeded/failed/cancelled/timed-out/resource-limit states. Invalid inputs, native nonzero exit, missing/truncated output, wrong version banner and parsing errors never become an empty successful result. Process-group termination/reaping occurs on terminal and exception paths. Mocked lifecycle tests are not proof of all real OS cancellation scenarios.

The child has kernel CPU/address-space/per-file/core limits. The watchdog's wall and recursive aggregate-artifact limits are sampled failure thresholds with a final check; output can overshoot between samples. `file_size_bytes` is distinct from `output_bytes`: runtime ordinary scratch files may require a larger per-file allowance than retained logs. Anonymous/shared-memory and all writable paths are not certified task-local. This is resource control, not a filesystem or hostile-code security sandbox. Typed result arrays also consume memory in the parent process. Save any additional result exports within your application's own resource budget.

## Results and scientific scope

`LAMMPSRunResult` contains typed trajectory and thermo frames. Trajectories map back to stable graph IDs and expose positions in m, velocities in m/s and forces in N. Thermo energies are J. The six global pressure components are Pa, compression-positive. This fixed-box mechanical virial is not a calibrated fiber or bulk material stress; site kinetic energy is not automatically physical material temperature.

Dump units, time, integer steps, complete unique identities, expected types, box, finite values, initial state, PE+KE consistency and a graph/contract-bound completion marker are checked. Returned energy is the engine's mechanical thermo energy. Explicit constant nodal forces and whole-node fixed constraints have compiler/protocol tests, but the native scientific qualification below uses zero loads and no fixed sites. External work is not integrated into the result, so do not apply closed-system energy conservation to loaded runs. Geometric boundary flags are preserved as provenance; fixed sites are an explicit separate argument.

Native evidence for the unchanged compiler protocol includes analytical two-site static and unequal-mass oscillator checks, and a heterogeneous 64-site/441-bond generator-defined braced-cubic graph with three mass and stiffness classes. Three network trajectories (100/200/400 steps, 703 frames total) passed saved-frame force/energy/conservation, local integration-residual and common-time refinement checks. Finest energy drift was about 1.48 ppm over the declared short interval. These are bounded correctness cases, not a large-scale performance, contact, beam, chemistry or MPI claim. The CLI-only publication revision itself still requires its final source/runtime admission review; earlier native evidence is not silently relabeled as a new execution.

The legacy `LAMMPSBridge`/`run_lammps_md` API remains for compatibility and is not covered by these checks. Its `_LAMMPS_AVAILABLE` module flag indicates adapter importability, not a verified installed engine.

## Representative example and CI

From a source checkout, `python examples/22_lammps_harmonic_network.py` compiles the frozen representative graph and prints its identity without starting an engine or writing output. Its data retain source-fiber/endpoint provenance, prescribed SI parameters and source geometry checks. `--execute` additionally requires a trusted runtime JSON and a new work directory. Do not use execution flags in ordinary CI.

Pure CI tests need only FiberNet's existing dependencies and test tools. They use synthetic parser fixtures and mocked process lifecycle; no LAMMPS download, credentials, network access or persistent service is required. POSIX-specific lifecycle tests are skipped on Windows; compilation, mapping, parsing and unsupported-runtime errors remain portable. Source parses under Python 3.9 grammar, but grammar checks do not replace actual Python 3.9 CI. The worker's former `entry_points(group=...)` incompatibility is removed with the unqualified linked-engine route.

See [LAMMPS license and provider notice](LAMMPS_LICENSE_NOTICE.md) before distributing an engine or a linked interface.

## Explicit serial-static CLI profile

`RuntimeSpec(mode='cli_serial_static', ...)` admits a separately verified Linux
x86_64 static CLI executable without the MPI wheel layout required by `cli`.
Provide existing canonical absolute executable paths (no symlink aliases) for the
supervisor Python and native CLI, canonical resolved-path SHA256 pins for both, `package_dir=''` and
`environment=()`. Every additional supplied pin is checked; duplicates, malformed
or nonhex pins are refused. This profile does not install or discover a runtime. Resolve supervisor paths
explicitly, for example `str(Path(sys.executable).resolve())`. Canonical paths and
launch-time rehashing do not eliminate arbitrary filesystem replacement races;
keep the pinned runtime in a controlled workspace.

The nonexecuting ELF filter accepts only little-endian ELF64 System V x86_64
ET_EXEC with bounded standard header tables, file-backed executable entry code,
and LOAD/TLS/GNU_STACK/GNU_RELRO segments. It rejects interpreters, dynamic
tables, executable stacks and unsupported layouts. It is deliberately narrower
than general static ELF support and does not establish trust, native capabilities
or physical correctness. Use independently verified artifact provenance, license
and capability evidence tied to the exact native executable hash.

The existing `cli` MPI validation and all compiler, task lifecycle, limits,
output parsing and result contracts are unchanged. The worker rechecks the
provided hashes before the same fixed no-shell native launch. Serial mode does
not route the legacy LAMMPSBridge API into this backend.

`expected_banner` is checked against native output. `expected_version` remains
range-validated metadata, not an independent query of the engine version; bind
the date, update and required capabilities to the binary in preflight evidence.
