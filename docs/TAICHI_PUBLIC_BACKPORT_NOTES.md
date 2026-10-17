# Python and source compatibility

The package declares Python >=3.9 support.
The completed-state observer uses the ordinary frozen/slotted dataclass on
Python >=3.10. A narrowly scoped Python 3.9 fallback constructs the same event
fields in slots, preserving frozen field assignment and excluding position and
velocity arrays from generated equality, hash and repr. It is not a general
replacement for Python's dataclass decorator or a security boundary.

The event emitted by the solver owns immutable-byte-backed float64 snapshots.
Normal copy/pickle reconstruction preserves event fields and values but does not
promise NumPy buffer flags; copied objects are not newly emitted solver events.
The callback restrictions, synchronous completed-step cadence, exception
propagation and observer=None behavior remain unchanged.

Exercising a compatibility branch under a newer interpreter does not prove that
Python 3.9 imports and runtime behavior work. Run the supported-version checks
on their actual interpreters. Source/mock checks alone do not establish native
numerical parity or performance for an installation. See [the API guide](TAICHI_API_DISPATCH.md),
including the observed floating-point side effect and caller-managed process
isolation.

Optional state, explicit masses, units and result serialization do not infer
material properties automatically. Existing
source-bound generation/batch resumes intentionally reject changed source hashes.
Use a fresh output directory or retain the frozen old source; do not relabel
stored checkpoints to bypass that mismatch.
