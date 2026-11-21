# FiberNet 4.9.2

## Reopen a saved rotational BPM run offline

The existing `read_saved_lammps_run` reader now accepts completed rotational
BPM task folders saved with restart disabled. On a supported POSIX system,
keep the folder's original files and the SHA-256 of `manifest.json` that you
recorded when saving the run. A folder that still matches the current input
and output contracts can be reopened without LAMMPS installed:

```python
from fibernet.integrations import read_saved_lammps_run

# retained_manifest_sha256 is the digest you recorded for the original manifest.
saved = read_saved_lammps_run(
    "saved-task", expected_manifest_sha256=retained_manifest_sha256,
)
print(len(saved.result.frames), saved.result.breaks)
```

The result keeps the original node and edge IDs, motion and break events.
It does not include a thermo energy history. This reader does not run or
restart the simulation, authenticate engine output, or verify physical
results. Folder readback is not supported on Windows. See the
[saved-task guide](integrations/LAMMPS_SAVED_RUN.md) for required files and
compatibility checks.

## Install and find the examples

```bash
python -m pip install "fibernet==4.9.2"
python -m fibernet.examples.basic_usage --output-dir fibernet_demo
```

The basic example is installed with the package. Standalone scripts under the
repository's top-level `examples/` directory are not installed by the wheel.
To run a matching script, use the 4.9.2 source distribution rather than files
from a newer GitHub checkout:

```bash
python -m pip download --no-deps --no-binary=fibernet "fibernet==4.9.2"
python -m tarfile -e fibernet-4.9.2.tar.gz
```
