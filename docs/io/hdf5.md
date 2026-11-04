# Saving a FiberNetwork as HDF5

`FiberNetwork.save_hdf5(path)` writes the legacy HDF5 representation. It requires
the optional `h5py` package and accepts a string or `os.PathLike[str]`, including
`pathlib.Path`. Byte paths are not supported. Create the parent directory first.

```python
from pathlib import Path

from fibernet.core.fiber import Fiber
from fibernet.core.network import FiberNetwork

network = FiberNetwork(
    fibers=[Fiber.straight([0, 0, 0], [1, 0, 0], radius=0.1)],
    dimension=2,
)
network.save_hdf5(Path("network.h5"))
```

Each fiber must have a serializable material. Network metadata and material
records must be JSON-serializable. This method does not assign missing materials
or convert a `StructureGraph` into a `FiberNetwork`.

## What the file contains

The existing format remains version `0.1.0`:

| Location | Stored data |
| --- | --- |
| Root attributes | `version`, `dimension`, `periodic`, optional `box_size`, and JSON `metadata` |
| `fibers/fiber_N` | `centerline` dataset; `radius`, `fiber_id`, `cross_section`, `segments`, and JSON `material` attributes |
| `crosslinks/cl_N` | `position` dataset; `fiber_i`, `fiber_j`, `param_i`, `param_j`, and `crosslink_type` attributes |

This is not a lossless serialization of every model field. In particular, the
legacy HDF5 format does not store fiber `cross_section_params` or crosslink
`strength` and `stiffness`. Do not infer those values from the saved file. Use
`FiberNetwork.save_json` and `FiberNetwork.load_json` when those fields must be
preserved in the native network representation.

Files can be inspected directly with h5py:

```python
import h5py

with h5py.File("network.h5", "r") as saved:
    centerline = saved["fibers/fiber_0/centerline"][...]
```

## Replacement and failure behavior

Metadata and material JSON are serialized before an output file is created. The
HDF5 data is then written to a temporary file in the destination directory. Only
after HDF5 closes successfully is the destination replaced with `os.replace`.
A serialization or write failure before this commit leaves an existing
destination unchanged; an absent destination remains absent.

Temporary-file cleanup is attempted on failure. If cleanup is also rejected by
the operating system, a temporary file may remain. The original failure is
re-raised; secondary close or cleanup failures are available through
`hdf5_close_error` and `hdf5_cleanup_error` when present. A failure closing the
initial temporary-file descriptor is marked `hdf5_descriptor_close_uncertain`;
the method does not blindly retry that close.

## Destination restrictions

- Existing destinations must be regular files with exactly one hard link.
  Symbolic links, directories, and other file types are rejected.
- Existing files with no owner, group, or other write-mode bit are rejected.
  This conservative check is not an ACL authorization check; filesystem
  operations can still fail for permission or other reasons.
- Existing permission bits are retained. New files use the private permissions
  provided by `tempfile.mkstemp` (normally `0600` on POSIX).
- Replacement creates a new file object. Ownership, ACLs, extended attributes,
  and inode identity are **not preserved**.

Same-directory atomic replacement is not a power-loss durability guarantee:
the method does not promise `fsync` persistence. It also does not provide
concurrent-writer arbitration or protection against concurrent changes to the
destination directory. Use a destination directory under your control.
