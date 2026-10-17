# Saving easy-training model artifacts

Both CSV and native `fibernet.train_model(..., save_path=...)` use the same
single-file staged save. The separate Torch training API is unchanged.

```python
from fibernet import train_model

# The models/ directory must already exist.
metrics = train_model(
    "training.csv", target="force_N", features=["length_m", "area_m2"],
    model_type="lr", save_path="models/current.pkl",
)
```

For a native Dataset, pass the object and its explicit split choice as described
in [native training](native-dataset-training-usage.md); save behavior is identical.
String, path-like, and byte-string filesystem paths are supported. Omitting
`save_path` keeps the existing train-without-saving behavior.

## Publication and ordinary failures

Training, splitting, metrics, feature order, target selection, and estimator
choices are unchanged. Saving creates one exclusive temporary sibling file in
the destination directory. It writes the complete pickle and closes the stream
before applying destination permissions to that stage and calling `os.replace`.
The previous destination is never opened for writing or truncated.

An ordinary serialization, write, stream-close, permission-preparation, or
replacement failure leaves the original file's exact bytes and ordinary mode
unchanged. If the destination was absent, it stays absent. The original save
error propagates. The helper removes only its own temporary name; it never scans
for or removes similarly named files. No backup, recovery descriptor, or
multi-file transaction machinery is needed for this single replacement.

The parent directory must already exist, as with the previous writer. Saving
does not create missing ancestors. Directory write access and space for the new
pickle are required. Being allowed to truncate an existing file does not imply
permission to replace its directory entry; such replacement errors are surfaced.

## Targets and metadata

- An existing target must be a regular file. Symlinks, including dangling links,
  directories, FIFOs, sockets, and other nonregular targets reject before staging
- Existing files with setuid, setgid, or sticky permission bits reject before
  staging. Those bits are not silently removed or copied
- The stage starts with restrictive permissions no broader than `0600`. After
  serialization and close, only the old file's ordinary POSIX `rwx` bits are
  applied to the stage. A newly published destination uses `0600`
- No permission, ownership, or ACL-changing operation touches the original file
  or unrelated paths. Original ordinary permission bits are the only metadata
  explicitly preserved
- Replacement creates a different inode. Ownership/group, ACLs, extended
  attributes, timestamps, and other inode metadata are not preserved or
  guaranteed. No `chown` or ACL adjustment is attempted
- Hard-link aliases keep the original inode and bytes. Saving through one name
  replaces only that directory entry; it does not update its other hard links

Mode checks and exact `0600`/`rwx` behavior are tested on POSIX. Other platforms
may implement permission modes differently. In particular, Windows `chmod` only
supports the read-only flag; `0600` does not establish a private Windows ACL.
Directory/inherited ACLs govern access. Exclusive staging and target validation
still apply. This intentionally changes the old
writer's behavior for symlink/special-mode destinations and new-file permissions.
The caller must control the destination directory: ancestor symlinks and path
replacement races are not secured by this helper.

## Cleanup diagnostics

If removing the owned temporary file fails, a `RuntimeWarning` identifies its
path and explains that the save outcome is unchanged. A cleanup failure cannot
replace the original save exception or turn an already committed save into a
reported failure. If warnings are treated as errors, the diagnostic falls back
to logging; failure of that fallback also leaves the save outcome unchanged.
A remaining stage may contain model data and may need deliberate cleanup after
inspection. Never bulk-delete similarly named files.

## Compatibility and limits

The pickle format stays the existing v1 dictionary with exactly these keys:
`model`, `features`, `metrics`, `artifact_version`, and `target`. The model remains
the bare sklearn estimator. Default pickle protocol, CSV metrics, and native
provenance contents are unchanged. No migration or new loading API is introduced.
Tests compare saved model predictions with independently trained estimators and
with [named inference](named-inference-usage.md).

Only deserialize artifacts you created or otherwise trust. Pickle can execute
code during loading; staged publication does not make an untrusted pickle safe.
The implementation never deserializes an artifact to validate it.

This is a bounded ordinary-exception contract using same-filesystem single-file
replacement. It is not a promise of crash or power-loss durability: no file or
directory `fsync` is performed. Process termination, disk corruption, unusual
network-filesystem semantics, concurrent writers/readers, concurrent changes to
source data, and adversarial path races are outside the guarantee. Coordinate
access externally; no lock or compare-and-swap generation check is added.

Coverage is in `tests/test_model_artifact_persistence.py`. The earlier native
training proposal and pre-repair evidence remain historical records.
