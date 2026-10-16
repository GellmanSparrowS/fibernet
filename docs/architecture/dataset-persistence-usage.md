# FiberNetDataset persistence and recovery

`FiberNetDataset.save(directory)` still writes the same three files:
`features.npy`, `labels.npy`, and `metadata.json`. Existing valid three-file
datasets need no migration or manifest. `load(directory)` now validates structure
and selected numerical values; see [load validation](dataset-load-validation-usage.md)
for compatibility and error details. Both methods keep their existing signatures;
`save` returns `None` on success.

```python
dataset.save("datasets/current")
loaded = FiberNetDataset.load("datasets/current")
loaded.to_csv("datasets/current.csv")
```

## Saving and overwriting

An absent, empty, checkpoint-only, complete, or partly populated destination
directory is supported. Repeated saves replace the three managed files while
leaving `checkpoint.json`, nested notes, and other unrelated files in place.
`generate(save_dir=...)` can continue saving into the directory that contains its
checkpoint. This change does not alter checkpoint generation/resume semantics.

Selected numerical values and the complete metadata JSON are prepared before the
destination is changed. Array/metadata writes happen in a private staging
directory. Existing managed files are backed up before publication; then the
three new files are individually replaced, with metadata last.

If preparation or staging fails, the previous destination remains unchanged. If
publication fails, completed replacements are rolled back. When rollback
succeeds, old managed-file bytes and ordinary POSIX permission bits are restored;
files originally absent remain absent. A destination created by this save is
removed only if empty. Missing ancestor directories created by
`out.parent.mkdir(parents=True)` during preparation may remain after a failed
fresh save; the absence promise concerns the destination itself. The original
exception is re-raised.

Unsupported metadata still raises rather than being coerced to strings or
dropped. JSON conversion rules, numerical shapes/order, legitimate zero values,
empty datasets/selections, and null/unavailable metadata retain their existing
semantics. Metadata's existing JSON NaN permissiveness is unchanged; this repair
does not add a scientific schema or infer missing-data provenance.

## CSV

`to_csv(path)` writes and closes a complete sibling temporary file, then performs
one replacement. Numerical preflight, malformed rows, quoting/formatting errors,
write errors, and a failed replacement leave the previous CSV bytes or its
previous absence unchanged. The parent directory must already exist, as before.

Column names/order, raw scalar formatting, quoted or multiline text, and CSV line
endings remain compatible with the previous writer. CSV does not use the
three-file backup/recovery descriptor.

## Supported paths and permission behavior

- The dataset destination must be a real directory. Managed targets and the CSV
  destination must be regular files when present. Symlinks, including dangling
  symlinks, and nonregular managed targets are rejected before publication
- Existing ordinary POSIX `rwx` bits are copied onto private staged files and
  private rollback files. Original files, directories, checkpoints, and unrelated
  paths are never passed to `chmod`, `chown`, or ACL-changing operations
- Managed files with setuid, setgid, or sticky bits are rejected instead of
  propagating those bits
- New managed files and new CSVs use restrictive `0600` permissions on POSIX.
  A newly created dataset directory uses ordinary directory creation/umask
  behavior. Private staging directories are created by `tempfile`
- Replacement changes the inode and may change ownership, ACLs, extended
  attributes, timestamps, or behavior through hard-link aliases. These properties
  are not preserved or guaranteed. No ownership or ACL adjustment is attempted
- Directory write permissions and enough temporary space for new files plus old
  managed-file backups are required. A filesystem that permits truncating an old
  file may still refuse directory-based replacement; that error is surfaced
- This contract assumes a supported local filesystem with same-filesystem
  replacement, exclusive access during the save, and no concurrent mutation of
  the in-memory dataset. It does not secure ancestor symlinks or path races

The mode checks are tested on POSIX. Exact POSIX mode semantics are not promised
on platforms that do not implement them. Symlink rejection and restrictive new
file modes are intentional safety changes from the previous direct-write behavior.

## If rollback also fails

A second filesystem error can prevent complete restoration. In this case `save`
raises a recovery-required `RuntimeError` with the original error as its cause.
Its message identifies the destination and retained private recovery directory,
and says not to load the destination until recovered. The exception also provides
`destination`, `recovery_directory`, `original_error`, and `rollback_errors` for
diagnostics; the implementation subtype is private.

The retained workspace contains:

- `original/`: complete copies of every managed file present before the save.
  Rollback copies these files; it never consumes them
- `recovery.json`: the absolute destination, whether its directory existed, the
  managed filenames, which files originally existed, their ordinary modes, and
  SHA-256 hashes of the original copies
- Staged or restore files left from the failed operation, useful for diagnosis

Do not treat a successful `load()` as proof that this destination is consistent.
The legacy loader can load same-shaped arrays from different generations.
Neither `load` nor a later `save` automatically scans or replays recovery folders.

For deliberate manual recovery:

1. Stop readers/writers and preserve a diagnostic copy of the failed destination
2. Verify the reported recovery path, destination, file list, and original-copy
   hashes. Keep all originals until recovery is confirmed
3. For each managed file listed as originally present, copy the retained original
   to a private restore file, apply only its recorded ordinary mode, then replace
   the corresponding destination file
4. Remove only managed files recorded as originally absent. Leave checkpoints and
   unrelated data alone. If the destination directory was originally absent,
   remove it only after it is empty
5. Verify restored file hashes and expected dataset contents before resuming use.
   Retained recovery folders may then be removed deliberately; never bulk-delete
   `.save-*` folders or recursively replace the dataset directory

If recovery still encounters I/O/permission failures, retain the evidence and
original copies. This repair cannot guarantee restoration through persistent
filesystem failure or reconstruct originals already lost by older saves.

If `KeyboardInterrupt` or `SystemExit` interrupts publication or rollback, that
same interruption propagates. The workspace, original copies, and recovery
descriptor are retained once publication has begun rather than deleted in
cleanup. No rollback is promised in that case. Inspect the private sibling
`.DESTINATION.save-*` workspace and verify its descriptor before deliberate
recovery. This narrow preservation behavior is not process-crash recovery.

## Cleanup warnings and limits

Failure to remove a private temporary workspace emits a `RuntimeWarning` with its
location. It does not turn a committed save into a reported failure or replace
the original exception from a failed save. If warning filters convert warnings
to errors, cleanup diagnostics fall back to logging while the save outcome stays
unchanged. Leftover data may require deliberate cleanup after verification.

This is an ordinary-exception/rollback contract, not a crash-atomic multi-file
transaction. There is no guarantee for process termination, power loss,
`KeyboardInterrupt`, disk corruption, network filesystems, or concurrent readers
and writers. A reader may see mixed generations during publication or rollback;
coordinate exclusive access externally. Publishing metadata last does not remove
that window. No `fsync` durability or snapshot-isolation claim is made.

Coverage lives in `tests/test_dataset_persistence.py`, alongside the existing
selected-value and reference-journey tests. The earlier proposal and disposable
failure evidence remain historical, unchanged records of the pre-fix behavior.
