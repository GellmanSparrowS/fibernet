# SimResult JSON save ownership

`SimResult.save` preserves its JSON field layout, indentation, NumPy conversion,
and recursive parent-directory creation. It stages through an exclusively created,
random sibling file and atomically replaces the destination after serialization
and close succeed. A pre-existing `<destination>.tmp` is never used or removed.

Existing regular-file ordinary permission bits are retained. New files use 0600
(private), rather than the former umask-dependent open default. Symlinks (including
dangling links), directories, FIFOs and other nonregular targets are rejected, as
are targets carrying setuid, setgid or sticky mode bits. These checks occur before
staging. This does not preserve inode identity, hardlink aliases, ownership, ACLs,
extended attributes or timestamps.

Serialization, write, permission-setting or replacement failure leaves the old
result intact (or the new destination absent). Cleanup only attempts this call's
owned temporary. Cleanup errors emit a best-effort RuntimeWarning and cannot
replace the original failure or turn a committed save into an exception, even
with warning-as-error filters; an owned temporary may remain after cleanup fails.

This is same-directory atomic replacement, not fsync-backed crash durability,
concurrent-writer locking or protection from hostile directory mutation. Concurrent
successful writers have last-replacement-wins behavior. Parent directories must be
trusted; destination validation is a preflight check, not a race-proof filesystem
security boundary.
