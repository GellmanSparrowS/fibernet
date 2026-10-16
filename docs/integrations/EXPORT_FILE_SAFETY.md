# Solid and route file replacement

`export_solid` (STL/3MF) and `export_route` (CSV) stage output in an exclusively created, uniquely named sibling file, then replace the requested destination with the existing `atomic_replace` hook. A pre-existing `destination.tmp` is unrelated and is never consumed or cleaned up. Supported format/input checks occur before staging. Parent directories must already exist, as before.

For an existing ordinary regular file, ordinary permission bits are copied onto the staging file before replacement. A newly created destination is private (0600). Symbolic links, nonregular destinations and special mode bits are rejected before writing. The exporter does not chmod/chown the original target or alter its ACL; replacement does not promise to reproduce ownership, ACLs, extended attributes or timestamps on the new file.

Ordinary validation, writing or replace failures preserve the old destination bytes (or its absence). Cleanup only addresses the temporary file created by that invocation. Cleanup failure is warned about without replacing the original exception or converting an already committed export into a failure; a retained private temporary may need manual removal. Warning filters do not change this outcome contract.

There is no concurrent-writer, malicious-directory-race or power-loss durability guarantee. Atomic replacement is one-file publication, not a multi-file STL/3MF/CSV transaction.

Geometry, coordinates, units, mesh/route algorithms and formats are unchanged. Tiny tetrahedron/route regression outputs match original STL and CSV bytes, and every complete 3MF ZIP member's bytes (ZIP timestamps are not compared). This persistence check is not a new manufacturing qualification.
