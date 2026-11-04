"""Count-only storage accounting; no graph, force law, provider or execution."""


def payload_counts(n, e, frames, thermo_frames, output_frames, *, row_bytes,
                   prepared_bytes, setup_bytes, metadata_reserve_bytes,
                   raw_reserve_bytes, export_reserve_bytes, setup_copies=1,
                   type_table_bytes=0, record_reference=True):
    """Reuse existing conservative arithmetic without importing profile policy.

    All returned sizes are estimates except caller-supplied exact byte counts.
    This models Python copies, not measured RSS or a general memory bound.
    """
    values = (n, e, frames, thermo_frames, output_frames, row_bytes,
              prepared_bytes, setup_bytes, metadata_reserve_bytes,
              raw_reserve_bytes, export_reserve_bytes, setup_copies,
              type_table_bytes)
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError('forecast counts must be nonnegative integers')
    if min(frames, thermo_frames, output_frames, row_bytes, setup_copies) < 1:
        raise ValueError('forecast frame, row and setup counts must be positive')
    if type(record_reference) is not bool:
        raise ValueError('record_reference must be boolean')
    dump = frames*(row_bytes*n+1024)+32
    thermo = thermo_frames*256
    reserve = 1024*1024
    slots = 12*n*output_frames+16*n+2*e+16*output_frames+64
    # Three original-SI coordinates plus a compiler-bounded (<2**31) stable ID
    # per node. Fixed schema/source digests and container overhead get 2 KiB.
    # Retain the existing conservative four simultaneous selected-copy model.
    reference_bytes = 4*n*64+2048 if record_reference else 0
    export = 64*slots+metadata_reserve_bytes+reserve+reference_bytes
    raw_parse = 4*raw_reserve_bytes
    parsed = frames*(9*n+16)*64
    selected = 4*(slots*64+reference_bytes)
    serialization = 3*export_reserve_bytes
    return dict(trajectory_bytes=dump, thermo_bytes=thermo,
                prepared_artifact_bytes=setup_bytes,
                diagnostic_reserve_bytes=reserve,
                native_aggregate_forecast_bytes=dump+2*thermo+setup_copies*setup_bytes+reserve,
                detailed_export_forecast_bytes=export,
                raw_parse_memory_bytes=raw_parse,
                parsed_state_memory_bytes=parsed,
                selected_copy_memory_bytes=selected,
                serialization_memory_bytes=serialization,
                planned_memory_bytes=raw_parse+parsed+selected+serialization+4*prepared_bytes+type_table_bytes)
