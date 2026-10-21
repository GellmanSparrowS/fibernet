# Legacy LAMMPS compatibility

The legacy `LAMMPSBridge.run_simulation` and `run_lammps_md` APIs now raise provider, execution, extraction and engine-close errors instead of returning an empty result after a warning. Function arguments and generated data/input text are unchanged. Constructing a bridge without the optional provider still permits export; running it raises the original import error before creating working files or a temporary directory.

Callers that previously treated an empty result as a failed run should catch exceptions around the simulation call and report or handle the failure. An exception does not imply that working files were removed. Every successfully constructed engine receives one close attempt, including on failure or a catchable interruption. If both simulation work and closing fail, the simulation error remains primary; the close error is available through its `lammps_close_error` attribute when the exception supports it. A close-only failure also raises.

`LAMMPSResult.to_dict()` returns `None` (JSON `null`) for unavailable final energy or temperature. Explicit measured zero remains zero. Consumers must handle unavailable measurements rather than assume numeric values. The legacy execution path does not collect energy or temperature, so those measurements remain unavailable after an otherwise successful run.

Engine closing does not delete working directories or generated files. Automatically created temporary directories and caller-supplied workdirs remain after success or failure.

The legacy bridge runs in-process without resource bounds. It does not use the separate bounded harmonic backend, and this change does not alter force fields, timesteps or simulation capability.

Known limitations remain: input and provider-output validation does not comprehensively check coordinate count, shape or finiteness; legacy bond-style atom rows omit the molecule-ID column; and the general LAMMPS exporter comments out its Bond Coeffs header, does not convert its stated units, and uses EA/L without the harmonic half-factor. Successful execution or engine closing is not evidence that these data and model limitations have been resolved.
