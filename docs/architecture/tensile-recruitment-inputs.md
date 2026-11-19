# Observed inputs for tensile recruitment

`fibernet.analysis.analyze_tensile_recruitment` accepts a finite strain history
with shape `(frames, edges)`, endpoint pairs in matching column order, and
left/right grip node indices. Endpoint and grip indices refer to the rows of
the corresponding node array.

Strains, endpoints and grips must be fully observed. Plain arrays and NumPy
masked arrays with all-false masks produce the same results. If any entry is
masked, the function identifies the affected input before converting it or
computing recruitment. This includes masks nested inside array-like inputs.
Handle missing observations explicitly before analysis; the function never
uses their underlying stored values as measurements.

The existing threshold, hysteresis, grip-connected pruning and spanning
statistics are unchanged. The legacy `compute_percolation` entry point uses
the same observed-input checks.
