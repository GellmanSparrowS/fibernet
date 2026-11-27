# FiberNet 4.40.0

## Compare two saved velocity fields without rerunning Stokes

A change in total flow need not tell you whether velocity has redistributed
across the inlet or outlet. Compare two native quadratic VTU files to measure
both on the same physical cap.

```sh
python -m pip install "fibernet==4.40.0"
```

```python
from fibernet.analysis.stokes_cap_comparison import compare_saved_stokes_caps

report = compare_saved_stokes_caps("reference/velocity.vtu", "loaded/velocity.vtu")
outlet = report["x_max"]
print(outlet["throughput_ratio"], outlet["normalized_profile_residual_L2"])
```

The first value compares signed flow. The second measures the normal-velocity
change after removing one uniform speed ratio. Both files must have matching
cap planes, physical triangles and edge midpoints, with resolved flow in the
same direction on each cap. The previous one-file cap flux audit is unchanged.
See the [saved-field comparison guide](TET4_OPEN_STOKES.md#compare-throughput-and-velocity-redistribution-on-the-same-caps).

Cap extraction uses batched vertex and edge lookups. On two saved field pairs
with 12,343 and 19,875 tetrahedra, the median local from-file comparison took
about 26–28% less time and returned the same report values. This measurement is
for post-processing, not the Stokes solve.
