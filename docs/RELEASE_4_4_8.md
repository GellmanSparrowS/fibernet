# FiberNet 4.4.8

This patch fixes precision loss in user-supplied point and cell fields
exported through `fibernet.io.meshio_adapter.to_meshio` and `fibernet.io.meshio_adapter.write_vtu_bundle`.
It affects both row-ordered field sequences and fields provided by ID.

For example, a custom label `9223372036854775809` supplied as a NumPy
unsigned integer could be rounded to `9223372036854775808` when paired with
a small signed integer in the same field. FiberNet now retains integer
values in a common signed or unsigned 64-bit type when possible. Mixed
integer and floating-point data that would round an integer raise before
a bundle is written. This does not change geometry, field alignment,
stable graph identifiers, materials, or explicitly typed arrays.

## Migration

If earlier VTU files carry large numeric labels or counters, re-export from
the original, exact values. Already-rounded floating-point arrays cannot be
recovered by the exporter. Keep integers in a separate field or use an
explicit floating-point array if approximation is intentional.

## Install

```bash
python -m pip install "fibernet==4.4.8"
python -m fibernet.examples.basic_usage --output-dir fibernet_demo
```

Public function signatures and package requirements are unchanged.
