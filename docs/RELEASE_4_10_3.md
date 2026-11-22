# FiberNet 4.10.3

## Use the original metamaterial factories by their historical names

`fibernet.api.create_metamaterial()` again defaults to the original
re-entrant honeycomb. The helper accepts its six documented historical names:
`reentrant_honeycomb_2d`, `chiral_honeycomb_2d`, `star_honeycomb_2d`,
`arrowhead_auxetic_2d`, `hierarchical_lattice_2d`, and
`missing_rib_auxetic_2d`. Each uses its original factory's parameters
and geometry, then applies the existing outer tiling and weld detection.

```python
from fibernet import api

network = api.create_metamaterial(
    array_size=(2, 1), grid_size=(1, 1),
    cell_height=7, cell_width=9, reentrant_angle=135,
)
```

The historical names are local to this helper. Already registered generators
still take precedence; the shared registry, the separate `metamaterial_2d`
geometry, and direct `fibernet.gen` calls do not change. On a historical
fallback path, the helper rejects nondefault values for the original
factory's unused option rather than silently producing the default geometry.
It does not certify mechanical behavior or periodic seams.
