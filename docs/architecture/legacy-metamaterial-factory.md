# Legacy metamaterial factory compatibility

`fibernet.api.create_metamaterial()` again resolves its default
`reentrant_honeycomb_2d`. The six historical 2D names listed below are restored
to the legacy registry and work with both `create_metamaterial(name, ...)` and
`create(name, ...)`.

```python
import fibernet.api as legacy

network = legacy.create_metamaterial(
    "reentrant_honeycomb_2d",
    array_size=(2, 1),
    grid_size=(1, 1),
    cell_height=7,
    cell_width=9,
    reentrant_angle=135,
    radius=0.3,
    weld_threshold=0.15,
)
```

These names select the original `fibernet.gen.metamaterials` implementations.
Their parameters are forwarded unchanged, without translation to the distinct
unified `metamaterial_2d(mode=...)` generator:

- `reentrant_honeycomb_2d`: `cell_height`, `cell_width`, `reentrant_angle`.
- `chiral_honeycomb_2d`: `node_radius`, `ligament_length`.
- `star_honeycomb_2d`: `cell_size`, `star_angle`, `num_arms`.
- `arrowhead_auxetic_2d`: `cell_size`, `arrow_angle`.
- `hierarchical_lattice_2d`: `cell_size`, `levels`.
- `missing_rib_auxetic_2d`: `cell_size`.

All six also accept `grid_size`, `radius` and `material`. Angles in these
historical signatures are in degrees. `grid_size` controls the internal
historical generator, while `array_size` controls the separate outer tiling.
Their defaults are retained: internal grids are `(5, 5)` except hierarchical
`(4, 4)`, and the outer array is `(3, 3)`.

## Unimplemented options are explicit

Each original implementation accepts one option it does not use. On the
restored **legacy registry routes only**, an explicitly supplied nondefault
value raises `ValueError` naming the option and its fixed historical default:

- Reentrant: `vertical_thickness=1.0`.
- Chiral: `num_node_points=6`.
- Star: `star_inner_angle=30.0`.
- Arrowhead: `arm_angle=None`.
- Hierarchical: `base_type="triangular"`.
- Missing-rib: `rib_angle=45.0`.

Omitting those options or explicitly passing the defaults preserves historical
output. This validation is deliberate: accepting a different value would
silently return geometry that does not represent it. Do not remove an option
merely to get a run to finish if your study requires that geometry. Use
`star_angle` or `arrow_angle` only when those are the parameters you intend.
Unknown keywords raise `TypeError`; there is no catch-all keyword translation.

Direct calls to `fibernet.gen.metamaterials` are unchanged, including their
historical no-op behavior. The unified registry entry `metamaterial_2d` is
also unchanged, including its own parameter contract and limitations. Its
`angle`/`cell_size` options are not interchangeable with the historical
`reentrant_angle`/`cell_height`/`cell_width` options, and its geometries differ.

## Preserved workflow and scope

`create_metamaterial` still tiles the generated network with bounding-box
spacing plus `0.1` in caller length units, clears the tiled crosslinks, then
runs `auto_crosslink` with the requested threshold. Its return type, metadata,
outer-array defaults and custom registered generator support are unchanged.
The helper does not guarantee a periodic seam, connectivity or auxetic
response. Validate the resulting network for the intended simulation.

The top-level `fibernet` facade is unchanged: import `fibernet.api` explicitly
for this legacy workflow. This correction does not redesign the unit cells,
change direct generators, modify saved networks, or publish a new package.
Regression coverage is in `tests/test_legacy_metamaterial_factory.py`.
