# Converting legacy FiberNetwork units

`fibernet.utils.units.convert_network_units(network, from_system, to_system)`
returns an independent converted `FiberNetwork`. The input is unchanged, including
when conversion raises an exception. Shared materials remain shared within the
returned network and are converted once.

```python
from fibernet.utils.units import convert_network_units

network_cgs = convert_network_units(network_si, "SI", "CGS")
```

Pass the actual source units explicitly. The function does not infer units from
coordinates, material values, or metadata. Supported system names are `SI`, `CGS`,
`MICRO`, `NANO`, and `MOLECULAR`; names are case-insensitive. Unknown systems and
physical quantities raise `ValueError`.

## Converted fields

- Fiber centerlines and radii; periodic box dimensions.
- Rectangular width/height, elliptical semi-minor axis, and hollow-circle inner
  radius. Circular sections require no additional parameters.
- Crosslink positions, force strength, and translational stiffness. Positive
  infinity strength/stiffness sentinels are preserved.
- Material density, Young's/shear moduli, yield/tensile strengths, fracture
  toughness in stress times square-root length, and stiffness tensors.
- Thermal conductivity, specific heat, and thermal expansion.

Fiber IDs, crosslink endpoint indices and dimensionless parameters, Poisson's
ratio, relative permittivity, relative permeability, and descriptive labels are
preserved. Crosslink positions and fiber coordinates use the same length factor.

Network metadata is copied, with `unit_system` set to the canonical target name.
Other metadata is descriptive: this function does not convert loads, results or
physical quantities embedded in arbitrary metadata. Keep such quantities in an
explicitly dimensioned workflow and convert them separately.

## Input contract and compatibility

This helper accepts the base `FiberNetwork`, `Fiber`, `Crosslink` and `Material`
classes. Subclasses and additional object attributes are rejected because they
may introduce fields with undeclared dimensions. Every fiber must have an
explicit `Material`; `None` does not select a default during conversion.

Electrical conductivity is rejected, including a supplied zero value: these
unit systems have no electric-current base unit. Numeric values in
`Material.extra` require a declared dimensional contract and are rejected.
String, Boolean, null and nested list/dictionary labels are preserved. Custom or
triangular cross sections and unknown section parameter keys are rejected.

Unit factors must be real, finite and positive; unsupported numerical ranges
raise `ValueError`. Nonfinite converted data is rejected, except for the positive
infinity crosslink sentinels above. Conversion does not certify constitutive
validity, graph connectivity, solver compatibility or physical calibration.

Older versions silently treated unknown systems as SI, returned unchanged values
for unknown quantities, and converted only part of a network's dimensional data.
Unknown systems and quantities now fail explicitly; supported dimensional fields
are converted together. Existing import paths and the copy-returning behavior are
retained.

For independent scalar or array values, `UnitConverter.convert`, `to_si` and
`from_si` support length, mass, time, force, energy, stress, velocity and viscosity.
For explicit unit names such as `mm` and `MPa`, use the separate
`fibernet.units` scalar conversion helpers.
