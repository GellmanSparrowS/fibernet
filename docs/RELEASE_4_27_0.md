# FiberNet 4.27.0

## Larger elastic solids and proportional-stroke sweeps

Install the optional solver:

~~~sh
python -m pip install "fibernet[solid-iterative]==4.27.0"
~~~

Use `solver="amg-cg"` with `solve_static_tet4` for a single face-connected
elastic solid. The direct banded solver is still the default. Mesh units,
stable IDs, input loads and output fields are unchanged. See the
[standalone mechanics solver](TET4_STATIC.md) for numerical limits.

From the source distribution's root or a source checkout, run:

~~~sh
python -m examples.tet4_iterative_mechanoelectric_sweep
~~~

The example varies clamped stroke on a fused lattice. For fixed, linear
elasticity with zero preload and unchanged grips and proportional loading,
one elastic solve gives the displacement field for all five strokes. Each
resulting geometry still gets a separate DC solve. For this example the
electrical solution does not affect the mechanics.

## Less time checking and solving meshes

The existing iterative DC solver evaluates cell geometry in bounded
batches. Tet4 mesh validation certifies conventional nonzero determinants
with outward-rounded intervals and applies the exact binary64 test when
the interval cannot exclude zero or the coordinate scale is unsuitable.
The geometry decision and solver conditioning checks are unchanged.

The optional mechanics method, new example, and electrical acceleration do
not change the default banded solver or the limits of coupled interfaces.
