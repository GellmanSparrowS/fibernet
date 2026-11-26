# FiberNet 4.30.0

## Faster stationary pore flow and iterative steady DC

- The periodic Stokes solver can use a structure-aware mode of the same
  general sparse LU factorization without changing its pivot threshold.
  It falls back to its original LU settings if factorization or a numerical
  check rejects the first result. The flow, continuity and work checks are
  unchanged. On the four matched-force cases plus one reference of the
  same example, paired one-thread runs measured a median 12.78 s before
  and 4.78 s after for the complete script, with comparable flow and power.
- Iterative Tet4 DC conservatively certifies well-conditioned cells using
  the inverse residual, retaining the previous SVD/scalar checks for
  uncertified or borderline cells. On five matched deformation states
  of one 132,000-tet network, paired one-thread runs measured a median
  4.99 s before and 3.82 s after for the electrical phase, with identical
  compared field arrays and scalars.
- A regression on a heterogeneous connected solid checks the physical
  scaling of current, power and resistance at fixed voltage, and checks
  that the same geometry encoded in metres or millimetres gives the same
  SI fields.

The speed comparisons describe those workloads; actual timing and memory
depend on mesh, provider and machine. Model assumptions and API are unchanged.

## Run the matched-force and resistance examples

The examples are in the matching source distribution. A new Python 3.12
environment works for both; the base package supports Python 3.9+, while
the pore-flow extra requires Python 3.10+.

```sh
python -m pip install "fibernet[solid-iterative,pore-flow]==4.30.0"
python -m pip download --no-deps --no-binary=fibernet "fibernet==4.30.0"
python -m tarfile -e fibernet-4.30.0.tar.gz
cd fibernet-4.30.0
python -m examples.tet4_solid_to_pore_flow --matched-force
python -m examples.tet4_iterative_mechanoelectric_sweep --refinement 5
```

Use the 28,512-cell default first for the force–resistance sweep; add
`--refinement 5` for 132,000 tetrahedra. See the
[3D workflow guide](RELEASE_4_29_0.md) for mesh selection and Gmsh/VTU setup.
