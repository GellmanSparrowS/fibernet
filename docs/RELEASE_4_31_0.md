# FiberNet 4.31.0

## Run 3D examples after a single install

In a fresh Python 3.10+ environment:

```bash
python -m pip install "fibernet[solid-iterative,pore-flow]==4.31.0"
python -m fibernet.examples.tet4_iterative_mechanoelectric_sweep
python -m fibernet.examples.tet4_solid_to_pore_flow --matched-force
```

The force–resistance example reports five grip displacements on a
28,512-tetrahedron solid network. Add `--refinement 5` for 132,000 tetrahedra,
or `--initial-slope` for the tangent at zero stroke. The matched-force example
compares two elastic moduli on one scaffold, using the same signed grip forces
and driving pressure gradient. Both print JSON. Source-tree example commands
still work, and each installed example needs only its corresponding extra
(`solid-iterative` or `pore-flow`). The base package supports Python 3.9+.

## Faster steady flow through periodic pores

The periodic Stokes solver uses bounded block-preconditioned MINRES when the
three velocity blocks are identical, reusing one scalar factorization across
the three components. It falls back to the existing direct sparse LU paths
when the iterative result fails checks. Mesh limits and residual, continuity,
flux and work checks are unchanged. This is steady, inertia-free flow on a
fixed geometry; the example transfers solid deformation to the pore, not fluid
loads back to the solid.

On the same reference pore and four loaded conditions, paired one-thread
wheel tests measured a median 4.66 s on 4.30.0 and 1.90 s on the new solver,
with peak memory about 204 and 136 MiB. The corresponding bulk flow and work
agreed to floating-point precision. These timings describe that calculation,
not every geometry or machine.
