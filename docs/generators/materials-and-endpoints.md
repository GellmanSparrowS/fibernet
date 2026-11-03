# Generator materials and endpoint crosslinks

The following `fibernet.gen` generators interpret an omitted or `None` material
as a fresh generic `Material` for each generated fiber:

- `lattice_2d` and `metamaterial_2d`;
- `lattice_3d` through its cubic, octet and diamond construction paths;
- `curved_random_2d`, `entangled_3d`, and `hierarchical_lattice`.

The same behavior applies through `fibernet.api.create`. Explicitly supplied
material objects retain their identity: fibers share the caller's object rather
than receiving copies. Changing that shared object's properties therefore
affects all fibers referencing it. Default objects, including their `extra`
dictionaries, are independent. This does not change standalone
`Fiber(material=None)` semantics or every other generator's material policy.

## 2D shared-node connections

The shared edge-to-network conversion used by `lattice_2d` and
`metamaterial_2d` maps crosslinks to the fibers that remain after short-edge
filtering. Dropped edges do not leave references to nonexistent fibers.

Crosslink parameters identify the actual endpoint: `0.0` is the start and
`1.0` is the end. Reversing an edge reverses its corresponding parameter. The
crosslink position agrees with the referenced endpoint on each incident fiber.

**Serialization compatibility:** newly generated networks now store these
orientation-correct `0.0`/`1.0` values instead of the former `0.5` placeholders.
Geometry construction algorithms and the short-edge filtering threshold are
unchanged. Existing saved records are not automatically rewritten or migrated.
Consumers should interpret the stored parameters, not assume every crosslink
is at a fiber midpoint. Native network loading retains its existing fiber-ID
renumbering behavior; this change does not promise stable IDs across loading.
