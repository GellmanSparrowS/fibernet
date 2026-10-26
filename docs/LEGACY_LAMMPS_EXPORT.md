# Legacy FiberNetwork LAMMPS export is unsupported

`fibernet.io.to_lammps` now raises `NotImplementedError` immediately, without
resampling, computing parameters or opening a destination. The same applies to
`fibernet.api.export(..., format='lammps')` and inferred `.data`/`.lammps` output.
Existing files are left untouched. Changing atom style, bond style or units does
not bypass this boundary; even an unlinked straight chain is not supported.

This is a deliberate compatibility break to prevent misleading simulation input:

- The old default bond-style atom records omitted the molecule ID required by
  [LAMMPS read_data](https://docs.lammps.org/read_data.html); its coefficient
  section header was a comment, not an active `Bond Coeffs` section.
- The same two coefficients were emitted for every advertised bond style.
  [Harmonic](https://docs.lammps.org/bond_harmonic.html),
  [FENE](https://docs.lammps.org/bond_fene.html) and
  [Morse](https://docs.lammps.org/bond_morse.html) have different definitions and
  coefficient requirements. The implementation did not establish those mappings.
- The `units` argument did not convert or otherwise govern the output.
  [LAMMPS units](https://docs.lammps.org/units.html) affect geometry, masses and
  coefficients. Bead spacing was substituted for actual bond equilibrium lengths,
  material-name grouping could hide section differences, and declared crosslinks
  were omitted. No replacement calibration or nearest-bead connectivity is invented.

## Preserve data or prepare a model explicitly

Use `network.save_json(path)` or `api.export(..., format='json')` to retain native
FiberNetwork centerlines and declared crosslinks. JSON storage is not a solver
model and does not establish physical equivalence.

The separate StructureGraph LAMMPS backend remains available unchanged. Prepare
a graph with explicit topology, `length_unit`, `masses_kg`, bond parameters and
runtime/resource contracts (see `prepare_harmonic_beads` in
`fibernet.integrations.lammps_backend`, or `simulate(..., backend='lammps',
mode='dynamics', ...)`). This requires an intentional model; it is not an
automatic FiberNetwork migration or equivalent replacement.

The existing `from_lammps` reader is unchanged and remains importable; this patch
does not qualify its format coverage or round-trip fidelity. The separately
unsupported `export_to_lammps_fea` template and modern backend are not rerouted.
No LAMMPS executable, solver or physics model is introduced by this change.
