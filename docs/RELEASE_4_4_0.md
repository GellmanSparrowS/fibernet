# FiberNet 4.4.0

FiberNet 4.4.0 adds bounded external-result interfaces and improves graph
generation, interchange, and analysis reliability. Python >=3.9 and core
dependencies are unchanged from 4.3.0.

## Changes since 4.3.0

- Add L3 graph preparation and saved-frame projection/render inputs. These
  interfaces do not bundle or execute an L3 solver.
- Add offline reading of saved LAMMPS preparation/output files with explicit
  manifest checks. This validates recorded inputs and output format, not engine
  provenance or physical correctness.
- Add import of a single exact welded fiber junction, opt-in NetworkX metadata
  preservation, strict XYZ reading, and explicit R10 geometry ZIP members.
- Preserve float64 coordinate precision in VTK/XYZ exports and fix VTK point
  arrays, PDB connectivity, planar DataFrame import, and masked mesh fields.
- Correct plate/gyroid factory parameter forwarding, neighboring-cell
  crosslinks, reflected random walks, graph serialization, and batch
  output/input collision checks.
- Preserve sweep/sensitivity alignment and ensemble seed provenance; add
  explicit physical-regressor validation rows; preserve optimizer categories
  and caller plotting state.

## Migration

- The corrected gyroid factory now uses its documented `grid_size=(3, 3, 3)`
  and `radius=0.1` defaults. To preserve its previous default output, explicitly
  pass `grid_size=(2, 2, 2), radius=0.15`. Direct `gyroid_lattice_3d` defaults
  do not change. This remains a TPMS-inspired fiber-network generator, not a
  solid/shell gyroid; `threshold`, factory `seed`, and `perturbation` still do
  not affect this route.
- The plate factory uses `cell_size`, not `spacing`, and now forwards
  `grid_size`, `radius`, and `material` correctly.
- Invalid/nonfinite graph/FEM inputs and masked coordinates/fields are rejected
  more explicitly.
- Previously persisted source-bound resumes or exact compiler-bound saved runs
  may require regeneration when their source identity changes. Old artifacts
  are not universally interchangeable.

This core-package release does not qualify native providers or optional extras.
The historical `Unreleased` sections in `CHANGELOG.md` include work already
shipped in 4.3.0; they are not the 4.4.0 release scope.
