Full strain energy in linear beam FEM
=====================================

Linear ``BeamFrameFEM.solve_2d`` and ``solve_3d`` now return a
``beam_energy`` record. Conversion retains a detached copy in
``SimResult.metadata["beam_energy"]``, including in summary JSON. This is an
additive result: ``SimResult.energy`` still holds the historical axial-stress
proxy and keeps ``energy_kind="axial_stress_proxy"``.

The record's ``strain_energy`` is the scalar quadratic form
``0.5 * u.T @ K_physical @ u`` at the solved state. The displacement vector
includes rotations, and ``K_physical`` is the assembled stiffness before
adding the numerical regularization diagonal. The total includes axial and
Euler–Bernoulli bending strain energy in 2-D, and axial, two-plane bending
and Saint-Venant torsion in 3-D. It does not add a transverse-shear deformation
law, contact, damage, kinetic energy, or interface energy to the existing
straight, circular-section, welded-joint beam model. Selected parallel edges
and the existing optional deduplication follow the assembled stiffness.

Read the total and keep the proxy
---------------------------------

.. code-block:: python

   import fibernet as fn

   graph = fn.StructureGraph(dimension=2)
   for position in ([0, 0, 0], [1, 0.4, 0], [2, 0, 0]):
       graph.add_node(position)
   for i, modulus in enumerate((1e6, 4e6)):
       graph.add_edge(i, i + 1, radius=0.02,
                      material=fn.Material(youngs_modulus=modulus))
   result = fn.simulate(
       graph, backend="fem", stretch_ratio=1.001, nonlinear=False,
       material_mode="graph", length_unit="m",
   )
   record = result.metadata["beam_energy"]
   if record["status"] != "available":
       raise RuntimeError(record["reason"])
   assert record["unit_system"] == "SI"
   print("Full linear beam strain energy, J:", record["strain_energy"])
   print("Legacy axial proxy, J:", result.energy)
   result.save("beam_energy_result.json", detailed=True)
   restored = fn.SimResult.load("beam_energy_result.json")
   assert restored.metadata["beam_energy"] == record

Only the explicit graph-material route declares SI: geometry and radii in
metres, moduli in pascals, energy in joules. Raw array solves and the default
uniform route use ``unit_system="not_declared"``. They do not infer or
convert units. For consistent newtons and millimetres the value is N·mm,
not joules; divide by 1000 to convert N·mm to joules. The record's
``included_terms`` describes the model, not a component-wise energy breakdown.

Regularization, work, and validity
----------------------------------

The numerical diagonal is excluded from stored beam energy. It still affects
the solved displacement. With applied load vector ``f``, physical residual
``r = K_physical @ u - f``, prescribed DOFs ``c``, free DOFs ``q`` and the
current diagonal coefficient ``d``, the linear solve satisfies, up to numerical
error,

``U = 0.5*u.T@f + 0.5*u[c].T@r[c] - 0.5*d*u[q].T@u[q]``.

The final term is an algebraic correction for the numerical solve, not energy
of a physical foundation. Do not identify the total with half the applied
load or grip work without accounting for constraints and residuals. The mixed
translation/rotation diagonal remains coordinate- and unit-dependent.

``status="available"`` means the computed quadratic is finite and nonnegative.
It does not certify equilibrium, stability, uniqueness, conditioning, accuracy,
or applicability of small-deformation physics. A nonfinite or negative result
is marked ``unavailable`` with ``strain_energy=None`` and a specific reason.
No clipping is performed. Rigid-body motions can produce a tiny negative
quadratic through cancellation and are then unavailable too; this is numerical
roundoff, not a claim of negative physical energy. Use independently checked
kinematics and residuals when interpreting near-zero energies.

The experimental updated-geometry nonlinear path marks final energy unavailable
with ``final_nonlinear_energy_not_recovered``. It does not expose the energy of
its last inner linear increment as total nonlinear energy. Viscoelastic
history, sparse FEM, and other backends are unchanged.

Persistence and provenance
--------------------------

``beam_energy`` is a producer snapshot. Conversion validates its version,
dimension, model, term labels, finite-value/status consistency and allowed
units, then deep-copies it. It neither recomputes from a later graph nor guesses
an energy for a legacy dictionary without the record. Later edits to raw
displacements, stiffness, materials or geometry do not update the snapshot.
Conversion validation is not cryptographic provenance or a consistency check
against arbitrarily edited raw arrays. Saved JSON uses the existing flat
metadata namespace; loading retains the record without rerunning the solve or
this producer-record admission. Summary exports retain the scalar but are not
restart checkpoints.

Verification
------------

``tests/test_fem_energy_physics.py`` checks closed-form axial, bending and
torsional cases, mixed and rotated 3-D loading, unit rescaling, rigid motions,
and heterogeneous networks against independently integrated Hermite/Gauss
stiffness and physical work balances. ``tests/test_fem_energy_contract.py``
checks conversion, copying, nonlinear unavailability and JSON persistence.
These bounded numerical checks do not establish experimental calibration or
general mesh convergence.
