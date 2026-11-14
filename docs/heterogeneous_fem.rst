Per-edge materials in linear beam FEM
=====================================

Use the existing ``StructureGraph`` to assign a different isotropic elastic
material to each member. Select ``material_mode="graph"`` explicitly; the
default ``"uniform"`` mode still uses the solver's global ``E`` and ``nu``.

Two-material series bar
-----------------------

The example uses metres, pascals and newtons. ``length_unit="m"`` is an explicit
declaration that coordinates and radii are already in metres, not a conversion.
The two unit-length members have the same radius and moduli of 1 MPa and 4 MPa.

.. code-block:: python

   import numpy as np
   import fibernet as fn

   graph = fn.StructureGraph(dimension=2)
   for x in (0.0, 1.0, 2.0):
       graph.add_node([x, 0.0])
   for i, modulus in enumerate((1e6, 4e6)):
       graph.add_edge(i, i + 1, radius=.02,
                      material=fn.Material(youngs_modulus=modulus))

   result = fn.simulate(
       graph, backend="fem", stretch_ratio=1.001,
       nonlinear=False, material_mode="graph", length_unit="m",
   )
   area = np.pi * .02**2
   extension = 2 * (1.001 - 1)
   expected_force = extension / (1 / (1e6 * area) + 1 / (4e6 * area))
   np.testing.assert_allclose(result.edge_forces, expected_force, rtol=1e-8)
   print(result.edge_forces)  # About [2.0106193, 2.0106193] N
   print(result.metadata["material_contract"])

   graph.save_json("heterogeneous_graph.json")
   result.save("heterogeneous_result.json", detailed=True)
   restored = fn.SimResult.load("heterogeneous_result.json")
   assert restored.metadata["material_contract"] == result.metadata["material_contract"]

The soft member takes four times the extension of the stiff member. The
otherwise identical default uniform solve with ``E=1e6`` carries about
1.2566371 N. The small difference from the exact bar solution is the existing
diagonal numerical regularization, not a material or interface effect.

Selection and admission
-----------------------

- ``material_mode="uniform"`` preserves existing behavior, including ignoring
  stored Material records. Constructor ``E`` and ``nu`` remain authoritative.
- ``material_mode="graph"`` requires explicit ``nonlinear=False`` and
  ``length_unit="m"``. It supports the current 2-D and 3-D linear frame
  formulations. The convenient facade rejects explicit ``E`` or ``nu`` in
  this mode rather than silently discarding an override.
- Direct ``BeamFrameFEM(E=..., nu=...).stretch_test(...)`` has the same graph
  option. Its constructor values remain available for later uniform solves
  but are unused during graph-material solves. Reusing the solver does not
  retain a previous graph's material assignment.
- Every edge must contain an ordinary ``Material`` with finite real scalar
  ``E > 0`` and ``-1 < nu < .5``. Missing materials, boolean values, custom
  material subclasses and non-None ``stiffness_tensor`` reject. Admission
  validates every original edge, including rows discarded by deduplication.
- ``G`` is derived from ``E / (2 * (1 + nu))``. A stored non-None
  ``shear_modulus`` must be finite and positive and agree at relative tolerance
  ``1e-6`` with zero absolute tolerance, allowing float32 storage roundoff.
  Inconsistent independently specified shear behavior is not silently used.
- Coordinates must be finite, each member length must be at least ``1e-12 m``
  (the existing raw solver's omission threshold), and 2-D nodes must lie in
  one XY plane. Existing straight-member and circular-radius admission applies.

Only Young's modulus, Poisson's ratio and the derived shear modulus are
consumed. Density, yield/tensile strength, thermal properties and ``extra``
remain stored data. They do not enable inertia, plasticity, damage or thermal
strain. Joints are welded according to the graph topology; changing material
across a joint does not add contact, slip, cohesive or interface behavior.

Raw arrays and identity
-----------------------

``solve_2d``, ``solve_3d`` and ``build_stiffness_2d`` accept keyword-only
``edge_materials=[...]``. The sequence must match original input edge rows,
before deduplication. This explicitly overrides constructor properties for
that call. Raw arrays remain caller-controlled consistent units with no
conversion or SI inference. For millimetre geometry and forces in newtons,
the modulus numbers must be N/mm², not Pa. Prefer converting the graph to
metres and using the graph route when material records are in Pa.

Independent parallel edges remain separate by default, including different
materials. Legacy ``deduplicate=True`` retains the first original row of each
unordered endpoint pair, its radius and its material. It is not homogenization.

Use the existing explicit ``node_ids`` and ``edge_ids`` arguments when
converting permuted raw rows with ``to_sim_result``. The result's selected
material rows follow ``edge_list`` and the same stable edge IDs. Do not index
result arrays directly with graph IDs.

Result meaning
--------------

Opt-in results contain a detached ``material_contract`` record with the model,
source, unit declaration, original selected input rows and effective ``E``,
``nu`` and ``G``. Conversion adds stable ``edge_ids`` when a graph map is
available. Both summary and detailed result JSON retain this record. It
captures solve-time properties, so later changes to graph materials or the
converter's uniform modulus do not change axial-proxy energy. This is not a
cryptographic binding to geometry: preserve the original graph, geometry,
sections and row maps when converting raw results.

``SimResult.energy`` keeps ``energy_kind="axial_stress_proxy"`` and uses
``sum(sigma_axial**2 * A * L / (2 * E_edge))``. It excludes bending and torsion.
For full linear frame energy, retain the raw rotational DOFs and stiffness
and compute ``0.5 * u.ravel() @ K @ u.ravel()``. Numerical regularization is
not included in that physical stiffness. Three-dimensional shear/resultant
recovery remains limited exactly as documented by ``beam_resultant_contract``.

Verification
------------

The heterogeneous tests cover series compliance with unequal lengths/radii,
parallel-member load partition, 2-D networks against independently integrated
beam energy, and 3-D bending/torsion with different Poisson ratios. They also
exercise original-row permutations, stable IDs, deduplication, material/solver
mutation isolation and graph/result persistence. Uniform-route regression
checks remain in :doc:`static_network_workflow`.

These are small-deformation, straight-member, welded-frame numerical checks.
They do not establish material calibration, nonlinear equilibrium, failure,
contact or mesh convergence. The existing explicit uniform viscoelastic
``history`` API is unchanged and does not accept ``material_mode``.

.. code-block:: bash

   python -m pytest tests/test_fem_heterogeneous_contract.py tests/test_fem_heterogeneous_physics.py -q
