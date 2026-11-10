Stored materials, static FEM, analysis and export
=================================================

A graph can contain several ``Material`` records without its selected solver
using several constitutive laws. The FEM stretch API uses one explicit ``E``
and ``nu`` for every member by default. Stored edge moduli are preserved by native graph
JSON but do not override those parameters. Changing a stored material alone
does not change this solver's displacement or force result.

This page demonstrates that default contract, not heterogeneous mechanics.
For the explicit per-edge-material option, see :doc:`heterogeneous_fem`.
It needs the core dependencies only. Native VTK-reader verification is an
optional test; VTK is not needed to write the file below.

Reproducible small network
--------------------------

Run in a directory without an existing ``static_network_output`` subdirectory.
The values below use metres, pascals and newtons consistently. The static
solver does not infer units from the graph or convert them.

.. code-block:: python

   from pathlib import Path
   import numpy as np
   import fibernet as fn
   from fibernet.analysis import analyze_tensile_recruitment
   from fibernet.ml.beam_frame_fem import BeamFrameFEM
   from fibernet.io.vtk import to_vtk

   output = Path("static_network_output")
   output.mkdir()  # Refuse an existing destination for this example.
   graph = fn.pattern_2d(unit="honeycomb", grid=(2, 2), box=(1, 1),
                         radius=.02, seed=23)
   for i, edge in enumerate(graph.edges.values()):
       edge.material = fn.Material(name="soft" if i % 2 else "stiff",
                                   youngs_modulus=1e6 if i % 2 else 4e6)
   graph.save_json(output / "graph.json")
   graph = fn.StructureGraph.load_json(output / "graph.json")

   uniform_E = 1e6
   result = fn.simulate(graph, backend="fem", stretch_ratio=1.001,
                        nonlinear=False, E=uniform_E, nu=.3, pct=.1)
   result.save(output / "result.json", detailed=True)
   result = fn.SimResult.load(output / "result.json")

   # Stable IDs identify solver rows. Never use IDs as array indices.
   node_rows = {node_id: row for row, node_id in enumerate(result.metadata["node_ids"])}
   edge_rows = {edge_id: row for row, edge_id in enumerate(result.metadata["edge_ids"])}
   edges = np.array([[node_rows[graph.edges[eid].node_i],
                      node_rows[graph.edges[eid].node_j]]
                     for eid in result.metadata["edge_ids"]])
   grips = BeamFrameFEM.graph_to_fem_input(graph, dim=2, pct=.1)["boundaries"]

   # This is the solver's linear axial strain, not finite chord-length strain.
   axial_strain = np.asarray(result.metadata["sigma_axial"]) / uniform_E
   recruitment = analyze_tensile_recruitment(
       np.stack([np.zeros_like(axial_strain), axial_strain]), edges,
       grips["left"], grips["right"], graph.num_nodes,
       alpha=.05, min_active=1e-7,
   )

   # Legacy VTK has sampled per-fiber points and line segments, not graph rows.
   network = graph.to_fiber_network()
   displacement, force, stored_modulus, active = [], [], [], []
   for fiber, (eid, edge) in zip(network.fibers, graph.edges.items()):
       count = len(fiber.centerline)
       displacement.extend(np.linspace(result.displacements[node_rows[edge.node_i]],
                                       result.displacements[node_rows[edge.node_j]], count))
       force.extend([result.edge_forces[edge_rows[eid]]] * (count - 1))
       stored_modulus.extend([edge.material.youngs_modulus] * (count - 1))
       active.extend([recruitment.active_edges[-1, edge_rows[eid]]] * (count - 1))
   to_vtk(network, output / "fields.vtk",
          point_data={"endpoint_interpolated_displacement_m": displacement},
          cell_data={"signed_axial_force_N": force,
                     "stored_material_E_Pa": stored_modulus,
                     "used_uniform_E_Pa": np.full(len(force), uniform_E),
                     "threshold_active": active})


The straight, nondegenerate generated edges in this example are retained in
edge iteration order by the legacy conversion. This is not a promise for
arbitrary curved or degenerate input. Coincident crosslink coordinates remain
duplicated in VTK. The endpoint-interpolated field is a visualization convention,
not recovered Hermite beam-interior displacement. Keep graph and detailed result
JSON as the authoritative records. See `legacy VTK precision and layout <https://github.com/GellmanSparrowS/fibernet/blob/main/docs/io/legacy_vtk_attributes.md>`_.

The recruitment threshold is ``max(.05 * maximum_positive_strain, 1e-7)`` here.
The small floor is an explicit illustration parameter, not a yield strain or
material calibration. A spanning recruited component is a positive-strain
connectivity statistic, not a full load path: bending, compression and shear
can carry force outside it.

Numerical meaning and independent checks
----------------------------------------

``tests/test_static_network_workflow.py`` runs actual solves on 18-node/22-edge
and 60-node/84-edge honeycombs with two stored material records, alternating
radii, reordered insertion and noncontiguous IDs. It checks:

- Graph data against the solver's geometry, radius, connectivity and grip rows.
- An independently integrated Euler–Bernoulli energy Hessian. Two-point Gauss
  integration exactly integrates squared linear Hermite curvature. Closed-form
  axial, constant-curvature and rigid-motion checks also test this oracle.
- Dense displacement and reaction solutions, axial force, axial energy,
  positive constrained stiffness, and comparison with zero regularization.
- Full exercised graph and detailed-result JSON records, including identity,
  stresses, reactions, resultants and numerical diagnostics.
- Recruitment from reloaded linear strain against independent NetworkX
  component analysis, then VTK coordinates, line connectivity and field values.
- Optional actual ``vtkPolyDataReader`` readback, isolated from the core tests.

The linear solve adds a diagonal numerical coefficient of ``1e-6`` by default.
Physical free-DOF residuals therefore satisfy ``K u = -1e-6 u`` on free DOFs in
these zero-external-load cases. This is not a physical foundation. The tests
check that relation and account for it in the prescribed-work identity. The
reported stiffness conditioning is specific to the chosen coordinate and
mixed translation/rotation scaling, not a unit-invariant physical measure.

``SimResult.energy`` is labeled ``axial_stress_proxy``; it excludes bending. In
these examples, bending accounts for about 96–99.5% of full beam energy.
The additive ``result.metadata["beam_energy"]`` record retains the full
linear energy ``0.5 * u.ravel() @ K @ u.ravel()`` using physical stiffness.
Raw ``u`` includes rotations, while the canonical ``SimResult.displacements``
contains translations only. See :doc:`linear_beam_energy` for availability,
unit declarations and the numerical-regularization work correction. Do not
interpret the axial proxy as total work or compare it to total grip work.

A two-member series bar provides a separate material-semantics check. With
unit member lengths, radius ``.02``, stored moduli ``1e6`` and ``4e6``, uniform
solver ``E=1e6`` and total extension ``.002``, the solver approaches ``1.256637 N``.
A genuinely heterogeneous bar carries ``2.010619 N`` through the explicit
graph-material mode. The mismatch is expected for this default uniform route
and must not be described as heterogeneous validation.

These are bounded numerical and integration checks for straight, welded,
small-deformation 2D frames. They do not validate large deformation, material
calibration, heterogeneous constitutive response, failure, contact, transient
response, all loading histories, or general mesh convergence. The two network
sizes are different topologies, not a mesh-refinement study. The recruitment
check covers a zero-to-loaded pair, not hysteresis or depth over a trajectory.

Run the focused checks from a checkout:

.. code-block:: bash

   python -m pytest tests/test_static_network_workflow.py -q
