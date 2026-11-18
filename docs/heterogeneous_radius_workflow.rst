Per-member radii in 3D analysis
===============================

``GraphFeatureExtractor3D`` uses each represented edge's own radius and
endpoint-chord length for its cylinder estimates. This matters even for native
generators: ``pattern_3d(unit="octet", n_pts_per_side=0)`` contains frame
members at the requested radius and body diagonals at 0.7 times that radius.

For radii ``r_e`` and chord lengths ``L_e``, the additive volume is
``V = sum(pi * r_e**2 * L_e)`` and lateral area is
``A = sum(2 * pi * r_e * L_e)``. Averaging radii before these sums gives the
wrong volume for unequal radii and loses the radius-length association.

The five affected features retain their names, ordering and conventions:

* ``volume_fraction`` is ``min(V / max(B, 1e-12), 1)``, where ``B`` is the
  volume of the node-coordinate bounding box, not ``StructureGraph.box_size``.
* ``surface_area_approx`` is ``A``.
* ``specific_surface`` is ``A / max(V, 1e-12)``.
* ``sphericity`` compares the area of a sphere of volume ``V`` with
  ``max(A, 1e-12)``.
* ``compactness`` is ``V / max(A**1.5, 1e-12)``.

These are independent circular-cylinder estimates. Overlapping junctions are
counted repeatedly; end caps are excluded from the area. They do not measure
union-solid volume or exposed manufacturing surface. The historical volume
cap and denominator floors remain, so planar or extremely small geometries
retain those conventions. ``StructureGraph`` curves are still represented by
their endpoint chords. No units are inferred or converted; all lengths and
radii must use the same unit. With metre inputs, area is in square metres and
specific surface is in inverse metres.

The 60-feature interface and simple-undirected-graph restriction are unchanged.
For NetworkX graphs, a missing edge radius still defaults to 0.05.
The legacy ``FiberNetwork`` feature conversion still uses default-radius
centerline segments, rather than its stored fiber radii. Use the native
``StructureGraph`` directly for this workflow. This correction concerns the
five cylinder-based features; it does not redefine or qualify the other
historical features. Regenerate affected heterogeneous-radius features and
derived datasets before comparing them with newly calculated values.

Runnable workflow
-----------------

From the repository checkout, using the ordinary FiberNet dependencies:

.. code-block:: bash

   python -m examples.heterogeneous_radius_workflow --output-dir octet-radius-run

The output directory must not exist. The example:

1. Generates a 35-node, 118-member octet from a ``2 x 2 x 2`` grid. The per-cell
   dimensions are 10, 15 and 20 mm, and the two radii are 0.5 and 0.35 mm.
2. Saves and reloads the native graph, then applies a small 1.002 stretch ratio
   with linear 3D FEM, uniform ``E=1e6 Pa``, ``nu=0.3`` and the existing
   ``pct=0.1`` row-based grip selector. Stored material records are not selected.
   Straight beams share welded nodes. Numerical regularization is unchanged.
3. Saves and reloads detailed results and verifies exact displacement, deformed
   position, axial-force and stretch arrays. Stable ID maps associate rows with
   nodes and members.
4. Computes thresholded positive-chord-strain recruitment and the five
   reference-geometry cylinder features. The latter are independently checked
   against per-member sums. The analytical cylinder volume is not deformed
   volume or beam strain energy.
5. Writes ``graph.json``, ``result.json``, ``analysis.json``,
   ``cylinder_features.csv`` and ``member_fields.vtk``.

The VTK file carries source edge IDs, radii, signed axial forces, a recruitment
indicator and explicitly named endpoint-interpolated displacement. It is a
centerline visualization, not a beam-interior displacement reconstruction or a
welded mesh. Scalar/vector fields use the existing legacy float32 text
representation. See the `legacy VTK attribute guide
<https://github.com/GellmanSparrowS/fibernet/blob/main/docs/io/legacy_vtk_attributes.md>`_
for ordering and precision.

The example tests software continuity, identity and analytical cylinder sums.
It is not evidence of physical calibration, mesh convergence, nonlinear
equilibrium or solid-manufacturing validity. For solver qualifications and
numerical diagnostics, see :doc:`static_network_workflow` and
:doc:`heterogeneous_fem`.
