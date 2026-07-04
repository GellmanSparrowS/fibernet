Basic Workflow Tutorial
=======================

This tutorial demonstrates the complete FiberNet workflow.

1. Network Generation
---------------------

.. code-block:: python

   from fibernet import gen

   # Generate a random 2D fiber network
   net = gen.random_straight_2d(
       num_fibers=100,
       fiber_length=15.0,
       box_size=(50, 50),
       radius=0.1,
       seed=42
   )

   print(f"Generated: {net.num_fibers} fibers, {net.num_crosslinks} crosslinks")
   print(f"Dimension: {net.dimension}")

2. Structural Analysis
----------------------

.. code-block:: python

   from fibernet.analysis import MorphologyAnalyzer, TopologyAnalyzer

   # Morphological properties
   morph = MorphologyAnalyzer(net)
   report = morph.full_report()
   print(f"Nematic order: {report['nematic_order']:.3f}")
   print(f"Mean length: {report['mean_length']:.2f}")
   print(f"Mean tortuosity: {report.get('mean_tortuosity', 1.0):.3f}")

   # Topological properties
   topo = TopologyAnalyzer(net)
   topo_report = topo.full_report()
   print(f"Nodes: {topo_report['num_nodes']}")
   print(f"Connected: {topo_report['is_connected']}")

3. Transformations
------------------

.. code-block:: python

   from fibernet.core.transform import rotate, scale, mirror

   # Rotate by 45 degrees
   rotated = rotate(net, angle=3.14159/4, axis=[0, 0, 1])

   # Scale by 2x
   scaled = scale(net, factor=2.0)

   # Mirror along x-axis
   mirrored = mirror(net, axis=0)

4. Visualization
----------------

.. code-block:: python

   from fibernet.viz import plot_network

   # 2D plot
   plot_network(net, show_crosslinks=True)

   # Or export to VTK for 3D visualization
   from fibernet.io import to_vtk
   to_vtk(net, "network.vtk")

5. Simulation
-------------

.. code-block:: python

   from fibernet.sim.mechanical import FiberFEM

   fem = FiberFEM(net, segments_per_fiber=5)
   result = fem.apply_uniaxial_strain(strain=0.01, axis=0)

   print(f"Energy: {result.energy:.4e} J")
   print(f"Modulus: {fem.effective_modulus(strain=0.001, axis=0):.4e} Pa")
