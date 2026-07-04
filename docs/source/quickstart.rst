Quick Start Guide
=================

This guide walks you through the basic workflow of FiberNet in 5 minutes.

Step 1: Generate a Network
--------------------------

.. code-block:: python

   from fibernet import gen

   # Simple random 2D network
   net = gen.random_straight_2d(
       num_fibers=100,
       fiber_length=15.0,
       box_size=(50, 50),
       radius=0.1,
       seed=42
   )

   print(f"Fibers: {net.num_fibers}")
   print(f"Crosslinks: {net.num_crosslinks}")

Available generators include:

- ``random_straight_2d`` / ``random_straight_3d`` - Random straight fibers
- ``random_walk_fibers`` - Random walk polymer-like fibers
- ``square_lattice_2d`` / ``cubic_lattice_3d`` - Regular lattices
- ``honeycomb_lattice_2d`` - Honeycomb structure
- ``voronoi_network_2d`` / ``voronoi_network_3d`` - Voronoi tessellations
- ``cnt_network_2d`` / ``cnt_network_3d`` - Carbon nanotube networks
- ``paper_network`` - Cellulose paper fibers
- ``textile_weave`` - Woven textile structures
- ``electrospun_mat`` - Electrospun nanofiber mats
- ``double_helix`` - DNA-like double helix

Step 2: Analyze Structure
-------------------------

.. code-block:: python

   from fibernet.analysis import MorphologyAnalyzer, TopologyAnalyzer

   # Morphological analysis
   morph = MorphologyAnalyzer(net)
   print(f"Nematic order: {morph.nematic_order():.3f}")
   print(f"Mean length: {morph.mean_length():.2f}")
   print(f"Mean tortuosity: {morph.mean_tortuosity():.3f}")

   # Topological analysis
   topo = TopologyAnalyzer(net)
   print(f"Nodes: {topo.num_nodes}")
   print(f"Edges: {topo.num_edges}")
   print(f"Connected: {topo.is_connected()}")

Step 3: Simulate Physics
------------------------

.. code-block:: python

   from fibernet.sim.mechanical import FiberFEM

   # Mechanical simulation
   fem = FiberFEM(net, segments_per_fiber=5)
   result = fem.apply_uniaxial_strain(strain=0.01, axis=0)

   print(f"Energy: {result.energy:.4e} J")
   print(f"Max stress: {result.max_stress():.4e} Pa")

   # Effective modulus
   E = fem.effective_modulus(strain=0.001, axis=0)
   print(f"Modulus: {E:.4e} Pa")

Step 4: Export Results
----------------------

.. code-block:: python

   # Save to JSON
   net.save_json("my_network.json")

   # Export to VTK for visualization in Paraview
   from fibernet.io import to_vtk
   to_vtk(net, "network.vtk")

   # Export to LAMMPS for molecular dynamics
   from fibernet.io import to_lammps
   to_lammps(net, "network.lammps", bead_spacing=1.0)

Step 5: High-Level API
----------------------

For convenience, FiberNet provides a simplified API:

.. code-block:: python

   import fibernet as fn

   # Quick generation
   net = fn.create("random_2d", num_fibers=50, fiber_length=10, box_size=(30, 30))

   # Quick analysis
   results = fn.analyze(net)
   print(results)

   # Quick simulation
   mech = fn.simulate_mechanics(net, strain=0.01, axis=0, model="linear")
   print(f"Modulus: {mech['modulus']:.2e} Pa")

   # Quick export
   fn.export(net, "output.vtk")

Next Steps
----------

- See :doc:`tutorials/index` for detailed tutorials
- Explore :doc:`examples/index` for research-level examples
- Check :doc:`api/index` for complete API reference
