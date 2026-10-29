Tutorials
=========

These examples use the current development checkout. Install it with
``python -m pip install -e .`` from the repository root. See
:doc:`getting_started` for the published package and optional dependencies.

Tutorial 1: Getting Started
---------------------------

Basic network generation and analysis workflow.

.. code-block:: python

   import fibernet as fn
   from fibernet.analysis import analyze_topology

   # Generate network
   # Coordinates and radii are in m.
   graph = fn.pattern_2d(
       unit="honeycomb", box=(10e-3, 10e-3), grid=(2, 2),
       radius=0.05e-3, seed=23,
   )

   # Analyze
   topology = analyze_topology(graph)
   print(topology.n_nodes, topology.n_physical_edges, topology.n_components)

Tutorial 2: Mechanical Simulation
----------------------------------

Continue with the graph above. This explicitly selects a small-deformation
linear beam model with 1% extension and an illustrative modulus of 1 GPa.
Displacements are in m and forces are in N.

.. code-block:: python

   result = fn.simulate(
       graph, backend="fem", mode="stretch", stretch_ratio=1.01,
       nonlinear=False, E=1e9, nu=0.3,
   )
   result.save("fem_result.json", detailed=True)
   restored = fn.SimResult.load("fem_result.json")
   print(f"Maximum displacement: {restored.max_displacement:.3e} m")

See :doc:`quickstart` for output definitions and model limits.

Tutorial 3: Machine Learning
-----------------------------

Extract graph descriptors for a later structure-property model. This step
does not require a learning backend; it creates one row of input features.

.. code-block:: python

   import numpy as np
   from fibernet.ml import light_features

   positions, edges = graph.node_positions(), graph.edge_array()[:, :2]
   features = np.asarray(light_features(positions, edges))
   print(f"Feature vector: {features.shape}")

For a complete example that also works from an installed wheel:

.. code-block:: bash

   python -m fibernet.examples.basic_usage --output-dir fibernet_demo

It saves ``network.json``, ``network.png``, ``fem_result.json``, and
``summary.json``. The graph and simulation result are reloaded before the
summary is calculated.
