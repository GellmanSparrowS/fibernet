Tutorials
=========

Jupyter notebook tutorials are available in the ``tutorials/`` directory.

Tutorial 1: Getting Started
---------------------------

Basic network generation and analysis workflow.

.. code-block:: python

   import fibernet as fn
   from fibernet import gen

   # Generate network
   net = gen.random_straight_2d(num_fibers=100, fiber_length=10.0, 
                                 box_size=(30, 30), seed=42)

   # Analyze
   from fibernet.analysis import MorphologyAnalyzer
   morph = MorphologyAnalyzer(net)
   print(f"Order: {morph.nematic_order_parameter():.3f}")

Tutorial 2: Mechanical Simulation
----------------------------------

Running FEM simulations on fiber networks.

.. code-block:: python

   from fibernet.sim import FiberFEM

   fem = FiberFEM(net, segments_per_fiber=5)

   # Uniaxial tension
   result = fem.apply_uniaxial_strain(strain=0.01, axis=0)
   print(f"Energy: {result.energy:.2e} J")

   # Compute effective modulus
   E_eff = fem.effective_modulus(strain=0.001, axis=0)
   print(f"E_eff = {E_eff:.2e} Pa")

Tutorial 3: Machine Learning
-----------------------------

Feature extraction and property prediction.

.. code-block:: python

   from fibernet.ml import FeatureExtractor

   extractor = FeatureExtractor()
   features = extractor.extract_features(net)
   print(f"Feature vector: {features.shape}")
