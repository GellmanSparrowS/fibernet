Getting Started
===============

Installation
------------

FiberNet requires Python 3.9+ and depends on ``numpy`` and ``scipy``.

.. code-block:: bash

   # Basic installation
   pip install fibernet

   # With all optional features
   pip install fibernet[full]

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

FiberNet is designed to work with only ``numpy`` and ``scipy`` as required
dependencies. Additional features require optional packages:

* ``pip install fibernet[viz]`` — matplotlib + pyvista for visualization
* ``pip install fibernet[graph]`` — networkx for topology analysis
* ``pip install fibernet[ml]`` — scikit-learn + tqdm for ML integration
* ``pip install fibernet[accel]`` — taichi for GPU-accelerated FEM
* ``pip install fibernet[io]`` — h5py for HDF5 I/O
* ``pip install fibernet[full]`` — all optional dependencies

First Example
-------------

.. code-block:: python

   import fibernet as fn

   # Generate a random 2D fiber network
   net = fn.create(
       'random_2d',
       num_fibers=100,
       fiber_length=10.0,
       box_size=(30, 30),
       seed=42
   )

   # Analyze structure
   stats = fn.analyze(net)
   print(f"Fibers: {stats['num_fibers']}")
   print(f"Nematic order: {stats['nematic_order']:.3f}")

   # Run mechanical simulation (FEM)
   result = fn.simulate_mechanics(net, strain=0.01)

   # Export to file
   fn.export(net, 'network.vtk', format='vtk')

Network Types
-------------

Random Networks
~~~~~~~~~~~~~~~

.. code-block:: python

   net = fn.create('random_2d', num_fibers=100, fiber_length=10.0, box_size=(30, 30))
   net = fn.create('random_3d', num_fibers=100, fiber_length=15.0, box_size=(30, 30, 30))

Ordered Lattices
~~~~~~~~~~~~~~~~

.. code-block:: python

   net = fn.create('square_2d', spacing=5.0, grid_size=(10, 10))
   net = fn.create('honeycomb_2d', cell_size=5.0, grid_size=(10, 10))
   net = fn.create('triangular_2d', spacing=5.0, grid_size=(10, 10))
   net = fn.create('cubic_3d', spacing=5.0, grid_size=(5, 5, 5))

Specialized Structures
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from fibernet import gen
   net = gen.chiral_network_2d(num_fibers=50, chirality=0.5, seed=42)
   net = gen.single_helix(radius=5.0, pitch=2.0, turns=3)
   net = gen.plain_weave_2d(spacing=5.0, warp_count=10, weft_count=10)
   net = gen.biomimetic_collagen(num_fibers=100, seed=42)

Simulations
-----------

Mechanical FEM
~~~~~~~~~~~~~~

.. code-block:: python

   from fibernet.sim import FiberFEM

   fem = FiberFEM(net, segments_per_fiber=5)
   result = fem.apply_uniaxial_strain(strain=0.01, axis=0)
   E_eff = fem.effective_modulus(strain=0.001)

Thermal
~~~~~~~

.. code-block:: python

   result = fn.simulate_thermal(net, T_hot=100.0, T_cold=0.0)

Electromagnetic
~~~~~~~~~~~~~~~

.. code-block:: python

   from fibernet.sim import EMSolver
   solver = EMSolver(net)
   result = solver.solve_conductivity(voltage=1.0)

Next Steps
----------

* Browse the :doc:`api/index` for detailed function reference
* Try the :doc:`tutorials` for step-by-step workflows
