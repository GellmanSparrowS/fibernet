Installation
============

Requirements
------------

FiberNet requires Python 3.9 or later. The core dependencies are:

- ``numpy>=1.21``
- ``scipy>=1.7``

Optional dependencies enable additional features:

- **Visualization**: ``pyvista>=0.37``, ``matplotlib>=3.5``
- **I/O formats**: ``h5py>=3.7``
- **GPU acceleration**: ``taichi>=1.6``
- **Machine learning**: ``scikit-learn>=1.0``, ``tqdm>=4.60``
- **Graph analysis**: ``networkx>=3.0``

Install from PyPI
-----------------

.. code-block:: bash

   pip install fibernet

Install with optional dependencies:

.. code-block:: bash

   # All optional dependencies
   pip install fibernet[full]

   # Visualization only
   pip install fibernet[viz]

   # Machine learning
   pip install fibernet[ml]

   # Graph analysis
   pip install fibernet[graph]

Install from Source
-------------------

.. code-block:: bash

   git clone https://github.com/GellmanSparrowS/fibernet.git
   cd fibernet
   pip install -e ".[dev]"

This installs FiberNet in development mode with all development dependencies.

Verify Installation
-------------------

.. code-block:: python

   import fibernet
   print(fibernet.__version__)  # Should print 0.4.0

   # Quick test
   from fibernet import gen
   net = gen.random_straight_2d(num_fibers=10, fiber_length=5, box_size=(20, 20), seed=42)
   print(f"Generated network with {net.num_fibers} fibers")
