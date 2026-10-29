Getting Started
===============

Choose a workflow
-----------------

FiberNet is the Python library. FiberScope is a separate desktop application
in the same repository. Installing the library does not install the desktop UI.
Start with :doc:`quickstart` for a small graph, a beam-FEM simulation, and a
saved result. No GPU, Taichi, or desktop application is needed for that example.

Installation
------------

The library requires Python 3.9 or newer. Its base dependencies are NumPy,
SciPy, NetworkX, and Matplotlib. Use a virtual environment to keep optional
workflows separate from your other projects:

.. code-block:: bash

   python -m venv .venv
   # Linux/macOS:
   source .venv/bin/activate
   # Windows PowerShell instead:
   # .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install fibernet

To use the examples included in a source checkout:

.. code-block:: bash

   git clone https://github.com/GellmanSparrowS/fibernet.git
   cd fibernet
   python -m pip install -e .
   python -m examples.open_source_workflow --output-dir demo_output

Source and published package versions can differ. Check the version and
location actually imported before reporting a problem:

.. code-block:: python

   import fibernet as fn
   print(fn.__version__)
   print(fn.__file__)

Optional dependencies
~~~~~~~~~~~~~~~~~~~~~

Install only the extras needed for your workflow. Quote the package expression
so shells do not interpret square brackets:

* ``python -m pip install "fibernet[viz]"``: PyVista 3D visualization
* ``python -m pip install "fibernet[mesh]"``: Trimesh-based mesh tools
* ``python -m pip install "fibernet[manufacturing]"``: solid generation and mesh reduction
* ``python -m pip install "fibernet[ml]"``: scikit-learn, pandas, and tqdm
* ``python -m pip install "fibernet[rl]"``: reinforcement-learning dependencies
* ``python -m pip install "fibernet[accel]"``: optional Taichi backend
* ``python -m pip install "fibernet[io]"``: HDF5 support
* ``python -m pip install -e ".[fracture]"``: experimental fracture backend (4.3.0 source)

For a checkout, use the equivalent editable command, for example
``python -m pip install -e ".[ml]"``. The desktop application has its own
requirements; library extras do not replace them.

Choose the model deliberately
-----------------------------

* ``fn.simulate(..., backend="fem", mode="stretch", strain=1.01)`` uses the
  beam-frame FEM shortcut. Despite the parameter name, ``strain`` here is a
  stretch ratio: 1.01 means 1% extension. Use a small extension for this
  small-strain model.
* ``ReducedBeamSolver`` and ``ReducedBeamConfig`` expose the desktop's reduced
  axial-spring, degree-two bending, and optional node-contact model. The
  source example ``python -m examples.reduced_recruitment_workflow`` saves
  trajectories and recruitment results.
* ``analyze_tensile_recruitment`` analyzes positive axial edge-strain
  thresholds on an existing trajectory. Its spanning components are graph
  statistics, not a complete mechanical force-flow calculation.
* The experimental fracture API is a separate linear-static member-removal
  model. Read its methods and limits before using it; selecting a regular
  spring or FEM shortcut does not enable fracture.

These models have different assumptions and output meanings. Their forces
are not calibrated material measurements. See the repository's
`beam-FEM contract <https://github.com/GellmanSparrowS/fibernet/blob/main/docs/METHODS_BEAM_FEM_CONTRACT.md>`_,
`reduced-model methods <https://github.com/GellmanSparrowS/fibernet/blob/main/docs/METHODS_REDUCED_SOLVER_UNIFIED.md>`_,
and `fracture methods <https://github.com/GellmanSparrowS/fibernet/blob/main/docs/FRACTURE_METHODS.md>`_.

Desktop application
-------------------

The documented FiberScope 3.1 Windows build is distributed with the
`4.2.0 release <https://github.com/GellmanSparrowS/fibernet/releases/tag/v4.2.0>`_.
Extract the entire ZIP before running ``FiberScope.exe``. Python is not
required for that executable. Source launch, optional external RL setup, and
first-run steps are in the
`desktop guide <https://github.com/GellmanSparrowS/fibernet/blob/main/FiberScope/README.md>`_.

Next steps
----------

* Follow :doc:`quickstart` for a complete minimal Python workflow
* Browse :doc:`api/index` for API reference
* Use the repository's ``examples/`` directory for runnable source workflows
