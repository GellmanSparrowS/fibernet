Quick Start Guide
=================

This guide uses the current 4.3.0 development checkout, not the published PyPI
4.2.0 API, which does not accept the ``stretch_ratio`` keyword below. From the
cloned repository's root, install the development checkout and its base Python
dependencies before running the examples:

.. code-block:: bash

   python -m pip install -e .

This command installs the source checkout rather than a PyPI release.
See :doc:`getting_started` for other installation options. The example
generates a graph, runs a small-extension beam-FEM simulation, and verifies that
a saved result can be read back. It does not require FiberScope or an optional
GPU backend.

Generate, simulate, and save
----------------------------

Run this complete block in a notebook or Python script. It writes into a
``demo_output`` directory beneath your current working directory.

.. code-block:: python

   from pathlib import Path
   import numpy as np
   import matplotlib.pyplot as plt
   import fibernet as fn

   output = Path("demo_output")
   output.mkdir(parents=True, exist_ok=True)
   # SI units: coordinates and radius in m; E in Pa; forces in N.
   graph = fn.pattern_2d(
       unit="honeycomb", box=(10, 10), grid=(2, 2), radius=0.05, seed=23
   )
   figure = fn.show(graph, theme="publication",
                    save_path=str(output / "network.png"))
   plt.close(figure)
   result = fn.simulate(
       graph, backend="fem", mode="stretch", stretch_ratio=1.01,
       nonlinear=False, E=1e9, nu=0.3,
       save_path=str(output / "fem_result.json")
   )
   restored = fn.SimResult.load(str(output / "fem_result.json"))
   assert np.isclose(result.max_displacement, restored.max_displacement)
   assert np.isfinite(result.max_displacement)
   print(graph.num_nodes, graph.num_edges, result.max_displacement)

``stretch_ratio=1.01`` means 1% extension; ``nonlinear=False`` selects the
small-deformation linear beam solver. Coordinates and radii are in m, ``E``
is in Pa, and ``nu`` is dimensionless; displacement is in m and force in N.
The solver does not convert units or infer moduli from graph material records.
Omitting the ratio retains the legacy default 2.0, which automatically selects
experimental 2-D updated-geometry increments with unverified final equilibrium.
That route is not a validated nonlinear beam solver. The warning switch at
|stretch-1| > 0.3 is not a physical validity bound; no warning does not certify
that a structure remains within small-deformation assumptions.
The example checks the software workflow; it is not a calibrated material
prediction. ``max_force`` is the largest individual beam axial force, not a
grip reaction or effective modulus. The result's ``energy`` is an axial-stress
proxy rather than total bending-plus-contact energy.

Analyze tensile recruitment
---------------------------

This independent example supplies edge-strain data directly, without running
a mechanical solver:

.. code-block:: python

   import numpy as np
   from fibernet.analysis import analyze_tensile_recruitment

   edges = np.array([[0, 1], [1, 2], [0, 2]])
   edge_strain = np.array([[0.0, 0.0, 0.0], [0.2, 0.3, -0.1]])
   recruitment = analyze_tensile_recruitment(
       edge_strain, edges, left_nodes=[0], right_nodes=[2], n_nodes=3
   )
   assert recruitment.perc_frame == 1
   print(recruitment.perc_frame)

Rows of ``edge_strain`` are frames; columns match the order of ``edges``.
Node indices in the edges and grips must refer to the same node ordering.
A spanning recruited component indicates thresholded tensile connectivity,
not complete force transfer. Changing the threshold can change this result
without changing the simulated trajectory. By default, the threshold is
``max(alpha * frame_max_positive_strain, 0.0001)`` with hysteresis for
already recruited edges; ``alpha=0.05`` is not a fixed 5% strain cutoff.

Runnable source examples
------------------------

The ``examples`` directory belongs to the source checkout, not the installed
library package. From the repository root:

.. code-block:: bash

   python -m examples.open_source_workflow --output-dir demo_output
   python -m examples.reduced_recruitment_workflow --output-dir demo_stretch
   python -m examples.manufacturing_workflow --output-dir demo_routes

The first command combines graph generation, a network image, FEM result
serialization, and recruitment analysis. It writes ``network.png``,
``fem_result.json``, and ``summary.json``. The second uses the desktop's
reduced model and saves its trajectory. The third compiles closed routes;
a closed graph route alone does not certify physical printability.

Troubleshooting
---------------

* If ``examples`` cannot be imported, run from a source checkout's root after
  ``python -m pip install -e .``
* If the wrong package is imported, inspect ``fibernet.__file__`` and
  ``fibernet.__version__`` in the same Python environment as your script
* On a machine without a display, set ``MPLBACKEND=Agg`` before running image
  examples; inspect the saved PNG instead of opening an interactive window
* For a bug report, include your Python and FiberNet versions, the command,
  and the complete error at the
  `issue tracker <https://github.com/GellmanSparrowS/fibernet/issues>`_

Next steps
----------

Read :doc:`getting_started` for model choices and optional dependencies, or
:doc:`api/index` for the API reference.
