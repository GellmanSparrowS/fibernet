Scaling graph geometry
======================

``fibernet.core.transforms.scale`` returns a detached ``StructureGraph``.
A real numeric scalar scales both the coordinates and the circular member
radii. Python integers/floats, NumPy integer/floating scalars and zero-dimensional
numeric arrays select this same operation. In earlier versions, some NumPy
scalars resized the coordinates but left the radii unchanged.

.. code-block:: python

   import numpy as np
   from fibernet.core.structure_graph import StructureGraph
   from fibernet.core.transforms import scale

   graph = StructureGraph(dimension=2, box_size=[1, 1])
   left = graph.add_node([0, 0])
   right = graph.add_node([1, 0])
   graph.add_edge(left, right, radius=0.05)

   larger = scale(graph, np.float32(2))
   assert larger.edge_lengths()[0] == 2
   assert larger.edges[0].radius == 0.1
   assert graph.edges[0].radius == 0.05

An array-like vector retains the existing coordinate-only operation, even when
its components are equal. ``scale(graph, [2, 2, 2])`` doubles all coordinates
but leaves radii unchanged. A two-component vector leaves z unchanged.
Length-one vectors broadcast to the stored coordinates and also leave radii
unchanged. ``make_scale`` and ``compose`` follow the same distinction.
Masked real-numeric scalar values raise ``ValueError``; unmasked zero-dimensional numeric
arrays are accepted. Previously unrecognized numeric scalar representations
use their float64 value, consistent with coordinate calculations. Existing
Python/NumPy float64 scalar arithmetic and its radius type promotion are kept.
This does not introduce a general validator for other scalar-like objects,
vector inputs, or representability of scaled geometry and radii.

For scalar ``s``, node and stored internal-point coordinates about ``center``
are multiplied by ``s``; radii by ``abs(s)``; and box-size coordinates by ``s``.
Negative box-size entries and unchanged boundary flags follow the historical
scaling policy. Zero scaling retains node/edge identities, even though the
collapsed geometry and zero radii are unsuitable for a beam FEM solve.
Scaling does not weld nodes, reallocate IDs or alter segment counts.

This is geometric resizing in the existing unit system, not a unit conversion.
Materials, metadata and boundary labels are copied without reinterpretation.
Young's modulus, Poisson's ratio, density, stiffness tensors, prescribed loads
and displacements are not rescaled. If coordinates are being converted between
units instead, every dimensioned input and result must be converted consistently
by the caller; changing geometry alone is insufficient.

For a positive scalar ``s``, a straight circular beam's length-to-radius ratio
is preserved. Area scales by ``s**2`` and second moment by ``s**4``. At unchanged
Young's modulus, axial stiffness ``EA/L`` and cantilever transverse stiffness
``3EI/L**3`` both scale by ``s``. Thus the same force produces displacement
divided by ``s``. These analytical checks validate the scalar resizing contract,
not general convergence, nonlinear mechanics or curved-beam discretization.
