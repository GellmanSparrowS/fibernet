Periodic pore flow with Tet4 geometry
=====================================

The optional pore-flow solver computes steady, incompressible Newtonian
Stokes flow through a bounded, connected pore. Supply the **fluid domain** as
a :class:`Tet4Bundle`, not the solid fiber geometry. The two end planes normal
to x must have matching triangulations under a translation; the other
boundary faces use no-slip conditions. The input bundle's material entries
keep geometric labels stable but do not set fluid properties. Supply dynamic
viscosity separately in Pa s.

On Python 3.10 or newer, install the ``pore-flow`` extra. From the repository
checkout run ``python -m examples.tet4_periodic_pore`` for an empty duct,
a synthetic fused three-path obstacle, and a manufactured scale check. No
precomputed or external mesh files are needed.

.. code-block:: python

   from fibernet.sim.tet4_pore import solve_x_periodic_stokes

   flow = solve_x_periodic_stokes(
       pore_bundle,
       viscosity_Pa_s=1e-3,
       pressure_gradient_Pa_per_m=500.0,
   )
   print(flow.axial_flow_m3_s, flow.cap_flow_m3_s)

A positive driver is minus the axial pressure derivative, and gives positive
flow along +x. Both cap flow values use the same +x direction. The returned
bulk and cap values are separate finite-element readouts, not interchangeable
when assessing mesh convergence. The result includes a power check, pressure
residual, end balance and relative cap/bulk difference.

``solve_matched_pore_geometry`` applies the same viscosity and axial pressure
gradient to two supplied geometries, checking stable node and ordered cell IDs
and small positive volume ratios. If the axial period changes, fixed gradient
does not mean fixed end-to-end pressure drop. For a fixed drop, call
``solve_x_periodic_stokes`` separately for each geometry with
``pressure_gradient_Pa_per_m`` equal to the drop in Pa divided by that mesh's
axial period in m. The matched solve neither computes the wall displacement
nor moves the solid under the fluid forces. The example's uniform scale
change is an analytical covariance check, not a mechanical material law.

The reference scope is P2 velocity and P1 pressure on straight-sided
tetrahedra, small meshes, a periodically driven connected pore, and a
bounded sparse direct solve. The workload estimate is a pre-assembly
screening check, not a guaranteed ceiling on total process memory.
