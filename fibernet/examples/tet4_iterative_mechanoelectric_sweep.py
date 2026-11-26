# SPDX-License-Identifier: MIT
"""Compute the resistance of a fused lattice over a small proportional stroke sweep.

Run after installing FiberNet with the solid-iterative extra:
python -m fibernet.examples.tet4_iterative_mechanoelectric_sweep [--refinement {3,4,5}] [--initial-slope]
"""
from dataclasses import replace
from math import fsum, isclose
import argparse
import json
import numpy as np

from fibernet.sim.tet4 import solve_static_tet4
from fibernet.sim.tet4_dc import solve_dc_tet4
from .tet4_mechanoelectric_looped_network import _bundle, _fixture, _lattice


def _reference_mechanics(refinement):
    points,tets=_lattice(refinement=refinement)
    mesh=_bundle(points,tets,'stiff_half')
    high,grip,fixed,electrodes,_=_fixture(mesh)
    stroke_m=1.5e-6
    displacement={nid:{'x':stroke_m} for nid in grip}
    mech=solve_static_tet4(mesh,forces_by_id={},fixed_dofs_by_id=fixed,
                           prescribed_displacements_m_by_id=displacement,
                           solver='amg-cg')
    reaction=fsum(mech.reaction_N[high,0])
    if reaction <= 0 or not isclose(mech.strain_energy_J,.5*reaction*stroke_m,rel_tol=1e-8):
        raise ValueError("Calculated grip reaction and elastic energy disagree")
    return mesh,high,electrodes,stroke_m,mech,reaction


def run(*, refinement=3):
    mesh,high,electrodes,stroke_m,mech,reaction=_reference_mechanics(refinement)
    points,tets=mesh.points,mesh.tetrahedra
    xyz=points[tets]
    initial_vol=np.linalg.det(xyz[:,1:]-xyz[:,:1])
    result=[]
    for fraction in (-1.,-.5,0.,.5,1.):
        # Fixed linear elasticity and zero preload: any proportional signed
        # stroke has the same displacement shape. DC is solved at each shape.
        deformed=points + (1000.*fraction)*mech.displacement_m
        tet=deformed[tets]
        volume=np.linalg.det(tet[:,1:]-tet[:,:1])
        if not np.all(np.isfinite(volume) & (volume*initial_vol > 0.)):
            raise ValueError("Stroke would invert or collapse a solid cell")
        dc=solve_dc_tet4(replace(mesh,points=deformed),
                        conductivity_S_per_m_by_material=dict.fromkeys(mesh.materials,1.),
                        electrode_voltage_V_by_id=electrodes,solver='cg')
        current=fsum(dc.electrode_current_into_solid_A[high])
        if not current>0.:
            raise ValueError("Expected positive current at the high electrode")
        result.append(dict(stroke_m=stroke_m*fraction, reaction_N=reaction*fraction,
                    strain_energy_J=mech.strain_energy_J*fraction**2, resistance_ohm=1./current))
    return dict(nodes=len(points),tetrahedra=len(tets),states=result)


def _power_weighted_strain(points_m, cells, displacement_m, current, power_w):
    """Return two compensated strain contributions for this bounded fixture."""
    stretch = []
    contraction = []
    if not (np.isfinite(current).all() and np.isfinite(power_w).all() and
            np.all(power_w >= 0.)):
        raise ValueError("Expected finite current and nonnegative cell power")
    for start in range(0, len(cells), 4096):
        sl = slice(start, start + 4096)
        tet = cells[sl]
        x = points_m[tet[:, 1:]] - points_m[tet[:, :1]]
        scale = np.max(np.abs(x), axis=(1, 2))
        if np.any(scale < 1e-12):
            raise ValueError("Cell is outside the reference geometry scale")
        shape = x / scale[:, None, None]
        if np.any(np.linalg.cond(shape) > 1e6):
            raise ValueError("Cell shape is outside the reference geometry range")
        du = displacement_m[tet[:, 1:]] - displacement_m[tet[:, :1]]
        gradient_transpose = np.linalg.solve(shape, du / scale[:, None, None])
        # For trace and n^T * grad(u) * n, the antisymmetric part vanishes.
        j = current[sl]
        magnitude_scale = np.max(np.abs(j), axis=1, keepdims=True)
        q = np.divide(j, magnitude_scale, out=np.zeros_like(j), where=magnitude_scale > 0.)
        norm = np.linalg.norm(q, axis=1, keepdims=True)
        direction = np.divide(q, norm, out=np.zeros_like(q), where=norm > 0.)
        w = power_w[sl]
        if np.any(w[norm[:, 0] == 0.] != 0.):
            raise ValueError("Zero-current cells must have zero Joule power")
        along = np.einsum("ni,nij,nj->n", direction, gradient_transpose, direction)
        across = np.trace(gradient_transpose, axis1=1, axis2=2) - along
        stretch.append(fsum(w * along))
        contraction.append(-fsum(w * across))
    return fsum(stretch), fsum(contraction)


def run_initial_slope(*, refinement=3):
    """Find the zero-stroke geometry tangent at fixed conductivity and voltage.

    The DC potential minimizes Joule power at the reference geometry, so
    its own variation contributes zero to the first-order energy balance.
    The two terms are power-weighted strain along and across local current.
    This does not approximate the resistance at a finite stroke.
    """
    mesh, high, electrodes, stroke_m, mechanical, _ = _reference_mechanics(refinement)
    dc = solve_dc_tet4(mesh, conductivity_S_per_m_by_material=dict.fromkeys(mesh.materials, 1.),
                       electrode_voltage_V_by_id=electrodes, solver='cg')
    if not (np.array_equal(mesh.cell_ids, dc.cell_ids) and
            np.array_equal(mesh.node_ids, mechanical.node_ids) and
            np.array_equal(mesh.node_ids, dc.node_ids)):
        raise ValueError("Reference mechanics and DC must share the mesh row IDs")
    current = fsum(dc.electrode_current_into_solid_A[high])
    voltage_span = max(electrodes.values()) - min(electrodes.values())
    power = fsum(dc.joule_power_W_by_cell)
    if current <= 0 or power <= 0 or not isclose(power, current * voltage_span, rel_tol=3e-8):
        raise ValueError("Reference power and electrode current disagree")
    along, cross = _power_weighted_strain(mesh.points * 1.e-3, mesh.tetrahedra,
                                          mechanical.displacement_m, dc.current_density_A_per_m2,
                                          dc.joule_power_W_by_cell)
    scale = power * (stroke_m * 1.e6)
    return dict(nodes=len(mesh.points), tetrahedra=len(mesh.tetrahedra),
                resistance_at_zero_ohm=voltage_span / current,
                current_aligned_stretch_per_um=along / scale,
                transverse_contraction_per_um=cross / scale,
                relative_resistance_slope_per_um=(along + cross) / scale)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-slope", action="store_true",
                        help="compute only d(R/R0)/d(stroke) at zero stroke")
    parser.add_argument("--refinement", type=int, choices=(3, 4, 5), default=3,
                        help="subdivisions per voxel axis: 28,512 (default), "
                             "67,584, or 132,000 tetrahedra")
    args = parser.parse_args(argv)
    solve = run_initial_slope if args.initial_slope else run
    result = solve(refinement=args.refinement)
    print(json.dumps(result, indent=2))


if __name__=='__main__':
    main()
