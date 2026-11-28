# SPDX-License-Identifier: MIT
"""Compare two fixed three-dimensional Stokes pore shapes under one pressure drop.

The open channels have one shared, narrow feed and one shared exit. A
synthetic local constriction changes only one of the three channels. This
example is not a time-dependent or fluid–structure solver.
"""

from argparse import ArgumentParser
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np

from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim.tet4_open_stokes import solve_open_pressure_stokes, write_open_stokes_velocity_vtu


def mesh_case(output: Path, mode: bool) -> None:
    """Generate a bounded mesh from the synthetic channel geometry (mm)."""
    import gmsh

    gmsh.initialize(readConfigFiles=False)
    try:
        gmsh.model.add("three-route-pore")
        cad = gmsh.model.occ
        pieces = [(3, cad.addCylinder(x, 0, 0, .53, 0, 0, .43)) for x in (0, 2.97)]
        pieces.extend([(3, cad.addCylinder(-.72, 0, 0, .528, 0, 0, .122)),
                       (3, cad.addCone(-.194, 0, 0, .30, 0, 0, .122, .43))])
        for route in (0, 1, 2):
            wires = []
            for i in range(29):
                t = i / 28
                x = .30 + 2.90 * t
                w = math.sin(math.pi * t) ** 1.25
                phase = 2 * math.pi * t
                if route == 0:
                    y = .153 + w * (.175 + .044 * math.cos(2 * phase + .6))
                    z = .025 + w * (.105 * math.sin(1.25 * phase - .65) + .049)
                    ry = .139 + w * (.036 * math.cos(2 * phase - .3) + .035)
                    rz = .143 + w * (.047 * math.cos(1.65 * phase + .45) + .023)
                    if mode and abs(x - 1.75) < .54:
                        blend = (1 + math.cos(math.pi * (x - 1.75) / .54)) / 2
                        ry = (1 - blend) * ry + .055 * blend
                        rz = (1 - blend) * rz + .043 * blend
                elif route == 1:
                    y = -.144 - w * (.185 + .054 * math.sin(1.35 * phase + .2))
                    z = -.018 - w * (.13 + .09 * math.sin(1.26 * phase + .5))
                    ry = .135 + w * (.025 * math.sin(2 * phase + .32) + .026)
                    rz = .148 + w * (.055 * math.sin(1.7 * phase + .45) + .024)
                else:
                    y = -.148 - w * (.125 + .043 * math.sin(1.3 * phase + .35))
                    z = .12 + w * (.153 + .071 * math.cos(1.2 * phase + .15))
                    ry = .085 + w * (.026 * math.sin(2 * phase - .65) + .018)
                    rz = .091 + w * (.023 * math.sin(1.55 * phase + .6) + .013)
                rz = min(rz, ry * .94)
                angle = .34 * math.sin(phase + .7 * route)
                ellipse = cad.addEllipse(x, y, z, ry, rz, zAxis=[1, 0, 0],
                                         xAxis=[0, math.cos(angle), math.sin(angle)])
                wires.append(cad.addWire([ellipse]))
            pieces.extend(cad.addThruSections(wires, makeSolid=True, maxDegree=3))
        volumes, _ = cad.fuse(pieces[:1], pieces[1:])
        cad.synchronize()
        if len(volumes) != 1:
            raise ValueError("Expected one connected fluid volume")
        surfaces = [[], [], []]
        for dim, surface in gmsh.model.getBoundary(volumes, oriented=False):
            x_min, _, _, x_max, _, _ = gmsh.model.getBoundingBox(dim, surface)
            if abs(x_min + .72) + abs(x_max + .72) < 2.e-5:
                surfaces[0].append(surface)
            elif abs(x_min - 3.5) + abs(x_max - 3.5) < 2.e-5:
                surfaces[1].append(surface)
            else:
                surfaces[2].append(surface)
        for group, cap_radius in zip(surfaces[:2], (.122, .43)):
            if len(group) != 1 or abs(cad.getMass(2, group[0]) - math.pi * cap_radius ** 2) > 1.e-10:
                raise ValueError("Expected the original registered pressure cap")
        for tag, (name, group) in enumerate(zip(("inlet", "outlet", "wall"), surfaces), 10):
            gmsh.model.addPhysicalGroup(2, group, tag, name)
        gmsh.model.addPhysicalGroup(3, [volumes[0][1]], 101, "water")
        options = {"General.NumThreads": 1, "Mesh.MeshSizeMin": .043,
                   "Mesh.MeshSizeMax": .157, "Mesh.MeshSizeFromCurvature": 7,
                   "Mesh.Algorithm3D": 1, "Mesh.RandomSeed": 1}
        for option, value in options.items():
            gmsh.option.setNumber(option, value)
        fld = gmsh.model.mesh.field.add("Box")
        settings = dict(VIn=.057, VOut=.157, XMin=1.36, XMax=2.14, YMin=.10,
                        YMax=.47, ZMin=-.25, ZMax=.25, Thickness=.1)
        for name, value in settings.items():
            gmsh.model.mesh.field.setNumber(fld, name, value)
        gmsh.model.mesh.field.setAsBackgroundMesh(fld)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(output))
    finally:
        gmsh.finalize()


_P2_EDGES = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))


def _section_flux(field, section_x_m=0.00175037):
    """Integrate signed P2 flow on a section where the two groups are disjoint."""
    cells = field.tetrahedra10
    x = field.points_m[cells[:, :4]]
    speed = field.velocity_m_s[cells, 0]
    ids = np.flatnonzero((x[:, :, 0].min(1) < section_x_m) & (x[:, :, 0].max(1) > section_x_m))
    flow = np.zeros(2)
    if np.any(x[:, :, 0] == section_x_m):
        raise ValueError("Section must not pass exactly through a mesh vertex")
    for i in ids:
        vertices = x[i]
        bary = []
        for a, b in combinations(range(4), 2):
            da, db = vertices[[a, b], 0] - section_x_m
            if (da < 0.0) == (db < 0.0):
                continue
            ratio = da / (da - db)
            row = np.zeros(4)
            row[a], row[b] = 1.0 - ratio, ratio
            bary.append(row)
        points = np.asarray(bary)
        yz = (points @ vertices)[:, 1:]
        side_mask = yz[:, 0] > 0
        if not (side_mask.all() or not side_mask.any()):
            raise ValueError("Channels must be separated by y=0 on the comparison section")
        side = int(side_mask.all())
        y, z = (yz - yz.mean(0)).T
        order = np.argsort(np.arctan2(z, y))
        points, yz = points[order], yz[order]
        for j in range(1, len(order) - 1):
            b = points[[0, j, j + 1]]
            tri = yz[[0, j, j + 1]]
            area = abs(np.linalg.det(tri[1:] - tri[:1])) * 0.5
            mid = (b + np.roll(b, -1, axis=0)) * 0.5
            shape = np.c_[mid * (2 * mid - 1),
                          np.array([4 * mid[:, x] * mid[:, y] for x, y in _P2_EDGES]).T]
            flow[side] += area * float(np.mean(shape @ speed[i]))
    return flow


def _run_case(path: Path, narrow: bool):
    import meshio

    mesh_case(path.with_suffix(".msh"), narrow)
    m = meshio.read(path.with_suffix(".msh"))
    points, tets = m.points, m.cells_dict["tetra"].copy()
    bad = np.linalg.det(points[tets[:, 1:]] - points[tets[:, :1]]) < 0
    tets[bad] = tets[bad][:, [1, 0, 2, 3]]
    material = {0: {"young_modulus_pa": 1e6, "poisson_ratio": 0.3}}
    bundle = Tet4Bundle(points, tets, np.arange(len(points)), np.arange(len(tets)),
                       np.zeros(len(tets), dtype=np.int64), material, "mm")
    answer = solve_open_pressure_stokes(
        bundle, pressure_drop_Pa=.0035, viscosity_Pa_s=.001,
        solver="direct", include_quadratic_velocity=True,
    )
    q = _section_flux(answer.quadratic_velocity)
    if np.any(q <= 0) or abs(q.sum() / answer.axial_flow_m3_s - 1) >= .025:
        raise ValueError("Signed section flux not consistent with the cap flux")
    write_open_stokes_velocity_vtu(path.with_suffix(".vtu"), answer.quadratic_velocity)
    return {"tetrahedra": len(tets), "cap_flow_m3_s": answer.axial_flow_m3_s,
            "unmodified_routes_section_m3_s": float(q[0]),
            "constricted_route_section_m3_s": float(q[1]),
            "constricted_route_share": float(q[1] / q.sum()),
            "relative_section_flux_difference": abs(float(q.sum()) / answer.axial_flow_m3_s - 1),
            "cap_balance": answer.cap_mass_relative_error}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New directory for the two meshes and VTU results")
    args = parser.parse_args()
    args.output.mkdir()
    reference = _run_case(args.output / "reference", False)
    narrowed = _run_case(args.output / "narrowed", True)
    summary = {
        "reference": reference, "narrowed": narrowed, "pressure_drop_Pa": .0035,
        "other_route_flux_ratio": narrowed["unmodified_routes_section_m3_s"] /
                                 reference["unmodified_routes_section_m3_s"],
        "total_flux_ratio": narrowed["cap_flow_m3_s"] / reference["cap_flow_m3_s"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
