# SPDX-License-Identifier: MIT
"""Run iterative Tet4 mechanics on a Gmsh mesh with named grips and materials.

The generated fixture needs only Meshio and the optional solid solver.
See the Tet4 bundle guide for the required physical groups of an input mesh.
"""
from collections import Counter
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
import argparse
import json
import math
import os
import re

import meshio
import numpy as np

from .tet4_mechanoelectric_looped_network import _lattice
from fibernet.io.tet4_meshio import read_tet4_vtu_bundle, write_tet4_vtu_bundle
from fibernet.sim.tet4 import _LENGTH_TO_M, solve_static_tet4


_MATERIAL_GROUPS = ('material_low', 'material_high')
_GRIP_GROUPS = ('hold_face', 'pull_face')


def make_fixture(path, *, refinement=3):
    p, t = _lattice(refinement=refinement)
    faces = Counter(face for cell in t for face in combinations(sorted(map(int, cell)), 3))
    left = [f for f, n in faces.items() if n == 1 and np.all(p[list(f), 0] == 0.)]
    right = [f for f, n in faces.items() if n == 1 and np.all(p[list(f), 0] == 3.)]
    tri = np.asarray(left + right, dtype=np.int64)
    tags = [np.where(np.all(p[t][:,:,0] >= 1.5, axis=1), 19, 7),
            np.repeat([31, 37], [len(left), len(right)])]
    named = {"material_low":[7,3], "material_high":[19,3],
             "hold_face":[31,2], "pull_face":[37,2]}
    m = meshio.Mesh(p, [("tetra",t),("triangle",tri)],
                    cell_data={"gmsh:physical":tags,"gmsh:geometrical":tags},field_data=named)
    meshio.write(path, m, file_format="gmsh22", binary=False)


def _volume_entity_tags(path, used_entities):
    # Gmsh 4.1 attaches all physical tags to each entity. Meshio's primary
    # cell labels carry only the first tag, and its sets omit unnamed tags.
    found_section = False
    matches = {}
    with path.open("rb") as stream:
        for line in stream:
            if line.strip() != b"$Entities":
                continue
            if found_section:
                raise ValueError("Duplicate Gmsh 4.1 Entities section")
            found_section = True
            try:
                counts = [int(v) for v in stream.readline().split()]
                if len(counts) != 4 or any(n < 0 for n in counts):
                    raise ValueError
                for _ in range(sum(counts[:3])):
                    row = stream.readline().strip()
                    if not row or row.startswith(b"$"):
                        raise ValueError
                for _ in range(counts[3]):
                    row = stream.readline().split()
                    ent = int(row[0])
                    n = int(row[7])
                    if n < 0 or len(row) < 9+n:
                        raise ValueError
                    if ent in used_entities:
                        if ent in matches:
                            raise ValueError
                        matches[ent] = tuple(map(int, row[8:8+n]))
                if stream.readline().strip() != b"$EndEntities":
                    raise ValueError
            except (IndexError, ValueError) as exc:
                raise ValueError("Invalid Gmsh 4.1 volume entity records") from exc
    if not found_section or matches.keys() != used_entities:
        raise ValueError("Missing Gmsh 4.1 entity records for tetrahedra")
    return matches


def prepare(source, out, *, length_unit, axis, stroke_m, moduli_Pa=(1e9, 2e9), poisson_ratio=.3,
            material_groups=_MATERIAL_GROUPS, grip_groups=_GRIP_GROUPS):
    def valid_names(group, lengths):
        return (isinstance(group, (tuple, list)) and len(group) in lengths
                and all(isinstance(name, str) and name.strip() for name in group)
                and len(set(group)) == len(group))

    if not valid_names(material_groups, (1, 2)) or not valid_names(grip_groups, (2,)):
        raise ValueError("Expected distinct names: one or two volume groups and two grip groups")
    if (not isinstance(moduli_Pa, (tuple, list)) or len(moduli_Pa) != len(material_groups)
            or any(isinstance(e, bool) or not math.isfinite(e) or not 1e3 <= e <= 1e13 for e in moduli_Pa)):
        raise ValueError("Expected one Young modulus within 1 kPa to 10 TPa for each volume group")
    if isinstance(poisson_ratio,bool) or not -.95 <= poisson_ratio <= .495:
        raise ValueError("Poisson ratio must be between −0.95 and 0.495")
    limit = 24 * 1024**2
    if not source.is_file() or source.stat().st_size > limit:
        raise ValueError("Expected a Gmsh mesh file not exceeding 24 MiB")
    complete_volume_tags = None
    with source.open('rb') as stream:
        marker = stream.readline(128).strip()
        format_fields = stream.readline(128).split()
        if marker != b'$MeshFormat' or format_fields not in ([b'2.2',b'0',b'8'],[b'4.1',b'0',b'8']):
            raise ValueError('Expected ASCII Gmsh 2.2 or 4.1')
        inspect_all_tags = format_fields[0] == b'4.1' and len(material_groups) == 1
        if inspect_all_tags:
            stream.seek(0)
            data = stream.read(limit+1)
            if len(data) > limit:
                raise ValueError("Expected a Gmsh mesh file not exceeding 24 MiB")
    if inspect_all_tags:
        # Both readers must see the same bounded source, not two separately
        # opened copies. Meshio 5.3.5's Gmsh reader requires a pathname.
        with TemporaryDirectory(prefix=".single-volume-", dir=out) as stage:
            snapshot = Path(stage) / "source.msh"
            snapshot.write_bytes(data)
            raw = meshio.read(snapshot)
            entities = raw.get_cell_data("gmsh:geometrical", "tetra")
            complete_volume_tags = _volume_entity_tags(snapshot, set(map(int, entities)))
    else:
        raw = meshio.read(source)
    if any(block.type not in {"tetra","triangle","line","vertex"} for block in raw.cells):
        raise ValueError("Expected linear tetrahedra and exterior physical tags")
    t = raw.get_cells_type("tetra")
    tri = raw.get_cells_type("triangle")
    if (not len(t) or len(t)>200_000 or len(raw.points)>60_000 or
            not np.array_equal(np.unique(t),np.arange(len(raw.points)))):
        raise ValueError("Expected 1–200,000 tetrahedra, at most 60,000 nodes, and no unused nodes")
    if not len(tri) or tri.shape[1:] != (3,) or np.any(tri < 0) or np.any(tri >= len(raw.points)):
        raise ValueError("Triangles must reference valid point rows")
    labels = raw.get_cell_data("gmsh:physical","tetra")
    groups = raw.get_cell_data("gmsh:physical","triangle")
    memberships = raw.cell_sets_dict
    if labels.shape != (len(t),) or groups.shape != (len(tri),):
        raise ValueError("Every tetrahedron and surface triangle needs a physical tag")

    def tag(name, dimension):
        val = raw.field_data.get(name)
        if val is None or tuple(np.shape(val)) != (2,) or int(val[1]) != dimension:
            raise ValueError(f"Missing named physical group {name!r}")
        return int(val[0])

    def checked_group(kind, name, dimension, recorded):
        selected=np.flatnonzero(recorded == tag(name,dimension))
        if memberships:
            all_members=np.asarray(memberships.get(name,{}).get(kind,[]))
            if len(all_members) != len(selected) or not np.array_equal(np.sort(all_members),selected):
                raise ValueError(f'Conflicting or overlapping physical membership: {name}')
        return selected

    if complete_volume_tags is not None:
        selected = tag(material_groups[0], 3)
        if any(tags != (selected,) for tags in complete_volume_tags.values()):
            raise ValueError("Single-material Gmsh 4.1 needs exactly one declared physical tag per tetrahedral volume")
    for name in material_groups:
        checked_group('tetra',name,3,labels)
    materials = {tag(name,3):dict(young_modulus_pa=E,poisson_ratio=poisson_ratio)
                 for name,E in zip(material_groups,moduli_Pa)}
    if len(materials) != len(material_groups) or set(labels) != set(materials):
        raise ValueError("Every tetrahedron must have exactly one of the declared material tags")
    faces = Counter(face for tet in t for face in combinations(sorted(map(int, tet)),3))

    def nodes_for(name):
        chosen = [tuple(sorted(map(int, a))) for a in tri[checked_group("triangle", name, 2, groups)]]
        if not chosen or len(chosen)!=len(set(chosen)) or any(faces[f]!=1 for f in chosen):
            raise ValueError(f"{name} must contain distinct exterior faces")
        return sorted({v for f in chosen for v in f})

    left, right = (nodes_for(name) for name in grip_groups)
    if set(left).intersection(right):
        raise ValueError("Grip node sets must be disjoint")
    n = np.arange(len(raw.points),dtype=np.uint64) + 2**53 + 1
    cells = np.arange(len(t),dtype=np.uint64) + 2**62
    result = write_tet4_vtu_bundle(out/"tagged-solid.zip",points=raw.points,tetrahedra=t,node_ids=n,
        cell_ids=cells,material_ids=labels,materials=materials,length_unit=length_unit)
    sidecar = dict(bundle_sha256=sha256(result.read_bytes()).hexdigest(), axis=axis, stroke_m=stroke_m,
        hold_ids=[str(int(n[j])) for j in left], pull_ids=[str(int(n[j])) for j in right])
    grip_bytes = (json.dumps(sidecar, indent=2) + "\n").encode("utf-8")
    (out/"grips.json").write_bytes(grip_bytes)
    return result, sha256(grip_bytes).hexdigest()


def _write_display_vtu(target, bundle, result):
    """Write same-run undeformed SI coordinates and two solved nodal fields."""
    if (not np.array_equal(result.node_ids, bundle.node_ids) or
            result.displacement_m.shape != bundle.points.shape or
            result.reaction_N.shape != bundle.points.shape):
        raise ValueError("Result rows do not match their originating mesh")
    if os.path.lexists(target):
        raise FileExistsError(f"Display VTU destination already exists: {target}")
    points = bundle.points * _LENGTH_TO_M[bundle.length_unit]
    if not np.isfinite(points).all():
        raise ValueError("Display VTU coordinates could not be converted to metres")
    point_data = {"node_id": bundle.node_ids.astype(np.uint64, copy=False),
                  "displacement_m": result.displacement_m,
                  "reaction_N": result.reaction_N}
    cell_data = {"cell_id": [bundle.cell_ids.astype(np.uint64, copy=False)],
                 "material_id": [bundle.material_ids.astype(np.uint64, copy=False)]}
    view = meshio.Mesh(points, [("tetra", bundle.tetrahedra)], point_data, cell_data)
    with TemporaryDirectory(prefix=".display-vtu-", dir=target.parent) as stage:
        staged = Path(stage) / "deformation.vtu"
        meshio.vtu.write(staged, view, binary=True, compression=None)
        if not 0 < staged.stat().st_size <= 32 * 1024**2:
            raise ValueError("Display VTU exceeds its 32 MiB file bound")
        os.link(staged, target)


def solve(bundle_path, sidecar_path, out, *, prepared_grip_sha256, write_vtu=False):
    if not sidecar_path.is_file() or sidecar_path.stat().st_size > 4 * 1024**2:
        raise ValueError('Expected a small grip sidecar')
    grip_bytes = sidecar_path.read_bytes()
    if sha256(grip_bytes).hexdigest() != prepared_grip_sha256:
        raise ValueError('Grip settings no longer match the prepared input')
    sidecar = json.loads(grip_bytes)
    if sha256(bundle_path.read_bytes()).hexdigest() != sidecar["bundle_sha256"]:
        raise ValueError("Grip sidecar does not match this bundle")
    b = read_tet4_vtu_bundle(bundle_path)
    def grip_ids(key):
        values=sidecar[key]
        if not isinstance(values,list) or not 0<len(values)<=len(b.node_ids):
            raise ValueError("Grip must supply its stable node IDs")
        if any(not isinstance(token,str) or not re.fullmatch(r"[0-9]{1,20}",token) for token in values):
            raise ValueError("Expected decimal stable node IDs")
        ids=tuple(map(int,values))
        if max(ids)>2**64-1 or len(set(ids))!=len(ids):
            raise ValueError("Grip IDs must be unique uint64 values")
        return ids
    left,right=grip_ids("hold_ids"),grip_ids("pull_ids")
    if set(left)&set(right):
        raise ValueError("Grip node sets must be disjoint")
    axis=sidecar["axis"]
    if axis not in ('x','y','z'):
        raise ValueError("Expected an x, y, or z displacement axis")
    col="xyz".index(axis)
    other=tuple(c for c in "xyz" if c!=axis)
    stroke=sidecar["stroke_m"]
    if isinstance(stroke,bool) or not isinstance(stroke,(int,float)) or not math.isfinite(stroke) or stroke==0.0:
        raise ValueError("Expected a finite nonzero prescribed stroke")
    holds = {**dict.fromkeys(left,("x","y","z")), **dict.fromkeys(right,other)}
    result=solve_static_tet4(b,forces_by_id={},fixed_dofs_by_id=holds,
        prescribed_displacements_m_by_id={n:{axis:stroke} for n in right},solver="amg-cg")
    mask=np.isin(result.node_ids,np.asarray(right,dtype=np.uint64))
    f=math.fsum(result.reaction_N[mask,col])
    if np.count_nonzero(mask)!=len(right):
        raise ValueError("The output is missing prescribed grip node IDs")
    np.testing.assert_allclose(result.displacement_m[mask,col],stroke,rtol=0.,atol=0.)
    gross_force = np.linalg.norm(result.reaction_N, axis=1).sum()
    if np.linalg.norm(result.reaction_N.sum(0)) > 2e-8*gross_force:
        raise ValueError("Prescribed-grip reactions are not in force balance")
    np.testing.assert_allclose(result.strain_energy_J,.5*f*stroke,rtol=1e-9)
    np.savez_compressed(out/"nodal-fields.npz",node_ids=result.node_ids,
                        displacement_m=result.displacement_m,reaction_N=result.reaction_N)
    with np.load(out/"nodal-fields.npz",allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored['node_ids'],b.node_ids)
    summary = dict(nodes=len(b.points),cells=len(b.tetrahedra),hold=len(left),pull=len(right),
                   axis=axis,stroke_m=stroke,reaction_N=f,strain_energy_J=result.strain_energy_J,
                   free_residual_inf_N=result.free_residual_inf_N)
    if write_vtu:
        display = out/"deformation.vtu"
        _write_display_vtu(display, b, result)
        summary["deformation_vtu"] = display.name
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    return summary


def main():
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument("--input",type=Path)
    args.add_argument("--output",type=Path,required=True)
    args.add_argument("--length-unit",choices=("m","cm","mm","um","nm"))
    args.add_argument("--axis",choices=("x","y","z"),default="x")
    args.add_argument("--stroke-um",type=float,default=1.5)
    args.add_argument("--material-low-gpa", type=float, help="first volume modulus (default 1 GPa)")
    args.add_argument("--material-high-gpa", type=float, help="second volume modulus (default 2 GPa)")
    args.add_argument("--poisson-ratio",type=float,default=.3)
    args.add_argument("--material-groups", nargs=2, metavar=("LOW", "HIGH"),
                      help="names of the two physical volumes, in modulus order")
    args.add_argument("--grip-groups", nargs=2, metavar=("HOLD", "PULL"),
                      help="names of the two physical surfaces, fixed then driven")
    args.add_argument("--write-vtu", action="store_true",
                      help="also write a ParaView-ready VTU from this solve")
    args.add_argument("--material-group", metavar="NAME", help="single physical volume of a homogeneous solid")
    args.add_argument("--young-modulus-gpa", type=float, help="modulus for the single physical volume")
    a=args.parse_args()
    single = a.material_group is not None or a.young_modulus_gpa is not None
    if single:
        if (a.input is None or a.material_group is None or not a.material_group.strip()
                or a.young_modulus_gpa is None):
            args.error("single-material mode requires --input, --material-group and --young-modulus-gpa")
        if a.material_groups is not None or a.material_low_gpa is not None or a.material_high_gpa is not None:
            args.error("do not mix single-material and two-material flags")
        materials = (a.material_group,)
        moduli = (a.young_modulus_gpa * 1e9,)
    else:
        materials = a.material_groups or _MATERIAL_GROUPS
        moduli = tuple((v if v is not None else fallback) * 1e9
                       for v, fallback in ((a.material_low_gpa, 1.), (a.material_high_gpa, 2.)))
    if any(not 1e3 <= e <= 1e13 for e in moduli):
        args.error("each Young modulus must be finite and within 1 kPa to 10 TPa")
    if a.input is None and (a.material_groups or a.grip_groups):
        args.error("--material-groups and --grip-groups require --input")
    if a.input is not None and a.length_unit is None:
        args.error("--length-unit is required for a supplied mesh")
    if a.input is None and a.length_unit not in (None, "mm"):
        args.error("The generated fixture is in mm; omit --length-unit")
    if not math.isfinite(a.stroke_um) or a.stroke_um==0.0:
        args.error("--stroke-um must be finite and nonzero")
    a.output.mkdir(parents=True,exist_ok=False)
    source=a.input
    if source is None:
        source=a.output/"tagged.msh"
        make_fixture(source)
    b, grip_hash=prepare(source,a.output,length_unit=a.length_unit or "mm",axis=a.axis,stroke_m=a.stroke_um*1e-6,
              moduli_Pa=moduli, poisson_ratio=a.poisson_ratio,
              material_groups=materials, grip_groups=a.grip_groups or _GRIP_GROUPS)
    print(json.dumps(solve(b,a.output/"grips.json",a.output,prepared_grip_sha256=grip_hash,write_vtu=a.write_vtu),indent=2))


if __name__=="__main__":
    main()
