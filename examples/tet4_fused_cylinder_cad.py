"""Remesh one tagged OCC cylinder union twice; export native Tet4 bundles."""
import argparse
from collections import Counter
from hashlib import sha256
from itertools import combinations
import json
from math import pi, sqrt
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

RADIUS_MM = 0.5
HALF_LENGTH_MM = 1.5
CASES = (("coarse", 0.33), ("fine", 0.24))
SPHERE_CASES = (("coarse", 0.33), ("fine", 0.26))
JUNCTION_RADIUS_MM = 0.30
JUNCTION_VOLUME_MM3 = 4 * pi * JUNCTION_RADIUS_MM**3 / 3
CAPS = {f"cap_{sign}_{ax}": (i, plane)
        for i, ax in enumerate("xyz")
        for sign, plane in (("neg", -HALF_LENGTH_MM), ("pos", HALF_LENGTH_MM))}
BOUNDARIES = (*CAPS, "ambient_side")
ANALYTIC_VOLUME_MM3 = 6 * HALF_LENGTH_MM * pi * RADIUS_MM**2 - 16 * RADIUS_MM**3 + 8 * (
    2 - sqrt(2)) * RADIUS_MM**3


def generate(gmsh, path, size_mm, junction_sphere=False):
    """Create a fresh CAD model and mesh; the only output is an ASCII MSH 4.1 file."""
    gmsh.clear()
    gmsh.model.add(path.stem)
    occ = gmsh.model.occ
    cylinders = []
    for axis in range(3):
        start, direction = [0.] * 3, [0.] * 3
        start[axis], direction[axis] = -HALF_LENGTH_MM, 2 * HALF_LENGTH_MM
        cylinders.append((3, occ.addCylinder(*start, *direction, RADIUS_MM)))
    union, _ = occ.fuse(cylinders[:1], cylinders[1:])
    occ.synchronize()
    if len(union) != 1 or union[0][0] != 3:
        raise ValueError("Expected a single fused CAD solid")
    volume = occ.getMass(*union[0])
    if abs(volume - ANALYTIC_VOLUME_MM3) > 1e-6:
        raise ValueError("Fused CAD solid does not match the analytic union volume")
    regions = {"solid": union[0][1]}
    if junction_sphere:
        sphere = occ.addSphere(0., 0., 0., JUNCTION_RADIUS_MM)
        union, provenance = occ.fragment(union, [(3, sphere)])
        occ.synchronize()
        if len(union) != 2 or set(provenance[0]) != set(union) or len(provenance[1]) != 1:
            raise ValueError("Expected two conformal CAD volumes from the junction fragment")
        junction = provenance[1][0]
        if junction not in union or any(dim != 3 for dim, _ in union):
            raise ValueError("Invalid sphere fragment provenance")
        bulk, = set(union) - {junction}
        sphere_volume = occ.getMass(*junction)
        if (abs(sphere_volume - JUNCTION_VOLUME_MM3) > 1e-8
                or abs(occ.getMass(*bulk) + sphere_volume - volume) > 1e-6):
            raise ValueError("CAD fragment volumes do not match the sphere and fused union")
        regions = {"bulk": bulk[1], "junction": junction[1]}
    exterior = {tag for dim, tag in gmsh.model.getBoundary(
        union, combined=True, oriented=False) if dim == 2}
    remaining = set(exterior)
    for name, (axis, plane) in CAPS.items():
        tags = []
        for tag in exterior:
            bbox = gmsh.model.getBoundingBox(2, tag)
            if max(abs(bbox[axis] - plane), abs(bbox[axis + 3] - plane)) < 1e-5:
                tags.append(tag)
        if (len(tags) != 1 or tags[0] not in remaining
                or abs(occ.getMass(2, tags[0]) - pi * RADIUS_MM**2) > 1e-8):
            raise ValueError(f"Could not identify one full CAD disk for {name}")
        remaining.remove(tags[0])
        gmsh.model.addPhysicalGroup(2, tags, name=name)
    if not remaining:
        raise ValueError("No CAD side surfaces remain")
    gmsh.model.addPhysicalGroup(2, sorted(remaining), name="ambient_side")
    for name, tag in regions.items():
        gmsh.model.addPhysicalGroup(3, [tag], name=name)
    for key, val in {
        "General.NumThreads": 1,
        "Mesh.MaxNumThreads1D": 1, "Mesh.MaxNumThreads2D": 1, "Mesh.MaxNumThreads3D": 1,
        "Mesh.RandomSeed": 1, "Mesh.MeshSizeFromPoints": 0, "Mesh.MeshSizeFromCurvature": 0,
        "Mesh.MeshSizeExtendFromBoundary": 0, "Mesh.ElementOrder": 1,
        "Mesh.MeshSizeMin": size_mm, "Mesh.MeshSizeMax": size_mm,
        "Mesh.MshFileVersion": 4.1, "Mesh.Binary": 0, "Mesh.Algorithm3D": 1, "Mesh.SaveAll": 0,
    }.items():
        gmsh.option.setNumber(key, val)
    if junction_sphere:
        # Resolve the spherical interface more finely without globally refining the caps.
        local_mm = 0.5 * size_mm
        field = gmsh.model.mesh.field
        rule = field.add("MathEval")
        field.setString(rule, "F",
                        f"Min({size_mm}, {local_mm}+0.8*Abs(Sqrt(x*x+y*y+z*z)-{JUNCTION_RADIUS_MM}))")
        field.setAsBackgroundMesh(rule)
        gmsh.option.setNumber("Mesh.MeshSizeMin", local_mm)
    gmsh.model.mesh.generate(3)
    npoints = len(gmsh.model.mesh.getNodes()[0])
    ncells = sum(len(tags) for tags in gmsh.model.mesh.getElements(3)[1])
    if not (0 < npoints < 1024 and 0 < ncells < 8000):
        raise ValueError(f"Generated mesh exceeds example limits: {npoints} nodes, {ncells} tetrahedra")
    gmsh.write(str(path))
    return volume


def export_bundle(path, cad_volume, size_mm, junction_sphere=False):
    """Adapt only this generator's named meshes, not arbitrary MSH input."""
    try:
        import meshio
    except ImportError as e:
        raise ValueError("Conversion requires meshio==5.3.5") from e
    import numpy as np
    from fibernet.io.tet4_meshio import Tet4Bundle, write_tet4_vtu_bundle
    from fibernet.sim.tet4 import inspect_static_tet4_admission

    try:
        mesh = meshio.read(path, file_format="gmsh")
    except meshio.ReadError as e:
        raise ValueError(f"Could not read generated mesh {path.name}: {e}") from e
    tets, tris = (mesh.get_cells_type(kind) for kind in ("tetra", "triangle"))
    if set(mesh.cells_dict) != {"tetra", "triangle"} or not (
        0 < len(mesh.points) < 1024 and 0 < len(tets) < 8000
    ):
        raise ValueError("The example requires a bounded linear tetra/triangle mesh")
    groups = mesh.cell_sets_dict
    labels = {"bulk": 1, "junction": 2} if junction_sphere else {"solid": 1}
    kinds = {**{name: (3, "tetra") for name in labels},
             **{name: (2, "triangle") for name in BOUNDARIES}}
    if set(mesh.field_data) != set(kinds) or any(
        int(mesh.field_data[name][1]) != dim or kind not in groups.get(name, {})
        for name, (dim, kind) in kinds.items()
    ):
        raise ValueError("The generated MSH lost its expected named physical groups")
    material_ids = np.zeros(len(tets), dtype=np.uint64)
    covered = np.zeros(len(tets), dtype=int)
    for name, material_id in labels.items():
        rows = groups[name]["tetra"]
        if not len(rows):
            raise ValueError(f"Empty named material region: {name}")
        np.add.at(covered, rows, 1)
        material_ids[rows] = material_id
    if not np.all(covered == 1):
        raise ValueError("Named material regions must cover every tetrahedron exactly once")
    p = mesh.points
    volumes = np.linalg.det(p[tets[:, 1:]] - p[tets[:, :1]]) / 6.
    if not np.all(volumes > 0) or not np.array_equal(np.unique(tets), np.arange(len(p))):
        raise ValueError("Generated tetrahedra must have positive volume and use all points")
    mesh_volume = float(volumes.sum())
    if not 0.9 * cad_volume < mesh_volume < 1.001 * cad_volume:
        raise ValueError("Faceted volume differs too much from the exact CAD union")
    owners = Counter(face for cell in tets for face in combinations(sorted(map(int, cell)), 3))
    exterior = {f for f, n in owners.items() if n == 1}
    if set(owners.values()) != {1, 2}:
        raise ValueError("Unexpected tetrahedral face incidence")
    seen = Counter()
    boundary = {}
    for name in BOUNDARIES:
        faces = sorted(tuple(sorted(map(int, f))) for f in tris[groups[name]["triangle"]])
        if not faces:
            raise ValueError(f"Empty generated boundary: {name}")
        if name in CAPS:
            axis, plane = CAPS[name]
            if not np.allclose(p[np.array(faces), axis], plane, rtol=0, atol=1e-10):
                raise ValueError(f"Misplaced cap group: {name}")
        seen.update(faces)
        boundary[name] = faces
    if (set(seen) != exterior or set(seen.values()) != {1}
            or len(tris) != len(exterior)):
        raise ValueError("Named groups must partition the exterior exactly once")

    interface = set()
    if junction_sphere:
        shell = Counter(face for tet in tets[material_ids == labels["junction"]]
                        for face in combinations(sorted(map(int, tet)), 3))
        interface = {face for face, n in shell.items() if n == 1}
        if not interface or any(owners[face] != 2 for face in interface):
            raise ValueError("Junction must have an internal conformal interface")
        edges = Counter(edge for face in interface for edge in combinations(face, 2))
        vertices = sorted({v for face in interface for v in face})
        if set(edges.values()) != {2} or len(vertices)-len(edges)+len(interface) != 2:
            raise ValueError("Material interface is not a closed sphere shell")
        if not np.allclose(np.linalg.norm(p[vertices], axis=1), JUNCTION_RADIUS_MM, rtol=0., atol=1e-10):
            raise ValueError("Interface vertices do not match the CAD sphere")
        junction_volume = float(volumes[material_ids == labels["junction"]].sum())
        if not .85 * JUNCTION_VOLUME_MM3 < junction_volume < 1.001 * JUNCTION_VOLUME_MM3:
            raise ValueError("Faceted junction volume differs too much from the CAD sphere")

    # Bundle node/cell IDs are row-based; never confuse them with Gmsh tags.
    node_ids = np.arange(len(p), dtype=np.uint64)
    cell_ids = np.arange(len(tets), dtype=np.uint64)
    props = {i: {"young_modulus_pa": 1e9, "poisson_ratio": .25} for i in labels.values()}
    bundle = Tet4Bundle(p, tets, node_ids, cell_ids, material_ids, props, "mm")
    holds = {node: ("x", "y", "z") for face in boundary["cap_neg_x"] for node in face}
    plan = inspect_static_tet4_admission(bundle, fixed_dofs_by_id=holds)
    if not plan.within_factor_budget:
        raise ValueError("Example mesh exceeds the structural factor budget")
    write_tet4_vtu_bundle(path.with_suffix(".zip"), points=bundle.points,
                          tetrahedra=bundle.tetrahedra, node_ids=bundle.node_ids,
                          cell_ids=bundle.cell_ids, material_ids=bundle.material_ids,
                          materials=bundle.materials, length_unit=bundle.length_unit)
    summary = {
        "mesh_size_mm": size_mm, "length_unit": bundle.length_unit,
        "nodes": len(p), "tetrahedra": len(tets),
        "cad_volume_mm3": cad_volume, "mesh_volume_mm3": mesh_volume,
        "exterior_triangles": len(exterior), "static_factor_entries": plan.factor_entries,
        "bundle_sha256": sha256(path.with_suffix(".zip").read_bytes()).hexdigest(),
        # Strings preserve the native bundle's uint64 node-ID representation.
        "boundary_node_ids": {name: [[str(i) for i in f] for f in faces]
                              for name, faces in boundary.items()},
    }
    if junction_sphere:
        summary.update(material_id_by_name={k: str(v) for k, v in labels.items()},
                       junction_radius_mm=JUNCTION_RADIUS_MM,
                       mesh_junction_volume_mm3=junction_volume,
                       material_interface_triangles=len(interface))
        boundary["material_interface"] = sorted(interface)
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    # Compare physical triangles rather than forge a difference by renumbering rows.
    return {name: {tuple(sorted(tuple(v) for v in p[list(f)].tolist())) for f in faces}
            for name, faces in boundary.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory (must not exist)")
    parser.add_argument("--junction-sphere", action="store_true",
                        help="fragment a second material with a centered 0.30 mm sphere")
    args = parser.parse_args(argv)
    cases = SPHERE_CASES if args.junction_sphere else CASES
    out = args.output_dir
    if out.exists() or out.is_symlink() or not out.parent.is_dir():
        parser.error("--output-dir must be new and its parent must already exist")
    try:
        import gmsh
    except (ImportError, OSError) as e:
        parser.error(f"Gmsh unavailable: install gmsh==4.13.1 and its platform runtime ({e})")
    try:
        from fibernet.io import tet4_meshio  # standard FiberNet module; meshio is lazy here
    except (ImportError, OSError) as e:
        parser.error(f"Install FiberNet >=4.21.0 ({e})")
    created_output = False
    try:
        with TemporaryDirectory(prefix=".fused-cad-", dir=out.parent) as tmp:
            gmsh.initialize(readConfigFiles=False)
            shapes = []
            try:
                gmsh.option.setNumber("General.Terminal", 0)
                for name, h in cases:
                    path = Path(tmp) / f"{name}.msh"
                    volume = generate(gmsh, path, h, args.junction_sphere)
                    shapes.append(export_bundle(path, volume, h, args.junction_sphere))
            finally:
                gmsh.finalize()
            if any(shapes[0][name] == shapes[1][name] for name in shapes[0]):
                raise ValueError("Expected distinct triangulations for every boundary/interface")
            staged = [Path(tmp) / f"{name}.{ext}" for name, _ in cases
                      for ext in ("msh", "zip", "json")]
            if not all(path.is_file() for path in staged):
                raise ValueError("Expected all six staged output files")
            out.mkdir(exist_ok=False)
            created_output = True
            for path in staged:
                shutil.move(path, out / path.name)
        # Empty marker, created exclusively after all six files are in place.
        (out / "COMPLETE").touch(exist_ok=False)
    except (OSError, ValueError) as e:
        if created_output:
            parser.exit(1, f"{parser.prog}: incomplete output at {out}: {e}\n"
                        "Do not use files without COMPLETE. Retry with a new output directory.\n")
        parser.exit(1, f"{parser.prog}: {e}\n")
    print(f"Wrote two independent named meshes, bundles, and boundary manifests to {out}")


if __name__ == "__main__":
    main()
