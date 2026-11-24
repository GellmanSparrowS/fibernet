"""Local Tet4 region-shell diagnostic, not a slicer or an intersection test."""

from collections import defaultdict
from math import fsum, isclose
from pathlib import Path

import numpy as np

from fibernet.io.tet4_meshio import read_tet4_vtu_bundle


def oriented_shells(bundle):
    """Return outward triangles (row indices), grouped by stable material ID."""
    p = bundle.points
    owners = defaultdict(list)
    if len(bundle.tetrahedra) != len(bundle.material_ids):
        raise ValueError("Expected one material ID per tetrahedron")
    for tet, mid in zip(bundle.tetrahedra, bundle.material_ids):
        for i in range(4):
            a, b, c = map(int, (*tet[:i], *tet[i + 1:]))
            side = np.dot(np.cross(p[b] - p[a], p[c] - p[a]), p[tet[i]] - p[a])
            if not np.isfinite(side) or side == 0:
                raise ValueError("Cannot orient a degenerate tetrahedron")
            key = tuple(sorted((a, b, c)))
            owners[key].append((int(mid), (a, c, b) if side > 0 else (a, b, c)))
    shells = defaultdict(list)
    for records in owners.values():
        if len(records) > 2:
            raise ValueError("Tet4 face has nonconforming owners")
        if len(records) == 2:
            (_, f), (_, g) = records
            if set(zip(f, f[1:] + f[:1])) == set(zip(g, g[1:] + g[:1])):
                raise ValueError("Tet4 face has owners on the same side")
            if records[0][0] == records[1][0]:
                continue
        for mid, face in records:
            shells[mid].append(face)
    return shells


def shell_volume(points, faces):
    edges = {}
    links = defaultdict(dict)
    for i, (a, b, c) in enumerate(faces):
        for x, y, z in ((a, b, c), (b, c, a), (c, a, b)):
            if (x, y) in edges or y in links[x]:
                raise ValueError("Material shell has a duplicated oriented edge")
            edges[x, y] = i
            links[x][y] = z
    if any((y, x) not in edges for x, y in edges):
        raise ValueError("Material shell has an open edge")
    for link in links.values():
        start = current = next(iter(link))
        seen = set()
        while current not in seen:
            seen.add(current)
            current = link[current]
        if current != start or len(seen) != len(link):
            raise ValueError("Material shell has a nonmanifold vertex")
    # Recenter each edge-connected component separately; a remote shell
    # would still suffer cancellation if all used the same origin.
    face_rows = np.asarray(faces)
    pending = set(range(len(faces)))
    contributions = []
    while pending:
        seed = pending.pop()
        todo, component = [seed], [seed]
        while todo:
            x, y, z = faces[todo.pop()]
            for edge in ((y, x), (z, y), (x, z)):
                other = edges[edge]
                if other in pending:
                    pending.remove(other)
                    todo.append(other)
                    component.append(other)
        centre = points[faces[seed][0]]
        a, b, c = (points[face_rows[component]] - centre).transpose(1, 0, 2)
        contributions.append(fsum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6)
    # Negative cavity components subtract from the enclosing shell.
    return fsum(contributions)


def check(bundle):
    """Check local topology and volume agreement in the declared length unit cubed."""
    if len(bundle.points) > 1024 or len(bundle.tetrahedra) > 8000:
        raise ValueError("Mesh exceeds this example's diagnostic budget")
    shells = oriented_shells(bundle)
    xyz = bundle.points[bundle.tetrahedra]
    volume = np.abs(np.linalg.det(xyz[:, 1:] - xyz[:, :1])) / 6
    report = {}
    for mid, faces in sorted(shells.items()):
        tv = fsum(volume[bundle.material_ids == mid])
        sv = shell_volume(bundle.points, faces)
        if not isclose(sv, tv, rel_tol=1e-10) or not tv > 0:
            raise ValueError(f"Material {mid} signed shell and Tet4 volumes disagree")
        report[str(mid)] = {"faces": len(faces), "signed_shell_volume": sv, "tet4_volume": tv}
    if set(report) != set(map(str, bundle.material_ids)):
        raise ValueError("A used material has no shell")
    return {"volume_unit": bundle.length_unit + "^3", "materials": report}


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="trusted local Tet4 ZIP (requires Meshio)")
    args = parser.parse_args()
    try:
        report = check(read_tet4_vtu_bundle(args.bundle))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))
