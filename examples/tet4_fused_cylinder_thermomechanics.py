"""One-way electrothermal to elastic handoff for the tagged homogeneous CAD example."""
import argparse
from collections import Counter
from hashlib import sha256
from itertools import combinations
import json
from math import fsum
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from fibernet.io.tet4_meshio import read_tet4_vtu_bundle
from fibernet.sim.tet4_dc_thermal import solve_dc_thermal_tet4_with_convection
from fibernet.sim.tet4_thermoelastic import solve_prescribed_temperature_tet4

CAPS = {f"cap_{side}_{axis}": (i, sign)
        for i, axis in enumerate("xyz") for side, sign in (("neg", -1), ("pos", 1))}
LEFT, RIGHT = "cap_neg_x", "cap_pos_x"
VOLTAGE_V, CAP_K, AMBIENT_K, FILM = .3, 300., 299.95, 100.


def snapshot(path, limit):
    # Read one handle once. The exact bytes checked are those later parsed.
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"{path.name} exceeds this example's input size limit")
    return data


def load_named_mesh(bundle_path, manifest_path):
    def unique_object(items):
        obj = dict(items)
        if len(obj) != len(items):
            raise ValueError("Duplicate keys in boundary manifest")
        return obj

    meta = json.loads(snapshot(manifest_path, 2 * 1024**2), object_pairs_hook=unique_object)
    data = snapshot(bundle_path, 8 * 1024**2)
    if not isinstance(meta, dict) or sha256(data).hexdigest() != meta.get("bundle_sha256"):
        raise ValueError("Bundle SHA-256 does not match the boundary manifest")
    with TemporaryDirectory(prefix="fiber-cad-input-") as folder:
        archive = Path(folder) / "input.zip"
        archive.write_bytes(data)
        mesh = read_tet4_vtu_bundle(archive)

    if mesh.length_unit != "mm" or meta.get("length_unit") != "mm":
        raise ValueError("The bundle and manifest must declare coordinates in mm")
    ids = [int(i) for i in mesh.node_ids]
    lookup = {str(i): row for row, i in enumerate(ids)}
    material = next(iter(mesh.materials)) if len(mesh.materials) == 1 else None
    if not (0 < len(ids) < 1024 and 0 < len(mesh.tetrahedra) < 8000
            and material is not None and set(map(int, mesh.material_ids)) == {material}):
        raise ValueError("Expected one homogeneous solid in the CAD example's mesh budget")
    groups = meta.get("boundary_node_ids")
    if not isinstance(groups, dict) or set(groups) != {*CAPS, "ambient_side"}:
        raise ValueError("Expected all six named cap groups and ambient_side")
    counts = Counter(face for tet in mesh.tetrahedra
                     for face in combinations(sorted(map(int, tet)), 3))
    exterior = {f for f, n in counts.items() if n == 1}
    found = Counter()
    faces = {}
    for name, triangles in groups.items():
        if not isinstance(triangles, list) or not triangles:
            raise ValueError(f"Missing triangles in {name}")
        rows = []
        for tri in triangles:
            if not isinstance(tri, list) or len(tri) != 3 or any(
                type(i) is not str or i not in lookup for i in tri
            ):
                raise ValueError(f"Expected three stable decimal-string node IDs in {name}")
            face = tuple(sorted(lookup[i] for i in tri))
            rows.append(face)
            found[face] += 1
        faces[name] = np.array(rows)
        if name in CAPS:
            axis, sign = CAPS[name]
            plane = (np.min if sign < 0 else np.max)(mesh.points[:, axis])
            expected = {face for face in exterior if np.allclose(
                mesh.points[list(face), axis], plane, rtol=0., atol=1e-10)}
            if set(rows) != expected:
                raise ValueError(f"Missing or misplaced triangles in cap {name}")
    if set(counts.values()) != {1, 2} or set(found) != exterior or set(found.values()) != {1}:
        raise ValueError("Named boundary faces must partition the exterior exactly once")
    return mesh, faces, material, meta["bundle_sha256"]


def solve(mesh, faces, material, digest, supports):
    left, right = (np.unique(faces[k]) for k in (LEFT, RIGHT))
    cap_rows = np.r_[left, right]
    if len(np.unique(cap_rows)) != len(cap_rows):
        raise ValueError("Electrical caps must not share nodes")
    nid = mesh.node_ids
    potentials = {**dict.fromkeys(map(int, nid[left]), 0.),
                  **dict.fromkeys(map(int, nid[right]), VOLTAGE_V)}
    robin = np.concatenate([faces[k] for k in sorted(faces) if k not in (LEFT, RIGHT)])
    coupled = solve_dc_thermal_tet4_with_convection(
        mesh, conductivity_S_per_m_by_material={material: 2.},
        electrode_voltage_V_by_id=potentials,
        conductivity_W_per_m_K_by_material={material: .25},
        prescribed_K_by_id=dict.fromkeys(potentials, CAP_K),
        faces_by_node_id=nid[robin], convection_W_per_m2_K=FILM, ambient_K=AMBIENT_K,
    )
    dc, thermal = coupled.electrical, coupled.thermal
    np.testing.assert_array_equal(dc.node_ids, nid)
    np.testing.assert_array_equal(thermal.node_ids, nid)
    np.testing.assert_array_equal(dc.cell_ids, mesh.cell_ids)
    np.testing.assert_array_equal(thermal.cell_ids, mesh.cell_ids)
    np.testing.assert_array_equal(dc.joule_power_W_by_cell, thermal.integrated_source_W_by_cell)
    holds = left if supports == "left-clamped" else cap_rows
    elastic = solve_prescribed_temperature_tet4(
        mesh, temperature_K_by_id=dict(zip(map(int, thermal.node_ids), map(float, thermal.temperature_K))),
        stress_free_temperature_K=CAP_K, alpha_per_K_by_material={material: 1e-4},
        forces_by_id={}, fixed_dofs_by_id=dict.fromkeys(map(int, nid[holds]), ("x", "y", "z")),
    )
    np.testing.assert_array_equal(elastic.node_ids, nid)
    np.testing.assert_array_equal(elastic.cell_ids, mesh.cell_ids)
    np.testing.assert_array_equal(elastic.material_ids, mesh.material_ids)
    p_m = mesh.points * 1e-3
    tri = p_m[robin]
    area = .5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    film_out = FILM * fsum(area * np.mean(thermal.temperature_K[robin] - AMBIENT_K, axis=1))
    joule = fsum(dc.joule_power_W_by_cell)
    electrode = VOLTAGE_V * fsum(dc.electrode_current_into_solid_A[right])
    cap_out = -fsum(thermal.heat_into_solid_W[cap_rows])
    thermal_closure = joule - film_out - cap_out
    np.testing.assert_allclose(electrode, joule, rtol=1e-9, atol=1e-17)
    np.testing.assert_allclose(film_out, -thermal.total_convective_heat_into_solid_W, rtol=1e-9, atol=1e-17)
    np.testing.assert_allclose(film_out + cap_out, joule, rtol=1e-9, atol=1e-17)
    force = elastic.reaction_N.sum(axis=0)
    moment = np.cross(p_m, elastic.reaction_N).sum(axis=0)
    np.testing.assert_allclose(force, 0., rtol=0., atol=1e-10)
    np.testing.assert_allclose(moment, 0., rtol=0., atol=1e-13)
    # Held displacements and applied external loads are both zero; elastic
    # energy need not be zero under this nonuniform temperature field.
    assert elastic.terminal_mechanical_pairing_J == 0.
    # The zero-displacement field is an admissible trial for either clamp.
    cells = p_m[mesh.tetrahedra]
    volume = np.abs(np.linalg.det(cells[:, 1:] - cells[:, :1])) / 6.
    delta = thermal.temperature_K[mesh.tetrahedra] - CAP_K
    mean = delta.mean(axis=1)
    variance = np.sum((delta - mean[:, None])**2, axis=1) / 20.
    props = mesh.materials[material]
    bulk_modulus = props["young_modulus_pa"] / (3. * (1. - 2. * props["poisson_ratio"]))
    upper = 4.5 * bulk_modulus * 1e-8 * fsum(volume * (mean**2 + variance))
    assert 0. <= elastic.elastic_energy_J <= upper * (1. + 1e-9)
    if supports == "left-clamped":
        np.testing.assert_array_equal(elastic.reaction_N[right], 0.)
    else:
        np.testing.assert_array_equal(elastic.displacement_m[right], 0.)
    return {
        "bundle_sha256": digest, "solid_material_id": str(material),
        "supports": supports, "nodes": len(nid), "tetrahedra": len(mesh.cell_ids),
        "electrode_power_W": electrode, "joule_W": joule, "robin_heat_out_W": film_out,
        "x_caps_heat_out_W": cap_out, "thermal_residual_W": thermal_closure,
        "temperature_range_K": [float(thermal.temperature_K.min()), float(thermal.temperature_K.max())],
        "right_cap_mean_x_displacement_m": float(elastic.displacement_m[right, 0].mean()),
        "right_cap_x_reaction_N": float(elastic.reaction_N[right, 0].sum()),
        "net_force_N": force.tolist(), "net_moment_N_m": moment.tolist(),
        "elastic_energy_J": elastic.elastic_energy_J, "zero_motion_trial_energy_J": upper,
        "terminal_mechanical_pairing_J": elastic.terminal_mechanical_pairing_J,
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("bundle", type=Path)
    cli.add_argument("boundary_manifest", type=Path)
    cli.add_argument("--supports", choices=("left-clamped", "both-clamped"), default="left-clamped")
    a = cli.parse_args()
    try:
        print(json.dumps(solve(*load_named_mesh(a.bundle, a.boundary_manifest), a.supports),
                         indent=2, allow_nan=False))
    except (ValueError, OSError, ImportError, AssertionError) as e:
        cli.exit(1, f"{cli.prog}: {e}\n")


if __name__ == "__main__":
    main()
