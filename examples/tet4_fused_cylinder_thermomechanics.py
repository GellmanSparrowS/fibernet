"""One-way electrothermal to elastic handoff for the tagged CAD example.

Equal-power reports retain the axial cap node mean and add the facet-area
mean of axial displacement (both in metres), with cap area in square metres.
"""
import argparse
from collections import Counter
from hashlib import sha256
from itertools import combinations
import json
from math import fsum, sqrt
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from fibernet.io.tet4_meshio import read_tet4_vtu_bundle
from fibernet.sim.tet4_dc import solve_dc_tet4
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


def load_named_mesh(bundle_path, manifest_path, compare_junction=False):
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
    if not (0 < len(ids) < 1024 and 0 < len(mesh.tetrahedra) < 8000):
        raise ValueError("Input exceeds the CAD example's mesh budget")
    if compare_junction:
        named = meta.get("material_id_by_name")
        if (not isinstance(named, dict) or set(named) != {"bulk", "junction"}
                or any(type(v) is not str for v in named.values())):
            raise ValueError("Comparison requires a generated bulk/junction material map")
        by_id = {str(i): i for i in mesh.materials}
        if (len(by_id) != 2 or set(named.values()) != set(by_id)
                or set(map(int, mesh.material_ids)) != set(mesh.materials)
                or meta.get("junction_radius_mm") != .30):
            raise ValueError("Expected two distinct used material IDs and a 0.30 mm junction sphere")
        material = {name: by_id[value] for name, value in named.items()}
        if mesh.materials[material["bulk"]] != mesh.materials[material["junction"]]:
            raise ValueError("This comparison holds elastic properties equal in both regions")
    else:
        material = next(iter(mesh.materials)) if len(mesh.materials) == 1 else None
        if material is None or set(map(int, mesh.material_ids)) != {material}:
            raise ValueError("Expected one homogeneous solid unless comparison is requested")
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
    if compare_junction:
        shell = Counter(face for tet in mesh.tetrahedra[mesh.material_ids == material["junction"]]
                        for face in combinations(sorted(map(int, tet)), 3))
        interface = {f for f, n in shell.items() if n == 1}
        edges = Counter(edge for face in interface for edge in combinations(face, 2))
        vertices = sorted({v for face in interface for v in face})
        if (not interface or interface & exterior or set(edges.values()) != {2}
                or len(vertices) - len(edges) + len(interface) != 2
                or not np.allclose(np.linalg.norm(mesh.points[vertices], axis=1), .30, rtol=0., atol=1e-10)):
            raise ValueError("Named junction is not the expected closed interior sphere")
    return mesh, faces, material, meta["bundle_sha256"]


def cap_area_mean_x(points_m, cap_faces, displacement_m):
    """Integrate the linear axial displacement over the named terminal faces."""
    triangles = points_m[cap_faces]
    area = .5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    if not len(area) or not np.all(np.isfinite(area) & (area > 0.)):
        raise ValueError("Terminal cap requires finite positive facet areas")
    total_area = fsum(area)
    face_mean = displacement_m[cap_faces, 0].mean(axis=1)
    return fsum(area * face_mean) / total_area, total_area


def solve(mesh, faces, material, digest, supports, *, conductivity=None, voltage=VOLTAGE_V):
    if conductivity is None:
        conductivity = {material: 2.}
    left, right = (np.unique(faces[k]) for k in (LEFT, RIGHT))
    cap_rows = np.r_[left, right]
    if len(np.unique(cap_rows)) != len(cap_rows):
        raise ValueError("Electrical caps must not share nodes")
    nid = mesh.node_ids
    potentials = {**dict.fromkeys(map(int, nid[left]), 0.),
                  **dict.fromkeys(map(int, nid[right]), voltage)}
    robin = np.concatenate([faces[k] for k in sorted(faces) if k not in (LEFT, RIGHT)])
    coupled = solve_dc_thermal_tet4_with_convection(
        mesh, conductivity_S_per_m_by_material=conductivity,
        electrode_voltage_V_by_id=potentials,
        conductivity_W_per_m_K_by_material=dict.fromkeys(conductivity, .25),
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
        stress_free_temperature_K=CAP_K, alpha_per_K_by_material=dict.fromkeys(conductivity, 1e-4),
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
    electrode = voltage * fsum(dc.electrode_current_into_solid_A[right])
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
    result = {
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
    if len(conductivity) > 1:
        result.pop("solid_material_id")
        result["voltage_V"] = voltage
        cap_mean, cap_area = cap_area_mean_x(p_m, faces[RIGHT], elastic.displacement_m)
        result["right_cap_area_weighted_mean_x_displacement_m"] = cap_mean
        result["right_cap_area_m2"] = cap_area
        result["joule_W_by_material_id"] = {
            str(i): fsum(dc.joule_power_W_by_cell[mesh.material_ids == i]) for i in conductivity
        }
    return result


def compare(mesh, faces, materials, digest, supports):
    bulk, junction = (materials[name] for name in ("bulk", "junction"))
    baseline = solve(mesh, faces, bulk, digest, supports, conductivity={bulk: 2., junction: 2.})
    changed = {bulk: 2., junction: .5}
    left, right = (mesh.node_ids[np.unique(faces[k])] for k in (LEFT, RIGHT))
    potentials = {**dict.fromkeys(map(int, left), 0.),
                  **dict.fromkeys(map(int, right), VOLTAGE_V)}
    pilot = solve_dc_tet4(mesh, conductivity_S_per_m_by_material=changed,
                        electrode_voltage_V_by_id=potentials)
    power_at_fixed_voltage = fsum(pilot.joule_power_W_by_cell)
    if not 0. < power_at_fixed_voltage < baseline["joule_W"]:
        raise ValueError("Expected reduced total Joule power at fixed voltage in this example")
    # Fixed conductivities make electrical power quadratic in applied voltage.
    voltage = VOLTAGE_V * sqrt(baseline["joule_W"] / power_at_fixed_voltage)
    matched = solve(mesh, faces, bulk, digest, supports, conductivity=changed, voltage=voltage)
    np.testing.assert_allclose(matched["joule_W"], baseline["joule_W"], rtol=1e-12, atol=0.)
    return {
        "material_id_by_name": {name: str(i) for name, i in materials.items()},
        "lower_junction_joule_at_0_3_V_W": power_at_fixed_voltage,
        "uniform_conductivity": baseline,
        "lower_junction_equal_power": matched,
    }


def compare_junction_cte(mesh, faces, materials, digest):
    """Compare free thermal expansion with and without a junction CTE mismatch."""
    points = mesh.points * 1e-3
    ids = list(map(int, mesh.node_ids))
    cap = np.unique(faces[LEFT])
    if len(cap) < 3 or not np.all(points[cap, 0] == points[cap[0], 0]):
        raise ValueError("CTE comparison needs three noncollinear nodes on one exact x plane")
    low = min(cap, key=lambda i: (points[i, 2], ids[i]))
    high = min(cap, key=lambda i: (-points[i, 2], ids[i]))
    a, b = sorted((low, high), key=lambda i: ids[i])
    yz = points[cap, 1:] - points[a, 1:]
    dy, dz = points[b, 1:] - points[a, 1:]
    areas = abs(dy * yz[:, 1] - dz * yz[:, 0])
    c = cap[min(range(len(cap)), key=lambda i: (-areas[i], ids[cap[i]]))]
    if dz == 0. or float(areas.max()) == 0.:
        raise ValueError("CTE comparison needs three noncollinear nodes on one exact x plane")

    # These six Cartesian restraints fix only the rigid-body reference. The
    # one nonzero gauge value keeps homogeneous free expansion admissible.
    alpha_bulk, reference_K, delta_K = 10e-6, 300., 10.
    eps_bulk = alpha_bulk * delta_K
    held = {ids[a]: ("x", "y", "z"), ids[b]: ("x",), ids[c]: ("x",)}
    gauge_y_m = float(eps_bulk * dy)
    prescribed = {ids[b]: {"y": gauge_y_m}}
    thermal = dict.fromkeys(ids, reference_K + delta_K)
    cells = points[mesh.tetrahedra]
    volume = abs(np.linalg.det(cells[:, 1:] - cells[:, :1])) / 6.
    affine_displacement = eps_bulk * (points - points[a])
    bulk, junction = materials["bulk"], materials["junction"]
    cases = {}
    for label, alpha_junction in (("uniform_cte", alpha_bulk), ("junction_3x_cte", 3 * alpha_bulk)):
        result = solve_prescribed_temperature_tet4(
            mesh, temperature_K_by_id=thermal, stress_free_temperature_K=reference_K,
            alpha_per_K_by_material={bulk: alpha_bulk, junction: alpha_junction},
            forces_by_id={}, fixed_dofs_by_id=held,
            prescribed_displacements_m_by_id=prescribed,
        )
        np.testing.assert_array_equal(result.node_ids, mesh.node_ids)
        np.testing.assert_array_equal(result.cell_ids, mesh.cell_ids)
        np.testing.assert_array_equal(result.material_ids, mesh.material_ids)
        if np.count_nonzero(result.fixed_mask) != 6:
            raise ValueError("CTE comparison expected exactly six held Cartesian degrees of freedom")
        np.testing.assert_allclose(result.reaction_N, 0., rtol=0., atol=2e-9)
        force = result.reaction_N.sum(axis=0)
        moment = np.cross(points, result.reaction_N).sum(axis=0)
        np.testing.assert_allclose(force, 0., rtol=0., atol=2e-9)
        np.testing.assert_allclose(moment, 0., rtol=0., atol=2e-12)
        pairing = fsum((result.reaction_N * result.displacement_m).ravel())
        np.testing.assert_allclose(pairing, 0., rtol=0., atol=2e-18)
        np.testing.assert_allclose(pairing, result.terminal_mechanical_pairing_J,
                                   rtol=1e-8, atol=1e-25)
        energy = result.elastic_energy_J
        departure = float(np.max(abs(result.displacement_m - affine_displacement)))
        if label == "uniform_cte":
            if not (departure <= 2e-15 and energy <= 1e-20):
                raise ValueError("Equal CTE did not reproduce stress-free affine expansion")
            np.testing.assert_allclose(result.mean_stress_Pa, 0., rtol=0., atol=1e-3)
        else:
            if not energy > 1e-13:
                raise ValueError("CTE mismatch did not produce the expected nonzero elastic energy")
            np.testing.assert_allclose(2 * energy, pairing - result.stress_eigenstrain_integral_J,
                                       rtol=2e-8, atol=1e-20)
        hydrostatic = np.trace(result.mean_stress_Pa, axis1=1, axis2=2) / 3.
        cases[label] = {
            "junction_alpha_per_K": alpha_junction,
            "elastic_energy_J": energy,
            "volume_weighted_hydrostatic_mean_Pa": {
                name: float(np.average(hydrostatic[mesh.material_ids == mid],
                                       weights=volume[mesh.material_ids == mid]))
                for name, mid in materials.items()
            },
            "max_departure_from_bulk_affine_expansion_m": departure,
            "max_abs_support_reaction_N": float(np.max(abs(result.reaction_N))),
            "net_force_N": force.tolist(), "net_moment_N_m": moment.tolist(),
            "terminal_mechanical_pairing_J": pairing,
        }
    return {
        "bundle_sha256": digest, "nodes": len(ids), "tetrahedra": len(volume),
        "stress_free_temperature_K": reference_K, "uniform_temperature_K": reference_K + delta_K,
        "bulk_alpha_per_K": alpha_bulk, "external_mechanical_force": False,
        "gauge": {"A_xyz": str(ids[a]), "B_x_and_y": str(ids[b]), "C_x": str(ids[c]),
                  "B_y_prescribed_m": gauge_y_m},
        **cases,
    }


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("bundle", type=Path)
    cli.add_argument("boundary_manifest", type=Path)
    cli.add_argument("--supports", choices=("left-clamped", "both-clamped"))
    comparisons = cli.add_mutually_exclusive_group()
    comparisons.add_argument("--compare-junction-conductivity", action="store_true")
    comparisons.add_argument("--compare-junction-cte", action="store_true",
                             help="compare uniform heating with equal and unequal expansion coefficients")
    a = cli.parse_args(argv)
    if a.compare_junction_cte and a.supports is not None:
        cli.error("--compare-junction-cte uses its own free-expansion gauge, not --supports")
    try:
        load = load_named_mesh(a.bundle, a.boundary_manifest,
                               a.compare_junction_conductivity or a.compare_junction_cte)
        if a.compare_junction_cte:
            report = compare_junction_cte(*load)
        else:
            runner = compare if a.compare_junction_conductivity else solve
            report = runner(*load, a.supports or "left-clamped")
        print(json.dumps(report, indent=2, allow_nan=False))
    except (ValueError, OSError, ImportError, AssertionError) as e:
        cli.exit(1, f"{cli.prog}: {e}\n")


if __name__ == "__main__":
    main()
