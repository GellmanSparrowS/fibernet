"""Provider-free checks for the tagged Tet4 comparison's CLI input controls."""
from collections import Counter
from hashlib import sha256
import importlib.util
from itertools import combinations, permutations, product
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from fibernet.io.tet4_meshio import Tet4Bundle


SPEC = importlib.util.spec_from_file_location(
    "fused_thermomechanics_example",
    Path(__file__).resolve().parents[1] / "examples" / "tet4_fused_cylinder_thermomechanics.py",
)
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)


def test_area_weighted_cap_integrates_affine_values_on_nonuniform_triangles():
    points = np.array([[0., 0., 0.], [0., 1., 0.],
                       [0., 0., 1.], [0., 0., -4.]])
    triangles = np.array([[0, 1, 2], [0, 1, 3]])
    # P1 u_x=z: triangle areas 1/2 and 2, means 1/3 and -4/3.
    displacement = np.zeros_like(points)
    displacement[:, 0] = points[:, 2]
    mean, area = example.cap_area_mean_x(points, triangles, displacement)
    assert (mean, area) == pytest.approx((-1., 2.5))
    assert mean != pytest.approx(displacement[:, 0].mean())
    assert example.cap_area_mean_x(points, triangles[::-1, ::-1], displacement) == pytest.approx((mean, area))
    with pytest.raises(ValueError, match="positive facet areas"):
        example.cap_area_mean_x(points, np.array([[0, 1, 1]]), displacement)


@pytest.fixture
def tagged_pair(tmp_path):
    # Toy conforming 3x3x3 brick partition. The middle brick is a cube
    # inscribed in the declared sphere, not a generated Gmsh mesh.
    inner = .30 / np.sqrt(3.)
    axis = [-1.5, -inner, inner, 1.5]
    grid = list(product(range(4), repeat=3))
    p = np.array([[axis[i] for i in node] for node in grid])
    rows = {node: i for i, node in enumerate(grid)}
    # Bevel one exterior corner to produce a non-cap side group.
    p[rows[(3, 3, 3)]] -= .05
    tets, tags = [], []
    for start in product(range(3), repeat=3):
        for order in permutations(range(3)):
            corner = list(start)
            tet = [rows[tuple(corner)]]
            for i in order:
                corner[i] += 1
                tet.append(rows[tuple(corner)])
            if np.linalg.det(p[tet[1:]] - p[tet[0]]) < 0.:
                tet[0], tet[1] = tet[1], tet[0]
            tets.append(tet)
            tags.append(2 if start == (1, 1, 1) else 1)
    node_ids = np.arange(len(p), dtype=np.uint64) + np.uint64(2**63)
    cell_ids = np.arange(len(tets), dtype=np.uint64) + np.uint64(2**63 + 1024)
    props = {i: {"young_modulus_pa": 1e9, "poisson_ratio": .25} for i in (1, 2)}
    mesh = Tet4Bundle(p, np.array(tets), node_ids, cell_ids,
                      np.array(tags, dtype=np.uint64), props, "mm")
    counts = Counter(tuple(sorted(f)) for tet in tets for f in combinations(tet, 3))
    boundary = {name: [] for name in (*example.CAPS, "ambient_side")}
    for face, count in sorted(counts.items()):
        if count != 1:
            continue
        for name, (axis_i, sign) in example.CAPS.items():
            if np.all(p[list(face), axis_i] == sign * 1.5):
                boundary[name].append([str(node_ids[i]) for i in face])
                break
        else:
            boundary["ambient_side"].append([str(node_ids[i]) for i in face])

    raw = b"test-double Tet4 archive bytes"
    archive = tmp_path / "input.zip"
    archive.write_bytes(raw)
    manifest = tmp_path / "input.json"
    payload = dict(bundle_sha256=sha256(raw).hexdigest(), length_unit="mm",
                   boundary_node_ids=boundary, junction_radius_mm=.30,
                   material_id_by_name={"bulk": "1", "junction": "2"})
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return archive, manifest, payload, mesh, raw


def _run_cli(monkeypatch, archive, manifest):
    monkeypatch.setattr(sys, "argv", [
        str(SPEC.origin), str(archive), str(manifest), "--compare-junction-conductivity",
    ])
    example.main()


def test_matching_pair_reaches_comparison_once(tagged_pair, monkeypatch, capsys):
    archive, manifest, payload, mesh, raw = tagged_pair

    def read_snapshot(path):
        assert path != archive and path.read_bytes() == raw
        return mesh

    def compare(actual_mesh, faces, materials, digest, supports):
        assert actual_mesh is mesh
        assert materials == {"bulk": 1, "junction": 2}
        assert digest == payload["bundle_sha256"] and supports == "left-clamped"
        assert set(faces) == set(payload["boundary_node_ids"])
        for group, rows in faces.items():
            # Above-2^53 stable input IDs must map to the actual row triangles.
            actual = [sorted(map(str, mesh.node_ids[face])) for face in rows]
            assert actual == [sorted(face) for face in payload["boundary_node_ids"][group]]
        calls.append(digest)
        return {"accepted_pair": digest}

    calls = []
    monkeypatch.setattr(example, "read_tet4_vtu_bundle", read_snapshot)
    monkeypatch.setattr(example, "compare", compare)
    _run_cli(monkeypatch, archive, manifest)
    out = capsys.readouterr()
    assert calls == [payload["bundle_sha256"]]
    assert out.err == "" and json.loads(out.out) == {"accepted_pair": calls[0]}


def test_cap_both_clamped_comparison_report_and_default_shape(tagged_pair, monkeypatch, capsys):
    archive, manifest, payload, mesh, raw = tagged_pair
    decoded = mesh

    def read_snapshot(path):
        assert path != archive and path.read_bytes() == raw
        return decoded

    monkeypatch.setattr(example, "read_tet4_vtu_bundle", read_snapshot)
    argv = [str(SPEC.origin), str(archive), str(manifest)]
    monkeypatch.setattr(sys, "argv", [*argv, "--compare-junction-conductivity",
                                     "--supports", "both-clamped"])
    example.main()
    output = json.loads(capsys.readouterr().out)
    mean_key = "right_cap_area_weighted_mean_x_displacement_m"
    area_key = "right_cap_area_m2"
    legacy_key = "right_cap_mean_x_displacement_m"
    rows = {str(n): i for i, n in enumerate(mesh.node_ids)}
    expected = 0.
    for face in payload["boundary_node_ids"]["cap_pos_x"]:
        a, b, c = mesh.points[[rows[n] for n in face], 1:] * 1e-3
        ab, ac = b - a, c - a
        expected += abs(ab[0] * ac[1] - ab[1] * ac[0]) / 2
    assert expected > 0.
    base, changed = (output[k] for k in ("uniform_conductivity", "lower_junction_equal_power"))
    assert base[area_key] == changed[area_key] == pytest.approx(expected)
    assert base["joule_W"] == pytest.approx(changed["joule_W"], rel=1e-12, abs=0.)
    for state in (base, changed):
        assert state["supports"] == "both-clamped"
        assert state[mean_key] == state[legacy_key] == 0.

    decoded = Tet4Bundle(mesh.points, mesh.tetrahedra, mesh.node_ids, mesh.cell_ids,
                         np.ones_like(mesh.material_ids), {1: mesh.materials[1]}, "mm")
    monkeypatch.setattr(sys, "argv", argv)
    example.main()
    default = json.loads(capsys.readouterr().out)
    assert {mean_key, area_key, "voltage_V", "joule_W_by_material_id"}.isdisjoint(default)
    assert set(default) - {"solid_material_id"} == set(base) - {
        mean_key, area_key, "voltage_V", "joule_W_by_material_id",
    }
    assert default["solid_material_id"] == "1"
    assert default["supports"] == "left-clamped" and default["right_cap_x_reaction_N"] == 0.


@pytest.mark.parametrize("mismatch", ["archive", "labels"])
def test_stale_or_mislabeled_pair_does_not_reach_comparison(
    tagged_pair, mismatch, monkeypatch, capsys
):
    archive, manifest, payload, mesh, raw = tagged_pair

    def read_snapshot(path):
        if mismatch == "archive":
            pytest.fail("mismatched bytes reached bundle decoding")
        assert path.read_bytes() == raw
        return mesh

    def forbidden_compare(*args):
        pytest.fail("invalid input reached the native comparison workflow")

    if mismatch == "archive":
        archive.write_bytes(b"a different archive")
        expected = "Bundle SHA-256 does not match the boundary manifest"
    else:
        payload["material_id_by_name"] = {"bulk": "2", "junction": "1"}
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        expected = "Named junction is not the expected closed interior sphere"

    monkeypatch.setattr(example, "read_tet4_vtu_bundle", read_snapshot)
    monkeypatch.setattr(example, "compare", forbidden_compare)
    with pytest.raises(SystemExit) as failure:
        _run_cli(monkeypatch, archive, manifest)
    out = capsys.readouterr()
    assert failure.value.code == 1
    assert not out.out and expected in out.err
