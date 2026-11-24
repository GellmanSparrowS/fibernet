from dataclasses import replace
from itertools import permutations, product

import numpy as np
import pytest

from examples.tet4_material_shell_check import check
from fibernet.io.tet4_meshio import Tet4Bundle


def cavity():
    """3³ unit cubes; the centre cube has a different material."""
    points = np.array(list(product(range(4), repeat=3)), dtype=float)
    index = {tuple(row): i for i, row in enumerate(points)}
    rows, labels = [], []
    for xyz in product(range(3), repeat=3):
        for order in permutations(np.eye(3, dtype=int)):
            corners = np.asarray(xyz) + np.vstack((np.zeros(3), np.cumsum(order, axis=0)))
            rows.append([index[tuple(corner)] for corner in corners])
            labels.append(2 if xyz == (1, 1, 1) else 1)
    attributes = {"young_modulus_pa": 1e9, "poisson_ratio": .25}
    return Tet4Bundle(points, np.array(rows), np.arange(len(points)), np.arange(len(rows)),
                      np.array(labels), {1: attributes, 2: attributes}, "mm")


def test_cavity_has_negative_inner_orientation():
    report = check(cavity())
    assert report["volume_unit"] == "mm^3"
    assert report["materials"]["1"]["signed_shell_volume"] == pytest.approx(26)
    assert report["materials"]["2"]["signed_shell_volume"] == pytest.approx(1)


def test_same_side_shared_face_rejected():
    mesh = cavity()
    # Isolate one tetrahedron and duplicate it in cyclic vertex order.
    t = mesh.tetrahedra[0]
    with pytest.raises(ValueError, match="owners on the same side"):
        check(replace(mesh, tetrahedra=np.array([t, t[[1, 2, 0, 3]]]),
                      cell_ids=np.array([1, 2]), material_ids=np.array([1, 1])))


def test_two_shells_touching_only_at_one_vertex_rejected():
    mesh = cavity()
    points = np.vstack((np.zeros(3), np.eye(3), -np.eye(3)))
    with pytest.raises(ValueError, match="nonmanifold vertex"):
        check(replace(mesh, points=points, node_ids=np.arange(7),
                      tetrahedra=np.array([[0, 1, 2, 3], [0, 4, 5, 6]]),
                      cell_ids=np.array([1, 2]), material_ids=np.array([1, 1])))


def test_rejects_missing_material_id():
    mesh = cavity()
    with pytest.raises(ValueError, match="one material ID per tetrahedron"):
        check(replace(mesh, material_ids=mesh.material_ids[:-1]))


def test_disjoint_shells_use_local_origins():
    mesh = cavity()
    first = np.vstack((np.zeros(3), np.eye(3)))
    points = np.vstack((first, first + (100000010, 100000033, 100000041)))
    report = check(replace(mesh, points=points, node_ids=np.arange(8),
                           tetrahedra=np.arange(8).reshape(2, 4),
                           cell_ids=np.array([1, 2]), material_ids=np.array([1, 1])))
    assert report["materials"]["1"]["signed_shell_volume"] == pytest.approx(1 / 3)
