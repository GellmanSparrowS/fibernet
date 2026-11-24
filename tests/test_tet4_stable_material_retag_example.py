"""Example-level mesh identity and stable material selection checks."""
from dataclasses import replace

import numpy as np
import pytest

from examples.tet4_stable_material_retag import mesh_identity, reassign_cells
from fibernet.io.tet4_meshio import Tet4Bundle
from fibernet.sim.tet4 import inspect_static_tet4_admission

TOP = 2**64 - 1
RED = {'young_modulus_pa': 3e9, 'poisson_ratio': .25}


def bundle():
    return Tet4Bundle(
        [[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[0.,0.,-1.]],
        [[0,1,2,3],[0,2,1,4]], [0,7,2**53+3,9,TOP], [0,TOP],
        [0,2], {0:{'young_modulus_pa':1e9,'poisson_ratio':.25},
                2:{'young_modulus_pa':2e9,'poisson_ratio':.25}}, 'mm')


def test_row_permutations_keep_selection_and_new_material_prunes_only_unused():
    b = bundle()
    n = [4,2,0,1,3]
    c = [1,0]
    p = replace(b, points=np.asarray(b.points)[n], node_ids=[b.node_ids[i] for i in n],
                tetrahedra=np.argsort(n)[np.asarray(b.tetrahedra)[c]],
                cell_ids=[b.cell_ids[i] for i in c], material_ids=np.asarray(b.material_ids)[c])
    signature = mesh_identity(b)
    assert mesh_identity(p) == signature
    options = dict(expected_mesh_identity=signature, cell_ids=[TOP], material_id=TOP, material=RED)
    a, z = (reassign_cells(x, **options) for x in (b, p))
    np.testing.assert_array_equal(a.material_ids, np.array([0, TOP], dtype=np.uint64))
    np.testing.assert_array_equal(z.material_ids, np.array([TOP, 0], dtype=np.uint64))
    assert z.materials == {0:b.materials[0], TOP:RED}
    assert b.material_ids == [0,2] and set(b.materials) == {0,2}
    for name in ('points','tetrahedra','node_ids','cell_ids'):
        ref = np.array(getattr(p, name), dtype=getattr(z,name).dtype)
        np.testing.assert_array_equal(getattr(z,name),ref)
        assert not getattr(z,name).flags.writeable
    fixed = dict.fromkeys(b.node_ids[:3], ('x','y','z'))
    assert inspect_static_tet4_admission(a, fixed_dofs_by_id=fixed) == inspect_static_tet4_admission(z, fixed_dofs_by_id=fixed)


def test_existing_zero_target_keeps_definition():
    b = bundle()
    kw = dict(expected_mesh_identity=mesh_identity(b), cell_ids=[TOP], material_id=0)
    out = reassign_cells(b, **kw)
    assert out.material_ids.tolist() == [0,0] and out.materials == {0:b.materials[0]}
    with pytest.raises(ValueError, match='existing target'):
        reassign_cells(b, **kw, material=RED)


@pytest.mark.parametrize('ids', [[], [0,0], [7], [False], [0.], [2**64], [-1],
                                 np.ma.array([0], mask=False)])
def test_invalid_selection_fails(ids):
    b = bundle()
    with pytest.raises(ValueError):
        reassign_cells(b, expected_mesh_identity=mesh_identity(b), cell_ids=ids,
                       material_id=7, material=RED)


def test_changed_mesh_and_bad_material_table_reject():
    b = bundle()
    kw = dict(expected_mesh_identity=mesh_identity(b), cell_ids=[0], material_id=7, material=RED)
    points=np.array(b.points)
    points[0,0]=1e-9
    with pytest.raises(ValueError, match='mesh identity mismatch'):
        reassign_cells(replace(b, points=points), **kw)
    with pytest.raises(ValueError, match='referenced material'):
        reassign_cells(replace(b, materials={**b.materials,9:RED}), **kw)


def test_exact_binary64_admission_before_selection_or_fingerprint():
    b = bundle()
    kw = dict(expected_mesh_identity=mesh_identity(b), cell_ids=[0], material_id=7)
    with pytest.raises(ValueError, match='exactly representable'):
        reassign_cells(b, **kw, material={'young_modulus_pa':2**53+1,'poisson_ratio':.25})
    if np.finfo(np.longdouble).nmant > np.finfo(float).nmant:
        points = np.array(b.points, dtype=np.longdouble)
        points[1,0] = np.nextafter(np.longdouble(1),np.longdouble(2))
        with pytest.raises(ValueError, match='exactly representable'):
            mesh_identity(replace(b, points=points))
