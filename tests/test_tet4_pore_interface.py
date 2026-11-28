from dataclasses import replace

import numpy as np
import pytest

from fibernet.examples.tet4_periodic_pore import _scaffold_voxels, _voxel_bundle, bundle
from fibernet.sim import tet4_pore_interface as transfer


def fixtures():
    return _voxel_bundle(_scaffold_voxels(), elastic_modulus=2.e9), bundle()


@pytest.mark.parametrize("first_id", [None, 2**53 + 1, 2**63])
def test_prepared_wall_reuses_one_factor_and_returns_detached_geometry(monkeypatch, first_id):
    solid, pore = fixtures()
    if first_id is not None:
        tags = np.arange(len(solid.points), dtype=np.uint64) + np.uint64(first_id)
        solid = replace(solid, node_ids=tags)
    calls = []
    sparse_lu = transfer.splu

    def counted(system):
        calls.append(system.shape)
        return sparse_lu(system)

    monkeypatch.setattr(transfer, 'splu', counted)
    wall = transfer.prepare_conforming_pore_wall(solid, pore)
    u = solid.points * np.array([2e-6, -1e-6, -1e-6])
    p = wall.deform(u, node_ids=solid.node_ids)
    order = np.arange(len(u))[::-1]
    q = wall.deform(-u[order], node_ids=solid.node_ids[order])
    assert len(calls) == 1
    np.testing.assert_allclose(p.points + q.points, 2*pore.points, atol=4e-15, rtol=0)

    surface = np.unique(transfer._faces(solid.tetrahedra))
    locator = {tuple(row): i for i, row in enumerate(pore.points)}
    closest = [locator[tuple(solid.points[j])] for j in surface]
    np.testing.assert_array_equal(p.points[closest], pore.points[closest]+u[surface]*1e3)
    outside = np.any((pore.points == pore.points.min(axis=0)) |
                     (pore.points == pore.points.max(axis=0)), axis=1)
    np.testing.assert_array_equal(p.points[outside], pore.points[outside])
    assert np.count_nonzero(np.linalg.norm(p.points-pore.points, axis=1)) > len(surface)

    p.points[:] = -1
    p.node_ids[:] = 1
    p.materials[1]['young_modulus_pa'] = 1
    z = wall.deform(np.zeros_like(u), node_ids=solid.node_ids)
    np.testing.assert_array_equal(z.points, pore.points)
    np.testing.assert_array_equal(z.node_ids, pore.node_ids)
    assert z.materials == pore.materials
    assert len(calls) == 1


def test_prepared_wall_uses_declared_mesh_units_and_exact_node_identity():
    solid, pore = fixtures()
    tags = np.arange(len(solid.node_ids), dtype=np.uint64)+np.uint64(2**63+123)
    solid = replace(solid, node_ids=tags)
    motion_m = solid.points * np.array([1e-7, -2e-7, 1e-7])
    w_mm = transfer.prepare_conforming_pore_wall(solid, pore)
    y_millimeter = w_mm.deform(motion_m, node_ids=tags)
    w_m = transfer.prepare_conforming_pore_wall(
        replace(solid, points=solid.points*1e-3, length_unit='m'),
        replace(pore, points=pore.points*1e-3, length_unit='m'),
    )
    y_meter = w_m.deform(motion_m, node_ids=tags)
    np.testing.assert_allclose(y_millimeter.points*1e-3, y_meter.points, atol=2e-18, rtol=0)
    wrong = tags.copy()
    wrong[-1] += np.uint64(1)
    with pytest.raises(ValueError, match='node IDs must match'):
        w_mm.deform(motion_m, node_ids=wrong)
    with pytest.raises(ValueError, match='unique'):
        w_mm.deform(motion_m, node_ids=np.repeat(tags[0], len(tags)))


def test_prepared_wall_fails_on_nonconforming_or_unusable_geometry():
    solid, pore = fixtures()
    with pytest.raises(ValueError, match='same length unit'):
        transfer.prepare_conforming_pore_wall(solid, replace(pore, length_unit='m'))
    with pytest.raises(ValueError, match='no matching pore point'):
        transfer.prepare_conforming_pore_wall(replace(solid, points=solid.points+.013), pore)
    with pytest.raises(ValueError, match='distinct coordinates'):
        transfer.prepare_conforming_pore_wall(solid, replace(
            pore, points=np.vstack((pore.points, pore.points[0])),
            node_ids=np.r_[pore.node_ids, pore.node_ids.max()+1],
        ))
    wall = transfer.prepare_conforming_pore_wall(solid, pore)
    with pytest.raises(ValueError, match='finite'):
        wall.deform(np.full_like(solid.points, np.nan), node_ids=solid.node_ids)
    with pytest.raises(ValueError, match='reverses a cell|collapsed'):
        wall.deform(np.ones_like(solid.points)*.1, node_ids=solid.node_ids)


def test_prepared_wall_is_independent_of_bundle_row_order_and_later_edits():
    solid, pore = fixtures()
    rng = np.random.default_rng(62)

    def permute(b, ids_offset=0):
        order = rng.permutation(len(b.points))
        take = np.argsort(order)
        cells = rng.permutation(len(b.tetrahedra))
        return replace(b, points=b.points[order], node_ids=b.node_ids[order]+ids_offset,
                       tetrahedra=take[b.tetrahedra[cells]], cell_ids=b.cell_ids[cells],
                       material_ids=b.material_ids[cells])

    s = permute(solid, ids_offset=1_000_000)
    p = permute(pore)
    values = s.points * np.array([3e-7,-2e-7,1e-7])
    plan = transfer.prepare_conforming_pore_wall(s, p)
    first = plan.deform(values, node_ids=s.node_ids)
    ids = s.node_ids.copy()
    ref_points = first.points.copy()
    s.points[:, :] += 20.
    p.points[:, :] += 10.
    np.testing.assert_array_equal(plan.deform(values, node_ids=ids).points, ref_points)

    simple = transfer.prepare_conforming_pore_wall(solid, pore).deform(
        solid.points * np.array([3e-7,-2e-7,1e-7]), node_ids=solid.node_ids
    )
    indexing = {int(key): i for i,key in enumerate(first.node_ids)}
    np.testing.assert_allclose(first.points[[indexing[i] for i in simple.node_ids]],
                               simple.points, atol=2e-15, rtol=0)


@pytest.mark.parametrize('side', ['solid', 'pore'])
def test_prepared_wall_rejects_duplicate_unordered_tetrahedra(side):
    solid, pore = fixtures()

    def duplicate(mesh):
        return replace(mesh, tetrahedra=np.vstack([mesh.tetrahedra, mesh.tetrahedra[0, ::-1]]),
                       cell_ids=np.r_[mesh.cell_ids, mesh.cell_ids.max()+1],
                       material_ids=np.r_[mesh.material_ids, mesh.material_ids[0]])

    a, b = (duplicate(solid), pore) if side == 'solid' else (solid, duplicate(pore))
    with pytest.raises(ValueError, match='duplicate tetrahedra'):
        transfer.prepare_conforming_pore_wall(a, b)
