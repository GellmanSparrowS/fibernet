"""Legacy public MSH export preserves representable geometric distinctions."""
import numpy as np
import pytest

from fibernet import Fiber, FiberNetwork, api
from fibernet.io import to_gmsh


def _mesh(path):
    words = path.read_text()
    def rows(name):
        header, *body = words.split(f'${name}\n')[1].split(f'$End{name}\n')[0].splitlines()
        assert int(header) == len(body)
        return [row.split() for row in body]
    return (
        {int(key): tuple(map(float, pos)) for key, *pos in rows('Nodes')},
        [(int(row[4]), int(row[5]), int(row[6])) for row in rows('Elements')],
    )


def _export(tmp_path, points):
    source = FiberNetwork(fibers=[Fiber(centerline=p, fiber_id=label)
                                  for label, p in zip([81, 17], points)])
    before = source.to_dict()
    target = tmp_path / 'sample.msh'
    assert api.export(source, target, segments_per_fiber=1) == target
    assert source.to_dict() == before
    return target


def test_nearby_distinct_sparse_label_rods_do_not_become_one_line(tmp_path):
    target = _export(tmp_path, [[[0., 0., 0.], [1., 0., 0.]],
                                [[0., 4e-9, 0.], [1., 4e-9, 0.]]])
    assert _mesh(target) == (
        {1: (0., 0., 0.), 2: (1., 0., 0.), 3: (0., 4e-9, 0.), 4: (1., 4e-9, 0.)},
        [(1, 1, 2), (2, 3, 4)],
    )


def test_short_nonzero_segment_does_not_become_self_loop_or_round_off(tmp_path):
    end = 1. + 4e-12
    target = _export(tmp_path, [[[1., 0., 0.], [end, 0., 0.]]])
    assert _mesh(target) == ({1: (1., 0., 0.), 2: (end, 0., 0.)}, [(1, 1, 2)])


def test_exact_coincident_endpoint_still_shares_node_after_resampling(tmp_path):
    target = _export(tmp_path, [[[0., 0., 0.], [.25, .4, .08], [.3, .12, .09]],
                                [[.3, .12, .09], [.6, .15, .1]]])
    assert _mesh(target) == (
        {1: (0., 0., 0.), 2: (.3, .12, .09), 3: (.6, .15, .1)},
        [(1, 1, 2), (2, 2, 3)],
    )


def test_real_meshio_roundtrip_keeps_disjoint_rod_rows(tmp_path):
    meshio = pytest.importorskip('meshio')
    target = _export(tmp_path, [[[0., 0., 0.], [1., 0., 0.]],
                                [[0., 4e-9, 0.], [1., 4e-9, 0.]]])
    mesh = meshio.read(target)
    np.testing.assert_array_equal(mesh.points, [[0., 0., 0.], [1., 0., 0.],
                                                 [0., 4e-9, 0.], [1., 4e-9, 0.]])
    np.testing.assert_array_equal(mesh.cells_dict['line'], [[0, 1], [2, 3]])
    np.testing.assert_array_equal(mesh.cell_data_dict['gmsh:physical']['line'], [1, 1])
    np.testing.assert_array_equal(mesh.cell_data_dict['gmsh:geometrical']['line'], [1, 2])


def test_unrepresentable_resampled_segment_rejected_before_write(tmp_path):
    # The midpoint of adjacent float64 numbers rounds to an endpoint.
    end = np.nextafter(1., 2.)
    source = FiberNetwork(fibers=[Fiber([[1., 0., 0.], [end, 0., 0.]])])
    target = tmp_path / 'keep.msh'
    target.write_bytes(b'keep')
    with pytest.raises(ValueError, match='zero-length'):
        to_gmsh(source, target, segments_per_fiber=2)
    assert target.read_bytes() == b'keep'


def test_far_offset_nodes_with_distinct_tags_keep_distinct_coordinates(tmp_path):
    offset, end = 1e6, 1e6 + 4e-7
    target = _export(tmp_path, [[[offset, 0., 0.], [end, 0., 0.]]])
    # Old welding pass gives two tags here, but %.10e serializes identical x.
    assert _mesh(target) == ({1: (offset, 0., 0.), 2: (end, 0., 0.)}, [(1, 1, 2)])


def test_exact_sampled_interior_intersection_keeps_legacy_shared_node(tmp_path):
    source = FiberNetwork(fibers=[
        Fiber([[-1., 0., 0.], [1., 0., 0.]], fiber_id=81),
        Fiber([[0., -1., 0.], [0., 1., 0.]], fiber_id=17),
    ])
    target = tmp_path / 'crossing.msh'
    to_gmsh(source, target, segments_per_fiber=2)
    assert _mesh(target) == (
        {1: (-1., 0., 0.), 2: (0., 0., 0.), 3: (1., 0., 0.),
         4: (0., -1., 0.), 5: (0., 1., 0.)},
        [(1, 1, 2), (1, 2, 3), (2, 4, 2), (2, 2, 5)],
    )
