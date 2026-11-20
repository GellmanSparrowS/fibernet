"""VTU numeric fields do not silently lose integer-valued external labels."""
import numpy as np
import pytest

from fibernet.io.meshio_adapter import to_meshio, write_vtu_bundle
from test_meshio_id_fields import fixture


def kwargs(location, field, value, *, by_id):
    return {
        location + '_data' + ('_by_id' if by_id else ''): {field: value},
        'field_metadata': {location: {field: {'units': '1', 'meaning': 'external label'}}},
        'length_unit': 'mm',
    }


@pytest.mark.parametrize('by_id', [False, True])
@pytest.mark.parametrize('boxed', [False, True])
def test_unsigned_exact_by_physical_edge_id(tmp_path, by_id, boxed):
    meshio = pytest.importorskip('meshio')
    graph = fixture()
    tags = [np.uint64(2**63+1), np.int64(42)]
    if boxed:
        tags = [np.array(value) for value in tags]
    value = {11: tags[1], 24: tags[0]} if by_id else tags
    target = write_vtu_bundle(tmp_path/'model', graph, **kwargs('cell', 'external_tag', value, by_id=by_id))
    mesh = meshio.read(target/'network.vtu')
    assert mesh.cell_data_dict['edge_id']['line'].tolist() == [24, 11]
    assert mesh.cell_data_dict['external_tag']['line'].dtype.kind == 'u'
    assert mesh.cell_data_dict['external_tag']['line'].tolist() == [2**63+1, 42]


def test_signed_exact_vectors_keep_point_rows():
    pytest.importorskip('meshio')
    graph = fixture()
    # Includes ndarray rows of different integer widths, not just scalars.
    values = {103: np.array([2, 3, 4], dtype=np.uint32),
              7: np.array([2**53+1, -1, 0], dtype=np.int64),
              29: np.array([5, 6, 7], dtype=np.uint64)}
    mesh,_ = to_meshio(graph, **kwargs('point', 'external_vector', values, by_id=True))
    assert mesh.point_data['node_id'].tolist() == [7, 29, 103]
    assert mesh.point_data['external_vector'].dtype.kind == 'i'
    assert mesh.point_data['external_vector'].tolist() == [[2**53+1,-1,0],[5,6,7],[2,3,4]]


@pytest.mark.parametrize('tags', [
    [np.uint64(2**63+1), np.int64(-1)],
    [2**53+1, 0.25],
    [np.uint64(2**64-1), 0.25],
])
@pytest.mark.parametrize('by_id', [False, True])
def test_unrepresentable_fields_reject_before_publication(tmp_path, tags, by_id):
    graph = fixture()
    value = {24: tags[0], 11: tags[1]} if by_id else tags
    with pytest.raises(ValueError, match='exact|signed or unsigned'):
        write_vtu_bundle(tmp_path/'model', graph, **kwargs('cell', 'external_tag', value, by_id=by_id))
    assert not list(tmp_path.iterdir())


def test_existing_explicit_array_dtype_and_exact_mixed_float_unchanged():
    pytest.importorskip('meshio')
    graph = fixture()
    for raw, dtype in [(np.array([1,2],dtype=np.int16), np.int16), ([2**53,0.5], np.float64)]:
        mesh,_ = to_meshio(graph, **kwargs('cell','data',raw,by_id=False))
        arr = mesh.cell_data_dict['data']['line']
        assert arr.dtype == dtype and arr.tolist() == list(raw)


def test_small_signed_unsigned_and_ordinary_floats_still_export():
    pytest.importorskip('meshio')
    graph = fixture()
    for raw, want in [([np.uint64(3), np.int64(-1)], [3,-1]),
                      ([np.float64(0.125), np.float64(0.25)], [0.125,0.25])]:
        mesh, _ = to_meshio(graph, **kwargs('cell','data',raw,by_id=False))
        assert mesh.cell_data_dict['data']['line'].tolist() == want


def test_incomplete_ids_and_nan_are_not_missing_value_fills(tmp_path):
    graph = fixture()
    for tags, error in [({24:np.uint64(2**63+1)}, 'exact ID coverage'),
                        ({24:np.uint64(2**63), 11:float('nan')}, 'must be finite'),
                        ({24:np.uint64(2**63+1), 11:float('nan')}, 'represent this integer exactly')]:
        with pytest.raises(ValueError, match=error):
            write_vtu_bundle(tmp_path/'model', graph, **kwargs('cell','data',tags,by_id=True))
        assert not list(tmp_path.iterdir())
