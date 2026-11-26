"""Optional v1 Tet4 bundles: synthetic geometry and actual Meshio 5.x."""
import base64
import hashlib
import json
import subprocess
import sys
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

from fibernet.io import tet4_meshio as tet4


def fixture(scale=1.0):
    big = 2**64 - 1
    # Construct the exact coordinate words; multiplication may flush subnormals.
    step = int(np.asarray(scale, dtype=np.float64).view(np.uint64))
    negative = step | (1 << 63)
    return dict(points=np.array([[0, 0, 0], [step, 0, 0], [0, step, 0],
                                 [0, 0, step], [0, 0, negative]], dtype=np.uint64).view(np.float64),
                # Deliberately different winding; no permutation during I/O.
                tetrahedra=[[1, 0, 2, 3], [0, 2, 1, 4]],
                node_ids=[np.int64(709), np.uint64(big), 17, 7, 3],
                cell_ids=[np.uint64(2**63 + 1), np.int64(20)],
                material_ids=[37, np.uint64(big)],
                materials={big: dict(young_modulus_pa=2.4e9, poisson_ratio=.33),
                           37: dict(young_modulus_pa=1e9, poisson_ratio=.25)},
                length_unit='mm')


@pytest.mark.parametrize('scale', [1.0, 1e-200, np.uint64(1).view(np.float64), 1e200])
def test_real_tet4_interchange_no_unit_or_id_change(tmp_path, scale):
    meshio = pytest.importorskip('meshio')
    data = fixture(scale)
    path = tet4.write_tet4_vtu_bundle(tmp_path / 'test.zip', **data)
    with ZipFile(path) as bundle:
        assert set(bundle.namelist()) == {'mesh.vtu', 'manifest.json'}
        ordinary_vtu = bundle.read('mesh.vtu')
        metadata = json.loads(bundle.read('manifest.json'))
    assert metadata == {'schema': 'fibernet-tet4-vtu-1', 'length_unit': 'mm',
                        'mesh_sha256': hashlib.sha256(ordinary_vtu).hexdigest(),
                        'materials': {str(k): v for k, v in data['materials'].items()}}
    internal = tmp_path / 'mesh.vtu'
    internal.write_bytes(ordinary_vtu)
    public = meshio.read(internal)
    np.testing.assert_array_equal(public.points.view(np.uint64), data['points'].view(np.uint64))
    assert len(public.cells) == 1 and public.cells[0].type == 'tetra'
    np.testing.assert_array_equal(public.cells[0].data, data['tetrahedra'])
    assert public.cell_data['cell_id'][0].tolist() == [2**63 + 1, 20]
    assert public.cell_data['material_id'][0].tolist() == [37, 2**64 - 1]
    assert public.point_data['node_id'].tolist() == [709, 2**64 - 1, 17, 7, 3]
    reread = tet4.read_tet4_vtu_bundle(path)
    np.testing.assert_array_equal(reread.points.view(np.uint64), data['points'].view(np.uint64))
    np.testing.assert_array_equal(reread.tetrahedra, data['tetrahedra'])
    assert reread.materials == data['materials']
    assert reread.length_unit == 'mm'
    assert reread.node_ids.tolist() == [int(k) for k in data['node_ids']]
    assert reread.cell_ids.tolist() == [int(k) for k in data['cell_ids']]
    assert reread.material_ids.tolist() == [int(k) for k in data['material_ids']]
    target = tet4.write_tet4_vtu_bundle(tmp_path / 'rewritten.zip', **vars(reread))
    with ZipFile(target) as bundle:
        assert bundle.read('mesh.vtu') == ordinary_vtu


def test_int64_ids_and_large_offset_geometry(tmp_path):
    pytest.importorskip('meshio')
    data = fixture()
    data.update(points=2**52 + data['points'], node_ids=[709, 2**63 - 1, 17, 7, 3],
                cell_ids=[2**53 + 1, 20], material_ids=np.array([37, 37], dtype=np.int32),
                materials={37: data['materials'][37]}, length_unit='nm')
    reread = tet4.read_tet4_vtu_bundle(tet4.write_tet4_vtu_bundle(tmp_path / 'small.zip', **data))
    assert [reread.node_ids.dtype, reread.cell_ids.dtype, reread.material_ids.dtype] == [np.int64] * 3
    assert reread.node_ids.tolist() == data['node_ids']
    assert reread.cell_ids.tolist() == data['cell_ids']
    np.testing.assert_array_equal(reread.points, data['points'])
    assert reread.length_unit == 'nm'


@pytest.mark.parametrize('invalid', [
    {'node_ids': [1, -1, 2, 3, 4]}, {'node_ids': [0, 2**64, 2, 3, 4]},
    {'node_ids': [0, 2.0, 2, 3, 4]}, {'node_ids': np.arange(5, dtype=float)},
    {'node_ids': [0, True, 2, 3, 4]}, {'node_ids': [1, 1, 2, 3, 4]},
    {'node_ids': np.ma.array(np.arange(5), mask=False)}, {'cell_ids': [1, 1]},
    {'tetrahedra': [[0, 1, 2, 6], [0, 1, 2, 4]]}, {'tetrahedra': [[0, 1, 2, 2]]},
    {'tetrahedra': [[0, 1, 2, 3.0]]}, {'tetrahedra': [[0, 1, 2, 2**63]]},
    {'material_ids': [37, 999]}, {'materials': {}}, {'length_unit': 'inch'},
    {'points': [[0, 0, 0], [2**53 + 1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]]},
    {'points': [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, -1]]},
    {'points': [[0, 0, 0], [float('inf'), 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]]},
])
def test_invalid_model_refused_before_optional_io(tmp_path, monkeypatch, invalid):
    def no_meshio():
        pytest.fail('invalid model reached the optional writer')

    monkeypatch.setattr(tet4, '_meshio', no_meshio)
    with pytest.raises(ValueError):
        tet4.write_tet4_vtu_bundle(tmp_path / 'bad.zip', **(fixture() | invalid))
    assert list(tmp_path.iterdir()) == []


def test_nonpositive_elastic_modulus_incompressibility_and_unknown_material_properties(tmp_path):
    base = fixture()
    for update in ({'young_modulus_pa': 0}, {'poisson_ratio': .5}, {'density': 1200}):
        base['materials'][37] = {**fixture()['materials'][37], **update}
        with pytest.raises(ValueError):
            tet4.write_tet4_vtu_bundle(tmp_path / 'bad.zip', **base)
    assert list(tmp_path.iterdir()) == []


def test_no_clobber_and_cleanup_on_race_or_backend_failure(tmp_path, monkeypatch):
    meshio = pytest.importorskip('meshio')
    path = tmp_path / 'target.zip'
    path.mkdir()
    with pytest.raises(FileExistsError):
        tet4.write_tet4_vtu_bundle(path, **fixture())
    assert path.is_dir() and not list(path.iterdir())
    path.rmdir()
    link = tet4.os.link

    def raced(src, dst):
        dst.write_bytes(b'another writer')
        link(src, dst)

    with monkeypatch.context() as mp:
        mp.setattr(tet4.os, 'link', raced)
        with pytest.raises(FileExistsError):
            tet4.write_tet4_vtu_bundle(path, **fixture())
    assert path.read_bytes() == b'another writer'
    path.unlink()

    def fail_meshio(vtu, *args, **kwargs):
        vtu.write_bytes(b'partial')
        raise RuntimeError('fake backend write failure')

    monkeypatch.setattr(meshio.vtu, 'write', fail_meshio)
    with pytest.raises(RuntimeError, match='fake backend'):
        tet4.write_tet4_vtu_bundle(path, **fixture())
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('change', [
    'extra member', 'wrong path', 'compressed', 'manifest extra', 'duplicate json key',
    'wrong hash', 'bad unit', 'unknown point field', 'mixed or wrong type', 'unknown cell field',
    'duplicate cell field', 'FieldData', 'missing field', 'multiple pieces', 'document type',
])
def test_reader_refuses_non_v1_content(tmp_path, change):
    meshio = pytest.importorskip('meshio')
    path = tet4.write_tet4_vtu_bundle(tmp_path / 'source.zip', **fixture())
    with ZipFile(path) as src:
        meta, xml = src.read('manifest.json'), src.read('mesh.vtu')
    record, root = json.loads(meta), ET.fromstring(xml)
    grid = root.find('UnstructuredGrid')
    piece = grid.find('Piece')
    if change == 'manifest extra':
        record['future_field'] = 1
    elif change == 'bad unit':
        record['length_unit'] = 'inches'
    elif change == 'unknown point field':
        piece.find('PointData/DataArray').set('Name', 'temperature')
    elif change == 'unknown cell field':
        ET.SubElement(piece.find('Cells'), 'DataArray', Name='extra')
    elif change == 'duplicate cell field':
        child = piece.find('CellData/DataArray')
        piece.find('CellData').append(ET.fromstring(ET.tostring(child)))
    elif change == 'FieldData':
        ET.SubElement(grid, 'FieldData')
    elif change == 'missing field':
        box = piece.find('CellData')
        box.remove(box[0])
    elif change == 'multiple pieces':
        grid.append(ET.fromstring(ET.tostring(piece)))
    elif change == 'mixed or wrong type':
        types = piece.find("Cells/DataArray[@Name='types']")
        buf = bytearray(base64.b64decode(types.text))
        endian = 'little' if root.get('byte_order') == 'LittleEndian' else 'big'
        buf[4:12] = (5).to_bytes(8, endian)
        types.text = base64.b64encode(buf).decode()
    xml = ET.tostring(root)
    if change == 'document type':
        xml = b'<!DOCTYPE VTKFile []>' + xml
    record['mesh_sha256'] = hashlib.sha256(xml).hexdigest() if change != 'wrong hash' else '0' * 64
    meta = json.dumps(record)
    if change == 'duplicate json key':
        meta = meta.replace('{', '{"length_unit": "mm", ', 1)
    bad = tmp_path / 'bad.zip'
    with ZipFile(bad, 'w', compression=ZIP_DEFLATED if change == 'compressed' else 0) as target:
        target.writestr('../mesh.vtu' if change == 'wrong path' else 'mesh.vtu', xml)
        target.writestr('manifest.json', meta)
        if change == 'extra member':
            target.writestr('extra', '')
    with pytest.raises(ValueError):
        tet4.read_tet4_vtu_bundle(bad)
    assert not (tmp_path / 'mesh.vtu').exists()


def test_lazy_import_even_where_meshio_is_installed():
    command = '''
import importlib.abc, sys
class BlockMeshio(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "meshio":
            raise ImportError("deliberate missing optional dependency")
sys.meta_path.insert(0, BlockMeshio())
import fibernet
from fibernet.io.tet4_meshio import read_tet4_vtu_bundle
assert "meshio" not in sys.modules
try:
    read_tet4_vtu_bundle("unused.zip")
except ImportError as exc:
    assert "Optional Tet4" in str(exc)
else:
    raise RuntimeError("missing dependency silently accepted")
'''
    subprocess.run([sys.executable, '-c', command], check=True)


@pytest.mark.parametrize(('key', 'label'), [('points', 'point rows'), ('tetrahedra', 'tetrahedra rows')])
def test_writer_checks_row_limits_before_iterating_or_staging(tmp_path, monkeypatch, key, label):
    class OversizedRows:
        def __len__(self):
            return 250_001

        def __iter__(self):
            pytest.fail('over-limit rows were expanded')

    def no_staging(*args, **kwargs):
        pytest.fail('over-limit model reached the filesystem')

    monkeypatch.setattr(tet4, 'TemporaryDirectory', no_staging)
    with pytest.raises(ValueError, match=label):
        tet4.write_tet4_vtu_bundle(tmp_path / 'oversize.zip', **(fixture() | {key: OversizedRows()}))


@pytest.mark.parametrize(('limit', 'cap', 'label'), [
    ('_MAX_POINTS', 4, 'point rows'), ('_MAX_TETS', 1, 'tetrahedra rows'),
])
def test_reader_checks_header_row_limits_before_meshio(tmp_path, monkeypatch, limit, cap, label):
    meshio = pytest.importorskip('meshio')
    path = tet4.write_tet4_vtu_bundle(tmp_path / 'small.zip', **fixture())

    def unexpected(*args, **kwargs):
        pytest.fail('over-limit header reached Meshio')

    monkeypatch.setattr(tet4, limit, cap)
    monkeypatch.setattr(meshio.vtu, 'read', unexpected)
    with pytest.raises(ValueError, match=label):
        tet4.read_tet4_vtu_bundle(path)


def test_batched_dyadic_check_keeps_absolute_row_and_tiny_coordinates(monkeypatch):
    data = fixture(np.uint64(1).view(np.float64))
    monkeypatch.setattr(tet4, '_DYADIC_BATCH', 1)
    good = np.asarray(data['tetrahedra'])
    tet4._nondegenerate(data['points'], good)
    with pytest.raises(ValueError, match='row 2 has zero volume'):
        tet4._nondegenerate(data['points'], np.vstack([good, [0, 1, 2, 2]]))


def test_outward_filter_keeps_exact_branch_for_subnormal_and_ambiguous_cells():
    from fractions import Fraction
    unit = np.vstack([np.zeros(3), np.eye(3)])
    assert tet4._nonzero_interval_certificate(unit[None]).tolist() == [True]
    subnormal_axes = unit.view(np.uint64).copy()
    subnormal_axes[subnormal_axes != 0] = 1
    small = subnormal_axes.view(np.float64)
    assert tet4._nonzero_interval_certificate(small[None]).tolist() == [False]
    tet4._nondegenerate(small, np.array([[0, 1, 2, 3]]))

    # Exactly integer coordinates with a translated plane and one changed
    # coordinate at its next represented value. All volumes below are
    # independently classified from the original binary64 coordinates.
    for shift in (1, 5, 13, 31, 45, 51):
        b = float(2**shift)
        planar = np.array([[b,b,b],[b+2,b+4,b+6],[b+3,b+6,b+3],[b+5,b+10,b+9]])
        changed = planar.copy()
        changed[-1,1] = np.nextafter(changed[-1,1], np.inf)
        rows = np.stack([planar,changed])
        assert not tet4._nonzero_interval_certificate(rows)[0]
        for i,points in enumerate(rows):
            p,q,r,s = [tuple(map(Fraction, y)) for y in points]
            u,v,w = [tuple(a-b for a,b in zip(k,p)) for k in (q,r,s)]
            cross = (v[1]*w[2]-v[2]*w[1], v[2]*w[0]-v[0]*w[2], v[0]*w[1]-v[1]*w[0])
            result = sum(a*b for a,b in zip(u,cross))
            assert bool(result)==bool(i)
            if result:
                tet4._nondegenerate(points, np.array([[0,1,2,3]]))
            else:
                with pytest.raises(ValueError, match='row 0 has zero volume'):
                    tet4._nondegenerate(points, np.array([[0,1,2,3]]))


def test_outward_filter_fallback_reports_original_row_across_two_batches():
    p=np.vstack([np.zeros(3),np.eye(3)])
    good=np.tile([0,1,2,3],(8200,1))
    combined=np.vstack([good, [0,1,2,2]])
    with pytest.raises(ValueError, match='row 8200 has zero volume'):
        tet4._nondegenerate(p, combined)
