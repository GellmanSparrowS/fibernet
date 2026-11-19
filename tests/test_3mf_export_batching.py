"""3MF record batching preserves XML and atomic export behavior."""

from contextlib import contextmanager
import stat
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pytest

from fibernet.gen.solid_export import FiberSolid, PrintSettings, export_solid


MODEL = '3D/3dmodel.model'
NS = 'http://schemas.microsoft.com/3dmanufacturing/core/2015/02'


def _closed_shells(count):
    tetrahedron = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    vertices = np.concatenate([tetrahedron + [4 * i, 0., 0.] for i in range(count)])
    vertices[0, 0] = -0.
    vertices[-1, 2] = np.nextafter(1., 2.)
    faces = np.concatenate([
        [[i, i+2, i+1], [i, i+1, i+3], [i, i+3, i+2], [i+1, i+2, i+3]]
        for i in range(0, len(vertices), 4)
    ])
    return FiberSolid(vertices, faces, np.zeros((0, 3)), .1, PrintSettings())


def _observe_writes(monkeypatch, observe):
    original = zipfile.ZipFile.open

    @contextmanager
    def observed(stream):
        class Writer:
            def write(self, data):
                observe(data)
                return stream.write(data)

        with stream:
            yield Writer()

    def open_entry(archive, name, mode='r', *args, **kwargs):
        stream = original(archive, name, mode, *args, **kwargs)
        return observed(stream) if name == MODEL and mode == 'w' else stream

    monkeypatch.setattr(zipfile.ZipFile, 'open', open_entry)


def test_batched_xml_preserves_all_member_bytes_and_record_order(tmp_path, monkeypatch):
    solid = _closed_shells(257)
    writes = []
    _observe_writes(monkeypatch, writes.append)
    path = tmp_path / 'mesh.3mf'
    export_solid(path, solid)
    assert [block.count(b'<vertex ') for block in writes if block.startswith(b'<vertex ')] == [1024, 4]
    assert [block.count(b'<triangle ') for block in writes if block.startswith(b'<triangle ')] == [1024, 4]

    with zipfile.ZipFile(path) as archive:
        assert archive.namelist() == ['[Content_Types].xml', '_rels/.rels', MODEL]
        assert archive.read('[Content_Types].xml') == (
            b'<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            b'<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            b'<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
            b'</Types>'
        )
        assert archive.read('_rels/.rels') == (
            b'<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            b'<Relationship Target="/3D/3dmodel.model" Id="rel0" '
            b'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>'
        )
        model = archive.read(MODEL)

    vertices = ''.join(
        f'<vertex x="{x:.17g}" y="{y:.17g}" z="{z:.17g}"/>' for x, y, z in solid.vertices
    )
    faces = ''.join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in solid.faces)
    assert model == (
        f'<?xml version="1.0" encoding="UTF-8"?><model unit="millimeter" xml:lang="en-US" xmlns="{NS}">'
        f'<resources><object id="1" type="model"><mesh><vertices>{vertices}</vertices>'
        f'<triangles>{faces}</triangles></mesh></object></resources><build><item objectid="1"/></build></model>'
    ).encode()
    root = ET.fromstring(model)
    assert root.attrib['unit'] == 'millimeter'
    read_vertices = np.array([[float(p.attrib[k]) for k in ('x', 'y', 'z')]
                              for p in root.iter(f'{{{NS}}}vertex')])
    read_faces = [[int(f.attrib[k]) for k in ('v1', 'v2', 'v3')] for f in root.iter(f'{{{NS}}}triangle')]
    np.testing.assert_array_equal(read_vertices.view(np.uint64), solid.vertices.view(np.uint64))
    np.testing.assert_array_equal(read_faces, solid.faces)


def test_mutated_invalid_geometry_preserves_existing_3mf(tmp_path, monkeypatch):
    path = tmp_path / 'mesh.3mf'
    solid = _closed_shells(1)
    export_solid(path, solid)
    before = path.read_bytes()

    def no_write(data):
        pytest.fail('invalid geometry reached the staged XML writer')

    _observe_writes(monkeypatch, no_write)
    solid.faces[0, 1] = solid.faces[0, 0]
    with pytest.raises(ValueError, match='three distinct vertices'):
        export_solid(path, solid)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_failed_geometry_write_preserves_existing_file_and_mode(tmp_path, monkeypatch):
    path = tmp_path / 'mesh.3mf'
    path.write_bytes(b'original')
    mode = stat.S_IMODE(path.stat().st_mode)
    writes = []

    def interrupt(data):
        if data.startswith(b'<triangle '):
            raise OSError('injected batched model write failure')
        writes.append(data)

    _observe_writes(monkeypatch, interrupt)
    with pytest.raises(OSError, match='injected batched model write failure'):
        export_solid(path, _closed_shells(1))
    assert any(block.startswith(b'<vertex ') for block in writes)
    assert path.read_bytes() == b'original'
    assert stat.S_IMODE(path.stat().st_mode) == mode
    assert list(tmp_path.iterdir()) == [path]
