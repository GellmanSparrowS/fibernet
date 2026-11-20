"""Strict optional Tet4 geometry/material interchange, not a solver."""

from __future__ import annotations

import base64
from binascii import Error as Base64Error
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shutil
from struct import unpack
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET
from zipfile import BadZipFile, ZIP_STORED, ZipFile

import numpy as np

from .meshio_adapter import _UNITS, _unmasked_field


_SCHEMA = 'fibernet-tet4-vtu-1'
# 250k rows of each kind are under 32 MiB of base64 data plus XML.
_LIMITS = {'mesh.vtu': 40 * 1024**2, 'manifest.json': 1024**2}
_MAX_POINTS = _MAX_TETS = 250_000
_DYADIC_BATCH = 4_096
_MAX_ARCHIVE = sum(_LIMITS.values()) + 1024
_MAX_ZIP_DIRECTORY = 512
_I64_MAX, _U64_MAX = 2**63 - 1, 2**64 - 1
_MATERIAL_KEYS = {'young_modulus_pa', 'poisson_ratio'}
_MANIFEST_KEYS = {'schema', 'length_unit', 'materials', 'mesh_sha256'}


@dataclass
class Tet4Bundle:
    """Binary64 points, row-index tetrahedra, and stable IDs in stored order."""

    points: np.ndarray
    tetrahedra: np.ndarray
    node_ids: np.ndarray
    cell_ids: np.ndarray
    material_ids: np.ndarray
    materials: dict
    length_unit: str


def _meshio():
    try:
        import meshio
    except ImportError as exc:
        raise ImportError('Optional Tet4 interchange requires meshio; install meshio==5.3.5') from exc
    return meshio


def _integer(value, label):
    if (isinstance(value, (bool, np.bool_, np.timedelta64)) or
            not isinstance(value, (int, np.integer)) or
            not 0 <= int(value) <= _U64_MAX):
        raise ValueError(f'{label} must contain exact nonnegative uint64 identifiers or indices')
    return int(value)


def _row_count(values, label):
    try:
        return len(values)
    except TypeError as exc:
        raise ValueError(f'{label} must be a sized array or sequence') from exc


def _bounded_counts(n, m):
    if not 1 <= n <= _MAX_POINTS:
        raise ValueError(f'Tet4 v1 requires 1 to {_MAX_POINTS:,} point rows')
    if not 1 <= m <= _MAX_TETS:
        raise ValueError(f'Tet4 v1 requires 1 to {_MAX_TETS:,} tetrahedra rows')


def _integers(values, shape, label, signed_only=False, unique=False):
    if _row_count(values, label) != shape[0]:
        raise ValueError(f'{label} must have shape {shape}')
    _unmasked_field(values, label)
    # Do not allow NumPy to infer a dtype for mixed Python/NumPy integers.
    raw = values if isinstance(values, np.ndarray) else np.asarray(values, dtype=object)
    if raw.shape != shape:
        raise ValueError(f'{label} must have shape {shape}')
    if raw.dtype.kind in 'iu' and raw.dtype.itemsize <= 8:
        if raw.size and raw.dtype.kind == 'i' and raw.min() < 0:
            raise ValueError(f'{label} must be nonnegative')
        highest = int(raw.max()) if raw.size else 0
        data = raw
    else:
        data = [_integer(item, label) for item in raw.flat]
        highest = max(data, default=0)
    if signed_only and highest > _I64_MAX:
        raise ValueError(f'{label} must fit int64 row indices')
    array = np.array(data, dtype=np.uint64 if highest > _I64_MAX else np.int64).reshape(shape)
    if unique and len(np.unique(array)) != len(array):
        raise ValueError(f'{label} must be unique')
    return array


def _float64(value, label):
    if (isinstance(value, (bool, np.bool_, np.timedelta64)) or
            not isinstance(value, (float, np.floating, int, np.integer))):
        raise ValueError(f'{label} must contain finite binary64 numbers')
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f'{label} must contain finite binary64 numbers') from exc
    if (not math.isfinite(number) or
            (int(number) != int(value) if isinstance(value, (int, np.integer)) else value != number)):
        raise ValueError(f'{label} must be finite and exactly representable as binary64')
    return number


def _points(values):
    _unmasked_field(values, 'points')
    raw = values if isinstance(values, np.ndarray) else np.asarray(values, dtype=object)
    if raw.ndim != 2 or raw.shape[1] != 3 or not len(raw):
        raise ValueError('points must be a nonempty (N, 3) array')
    if raw.dtype.kind == 'f' and raw.dtype.itemsize <= 8:
        if not np.isfinite(raw).all():
            raise ValueError('points must be finite')
        return raw.astype(np.float64, copy=True)
    return np.array([_float64(item, 'points') for item in raw.flat], dtype=np.float64).reshape(raw.shape)


def _nondegenerate(points, cells):
    # Every binary64 value is a dyadic rational. Integer arithmetic evaluates
    # the zero/nonzero determinant exactly, with no length or volume cutoff.
    # Convert only vertices referenced by this batch and discard its Python
    # integers and tuples before the next batch. Never expand the whole mesh.
    for start in range(0, len(cells), _DYADIC_BATCH):
        batch = cells[start:start + _DYADIC_BATCH]
        references, local = np.unique(batch.ravel(), return_inverse=True)
        dyadics = [tuple(x.as_integer_ratio() for x in row) for row in points[references].tolist()]
        for i, indices in enumerate(local.reshape(batch.shape).tolist(), start):
            vertices = [dyadics[k] for k in indices]
            differences = []
            for axis in zip(*vertices):
                denominator = max(d for _, d in axis)
                a, b, c, d = (n * (denominator // e) for n, e in axis)
                differences.append((b - a, c - a, d - a))
            (a, b, c), (d, e, f), (g, h, j) = differences
            if a * (e * j - f * h) - b * (d * j - f * g) + c * (d * h - e * g) == 0:
                raise ValueError(f'tetrahedra row {i} has zero volume')


def _validate(points, tetrahedra, node_ids, cell_ids, material_ids, materials, length_unit):
    if not isinstance(length_unit, str) or length_unit not in _UNITS:
        raise ValueError('length_unit must be explicitly declared: m, cm, mm, um, nm')
    _bounded_counts(_row_count(points, 'points'), _row_count(tetrahedra, 'tetrahedra'))
    points = _points(points)
    _unmasked_field(tetrahedra, 'tetrahedra')
    raw = tetrahedra if isinstance(tetrahedra, np.ndarray) else np.asarray(tetrahedra, dtype=object)
    if raw.ndim != 2 or raw.shape[1] != 4 or not len(raw):
        raise ValueError('tetrahedra must be a nonempty (M, 4) array of point row indices')
    cells = _integers(raw, raw.shape, 'tetrahedra', signed_only=True)
    if int(cells.max()) >= len(points):
        raise ValueError('tetrahedra must reference existing point rows')
    node_ids = _integers(node_ids, (len(points),), 'node_ids', unique=True)
    cell_ids = _integers(cell_ids, (len(cells),), 'cell_ids', unique=True)
    material_ids = _integers(material_ids, (len(cells),), 'material_ids')
    if not isinstance(materials, Mapping) or len(materials) > len(cells):
        raise ValueError('materials must map each referenced material ID to its properties')
    _unmasked_field(materials, 'materials')
    properties = {}
    for key, value in materials.items():
        mid = _integer(key, 'materials keys')
        if mid in properties or not isinstance(value, Mapping) or set(value) != _MATERIAL_KEYS:
            raise ValueError('a material needs a unique ID, young_modulus_pa and poisson_ratio only')
        modulus = _float64(value['young_modulus_pa'], 'young_modulus_pa')
        nu = _float64(value['poisson_ratio'], 'poisson_ratio')
        if not modulus > 0 or not -1 < nu < 0.5:
            raise ValueError('materials require E > 0 Pa and -1 < poisson_ratio < 0.5')
        properties[mid] = {'young_modulus_pa': modulus, 'poisson_ratio': nu}
    if set(properties) != {int(v) for v in np.unique(material_ids)}:
        raise ValueError('materials must cover exactly the referenced material_ids')
    _nondegenerate(points, cells)
    return Tet4Bundle(points, cells, node_ids, cell_ids, material_ids, properties, length_unit)


def _hash(path):
    digest = sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_tet4_vtu_bundle(path, *, points, tetrahedra, node_ids, cell_ids, material_ids,
                          materials, length_unit):
    """Write a new ZIP containing mesh.vtu and manifest.json, without replacing a destination.

    Requires Meshio and a destination filesystem supporting atomic hard links.
    See docs/TET4_VTU_BUNDLE.md for the strict v1 interchange contract.
    """
    target = Path(path)
    if os.path.lexists(target):
        raise FileExistsError(f'Tet4 bundle destination already exists: {target}')
    data = _validate(points, tetrahedra, node_ids, cell_ids, material_ids, materials, length_unit)
    meshio = _meshio()
    mesh = meshio.Mesh(data.points, [('tetra', data.tetrahedra)],
                       point_data={'node_id': data.node_ids},
                       cell_data={'cell_id': [data.cell_ids], 'material_id': [data.material_ids]})
    with TemporaryDirectory(prefix='.tet4-', dir=target.parent) as staging:
        staging = Path(staging)
        vtu, archive = staging / 'mesh.vtu', staging / 'bundle.zip'
        meshio.vtu.write(vtu, mesh, binary=True, compression=None)
        metadata = json.dumps({'schema': _SCHEMA, 'length_unit': data.length_unit,
                               'materials': data.materials, 'mesh_sha256': _hash(vtu)},
                              allow_nan=False, sort_keys=True, indent=2) + '\n'
        if vtu.stat().st_size > _LIMITS['mesh.vtu'] or len(metadata) > _LIMITS['manifest.json']:
            raise ValueError('Tet4 bundle exceeds v1 size limits')
        _check_vtu(vtu)
        with ZipFile(archive, 'w', compression=ZIP_STORED) as output:
            output.write(vtu, 'mesh.vtu')
            output.writestr('manifest.json', metadata)
        try:
            os.link(archive, target)
        except FileExistsError:
            raise
        except OSError as exc:
            raise OSError(f'Tet4 atomic publication requires a writable destination '
                          f'filesystem supporting hard links: {exc}') from exc
    return target


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Tet4 manifest has duplicate key: {key}')
        result[key] = value
    return result


def _manifest(raw):
    metadata = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique_object)
    if (not isinstance(metadata, dict) or set(metadata) != _MANIFEST_KEYS or
            metadata['schema'] != _SCHEMA or
            not isinstance(metadata['mesh_sha256'], str) or
            not re.fullmatch(r'[0-9a-f]{64}', metadata['mesh_sha256']) or
            not isinstance(metadata['materials'], dict)):
        raise ValueError('invalid Tet4 v1 manifest')
    for key in metadata['materials']:
        if not re.fullmatch(r'0|[1-9][0-9]{0,19}', key):
            raise ValueError('Tet4 material IDs must be canonical decimal strings in the manifest')
    return metadata


def _children(element, expected, attr=None):
    children = dict((item.tag if attr is None else item.get(attr), item) for item in element)
    if (len(children) != len(element) or set(children) != set(expected) or
            (element.text or '').strip() or any((item.tail or '').strip() for item in element)):
        raise ValueError(f'Tet4 v1 has missing, duplicate or unsupported VTU fields/blocks in {element.tag}')
    return children


def _array(element, entries, dtype, endian, header, components=None):
    allowed = {'type', 'Name', 'format'}
    if components:
        allowed.add('NumberOfComponents')
    if (element.tag != 'DataArray' or set(element.attrib) != allowed or len(element) or
            element.get('format') != 'binary' or element.get('type') not in dtype or
            (components and element.get('NumberOfComponents') != str(components))):
        raise ValueError('unsupported Tet4 v1 VTU DataArray')
    encoded = ''.join((element.text or '').split())
    expected_bytes = entries * 8
    if len(encoded) != 4 * ((header + expected_bytes + 2) // 3):
        raise ValueError('Tet4 VTU binary array has incorrect size')
    decoded = base64.b64decode(encoded, validate=True)
    if len(decoded) != header + expected_bytes or int.from_bytes(decoded[:header], endian) != expected_bytes:
        raise ValueError('Tet4 VTU binary array has incorrect length header')
    return memoryview(decoded)[header:]


class _VTUTree(ET.TreeBuilder):
    def doctype(self, name, pubid, system):
        raise ValueError('Tet4 v1 VTU does not allow a document type declaration')


def _check_vtu(vtu):
    # Meshio accepts/merges several shapes (pieces, unknown cell arrays and
    # FieldData) this geometry-only schema must never silently discard.
    root = ET.parse(vtu, parser=ET.XMLParser(target=_VTUTree())).getroot()
    root_keys = {'type', 'version', 'byte_order', 'header_type'}
    if (root.tag != 'VTKFile' or not set(root.attrib) <= root_keys or
            root.get('type') != 'UnstructuredGrid' or root.get('version') not in ('0.1', '1.0') or
            root.get('byte_order') not in ('LittleEndian', 'BigEndian') or
            root.get('header_type', 'UInt32') not in ('UInt32', 'UInt64')):
        raise ValueError('unsupported Tet4 v1 VTU header')
    grid = _children(root, {'UnstructuredGrid'})['UnstructuredGrid']
    piece = _children(grid, {'Piece'})['Piece']
    if grid.attrib or set(piece.attrib) != {'NumberOfPoints', 'NumberOfCells'}:
        raise ValueError('unsupported Tet4 v1 VTU grid/piece attributes')
    n, m = int(piece.get('NumberOfPoints')), int(piece.get('NumberOfCells'))
    _bounded_counts(n, m)
    blocks = _children(piece, {'Points', 'Cells', 'PointData', 'CellData'})
    if any(block.attrib for block in blocks.values()):
        raise ValueError('unsupported Tet4 v1 VTU block attributes')
    points = _children(blocks['Points'], {'Points'}, 'Name')
    cells = _children(blocks['Cells'], {'connectivity', 'offsets', 'types'}, 'Name')
    point_data = _children(blocks['PointData'], {'node_id'}, 'Name')
    cell_data = _children(blocks['CellData'], {'cell_id', 'material_id'}, 'Name')
    endian = 'little' if root.get('byte_order') == 'LittleEndian' else 'big'
    header = 4 if root.get('header_type', 'UInt32') == 'UInt32' else 8
    _array(points['Points'], n * 3, {'Float64'}, endian, header, components=3)
    _array(cells['connectivity'], m * 4, {'Int64'}, endian, header)
    for fields, count in ((point_data, n), (cell_data, m)):
        for field in fields.values():
            _array(field, count, {'Int64', 'UInt64'}, endian, header)
    dtype = '<i8' if endian == 'little' else '>i8'
    types = np.frombuffer(_array(cells['types'], m, {'Int64'}, endian, header), dtype=dtype)
    offsets = np.frombuffer(_array(cells['offsets'], m, {'Int64'}, endian, header), dtype=dtype)
    if not np.all(types == 10) or not np.array_equal(offsets, 4 * np.arange(1, m + 1, dtype=np.int64)):
        raise ValueError('Tet4 v1 requires exactly one fixed-width tetra block')


def _validate_zip_catalog(source):
    # A v1 writer emits the fixed, uncommented end record, with no ZIP64 or
    # multiple disks. Bound the directory before ZipFile allocates ZipInfos.
    size = source.seek(0, os.SEEK_END)
    if not 22 <= size <= _MAX_ARCHIVE:
        raise ValueError('Tet4 bundle exceeds v1 archive size limit or is truncated')
    source.seek(-22, os.SEEK_END)
    magic, disk, start_disk, count, total, length, offset, comment = unpack('<4s4H2IH', source.read(22))
    if (magic != b'PK\x05\x06' or disk or start_disk or count != 2 or total != 2 or
            not 0 < length <= _MAX_ZIP_DIRECTORY or offset + length != size - 22 or comment):
        raise ValueError('Tet4 v1 requires a bounded ZIP catalog with exactly two entries')
    source.seek(0)


def _read_mesh(archive, scratch, meshio):
    entries = archive.infolist()
    if (len(entries) != 2 or {entry.filename for entry in entries} != set(_LIMITS) or
            any(entry.compress_type != ZIP_STORED or entry.flag_bits & 1 or
                entry.compress_size != entry.file_size or
                not 0 < entry.file_size <= _LIMITS[entry.filename] for entry in entries)):
        raise ValueError('Tet4 v1 requires exactly two bounded, stored ZIP members')
    metadata = _manifest(archive.read('manifest.json'))
    vtu = Path(scratch) / 'mesh.vtu'
    with archive.open('mesh.vtu') as source, vtu.open('wb') as dest:
        shutil.copyfileobj(source, dest)
    if _hash(vtu) != metadata['mesh_sha256']:
        raise ValueError('Tet4 VTU SHA-256 does not match the manifest')
    _check_vtu(vtu)
    return meshio.vtu.read(vtu), metadata


def read_tet4_vtu_bundle(path):
    """Read a v1 bundle produced by the matching writer, with no unit conversion."""
    meshio = _meshio()
    try:
        with Path(path).open('rb') as source, TemporaryDirectory(prefix='fibernet-tet4-') as scratch:
            _validate_zip_catalog(source)
            with ZipFile(source) as archive:
                mesh, metadata = _read_mesh(archive, scratch, meshio)
    except (BadZipFile, Base64Error, ET.ParseError, meshio.ReadError, RecursionError) as exc:
        raise ValueError(f'invalid Tet4 v1 bundle: {exc}') from exc
    if (len(mesh.cells) != 1 or mesh.cells[0].type != 'tetra' or
            set(mesh.point_data) != {'node_id'} or set(mesh.cell_data) != {'cell_id', 'material_id'} or
            any(len(blocks) != 1 for blocks in mesh.cell_data.values()) or
            mesh.field_data or mesh.point_sets or mesh.cell_sets):
        raise ValueError('unsupported Tet4 v1 VTU mesh or fields')
    return _validate(mesh.points, mesh.cells[0].data, mesh.point_data['node_id'],
                     mesh.cell_data['cell_id'][0], mesh.cell_data['material_id'][0],
                     {int(k): v for k, v in metadata['materials'].items()}, metadata['length_unit'])
