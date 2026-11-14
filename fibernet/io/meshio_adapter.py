"""Optional, straight-line VTU interoperability; not a solver or project format.

Import directly from this module. ``meshio`` is loaded only when exporting.
Geometric units are declared, never inferred or rescaled. Material properties
retain the SI conventions documented by :class:`Material`.
"""
from __future__ import annotations

from collections.abc import Mapping

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np


_RESERVED = {'node_id', 'edge_id', 'radius', 'material_index'}
_UNITS = {'m', 'mm', 'um', 'nm', 'cm'}


def _json_copy(value):
    """Reject lossy or non-JSON provenance and detach nested dictionaries."""
    def check_keys(item):
        if isinstance(item, dict):
            if any(not isinstance(k, str) for k in item):
                raise ValueError('JSON provenance object keys must be strings')
            for child in item.values():
                check_keys(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check_keys(child)
    check_keys(value)
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError('provenance and materials must be finite JSON data') from exc


def _ids(values, label):
    values = list(values)
    for value in values:
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer)) or
                value < 0 or value > np.iinfo(np.int64).max):
            raise ValueError(f'{label} must be nonnegative int64 identifiers')
    return np.asarray(values, dtype=np.int64)


def _unmasked_field(value, label):
    """Reject masks before array coercion, including nested ID-aligned values."""
    pending, seen = [value], set()
    while pending:
        item = pending.pop()
        if np.ma.isMaskedArray(item):
            raise ValueError(f'{label}: masked arrays/scalars are unsupported, including all-false masks')
        if isinstance(item, (list, tuple)) and id(item) not in seen:
            seen.add(id(item))
            pending.extend(item)


def _fields(values, count, specs, location):
    result = {}
    if values is not None and not isinstance(values, dict):
        raise ValueError(f'{location} data must be a dictionary')
    if not isinstance(specs.get(location, {}), dict):
        raise ValueError(f'{location} field metadata must be a dictionary')
    for name, value in (values or {}).items():
        if not isinstance(name, str) or not name or name in _RESERVED:
            raise ValueError('field names must be nonempty and not reserved')
        _unmasked_field(value, f'{location} field {name}')
        arr = np.asarray(value)
        if arr.shape not in ((count,), (count, 3)) or arr.dtype.kind not in 'fiu':
            raise ValueError(f'{location} field {name} must be numeric scalar or 3-vector data')
        if not np.isfinite(arr).all():
            raise ValueError(f'{location} field {name} must be finite')
        spec = specs.get(location, {}).get(name)
        if not isinstance(spec, dict) or not all(
                isinstance(spec.get(k), str) and spec[k] for k in ('units', 'meaning')):
            raise ValueError(f'{location} field {name} needs units and meaning')
        result[name] = arr.copy()
    return result


def _fields_by_id(values, ids, location):
    """Align complete ID mappings; never infer or fill unavailable values."""
    if not isinstance(values, Mapping):
        raise ValueError(f'{location} data by ID must be a mapping')
    result = {}
    expected = set(map(int, ids))
    for name, field in values.items():
        if not isinstance(field, Mapping):
            raise ValueError(f'{location} field {name} must be an ID mapping')
        aligned = {}
        for key, value in field.items():
            identifier = int(_ids([key], f'{location} field {name} IDs')[0])
            if identifier in aligned:
                raise ValueError(f'{location} field {name} has duplicate IDs')
            aligned[identifier] = value
        actual = set(aligned)
        if actual != expected:
            raise ValueError(f'{location} field {name} requires exact ID coverage; '
                             f'missing IDs: {sorted(expected - actual)}; '
                             f'foreign IDs: {sorted(actual - expected)}')
        result[name] = [aligned[int(identifier)] for identifier in ids]
    return result


def to_meshio(graph, *, length_unit, point_data=None, cell_data=None,
              field_metadata=None, provenance=None, point_data_by_id=None,
              cell_data_by_id=None):
    """Return ``(meshio.Mesh, manifest)`` without changing the graph.

    Points follow sorted node IDs. Cells follow graph edge insertion order.
    Field arrays must already follow those orders; ID mappings are aligned here.
    Each location accepts arrays or complete ID mappings, never both.
    Masked arrays/scalars are unsupported, including all-false masks.
    Only edges without internal
    points are accepted. Parallel edges and isolated nodes are retained.
    Units are one of m, cm, mm, um, nm; no conversion is performed.
    """
    if point_data is not None and point_data_by_id is not None:
        raise ValueError('point_data and point_data_by_id are mutually exclusive')
    if cell_data is not None and cell_data_by_id is not None:
        raise ValueError('cell_data and cell_data_by_id are mutually exclusive')
    if length_unit not in _UNITS:
        raise ValueError('length_unit must be explicitly declared: m, cm, mm, um, nm')
    node_ids = _ids(graph.nodes, 'node IDs')
    node_ids.sort()
    edge_ids = _ids(graph.edges, 'edge IDs')
    if not len(node_ids) or not len(edge_ids):
        raise ValueError('line export requires at least one edge and its nodes')
    rows = {int(nid): i for i, nid in enumerate(node_ids)}
    positions = np.array([graph.nodes[int(nid)].position for nid in node_ids], dtype=np.float64)
    if positions.shape != (len(node_ids), 3) or not np.isfinite(positions).all():
        raise ValueError('node positions must be finite (N,3) coordinates')
    connectivity, radii, materials, material_indices = [], [], [], []
    material_lookup = {}
    for eid in edge_ids:
        edge = graph.edges[int(eid)]
        if edge.internal_points is not None and np.asarray(edge.internal_points).size:
            raise ValueError('unsupported geometry: internal-point edges require explicit subdivision')
        ends = _ids([edge.node_i, edge.node_j], 'edge endpoints')
        if any(int(nid) not in rows for nid in ends):
            raise ValueError('edge endpoints must reference existing nodes')
        a, b = [rows[int(nid)] for nid in ends]
        if a == b or np.linalg.norm(positions[b] - positions[a]) == 0:
            raise ValueError('line edges must have nonzero length')
        radius = float(edge.radius)
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError('edge radii must be finite and positive')
        material = _json_copy(edge.material.to_dict())
        key = json.dumps(material, sort_keys=True, allow_nan=False)
        if key not in material_lookup:
            material_lookup[key] = len(materials)
            materials.append(material)
        connectivity.append([a, b])
        radii.append(radius)
        material_indices.append(material_lookup[key])
    specs = _json_copy(field_metadata or {})
    if not isinstance(specs, dict):
        raise ValueError('field_metadata must be a dictionary')
    if point_data_by_id is not None:
        point_data = _fields_by_id(point_data_by_id, node_ids, 'point')
    if cell_data_by_id is not None:
        cell_data = _fields_by_id(cell_data_by_id, edge_ids, 'cell')
    pd = _fields(point_data, len(node_ids), specs, 'point')
    cd = _fields(cell_data, len(edge_ids), specs, 'cell')
    pd['node_id'] = node_ids.copy()
    cd.update(edge_id=edge_ids.copy(), radius=np.array(radii, dtype=np.float64),
              material_index=np.array(material_indices, dtype=np.int64))
    manifest = _json_copy({
        'schema': 'fibernet-line-vtu-1', 'geometry_dimension': graph.dimension,
        'length_unit': length_unit, 'coordinate_conversion': 'none',
        'material_units': 'SI per fibernet.core.material.Material; extra fields user-defined',
        'node_ids': node_ids.tolist(), 'edge_ids': edge_ids.tolist(),
        'cell_type': 'line', 'materials': materials, 'field_metadata': specs,
        'graph_metadata': graph.metadata,
        'node_metadata': [graph.nodes[int(i)].metadata for i in node_ids],
        'node_boundary': [[bool(v) for v in graph.nodes[int(i)].boundary] for i in node_ids],
        'edge_metadata': [graph.edges[int(i)].metadata for i in edge_ids],
        'edge_segments': [graph.edges[int(i)].segments for i in edge_ids],
        'box_size': None if graph.box_size is None else graph.box_size.tolist(),
        'provenance': provenance or {},
        'scope': 'straight line interchange; no beam law, joint law or volume mesh',
    })
    try:
        import meshio
    except ImportError as exc:
        raise ImportError('Optional line export requires meshio; install meshio==5.3.5') from exc
    manifest['meshio_version'] = meshio.__version__
    mesh = meshio.Mesh(positions, [('line', np.array(connectivity, dtype=np.int64))],
                       point_data=pd, cell_data={k: [v] for k, v in cd.items()})
    return mesh, manifest


def write_vtu_bundle(directory, graph, **kwargs):
    """Publish a new directory containing network.vtu and manifest.json.

    Existing destinations are never overwritten. A sibling staging directory is
    removed on failure. Directory publication assumes ordinary local filesystem
    rename semantics; concurrent writers to the same destination are unsupported.
    """
    target = Path(directory)
    if target.exists():
        raise FileExistsError(f'export destination already exists: {target}')
    mesh, manifest = to_meshio(graph, **kwargs)
    staging = Path(tempfile.mkdtemp(prefix=f'.{target.name}-', dir=target.parent))
    try:
        vtu = staging / 'network.vtu'
        mesh.write(vtu, file_format='vtu')
        manifest['mesh_sha256'] = hashlib.sha256(vtu.read_bytes()).hexdigest()
        (staging / 'manifest.json').write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        if target.exists():
            raise FileExistsError(f'export destination already exists: {target}')
        os.rename(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target
