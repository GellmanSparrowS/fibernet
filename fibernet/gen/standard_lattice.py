"""Explicit SI octet geometry, not a mechanics qualification.

Exported as octet_fcc_v1 without changing legacy generator registrations.
Material semantics are canonical native values, never Python object identity.
"""
from __future__ import annotations

import copy
import json
import math
import sys
from numbers import Real

from fibernet.core.material import Material
from fibernet.core.structure_graph import StructureGraph, _material_record
from fibernet.core.tiling import tile_3d
from fibernet.gen.pattern import _unit_fcc_3d

_DEFINITION = 'octet_fcc_v1'
_TAU0 = 1e-6
_MAX_NODES = 100_000
_MAX_EDGES = 600_000
_MAX_MATERIAL_BYTES = 16_384
_MAX_ESTIMATED_PEAK_BYTES = 256 * 1024 * 1024
_MAX_LOOKUP_CANDIDATES = 4096
_SOURCE_BLOBS = {
    'fibernet/core/material.py': '74187e8d41b3b631bd50d96e47d5545f915af734',
    'fibernet/gen/pattern.py': 'e8052eace7a3743505a0784440c84b2d630d8f52',
    'fibernet/core/tiling.py': 'af728721070b9c757e6be8ba76d6ce0bf2128fce',
    'fibernet/core/structure_graph.py': 'd78406fafb4a205beb1aad731cbeb69774d4d035',
}


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(name + ' must be a finite positive real scalar')
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(name + ' must be representable as float64') from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(name + ' must be a finite positive real scalar')
    return result


def _preflight(a, grid, radius, material):
    a = _positive(a, 'a')
    radius = _positive(radius, 'radius')
    if (type(grid) is not tuple or len(grid) != 3
            or any(type(n) is not int or n <= 0 for n in grid)):
        raise ValueError('grid must be a tuple of three positive Python integers')
    nx, ny, nz = grid
    faces = nx * ny * (nz + 1) + nx * (ny + 1) * nz + (nx + 1) * ny * nz
    expected_n = (nx + 1) * (ny + 1) * (nz + 1) + faces
    expected_e = 4 * faces + 12 * nx * ny * nz
    if expected_n > _MAX_NODES or expected_e > _MAX_EDGES:
        raise ValueError('lattice exceeds V1 node/edge allocation budget')
    if 64 * sys.float_info.epsilon * max(grid) >= _TAU0:
        raise ValueError('grid exceeds normalized coordinate resolution')
    tau = a * _TAU0
    extent = [a * n for n in grid]
    if (not math.isfinite(tau) or tau <= 0
            or any(not math.isfinite(x) for x in extent)):
        raise ValueError('final tolerance or coordinate extent is not representable')
    # Protect tolerance-scale lookup tests from squared-norm underflow and
    # conservative three-dimensional extent distances from norm overflow.
    if tau / 4 < math.sqrt(sys.float_info.min):
        raise ValueError('tolerance-scale distances are outside float64 norm range')
    if max(extent) > math.sqrt(sys.float_info.max) / 16:
        raise ValueError('extent is outside float64 norm range')
    if 64 * sys.float_info.epsilon * max(extent) >= tau:
        raise ValueError('final coordinates cannot resolve the requested tolerance')
    h = max(tau, 1e-12)
    if not math.isfinite(max(extent) / h):
        raise ValueError('final spatial keys are not representable')
    span = math.ceil(4 * h / a) + 2
    candidates = 27 * min(expected_n, span ** 3)
    if candidates > _MAX_LOOKUP_CANDIDATES:
        raise ValueError('lattice exceeds conservative spatial-bin lookup budget')
    if not isinstance(material, Material):
        raise ValueError('material must use the native Material serialization contract')
    record = _material_record(material.to_dict())
    restored = Material.from_dict(copy.deepcopy(record))
    if _material_record(restored.to_dict()) != record:
        raise ValueError('material must have canonical lossless native serialization')
    material_bytes = len(json.dumps(record, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False, ensure_ascii=True).encode('utf-8'))
    if material_bytes > _MAX_MATERIAL_BYTES:
        raise ValueError('material record exceeds V1 payload budget')
    # Conservative WORK accounting, not a measured or guaranteed RSS bound.
    # Includes normalized graph, detached record and rebuilt graph, plus
    # repeated per-edge material data and Python-object overhead allowance.
    peak_work_bytes = expected_n * 4096 + expected_e * (8192 + 64 * material_bytes)
    if peak_work_bytes > _MAX_ESTIMATED_PEAK_BYTES:
        raise ValueError('lattice exceeds V1 peak-work allocation budget')
    return a, radius, tau, extent, expected_n, expected_e, record, peak_work_bytes


def octet_fcc_v1(*, a, grid, radius, material):
    """Return a finite closed FCC nearest-neighbor StructureGraph in SI metres.

    Required grid is a tuple of three positive integers. No stochastic,
    anisotropic-cell, subdivision, periodic-boundary or mechanics options.
    Radius does not select bonds. Caller Material is detached by native record.
    """
    (a, radius, tau, extent, expected_n, expected_e,
     requested_material, peak_work_bytes) = _preflight(a, grid, radius, material)
    # Never expose caller Material to metadata-sensitive shared-fiber matching.
    unit = _unit_fcc_3d(box=(1., 1., 1.), radius=1., material=Material(),
                        n_pts_per_side=0, point_displacements=None, n_internal=0)
    normalized = tile_3d(unit, grid=grid, box_size=(1., 1., 1.))
    if normalized.num_nodes != expected_n or normalized.num_edges != expected_e:
        raise ValueError('normalized FCC counts violate the versioned definition')
    labels = {}
    for nid, node in normalized.nodes.items():
        twice = [2 * float(x) for x in node.position]
        if any(not math.isfinite(x) or not x.is_integer() for x in twice):
            raise ValueError('normalized FCC vertex is not on the half-coordinate lattice')
        q = tuple(int(x) for x in twice)
        if sum(q) % 2 or any(x < 0 or x > 2 * n for x, n in zip(q, grid)):
            raise ValueError('normalized FCC vertex violates parity or finite bounds')
        labels[nid] = q
    if len(set(labels.values())) != expected_n:
        raise ValueError('normalized FCC has duplicate physical vertices')
    pairs = set()
    for edge in normalized.edges.values():
        qi, qj = labels[edge.node_i], labels[edge.node_j]
        if sorted(abs(x - y) for x, y in zip(qi, qj)) != [0, 1, 1]:
            raise ValueError('normalized FCC has a non-nearest-neighbor bar')
        if edge.internal_points is not None:
            raise ValueError('sampled or curved bars are outside V1')
        pairs.add(tuple(sorted((qi, qj))))
    if len(pairs) != expected_e:
        raise ValueError('normalized FCC has duplicate physical bars')
    record = copy.deepcopy(normalized.to_dict())
    record['tolerance'] = tau
    record['box_size'] = extent
    # Preserve existing provenance but qualify all dimensionless construction
    # data rather than leaving unit_box=[1,1,1] to appear to mean metres.
    record['metadata'] = {
        'definition_id': _DEFINITION,
        'definition_schema': 'fibernet.standard-lattice/1',
        'generator_revision': 'fibernet.standard-lattice/1.0',
        'length_unit': 'm', 'a_m': a, 'grid': list(grid), 'radius_m': radius,
        'boundary_policy': 'closed_box_induced_nearest_neighbor',
        'nominal_extent_m': list(extent),
        'geometric_bounds_m': [[0., 0., 0.], list(extent)],
        'coordinate_convention': 'right-handed Cartesian; q=2*x/a; even parity',
        'tolerance_m': tau, 'source_blobs': dict(_SOURCE_BLOBS),
        'construction_normalized': record['metadata'],
        'estimated_peak_work_bytes': peak_work_bytes,
        'capability_scope': 'geometry_only_not_mechanics',
    }
    for node in record['nodes']:
        q = labels[node['id']]
        node['position'] = [a * (x / 2) for x in q]
        expected_boundary = [flag for x, n in zip(q, grid)
                             for flag in (x == 0, x == 2 * n)]
        if node['boundary'] != expected_boundary:
            raise ValueError('tiler boundary flags violate exact FCC labels')
        node['metadata']['lattice_index'] = list(q)
        node['metadata']['canonical_node_key'] = 'fcc:' + ','.join(map(str, q))
    for edge in record['edges']:
        lo, hi = sorted((labels[edge['node_i']], labels[edge['node_j']]))
        edge['radius'] = radius
        edge['material'] = copy.deepcopy(requested_material)
        edge['metadata']['physical_edge_key'] = [_DEFINITION, list(lo), list(hi)]
    # Native reconstruction sets final tolerance first, rebuilds both lookup
    # structures, and preserves IDs/counters, boundaries, segments and records.
    return StructureGraph.from_dict(record)
