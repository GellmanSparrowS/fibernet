"""Pure GINE material-input contracts; no Torch/PyG model or physics execution.

These checks establish input representation and refusal behavior only. They do
not establish numerical operator consumption, trainability, or physical accuracy.
The optional module is loaded in isolation with both optional backends blocked.
"""
import ast
import builtins
import importlib.util
import json
import struct
import sys
from dataclasses import fields as dataclass_fields, replace
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

import numpy as np
import pytest

from fibernet.core.material import Material
from fibernet.core.structure_graph import SEdge, SNode, StructureGraph
from fibernet.ml.graph_data import (
    GraphBatch, MaterialFeatureView, graph_batch, material_features_from_batch,
)


SOURCE = Path(__file__).parents[1] / 'fibernet/ml/edge_gine.py'
MATERIAL_FIELDS = ('youngs_modulus', 'shear_modulus', 'density', 'poissons_ratio')
UNITS = dict(zip(MATERIAL_FIELDS, ('Pa', 'Pa', 'kg/m^3', '1')))
SCALES = dict(zip(MATERIAL_FIELDS, (100., 10., 1000., .5)))
GEOMETRY_FIELDS = ('radius', 'chord_length', 'polyline_length')
VERSION = 'fibernet.edge_gine.material_view.v1'
OLD_PROVENANCE_KEYS = {
    'graph_ids', 'source_groups', 'node_ids', 'physical_edge_ids', 'node_ptr',
    'edge_ptr', 'physical_edge_index', 'message_parent_rows', 'message_parent_ids',
    'message_orientation', 'positions', 'internal_points', 'internal_ptr',
    'internal_available', 'graph_metadata', 'node_metadata', 'edge_metadata',
}


@pytest.fixture(scope='module')
def gine():
    """Exercise the real optional-import refusal path without loading a backend."""
    name = '_fibernet_edge_gine_material_contract_no_backend'
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    module = importlib.util.module_from_spec(spec)
    original_import = builtins.__import__

    def no_optional_backend(module_name, *args, **kwargs):
        if module_name.split('.')[0] in ('torch', 'torch_geometric'):
            raise ImportError('optional backend deliberately unavailable in pure contract test')
        return original_import(module_name, *args, **kwargs)

    with patch.dict(sys.modules, {name: module}):
        with patch.object(builtins, '__import__', no_optional_backend):
            spec.loader.exec_module(module)
        yield module


def readonly(value, dtype=None):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def bits32(value):
    return struct.unpack('=I', np.asarray(value, dtype=np.float32).tobytes())[0]


def graph():
    """Parallel members, reverse physical direction and one explicit curve."""
    g = StructureGraph(3)
    g.nodes[20] = SNode(np.array([2., 0., 0.]), 20,
                       boundary=(True, False, False, False, False, False))
    g.nodes[4] = SNode(np.array([0., 0., 0.]), 4)
    g.edges[9] = SEdge(20, 4, 9, radius=.2, material=Material(name='same',
        youngs_modulus=200., shear_modulus=70., density=0., poissons_ratio=.25),
        internal_points=np.array([[1., 1., 0.]]), segments=3)
    g.edges[2] = SEdge(4, 20, 2, radius=.4, material=Material(name='same',
        youngs_modulus=800., shear_modulus=320., density=2000., poissons_ratio=.25),
        internal_points=None, segments=7)
    g.edges[7] = SEdge(20, 4, 7, radius=.6, material=Material(name='same',
        youngs_modulus=600., shear_modulus=240., density=6000., poissons_ratio=.25),
        internal_points=np.empty((0, 3)), segments=2)
    return g


def snapshot(*graphs, **changes):
    options = dict(graph_ids=['g' + str(i) for i in range(len(graphs))],
                   length_unit='m', coordinate_frame='laboratory',
                   edge_features=GEOMETRY_FIELDS, include_provenance=True)
    options.update(changes)
    return graph_batch(graphs, **options)


def profile(names=MATERIAL_FIELDS, scales=None):
    return dict(view_schema='fibernet.material_feature_view.v1',
                batch_schema='fibernet.graph_batch.v2',
                provenance_schema='fibernet.graph_batch.provenance.v1',
                fields=list(names), units={k: UNITS[k] for k in names},
                scales={k: (SCALES if scales is None else scales)[k] for k in names},
                material_assumption='isotropic_scalar',
                origin_policy='retain_stored_origin_unknown')


def feature(name, source, unit, scale, required=False, offset=0.):
    return dict(name=name, source=source, unit=unit, scale=scale, offset=offset,
                required=required, parity='even')


def descriptors(p):
    result = [feature(k, {'snapshot': k}, 'm', scale)
              for k, scale in zip(GEOMETRY_FIELDS, (.1, 2., 2.))]
    result += [feature(k, {'material_view': k}, p['units'][k], p['scales'][k])
               for k in p['fields']]
    return result


def config(gine, p=None, **changes):
    p = profile() if p is None else p
    selected_features = changes.pop('features') if 'features' in changes else descriptors(p)
    options = dict(features=selected_features, length_unit='m',
                   coordinate_frame='laboratory', node_offset=[0., 0., 0., 0.],
                   node_scale=[2., 2., 2., 3.], pooling='mean', material_profile=p)
    options.update(changes)
    return gine.make_config(**options)


def view(batch, p=None):
    p = profile() if p is None else p
    return material_features_from_batch(batch, fields=p['fields'], units=p['units'],
        scales=p['scales'], material_assumption=p['material_assumption'],
        origin_policy=p['origin_policy'])


def inputs(gine, batch, *, c=None, v=None, groups=None):
    return gine._material_inputs(batch, config=config(gine) if c is None else c,
        source_groups={g: 'independent-source-' + g for g in batch.graph_ids}
                      if groups is None else groups,
        material_view=view(batch) if v is None else v)


def replace_record(batch, record, row=0):
    records = list(batch.provenance.edge_material_records)
    records[row] = record
    present = [r is not None for r in records]
    payload = replace(batch.provenance, edge_material_records=tuple(records),
        edge_material_available=readonly(present, bool),
        edge_material_origin=tuple('stored_origin_unknown' if p else 'missing'
                                   for p in present))
    return replace(batch, provenance=payload)


def test_closed_config_and_original_prepared_shape(gine):
    p = profile()
    c = config(gine, p)
    assert c['version'] == VERSION
    assert c['material_profile'] == p
    assert c['node_schema'] == ['x', 'y', 'z', 'dimension']
    assert (c['dtype'], c['device'], c['hidden'], c['layers'], c['eps']) == (
        'float32', 'cpu', 16, 2, 0.)
    assert [f['source'] for f in c['features']] == [
        {'snapshot': k} for k in GEOMETRY_FIELDS] + [
        {'material_view': k} for k in MATERIAL_FIELDS]
    assert [f.name for f in dataclass_fields(gine.PreparedGINE)] == [
        'node_features', 'edge_features', 'edge_index', 'graph_index',
        'config_json', 'provenance_json']
    # Caller-owned declarations must not remain mutable aliases of the config.
    p['fields'].reverse()
    p['units']['density'] = 'wrong'
    p['scales']['density'] = 999.
    assert c['material_profile'] == profile()


def test_all_declared_columns_once_and_authoritative_message_parentage(gine):
    a, b = graph(), graph()
    b._edges = dict(reversed(list(b.edges.items())))
    s = snapshot(a, b)
    v = view(s)
    nodes, edges, index, membership, config_json, provenance_json = inputs(gine, s, v=v)
    assert nodes.dtype == edges.dtype == np.dtype('float32')
    assert index.dtype == membership.dtype == np.dtype('int64')
    assert (nodes.shape, edges.shape, index.shape, membership.shape) == (
        (4, 4), (12, 14), (2, 12), (4,))
    physical = np.array([[2., 1., np.sqrt(2.), 2., 7., 0., .5],
                         [4., 1., 1., 8., 32., 2., .5],
                         [6., 1., 1., 6., 24., 6., .5]], dtype=np.float32)
    physical = np.vstack([physical, physical[::-1]])
    expected = np.concatenate([physical, np.ones_like(physical)], axis=1)
    np.testing.assert_array_equal(edges, expected[s.message_parent_rows])
    np.testing.assert_array_equal(nodes, [[0., 0., 0., 1.], [1., 0., 0., 1.]] * 2)
    np.testing.assert_array_equal(index, s.message_index)
    np.testing.assert_array_equal(membership, [0, 0, 1, 1])
    assert s.edge_ids.tolist() == [9, 2, 7, 7, 2, 9]
    assert json.loads(config_json) == config(gine)
    prov = json.loads(provenance_json)
    assert set(prov) == OLD_PROVENANCE_KEYS | {'material_profile'}
    assert prov['physical_edge_ids'] == s.edge_ids.tolist()
    assert prov['message_parent_rows'] == s.message_parent_rows.tolist()
    assert prov['message_parent_ids'] == s.message_parent_ids.tolist()
    assert prov['message_orientation'] == [1, -1] * 6
    assert prov['internal_available'] == [True, False, True, True, False, True]
    assert prov['internal_points'] == s.internal_points.tolist()
    material = prov['material_profile']
    assert set(material) == {'config', 'property_status', 'physical_available'} | {
        f.name for f in dataclass_fields(s.provenance) if f.name != 'schema_version'}
    assert material['config'] == profile()
    assert material['property_status'] == [list(row) for row in v.property_status]
    assert material['physical_available'] == v.available.tolist()
    assert material['node_boundary_face_flags'] == s.provenance.node_boundary_face_flags.tolist()
    assert material['edge_segments'] == [3, 7, 2, 2, 7, 3]
    assert material['edge_material_records'][0]['name'] == 'same'
    assert material['edge_material_records'][0]['youngs_modulus'] == 200.
    assert material['edge_material_records'][0]['shear_modulus'] == 70.
    assert material['edge_material_origin'] == ['stored_origin_unknown'] * 6
    assert not any(np.shares_memory(out, original) for out in (nodes, edges, index, membership)
                   for original in (s.positions, s.message_index, v.values, v.available))
    before = v.values.tobytes(), s.message_index.tobytes()
    edges[0, 3] = 123.
    index[0, 0] = 99
    assert (v.values.tobytes(), s.message_index.tobytes()) == before


def test_material_values_are_not_scaled_or_centered_twice(gine):
    p = profile(names=('density', 'youngs_modulus'), scales={'density': 10., 'youngs_modulus': 4.})
    s = snapshot(graph())
    v = view(s, p)
    c = config(gine, p)
    edge = inputs(gine, s, c=c, v=v)[1]
    np.testing.assert_array_equal(edge[::2, 3:5], [[0., 50.], [200., 200.], [600., 150.]])
    assert edge[2, 3] != np.float32(v.values[1, 0] / p['scales']['density'])
    assert c['features'][3]['scale'] == 10.


def test_material_missing_null_missing_property_unknown_and_true_zero(gine):
    g = graph()
    g.edges[9].material = None
    g.edges[2].material.shear_modulus = None
    g.edges[7].material.density = -0.
    s = snapshot(g)
    record = dict(s.provenance.edge_material_records[1])
    del record['youngs_modulus']
    s = replace_record(s, MappingProxyType(record), row=1)
    v = view(s)
    result = inputs(gine, s, v=v)
    edge = result[1]
    assert v.property_status == (
        ('material_missing',) * 4,
        ('property_missing', 'property_null', 'stored_origin_unknown', 'stored_origin_unknown'),
        ('stored_origin_unknown',) * 4,
    )
    np.testing.assert_array_equal(edge[::2, 10:], v.available.astype(np.float32))
    assert bits32(edge[0, 3]) == bits32(edge[2, 3]) == bits32(edge[2, 4]) == 0
    assert bits32(edge[4, 5]) == 1 << 31
    assert edge[4, 12] == 1.  # A signed true zero remains available.
    payload = json.loads(result[5])['material_profile']
    assert payload['property_status'] == [list(row) for row in v.property_status]
    assert payload['physical_available'] == v.available.tolist()
    assert payload['edge_material_records'][0] is None
    assert 'youngs_modulus' not in payload['edge_material_records'][1]
    assert payload['edge_material_records'][1]['shear_modulus'] is None
    assert payload['edge_material_origin'] == ['missing', 'stored_origin_unknown', 'stored_origin_unknown']


def test_legitimate_nonzero_raw_equal_to_offset_centers_to_zero(gine):
    s = snapshot(graph())
    p = profile()
    f = descriptors(p)
    f[0]['offset'] = .2
    c = config(gine, p, features=f, node_offset=[2., 0., 0., 3.])
    nodes, edges, *_ = inputs(gine, s, c=c)
    assert bits32(edges[0, 0]) == 0 and edges[0, 7] == 1.
    assert bits32(nodes[1, 0]) == 0
    assert [bits32(x) for x in nodes[:, 3]] == [0, 0]
    assert nodes[0, 0] == -1.


@pytest.mark.parametrize('value', [0., -0., 2. ** -126, -(2. ** -126), 2. ** 100, -(2. ** 100)])
def test_material_float32_finite_sign_and_genuine_zero_retention(gine, value):
    g = graph()
    g.edges[9].material.density = value
    p = profile(names=('density',), scales={'density': 1.})
    s = snapshot(g)
    edge = inputs(gine, s, c=config(gine, p), v=view(s, p))[1]
    assert bits32(edge[0, 3]) == bits32(value)
    assert edge[0, 7] == 1.


@pytest.mark.parametrize('value', [1e39, -1e39, 2. ** -150, -(2. ** -150)])
def test_material_float32_overflow_and_new_nonzero_loss_reject(gine, value):
    g = graph()
    g.edges[9].material.density = value
    p = profile(names=('density',), scales={'density': 1.})
    s = snapshot(g)
    with pytest.raises(ValueError):
        inputs(gine, s, c=config(gine, p), v=view(s, p))


@pytest.mark.parametrize('value', [2. ** -149, -(2. ** -149)])
def test_float32_subnormal_is_retained_or_precisely_refused_if_host_flushes(gine, value):
    g = graph()
    g.edges[9].material.density = value
    p = profile(names=('density',), scales={'density': 1.})
    s = snapshot(g)
    v = view(s, p)
    # Observe the current mode, without changing it or importing a native backend.
    converted_bits = bits32(value)
    if converted_bits & 0x7fffffff:
        out = inputs(gine, s, c=config(gine, p), v=v)[1]
        assert bits32(out[0, 3]) == converted_bits
    else:
        with pytest.raises(ValueError, match='float32|nonzero|range|sign'):
            inputs(gine, s, c=config(gine, p), v=v)


@pytest.mark.parametrize('value', [0., -0.])
def test_geometry_and_coordinate_signed_zeros(gine, value):
    g = graph()
    g.edges[9].radius = value
    g.nodes[4].position[0] = value
    s = snapshot(g)
    nodes, edge, *_ = inputs(gine, s)
    assert bits32(nodes[0, 0]) == bits32(value)
    assert bits32(edge[0, 0]) == bits32(value)


@pytest.mark.parametrize('raw,offset,scale', [
    (1e308, -1e308, 1.), (-1e308, 1e308, 1.),
    (1., 0., 1e-308), (-1., 0., 1e-308),
    (1e-308, 0., 1e308), (-1e-308, 0., 1e308),
    (1e39, 0., 1.), (-1e39, 0., 1.),
    (2. ** -150, 0., 1.), (-(2. ** -150), 0., 1.),
    (5e-324, 0., 1.), (-5e-324, 0., 1.),
])
def test_geometry_subtraction_division_and_float32_new_loss_reject(gine, raw, offset, scale):
    g = graph()
    g.edges[9].radius = raw
    s = snapshot(g)
    p = profile()
    f = descriptors(p)
    f[0].update(offset=offset, scale=scale)
    with pytest.raises(ValueError):
        inputs(gine, s, c=config(gine, p, features=f))


@pytest.mark.parametrize('key,value', [
    ('view_schema', 'unknown'), ('batch_schema', 'fibernet.graph_batch.v1'),
    ('provenance_schema', 'unknown'), ('fields', []), ('fields', 'density'),
    ('fields', ['density', 'density']), ('fields', ['stiffness_tensor']),
    ('material_assumption', 'anisotropic'), ('origin_policy', 'assume_explicit'),
    ('units', {}), ('units', dict(UNITS, density='g/cm^3')),
    ('units', dict(UNITS, extra='Pa')), ('scales', {}),
    ('scales', dict(SCALES, extra=1.)), ('scales', dict(SCALES, density=True)),
    ('scales', dict(SCALES, density=0.)), ('scales', dict(SCALES, density=-0.)),
    ('scales', dict(SCALES, density=-1.)), ('scales', dict(SCALES, density=float('nan'))),
    ('scales', dict(SCALES, density=float('inf'))),
    ('scales', dict(SCALES, density=np.float64(1.))),
])
def test_invalid_profile_declarations(gine, key, value):
    p = profile()
    f = descriptors(p)
    p[key] = value
    with pytest.raises(ValueError):
        config(gine, p, features=f)


@pytest.mark.parametrize('mutation', ['extra', 'missing', 'mapping', 'none', 'bool'])
def test_profile_closed_type_and_keys(gine, mutation):
    p = profile()
    f = descriptors(p)
    if mutation == 'extra': p['unexpected'] = True
    elif mutation == 'missing': del p['origin_policy']
    elif mutation == 'mapping': p = MappingProxyType(p)
    elif mutation == 'none': p = None
    elif mutation == 'bool': p = True
    with pytest.raises(ValueError):
        gine.make_config(features=f, length_unit='m', coordinate_frame='laboratory',
            node_offset=[0.] * 4, node_scale=[1.] * 4, pooling='mean', material_profile=p)


@pytest.mark.parametrize('mutation', [
    'offset', 'scale', 'unit', 'order', 'omit', 'duplicate', 'extra-material',
    'external', 'mixed-source', 'unknown-source', 'required-type', 'parity',
    'identity-name', 'geometry-unit', 'unknown-length-unit', 'no-material',
])
def test_descriptor_and_profile_disagreement_rejects(gine, mutation):
    p = profile()
    f = descriptors(p)
    options = {}
    if mutation == 'offset': f[3]['offset'] = 1.
    elif mutation == 'scale': f[3]['scale'] *= 2.
    elif mutation == 'unit': f[3]['unit'] = 'MPa'
    elif mutation == 'order': f[3], f[4] = f[4], f[3]
    elif mutation == 'omit': del f[4]
    elif mutation == 'duplicate': f.append(dict(f[3], name='other'))
    elif mutation == 'extra-material':
        f.append(feature('custom', {'material_view': 'yield_strength'}, 'Pa', 1.))
    elif mutation == 'external': f.append(feature('custom', {'external': 'custom'}, '1', 1.))
    elif mutation == 'mixed-source': f[3]['source']['external'] = 'youngs_modulus'
    elif mutation == 'unknown-source': f[3]['source'] = {'material': 'youngs_modulus'}
    elif mutation == 'required-type': f[3]['required'] = 1
    elif mutation == 'parity': f[3]['parity'] = 'odd'
    elif mutation == 'identity-name': f[3]['name'] = 'edge_id'
    elif mutation == 'geometry-unit': f[0]['unit'] = 'mm'
    elif mutation == 'unknown-length-unit':
        options['length_unit'] = 'unspecified'
        for col in f[:3]: col['unit'] = 'unspecified'
    elif mutation == 'no-material': f = f[:3]
    with pytest.raises(ValueError):
        config(gine, p, features=f, **options)


def test_tuple_fields_and_interleaved_geometry_preserve_material_relative_order(gine):
    p = profile()
    p['fields'] = tuple(p['fields'])
    original = descriptors(p)
    f = [original[3], original[0], original[4], original[1], original[5], original[2], original[6]]
    c = config(gine, p, features=f)
    s = snapshot(graph())
    edge = inputs(gine, s, c=c, v=view(s, p))[1]
    np.testing.assert_array_equal(edge[0, :7], np.array([2., 2., 7., 1., 0., np.sqrt(2.), .5], np.float32))


@pytest.mark.parametrize('mutation', [
    'schema', 'snapshot-identity', 'wrong-type', 'names', 'units', 'scales',
    'assumption', 'origin-policy', 'values-writable', 'values-dtype', 'values-shape',
    'values-changed', 'values-zero-sign', 'values-nonfinite',
    'available-writable', 'available-dtype', 'available-changed',
    'status-length', 'status-origin', 'status-missing',
])
def test_material_view_is_exact_canonical_consistent_and_same_snapshot(gine, mutation):
    s = snapshot(graph())
    v = view(s)
    if mutation == 'schema': v = replace(v, schema_version='unknown')
    elif mutation == 'snapshot-identity': v = replace(v, snapshot=replace(s))
    elif mutation == 'wrong-type': v = dict(values=v.values, available=v.available)
    elif mutation == 'names': v = replace(v, feature_names=tuple(reversed(v.feature_names)))
    elif mutation == 'units': v = replace(v, property_units=('MPa',) + v.property_units[1:])
    elif mutation == 'scales': v = replace(v, scales=(200.,) + v.scales[1:])
    elif mutation == 'assumption': v = replace(v, material_assumption='other')
    elif mutation == 'origin-policy': v = replace(v, origin_policy='explicit')
    elif mutation == 'values-writable': v = replace(v, values=np.array(v.values))
    elif mutation == 'values-dtype': v = replace(v, values=readonly(v.values, np.float32))
    elif mutation == 'values-shape': v = replace(v, values=readonly(v.values[:-1]))
    elif mutation == 'values-changed':
        values = np.array(v.values); values[0, 0] += 1.
        v = replace(v, values=readonly(values))
    elif mutation == 'values-zero-sign':
        values = np.array(v.values); values[0, 2] = -0.
        v = replace(v, values=readonly(values))
    elif mutation == 'values-nonfinite':
        values = np.array(v.values); values[0, 0] = float('nan')
        v = replace(v, values=readonly(values))
    elif mutation == 'available-writable': v = replace(v, available=np.array(v.available))
    elif mutation == 'available-dtype': v = replace(v, available=readonly(v.available, np.int8))
    elif mutation == 'available-changed':
        available = np.array(v.available); available[0, 0] = False
        v = replace(v, available=readonly(available))
    elif mutation == 'status-length': v = replace(v, property_status=v.property_status[:-1])
    elif mutation == 'status-origin':
        v = replace(v, property_status=(('explicit',) * 4,) + v.property_status[1:])
    elif mutation == 'status-missing':
        v = replace(v, property_status=(('material_missing',) * 4,) + v.property_status[1:])
    with pytest.raises(ValueError):
        inputs(gine, s, v=v)


@pytest.mark.parametrize('field,value,dtype', [
    ('node_ids', [20, 4], np.int64), ('edge_ids', [9, 9, 7], np.int64),
    ('node_ptr', [0, 1], np.int64), ('edge_ptr', [0, 2], np.int64),
    ('message_parent_rows', [1, 1, 0, 0, 2, 2], np.int64),
    ('message_parent_ids', [2, 2, 9, 9, 7, 7], np.int64),
    ('message_orientation', [-1, 1, -1, 1, -1, 1], np.int8),
    ('edge_ids', [9, 2, 7], np.int32),
    ('edge_index', [[1, 0, 1], [0, 2, 0]], np.int64),
])
def test_graph_identity_corruptions_reject_before_message_material_expansion(gine, field, value, dtype):
    s = snapshot(graph())
    v = view(s)
    bad = replace(s, **{field: readonly(value, dtype)})
    with pytest.raises(ValueError):
        inputs(gine, bad, v=replace(v, snapshot=bad))


@pytest.mark.parametrize('mutation', [
    'v1', 'no-provenance', 'provenance-schema', 'writable-positions',
    'float32-positions', 'wrong-frame', 'wrong-unit', 'material-origin',
    'material-availability', 'material-record-mutable', 'writable-internal-points',
    'float32-internal-points', 'writable-internal-ptr', 'int32-internal-ptr',
    'writable-internal-available',
])
def test_batch_and_stored_provenance_corruptions_reject(gine, mutation):
    s = snapshot(graph())
    v = view(s)
    if mutation == 'v1': bad = replace(s, schema_version='fibernet.graph_batch.v1')
    elif mutation == 'no-provenance': bad = replace(s, provenance=None)
    elif mutation == 'provenance-schema': bad = replace(s, provenance=replace(s.provenance, schema_version='other'))
    elif mutation == 'writable-positions': bad = replace(s, positions=np.array(s.positions))
    elif mutation == 'float32-positions': bad = replace(s, positions=readonly(s.positions, np.float32))
    elif mutation == 'wrong-frame': bad = replace(s, coordinate_frame='other')
    elif mutation == 'wrong-unit': bad = replace(s, length_unit='mm')
    elif mutation == 'material-origin':
        bad = replace(s, provenance=replace(s.provenance, edge_material_origin=('explicit',) * 3))
    elif mutation == 'material-availability':
        bad = replace(s, provenance=replace(s.provenance, edge_material_available=readonly([False, True, True], bool)))
    elif mutation == 'material-record-mutable': bad = replace_record(s, dict(s.provenance.edge_material_records[0]))
    elif mutation == 'writable-internal-points': bad = replace(s, internal_points=np.array(s.internal_points))
    elif mutation == 'float32-internal-points': bad = replace(s, internal_points=readonly(s.internal_points, np.float32))
    elif mutation == 'writable-internal-ptr': bad = replace(s, internal_ptr=np.array(s.internal_ptr))
    elif mutation == 'int32-internal-ptr': bad = replace(s, internal_ptr=readonly(s.internal_ptr, np.int32))
    elif mutation == 'writable-internal-available': bad = replace(s, internal_available=np.array(s.internal_available))
    with pytest.raises(ValueError):
        inputs(gine, bad, v=replace(v, snapshot=bad))


@pytest.mark.parametrize('key,value', [
    ('stiffness_tensor', tuple((1.,) * 6 for _ in range(6))),
    ('extra', MappingProxyType({'custom': 1.})), ('unknown_property', 1.),
    ('density', True), ('density', np.float64(1.)),
])
def test_unsupported_stored_material_cannot_be_hidden_behind_an_existing_view(gine, key, value):
    s = snapshot(graph())
    v = view(s)
    record = dict(s.provenance.edge_material_records[0])
    record[key] = value
    bad = replace_record(s, MappingProxyType(record))
    with pytest.raises(ValueError):
        inputs(gine, bad, v=replace(v, snapshot=bad))


def test_custom_declaration_and_view_hooks_are_not_executed(gine):
    class BadFloat(float):
        def __float__(self): raise AssertionError('custom float hook executed')
    class BadDict(dict):
        def __iter__(self): raise AssertionError('custom dict hook executed')
    class BadList(list):
        def __iter__(self): raise AssertionError('custom list hook executed')
    class BadArray(np.ndarray):
        def __array__(self, *args, **kwargs): raise AssertionError('custom array hook executed')
    for key, value in [('scales', dict(SCALES, density=BadFloat(1.))),
                       ('units', BadDict(UNITS)), ('fields', BadList(MATERIAL_FIELDS))]:
        p = profile()
        f = descriptors(p)
        p[key] = value
        with pytest.raises(ValueError): config(gine, p, features=f)
    s = snapshot(graph())
    v = view(s)
    bad_values = v.values.view(BadArray)
    with pytest.raises(ValueError): inputs(gine, s, v=replace(v, values=bad_values))


@pytest.mark.parametrize('name,value', [
    ('node_offset', [0., 0., 0.]), ('node_offset', [True, 0., 0., 0.]),
    ('node_offset', [float('inf'), 0., 0., 0.]),
    ('node_scale', [1., 1., 1., 0.]), ('node_scale', [1., 1., 1., -0.]),
    ('node_scale', [1., 1., 1., -1.]), ('node_scale', [1., 1., 1., True]),
    ('node_scale', [1., 1., 1., float('nan')]),
])
def test_node_normalization_types_and_positive_scales(gine, name, value):
    with pytest.raises(ValueError): config(gine, **{name: value})


@pytest.mark.parametrize('which', ['material', 'geometry'])
def test_required_missing_rejects_and_optional_missing_retains_mask(gine, which):
    g = graph()
    if which == 'material': g.edges[9].material = None
    else: g.edges[9].radius = None
    s = snapshot(g)
    p = profile()
    f = descriptors(p)
    column = 3 if which == 'material' else 0
    optional = inputs(gine, s, c=config(gine, p, features=f))[1]
    assert bits32(optional[0, column]) == 0 and optional[0, column + 7] == 0.
    f[column]['required'] = True
    with pytest.raises(ValueError, match='required'):
        inputs(gine, s, c=config(gine, p, features=f))


@pytest.mark.parametrize('groups', [{}, {'g0': ''}, {'g0': 3}, {'g0': 'a', 'extra': 'b'}])
def test_exact_source_group_coverage(gine, groups):
    with pytest.raises(ValueError):
        inputs(gine, snapshot(graph()), groups=groups)


@pytest.mark.parametrize('key,value', [
    ('version', 'fibernet.edge_gine.v1'), ('version', 'unknown'), ('hidden', 17),
    ('layers', 3), ('dtype', 'float64'), ('device', 'cuda'), ('extra', True),
])
def test_complete_config_is_revalidated_at_bridge(gine, key, value):
    s = snapshot(graph())
    c = config(gine)
    c[key] = value
    with pytest.raises(ValueError):
        inputs(gine, s, c=c)


def test_same_width_reordered_valid_profile_still_rejects_original_view(gine):
    s = snapshot(graph())
    v = view(s)
    p = profile(names=tuple(reversed(MATERIAL_FIELDS)))
    with pytest.raises(ValueError):
        inputs(gine, s, c=config(gine, p), v=v)


@pytest.mark.parametrize('mutation', [
    'missing-profile', 'extra-profile-key', 'wrong-config', 'missing-status',
    'false-origin', 'wrong-availability', 'physical-parent', 'record-null',
    'face-origin', 'segments', 'duplicate-json-key',
])
def test_serialized_material_provenance_is_revalidated_without_a_backend(gine, mutation):
    s = snapshot(graph())
    c = config(gine)
    result = inputs(gine, s, c=c)
    p = json.loads(result[5])
    q = p['material_profile']
    if mutation == 'missing-profile': del p['material_profile']
    elif mutation == 'extra-profile-key': q['extra'] = True
    elif mutation == 'wrong-config': q['config']['scales']['density'] = 99.
    elif mutation == 'missing-status': del q['property_status']
    elif mutation == 'false-origin': q['property_status'][0][0] = 'explicit'
    elif mutation == 'wrong-availability': q['physical_available'][0][0] = False
    elif mutation == 'physical-parent': p['message_parent_rows'][0] = 1
    elif mutation == 'record-null': q['edge_material_records'][0] = None
    elif mutation == 'face-origin': q['node_boundary_face_origin'][0] = 'mechanical_constraint'
    elif mutation == 'segments': q['edge_segments'][0] = 0
    text = json.dumps(p)
    if mutation == 'duplicate-json-key': text = text[:-1] + ',"graph_ids":["g0"]}'
    with pytest.raises(ValueError):
        gine._provenance(text, config=c)


def test_edge_free_graph_supported_but_zero_nodes_and_self_loops_reject(gine):
    g = StructureGraph(3)
    g.nodes[3] = SNode(np.array([0., 0., 0.]), 3)
    s = snapshot(g)
    nodes, edge, index, membership, *_ = inputs(gine, s)
    assert nodes.shape == (1, 4) and edge.shape == (0, 14) and index.shape == (2, 0)
    assert membership.tolist() == [0]
    for bad in (snapshot(StructureGraph(3)), snapshot()):
        with pytest.raises(ValueError): inputs(gine, bad)
    loop = graph()
    loop.edges[9].node_j = loop.edges[9].node_i
    with pytest.raises(ValueError): inputs(gine, snapshot(loop))


def test_optional_backend_absence_never_returns_fake_prepared_or_model(gine):
    assert gine.torch is gine.nn is gine.GINEConv is None
    assert isinstance(gine._BACKEND_ERROR, ImportError)
    s = snapshot(graph())
    c = config(gine)
    with pytest.raises(ImportError, match='no topology-only fallback'):
        gine.prepare_gine(s, config=c, source_groups={'g0': 'parent'}, material_view=view(s))
    with pytest.raises(ImportError, match='no topology-only fallback'):
        gine.GINERegressor(c)


def test_default_v1_config_and_pure_normalization_contract_remain(gine):
    f = [feature('radius', {'snapshot': 'radius'}, 'native', 2.),
         feature('conductivity', {'external': 'conductivity'}, 'S/m', 10.)]
    c = gine.make_config(features=f, length_unit='native', coordinate_frame='source',
        node_offset=[0.] * 4, node_scale=[1.] * 4, pooling='sum')
    assert c['version'] == 'fibernet.edge_gine.v1' and 'material_profile' not in c
    assert gine._config(c) == c
    normalized = gine._normalized(np.array([[2., 10.], [0., 0.]]),
        np.array([[True, True], [True, False]]), np.array([0., 0.]), np.array([2., 10.]))
    np.testing.assert_array_equal(normalized, np.array([[1., 1.], [0., 0.]], np.float32))
    with pytest.raises(ImportError, match='no topology-only fallback'):
        gine.prepare_gine(snapshot(graph(), include_provenance=False), config=c,
                          source_groups={'g0': 'parent'})


def test_existing_model_uses_both_full_edge_channels_without_alternate_architecture():
    tree = ast.parse(SOURCE.read_text())
    model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GINERegressor')
    init = next(n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    forward = next(n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    init_source, forward_source = ast.unparse(init), ast.unparse(forward)
    assert 'self.input_proj = nn.Linear(4, 16)' in init_source
    assert 'self.head = nn.Linear(16, 1)' in init_source
    assert 'edge_dim=2 * len(c[\'features\'])' in init_source
    assert 'range(2)' in init_source and 'train_eps=False' in init_source
    assert 'layer(x, idx, a)' in forward_source
    assert 'dropout' not in init_source.lower() and 'batchnorm' not in init_source.lower()
