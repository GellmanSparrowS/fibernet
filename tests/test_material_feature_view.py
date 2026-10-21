"""Pure representation/identity checks; no model or constitutive validation."""
import ast
import struct
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, fields as dataclass_fields, replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from fibernet.core.material import Material
from fibernet.core.structure_graph import StructureGraph, SNode, SEdge
from fibernet.ml.graph_data import (
    GraphBatch, MaterialFeatureView, graph_batch, material_features_from_batch,
)


UNITS = {'youngs_modulus': 'Pa', 'shear_modulus': 'Pa', 'density': 'kg/m^3', 'poissons_ratio': '1'}


def graph():
    g = StructureGraph(3)
    g.nodes[20] = SNode(np.array([1., 0., 2.]), 20, boundary=(False, True, False, False, False, True))
    g.nodes[4] = SNode(np.array([1., 0., 0.]), 4)
    g.edges[9] = SEdge(20, 4, 9, material=Material(name='same', youngs_modulus=200.,
        poissons_ratio=.25, shear_modulus=77., density=0., thermal_conductivity=0.), segments=3)
    g.edges[2] = SEdge(4, 20, 2, material=Material(name='same', youngs_modulus=800.,
        poissons_ratio=.25, density=2000.), segments=7)
    return g


def snapshot(*graphs, provenance=True):
    return graph_batch(graphs, graph_ids=[str(i) for i in range(len(graphs))],
        length_unit='fixture_native_length_unit_unspecified', coordinate_frame='source',
        include_provenance=provenance)


def convert(batch, **kwargs):
    options = dict(fields=tuple(UNITS), units=dict(UNITS), scales={k: 1. for k in UNITS},
        material_assumption='isotropic_scalar', origin_policy='retain_stored_origin_unknown')
    options.update(kwargs)
    return material_features_from_batch(batch, **options)


def readonly(value, dtype=None):
    a = np.array(value, dtype=dtype, copy=True); a.setflags(write=False); return a


def with_record(batch, record, row=0):
    records = list(batch.provenance.edge_material_records); records[row] = record
    present = [r is not None for r in records]
    return replace(batch, provenance=replace(batch.provenance, edge_material_records=tuple(records),
        edge_material_available=readonly(present, bool),
        edge_material_origin=tuple('stored_origin_unknown' if p else 'missing' for p in present)))


def bits(x):
    return struct.unpack('>Q', struct.pack('>d', x))[0]


def test_order_units_scaling_independent_properties_and_snapshot_identity():
    s = snapshot(graph())
    names = ('poissons_ratio', 'density', 'shear_modulus', 'youngs_modulus')
    view = convert(s, fields=names, units={k: UNITS[k] for k in names},
        scales=dict(poissons_ratio=.5, density=1000., shear_modulus=10., youngs_modulus=100.))
    assert type(view) is MaterialFeatureView and view.snapshot is s
    assert view.schema_version == 'fibernet.material_feature_view.v1'
    assert view.feature_names == names and view.property_units == ('1', 'kg/m^3', 'Pa', 'Pa')
    assert view.scales == (.5, 1000., 10., 100.)
    np.testing.assert_array_equal(view.values, [[.5, 0., 7.7, 2.], [.5, 2., 32., 8.]])
    assert view.available.all() and view.values.dtype == np.float64
    assert all(x == 'stored_origin_unknown' for row in view.property_status for x in row)
    # Explicit G=77 is retained independently of E=200, nu=.25; no reconciliation.
    assert view.values[0, 2] == 7.7
    assert view.snapshot.length_unit == 'fixture_native_length_unit_unspecified'
    assert view.snapshot.provenance.node_boundary_face_flags[1, 1]
    assert view.snapshot.provenance.edge_segments.tolist() == [3, 7]
    assert not isinstance(view, Mapping) and not hasattr(view, '__getitem__')
    assert not any(hasattr(view, x) for x in ('node_features', 'adjacency', 'edge_index', 'to_dict', 'targets'))


def test_missing_null_unknown_and_signed_zero_are_distinct():
    g = graph(); g.edges[9].material = None; g.edges[2].material.shear_modulus = None
    g.edges[2].material.density = -0.
    s = snapshot(g); v = convert(s)
    assert v.property_status[0] == ('material_missing',)*4 and not v.available[0].any()
    assert v.property_status[1] == ('stored_origin_unknown', 'property_null', 'stored_origin_unknown', 'stored_origin_unknown')
    assert not v.available[1, 1] and v.available[1, 2]
    assert bits(v.values[0, 0]) == bits(v.values[1, 1]) == 0
    assert bits(v.values[1, 2]) == 1 << 63
    record = dict(s.provenance.edge_material_records[1]); del record['density']
    v = convert(with_record(s, MappingProxyType(record), row=1))
    assert v.property_status[1][2] == 'property_missing' and not v.available[1, 2]
    assert bits(v.values[1, 2]) == 0


def test_defaults_and_supplied_equivalent_values_keep_unknown_origin():
    g = graph(); g.edges[9].material = Material()
    g.edges[2].material = Material(density=1000., youngs_modulus=1e9, poissons_ratio=.3,
        shear_modulus=1e9/(2*(1+.3)))
    v = convert(snapshot(g))
    assert v.values[0].tobytes() == v.values[1].tobytes()
    assert v.property_status[0] == v.property_status[1] == ('stored_origin_unknown',)*4


def test_parallel_reverse_repeated_ids_empty_graphs_and_directed_parentage():
    a, b = graph(), graph(); b._edges = dict(reversed(list(b.edges.items())))
    s = snapshot(StructureGraph(), a, StructureGraph(), b); v = convert(s)
    assert s.edge_ids.tolist() == [9, 2, 2, 9] and s.edge_ptr.tolist() == [0, 0, 2, 2, 4]
    assert v.values[:, 0].tolist() == [200., 800., 800., 200.]
    assert v.values[s.message_parent_rows, 0].tolist() == [200., 200., 800., 800., 800., 800., 200., 200.]
    assert s.message_parent_ids.tolist() == [9, 9, 2, 2, 2, 2, 9, 9]
    assert s.message_orientation.tolist() == [1, -1]*4
    for empty in (snapshot(), snapshot(StructureGraph())):
        out = convert(empty)
        assert out.values.shape == out.available.shape == (0, 4) and out.property_status == ()


def test_derived_arrays_are_detached_and_readonly():
    g = graph(); s = snapshot(g); v = convert(s)
    g.edges[9].material.youngs_modulus = 999.; g.nodes[20].boundary = (False,)*6
    assert v.values[0, 0] == 200. and s.provenance.node_boundary_face_flags[1, 1]
    with pytest.raises(FrozenInstanceError): v.snapshot = None
    for array in (v.values, v.available):
        assert not array.flags.writeable
        with pytest.raises(ValueError): array[0, 0] = 0
    v.values.setflags(write=True); v.values[0, 0] = 123.
    assert s.provenance.edge_material_records[0]['youngs_modulus'] == 200.


@pytest.mark.parametrize('names', [(), [], 'density', ('name',), ('density', 'density'), (1,), (True,)])
def test_invalid_selection(names):
    with pytest.raises(ValueError, match='fields'): convert(snapshot(graph()), fields=names)


@pytest.mark.parametrize('option,value', [
    ('units', {}), ('units', dict(UNITS, extra='Pa')), ('units', dict(UNITS, density='g/cm^3')),
    ('units', dict(UNITS, youngs_modulus='MPa')), ('units', dict(UNITS, density={0: 'kg/m^3'})),
    ('scales', {}), ('scales', dict.fromkeys(UNITS, True)), ('scales', dict.fromkeys(UNITS, 0.)),
    ('scales', dict.fromkeys(UNITS, -0.)), ('scales', dict.fromkeys(UNITS, -1.)),
    ('scales', dict.fromkeys(UNITS, float('nan'))), ('scales', dict.fromkeys(UNITS, float('inf'))),
    ('scales', dict.fromkeys(UNITS, 10**400)), ('scales', dict.fromkeys(UNITS, np.float64(1.))),
    ('material_assumption', 'inferred'), ('origin_policy', 'assume_explicit'),
])
def test_invalid_declarations(option, value):
    with pytest.raises(ValueError): convert(snapshot(graph()), **{option: value})


@pytest.mark.parametrize('key,value', [
    ('stiffness_tensor', ((1.,)*6,)*6), ('extra', MappingProxyType({'custom': 1.})),
    ('extra', {}), ('unknown', 1.), ('density', None), ('youngs_modulus', None),
    ('poissons_ratio', None), ('name', 1.), ('density', True), ('density', np.float64(1.)),
    ('density', float('inf')), ('density', float('nan')), ('density', 10**400),
    ('thermal_conductivity', float('inf')),
])
def test_unsupported_or_malformed_records(key, value):
    s = snapshot(graph()); record = dict(s.provenance.edge_material_records[0]); record[key] = value
    with pytest.raises(ValueError): convert(with_record(s, MappingProxyType(record)))


def test_known_unselected_optional_properties_are_provenance_only():
    s = snapshot(graph()); record = dict(s.provenance.edge_material_records[0])
    record.update(stiffness_tensor=None, extra=MappingProxyType({}), yield_strength=0., permittivity=None)
    v = convert(with_record(s, MappingProxyType(record)), fields=('density',),
        units={'density': 'kg/m^3'}, scales={'density': 1})
    assert v.values.shape == (2, 1) and v.available.all()
    assert v.snapshot.provenance.edge_material_records[0]['yield_strength'] == 0.


@pytest.mark.parametrize('value,scale', [(5e-324, 2.), (-5e-324, 2.),
    (1., 5e-324), (-1., 5e-324), (1e308, 1e-308), (-1e308, 1e-308)])
def test_binary64_range_and_nonzero_loss_are_rejected(value, scale):
    s = snapshot(graph()); record = dict(s.provenance.edge_material_records[0]); record['density'] = value
    with pytest.raises(ValueError, match='normalization'):
        convert(with_record(s, MappingProxyType(record)), fields=('density',),
            units={'density': 'kg/m^3'}, scales={'density': scale})


@pytest.mark.parametrize('value', [0., -0., 5e-324, -5e-324, 1e308, -1e308])
def test_representable_values_and_signs_are_retained(value):
    s = snapshot(graph()); record = dict(s.provenance.edge_material_records[0]); record['density'] = value
    scale = 1.
    original, divided = bits(value), bits(value / scale)
    magnitude_mask = (1 << 63) - 1
    # Observe ordinary arithmetic independently of the candidate. Importing a
    # native extension may enable FTZ/DAZ; never reset the process mode here.
    subnormal = 0 < original & magnitude_mask < 1 << 52
    if subnormal and divided & magnitude_mask == 0:
        assert divided >> 63 == original >> 63
        with pytest.raises(ValueError, match=
                '^material normalization lost finite range, a nonzero value or sign$'):
            convert(with_record(s, MappingProxyType(record)), fields=('density',),
                units={'density': 'kg/m^3'}, scales={'density': scale})
    else:
        # Genuine signed zero and normal controls always require exact success.
        assert divided == original
        v = convert(with_record(s, MappingProxyType(record)), fields=('density',),
            units={'density': 'kg/m^3'}, scales={'density': scale})
        assert bits(v.values[0, 0]) == original and v.available[0, 0]


def test_no_custom_scalar_array_or_declaration_hooks():
    class BadFloat(float):
        def __float__(self): raise AssertionError('float hook called')
    class BadDict(dict):
        def __iter__(self): raise AssertionError('mapping hook called')
    class BadList(list):
        def __iter__(self): raise AssertionError('iterable hook called')
    class BadArray(np.ndarray):
        def __array__(self, *args, **kwargs): raise AssertionError('array hook called')
    s = snapshot(graph())
    for options in (dict(scales=dict.fromkeys(UNITS, BadFloat(1.))), dict(units=BadDict(UNITS)),
                    dict(fields=BadList(UNITS))):
        with pytest.raises(ValueError): convert(s, **options)
    for value in (BadFloat(1.), object()):
        record = dict(s.provenance.edge_material_records[0]); record['density'] = value
        with pytest.raises(ValueError): convert(with_record(s, MappingProxyType(record)))
    with pytest.raises(ValueError): convert(with_record(s, BadDict(density=1.)))
    with pytest.raises(ValueError): convert(replace(s, edge_ids=s.edge_ids.view(BadArray)))


@pytest.mark.parametrize('name,value,dtype', [
    ('node_ptr', [0, 1], np.int64), ('edge_ptr', [0, 3], np.int64),
    ('edge_ids', [9, 9], np.int64), ('node_ids', [20, 4], np.int64),
    ('edge_index', [[9, 0], [0, 1]], np.int64), ('message_parent_rows', [0, 1, 0, 1], np.int64),
    ('message_parent_ids', [2, 2, 9, 9], np.int64), ('message_orientation', [-1, 1, -1, 1], np.int8),
    ('dimensions', [4], np.int64), ('edge_ids', [9, 2], np.int32),
])
def test_malformed_identity_is_rejected(name, value, dtype):
    with pytest.raises(ValueError): convert(replace(snapshot(graph()), **{name: readonly(value, dtype)}))


@pytest.mark.parametrize('change', ['v1', 'version', 'no_provenance', 'payload_version', 'writable',
    'origin', 'availability', 'record_length', 'record_container', 'graph_ids'])
def test_schema_payload_and_mutable_maps_fail_closed(change):
    s = snapshot(graph()); p = s.provenance
    if change == 'v1': s = snapshot(graph(), provenance=False)
    elif change == 'version': s = replace(s, schema_version='unknown')
    elif change == 'no_provenance': s = replace(s, provenance=None)
    elif change == 'payload_version': s = replace(s, provenance=replace(p, schema_version='unknown'))
    elif change == 'writable': s = replace(s, edge_ids=np.array(s.edge_ids))
    elif change == 'origin': s = replace(s, provenance=replace(p, edge_material_origin=('explicit',)*2))
    elif change == 'availability': s = replace(s, provenance=replace(p, edge_material_available=readonly([False, True], bool)))
    elif change == 'record_length': s = replace(s, provenance=replace(p, edge_material_records=()))
    elif change == 'record_container': s = with_record(s, dict(p.edge_material_records[0]))
    elif change == 'graph_ids': s = replace(s, graph_ids=('a', 'a'))
    with pytest.raises(ValueError): convert(s)


def test_default_batch_fields_and_v1_adapter_guard_remain_unchanged():
    g = graph(); before = snapshot(g, provenance=False); convert(snapshot(g)); after = snapshot(g, provenance=False)
    for field in dataclass_fields(GraphBatch):
        a, b = getattr(before, field.name), getattr(after, field.name)
        if isinstance(a, np.ndarray): assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
        else: assert a == b
    # Source-level contract: no Torch import needed for the unchanged strict adapter.
    tree = ast.parse((Path(__file__).parents[1]/'fibernet/ml/gnn.py').read_text())
    adapter = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'graphs_from_batch')
    assert 'fibernet.graph_batch.v1' in ast.unparse(adapter)
    assert 'fibernet.material_feature_view.v1' not in ast.unparse(adapter)
