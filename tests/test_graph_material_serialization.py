"""Native material records are lossless JSON, with legacy-safe graph identities."""
import copy
from dataclasses import fields
import json

import numpy as np
import pytest

from fibernet import Material, StructureGraph


def _graph():
    graph = StructureGraph()
    for point in ([99, 99], [0, 0], [88, 88], [1, 0], [77, 77]):
        graph.add_node(point, merge=False)
    for node_id in (0, 2, 4):
        graph.remove_node(node_id)
    material = Material(name='full', density=1200., youngs_modulus=1234.,
                        poissons_ratio=.25, shear_modulus=321.,
                        stiffness_tensor=np.eye(6) * 2,
                        extra={'nested': {'tags': ['a', None, True], 'n': 3}})
    required = {'name', 'density', 'youngs_modulus', 'poissons_ratio',
                'shear_modulus', 'stiffness_tensor', 'extra'}
    for item in fields(Material):
        if item.name not in required:
            setattr(material, item.name, 5.5)
    graph.add_edge(1, 3, material=material, radius=.1)
    graph.add_edge(1, 3, material=Material(name='other', youngs_modulus=777.),
                   radius=.2, allow_parallel=True)
    graph.add_edge(1, 3, allow_parallel=True)
    graph.remove_edge(2)
    return graph


def test_all_material_fields_and_sparse_parallel_identity_roundtrip(tmp_path):
    graph = _graph()
    original = graph.to_dict()
    assert original['version'] == '3.1'
    assert set(original['edges'][0]['material']) == {item.name for item in fields(Material)}
    path = tmp_path / 'graph.json'
    graph.save_json(path)
    # Every material value is ordinary JSON (strict finite values).
    json.dumps(original['edges'][0]['material'], allow_nan=False)
    restored = StructureGraph.load_json(path)
    assert restored.to_dict() == original
    assert restored.edges[0].material.youngs_modulus == 1234.
    assert restored.edges[0].material.shear_modulus == 321.
    assert restored.edges[1].material.youngs_modulus == 777.
    np.testing.assert_array_equal(restored.edges[0].material.stiffness_tensor,
                                   graph.edges[0].material.stiffness_tensor)
    assert restored.add_node([2, 0]) == 5
    assert restored.add_edge(1, 3, allow_parallel=True) == 3


def test_record_and_restored_material_mutation_are_isolated():
    graph = _graph()
    data = graph.to_dict()
    before = copy.deepcopy(data)
    restored = StructureGraph.from_dict(data)
    assert data == before  # Material.from_dict cannot mutate caller tensor data.
    restored.edges[0].material.extra['nested']['tags'].append('new')
    restored.edges[0].material.stiffness_tensor[0, 0] = 99
    assert data == before
    data['edges'][0]['material']['extra']['nested']['n'] = 8
    data['edges'][0]['material']['stiffness_tensor'][0][0] = 88
    assert graph.edges[0].material.extra['nested']['n'] == 3
    assert graph.edges[0].material.stiffness_tensor[0, 0] == 2


@pytest.mark.parametrize('version', [None, '3.0', '3.1'])
def test_legacy_missing_material_defaults_are_fresh(version):
    data = _graph().to_dict()
    if version is None:
        data.pop('version')
    else:
        data['version'] = version
    for edge in data['edges']:
        edge.pop('material')
    restored = StructureGraph.from_dict(data)
    assert restored.edges[0].material.to_dict() == Material().to_dict()
    assert restored.edges[1].material.to_dict() == Material().to_dict()
    restored.edges[0].material.extra['test'] = True
    assert not restored.edges[1].material.extra


@pytest.mark.parametrize('material', [
    None, [], 'steel', 3,
    {'class': 'arbitrary.module.Type'}, {'name': 3},
    {'youngs_modulus': '123'}, {'youngs_modulus': True},
    {'youngs_modulus': float('nan')}, {'density': float('inf')},
    {'poissons_ratio': None}, {'shear_modulus': 'bad'},
    {'yield_strength': []}, {'extra': []}, {'extra': {1: 'bad key'}},
    {'extra': {'bad': object()}}, {'extra': {'bad': float('nan')}},
    {'stiffness_tensor': [[1, 2], [3, 4]]},
    {'stiffness_tensor': [['1'] * 6] * 6},
    {'stiffness_tensor': [[True] * 6] * 6},
    {'stiffness_tensor': [[float('inf')] * 6] * 6},
])
def test_malformed_material_payload_rejected(material):
    data = _graph().to_dict()
    data['edges'][0]['material'] = material
    with pytest.raises(ValueError, match='material'):
        StructureGraph.from_dict(data)


@pytest.mark.parametrize('version', ['4.0', 'garbage', 3.1, None])
def test_unknown_schema_rejected(version):
    data = _graph().to_dict()
    data['version'] = version
    with pytest.raises(ValueError, match='schema version'):
        StructureGraph.from_dict(data)


def test_optional_nulls_and_computed_shear():
    data = _graph().to_dict()
    data['edges'][0]['material'] = dict(youngs_modulus=100., poissons_ratio=.25,
                                      shear_modulus=None, stiffness_tensor=None,
                                      tensile_strength=None)
    restored = StructureGraph.from_dict(data)
    assert restored.edges[0].material.shear_modulus == 40.
    assert restored.edges[0].material.stiffness_tensor is None
    assert restored.edges[0].material.tensile_strength is None


def test_fingerprint_scope_is_unchanged():
    graph = _graph()
    before = graph.fingerprint()
    graph.edges[0].material.youngs_modulus *= 2
    assert graph.fingerprint() == before


@pytest.mark.parametrize('bad_value', [float('nan'), 'bad', True])
def test_writer_rejects_invalid_material_numbers(bad_value):
    graph = _graph()
    graph.edges[0].material.youngs_modulus = bad_value
    with pytest.raises(ValueError, match='material'):
        graph.to_dict()


def test_invalid_default_shear_construction_has_material_error():
    data = _graph().to_dict()
    data['edges'][0]['material'] = {'poissons_ratio': -1.}
    with pytest.raises(ValueError, match='invalid material for edge 0'):
        StructureGraph.from_dict(data)


def test_cyclic_extra_rejected_cleanly():
    data = _graph().to_dict()
    extra = {}
    extra['cycle'] = extra
    data['edges'][0]['material']['extra'] = extra
    with pytest.raises(ValueError, match='material extra is cyclic'):
        StructureGraph.from_dict(data)


def test_invalid_writer_tensor_rejected_cleanly():
    graph = _graph()
    graph.edges[0].material.stiffness_tensor = object()
    with pytest.raises(ValueError, match='material'):
        graph.to_dict()
