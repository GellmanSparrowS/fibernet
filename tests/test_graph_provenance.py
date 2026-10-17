"""Pure stored-provenance contracts; no model, solver or predictive benchmark."""
import ast
import json
from collections.abc import Mapping
from dataclasses import fields, FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from fibernet.core.material import Material
from fibernet.core.structure_graph import StructureGraph, SNode, SEdge
from fibernet.ml.graph_data import GraphBatch, GraphBatchProvenance, graph_batch


def fixture():
    graph = StructureGraph(3)
    graph.nodes[20] = SNode(np.array([1., 0., 2.]), 20,
        boundary=(False, True, False, False, False, True))
    graph.nodes[4] = SNode(np.array([1., 0., 0.]), 4)
    graph.edges[9] = SEdge(20, 4, 9, material=Material(name='same-name', youngs_modulus=200.,
        thermal_conductivity=0., extra={'source': {'tags': ['a']}}),
        internal_points=np.array([[2., 0., 1.]]), segments=3,
        metadata={'material_id': 'assertion-not-constitutive-properties'})
    graph.edges[2] = SEdge(4, 20, 2, material=Material(name='same-name', youngs_modulus=800.,
        stiffness_tensor=np.eye(6)), segments=7)
    graph.metadata['nested'] = {'ids': [1, 2]}
    return graph


def batch(*graphs, **kwargs):
    return graph_batch(graphs, graph_ids=[str(i) for i in range(len(graphs))],
        length_unit='mm', coordinate_frame='source-frame', **kwargs)


def thaw(value):
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw(item) for item in value]
    return value


def assert_base_equal(a, b):
    for field in fields(GraphBatch):
        if field.name in ('provenance', 'schema_version'):
            continue
        x, y = getattr(a, field.name), getattr(b, field.name)
        if isinstance(x, np.ndarray):
            assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
        else:
            assert x == y


def test_default_explicit_false_and_old_positional_constructor():
    graph = fixture()
    before = json.dumps(graph.to_dict(), sort_keys=True)
    default, explicit = batch(graph), batch(graph, include_provenance=False)
    assert default.schema_version == explicit.schema_version == 'fibernet.graph_batch.v1'
    assert default.provenance is explicit.provenance is None
    old_arguments = [getattr(default, f.name) for f in fields(GraphBatch) if f.name != 'provenance']
    manually_constructed = GraphBatch(*old_arguments)
    assert manually_constructed.provenance is None
    assert_base_equal(default, explicit)
    assert_base_equal(default, manually_constructed)
    opted = batch(graph, include_provenance=True)
    assert_base_equal(default, opted)
    assert json.dumps(graph.to_dict(), sort_keys=True) == before
    assert fields(GraphBatch)[-1].name == 'provenance'
    # Introspection gains one documented field; no new persistence format exists.
    assert not hasattr(default, 'to_dict')


def test_material_records_faces_and_meshing_are_separate():
    graph = fixture()
    view = batch(graph, include_provenance=True)
    p = view.provenance
    assert type(view) is GraphBatch and isinstance(p, GraphBatchProvenance)
    assert view.schema_version == 'fibernet.graph_batch.v2'
    assert p.schema_version == 'fibernet.graph_batch.provenance.v1'
    assert view.node_ids.tolist() == [4, 20] and view.edge_ids.tolist() == [9, 2]
    assert p.node_boundary_face_flags.tolist() == [[False]*6, [False, True, False, False, False, True]]
    assert p.node_boundary_face_available.tolist() == [True, True]
    assert p.edge_segments.tolist() == [3, 7] and p.edge_segments_available.all()
    assert p.edge_material_available.tolist() == [True, True]
    records = thaw(p.edge_material_records)
    assert records == [graph.edges[eid].material.to_dict() for eid in (9, 2)]
    assert records[0]['name'] == records[1]['name'] == 'same-name'
    assert [row['youngs_modulus'] for row in records] == [200., 800.]
    assert records[0]['thermal_conductivity'] == 0.
    assert 'yield_strength' not in records[0]
    assert view.edge_metadata[0]['material_id'] == 'assertion-not-constitutive-properties'
    assert all(origin == 'stored_origin_unknown' for origin in p.edge_material_origin)
    assert all(origin == 'stored_origin_unknown' for origin in p.node_boundary_face_origin)
    assert all(origin == 'stored_origin_unknown' for origin in p.edge_segments_origin)
    assert not hasattr(p, 'boundary_conditions') and not hasattr(p, 'targets')
    assert view.edge_features.shape == (2, 0)


def test_parallel_reverse_repeated_ids_and_empty_graph_alignment():
    a, b = fixture(), fixture()
    b._nodes = dict(reversed(list(b.nodes.items())))
    b._edges = dict(reversed(list(b.edges.items())))
    view = batch(a, StructureGraph(), b, include_provenance=True)
    p = view.provenance
    assert view.node_ptr.tolist() == [0, 2, 2, 4]
    assert view.edge_ptr.tolist() == [0, 2, 2, 4]
    assert view.node_ids.tolist() == [4, 20, 4, 20]
    assert view.edge_ids.tolist() == [9, 2, 2, 9]
    assert view.edge_index.tolist() == [[1, 0, 2, 3], [0, 1, 3, 2]]
    assert view.message_parent_ids.tolist() == [9, 9, 2, 2, 2, 2, 9, 9]
    assert view.message_orientation.tolist() == [1, -1]*4
    assert [r['youngs_modulus'] for r in p.edge_material_records] == [200., 800., 800., 200.]
    assert p.edge_segments.tolist() == [3, 7, 7, 3]
    assert view.internal_ptr.tolist() == [0, 1, 1, 1, 2]
    for empty in (batch(include_provenance=True), batch(StructureGraph(), include_provenance=True)):
        assert empty.provenance.node_boundary_face_flags.shape == (0, 6)
        assert empty.provenance.edge_segments.shape == (0,)
        assert empty.provenance.edge_material_records == ()


def test_missing_present_zero_and_unknown_origin():
    graph = fixture()
    graph.nodes[4].boundary = None
    graph.edges[9].material = None
    graph.edges[9].segments = None
    graph.edges[2].material.shear_modulus = None  # retain a stored null; do not rederive G
    graph.edges[2].material.density = 0.
    p = batch(graph, include_provenance=True).provenance
    assert p.node_boundary_face_flags[0].tolist() == [False]*6
    assert p.node_boundary_face_available.tolist() == [False, True]
    assert p.node_boundary_face_origin == ('missing', 'stored_origin_unknown')
    assert p.edge_material_records[0] is None
    assert p.edge_material_available.tolist() == [False, True]
    assert p.edge_material_origin == ('missing', 'stored_origin_unknown')
    assert p.edge_material_records[1]['shear_modulus'] is None
    assert p.edge_material_records[1]['density'] == 0.
    assert p.edge_segments.tolist() == [0, 7]
    assert p.edge_segments_available.tolist() == [False, True]
    assert p.edge_segments_origin == ('missing', 'stored_origin_unknown')


def test_detachment_and_readonly_records():
    graph = fixture()
    p = batch(graph, include_provenance=True).provenance
    graph.nodes[20].boundary = (False,)*6
    graph.edges[9].material.youngs_modulus = 999.
    graph.edges[9].material.extra['source']['tags'].append('changed')
    graph.edges[2].material.stiffness_tensor[:] = 0
    graph.edges[9].segments = 99
    assert p.node_boundary_face_flags[1, 1]
    assert p.edge_material_records[0]['youngs_modulus'] == 200.
    assert p.edge_material_records[0]['extra']['source']['tags'] == ('a',)
    assert p.edge_material_records[1]['stiffness_tensor'][0][0] == 1.
    assert p.edge_segments.tolist() == [3, 7]
    with pytest.raises(TypeError):
        p.edge_material_records[0]['extra']['source']['new'] = 'no'
    with pytest.raises(FrozenInstanceError):
        p.schema_version = 'changed'
    for field in fields(p):
        value = getattr(p, field.name)
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable
            with pytest.raises(ValueError):
                value.flat[0] = 0
    # Write flags are accidental-write protection, not a security boundary.
    p.edge_segments.setflags(write=True)
    p.edge_segments[1] = 100
    assert graph.edges[2].segments == 7


@pytest.mark.parametrize('value', [0, 1, None, 'yes', np.bool_(True)])
def test_option_requires_actual_bool(value):
    with pytest.raises(ValueError, match='must be a bool'):
        batch(fixture(), include_provenance=value)


@pytest.mark.parametrize('value', [(), [False]*5, [False]*7, [0]*6, '000000',
    np.zeros((6, 1), dtype=bool), np.array(True), np.ma.array([False]*6, mask=False), object()])
def test_malformed_boundary_only_rejected_for_opt_in(value):
    graph = fixture(); graph.nodes[4].boundary = value
    assert batch(graph).provenance is None
    with pytest.raises(ValueError, match='boundary face'):
        batch(graph, include_provenance=True)


@pytest.mark.parametrize('value', [True, np.bool_(True), 0, -1, 1.5, '3', 2**63, object()])
def test_malformed_segments_only_rejected_for_opt_in(value):
    graph = fixture(); graph.edges[9].segments = value
    assert batch(graph, include_provenance=False).provenance is None
    with pytest.raises(ValueError, match='segments'):
        batch(graph, include_provenance=True)


@pytest.mark.parametrize('change', ['object', 'nan', 'inf_extra', 'cycle', 'bad_tensor', 'unknown_extra_object', 'required_null'])
def test_unsupported_material_records_fail_closed_only_under_opt_in(change):
    graph = fixture(); material = graph.edges[9].material
    if change == 'object': graph.edges[9].material = {'youngs_modulus': 200.}
    elif change == 'nan': material.youngs_modulus = np.nan
    elif change == 'inf_extra': material.extra['bad'] = np.inf
    elif change == 'cycle': material.extra['cycle'] = material.extra
    elif change == 'bad_tensor': material.stiffness_tensor = np.zeros((2, 2))
    elif change == 'unknown_extra_object': material.extra['bad'] = object()
    elif change == 'required_null': material.density = None
    assert batch(graph).provenance is None
    with pytest.raises(ValueError, match='material'):
        batch(graph, include_provenance=True)


def test_material_subclass_serializer_is_not_executed():
    class CustomMaterial(Material):
        def to_dict(self):
            raise AssertionError('custom serialization must not execute')
    graph = fixture(); graph.edges[9].material = CustomMaterial()
    assert batch(graph).provenance is None
    with pytest.raises(ValueError, match='requires Material'):
        batch(graph, include_provenance=True)


def test_numpy_boolean_faces_and_integral_segments():
    graph = fixture(); graph.nodes[4].boundary = np.array([False]*6)
    graph.edges[9].segments = np.int64(4)
    p = batch(graph, include_provenance=True).provenance
    assert p.node_boundary_face_available.all() and p.edge_segments.tolist() == [4, 7]


def test_current_adapter_rejects_v2_before_tensor_work_without_torch():
    # Execute the actual unchanged adapter function AST with only its initial
    # optional-backend guard replaced; v2 must reject before any torch access.
    import fibernet.ml.graph_data as data_module
    source = Path(data_module.__file__).with_name('gnn.py')
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'graphs_from_batch')
    class NoTorch:
        def __getattr__(self, name):
            raise AssertionError('adapter must reject v2 before tensor work')
    namespace = {'_require_torch': lambda: None, 'torch': NoTorch()}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    with pytest.raises(ValueError, match='GraphBatch v1'):
        namespace['graphs_from_batch'](batch(fixture(), include_provenance=True), source_groups=['source'])


def test_public_lazy_provenance_export():
    from fibernet.ml import GraphBatchProvenance as exported
    assert exported is GraphBatchProvenance
