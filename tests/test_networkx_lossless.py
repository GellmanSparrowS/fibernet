"""Tiny native NetworkX interchange contracts; no simulation or training."""
import copy
import inspect
import json

import networkx as nx
import numpy as np
import pytest

from fibernet import Material, StructureGraph


NS = "_fibernet"


def _graph():
    material = Material(name="full", youngs_modulus=1234., shear_modulus=321.,
                        stiffness_tensor=np.eye(6) * 2,
                        extra={"nested": {"tags": ["a", True, None]}})
    for name in Material.__dataclass_fields__:
        if getattr(material, name) is None:
            setattr(material, name, 5.5)
    return StructureGraph.from_dict({
        "version": "3.1", "dimension": 3, "tolerance": 0.002,
        "box_size": [4., 5., 6.], "next_node_id": 100, "next_edge_id": 1000,
        "metadata": {"units": "mm", "periodic": False, "nested": {"v": [1]}},
        "nodes": [{"id": 70, "position": [0., 0., 0.],
                   "boundary": [True, False, False, False, False, False],
                   "metadata": {"pos": "metadata", NS: {"v": [2]}}},
                  {"id": 10, "position": [1., 0., 2.]},
                  {"id": 30, "position": [3., 2., 1.]},
                  {"id": 80, "position": [7., 7., 7.]}],
        "edges": [{"id": 90, "node_i": 10, "node_j": 30,
                   "material": material.to_dict(), "segments": 7,
                   "internal_points": [[1.1, .3, 1.7], [2.9, 1.6, 1.2]],
                   "metadata": {"radius": "metadata", "edge_id": "also metadata",
                                "material": "tag", NS: {"v": [3]}}},
                  {"id": 7, "node_i": 70, "node_j": 30, "radius": .2},
                  {"id": 50, "node_i": 30, "node_j": 10, "radius": .3}]})


def _edge(G, eid=90):
    return next(attrs[NS] for _, _, key, attrs in G.edges(keys=True, data=True) if key == eid)


def test_native_rich_roundtrip_preserves_records_order_and_next_ids():
    graph = _graph()
    before = copy.deepcopy(graph.to_dict())
    G = graph.to_networkx_lossless()
    assert type(G) is nx.MultiGraph and not G.is_directed()
    assert list(G) == [70, 10, 30, 80]
    assert G.graph[NS]["edge_order"] == [90, 7, 50]
    assert [key for _, _, key in G.edges(keys=True)] != [90, 7, 50]
    # This native envelope contains only JSON data in each attribute record.
    json.dumps(G.graph, allow_nan=False)
    for _, attrs in G.nodes(data=True):
        json.dumps(attrs, allow_nan=False)
    for _, _, _, attrs in G.edges(keys=True, data=True):
        json.dumps(attrs, allow_nan=False)
    restored = StructureGraph.from_networkx_lossless(G)
    assert restored.to_dict() == before == graph.to_dict()
    assert list(restored.nodes) == list(graph.nodes)
    assert list(restored.edges) == list(graph.edges)
    np.testing.assert_array_equal(restored.edges[90].internal_points, graph.edges[90].internal_points)
    np.testing.assert_array_equal(restored.edges[90].material.stiffness_tensor,
                                  graph.edges[90].material.stiffness_tensor)
    assert restored.add_node([8, 0, 0]) == 100
    assert restored.add_edge(10, 30, allow_parallel=True) == 1000
    assert restored.add_node([0, 0, 0]) == 70  # Spatial hash reconstructed.
    assert restored.add_edge(10, 30) == 90    # Parallel lookup reconstructed.


def test_oppositely_ordered_parallel_members_do_not_flip():
    graph = StructureGraph()
    a, b = graph.add_node([0, 0]), graph.add_node([1, 0])
    graph.add_edge(a, b)
    graph.add_edge(b, a, allow_parallel=True)
    native = StructureGraph.from_networkx_lossless(graph.to_networkx_lossless())
    assert native.edge_array().tolist() == [[0, 1], [1, 0]]
    # The old projection remains exactly the count-preserving projection.
    assert StructureGraph.from_networkx(graph.to_networkx()).edge_array().tolist() == [[0, 1], [0, 1]]


@pytest.mark.parametrize("kind", ["empty", "isolate", "simple", "empty_curve", "two_dimensional_z"])
def test_empty_isolated_simple_and_full_coordinates(kind):
    graph = StructureGraph(dimension=2, tolerance=1e-5)
    if kind != "empty":
        a = graph.add_node([0, 0, 2])
    if kind in ("simple", "empty_curve", "two_dimensional_z"):
        b = graph.add_node([1, 0, 3])
        graph.add_edge(b, a, internal_points=np.empty((0, 3)) if kind == "empty_curve" else None)
    G = graph.to_networkx_lossless()
    restored = StructureGraph.from_networkx_lossless(G)
    assert type(G) is nx.MultiGraph
    assert restored.to_dict() == graph.to_dict()
    if kind == "empty_curve":
        assert restored.edges[0].internal_points.shape == (0, 3)


def test_current_native_attribute_edits_are_authoritative():
    graph = _graph()
    G = graph.to_networkx_lossless()
    G.nodes[70][NS]["position"] = [2., 3., 4.]
    G.nodes[70][NS]["metadata"]["score"] = .75
    edge = _edge(G)
    edge["radius"] = .9
    edge["segments"] = 11
    edge["material"]["youngs_modulus"] = 9876.
    edge["internal_points"] = [[2., 4., 6.]]
    G.graph[NS]["metadata"]["note"] = "edited"
    before = copy.deepcopy(dict(G.graph))
    restored = StructureGraph.from_networkx_lossless(G)
    assert restored.nodes[70].position.tolist() == [2., 3., 4.]
    assert restored.nodes[70].metadata["score"] == .75
    assert restored.edges[90].radius == .9 and restored.edges[90].segments == 11
    assert restored.edges[90].material.youngs_modulus == 9876.
    assert restored.edges[90].internal_points.tolist() == [[2., 4., 6.]]
    assert restored.metadata["note"] == "edited" and G.graph == before
    assert graph.nodes[70].position.tolist() == [0., 0., 0.]


def test_both_directions_are_deeply_independent():
    graph = _graph()
    G = graph.to_networkx_lossless()
    other = graph.to_networkx_lossless()
    original = copy.deepcopy(graph.to_dict())
    G.graph[NS]["metadata"]["nested"]["v"].append(100)
    G.nodes[70][NS]["metadata"][NS]["v"].append(200)
    _edge(G)["metadata"][NS]["v"].append(300)
    _edge(G)["material"]["extra"]["nested"]["tags"].append("export")
    _edge(G)["material"]["stiffness_tensor"][0][0] = 99
    _edge(G)["internal_points"][0][0] = 99
    assert graph.to_dict() == original
    assert StructureGraph.from_networkx_lossless(other).to_dict() == original
    restored = StructureGraph.from_networkx_lossless(G)
    second = StructureGraph.from_networkx_lossless(G)
    expected = copy.deepcopy(restored.to_dict())
    restored.metadata["nested"]["v"].append(400)
    restored.nodes[70].metadata[NS]["v"].append(500)
    restored.edges[90].metadata[NS]["v"].append(600)
    restored.edges[90].material.extra["nested"]["tags"].append("import")
    restored.edges[90].material.stiffness_tensor[0, 0] = 100
    restored.edges[90].internal_points[0, 0] = 100
    assert second.to_dict() == expected
    assert StructureGraph.from_networkx_lossless(G).to_dict() == expected
    # Source-to-export independence also holds after export has completed.
    graph.nodes[70].metadata[NS]["v"].append(999)
    graph.edges[90].material.stiffness_tensor[1, 1] = 999
    assert StructureGraph.from_networkx_lossless(other).to_dict() == original


@pytest.mark.parametrize("operation", ["add_node", "remove_node", "add_edge", "remove_edge",
    "subgraph", "edge_subgraph", "relabel", "permute_labels", "reconnect", "duplicate_edge_id"])
def test_actual_networkx_topology_edits_fail(operation):
    G = _graph().to_networkx_lossless()
    if operation == "add_node":
        G.add_node(99, **{NS: {"id": 99, "position": [0, 0, 0], "boundary": [False] * 6, "metadata": {}}})
    elif operation == "remove_node":
        G.remove_node(80)
    elif operation == "add_edge":
        attrs = copy.deepcopy(_edge(G)); attrs.update(id=999, node_i=70, node_j=80)
        G.add_edge(70, 80, key=999, **{NS: attrs})
    elif operation == "remove_edge":
        G.remove_edge(10, 30, 90)
    elif operation == "subgraph":
        G = G.subgraph([70, 10, 30]).copy()
    elif operation == "edge_subgraph":
        G = G.edge_subgraph([(10, 30, 90)]).copy()
    elif operation == "relabel":
        G = nx.relabel_nodes(G, {70: 99})
    elif operation == "permute_labels":
        G = nx.relabel_nodes(G, {70: 10, 10: 70})
    elif operation == "reconnect":
        attrs = copy.deepcopy(G.edges[10, 30, 90]); G.remove_edge(10, 30, 90)
        G.add_edge(70, 80, key=90, **attrs)
    else:
        attrs = copy.deepcopy(_edge(G)); attrs.update(node_i=70, node_j=80)
        G.add_edge(70, 80, key=90, **{NS: attrs})
    with pytest.raises(ValueError):
        StructureGraph.from_networkx_lossless(G)


@pytest.mark.parametrize("case", ["extra_graph", "extra_node", "extra_edge", "extra_record",
    "missing_record", "missing_field", "bad_format", "bad_version", "bool_version", "bad_native_schema",
    "bool_node_id", "negative_node_id", "float_edge_id", "bool_counter", "node_counter", "edge_counter",
    "duplicate_node_order", "duplicate_edge_order", "bool_order", "foreign_order", "float_dimension",
    "invalid_dimension", "nan_tolerance", "bad_box", "bad_position", "inf_position", "bad_boundary",
    "bool_radius", "float_segments", "negative_segments", "bad_curve", "inf_curve", "bad_material",
    "unknown_material", "bool_modulus", "null_shear", "nan_metadata", "object_metadata", "tuple_metadata",
    "nonstring_key", "list_metadata", "self_loop", "edge_key_mismatch"])
def test_malformed_native_records_rejected(case):
    G = _graph().to_networkx_lossless(); h = G.graph[NS]; n = G.nodes[70][NS]; e = _edge(G)
    if case == "extra_graph": G.graph["algorithm"] = 1
    elif case == "extra_node": nx.set_node_attributes(G, 1., "score")
    elif case == "extra_edge": G.edges[10, 30, 90]["weight"] = 1
    elif case == "extra_record": n["unknown"] = 1
    elif case == "missing_record": G.nodes[70].clear()
    elif case == "missing_field": del e["material"]
    elif case == "bad_format": h["format"] = "other"
    elif case == "bad_version": h["version"] = 2
    elif case == "bool_version": h["version"] = True
    elif case == "bad_native_schema": h["structure_graph_version"] = "4.0"
    elif case == "bool_node_id": n["id"] = True
    elif case == "negative_node_id": n["id"] = -1
    elif case == "float_edge_id": e["id"] = 90.
    elif case == "bool_counter": h["next_node_id"] = True
    elif case == "node_counter": h["next_node_id"] = 80
    elif case == "edge_counter": h["next_edge_id"] = 90
    elif case == "duplicate_node_order": h["node_order"].append(70)
    elif case == "duplicate_edge_order": h["edge_order"].append(90)
    elif case == "bool_order": h["node_order"][0] = True
    elif case == "foreign_order": h["edge_order"][0] = 999
    elif case == "float_dimension": h["dimension"] = 3.
    elif case == "invalid_dimension": h["dimension"] = 4
    elif case == "nan_tolerance": h["tolerance"] = float("nan")
    elif case == "bad_box": h["box_size"] = [1, 2]
    elif case == "bad_position": n["position"] = [1, 2]
    elif case == "inf_position": n["position"][0] = float("inf")
    elif case == "bad_boundary": n["boundary"][0] = 1
    elif case == "bool_radius": e["radius"] = True
    elif case == "float_segments": e["segments"] = 4.
    elif case == "negative_segments": e["segments"] = -1
    elif case == "bad_curve": e["internal_points"] = [[1, 2]]
    elif case == "inf_curve": e["internal_points"][0][0] = float("inf")
    elif case == "bad_material": e["material"] = []
    elif case == "unknown_material": e["material"]["class"] = "arbitrary"
    elif case == "bool_modulus": e["material"]["youngs_modulus"] = True
    elif case == "null_shear": e["material"]["shear_modulus"] = None
    elif case == "nan_metadata": n["metadata"]["value"] = float("nan")
    elif case == "object_metadata": n["metadata"]["value"] = object()
    elif case == "tuple_metadata": n["metadata"]["value"] = (1, 2)
    elif case == "nonstring_key": n["metadata"][2] = "invalid"
    elif case == "list_metadata": n["metadata"] = []
    elif case == "self_loop": e["node_j"] = e["node_i"]
    else: e["id"] = 91
    with pytest.raises(ValueError):
        StructureGraph.from_networkx_lossless(G)


@pytest.mark.parametrize("kind", ["cycle", "bad_node_id", "bad_edge_id", "boundary", "empty_curve_shape",
                                  "nonarray_curve", "material_subclass", "null_shear", "nonjson"])
def test_export_rejects_unrepresentable_source_without_mutation(kind):
    graph = _graph()
    if kind == "cycle": graph.metadata["cycle"] = graph.metadata
    elif kind == "bad_node_id": graph.nodes[70].node_id = 71
    elif kind == "bad_edge_id": graph.edges[90].edge_id = 91
    elif kind == "boundary": graph.nodes[70].boundary = (1,) * 6
    elif kind == "empty_curve_shape": graph.edges[90].internal_points = np.empty((0, 4))
    elif kind == "nonarray_curve": graph.edges[90].internal_points = [[1, 2, 3]]
    elif kind == "material_subclass":
        class CustomMaterial(Material): pass
        graph.edges[90].material = CustomMaterial()
    elif kind == "null_shear": graph.edges[90].material.shear_modulus = None
    else: graph.metadata["value"] = np.array([1, 2])
    orders = (list(graph.nodes), list(graph.edges))
    with pytest.raises(ValueError): graph.to_networkx_lossless()
    assert (list(graph.nodes), list(graph.edges)) == orders


@pytest.mark.parametrize("kind", [nx.Graph, nx.DiGraph, nx.MultiDiGraph])
def test_unsupported_graph_types_fail(kind):
    with pytest.raises(ValueError, match="MultiGraph"):
        StructureGraph.from_networkx_lossless(kind(_graph().to_networkx_lossless()))


def test_legacy_signatures_and_outputs_unchanged():
    signature = inspect.signature(StructureGraph.from_networkx)
    assert signature.parameters["dimension"].default == 2
    assert signature.parameters["tolerance"].default == 1e-6
    graph = StructureGraph(dimension=3)
    a, b = graph.add_node([0, 0, 1], label="a"), graph.add_node([1, 0, 2])
    graph.add_edge(b, a, radius=.3, segments=8, tag="x")
    G = graph.to_networkx()
    assert type(G) is nx.Graph and G.graph == {}
    assert list(G.nodes(data=True)) == [(0, {"pos": (0., 0., 1.), "boundary": (False,) * 6, "label": "a"}),
                                       (1, {"pos": (1., 0., 2.), "boundary": (False,) * 6})]
    assert list(G.edges(data=True)) == [(0, 1, {"radius": .3, "edge_id": 0, "segments": 8, "tag": "x"})]
    restored = StructureGraph.from_networkx(G)
    assert restored.dimension == 2 and restored.tolerance == 1e-6
    assert restored.edge_array().tolist() == [[0, 1]]
    assert not restored.nodes[0].metadata and not restored.edges[0].metadata
    with pytest.raises(TypeError):
        StructureGraph.from_networkx_lossless(graph.to_networkx_lossless(), dimension=2)
    with pytest.raises(TypeError):
        StructureGraph.from_networkx_lossless(graph.to_networkx_lossless(), tolerance=1e-6)


def test_structure_algorithms_work_and_metadata_is_the_edit_namespace():
    G = _graph().to_networkx_lossless()
    assert nx.number_connected_components(G) == 2
    assert nx.shortest_path(G, 70, 10) == [70, 30, 10]
    for node, score in nx.degree_centrality(G).items():
        G.nodes[node][NS]["metadata"]["score"] = score
    restored = StructureGraph.from_networkx_lossless(G)
    assert all("score" in node.metadata for node in restored.nodes.values())


@pytest.mark.parametrize("mask", [True, False])
@pytest.mark.parametrize("field", ["position", "curve", "box", "boundary", "radius", "segments", "tensor", "modulus"])
def test_masked_source_data_rejected_before_serialization(field, mask):
    graph = _graph()
    if field == "position": graph.nodes[70].position = np.ma.array([0., 0., 0.], mask=mask)
    elif field == "curve": graph.edges[90].internal_points = np.ma.array([[1., 2., 3.]], mask=mask)
    elif field == "box": graph._box_size = np.ma.array([4., 5., 6.], mask=mask)
    elif field == "boundary": graph.nodes[70].boundary = np.ma.array([False] * 6, mask=mask)
    elif field == "radius": graph.edges[90].radius = np.ma.array(.1, mask=mask)
    elif field == "segments": graph.edges[90].segments = np.ma.array(4, mask=mask)
    elif field == "tensor": graph.edges[90].material.stiffness_tensor = np.ma.array(np.eye(6), mask=mask)
    else: graph.edges[90].material.youngs_modulus = np.ma.array(1234., mask=mask)
    with pytest.raises(ValueError, match="masked"):
        graph.to_networkx_lossless()


@pytest.mark.parametrize("field", ["position", "radius", "tensor", "metadata"])
def test_masked_import_records_rejected(field):
    G = _graph().to_networkx_lossless()
    if field == "position": G.nodes[70][NS]["position"][0] = np.ma.masked
    elif field == "radius": _edge(G)["radius"] = np.ma.array(.1, mask=False)
    elif field == "tensor": _edge(G)["material"]["stiffness_tensor"] = np.ma.array(np.eye(6), mask=False)
    else: G.graph[NS]["metadata"]["hidden"] = np.ma.array([1], mask=False)
    with pytest.raises(ValueError, match="masked"):
        StructureGraph.from_networkx_lossless(G)


@pytest.mark.parametrize("tolerance", [0., -1.])
def test_positive_tolerance_even_for_empty_graph(tolerance):
    with pytest.raises(ValueError, match="positive"):
        StructureGraph(tolerance=tolerance).to_networkx_lossless()
    G = StructureGraph().to_networkx_lossless()
    G.graph[NS]["tolerance"] = tolerance
    with pytest.raises(ValueError, match="positive"):
        StructureGraph.from_networkx_lossless(G)


@pytest.mark.parametrize("field", ["position", "radius", "modulus", "tensor"])
def test_import_rejects_float64_value_loss(field):
    G = _graph().to_networkx_lossless(); too_precise = 2 ** 53 + 1
    if field == "position": G.nodes[70][NS]["position"][0] = too_precise
    elif field == "radius": _edge(G)["radius"] = too_precise
    elif field == "modulus": _edge(G)["material"]["youngs_modulus"] = too_precise
    else: _edge(G)["material"]["stiffness_tensor"][0][0] = too_precise
    with pytest.raises(ValueError, match="exactly as float64"):
        StructureGraph.from_networkx_lossless(G)


@pytest.mark.parametrize("field", ["radius", "position", "curve", "modulus", "tensor"])
def test_export_rejects_float64_value_loss(field):
    graph = _graph(); too_precise = np.uint64(2 ** 53 + 1)
    if field == "radius": graph.edges[90].radius = too_precise
    elif field == "position": graph.nodes[70].position = np.array([too_precise, 0, 0], dtype=np.uint64)
    elif field == "curve": graph.edges[90].internal_points = np.array([[too_precise, 0, 0]], dtype=np.uint64)
    elif field == "modulus": graph.edges[90].material.youngs_modulus = too_precise
    else: graph.edges[90].material.stiffness_tensor = np.full((6, 6), too_precise, dtype=np.uint64)
    with pytest.raises(ValueError, match="exactly as float64"):
        graph.to_networkx_lossless()
